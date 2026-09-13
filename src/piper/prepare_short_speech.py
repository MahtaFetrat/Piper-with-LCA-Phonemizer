"""Create a separate duration-enabled copy of the locally investigated Mana model.

This preparation step needs optional ``onnx``; normal synthesis does not.
The unique Ceil tensor is a duration tensor in this Mana export. That assumption
has not been established for arbitrary Piper models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def _local_voice_paths(path: str) -> tuple[Path, Path]:
    """Resolve a local model/configuration pair without downloading assets."""
    voice_path = Path(path).expanduser().resolve(strict=True)
    if voice_path.name.endswith(".onnx.json"):
        model_path, config_path = voice_path.with_suffix(""), voice_path
    elif voice_path.suffix == ".onnx":
        model_path, config_path = voice_path, Path(f"{voice_path}.json")
    else:
        raise ValueError("--voice must name an .onnx or .onnx.json file")
    if not model_path.is_file() or not config_path.is_file():
        raise ValueError(f"Missing model/configuration pair: {model_path}")
    return model_path, config_path


def _has_external_data(message) -> bool:
    if message.DESCRIPTOR.full_name == "onnx.TensorProto" and (
        message.external_data or message.data_location == 1
    ):
        return True
    for field, value in message.ListFields():
        if field.message_type is not None:
            children = value if field.is_repeated else (value,)
            if any(_has_external_data(child) for child in children):
                return True
    return False


def _expose_durations(model, config: dict, source_hash: str, onnx) -> None:
    """Validate the expected Mana interface, then expose an existing tensor only."""
    if _has_external_data(model):
        raise ValueError(
            "External tensor data is unsupported; use the original self-contained Mana model"
        )
    if config.get("espeak", {}).get("voice") != "fa" or config.get("num_speakers") != 1:
        raise ValueError(
            "This preparation tool is intended for the investigated single-speaker Persian Mana model"
        )
    hop_length = config.get("hop_length", 256)
    if type(hop_length) is not int or not 1 <= hop_length <= 4096:
        raise ValueError("Invalid hop_length in the voice configuration")
    expected = {
        "input": (onnx.TensorProto.INT64, 2),
        "input_lengths": (onnx.TensorProto.INT64, 1),
        "scales": (onnx.TensorProto.FLOAT, 1),
    }
    if {item.name for item in model.graph.input} != expected.keys():
        raise ValueError(
            "Expected exactly the Mana input, input_lengths and scales inputs"
        )
    for item in model.graph.input:
        tensor = item.type.tensor_type
        if (tensor.elem_type, len(tensor.shape.dim)) != expected[item.name]:
            raise ValueError(f"Unexpected type or rank for input {item.name}")
    if len(model.graph.output) != 1:
        raise ValueError(
            "Expected exactly one original audio output; already prepared models are unsupported"
        )
    audio = model.graph.output[0].type.tensor_type
    dimensions = audio.shape.dim
    # The investigated Mana export declares [batch, time, 1, samples].
    # Ordinary Piper exports can instead declare [batch, 1, samples].
    mono_shape = (len(dimensions) == 4 and dimensions[2].dim_value == 1) or (
        len(dimensions) == 3 and dimensions[1].dim_value == 1
    )
    if audio.elem_type != onnx.TensorProto.FLOAT or not mono_shape:
        raise ValueError("Expected a float mono audio output of rank three or four")
    ceil_nodes = [
        node for node in model.graph.node if node.op_type == "Ceil" and not node.domain
    ]
    if (
        len(ceil_nodes) != 1
        or len(ceil_nodes[0].output) != 1
        or not ceil_nodes[0].output[0]
    ):
        raise ValueError("Expected a unique top-level Ceil duration tensor")
    duration_name = ceil_nodes[0].output[0]
    if duration_name == model.graph.output[0].name:
        raise ValueError("The duration tensor must differ from the audio output")
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    if any(key.startswith("piper.duration_") for key in metadata):
        raise ValueError("The model already contains Piper duration metadata")
    onnx.checker.check_model(model)
    model.graph.output.append(
        onnx.helper.make_tensor_value_info(
            duration_name, onnx.TensorProto.FLOAT, ["batch_size", 1, "phonemes"]
        ),
    )
    metadata.update(
        {
            "piper.duration_output": duration_name,
            "piper.duration_unit": "frames",
            "piper.duration_hop_length": str(hop_length),
            "piper.source_model_sha256": source_hash,
        }
    )
    onnx.helper.set_model_props(model, metadata)
    onnx.checker.check_model(model)


def prepare(voice: str, output: str) -> Path:
    """Write a new model/config pair without overwriting an existing file."""
    model_path, config_path = _local_voice_paths(voice)
    requested_target = Path(output).expanduser()
    if os.path.lexists(requested_target) or os.path.lexists(f"{requested_target}.json"):
        raise ValueError(
            f"Refusing to overwrite an existing target model/configuration: {requested_target}"
        )
    target = requested_target.resolve()
    target_config = Path(f"{target}.json")
    if target.suffix != ".onnx":
        raise ValueError("--output must name a new .onnx file")
    for destination in (target, target_config):
        if destination in (
            model_path.resolve(),
            config_path.resolve(),
        ) or os.path.lexists(destination):
            raise ValueError(
                f"Refusing to overwrite an existing model, configuration or path: {destination}"
            )
    if not target.parent.is_dir():
        raise ValueError("The --output parent directory must already exist")
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError(
            "Preparation requires optional onnx: install it with python -m pip install onnx"
        ) from error
    config_bytes = config_path.read_bytes()
    model_bytes = model_path.read_bytes()
    model = onnx.load_model_from_string(model_bytes)
    _expose_durations(
        model, json.loads(config_bytes), hashlib.sha256(model_bytes).hexdigest(), onnx
    )
    # Hard-link publication is atomic and refuses existing targets, including a
    # target created after validation. Both scratch files are on the same volume.
    with tempfile.TemporaryDirectory(
        prefix=".piper-mana-", dir=target.parent
    ) as scratch:
        staged_model = Path(scratch) / "voice.onnx"
        staged_config = Path(scratch) / "voice.onnx.json"
        onnx.save_model(model, staged_model)
        staged_config.write_bytes(config_bytes)
        os.link(staged_config, target_config)
        try:
            os.link(staged_model, target)
        except BaseException:
            if target_config.is_file() and os.path.samefile(
                staged_config, target_config
            ):
                target_config.unlink()
            raise
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--voice", required=True, help="Original local Mana .onnx or .onnx.json"
    )
    parser.add_argument(
        "--output", required=True, help="New .onnx path in an existing directory"
    )
    args = parser.parse_args(argv)
    try:
        result = prepare(args.voice, args.output)
    except Exception as error:
        parser.exit(1, f"Could not prepare Mana: {error}\n")
    print(f"Prepared {result} and {result}.json; the original files are unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
