"""Exercise model preparation without a downloaded voice or ONNX Runtime."""

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from piper.prepare_short_speech import _expose_durations, prepare

try:
    import onnx
except ImportError:
    onnx = None


@unittest.skipIf(onnx is None, "Optional onnx package is not installed")
class PrepareShortSpeechTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name).resolve()
        self.config = {"espeak": {"voice": "fa"}, "num_speakers": 1}

    def make_model(self, rank=4):
        helper, tensor = onnx.helper, onnx.TensorProto
        nodes = [
            helper.make_node("Cast", ["input"], ["float_ids"], to=tensor.FLOAT),
            helper.make_node("Unsqueeze", ["float_ids"], ["mono"], axes=[1]),
            helper.make_node("Ceil", ["mono"], ["durations"]),
        ]
        if rank == 4:
            nodes.append(helper.make_node("Unsqueeze", ["mono"], ["output"], axes=[1]))
            audio_shape = ["batch_size", "time", 1, "samples"]
        else:
            nodes.append(helper.make_node("Identity", ["mono"], ["output"]))
            audio_shape = ["batch_size", 1, "samples"]
        graph = helper.make_graph(
            nodes,
            "synthetic_mana_interface",
            [
                helper.make_tensor_value_info(
                    "input", tensor.INT64, ["batch", "tokens"]
                ),
                helper.make_tensor_value_info("input_lengths", tensor.INT64, ["batch"]),
                helper.make_tensor_value_info("scales", tensor.FLOAT, [3]),
            ],
            [helper.make_tensor_value_info("output", tensor.FLOAT, audio_shape)],
        )
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])

    def write_source(self, model=None):
        source = self.directory / "mana.onnx"
        onnx.save_model(model or self.make_model(), source)
        Path(f"{source}.json").write_bytes(
            (json.dumps(self.config, indent=4) + "\n").encode()
        )
        return source

    def test_preserves_source_graph_audio_output_and_config_for_both_audio_ranks(self):
        for rank in (3, 4):
            with self.subTest(rank=rank):
                original = self.make_model(rank)
                source = self.write_source(original)
                original_bytes = source.read_bytes()
                config_bytes = Path(f"{source}.json").read_bytes()
                target = self.directory / f"prepared-{rank}.onnx"
                self.assertEqual(prepare(str(source), str(target)), target)
                prepared = onnx.load(target)
                self.assertEqual(source.read_bytes(), original_bytes)
                self.assertEqual(Path(f"{target}.json").read_bytes(), config_bytes)
                self.assertEqual(list(prepared.graph.node), list(original.graph.node))
                self.assertEqual(prepared.graph.initializer, original.graph.initializer)
                self.assertEqual(prepared.graph.output[0], original.graph.output[0])
                self.assertEqual(len(prepared.graph.output), 2)
                self.assertEqual(prepared.graph.output[1].name, "durations")
                self.assertEqual(
                    {item.key: item.value for item in prepared.metadata_props},
                    {
                        "piper.duration_output": "durations",
                        "piper.duration_unit": "frames",
                        "piper.duration_hop_length": "256",
                        "piper.source_model_sha256": hashlib.sha256(
                            original_bytes
                        ).hexdigest(),
                    },
                )

    def test_uses_config_hop_length_and_retains_metadata(self):
        model = self.make_model()
        onnx.helper.set_model_props(model, {"existing": "preserved"})
        self.config["hop_length"] = 512
        _expose_durations(model, self.config, "a" * 64, onnx)
        metadata = {item.key: item.value for item in model.metadata_props}
        self.assertEqual(metadata["piper.duration_hop_length"], "512")
        self.assertEqual(metadata["existing"], "preserved")

    def test_refuses_source_existing_model_and_existing_config(self):
        source = self.write_source()
        target = self.directory / "occupied.onnx"
        for occupied in (source, target, Path(f"{target}.json")):
            with self.subTest(occupied=occupied):
                if occupied != source:
                    occupied.write_bytes(b"user file")
                before = occupied.read_bytes()
                with self.assertRaises(ValueError):
                    prepare(str(source), str(source if occupied == source else target))
                self.assertEqual(occupied.read_bytes(), before)
                if occupied != source:
                    occupied.unlink()

    def test_rejects_ambiguous_or_incompatible_models(self):
        original = self.make_model()
        wrong_input = copy.deepcopy(original)
        wrong_input.graph.input[0].name = "wrong"
        duplicate_ceil = copy.deepcopy(original)
        duplicate_ceil.graph.node.append(
            onnx.helper.make_node("Ceil", ["mono"], ["another"])
        )
        extra_output = copy.deepcopy(original)
        extra_output.graph.output.append(original.graph.output[0])
        wrong_channel = copy.deepcopy(original)
        wrong_channel.graph.output[0].type.tensor_type.shape.dim[2].dim_value = 2
        already_marked = copy.deepcopy(original)
        onnx.helper.set_model_props(already_marked, {"piper.duration_unit": "frames"})
        for model in (
            wrong_input,
            duplicate_ceil,
            extra_output,
            wrong_channel,
            already_marked,
        ):
            with self.subTest(model=model.graph.name), self.assertRaises(ValueError):
                _expose_durations(model, self.config, "a" * 64, onnx)

    def test_rejects_external_tensors_before_loading_them(self):
        model = self.make_model()
        external = onnx.helper.make_tensor(
            "external", onnx.TensorProto.FLOAT, [1], [1.0]
        )
        external.data_location = onnx.TensorProto.EXTERNAL
        entry = external.external_data.add()
        entry.key, entry.value = "location", "missing.bin"
        model.graph.initializer.append(external)
        with self.assertRaisesRegex(ValueError, "External tensor data"):
            _expose_durations(model, self.config, "a" * 64, onnx)

    def test_missing_optional_dependency_has_installation_message(self):
        source = self.write_source()
        with patch.dict(sys.modules, {"onnx": None}):
            with self.assertRaisesRegex(RuntimeError, "pip install onnx"):
                prepare(str(source), str(self.directory / "new.onnx"))

    def test_failed_publication_cleans_only_own_files(self):
        source = self.write_source()
        target = self.directory / "failed.onnx"
        original_link = os.link

        def fail_model_publication(staged, destination):
            if Path(destination).suffix == ".onnx":
                raise OSError("simulated publication failure")
            return original_link(staged, destination)

        with patch(
            "piper.prepare_short_speech.os.link", side_effect=fail_model_publication
        ):
            with self.assertRaisesRegex(OSError, "simulated publication failure"):
                prepare(str(source), str(target))
        self.assertFalse(target.exists())
        self.assertFalse(Path(f"{target}.json").exists())
        self.assertEqual(list(self.directory.glob(".piper-mana-*")), [])
        self.assertTrue(source.is_file())
        self.assertTrue(Path(f"{source}.json").is_file())
