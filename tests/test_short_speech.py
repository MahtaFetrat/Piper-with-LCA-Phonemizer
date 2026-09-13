"""Exercise first-copy extraction through Piper's real synthesis pipeline with a fake model."""

import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from piper import PiperConfig, PiperVoice, SynthesisConfig
from piper.__main__ import main
from piper.config import PhonemeType
from piper.short_speech import (
    first_copy_sample_bounds,
    is_short_persian_text,
    plan_repeated_speech,
)

ID_MAP = {"^": [1], "_": [0], "$": [2], " ": [3], "a": [4], "b": [5]}


class ShortSpeechTests(unittest.TestCase):
    def setUp(self):
        self.config = PiperConfig(
            6, 1, 22050, "fa", ID_MAP, PhonemeType.ESPEAK, hop_length=4
        )
        self.metadata = {
            "piper.duration_output": "/Ceil_output_0",
            "piper.duration_unit": "frames",
            "piper.duration_hop_length": "4",
            "piper.source_model_sha256": "a" * 64,
        }
        self.outputs = [
            SimpleNamespace(name="output"),
            SimpleNamespace(
                name="/Ceil_output_0",
                type="tensor(float)",
                shape=["batch", 1, "phonemes"],
            ),
        ]
        self.session = SimpleNamespace(
            get_outputs=lambda: self.outputs,
            get_modelmeta=lambda: SimpleNamespace(custom_metadata_map=self.metadata),
            run=mock.Mock(side_effect=self.infer),
        )
        self.cancelled = threading.Event()
        self.chunks = [list("ab")]

    def infer(self, outputs, args):
        count = args["input"].shape[1]
        # Each input ID has a different amplitude, exposing accidental context playback.
        audio = np.repeat(np.arange(1, count + 1, dtype=np.float32) / 10, 4)
        return [audio.reshape(1, 1, 1, -1), np.ones((1, 1, count), dtype=np.float32)]

    def voice(self, enabled=True, language="fa"):
        voice = PiperVoice(
            self.session,
            replace(self.config, espeak_voice=language),
            use_short_speech_repeat=enabled,
        )
        voice.phonemize_stream = mock.Mock(side_effect=lambda *args: iter(self.chunks))
        return voice

    def test_eligibility_is_limited_to_bounded_persian_tokens(self):
        for text in ("دستیار", "دَستیار", "ی", "می‌رود", " دستیار. ", "«دستیار»"):
            with self.subTest(text=text):
                self.assertTrue(is_short_persian_text(text))
        for text in (
            "",
            "   ",
            "این دستیار است",
            "hello",
            "دستیارhello",
            "[[dastjar]]",
            "۱۲۳",
            "د" * 65,
        ):
            with self.subTest(text=text):
                self.assertFalse(is_short_persian_text(text))

    def test_plan_preserves_symbols_and_bounds_repeated_context(self):
        original = list("ab")
        plan = plan_repeated_speech(original, ID_MAP)
        self.assertEqual(original, list("ab"))
        self.assertEqual(plan.phonemes, tuple(" ".join(["ab"] * 11)))
        self.assertEqual((plan.first_start_id, plan.first_end_id), (2, 6))
        self.assertEqual(plan.phoneme_ids[2:6], (4, 0, 5, 0))
        self.assertIsNone(plan_repeated_speech([], ID_MAP))
        self.assertIsNone(plan_repeated_speech(["a"] * 41, ID_MAP))
        self.assertEqual(len(plan_repeated_speech(["a"] * 40, ID_MAP).phonemes), 81)
        with self.assertRaises(ValueError):
            plan_repeated_speech(list("ac"), ID_MAP)

    def test_boundaries_support_multiple_ids_per_symbol_and_pad(self):
        id_map = dict(ID_MAP, **{"^": [1, 9], "_": [0, 8], "a": [4, 7]})
        plan = plan_repeated_speech(list("ab"), id_map)
        counts = list(range(1, len(plan.phoneme_ids) + 1))
        self.assertEqual((plan.first_start_id, plan.first_end_id), (4, 11))
        self.assertEqual(
            first_copy_sample_bounds(plan, counts, sum(counts)),
            (sum(counts[:4]), sum(counts[:11])),
        )

    def test_cropped_audio_uses_common_normalization_volume_and_original_metadata(self):
        voice = self.voice()
        settings = SynthesisConfig(
            length_scale=1.5, noise_scale=0.3, noise_w_scale=0.2, volume=0.5
        )
        [chunk] = voice.synthesize("دستیار", settings, include_alignments=True)
        expected = np.repeat(np.array([3, 4, 5, 6], dtype=np.float32) / 6 * 0.5, 4)
        np.testing.assert_allclose(chunk.audio_float_array, expected)
        np.testing.assert_array_equal(
            chunk.audio_int16_array, (expected * 32767).astype(np.int16)
        )
        self.assertEqual(chunk.phonemes, list("ab"))
        self.assertEqual(chunk.phoneme_ids, voice.phonemes_to_ids(list("ab")))
        np.testing.assert_array_equal(chunk.phoneme_id_samples, [0, 0, 4, 4, 4, 4, 0])
        self.assertEqual(
            [item.phoneme for item in chunk.phoneme_alignments], ["^", "a", "b", "$"]
        )
        self.assertEqual(
            sum(item.num_samples for item in chunk.phoneme_alignments), len(expected)
        )
        np.testing.assert_allclose(
            self.session.run.call_args.args[1]["scales"], [0.3, 1.5, 0.2]
        )
        self.assertEqual(self.chunks, [list("ab")])
        voice.phonemize_stream.assert_called_once_with("دستیار", None)

    def test_no_normalization_and_no_requested_alignments_preserve_api_behavior(self):
        [chunk] = self.voice().synthesize(
            "د", SynthesisConfig(normalize_audio=False, volume=0.5)
        )
        expected = np.repeat(np.array([3, 4, 5, 6], dtype=np.float32) / 10 * 0.5, 4)
        np.testing.assert_array_equal(chunk.audio_float_array, expected)
        self.assertIsNone(chunk.phoneme_id_samples)
        self.assertIsNone(chunk.phoneme_alignments)

    def test_disabled_non_persian_sentences_and_explicit_phonemes_do_not_repeat(self):
        for text, enabled, language in (
            ("دستیار", False, "fa"),
            ("دستیار", True, "en-us"),
            ("این دستیار است", True, "fa"),
            ("[[dastjar]]", True, "fa"),
            ("hello", True, "fa"),
        ):
            with self.subTest(text=text, enabled=enabled, language=language):
                voice = self.voice(enabled, language)
                [chunk] = voice.synthesize(text)
                actual_ids = self.session.run.call_args.args[1]["input"][0]
                np.testing.assert_array_equal(
                    actual_ids, voice.phonemes_to_ids(list("ab"))
                )
                self.assertEqual(len(chunk.audio_float_array), 28)

    def test_multichunk_frontend_is_preserved_without_rephonemizing(self):
        self.chunks = [list("ab"), list("ab"), list("ab")]
        voice = self.voice()
        chunks = list(voice.synthesize("دستیار"))
        self.assertEqual([len(chunk.audio_float_array) for chunk in chunks], [28] * 3)
        voice.phonemize_stream.assert_called_once()
        self.assertEqual(self.session.run.call_count, 3)

    def test_missing_metadata_is_rejected_only_when_persian_feature_is_enabled(self):
        self.metadata.clear()
        with self.assertRaises(ValueError):
            self.voice()
        self.assertIsNotNone(self.voice(False))
        self.assertIsNotNone(self.voice(True, "en-us"))

    def test_invalid_duration_contract_fails_before_frontend(self):
        for field, value in (
            ("piper.duration_unit", "samples"),
            ("piper.duration_output", "unrelated"),
            ("piper.duration_hop_length", "512"),
            ("piper.source_model_sha256", "not-a-hash"),
        ):
            with (
                self.subTest(field=field),
                mock.patch.dict(self.metadata, {field: value}),
            ):
                with self.assertRaises(ValueError):
                    self.voice()
        self.session.run.assert_not_called()

    def test_missing_and_malformed_durations_never_emit_context(self):
        for defect in (
            "missing",
            "length",
            "negative",
            "fractional",
            "nan",
            "total",
            "shape",
        ):

            def infer(outputs, args):
                audio, durations = self.infer(outputs, args)
                if defect == "missing":
                    return [audio]
                if defect == "length":
                    durations = durations[..., :-1]
                elif defect == "negative":
                    durations[..., 0] = -1
                elif defect == "fractional":
                    durations[..., 0] = 0.5
                elif defect == "nan":
                    durations[..., 0] = np.nan
                elif defect == "total":
                    durations[..., 0] += 1
                elif defect == "shape":
                    durations = np.repeat(durations, 2, axis=1)
                return [audio, durations]

            with self.subTest(defect=defect):
                self.session.run.side_effect = infer
                with self.assertRaises(ValueError):
                    next(self.voice().synthesize("دستیار"))

    def test_first_copy_cannot_claim_the_whole_waveform_or_empty_later_copies(self):
        plan = plan_repeated_speech(list("ab"), ID_MAP)
        counts = [0] * len(plan.phoneme_ids)
        counts[plan.first_start_id] = 1000
        with self.assertRaises(ValueError):
            first_copy_sample_bounds(plan, counts, 1000)
        counts[-1] = 1  # An EOS tail still does not assign audio to subsequent copies.
        with self.assertRaises(ValueError):
            first_copy_sample_bounds(plan, counts, 1001)

    def test_cancellation_before_frontend_skips_work(self):
        voice = self.voice()
        self.cancelled.set()
        self.assertEqual(
            list(voice.synthesize("د", cancelled_callback=self.cancelled.is_set)), []
        )
        voice.phonemize_stream.assert_not_called()
        self.session.run.assert_not_called()

    def test_cancellation_after_frontend_skips_inference(self):
        voice = self.voice()

        def frontend(*args):
            yield list("ab")
            self.cancelled.set()

        voice.phonemize_stream.side_effect = frontend
        self.assertEqual(
            list(voice.synthesize("د", cancelled_callback=self.cancelled.is_set)), []
        )
        self.session.run.assert_not_called()

    def test_cancellation_after_inference_emits_no_audio(self):
        def infer(*args):
            result = self.infer(*args)
            self.cancelled.set()
            return result

        self.session.run.side_effect = infer
        self.assertEqual(
            list(
                self.voice().synthesize("د", cancelled_callback=self.cancelled.is_set)
            ),
            [],
        )

    def test_load_and_cli_expose_an_explicit_disabled_by_default_option(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "mana.onnx"
            model.touch()
            config_path = Path(f"{model}.json")
            import json

            config_path.write_text(json.dumps(self.config.to_dict()), encoding="utf-8")
            with mock.patch(
                "piper.voice.onnxruntime.InferenceSession", return_value=self.session
            ):
                self.assertFalse(PiperVoice.load(model).use_short_speech_repeat)
                self.assertTrue(
                    PiperVoice.load(
                        model, use_short_speech_repeat=True
                    ).use_short_speech_repeat
                )
            for extra, expected in (([], False), (["--short-speech-repeat"], True)):
                with (
                    self.subTest(enabled=expected),
                    mock.patch(
                        "sys.argv",
                        ["piper", "--model", str(model), "--output-raw", *extra],
                    ),
                    mock.patch("sys.stdin", []),
                    mock.patch(
                        "piper.__main__.PiperVoice.load", return_value=self.voice()
                    ) as load,
                    mock.patch("piper.__main__.logging.basicConfig"),
                ):
                    main()
                    self.assertEqual(
                        load.call_args.kwargs["use_short_speech_repeat"], expected
                    )


if __name__ == "__main__":
    unittest.main()
