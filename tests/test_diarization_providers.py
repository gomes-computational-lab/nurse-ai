from __future__ import annotations

from contextlib import nullcontext
import subprocess
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from sim.diarization import (
    DiarizationConfig,
    DiarizationProvider,
    ProviderDependencyError,
    ProviderInferenceError,
    ProviderLoadError,
    PyannoteDiarizationProvider,
    TranscriptionProvider,
    WhisperTranscriptionProvider,
    load_pyannote_provider,
    load_whisper_provider,
)


def annotation(turns: list[tuple[float, float, str]]) -> Mock:
    value = Mock()
    value.itertracks.return_value = [
        (SimpleNamespace(start=start, end=end), 0, speaker)
        for start, end, speaker in turns
    ]
    value.labels.return_value = sorted({speaker for _, _, speaker in turns})
    return value


class ProviderTests(unittest.TestCase):
    def test_pyannote_adapter_returns_library_independent_frame(self) -> None:
        regular = annotation([(0, 2, "LOCAL_00"), (2, 4, "LOCAL_01")])
        exclusive = annotation([(0, 2.5, "LOCAL_00"), (2.5, 4, "LOCAL_01")])
        output = SimpleNamespace(
            speaker_diarization=regular,
            exclusive_speaker_diarization=exclusive,
            speaker_embeddings=np.array([[1, 0], [0, 1]], dtype=float),
        )
        pipeline = Mock(return_value=output)
        tensor = Mock()
        torch = SimpleNamespace(
            inference_mode=lambda: nullcontext(),
            from_numpy=Mock(return_value=tensor),
        )
        provider = PyannoteDiarizationProvider(
            pipeline,
            torch,
            model_name="test/diarizer",
            device_name="cpu",
        )
        samples = np.zeros(4 * 16_000, dtype=np.float32)

        frame = provider.diarize(
            samples,
            sample_rate=16_000,
            min_speakers=1,
            max_speakers=2,
        )

        self.assertIsInstance(provider, DiarizationProvider)
        self.assertEqual(
            [(turn.start, turn.end, turn.speaker_id) for turn in frame.speaker_turns],
            [(0, 2, "LOCAL_00"), (2, 4, "LOCAL_01")],
        )
        self.assertEqual(frame.exclusive_speaker_turns[0].end, 2.5)
        self.assertEqual(frame.speaker_labels, ("LOCAL_00", "LOCAL_01"))
        self.assertEqual(frame.speaker_embeddings, ((1.0, 0.0), (0.0, 1.0)))
        tensor.unsqueeze.assert_called_once_with(0)
        self.assertEqual(
            pipeline.call_args.kwargs,
            {"min_speakers": 1, "max_speakers": 2},
        )
        self.assertEqual(pipeline.call_args.args[0]["sample_rate"], 16_000)

    def test_pyannote_adapter_treats_malformed_embeddings_as_unavailable(self) -> None:
        turns = annotation([(0, 1, "LOCAL_00")])
        output = SimpleNamespace(
            speaker_diarization=turns,
            exclusive_speaker_diarization=turns,
            speaker_embeddings=np.array([1, 0], dtype=float),
        )
        torch = SimpleNamespace(
            inference_mode=lambda: nullcontext(),
            from_numpy=lambda samples: Mock(unsqueeze=Mock(return_value=samples)),
        )
        provider = PyannoteDiarizationProvider(
            Mock(return_value=output),
            torch,
            model_name="test/diarizer",
            device_name="cpu",
        )

        frame = provider.diarize(
            np.zeros(16_000, dtype=np.float32),
            sample_rate=16_000,
            min_speakers=1,
            max_speakers=2,
        )

        self.assertIsNone(frame.speaker_embeddings)

    def test_whisper_adapter_returns_clipped_nonempty_words(self) -> None:
        model = Mock()
        model.transcribe.return_value = ([SimpleNamespace(words=[
            SimpleNamespace(start=-0.1, end=0.5, word=" Hello "),
            SimpleNamespace(start=0.5, end=2.5, word="world."),
            SimpleNamespace(start=1.5, end=1.5, word="zero"),
            SimpleNamespace(start=1.0, end=1.2, word=" "),
        ])], None)
        provider = WhisperTranscriptionProvider(model, DiarizationConfig())

        words = provider.transcribe(
            np.zeros(2 * 16_000, dtype=np.float32),
            sample_rate=16_000,
        )

        self.assertIsInstance(provider, TranscriptionProvider)
        self.assertEqual(
            [(word.start, word.end, word.text) for word in words],
            [(0, 0.5, "Hello"), (0.5, 2.0, "world.")],
        )
        kwargs = model.transcribe.call_args.kwargs
        self.assertEqual(kwargs["language"], "en")
        self.assertTrue(kwargs["word_timestamps"])
        self.assertTrue(kwargs["vad_filter"])

    def test_adapters_wrap_inference_failures(self) -> None:
        failing_pipeline = Mock(side_effect=RuntimeError("diarization exploded"))
        torch = SimpleNamespace(
            inference_mode=lambda: nullcontext(),
            from_numpy=lambda samples: Mock(unsqueeze=Mock(return_value=samples)),
        )
        diarizer = PyannoteDiarizationProvider(
            failing_pipeline,
            torch,
            model_name="test/diarizer",
            device_name="cpu",
        )
        with self.assertRaisesRegex(ProviderInferenceError, "Pyannote inference failed"):
            diarizer.diarize(
                np.zeros(16_000, dtype=np.float32),
                sample_rate=16_000,
                min_speakers=1,
                max_speakers=2,
            )

        whisper = WhisperTranscriptionProvider(
            Mock(transcribe=Mock(side_effect=RuntimeError("transcription exploded"))),
            DiarizationConfig(),
        )
        with self.assertRaisesRegex(ProviderInferenceError, "Whisper inference failed"):
            whisper.transcribe(np.zeros(16_000, dtype=np.float32), sample_rate=16_000)

    def test_pyannote_load_failure_redacts_token(self) -> None:
        token = "hf_secret-value"

        class Pipeline:
            @staticmethod
            def from_pretrained(model_name, *, token):
                del model_name
                raise RuntimeError(f"request authorization={token}")

        torch = types.ModuleType("torch")
        torch.device = lambda name: name
        torch.cuda = SimpleNamespace(is_available=lambda: False)
        pyannote = types.ModuleType("pyannote")
        pyannote.__path__ = []  # type: ignore[attr-defined]
        pyannote_audio = types.ModuleType("pyannote.audio")
        pyannote_audio.Pipeline = Pipeline

        with patch.dict(
            sys.modules,
            {"torch": torch, "pyannote": pyannote, "pyannote.audio": pyannote_audio},
        ):
            with self.assertRaises(ProviderLoadError) as error:
                load_pyannote_provider(DiarizationConfig(), token=token)

        self.assertNotIn(token, str(error.exception))
        self.assertIn("[REDACTED]", str(error.exception))

    def test_loaders_report_missing_dependencies_without_raw_import_errors(self) -> None:
        with patch.dict(sys.modules, {"torch": None}):
            with self.assertRaisesRegex(ProviderDependencyError, "requires.*Torch"):
                load_pyannote_provider(DiarizationConfig(), token="hf_valid-token")

        with patch.dict(sys.modules, {"faster_whisper": None}):
            with self.assertRaisesRegex(ProviderDependencyError, "requires `faster-whisper`"):
                load_whisper_provider(DiarizationConfig())

    def test_whisper_loader_uses_public_provider_boundary(self) -> None:
        created: list[tuple[str, str, str]] = []

        class WhisperModel:
            def __init__(self, model_name, *, device, compute_type):
                created.append((model_name, device, compute_type))

        faster_whisper = types.ModuleType("faster_whisper")
        faster_whisper.WhisperModel = WhisperModel
        with patch.dict(sys.modules, {"faster_whisper": faster_whisper}):
            provider = load_whisper_provider(DiarizationConfig(whisper_model_name="base.en"))

        self.assertIsInstance(provider, WhisperTranscriptionProvider)
        self.assertEqual(created, [("base.en", "cpu", "int8")])

    def test_whisper_loader_wraps_model_initialization_failure(self) -> None:
        class WhisperModel:
            def __init__(self, model_name, *, device, compute_type):
                del model_name, device, compute_type
                raise RuntimeError("model cache unavailable")

        faster_whisper = types.ModuleType("faster_whisper")
        faster_whisper.WhisperModel = WhisperModel
        with patch.dict(sys.modules, {"faster_whisper": faster_whisper}):
            with self.assertRaisesRegex(ProviderLoadError, "Could not load Whisper model"):
                load_whisper_provider(DiarizationConfig())

    def test_provider_module_imports_no_heavy_model_libraries(self) -> None:
        script = """
import sys
import sim.diarization.providers
heavy = ('torch', 'pyannote', 'pyannote.audio', 'faster_whisper', 'sounddevice')
print([name for name in heavy if name in sys.modules])
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.stdout.strip(), "[]")


if __name__ == "__main__":
    unittest.main()
