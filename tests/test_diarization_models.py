from __future__ import annotations

import subprocess
import sys
import unittest
from dataclasses import FrozenInstanceError

from sim.diarization import (
    DEFAULT_PYANNOTE_MODEL,
    DEFAULT_WHISPER_MODEL,
    AudioGap,
    DiarizationConfig,
    ProcessingDiagnostics,
    ProcessingResult,
    RecognizedWord,
    SpeakerTurn,
    Utterance,
    UtteranceStatus,
)


def diagnostics() -> ProcessingDiagnostics:
    return ProcessingDiagnostics(
        window_start=0,
        window_end=6,
        processing_seconds=0.25,
        input_rms_dbfs=-20,
        input_peak_dbfs=-5,
        local_speaker_count=2,
        tracked_speaker_count=2,
        profile_count=2,
    )


def utterance(status: UtteranceStatus) -> Utterance:
    return Utterance(1, 2, "SPEAKER_00", "Hello.", status)


class DiarizationConfigTests(unittest.TestCase):
    def test_defaults_preserve_current_demo_behavior(self) -> None:
        config = DiarizationConfig()

        self.assertEqual(config.pyannote_model_name, DEFAULT_PYANNOTE_MODEL)
        self.assertEqual(config.whisper_model_name, DEFAULT_WHISPER_MODEL)
        self.assertEqual(config.language, "en")
        self.assertEqual(config.sample_rate, 16_000)
        self.assertEqual(
            (config.window_seconds, config.step_seconds, config.finalization_delay_seconds),
            (6.0, 3.0, 3.0),
        )
        self.assertEqual((config.min_speakers, config.max_speakers), (1, 2))
        self.assertEqual(config.max_voice_profiles, 2)
        self.assertEqual(config.profile_seed_min_speech_seconds, 1.0)
        self.assertEqual(config.new_profile_max_similarity, 0.50)
        self.assertEqual(config.match_min_similarity, 0.60)
        self.assertEqual(config.match_min_margin, 0.10)
        self.assertEqual(config.profile_update_min_similarity, 0.75)
        self.assertEqual(config.profile_update_min_margin, 0.20)
        self.assertEqual(config.profile_update_weight, 0.05)

    def test_accepts_valid_custom_configuration_and_is_frozen(self) -> None:
        config = DiarizationConfig(
            pyannote_model_name="example/diarizer",
            whisper_model_name="base.en",
            language="pt",
            sample_rate=8_000,
            window_seconds=12,
            step_seconds=4,
            finalization_delay_seconds=5,
            min_speakers=2,
            max_speakers=3,
            max_voice_profiles=4,
            profile_seed_min_speech_seconds=1.5,
            new_profile_max_similarity=0.40,
            match_min_similarity=0.65,
            match_min_margin=0.15,
            profile_update_min_similarity=0.80,
            profile_update_min_margin=0.25,
            profile_update_weight=0.10,
        )

        self.assertEqual(config.whisper_model_name, "base.en")
        self.assertEqual(config.max_voice_profiles, 4)
        with self.assertRaises(FrozenInstanceError):
            config.sample_rate = 16_000  # type: ignore[misc]

    def test_rejects_invalid_configuration(self) -> None:
        invalid = (
            {"pyannote_model_name": " "},
            {"whisper_model_name": ""},
            {"language": ""},
            {"sample_rate": 0},
            {"sample_rate": True},
            {"window_seconds": float("nan")},
            {"window_seconds": 0},
            {"step_seconds": 7},
            {"finalization_delay_seconds": -1},
            {"finalization_delay_seconds": 4},
            {"min_speakers": 3, "max_speakers": 2, "max_voice_profiles": 3},
            {"min_speakers": 2, "max_voice_profiles": 1},
            {"profile_seed_min_speech_seconds": 0},
            {"new_profile_max_similarity": -1.1},
            {"match_min_similarity": 1.1},
            {"match_min_margin": -0.1},
            {"profile_update_min_margin": 2.1},
            {"profile_update_weight": 0},
            {"profile_update_weight": 1.1},
            {"new_profile_max_similarity": 0.60},
            {"profile_update_min_similarity": 0.55},
            {"match_min_margin": 0.30},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                DiarizationConfig(**values)


class DiarizationModelTests(unittest.TestCase):
    def test_result_converts_collections_to_immutable_tuples(self) -> None:
        turn = SpeakerTurn(0, 2, "SPEAKER_00", "LOCAL_01")
        word = RecognizedWord(1, 2, "Hello.")
        final = utterance(UtteranceStatus.FINALIZED)
        provisional = utterance(UtteranceStatus.PROVISIONAL)
        gap = AudioGap(2, 3, "capture buffer overwritten")
        result = ProcessingResult(
            [turn],  # type: ignore[arg-type]
            [word],  # type: ignore[arg-type]
            [final],  # type: ignore[arg-type]
            [provisional],  # type: ignore[arg-type]
            [gap],  # type: ignore[arg-type]
            diagnostics(),
        )

        self.assertEqual(result.speaker_turns, (turn,))
        self.assertEqual(result.recognized_words, (word,))
        self.assertEqual(result.finalized_utterances, (final,))
        self.assertEqual(result.provisional_utterances, (provisional,))
        self.assertEqual(result.audio_gaps, (gap,))
        with self.assertRaises(FrozenInstanceError):
            result.audio_gaps = ()  # type: ignore[misc]

    def test_rejects_invalid_timestamps_and_empty_values(self) -> None:
        invalid_factories = (
            lambda: SpeakerTurn(-1, 2, "SPEAKER_00", "LOCAL_00"),
            lambda: SpeakerTurn(2, 1, "SPEAKER_00", "LOCAL_00"),
            lambda: SpeakerTurn(0, 1, "", "LOCAL_00"),
            lambda: SpeakerTurn(0, 1, "SPEAKER_00", " "),
            lambda: RecognizedWord(0, float("inf"), "Hello"),
            lambda: RecognizedWord(0, 1, ""),
            lambda: Utterance(0, 1, "", "Hello", UtteranceStatus.FINALIZED),
            lambda: Utterance(0, 1, "SPEAKER_00", " ", UtteranceStatus.FINALIZED),
            lambda: AudioGap(3, 2, "overwritten"),
            lambda: AudioGap(2, 3, ""),
            lambda: ProcessingDiagnostics(0, 6, -1, -20, -5, 1, 1, 1),
            lambda: ProcessingDiagnostics(0, 6, 1, float("nan"), -5, 1, 1, 1),
            lambda: ProcessingDiagnostics(0, 6, 1, -20, -5, -1, 1, 1),
        )
        for factory in invalid_factories:
            with self.subTest(factory=factory), self.assertRaises(ValueError):
                factory()

    def test_rejects_invalid_or_misplaced_utterance_statuses(self) -> None:
        with self.assertRaises(ValueError):
            Utterance(0, 1, "SPEAKER_00", "Hello", "finalized")  # type: ignore[arg-type]

        final = utterance(UtteranceStatus.FINALIZED)
        provisional = utterance(UtteranceStatus.PROVISIONAL)
        with self.assertRaisesRegex(ValueError, "Finalized utterances"):
            ProcessingResult((), (), (provisional,), (), (), diagnostics())
        with self.assertRaisesRegex(ValueError, "Provisional utterances"):
            ProcessingResult((), (), (), (final,), (), diagnostics())

    def test_rejects_wrong_result_item_types_and_diagnostics(self) -> None:
        with self.assertRaisesRegex(ValueError, "Speaker turns"):
            ProcessingResult(("not a turn",), (), (), (), (), diagnostics())  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "Diagnostics"):
            ProcessingResult((), (), (), (), (), object())  # type: ignore[arg-type]

    def test_public_package_does_not_import_heavy_dependencies(self) -> None:
        script = """
import sys
import sim.diarization
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
