from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import unittest
from unittest.mock import Mock

import numpy as np

from sim.diarization import (
    DiarizationConfig,
    DiarizationFrame,
    DiarizationService,
    DiarizationServiceClosedError,
    DiarizationServiceError,
    LocalSpeakerTurn,
    ProviderInferenceError,
    RecognizedWord,
    UtteranceStatus,
)


SAMPLE_RATE = 16_000


def acoustic_frame(
    turns: tuple[tuple[float, float, str], ...] = ((0, 6, "LOCAL_A"),),
    embeddings: tuple[tuple[float, ...], ...] | None = ((1, 0),),
) -> DiarizationFrame:
    local_turns = tuple(LocalSpeakerTurn(start, end, speaker) for start, end, speaker in turns)
    return DiarizationFrame(
        speaker_turns=local_turns,
        exclusive_speaker_turns=local_turns,
        speaker_labels=tuple(sorted({speaker for _, _, speaker in turns})),
        speaker_embeddings=embeddings,
    )


def words(*items: tuple[float, float, str]) -> tuple[RecognizedWord, ...]:
    return tuple(RecognizedWord(start, end, text) for start, end, text in items)


def samples(seconds: int = 6, value: float = 0.1) -> np.ndarray:
    return np.full(seconds * SAMPLE_RATE, value, dtype=np.float32)


class DiarizationServiceTests(unittest.TestCase):
    def make_service(
        self,
        transcripts,
        *,
        config: DiarizationConfig | None = None,
        frames=None,
    ):
        diarization = Mock()
        diarization.diarize.side_effect = frames
        if frames is None:
            diarization.diarize.return_value = acoustic_frame()
        transcription = Mock()
        transcription.transcribe.side_effect = transcripts
        service = DiarizationService(diarization, transcription, config)
        return service, diarization, transcription

    def test_process_returns_structured_final_and_replaceable_provisional_results(self) -> None:
        service, diarization, transcription = self.make_service([
            words((1, 2, "early"), (4, 5, "tail")),
            words((1, 2, "tail revised"), (4, 5, "new")),
        ])
        stdout, stderr = StringIO(), StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            first = service.process(samples(), 0)
            second = service.process(samples(), 3)

        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(first.speaker_turns[0].speaker_id, "SPEAKER_00")
        self.assertEqual(first.speaker_turns[0].source_speaker_id, "LOCAL_A")
        self.assertEqual(
            [(word.start, word.end, word.text) for word in first.recognized_words],
            [(1, 2, "early"), (4, 5, "tail")],
        )
        self.assertEqual([item.text for item in first.finalized_utterances], ["early"])
        self.assertEqual([item.text for item in first.provisional_utterances], ["tail"])
        self.assertIs(first.finalized_utterances[0].status, UtteranceStatus.FINALIZED)
        self.assertIs(first.provisional_utterances[0].status, UtteranceStatus.PROVISIONAL)
        self.assertEqual(service.finalized_until, 6)

        # Final output is new-only; the provisional collection is a full replacement.
        self.assertEqual([item.text for item in second.finalized_utterances], ["tail revised"])
        self.assertNotIn("early", [item.text for item in second.finalized_utterances])
        self.assertEqual([item.text for item in second.provisional_utterances], ["new"])
        self.assertEqual(service.provisional_utterances, second.provisional_utterances)
        self.assertEqual([item.text for item in first.provisional_utterances], ["tail"])
        self.assertEqual(second.diagnostics.window_start, 3)
        self.assertEqual(second.diagnostics.window_end, 9)
        self.assertEqual(second.diagnostics.local_speaker_count, 1)
        self.assertEqual(second.diagnostics.tracked_speaker_count, 1)
        self.assertEqual(second.diagnostics.profile_count, 1)
        self.assertAlmostEqual(second.diagnostics.input_rms_dbfs, -20, places=3)
        self.assertAlmostEqual(second.diagnostics.input_peak_dbfs, -20, places=3)
        self.assertEqual(diarization.diarize.call_count, 2)
        self.assertEqual(transcription.transcribe.call_count, 2)
        self.assertEqual(
            diarization.diarize.call_args.kwargs,
            {"sample_rate": SAMPLE_RATE, "min_speakers": 1, "max_speakers": 2},
        )

    def test_flush_finalizes_the_pending_tail_once_and_closes_service(self) -> None:
        service, _, _ = self.make_service([
            words((1, 2, "early"), (4, 5, "pending tail")),
        ])
        service.process(samples(), 0)

        first_flush = service.flush()
        second_flush = service.flush()

        self.assertEqual([item.text for item in first_flush.finalized_utterances], ["pending tail"])
        self.assertIs(first_flush.finalized_utterances[0].status, UtteranceStatus.FINALIZED)
        self.assertEqual(first_flush.provisional_utterances, ())
        self.assertEqual(first_flush.speaker_turns, ())
        self.assertEqual(first_flush.recognized_words, ())
        self.assertEqual(second_flush.finalized_utterances, ())
        self.assertEqual(service.provisional_utterances, ())
        self.assertEqual(service.finalized_until, 6)
        self.assertTrue(service.is_flushed)
        with self.assertRaises(DiarizationServiceClosedError):
            service.process(samples(), 3)

    def test_provider_failures_do_not_advance_or_replace_service_state(self) -> None:
        service, diarization, transcription = self.make_service([
            words((1, 2, "early"), (4, 5, "pending")),
        ])
        service.process(samples(), 0)
        previous_frontier = service.finalized_until
        previous_provisional = service.provisional_utterances
        previous_profiles = service.profile_count

        transcription.transcribe.side_effect = ProviderInferenceError("Whisper failed")
        with self.assertRaisesRegex(ProviderInferenceError, "Whisper failed"):
            service.process(samples(), 3)
        self.assertEqual(service.finalized_until, previous_frontier)
        self.assertEqual(service.provisional_utterances, previous_provisional)
        self.assertEqual(service.profile_count, previous_profiles)

        diarization.diarize.side_effect = ProviderInferenceError("Pyannote failed")
        transcription.transcribe.side_effect = None
        transcription.transcribe.return_value = words((1, 2, "retry"))
        with self.assertRaisesRegex(ProviderInferenceError, "Pyannote failed"):
            service.process(samples(), 3)
        self.assertEqual(service.finalized_until, previous_frontier)
        self.assertEqual(service.provisional_utterances, previous_provisional)
        self.assertEqual(service.profile_count, previous_profiles)

        # The same window can be retried after a failure because no timeline state moved.
        diarization.diarize.side_effect = None
        diarization.diarize.return_value = acoustic_frame()
        retry = service.process(samples(), 3)
        self.assertEqual([item.text for item in retry.finalized_utterances], ["retry"])
        self.assertEqual(service.finalized_until, 6)

    def test_reports_audio_dropped_between_finalized_frontier_and_next_window(self) -> None:
        config = DiarizationConfig(finalization_delay_seconds=0)
        service, _, _ = self.make_service([(), ()], config=config)

        first = service.process(samples(), 0)
        second = service.process(samples(), 8)

        self.assertEqual(first.audio_gaps, ())
        self.assertEqual(len(second.audio_gaps), 1)
        self.assertEqual((second.audio_gaps[0].start, second.audio_gaps[0].end), (6, 8))
        self.assertIn("unavailable", second.audio_gaps[0].reason)
        self.assertEqual(service.finalized_until, 14)

    def test_rejects_invalid_or_backward_windows_without_calling_providers(self) -> None:
        service, diarization, transcription = self.make_service([()])
        invalid = (
            (np.array([], dtype=np.float32), 0),
            (np.zeros((2, 2), dtype=np.float32), 0),
            (np.array([float("nan")], dtype=np.float32), 0),
            (samples(), -1),
            (samples(), float("inf")),
        )
        for audio, start in invalid:
            with self.subTest(start=start, shape=audio.shape), self.assertRaises(ValueError):
                service.process(audio, start)
        diarization.diarize.assert_not_called()
        transcription.transcribe.assert_not_called()

        service.process(samples(), 3)
        with self.assertRaisesRegex(DiarizationServiceError, "must not move backward"):
            service.process(samples(), 0)

    def test_flush_before_processing_is_empty_and_idempotent(self) -> None:
        service, diarization, transcription = self.make_service([])

        first = service.flush()
        second = service.flush()

        self.assertEqual(first.finalized_utterances, ())
        self.assertEqual(second.finalized_utterances, ())
        self.assertEqual(first.diagnostics.window_start, 0)
        self.assertEqual(first.diagnostics.window_end, 0)
        diarization.diarize.assert_not_called()
        transcription.transcribe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
