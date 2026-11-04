from __future__ import annotations

from io import StringIO
import unittest

from sim.diarization import (
    ProcessingDiagnostics,
    ProcessingResult,
    RecognizedWord,
    SpeakerTurn,
)
from sim.diarization.cli import present_result


class InspectionTests(unittest.TestCase):
    def test_inspection_formats_structured_turns_words_and_diagnostics(self) -> None:
        result = ProcessingResult(
            speaker_turns=(
                SpeakerTurn(0, 3, "SPEAKER_00", "LOCAL_01"),
                SpeakerTurn(3, 6, "SPEAKER_01", "LOCAL_00"),
            ),
            recognized_words=(
                RecognizedWord(1, 2, "Hello."),
                RecognizedWord(4, 5, "Hi."),
            ),
            finalized_utterances=(),
            provisional_utterances=(),
            audio_gaps=(),
            diagnostics=ProcessingDiagnostics(0, 6, 0.25, -20, -5, 2, 2, 2),
        )
        stdout, stderr = StringIO(), StringIO()

        present_result(result, inspect=True, stdout=stdout, stderr=stderr)

        self.assertIn("[TURN 00:00.000-00:03.000] SPEAKER_00 (model=LOCAL_01)", stdout.getvalue())
        self.assertIn("[TURN 00:03.000-00:06.000] SPEAKER_01 (model=LOCAL_00)", stdout.getvalue())
        self.assertIn("[STT 00:01.000-00:05.000] Hello. Hi.", stdout.getvalue())
        self.assertIn("input RMS -20 dBFS, peak -5 dBFS", stdout.getvalue())
        self.assertIn("model=2, tracked=2, profiles=2", stdout.getvalue())
        self.assertNotIn("ROLE_", stdout.getvalue())
        self.assertEqual(stderr.getvalue(), "")

    def test_inspection_reports_empty_independent_transcription(self) -> None:
        result = ProcessingResult(
            speaker_turns=(SpeakerTurn(0, 6, "SPEAKER_00", "LOCAL_00"),),
            recognized_words=(),
            finalized_utterances=(),
            provisional_utterances=(),
            audio_gaps=(),
            diagnostics=ProcessingDiagnostics(0, 6, 0.1, -30, -10, 1, 1, 1),
        )
        stdout = StringIO()

        present_result(result, inspect=True, stdout=stdout, stderr=StringIO())

        self.assertIn("[STT 00:00.000-00:06.000] (no text)", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
