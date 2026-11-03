from contextlib import nullcontext, redirect_stderr, redirect_stdout
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from diarization_demo import DEFAULT_SAMPLE_RATE, Diarizer


def model_output(turns, embeddings):
    annotation = Mock()
    annotation.itertracks.return_value = [
        (SimpleNamespace(start=a, end=b), 0, label) for a, b, label in turns]
    annotation.labels.return_value = sorted({label for _, _, label in turns})
    return SimpleNamespace(speaker_diarization=annotation,
                           exclusive_speaker_diarization=annotation,
                           speaker_embeddings=np.asarray(embeddings, dtype=float))


def recognized(items):
    return [SimpleNamespace(words=[SimpleNamespace(start=a, end=b, word=text)
                                   for a, b, text in items])], None


class InspectionTests(unittest.TestCase):
    def make_diarizer(self, output, transcript, **kwargs):
        torch = Mock()
        torch.inference_mode.side_effect = nullcontext
        whisper = Mock()
        whisper.transcribe.return_value = transcript
        diarizer = Diarizer(Mock(return_value=output), torch, whisper,
                            track_speakers=True, **kwargs)
        return diarizer

    def capture(self, diarizer, start=0):
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            diarizer.process(np.full(6 * DEFAULT_SAMPLE_RATE, .1, dtype=np.float32), start)
        return stdout.getvalue(), stderr.getvalue()

    def test_speaker_turns_are_visible_when_whisper_returns_no_words(self):
        diarizer = self.make_diarizer(
            model_output([(0, 3, 'SPEAKER_00'), (3, 6, 'SPEAKER_01')], [[1, 0], [0, 1]]),
            ([], None), inspect=True)
        stdout, stderr = self.capture(diarizer)
        self.assertIn('[TURN 00:00.000-00:03.000] SPEAKER_00 (model=SPEAKER_00)', stdout)
        self.assertIn('[TURN 00:03.000-00:06.000] SPEAKER_01 (model=SPEAKER_01)', stdout)
        self.assertIn('[STT 00:00.000-00:06.000] (no text)', stdout)
        self.assertIn('input RMS -20 dBFS, peak -20 dBFS', stdout)
        self.assertIn('model=2, tracked=2, profiles=2/2', stdout)
        self.assertNotIn('ROLE_', stdout)
        self.assertNotIn('no new words', stdout + stderr)

    def test_independent_stt_runs_even_when_pyannote_detects_no_speech(self):
        diarizer = self.make_diarizer(model_output([], np.empty((0, 2))),
                                    recognized([(1, 2, 'Hello.')]), inspect=True)
        stdout, _ = self.capture(diarizer)
        self.assertIn('[STT 00:01.000-00:02.000] Hello.', stdout)
        self.assertNotIn('[TURN ', stdout)
        self.assertIn('model=0, tracked=0, profiles=0/2', stdout)
        diarizer.whisper.transcribe.assert_called_once()

    def test_independent_stt_is_not_hidden_by_nonoverlapping_speaker_turns(self):
        diarizer = self.make_diarizer(model_output([(0, 1, 'SPEAKER_00')], [[1, 0]]),
                                    recognized([(2, 3, 'Outside the turn.')]), inspect=True)
        stdout, _ = self.capture(diarizer)
        self.assertIn('[TURN 00:00.000-00:01.000]', stdout)
        self.assertIn('[STT 00:02.000-00:03.000] Outside the turn.', stdout)

    def test_tracking_without_roles_keeps_the_combined_transcript_format(self):
        diarizer = self.make_diarizer(model_output([(0, 6, 'SPEAKER_00')], [[1, 0]]),
                                    recognized([(1, 2, 'Hello.')]))
        stdout, _ = self.capture(diarizer)
        self.assertIsNotNone(diarizer.voices)
        self.assertIsNone(diarizer.roles)
        self.assertEqual(stdout, '[00:01.000-00:02.000] SPEAKER_00: Hello.\n')

    def test_repeated_words_in_later_audio_are_still_printed(self):
        diarizer = self.make_diarizer(model_output([(0, 6, 'SPEAKER_00')], [[1, 0]]),
                                    recognized([(1, 2, 'Yes.')]), inspect=True)
        first, _ = self.capture(diarizer)
        diarizer.whisper.transcribe.return_value = recognized([(3, 4, 'Yes.')])
        second, _ = self.capture(diarizer, start=3)
        self.assertIn('[STT 00:01.000-00:02.000] Yes.', first)
        self.assertIn('[STT 00:06.000-00:07.000] Yes.', second)


if __name__ == '__main__':
    unittest.main()
