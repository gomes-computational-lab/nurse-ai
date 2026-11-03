from contextlib import nullcontext, redirect_stderr, redirect_stdout
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from diarization_demo import DEFAULT_SAMPLE_RATE, Diarizer, main, run_live


class HoldbackTests(unittest.TestCase):
    def test_deferred_words_use_revised_speakers_and_final_flush_does_not_repeat(self):
        def output(turns):
            annotation = Mock()
            annotation.itertracks.return_value = [
                (SimpleNamespace(start=a, end=b), 0, label) for a, b, label in turns]
            return SimpleNamespace(speaker_diarization=annotation,
                                   exclusive_speaker_diarization=annotation)

        def words(items):
            return [SimpleNamespace(words=[SimpleNamespace(start=a, end=b, word=text)
                                           for a, b, text in items])], None

        # The first window incorrectly merges the patient's reply. A later
        # window corrects it before that reply has been finalized.
        pipeline = Mock(side_effect=[
            output([(0, 6, 'SPEAKER_00')]),
            output([(0, 1, 'SPEAKER_00'), (1, 6, 'SPEAKER_01')]),
            output([(0, 1, 'SPEAKER_00'), (1, 6, 'SPEAKER_01')]),
        ])
        whisper = Mock()
        whisper.transcribe.side_effect = [
            words([(.5, 1.5, 'Hello.'), (4, 5, 'Yes.')]),
            words([(1, 2, 'Yes.'), (3, 4, 'Thanks.')]),
            words([(1, 2, 'Yes.'), (3, 4, 'Thanks.')]),
        ]
        torch = Mock()
        torch.inference_mode.side_effect = nullcontext
        diarizer = Diarizer(pipeline, torch, whisper, holdback=3)
        diarizer.voices = Mock()
        diarizer.voices.assign.return_value = {'SPEAKER_00': 'SPEAKER_00', 'SPEAKER_01': 'SPEAKER_01'}
        samples = np.zeros(6 * DEFAULT_SAMPLE_RATE, dtype=np.float32)
        stdout = StringIO()
        with redirect_stdout(stdout), redirect_stderr(StringIO()):
            diarizer.process(samples, 0)
            self.assertEqual(diarizer.emitted_until, 3)
            self.assertNotIn('Yes.', stdout.getvalue())
            diarizer.process(samples, 3)
            self.assertEqual(diarizer.emitted_until, 6)
            self.assertNotIn('Thanks.', stdout.getvalue())
            diarizer.process(samples, 3, final=True)
        self.assertEqual(stdout.getvalue().splitlines(), [
            '[00:00.500-00:01.500] SPEAKER_00: Hello.',
            '[00:04.000-00:05.000] SPEAKER_01: Yes.',
            '[00:06.000-00:07.000] SPEAKER_01: Thanks.',
        ])
        self.assertEqual(diarizer.emitted_until, 9)

    def test_labels_without_transcription_are_held_and_flushed(self):
        annotation = Mock()
        annotation.itertracks.return_value = [(SimpleNamespace(start=0, end=6), 0, 'SPEAKER_00')]
        pipeline = Mock(return_value=SimpleNamespace(speaker_diarization=annotation))
        torch = Mock()
        torch.inference_mode.side_effect = nullcontext
        diarizer = Diarizer(pipeline, torch, holdback=3)
        samples = np.zeros(6 * DEFAULT_SAMPLE_RATE, dtype=np.float32)
        stdout = StringIO()
        with redirect_stdout(stdout), redirect_stderr(StringIO()):
            diarizer.process(samples, 0)
            diarizer.process(samples, 0, final=True)
        self.assertEqual(stdout.getvalue().splitlines(), [
            '[00:00.000-00:03.000] SPEAKER_00',
            '[00:03.000-00:06.000] SPEAKER_00',
        ])

    def test_ctrl_c_flushes_pending_capture_only_when_holdback_is_enabled(self):
        for holdback in (0, 1.5):
            with self.subTest(holdback=holdback):
                stream = Mock(active=True)
                stream.__exit__ = Mock(return_value=False)

                def input_stream(**kwargs):
                    def enter():
                        kwargs['callback'](np.ones((16000, 1), dtype=np.float32), 16000, None, None)
                        return stream
                    stream.__enter__ = Mock(side_effect=enter)
                    return stream

                sd = SimpleNamespace(InputStream=input_stream)
                diarizer = Mock(holdback=holdback)
                with patch.dict('sys.modules', {'sounddevice': sd}), \
                        patch('diarization_demo.time.sleep', side_effect=KeyboardInterrupt), \
                        redirect_stdout(StringIO()), self.assertRaises(KeyboardInterrupt):
                    run_live(diarizer, 1, 6 * DEFAULT_SAMPLE_RATE, 3 * DEFAULT_SAMPLE_RATE)
                if holdback:
                    stream.stop.assert_called_once()
                    diarizer.process.assert_called_once()
                    call = diarizer.process.call_args
                    self.assertEqual(len(call.args[0]), 16000)
                    self.assertEqual(call.args[1], 0)
                    self.assertEqual(call.kwargs, {'final': True})
                else:
                    diarizer.process.assert_not_called()

    def test_cli_rejects_holdback_that_cannot_fit_the_next_window(self):
        for value in ('-1', 'nan', 'inf', '13'):
            with self.subTest(value=value), redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
                main(['--window', '15', '--step', '3', '--holdback', value])
            self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
