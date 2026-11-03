from contextlib import nullcontext, redirect_stderr, redirect_stdout
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from diarization_demo import DEFAULT_SAMPLE_RATE, Diarizer, VoiceTracker


class DiarizationDemoTests(unittest.TestCase):
    def test_overlapping_windows_track_voices_and_emit_only_new_times(self):
        def output(turns, embeddings=((1, 0), (0, 1))):
            annotation = Mock()
            annotation.itertracks.return_value = [
                (SimpleNamespace(start=a, end=b), 0, label) for a, b, label in turns
            ]
            annotation.labels.return_value = sorted({label for _, _, label in turns})
            return SimpleNamespace(speaker_diarization=annotation,
                                   exclusive_speaker_diarization=annotation,
                                   speaker_embeddings=np.asarray(embeddings, dtype=float))

        pipeline = Mock(side_effect=[
            output([(1, 4, "SPEAKER_00"), (4, 6, "SPEAKER_01")]),
            # Same voices, but local cluster labels have swapped in the overlap.
            output([(0, 3, "SPEAKER_00"), (3, 6, "SPEAKER_01")], [[0, 1], [1, 0]]),
        ])
        torch = Mock()
        torch.inference_mode.side_effect = nullcontext
        diarizer = Diarizer(pipeline, torch)
        samples = np.zeros(6 * DEFAULT_SAMPLE_RATE, dtype=np.float32)
        stdout = StringIO()
        with redirect_stdout(stdout), redirect_stderr(StringIO()):
            diarizer.process(samples, start=0)
            diarizer.process(samples, start=3)

        self.assertEqual(stdout.getvalue().splitlines(), [
            "[00:01.000-00:04.000] SPEAKER_00",
            "[00:04.000-00:06.000] SPEAKER_01",
            "[00:06.000-00:09.000] SPEAKER_00",
        ])
        self.assertEqual(diarizer.emitted_until, 9)
        for call in pipeline.call_args_list:
            self.assertEqual(call.kwargs, {"min_speakers": 1, "max_speakers": 2})
            self.assertEqual(call.args[0]["sample_rate"], 16000)

        # Exercise transcription without importing or downloading Whisper.
        pipeline.side_effect = [
            output([(1, 4, "SPEAKER_00"), (4, 6, "SPEAKER_01")]),
            output([(0, 3, "SPEAKER_00"), (3, 6, "SPEAKER_01")], [[0, 1], [1, 0]]),
        ]
        def segment(words):
            return [SimpleNamespace(words=[
                SimpleNamespace(start=a, end=b, word=text) for a, b, text in words
            ])]

        whisper = Mock()
        whisper.transcribe.side_effect = [
            (segment([(0, .5, "noise"), (1, 2, "Hello"), (4.2, 5, "Hi.")]), None),
            (segment([(1.2, 2, "Hi."), (3.2, 4, "How"), (4, 5, "are you?")]), None),
        ]
        diarizer = Diarizer(pipeline, torch, whisper)
        stdout = StringIO()
        with redirect_stdout(stdout), redirect_stderr(StringIO()):
            diarizer.process(samples, start=0)
            diarizer.process(samples, start=3)
        self.assertEqual(stdout.getvalue().splitlines(), [
            "[00:01.000-00:02.000] SPEAKER_00: Hello",
            "[00:04.200-00:05.000] SPEAKER_01: Hi.",
            "[00:06.200-00:08.000] SPEAKER_00: How are you?",
        ])
        self.assertTrue(whisper.transcribe.call_args.kwargs["word_timestamps"])

        # Session voice matching survives a gap and reversed local cluster IDs.
        voices = VoiceTracker()
        duplicate = output([(0, 3, "SPEAKER_00"), (3, 6, "SPEAKER_01")],
                           [[1, 0, 0], [1, .01, 0]])
        self.assertEqual(voices.assign(duplicate, [(0, 3, "SPEAKER_00"), (3, 6, "SPEAKER_01")]),
                         {"SPEAKER_00": "SPEAKER_00", "SPEAKER_01": "SPEAKER_00"})
        second = output([(0, 6, "SPEAKER_00")], [[0, 1, 0]])
        self.assertEqual(voices.assign(second, [(20, 26, "SPEAKER_00")]),
                         {"SPEAKER_00": "SPEAKER_01"})
        reversed_voices = output([(0, 3, "SPEAKER_00"), (3, 6, "SPEAKER_01")],
                                 [[0, 1, 0], [1, 0, 0]])
        self.assertEqual(voices.assign(reversed_voices, [(50, 53, "SPEAKER_00"), (53, 56, "SPEAKER_01")]),
                         {"SPEAKER_00": "SPEAKER_01", "SPEAKER_01": "SPEAKER_00"})
        for vector in ([1, 1, 0], [0, 0, 1], [0, 0, 0], [float("nan"), 0, 0]):
            unclear = output([(0, 6, "SPEAKER_00")], [vector])
            self.assertEqual(voices.assign(unclear, [(60, 66, "SPEAKER_00")]),
                             {"SPEAKER_00": "SPEAKER_UNKNOWN"})
        self.assertEqual(len(voices.profiles), 2)


if __name__ == "__main__":
    unittest.main()
