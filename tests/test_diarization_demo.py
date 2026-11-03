from contextlib import nullcontext, redirect_stderr, redirect_stdout
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from diarization_demo import DEFAULT_SAMPLE_RATE, ConversationRoles, Diarizer, VoiceTracker


class DiarizationDemoTests(unittest.TestCase):
    def test_overlapping_windows_align_labels_and_emit_only_new_times(self):
        def output(turns):
            annotation = Mock()
            annotation.itertracks.return_value = [
                (SimpleNamespace(start=a, end=b), 0, label) for a, b, label in turns
            ]
            return SimpleNamespace(speaker_diarization=annotation)

        pipeline = Mock(side_effect=[
            output([(1, 4, "SPEAKER_00"), (4, 6, "SPEAKER_01")]),
            # Same voices, but local cluster labels have swapped in the overlap.
            output([(0, 3, "SPEAKER_00"), (3, 6, "SPEAKER_01")]),
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
            self.assertEqual(call.kwargs, {"min_speakers": 2, "max_speakers": 2})
            self.assertEqual(call.args[0]["sample_rate"], 16000)

        # Exercise transcription without importing or downloading Whisper.
        pipeline.side_effect = [
            output([(1, 4, "SPEAKER_00"), (4, 6, "SPEAKER_01")]),
            output([(0, 3, "SPEAKER_00"), (3, 6, "SPEAKER_01")]),
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
        def voiced_output(turns, embeddings):
            result = output(turns)
            result.speaker_diarization.labels.return_value = sorted({label for _, _, label in turns})
            result.exclusive_speaker_diarization = result.speaker_diarization
            result.speaker_embeddings = np.array(embeddings, dtype=float)
            return result

        voices = VoiceTracker()
        duplicate = voiced_output([(0, 3, "SPEAKER_00"), (3, 6, "SPEAKER_01")],
                                 [[1, 0, 0], [1, .01, 0]])
        self.assertEqual(voices.assign(duplicate, [(0, 3, "SPEAKER_00"), (3, 6, "SPEAKER_01")]),
                         {"SPEAKER_00": "SPEAKER_00", "SPEAKER_01": "SPEAKER_00"})
        second = voiced_output([(0, 6, "SPEAKER_00")], [[0, 1, 0]])
        self.assertEqual(voices.assign(second, [(20, 26, "SPEAKER_00")]),
                         {"SPEAKER_00": "SPEAKER_01"})
        reversed_voices = voiced_output([(0, 3, "SPEAKER_00"), (3, 6, "SPEAKER_01")],
                                       [[0, 1, 0], [1, 0, 0]])
        self.assertEqual(voices.assign(reversed_voices, [(50, 53, "SPEAKER_00"), (53, 56, "SPEAKER_01")]),
                         {"SPEAKER_00": "SPEAKER_01", "SPEAKER_01": "SPEAKER_00"})
        for vector in ([1, 1, 0], [0, 0, 1], [0, 0, 0], [float("nan"), 0, 0]):
            unclear = voiced_output([(0, 6, "SPEAKER_00")], [vector])
            self.assertEqual(voices.assign(unclear, [(60, 66, "SPEAKER_00")]),
                             {"SPEAKER_00": "SPEAKER_UNKNOWN"})
        self.assertEqual(len(voices.profiles), 2)

        # No forced role from greetings or mixed nurse/patient text.
        roles = ConversationRoles()
        self.assertFalse(roles.observe([(0, 2, "SPEAKER_00", "Hello. Good morning.")]))
        self.assertEqual(roles.describe("SPEAKER_00"), "SPEAKER_00 [ROLE_UNKNOWN]")
        self.assertFalse(roles.observe([(3, 6, "SPEAKER_00",
                                        "Let me check your blood pressure. I'm not feeling well.")]))
        self.assertEqual(sum(map(len, roles.evidence["SPEAKER_00"])), 0)

        # Same processing path prints inferred roles without any enrollment.
        pipeline.side_effect = [
            voiced_output([(0, 6, "SPEAKER_00")], [[1, 0]]),
            voiced_output([(0, 6, "SPEAKER_00")], [[0, 1]]),
        ]
        whisper.transcribe.side_effect = [
            (segment([(1, 3, "Let me check your blood pressure.")]), None),
            (segment([(1, 4, "I'm not feeling well. I've had a fever.")]), None),
        ]
        diarizer = Diarizer(pipeline, torch, whisper, nurse_patient=True)
        stdout = StringIO()
        with redirect_stdout(stdout), redirect_stderr(StringIO()):
            diarizer.process(samples, start=0)
            diarizer.process(samples, start=9)
        self.assertEqual(stdout.getvalue().splitlines(), [
            "[00:01.000-00:03.000] SPEAKER_00 [ROLE_UNKNOWN]: Let me check your blood pressure.",
            "Role estimate updated: SPEAKER_00=NURSE; SPEAKER_01=PATIENT. Earlier lines are unchanged.",
            "[00:10.000-00:13.000] SPEAKER_01 [PATIENT, estimated]: I'm not feeling well. I've had a fever.",
        ])
        self.assertEqual(diarizer.roles.describe("SPEAKER_00"), "SPEAKER_00 [NURSE, estimated]")
        # Unknown acoustic identities must never contribute role evidence.
        self.assertFalse(diarizer.roles.observe([(20, 25, "SPEAKER_UNKNOWN", "Let me check your temperature.")]))


if __name__ == "__main__":
    unittest.main()
