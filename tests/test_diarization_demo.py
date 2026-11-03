from contextlib import nullcontext, redirect_stderr, redirect_stdout
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from diarization_demo import DEFAULT_SAMPLE_RATE, Diarizer, VoiceTracker, main, run_live
from sim.diarization import DiarizationFrame, LocalSpeakerTurn


def tracking_frame(turns, embeddings):
    local_turns = tuple(LocalSpeakerTurn(a, b, label) for a, b, label in turns)
    return DiarizationFrame(
        speaker_turns=local_turns,
        exclusive_speaker_turns=local_turns,
        speaker_labels=tuple(sorted({label for _, _, label in turns})),
        speaker_embeddings=tuple(tuple(value for value in row) for row in embeddings),
    )


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
        samples = np.zeros(6 * DEFAULT_SAMPLE_RATE, dtype=np.float32)
        # Exercise transcription without importing or downloading Whisper.
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
        self.assertEqual(diarizer.emitted_until, 9)
        for call in pipeline.call_args_list:
            self.assertEqual(call.kwargs, {"min_speakers": 1, "max_speakers": 2})
            self.assertEqual(call.args[0]["sample_rate"], 16000)
        self.assertTrue(whisper.transcribe.call_args.kwargs["word_timestamps"])
        self.assertEqual(whisper.transcribe.call_args.kwargs["language"], "en")

        # Session voice matching survives a gap and reversed local cluster IDs.
        voices = VoiceTracker()
        duplicate = tracking_frame(
            [(0, 3, "SPEAKER_00"), (3, 6, "SPEAKER_01")],
            [[1, 0, 0], [1, .01, 0]],
        )
        self.assertEqual(voices.assign(duplicate),
                         {"SPEAKER_00": "SPEAKER_00", "SPEAKER_01": "SPEAKER_00"})
        second = tracking_frame([(0, 6, "SPEAKER_00")], [[0, 1, 0]])
        self.assertEqual(voices.assign(second),
                         {"SPEAKER_00": "SPEAKER_01"})
        reversed_voices = tracking_frame(
            [(0, 3, "SPEAKER_00"), (3, 6, "SPEAKER_01")],
            [[0, 1, 0], [1, 0, 0]],
        )
        self.assertEqual(voices.assign(reversed_voices),
                         {"SPEAKER_00": "SPEAKER_01", "SPEAKER_01": "SPEAKER_00"})
        for vector in ([1, 1, 0], [0, 0, 1], [0, 0, 0], [float("nan"), 0, 0]):
            unclear = tracking_frame([(0, 6, "SPEAKER_00")], [vector])
            self.assertEqual(voices.assign(unclear),
                             {"SPEAKER_00": "SPEAKER_UNKNOWN"})
        self.assertEqual(len(voices.profiles), 2)

    def test_live_command_loads_stt_and_preserves_window_and_step(self):
        sd = Mock()
        sd.query_devices.return_value = {"name": "Test microphone"}
        with (
            patch.dict("sys.modules", {"sounddevice": sd}),
            patch("diarization_demo.load_diarizer") as load,
            patch("diarization_demo.run_live") as live,
            redirect_stdout(StringIO()),
        ):
            self.assertEqual(main(["--window", "15", "--step", "3"]), 0)
        load.assert_called_once_with("tiny.en", inspect=False)
        live.assert_called_once_with(load.return_value, None, 240000, 48000)

    def test_ctrl_c_closes_capture_without_transcribing_buffered_tail(self):
        stream = Mock(active=True)
        stream.__exit__ = Mock(return_value=False)
        diarizer = Mock()

        def input_stream(**kwargs):
            def enter():
                # Enough audio for the first window; more arrives during inference.
                kwargs["callback"](np.ones((240000, 1), dtype=np.float32), 240000, None, None)
                return stream

            def process(*args):
                kwargs["callback"](np.ones((16000, 1), dtype=np.float32), 16000, None, None)

            stream.__enter__ = Mock(side_effect=enter)
            diarizer.process.side_effect = process
            return stream

        with (
            patch.dict("sys.modules", {"sounddevice": SimpleNamespace(InputStream=input_stream)}),
            patch("diarization_demo.time.sleep", side_effect=KeyboardInterrupt),
            redirect_stdout(StringIO()),
            self.assertRaises(KeyboardInterrupt),
        ):
            run_live(diarizer, None, 240000, 48000)
        diarizer.process.assert_called_once()
        self.assertEqual(len(diarizer.process.call_args.args[0]), 240000)
        self.assertEqual(diarizer.process.call_args.args[1], 0)
        self.assertEqual(diarizer.process.call_args.kwargs, {})
        stream.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()
