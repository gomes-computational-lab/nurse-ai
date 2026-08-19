from __future__ import annotations

import sys
import types
import unittest
from collections import deque
from unittest.mock import patch

import numpy as np

from sim.audio import RecordedAudio, record_microphone_until_silence


class FakeClock:
    def __init__(self, step: float = 0.001):
        self.value = -step
        self.step = step

    def __call__(self) -> float:
        self.value += self.step
        return self.value


def fake_audio_modules(flags: list[bool]):
    vad_flags = deque(flags)

    class Vad:
        def __init__(self, mode: int):
            self.mode = mode

        def is_speech(self, frame: bytes, sample_rate: int) -> bool:
            del frame, sample_rate
            return vad_flags.popleft() if vad_flags else False

    webrtcvad = types.SimpleNamespace(Vad=Vad)

    class RawInputStream:
        def __init__(self, *, callback, blocksize, **kwargs):
            del kwargs
            self.callback = callback
            self.blocksize = blocksize

        def __enter__(self):
            frame = b"\x00\x00" * self.blocksize
            for _ in flags:
                self.callback(frame, self.blocksize, None, False)
            return self

        def __exit__(self, *args):
            return False

    sounddevice = types.SimpleNamespace(RawInputStream=RawInputStream)
    return {"sounddevice": sounddevice, "webrtcvad": webrtcvad}


class AudioRecordingTests(unittest.TestCase):
    def test_vad_keeps_pre_roll_and_allows_short_internal_pause(self) -> None:
        flags = [False] * 12 + [True] * 10 + [False] * 5 + [True] * 10 + [False] * 24

        with patch.dict(sys.modules, fake_audio_modules(flags)):
            audio = record_microphone_until_silence()

        self.assertIsInstance(audio, RecordedAudio)
        self.assertEqual(audio.stop_reason, "silence")
        self.assertTrue(audio.has_speech)
        self.assertAlmostEqual(audio.speech_seconds, 0.6, places=2)
        self.assertGreaterEqual(audio.captured_seconds, 1.6)
        self.assertEqual(audio.samples.dtype, np.float32)

    def test_no_speech_timeout_returns_empty_audio(self) -> None:
        flags = [False] * 5

        with (
            patch.dict(sys.modules, fake_audio_modules(flags)),
            patch("sim.audio.time.perf_counter", new=FakeClock(step=0.01)),
        ):
            audio = record_microphone_until_silence(no_speech_timeout_seconds=0.08)

        self.assertEqual(audio.stop_reason, "no_speech")
        self.assertFalse(audio.has_speech)
        self.assertEqual(audio.samples.size, 0)

    def test_maximum_duration_stops_continuous_speech(self) -> None:
        flags = [True] * 20

        with (
            patch.dict(sys.modules, fake_audio_modules(flags)),
            patch("sim.audio.time.perf_counter", new=FakeClock(step=0.01)),
        ):
            audio = record_microphone_until_silence(
                min_speech_ms=30,
                max_recording_seconds=0.08,
            )

        self.assertEqual(audio.stop_reason, "max_duration")
        self.assertTrue(audio.has_speech)


if __name__ == "__main__":
    unittest.main()
