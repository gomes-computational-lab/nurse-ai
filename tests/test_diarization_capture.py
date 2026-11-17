from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
import unittest

import numpy as np

from sim.diarization.capture import (
    AudioSnapshot,
    MicrophoneCapture,
    RollingAudio,
    list_input_devices,
    validate_input_device,
)


class FakeStream:
    def __init__(self, callback, sample_rate):
        self.callback = callback
        self.sample_rate = sample_rate
        self.active = True
        self.stopped = False

    def __enter__(self):
        self.callback(
            np.arange(self.sample_rate, dtype=np.float32).reshape(-1, 1),
            self.sample_rate,
            None,
            None,
        )
        return self

    def __exit__(self, *args):
        self.active = False
        return False

    def stop(self):
        self.stopped = True
        self.active = False


class FakeSoundDevice:
    class CallbackAbort(Exception):
        pass

    def __init__(self):
        self.default = SimpleNamespace(device=(1, None))
        self.stream: FakeStream | None = None
        self.checked = None

    def InputStream(self, *, callback, samplerate, **kwargs):
        del kwargs
        self.stream = FakeStream(callback, samplerate)
        return self.stream

    def query_devices(self, device=None, kind=None):
        if device is not None or kind is not None:
            return {"name": "Boom microphone"}
        return [
            {"name": "Speakers", "max_input_channels": 0, "hostapi": 0},
            {"name": "Boom microphone", "max_input_channels": 1, "hostapi": 0},
        ]

    def query_hostapis(self, index):
        self.assert_index = index
        return {"name": "Core Audio"}

    def check_input_settings(self, **kwargs):
        self.checked = kwargs


class CaptureTests(unittest.TestCase):
    def test_rolling_audio_retains_samples_in_time_order(self) -> None:
        audio = RollingAudio(5)
        audio.append(np.array([1, 2, 3], dtype=np.float32))
        first = audio.snapshot(3)
        self.assertIsNotNone(first)
        self.assertEqual(first[1], 3)
        np.testing.assert_array_equal(first[0], [1, 2, 3])

        audio.append(np.array([4, 5, 6, 7], dtype=np.float32))
        wrapped = audio.snapshot(7)
        self.assertIsNotNone(wrapped)
        self.assertEqual(wrapped[1], 7)
        np.testing.assert_array_equal(wrapped[0], [3, 4, 5, 6, 7])
        self.assertIsNone(audio.snapshot(8))

    def test_audio_snapshot_uses_capture_sample_clock(self) -> None:
        snapshot = AudioSnapshot(np.zeros(6 * 16_000), 7 * 16_000, 16_000)
        self.assertEqual(snapshot.start_seconds, 1)
        self.assertEqual(snapshot.end_seconds, 7)

    def test_microphone_capture_has_no_presentation_or_model_responsibility(self) -> None:
        sd = FakeSoundDevice()
        stdout = StringIO()

        with redirect_stdout(stdout):
            with MicrophoneCapture(
                sample_rate=10,
                capacity_samples=10,
                sounddevice_module=sd,
            ) as capture:
                capture.ensure_healthy()
                snapshot = capture.snapshot(10)
                self.assertIsNotNone(snapshot)
                np.testing.assert_array_equal(snapshot.samples, np.arange(10, dtype=np.float32))
                capture.stop()

        self.assertTrue(sd.stream.stopped)
        self.assertEqual(stdout.getvalue(), "")

    def test_device_functions_return_data_instead_of_printing(self) -> None:
        sd = FakeSoundDevice()
        stdout = StringIO()

        with redirect_stdout(stdout):
            devices = list_input_devices(sounddevice_module=sd)
            name = validate_input_device(1, sample_rate=16_000, sounddevice_module=sd)

        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0].id, 1)
        self.assertEqual(devices[0].name, "Boom microphone")
        self.assertTrue(devices[0].is_default)
        self.assertEqual(devices[0].host_api, "Core Audio")
        self.assertEqual(name, "Boom microphone")
        self.assertEqual(sd.checked["device"], 1)
        self.assertEqual(sd.checked["samplerate"], 16_000)
        self.assertEqual(stdout.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
