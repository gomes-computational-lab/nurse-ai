"""Bounded microphone capture without model or presentation responsibilities."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from types import TracebackType

import numpy as np


class AudioCaptureError(RuntimeError):
    """The selected microphone could not continue capturing audio."""


@dataclass(frozen=True)
class AudioSnapshot:
    samples: np.ndarray
    total_samples: int
    sample_rate: int

    @property
    def start_seconds(self) -> float:
        return (self.total_samples - len(self.samples)) / self.sample_rate

    @property
    def end_seconds(self) -> float:
        return self.total_samples / self.sample_rate


@dataclass(frozen=True)
class InputDevice:
    id: int
    name: str
    host_api: str
    is_default: bool


class RollingAudio:
    """Thread-safe circular storage retaining only the newest audio window."""

    def __init__(self, capacity: int):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("Audio capacity must be a positive sample count.")
        self._data = np.empty(capacity, dtype=np.float32)
        self._position = 0
        self._total = 0
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return len(self._data)

    @property
    def total_samples(self) -> int:
        with self._lock:
            return self._total

    def append(self, samples: np.ndarray) -> None:
        values = np.asarray(samples, dtype=np.float32)
        if values.ndim != 1:
            raise ValueError("Captured audio blocks must be one-dimensional.")
        with self._lock:
            count = len(values)
            self._total += count
            if count >= len(self._data):
                self._data[:] = values[-len(self._data):]
                self._position = 0
                return
            first = min(count, len(self._data) - self._position)
            self._data[self._position:self._position + first] = values[:first]
            self._data[:count - first] = values[first:]
            self._position = (self._position + count) % len(self._data)

    def snapshot(self, minimum_total: int) -> tuple[np.ndarray, int] | None:
        with self._lock:
            if self._total < minimum_total:
                return None
            if self._total < len(self._data):
                samples = self._data[:self._total].copy()
            else:
                samples = np.concatenate(
                    (self._data[self._position:], self._data[:self._position])
                )
            return samples, self._total


class MicrophoneCapture:
    """Context-managed SoundDevice stream backed by bounded rolling storage."""

    def __init__(
        self,
        *,
        sample_rate: int,
        capacity_samples: int,
        device: int | None = None,
        stall_seconds: float = 5.0,
        sounddevice_module: object | None = None,
    ):
        self.sample_rate = sample_rate
        self.device = device
        self.stall_seconds = stall_seconds
        self._sounddevice = sounddevice_module
        self._audio = RollingAudio(capacity_samples)
        self._stream_context: object | None = None
        self._stream: object | None = None
        self._capture_error: str | None = None
        self._last_capture = time.monotonic()

    def __enter__(self) -> MicrophoneCapture:
        if self._sounddevice is None:
            import sounddevice as sd

            self._sounddevice = sd
        self._stream_context = self._sounddevice.InputStream(  # type: ignore[attr-defined]
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.sample_rate // 10,
            device=self.device,
            callback=self._capture,
        )
        self._stream = self._stream_context.__enter__()  # type: ignore[attr-defined]
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        if self._stream_context is None:
            return None
        return self._stream_context.__exit__(exc_type, exc_value, traceback)  # type: ignore[attr-defined]

    @property
    def total_samples(self) -> int:
        return self._audio.total_samples

    def snapshot(self, minimum_total: int) -> AudioSnapshot | None:
        value = self._audio.snapshot(minimum_total)
        if value is None:
            return None
        samples, total = value
        return AudioSnapshot(samples, total, self.sample_rate)

    def ensure_healthy(self) -> None:
        active = bool(self._stream is not None and self._stream.active)  # type: ignore[attr-defined]
        if self._capture_error or not active:
            raise AudioCaptureError(
                f"Microphone stream stopped: {self._capture_error or 'device disconnected'}"
            )
        if time.monotonic() - self._last_capture > self.stall_seconds:
            raise AudioCaptureError(
                "Microphone stopped delivering audio. Check the selected input device."
            )

    def stop(self) -> None:
        if self._stream is not None and self._stream.active:  # type: ignore[attr-defined]
            self._stream.stop()  # type: ignore[attr-defined]

    def _capture(self, indata, frames, time_info, status) -> None:
        del frames, time_info
        if status:
            self._capture_error = str(status)
            raise self._sounddevice.CallbackAbort  # type: ignore[attr-defined]
        self._audio.append(indata[:, 0])
        self._last_capture = time.monotonic()


def list_input_devices(*, sounddevice_module: object | None = None) -> tuple[InputDevice, ...]:
    if sounddevice_module is None:
        import sounddevice as sd

        sounddevice_module = sd
    default = sounddevice_module.default.device[0]  # type: ignore[attr-defined]
    devices: list[InputDevice] = []
    for index, device in enumerate(sounddevice_module.query_devices()):  # type: ignore[attr-defined]
        if device["max_input_channels"]:
            api = sounddevice_module.query_hostapis(device["hostapi"])["name"]  # type: ignore[attr-defined]
            devices.append(InputDevice(index, device["name"], api, index == default))
    return tuple(devices)


def validate_input_device(
    device: int | None,
    *,
    sample_rate: int,
    sounddevice_module: object | None = None,
) -> str:
    if sounddevice_module is None:
        import sounddevice as sd

        sounddevice_module = sd
    sounddevice_module.check_input_settings(  # type: ignore[attr-defined]
        device=device,
        channels=1,
        dtype="float32",
        samplerate=sample_rate,
    )
    return str(sounddevice_module.query_devices(device, "input")["name"])  # type: ignore[attr-defined]
