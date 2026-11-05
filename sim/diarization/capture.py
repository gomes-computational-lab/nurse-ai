"""Bounded microphone capture without model or presentation responsibilities.

SoundDevice invokes its callback on an audio thread while model inference runs
on the main thread.  ``RollingAudio`` provides the small synchronized boundary
between those two execution contexts and retains only the newest window.  The
capture layer returns audio and timing data; it never runs a model or prints.
"""

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
    """An immutable description of one copied rolling-buffer snapshot.

    ``total_samples`` counts every sample observed since capture began, even
    when older samples have fallen out of the bounded buffer.  It therefore
    provides a monotonic clock independent of wall-clock scheduling delays.
    """

    samples: np.ndarray
    total_samples: int
    sample_rate: int

    @property
    def start_seconds(self) -> float:
        # Subtract the retained length because total_samples refers to the end
        # of this snapshot, not the beginning.
        return (self.total_samples - len(self.samples)) / self.sample_rate

    @property
    def end_seconds(self) -> float:
        return self.total_samples / self.sample_rate


@dataclass(frozen=True)
class InputDevice:
    """Presentation-neutral information about one microphone input."""

    id: int
    name: str
    host_api: str
    is_default: bool


class RollingAudio:
    """Thread-safe circular storage retaining only the newest audio window.

    Memory use is fixed at ``capacity`` samples.  Once full, new audio
    overwrites the oldest audio.  The monotonic total counter still advances,
    allowing the service to detect and report intervals that were overwritten
    before inference reached them.
    """

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
        """Append a callback block, overwriting the oldest data if necessary."""

        values = np.asarray(samples, dtype=np.float32)
        if values.ndim != 1:
            raise ValueError("Captured audio blocks must be one-dimensional.")
        with self._lock:
            count = len(values)
            self._total += count
            # A block larger than the entire buffer supersedes everything;
            # retain only its newest capacity-sized suffix.
            if count >= len(self._data):
                self._data[:] = values[-len(self._data):]
                self._position = 0
                return
            # Write up to the physical end, then wrap the remaining prefix to
            # index zero.  ``_position`` always points to the next write slot.
            first = min(count, len(self._data) - self._position)
            self._data[self._position:self._position + first] = values[:first]
            self._data[:count - first] = values[first:]
            self._position = (self._position + count) % len(self._data)

    def snapshot(self, minimum_total: int) -> tuple[np.ndarray, int] | None:
        """Return ordered audio once enough total samples have arrived.

        The returned array is always a copy.  The callback may therefore keep
        writing immediately without changing audio already being processed.
        """

        with self._lock:
            if self._total < minimum_total:
                return None
            if self._total < len(self._data):
                samples = self._data[:self._total].copy()
            else:
                # Once full, the oldest sample is at the next write position;
                # concatenate from there to restore chronological order.
                samples = np.concatenate(
                    (self._data[self._position:], self._data[:self._position])
                )
            return samples, self._total


class MicrophoneCapture:
    """Context-managed SoundDevice stream backed by bounded rolling storage.

    SoundDevice is accepted as an optional constructor dependency so tests can
    use a fake module.  Production imports it lazily in ``__enter__``.
    """

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
        # Keep SoundDevice out of standard imports and non-microphone commands.
        if self._sounddevice is None:
            import sounddevice as sd

            self._sounddevice = sd
        # Ten callbacks per second balances responsiveness with callback
        # overhead.  The callback only copies samples; expensive inference is
        # intentionally kept off the real-time audio thread.
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
        """Wrap a raw buffer snapshot with sample-clock timing metadata."""

        value = self._audio.snapshot(minimum_total)
        if value is None:
            return None
        samples, total = value
        return AudioSnapshot(samples, total, self.sample_rate)

    def ensure_healthy(self) -> None:
        """Fail explicitly if the stream stopped, aborted, or stalled."""

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
        """Stop accepting samples while retaining buffered audio for shutdown."""

        if self._stream is not None and self._stream.active:  # type: ignore[attr-defined]
            self._stream.stop()  # type: ignore[attr-defined]

    def _capture(self, indata, frames, time_info, status) -> None:
        """SoundDevice callback: validate status and copy one mono block."""

        del frames, time_info
        if status:
            self._capture_error = str(status)
            raise self._sounddevice.CallbackAbort  # type: ignore[attr-defined]
        self._audio.append(indata[:, 0])
        self._last_capture = time.monotonic()


def list_input_devices(*, sounddevice_module: object | None = None) -> tuple[InputDevice, ...]:
    """Return input-capable devices without printing or loading any models."""

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
    """Check mono/sample-rate support and return the selected device name."""

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
