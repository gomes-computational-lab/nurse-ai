from __future__ import annotations

import math
import queue
import tempfile
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


DEFAULT_SAMPLE_RATE = 16000
DEFAULT_FRAME_DURATION_MS = 30
DEFAULT_PRE_ROLL_MS = 300
DEFAULT_END_SILENCE_MS = 700
DEFAULT_NO_SPEECH_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_RECORDING_SECONDS = 60.0
DEFAULT_MIN_SPEECH_MS = 250

RecordingStopReason = Literal["silence", "manual", "no_speech", "max_duration"]


@dataclass(frozen=True)
class RecordedAudio:
    samples: np.ndarray
    sample_rate: int
    captured_seconds: float
    speech_seconds: float
    endpoint_delay_seconds: float | None
    stop_reason: RecordingStopReason

    @property
    def has_speech(self) -> bool:
        return self.samples.size > 0 and self.speech_seconds > 0


class AudioRecordingError(RuntimeError):
    pass


class AudioDependencyError(AudioRecordingError):
    pass


class VoiceActivityDetectionUnavailable(AudioDependencyError):
    pass


def record_microphone_until_silence(
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = 1,
    vad_mode: int = 2,
    frame_duration_ms: int = DEFAULT_FRAME_DURATION_MS,
    pre_roll_ms: int = DEFAULT_PRE_ROLL_MS,
    end_silence_ms: int = DEFAULT_END_SILENCE_MS,
    no_speech_timeout_seconds: float = DEFAULT_NO_SPEECH_TIMEOUT_SECONDS,
    max_recording_seconds: float = DEFAULT_MAX_RECORDING_SECONDS,
    min_speech_ms: int = DEFAULT_MIN_SPEECH_MS,
    stop_event: threading.Event | None = None,
) -> RecordedAudio:
    if channels != 1:
        raise ValueError("Voice activity detection requires mono audio.")
    if frame_duration_ms not in (10, 20, 30):
        raise ValueError("WebRTC VAD frames must be 10, 20, or 30 milliseconds.")

    try:
        import sounddevice as sd
    except ImportError as exc:
        raise AudioDependencyError(
            "Missing audio dependency `sounddevice`. Install requirements with `pip install -r requirements.txt`."
        ) from exc

    try:
        import webrtcvad
    except ImportError as exc:
        raise VoiceActivityDetectionUnavailable(
            "Missing voice activity dependency `webrtcvad-wheels`."
        ) from exc

    try:
        vad = webrtcvad.Vad(vad_mode)
    except Exception as exc:
        raise VoiceActivityDetectionUnavailable(f"Could not initialize voice activity detection: {exc}") from exc

    stop_event = stop_event or threading.Event()
    frame_samples = sample_rate * frame_duration_ms // 1000
    frame_seconds = frame_duration_ms / 1000.0
    pre_roll_frames = max(1, math.ceil(pre_roll_ms / frame_duration_ms))
    trailing_frames_required = max(1, math.ceil(end_silence_ms / frame_duration_ms))
    minimum_voiced_frames = max(1, math.ceil(min_speech_ms / frame_duration_ms))
    start_frames_required = min(3, minimum_voiced_frames)

    incoming: queue.Queue[bytes] = queue.Queue()
    callback_error: list[str] = []

    def capture(indata, frame_count, time_info, status) -> None:
        del time_info
        if status and not callback_error:
            callback_error.append(str(status))
        if frame_count == frame_samples:
            incoming.put(bytes(indata))

    frames: list[bytes] = []
    pre_roll: deque[bytes] = deque(maxlen=pre_roll_frames)
    recording_started = time.perf_counter()
    speech_started = False
    consecutive_voiced = 0
    trailing_silence_frames = 0
    voiced_frames = 0
    last_speech_at: float | None = None
    stop_reason: RecordingStopReason = "no_speech"

    try:
        with sd.RawInputStream(
            samplerate=sample_rate,
            blocksize=frame_samples,
            channels=channels,
            dtype="int16",
            callback=capture,
        ):
            while True:
                now = time.perf_counter()
                elapsed = now - recording_started
                if stop_event.is_set():
                    stop_reason = "manual"
                    break
                if elapsed >= max_recording_seconds:
                    stop_reason = "max_duration"
                    break
                if not speech_started and elapsed >= no_speech_timeout_seconds:
                    stop_reason = "no_speech"
                    break

                try:
                    frame = incoming.get(timeout=0.05)
                except queue.Empty:
                    continue

                try:
                    is_speech = bool(vad.is_speech(frame, sample_rate))
                except Exception as exc:
                    raise VoiceActivityDetectionUnavailable(f"Voice activity detection failed: {exc}") from exc

                now = time.perf_counter()
                if not speech_started:
                    pre_roll.append(frame)
                    consecutive_voiced = consecutive_voiced + 1 if is_speech else 0
                    if consecutive_voiced < start_frames_required:
                        continue

                    speech_started = True
                    frames.extend(pre_roll)
                    pre_roll.clear()
                    voiced_frames = consecutive_voiced
                    last_speech_at = now
                    trailing_silence_frames = 0
                    continue

                frames.append(frame)
                if is_speech:
                    voiced_frames += 1
                    last_speech_at = now
                    trailing_silence_frames = 0
                    continue

                trailing_silence_frames += 1
                if trailing_silence_frames < trailing_frames_required:
                    continue

                if voiced_frames >= minimum_voiced_frames:
                    stop_reason = "silence"
                    break

                # Treat a short noise burst as a false start and keep listening.
                speech_started = False
                consecutive_voiced = 0
                trailing_silence_frames = 0
                voiced_frames = 0
                last_speech_at = None
                frames.clear()
    except VoiceActivityDetectionUnavailable:
        raise
    except Exception as exc:
        detail = f" ({callback_error[0]})" if callback_error else ""
        raise AudioRecordingError(
            f"Microphone recording failed{detail}. Check microphone permissions and the default input device."
        ) from exc

    stopped_at = time.perf_counter()
    if not speech_started or voiced_frames < minimum_voiced_frames:
        frames.clear()
        voiced_frames = 0
        last_speech_at = None

    return _recorded_audio_from_frames(
        frames,
        sample_rate=sample_rate,
        voiced_frames=voiced_frames,
        frame_seconds=frame_seconds,
        stop_reason=stop_reason,
        endpoint_delay_seconds=(
            max(0.0, stopped_at - last_speech_at) if last_speech_at is not None else None
        ),
    )


def record_microphone_until_stopped(
    stop_event: threading.Event,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = 1,
) -> RecordedAudio:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise AudioDependencyError(
            "Missing audio dependency `sounddevice`. Install requirements with `pip install -r requirements.txt`."
        ) from exc

    frames: list[bytes] = []
    callback_stopped = threading.Event()

    def capture(indata, frame_count, time_info, status) -> None:
        del frame_count, time_info, status
        frames.append(indata.copy().tobytes())
        if stop_event.is_set():
            callback_stopped.set()
            raise sd.CallbackStop()

    try:
        with sd.InputStream(samplerate=sample_rate, channels=channels, dtype="int16", callback=capture):
            while not stop_event.wait(0.05):
                pass
            if not callback_stopped.wait(1.0):
                callback_stopped.set()
    except Exception as exc:
        raise AudioRecordingError(
            "Microphone recording failed. Check microphone permissions and the default input device."
        ) from exc

    return _recorded_audio_from_frames(
        frames,
        sample_rate=sample_rate,
        voiced_frames=None,
        frame_seconds=None,
        stop_reason="manual",
        endpoint_delay_seconds=0.0,
    )


def _recorded_audio_from_frames(
    frames: list[bytes],
    *,
    sample_rate: int,
    voiced_frames: int | None,
    frame_seconds: float | None,
    stop_reason: RecordingStopReason,
    endpoint_delay_seconds: float | None,
) -> RecordedAudio:
    audio_bytes = b"".join(frames)
    if audio_bytes:
        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        samples = np.empty(0, dtype=np.float32)

    captured_seconds = samples.size / sample_rate
    speech_seconds = (
        voiced_frames * frame_seconds
        if voiced_frames is not None and frame_seconds is not None
        else captured_seconds
    )
    return RecordedAudio(
        samples=samples,
        sample_rate=sample_rate,
        captured_seconds=captured_seconds,
        speech_seconds=speech_seconds,
        endpoint_delay_seconds=endpoint_delay_seconds,
        stop_reason=stop_reason,
    )


def record_microphone_clip(
    duration_seconds: float,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = 1,
) -> Path:
    if duration_seconds <= 0:
        raise ValueError("Recording duration must be greater than zero.")

    try:
        import sounddevice as sd
    except ImportError as exc:
        raise AudioDependencyError(
            "Missing audio dependency `sounddevice`. Install requirements with `pip install -r requirements.txt`."
        ) from exc

    try:
        frame_count = int(duration_seconds * sample_rate)
        recording = sd.rec(frame_count, samplerate=sample_rate, channels=channels, dtype="int16")
        sd.wait()
    except Exception as exc:
        raise AudioRecordingError(
            "Microphone recording failed. Check microphone permissions and the default input device."
        ) from exc

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    with wave.open(str(temp_path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(recording.tobytes())

    return temp_path
