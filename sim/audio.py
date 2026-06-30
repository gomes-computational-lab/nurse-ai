from __future__ import annotations

import tempfile
import wave
from pathlib import Path


DEFAULT_SAMPLE_RATE = 16000


class AudioRecordingError(RuntimeError):
    pass


class AudioDependencyError(AudioRecordingError):
    pass


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
