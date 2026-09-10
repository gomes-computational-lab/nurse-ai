from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import tempfile
from typing import BinaryIO

import numpy as np


DEFAULT_STT_MODEL = "tiny.en"


class SpeechToTextError(RuntimeError):
    pass


class SpeechToTextDependencyError(SpeechToTextError):
    pass


def preload_speech_to_text_model(*, model_name: str = DEFAULT_STT_MODEL) -> None:
    _load_model(model_name)


def transcribe_audio(audio: Path | np.ndarray, *, model_name: str = DEFAULT_STT_MODEL) -> str:
    model = _load_model(model_name)

    try:
        segments, _ = model.transcribe(
            str(audio) if isinstance(audio, Path) else audio,
            language="en",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 300,
                "speech_pad_ms": 200,
            },
        )
    except Exception as exc:
        raise SpeechToTextError(f"Transcription failed: {exc}") from exc

    transcript = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
    normalized = " ".join(transcript.split())

    if not any(character.isalnum() for character in normalized):
        return ""

    return normalized


def transcribe_audio_bytes(
    audio: bytes | bytearray | BinaryIO,
    *,
    model_name: str = DEFAULT_STT_MODEL,
    suffix: str = ".wav",
) -> str:
    """Transcribe uploaded audio without making callers manage a temporary file."""
    if hasattr(audio, "read"):
        raw = audio.read()
    else:
        raw = bytes(audio)

    if not raw:
        return ""

    path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(raw)
            path = Path(temp_file.name)
        return transcribe_audio(path, model_name=model_name)
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise SpeechToTextDependencyError(
            "Missing speech-to-text dependency `faster-whisper`. Install requirements with `pip install -r requirements.txt`."
        ) from exc

    try:
        return WhisperModel(model_name, device="cpu", compute_type="int8")
    except Exception as exc:
        raise SpeechToTextError(f"Could not load STT model '{model_name}': {exc}") from exc
