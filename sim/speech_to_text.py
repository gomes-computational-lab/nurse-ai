from __future__ import annotations

from functools import lru_cache
from pathlib import Path


DEFAULT_STT_MODEL = "tiny.en"


class SpeechToTextError(RuntimeError):
    pass


class SpeechToTextDependencyError(SpeechToTextError):
    pass


def preload_speech_to_text_model(*, model_name: str = DEFAULT_STT_MODEL) -> None:
    _load_model(model_name)


def transcribe_audio(audio_path: Path, *, model_name: str = DEFAULT_STT_MODEL) -> str:
    model = _load_model(model_name)

    try:
        segments, _ = model.transcribe(
            str(audio_path),
            language="en",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,
        )
    except Exception as exc:
        raise SpeechToTextError(f"Transcription failed: {exc}") from exc

    transcript = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
    normalized = " ".join(transcript.split())

    if not any(character.isalnum() for character in normalized):
        return ""

    return normalized


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
