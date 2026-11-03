"""Public, dependency-light interfaces for standalone speaker diarization."""

from .config import (
    DEFAULT_PYANNOTE_MODEL,
    DEFAULT_WHISPER_MODEL,
    DiarizationConfig,
)
from .models import (
    AudioGap,
    ProcessingDiagnostics,
    ProcessingResult,
    RecognizedWord,
    SpeakerTurn,
    Utterance,
    UtteranceStatus,
)

__all__ = [
    "AudioGap",
    "DEFAULT_PYANNOTE_MODEL",
    "DEFAULT_WHISPER_MODEL",
    "DiarizationConfig",
    "ProcessingDiagnostics",
    "ProcessingResult",
    "RecognizedWord",
    "SpeakerTurn",
    "Utterance",
    "UtteranceStatus",
]
