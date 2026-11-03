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
from .providers import (
    DiarizationFrame,
    DiarizationProvider,
    LocalSpeakerTurn,
    ProviderDependencyError,
    ProviderError,
    ProviderInferenceError,
    ProviderLoadError,
    PyannoteDiarizationProvider,
    TranscriptionProvider,
    WhisperTranscriptionProvider,
    load_pyannote_provider,
    load_whisper_provider,
)

__all__ = [
    "AudioGap",
    "DEFAULT_PYANNOTE_MODEL",
    "DEFAULT_WHISPER_MODEL",
    "DiarizationConfig",
    "DiarizationFrame",
    "DiarizationProvider",
    "LocalSpeakerTurn",
    "ProcessingDiagnostics",
    "ProcessingResult",
    "ProviderDependencyError",
    "ProviderError",
    "ProviderInferenceError",
    "ProviderLoadError",
    "PyannoteDiarizationProvider",
    "RecognizedWord",
    "SpeakerTurn",
    "TranscriptionProvider",
    "Utterance",
    "UtteranceStatus",
    "WhisperTranscriptionProvider",
    "load_pyannote_provider",
    "load_whisper_provider",
]
