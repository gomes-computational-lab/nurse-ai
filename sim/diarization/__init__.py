"""Public, dependency-light interfaces for standalone speaker diarization."""

from .config import (
    DEFAULT_PYANNOTE_MODEL,
    DEFAULT_WHISPER_MODEL,
    DiarizationConfig,
)
from .alignment import (
    align_words_to_speakers,
    to_session_speaker_turns,
    to_session_words,
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
from .tracking import UNKNOWN_SPEAKER_ID, VoiceTracker

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
    "UNKNOWN_SPEAKER_ID",
    "VoiceTracker",
    "WhisperTranscriptionProvider",
    "align_words_to_speakers",
    "load_pyannote_provider",
    "load_whisper_provider",
    "to_session_speaker_turns",
    "to_session_words",
]
