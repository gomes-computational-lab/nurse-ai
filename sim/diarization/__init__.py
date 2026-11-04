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
from .capture import (
    AudioCaptureError,
    AudioSnapshot,
    InputDevice,
    MicrophoneCapture,
    RollingAudio,
    list_input_devices,
    validate_input_device,
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
from .service import (
    DiarizationService,
    DiarizationServiceClosedError,
    DiarizationServiceError,
)

__all__ = [
    "AudioGap",
    "AudioCaptureError",
    "AudioSnapshot",
    "DEFAULT_PYANNOTE_MODEL",
    "DEFAULT_WHISPER_MODEL",
    "DiarizationConfig",
    "DiarizationFrame",
    "DiarizationProvider",
    "DiarizationService",
    "DiarizationServiceClosedError",
    "DiarizationServiceError",
    "LocalSpeakerTurn",
    "InputDevice",
    "MicrophoneCapture",
    "ProcessingDiagnostics",
    "ProcessingResult",
    "ProviderDependencyError",
    "ProviderError",
    "ProviderInferenceError",
    "ProviderLoadError",
    "PyannoteDiarizationProvider",
    "RecognizedWord",
    "RollingAudio",
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
    "list_input_devices",
    "to_session_speaker_turns",
    "to_session_words",
    "validate_input_device",
]
