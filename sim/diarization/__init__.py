"""Public, dependency-light interfaces for standalone speaker diarization.

Importing this package is safe from the normal simulation environment: the
modules below import only the standard library and NumPy.  Torch, Pyannote,
faster-whisper, and SoundDevice are imported lazily when a caller explicitly
loads a model or opens a microphone.

The exports are intentionally library-independent wherever possible so the
service can later be reused by a UI without exposing third-party model types.
"""

# Re-export the supported public surface here.  Callers should not need to
# know which internal module owns a model, provider boundary, or helper.

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
