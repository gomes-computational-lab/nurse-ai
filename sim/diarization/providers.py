"""Lazy provider boundaries for Pyannote diarization and Whisper words."""

from __future__ import annotations

import math
import os
import warnings
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .config import DiarizationConfig
from .models import RecognizedWord


class ProviderError(RuntimeError):
    """Base error for a model provider."""


class ProviderDependencyError(ProviderError):
    """A provider's optional package is unavailable."""


class ProviderLoadError(ProviderError):
    """A provider could not initialize its model."""


class ProviderInferenceError(ProviderError):
    """An initialized provider could not process audio."""


@dataclass(frozen=True)
class LocalSpeakerTurn:
    """A provider-local acoustic cluster over a window-relative interval."""

    start: float
    end: float
    speaker_id: str

    def __post_init__(self) -> None:
        for name, value in (("start", self.start), ("end", self.end)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Local speaker turn {name} must be a finite number.")
            if value < 0:
                raise ValueError(f"Local speaker turn {name} must not be negative.")
        if self.end < self.start:
            raise ValueError("Local speaker turn end must not precede its start.")
        if not isinstance(self.speaker_id, str) or not self.speaker_id.strip():
            raise ValueError("Local speaker ID must be a non-empty string.")


@dataclass(frozen=True)
class DiarizationFrame:
    """Library-independent acoustic output for one audio window."""

    speaker_turns: tuple[LocalSpeakerTurn, ...]
    exclusive_speaker_turns: tuple[LocalSpeakerTurn, ...]
    speaker_labels: tuple[str, ...]
    speaker_embeddings: tuple[tuple[float, ...], ...] | None

    def __post_init__(self) -> None:
        turns = tuple(self.speaker_turns)
        exclusive_turns = tuple(self.exclusive_speaker_turns)
        labels = tuple(self.speaker_labels)
        if any(not isinstance(turn, LocalSpeakerTurn) for turn in turns + exclusive_turns):
            raise ValueError("Diarization turns must contain LocalSpeakerTurn values.")
        if any(not isinstance(label, str) or not label.strip() for label in labels):
            raise ValueError("Speaker labels must be non-empty strings.")
        object.__setattr__(self, "speaker_turns", turns)
        object.__setattr__(self, "exclusive_speaker_turns", exclusive_turns)
        object.__setattr__(self, "speaker_labels", labels)
        if self.speaker_embeddings is not None:
            embeddings = tuple(tuple(float(value) for value in row) for row in self.speaker_embeddings)
            object.__setattr__(self, "speaker_embeddings", embeddings)


@runtime_checkable
class DiarizationProvider(Protocol):
    def diarize(
        self,
        samples: np.ndarray,
        *,
        sample_rate: int,
        min_speakers: int,
        max_speakers: int,
    ) -> DiarizationFrame:
        """Return acoustic clusters with timestamps relative to this window."""


@runtime_checkable
class TranscriptionProvider(Protocol):
    def transcribe(
        self,
        samples: np.ndarray,
        *,
        sample_rate: int,
    ) -> tuple[RecognizedWord, ...]:
        """Return recognized words with timestamps relative to this window."""


def _redact_secret(detail: str, secret: str | None) -> str:
    return detail.replace(secret, "[REDACTED]") if secret else detail


def _annotation_turns(annotation: object) -> tuple[LocalSpeakerTurn, ...]:
    return tuple(
        LocalSpeakerTurn(float(turn.start), float(turn.end), str(speaker))
        for turn, _, speaker in annotation.itertracks(yield_label=True)  # type: ignore[attr-defined]
    )


def _speaker_embeddings(value: object) -> tuple[tuple[float, ...], ...] | None:
    if value is None:
        return None
    try:
        embeddings = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if embeddings.ndim != 2:
        return None
    return tuple(tuple(float(item) for item in row) for row in embeddings)


class PyannoteDiarizationProvider:
    """Adapter that keeps Pyannote and Torch objects behind a typed boundary."""

    def __init__(
        self,
        pipeline: object,
        torch_module: object,
        *,
        model_name: str,
        device_name: str,
    ):
        self._pipeline = pipeline
        self._torch = torch_module
        self.model_name = model_name
        self.device_name = device_name

    def diarize(
        self,
        samples: np.ndarray,
        *,
        sample_rate: int,
        min_speakers: int,
        max_speakers: int,
    ) -> DiarizationFrame:
        try:
            inference_mode: AbstractContextManager[object] = self._torch.inference_mode()  # type: ignore[attr-defined]
            with inference_mode:
                waveform = self._torch.from_numpy(samples).unsqueeze(0)  # type: ignore[attr-defined]
                output = self._pipeline(
                    {"waveform": waveform, "sample_rate": sample_rate},
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                )
            annotation = output.speaker_diarization
            exclusive_annotation = output.exclusive_speaker_diarization
            labels = tuple(str(label) for label in annotation.labels())
            return DiarizationFrame(
                speaker_turns=_annotation_turns(annotation),
                exclusive_speaker_turns=_annotation_turns(exclusive_annotation),
                speaker_labels=labels,
                speaker_embeddings=_speaker_embeddings(getattr(output, "speaker_embeddings", None)),
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderInferenceError(
                f"Pyannote inference failed for '{self.model_name}' ({type(exc).__name__}): {exc}"
            ) from exc


class WhisperTranscriptionProvider:
    """Adapter that converts faster-whisper output into timestamped words."""

    def __init__(self, model: object, config: DiarizationConfig):
        self._model = model
        self.model_name = config.whisper_model_name
        self.language = config.language

    def transcribe(
        self,
        samples: np.ndarray,
        *,
        sample_rate: int,
    ) -> tuple[RecognizedWord, ...]:
        try:
            segments, _ = self._model.transcribe(  # type: ignore[attr-defined]
                samples,
                language=self.language,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300, "speech_pad_ms": 200},
                word_timestamps=True,
            )
            window_end = len(samples) / sample_rate
            words: list[RecognizedWord] = []
            for segment in segments:
                for word in segment.words or ():
                    start = max(0.0, float(word.start))
                    end = min(window_end, float(word.end))
                    text = str(word.word).strip()
                    if end > start and text:
                        words.append(RecognizedWord(start, end, text))
            return tuple(words)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderInferenceError(
                f"Whisper inference failed for '{self.model_name}' ({type(exc).__name__}): {exc}"
            ) from exc


def load_pyannote_provider(
    config: DiarizationConfig,
    *,
    token: str | None = None,
) -> PyannoteDiarizationProvider:
    """Load Pyannote lazily so normal simulation paths never import Torch."""
    token = (token if token is not None else os.environ.get("HF_TOKEN", "")).strip()
    if not token:
        raise ProviderLoadError("HF_TOKEN is not set for the Pyannote model.")
    if not token.startswith("hf_") or any(character.isspace() for character in token):
        raise ProviderLoadError("HF_TOKEN must be a complete Hugging Face token beginning with hf_.")

    try:
        import torch
    except ImportError as exc:
        raise ProviderDependencyError(
            "Pyannote requires the isolated diarization dependencies, including Torch."
        ) from exc

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"\s*torchcodec is not installed correctly.*",
            )
            from pyannote.audio import Pipeline
    except ImportError as exc:
        raise ProviderDependencyError(
            "Pyannote requires `pyannote.audio` from demos/diarization/requirements.txt."
        ) from exc

    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline = Pipeline.from_pretrained(config.pyannote_model_name, token=token)
        if pipeline is None:
            raise RuntimeError("model access returned no pipeline")
        pipeline.to(device)
    except Exception as exc:
        detail = _redact_secret(str(exc), token)
        raise ProviderLoadError(
            f"Could not load Pyannote model '{config.pyannote_model_name}' "
            f"({type(exc).__name__}): {detail}"
        ) from None

    return PyannoteDiarizationProvider(
        pipeline,
        torch,
        model_name=config.pyannote_model_name,
        device_name=str(device),
    )


def load_whisper_provider(config: DiarizationConfig) -> WhisperTranscriptionProvider:
    """Load faster-whisper through the public timestamped provider boundary."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ProviderDependencyError(
            "Whisper requires `faster-whisper` from demos/diarization/requirements.txt."
        ) from exc

    try:
        model = WhisperModel(config.whisper_model_name, device="cpu", compute_type="int8")
    except Exception as exc:
        raise ProviderLoadError(
            f"Could not load Whisper model '{config.whisper_model_name}' "
            f"({type(exc).__name__}): {exc}"
        ) from None
    return WhisperTranscriptionProvider(model, config)
