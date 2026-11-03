"""Typed, library-independent results produced by live diarization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from typing import Iterable, TypeVar


class UtteranceStatus(str, Enum):
    """Whether an utterance may still be revised by a later audio window."""

    PROVISIONAL = "provisional"
    FINALIZED = "finalized"


def _validate_time_range(start: float, end: float, *, label: str) -> None:
    for name, value in (("start", start), ("end", end)):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{label} {name} must be a finite number.")
        if value < 0:
            raise ValueError(f"{label} {name} must not be negative.")
    if end < start:
        raise ValueError(f"{label} end must not precede its start.")


def _validate_nonempty(value: str, *, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")


def _validate_nonnegative_number(value: float, *, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number.")
    if value < 0:
        raise ValueError(f"{label} must not be negative.")


def _validate_nonnegative_count(value: int, *, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer.")


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker_id: str
    source_speaker_id: str

    def __post_init__(self) -> None:
        _validate_time_range(self.start, self.end, label="Speaker turn")
        _validate_nonempty(self.speaker_id, label="Speaker ID")
        _validate_nonempty(self.source_speaker_id, label="Source speaker ID")


@dataclass(frozen=True)
class RecognizedWord:
    start: float
    end: float
    text: str

    def __post_init__(self) -> None:
        _validate_time_range(self.start, self.end, label="Recognized word")
        _validate_nonempty(self.text, label="Recognized word text")


@dataclass(frozen=True)
class Utterance:
    start: float
    end: float
    speaker_id: str
    text: str
    status: UtteranceStatus

    def __post_init__(self) -> None:
        _validate_time_range(self.start, self.end, label="Utterance")
        _validate_nonempty(self.speaker_id, label="Speaker ID")
        _validate_nonempty(self.text, label="Utterance text")
        if not isinstance(self.status, UtteranceStatus):
            raise ValueError("Utterance status must be an UtteranceStatus value.")


@dataclass(frozen=True)
class AudioGap:
    start: float
    end: float
    reason: str

    def __post_init__(self) -> None:
        _validate_time_range(self.start, self.end, label="Audio gap")
        _validate_nonempty(self.reason, label="Audio gap reason")


@dataclass(frozen=True)
class ProcessingDiagnostics:
    window_start: float
    window_end: float
    processing_seconds: float
    input_rms_dbfs: float
    input_peak_dbfs: float
    local_speaker_count: int
    tracked_speaker_count: int
    profile_count: int

    def __post_init__(self) -> None:
        _validate_time_range(self.window_start, self.window_end, label="Processing window")
        _validate_nonnegative_number(self.processing_seconds, label="Processing duration")
        for name, value in (
            ("Input RMS dBFS", self.input_rms_dbfs),
            ("Input peak dBFS", self.input_peak_dbfs),
        ):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
        _validate_nonnegative_count(self.local_speaker_count, label="Local speaker count")
        _validate_nonnegative_count(self.tracked_speaker_count, label="Tracked speaker count")
        _validate_nonnegative_count(self.profile_count, label="Profile count")


_ResultItem = TypeVar("_ResultItem")


def _as_typed_tuple(
    values: Iterable[_ResultItem],
    expected_type: type[_ResultItem],
    *,
    label: str,
) -> tuple[_ResultItem, ...]:
    try:
        result = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{label} must be an iterable.") from exc
    if any(not isinstance(value, expected_type) for value in result):
        raise ValueError(f"{label} contains an invalid item.")
    return result


@dataclass(frozen=True)
class ProcessingResult:
    """One processing update with append-only final and replaceable provisional text."""

    speaker_turns: tuple[SpeakerTurn, ...]
    recognized_words: tuple[RecognizedWord, ...]
    finalized_utterances: tuple[Utterance, ...]
    provisional_utterances: tuple[Utterance, ...]
    audio_gaps: tuple[AudioGap, ...]
    diagnostics: ProcessingDiagnostics

    def __post_init__(self) -> None:
        collections = (
            ("speaker_turns", SpeakerTurn, "Speaker turns"),
            ("recognized_words", RecognizedWord, "Recognized words"),
            ("finalized_utterances", Utterance, "Finalized utterances"),
            ("provisional_utterances", Utterance, "Provisional utterances"),
            ("audio_gaps", AudioGap, "Audio gaps"),
        )
        for field_name, expected_type, label in collections:
            value = _as_typed_tuple(getattr(self, field_name), expected_type, label=label)
            object.__setattr__(self, field_name, value)

        if not isinstance(self.diagnostics, ProcessingDiagnostics):
            raise ValueError("Diagnostics must be a ProcessingDiagnostics value.")
        if any(
            utterance.status is not UtteranceStatus.FINALIZED
            for utterance in self.finalized_utterances
        ):
            raise ValueError("Finalized utterances must have FINALIZED status.")
        if any(
            utterance.status is not UtteranceStatus.PROVISIONAL
            for utterance in self.provisional_utterances
        ):
            raise ValueError("Provisional utterances must have PROVISIONAL status.")
