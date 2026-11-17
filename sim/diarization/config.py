"""Validated configuration for the standalone diarization service.

Keeping every tuning value here makes model adapters and processing classes
deterministic and easy to test.  The configuration is frozen so a running
session cannot silently change thresholds halfway through voice tracking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real


DEFAULT_PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"
DEFAULT_WHISPER_MODEL = "tiny.en"


def _require_nonempty_string(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")


def _require_integer(name: str, value: int, *, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}.")


def _require_finite(
    name: str,
    value: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_exclusive: bool = False,
) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    if minimum is not None and (
        value <= minimum if minimum_exclusive else value < minimum
    ):
        qualifier = "greater than" if minimum_exclusive else "at least"
        raise ValueError(f"{name} must be {qualifier} {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must not exceed {maximum}.")


@dataclass(frozen=True)
class DiarizationConfig:
    """Models, streaming behavior, and acoustic voice-matching heuristics.

    Similarity values are cosine similarities in the inclusive range [-1, 1].
    A margin is the difference between the best and second-best profile score;
    requiring a margin prevents a weak tie from being treated as a confident
    identity match.
    """

    # Model and audio settings.  Whisper receives the language explicitly so
    # it does not spend time auto-detecting the language for every window.
    pyannote_model_name: str = DEFAULT_PYANNOTE_MODEL
    whisper_model_name: str = DEFAULT_WHISPER_MODEL
    language: str = "en"
    sample_rate: int = 16_000

    # Capture is rolling and overlapping.  With the defaults the analyzed
    # windows are 0-6s, 3-9s, 6-12s, and so on.  The newest three seconds stay
    # provisional until they appear with more context in the next window.
    window_seconds: float = 6.0
    step_seconds: float = 3.0
    finalization_delay_seconds: float = 3.0

    # These limits describe acoustic clusters, not nurse/patient roles.
    min_speakers: int = 1
    max_speakers: int = 2
    max_voice_profiles: int = 2

    # A local voice must contain enough speech before it may seed or update a
    # session profile; very short sounds produce unreliable embeddings.
    profile_seed_min_speech_seconds: float = 1.0

    # A low maximum similarity allows a sufficiently different embedding to
    # create a new profile.  A separate, higher minimum accepts an existing
    # match.  The intentional gap between them is an uncertainty region.
    new_profile_max_similarity: float = 0.50
    match_min_similarity: float = 0.60
    match_min_margin: float = 0.10

    # Updating a stored voice profile is stricter than merely matching it.
    # Only five percent of a confident new vector is blended in by default,
    # which lets profiles adapt slowly without drifting toward another voice.
    profile_update_min_similarity: float = 0.75
    profile_update_min_margin: float = 0.20
    profile_update_weight: float = 0.05

    def __post_init__(self) -> None:
        # Validate related values together here rather than allowing failures
        # deep inside a model invocation after expensive models have loaded.
        _require_nonempty_string("Pyannote model name", self.pyannote_model_name)
        _require_nonempty_string("Whisper model name", self.whisper_model_name)
        _require_nonempty_string("Language", self.language)
        _require_integer("Sample rate", self.sample_rate)

        _require_finite("Window seconds", self.window_seconds, minimum=0, minimum_exclusive=True)
        _require_finite("Step seconds", self.step_seconds, minimum=0, minimum_exclusive=True)
        _require_finite("Finalization delay seconds", self.finalization_delay_seconds, minimum=0)
        if self.step_seconds > self.window_seconds:
            raise ValueError("Step seconds must not exceed window seconds.")

        # Text can only be safely reconsidered while it remains inside the
        # overlap shared with the next window.
        if self.finalization_delay_seconds > self.window_seconds - self.step_seconds:
            raise ValueError(
                "Finalization delay seconds must not exceed window seconds minus step seconds."
            )

        _require_integer("Minimum speakers", self.min_speakers)
        _require_integer("Maximum speakers", self.max_speakers)
        _require_integer("Maximum voice profiles", self.max_voice_profiles)
        if self.min_speakers > self.max_speakers:
            raise ValueError("Minimum speakers must not exceed maximum speakers.")
        if self.max_voice_profiles < self.min_speakers:
            raise ValueError("Maximum voice profiles must cover the minimum speaker count.")

        _require_finite(
            "Profile seed minimum speech seconds",
            self.profile_seed_min_speech_seconds,
            minimum=0,
            minimum_exclusive=True,
        )
        for name, value in (
            ("New-profile maximum similarity", self.new_profile_max_similarity),
            ("Match minimum similarity", self.match_min_similarity),
            ("Profile-update minimum similarity", self.profile_update_min_similarity),
        ):
            _require_finite(name, value, minimum=-1, maximum=1)
        for name, value in (
            ("Match minimum margin", self.match_min_margin),
            ("Profile-update minimum margin", self.profile_update_min_margin),
        ):
            _require_finite(name, value, minimum=0, maximum=2)
        _require_finite(
            "Profile-update weight",
            self.profile_update_weight,
            minimum=0,
            maximum=1,
            minimum_exclusive=True,
        )

        # Preserve a non-overlapping decision order: create, uncertain, match,
        # and finally (under stricter conditions) update.
        if self.new_profile_max_similarity >= self.match_min_similarity:
            raise ValueError(
                "New-profile maximum similarity must be below match minimum similarity."
            )
        if self.match_min_similarity > self.profile_update_min_similarity:
            raise ValueError(
                "Match minimum similarity must not exceed profile-update minimum similarity."
            )
        if self.match_min_margin > self.profile_update_min_margin:
            raise ValueError(
                "Match minimum margin must not exceed profile-update minimum margin."
            )
