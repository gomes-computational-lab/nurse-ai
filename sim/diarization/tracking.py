"""Cross-window acoustic voice tracking with no participant-role semantics.

Pyannote labels are local to one inference window and can swap between calls.
For example, the same person may be ``SPEAKER_00`` in one model result and
``SPEAKER_01`` in the next.  ``VoiceTracker`` compares normalized speaker
embeddings with session profiles to provide more stable acoustic IDs.

This is session-local clustering, not biometric identity recognition.  It does
not know whether a voice belongs to a nurse, patient, learner, or anyone else.
"""

from __future__ import annotations

import numpy as np

from .config import DiarizationConfig
from .providers import DiarizationFrame


UNKNOWN_SPEAKER_ID = "SPEAKER_UNKNOWN"


class VoiceTracker:
    """Match window-local clusters to session-stable acoustic speaker IDs."""

    def __init__(self, config: DiarizationConfig | None = None):
        self.config = config or DiarizationConfig()
        self._profiles: list[np.ndarray] = []

    @property
    def profiles(self) -> tuple[np.ndarray, ...]:
        """Return profile copies for diagnostics without exposing mutable state."""
        return tuple(profile.copy() for profile in self._profiles)

    @property
    def profile_count(self) -> int:
        return len(self._profiles)

    def clone(self) -> VoiceTracker:
        """Copy tracker state so a processing update can commit atomically.

        The service runs matching on this copy.  If a later provider or result
        validation step fails, the original profiles remain untouched.
        """
        tracker = VoiceTracker(self.config)
        tracker._profiles = [profile.copy() for profile in self._profiles]
        return tracker

    def assign(self, frame: DiarizationFrame) -> dict[str, str]:
        """Map local acoustic labels to session IDs, leaving ambiguity unknown."""

        labels = frame.speaker_labels
        # Conservative default: an unavailable, malformed, or ambiguous
        # embedding never receives a confident session identity.
        mapping = {label: UNKNOWN_SPEAKER_ID for label in labels}
        if frame.speaker_embeddings is None:
            return mapping
        try:
            embeddings = np.asarray(frame.speaker_embeddings, dtype=np.float64)
        except (TypeError, ValueError):
            return mapping
        if embeddings.ndim != 2 or len(embeddings) != len(labels):
            return mapping

        # Longer speech usually produces a more useful embedding.  Duration is
        # also used to prevent brief noises from seeding a permanent profile.
        durations = {
            label: sum(
                turn.end - turn.start
                for turn in frame.speaker_turns
                if turn.speaker_id == label
            )
            for label in labels
        }
        # Process the most strongly represented local voice first.  This makes
        # profile creation deterministic when multiple new voices arrive in a
        # single window and profile capacity is limited.
        order = sorted(range(len(labels)), key=lambda index: -durations[labels[index]])
        for index in order:
            label = labels[index]
            vector = embeddings[index]
            norm = np.linalg.norm(vector)
            if not np.isfinite(vector).all() or norm < 1e-8:
                continue
            vector = vector / norm
            # Both sides are unit vectors, so a dot product is cosine
            # similarity: 1 is identical direction and lower values are less
            # acoustically similar.
            scores = np.asarray(
                [np.dot(vector, profile) for profile in self._profiles],
                dtype=np.float64,
            )

            # Seed a new profile only when the voice is sufficiently long,
            # profile capacity remains, and it is unlike every known profile.
            if (
                len(self._profiles) < self.config.max_voice_profiles
                and durations[label] >= self.config.profile_seed_min_speech_seconds
                and (
                    not len(scores)
                    or float(scores.max()) < self.config.new_profile_max_similarity
                )
            ):
                match = len(self._profiles)
                self._profiles.append(vector.copy())
            elif len(scores):
                match = int(scores.argmax())
                # The best score alone is insufficient when two profiles are
                # nearly tied.  The margin measures that ambiguity.
                if len(scores) >= 2:
                    strongest = np.partition(scores, -2)[-2:]
                    margin = float(strongest.max() - strongest.min())
                else:
                    margin = 1.0
                # A weak or ambiguous match remains SPEAKER_UNKNOWN.
                if (
                    scores[match] < self.config.match_min_similarity
                    or margin < self.config.match_min_margin
                ):
                    continue
                # Profile updates use stricter gates than ordinary matches to
                # reduce long-term drift.  The small configured blend weight
                # makes adaptation gradual.
                if (
                    scores[match] >= self.config.profile_update_min_similarity
                    and margin >= self.config.profile_update_min_margin
                    and durations[label] >= self.config.profile_seed_min_speech_seconds
                ):
                    update_weight = self.config.profile_update_weight
                    profile = (
                        (1.0 - update_weight) * self._profiles[match]
                        + update_weight * vector
                    )
                    self._profiles[match] = profile / np.linalg.norm(profile)
            else:
                continue
            # These IDs identify acoustic profiles only; their numeric order
            # is simply creation order within this service instance.
            mapping[label] = f"SPEAKER_{match:02d}"
        return mapping
