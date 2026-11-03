"""Cross-window acoustic voice tracking with no participant-role semantics."""

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

    def assign(self, frame: DiarizationFrame) -> dict[str, str]:
        """Map local acoustic labels to session IDs, leaving ambiguity unknown."""
        labels = frame.speaker_labels
        mapping = {label: UNKNOWN_SPEAKER_ID for label in labels}
        if frame.speaker_embeddings is None:
            return mapping
        try:
            embeddings = np.asarray(frame.speaker_embeddings, dtype=np.float64)
        except (TypeError, ValueError):
            return mapping
        if embeddings.ndim != 2 or len(embeddings) != len(labels):
            return mapping

        durations = {
            label: sum(
                turn.end - turn.start
                for turn in frame.speaker_turns
                if turn.speaker_id == label
            )
            for label in labels
        }
        order = sorted(range(len(labels)), key=lambda index: -durations[labels[index]])
        for index in order:
            label = labels[index]
            vector = embeddings[index]
            norm = np.linalg.norm(vector)
            if not np.isfinite(vector).all() or norm < 1e-8:
                continue
            vector = vector / norm
            scores = np.asarray(
                [np.dot(vector, profile) for profile in self._profiles],
                dtype=np.float64,
            )

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
                if len(scores) >= 2:
                    strongest = np.partition(scores, -2)[-2:]
                    margin = float(strongest.max() - strongest.min())
                else:
                    margin = 1.0
                if (
                    scores[match] < self.config.match_min_similarity
                    or margin < self.config.match_min_margin
                ):
                    continue
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
            mapping[label] = f"SPEAKER_{match:02d}"
        return mapping
