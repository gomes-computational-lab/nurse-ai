"""Stateful, presentation-free orchestration for live diarization windows."""

from __future__ import annotations

import math
import time
from dataclasses import replace
from numbers import Real

import numpy as np

from .alignment import (
    align_words_to_speakers,
    to_session_speaker_turns,
    to_session_words,
)
from .config import DiarizationConfig
from .models import (
    AudioGap,
    ProcessingDiagnostics,
    ProcessingResult,
    RecognizedWord,
    Utterance,
    UtteranceStatus,
)
from .providers import DiarizationProvider, TranscriptionProvider
from .tracking import UNKNOWN_SPEAKER_ID, VoiceTracker


class DiarizationServiceError(RuntimeError):
    """The service lifecycle or input window is invalid."""


class DiarizationServiceClosedError(DiarizationServiceError):
    """Processing was requested after the service was flushed."""


class DiarizationService:
    """Coordinate providers, tracking, alignment, and transcript finalization."""

    def __init__(
        self,
        diarization_provider: DiarizationProvider,
        transcription_provider: TranscriptionProvider,
        config: DiarizationConfig | None = None,
    ):
        self.config = config or DiarizationConfig()
        self._diarization_provider = diarization_provider
        self._transcription_provider = transcription_provider
        self._voice_tracker = VoiceTracker(self.config)
        self._finalized_until = 0.0
        self._provisional_utterances: tuple[Utterance, ...] = ()
        self._last_window_end = 0.0
        self._last_diagnostics: ProcessingDiagnostics | None = None
        self._is_flushed = False

    @property
    def finalized_until(self) -> float:
        return self._finalized_until

    @property
    def provisional_utterances(self) -> tuple[Utterance, ...]:
        return self._provisional_utterances

    @property
    def profile_count(self) -> int:
        return self._voice_tracker.profile_count

    @property
    def is_flushed(self) -> bool:
        return self._is_flushed

    def process(self, samples: np.ndarray, start_seconds: float) -> ProcessingResult:
        """Analyze one rolling window and atomically commit its transcript state."""
        if self._is_flushed:
            raise DiarizationServiceClosedError("Cannot process audio after flush().")
        audio = self._validated_audio(samples)
        start = self._validated_start(start_seconds)
        end = start + len(audio) / self.config.sample_rate
        if end + 1e-9 < self._last_window_end:
            raise DiarizationServiceError(
                "Audio windows must not move backward on the session timeline."
            )

        began = time.perf_counter()
        # Providers do not mutate service state. If either raises, every service
        # frontier/profile/pending value remains exactly as it was before this call.
        frame = self._diarization_provider.diarize(
            audio,
            sample_rate=self.config.sample_rate,
            min_speakers=self.config.min_speakers,
            max_speakers=self.config.max_speakers,
        )
        local_words = self._transcription_provider.transcribe(
            audio,
            sample_rate=self.config.sample_rate,
        )

        tracker = self._voice_tracker.clone()
        mapping = tracker.assign(frame)
        speaker_turns = to_session_speaker_turns(
            frame.exclusive_speaker_turns,
            mapping,
            window_start=start,
            window_end=end,
        )
        recognized_words = tuple(sorted(
            to_session_words(
                local_words,
                window_start=start,
                window_end=end,
            ),
            key=lambda word: (word.start, word.end),
        ))

        previous_frontier = self._finalized_until
        cutoff = max(previous_frontier, end - self.config.finalization_delay_seconds)
        eligible_words = tuple(
            word for word in recognized_words if _midpoint(word) >= previous_frontier
        )
        final_words = tuple(word for word in eligible_words if _midpoint(word) < cutoff)
        provisional_words = tuple(word for word in eligible_words if _midpoint(word) >= cutoff)
        finalized_utterances = align_words_to_speakers(
            final_words,
            speaker_turns,
            status=UtteranceStatus.FINALIZED,
        )
        provisional_utterances = align_words_to_speakers(
            provisional_words,
            speaker_turns,
            status=UtteranceStatus.PROVISIONAL,
        )

        gaps = ()
        if start > previous_frontier + 0.001:
            gaps = (
                AudioGap(
                    previous_frontier,
                    start,
                    "audio unavailable before the current rolling window",
                ),
            )
        diagnostics = self._diagnostics(
            audio,
            start=start,
            end=end,
            began=began,
            local_speaker_count=len({turn.speaker_id for turn in frame.exclusive_speaker_turns}),
            tracked_speaker_count=len(
                {turn.speaker_id for turn in speaker_turns}
                - {UNKNOWN_SPEAKER_ID}
            ),
            profile_count=tracker.profile_count,
        )
        result = ProcessingResult(
            speaker_turns,
            recognized_words,
            finalized_utterances,
            provisional_utterances,
            gaps,
            diagnostics,
        )

        # Commit only after every provider, conversion, alignment, and result
        # validation step has completed successfully.
        self._voice_tracker = tracker
        self._finalized_until = cutoff
        self._provisional_utterances = provisional_utterances
        self._last_window_end = end
        self._last_diagnostics = diagnostics
        return result

    def flush(self) -> ProcessingResult:
        """Finalize the current provisional tail exactly once and close the service."""
        began = time.perf_counter()
        finalized = tuple(
            replace(utterance, status=UtteranceStatus.FINALIZED)
            for utterance in self._provisional_utterances
        ) if not self._is_flushed else ()
        diagnostics = self._flush_diagnostics(began)
        result = ProcessingResult((), (), finalized, (), (), diagnostics)
        if not self._is_flushed:
            self._provisional_utterances = ()
            self._finalized_until = max(self._finalized_until, self._last_window_end)
            self._is_flushed = True
        return result

    @staticmethod
    def _validated_audio(samples: np.ndarray) -> np.ndarray:
        try:
            audio = np.asarray(samples, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("Audio samples must be a one-dimensional numeric array.") from exc
        if audio.ndim != 1 or not audio.size:
            raise ValueError("Audio samples must be a non-empty one-dimensional array.")
        if not np.isfinite(audio).all():
            raise ValueError("Audio samples must contain only finite values.")
        return audio

    @staticmethod
    def _validated_start(start_seconds: float) -> float:
        if (
            isinstance(start_seconds, bool)
            or not isinstance(start_seconds, Real)
            or not math.isfinite(start_seconds)
            or start_seconds < 0
        ):
            raise ValueError("Window start must be a finite, non-negative number.")
        return float(start_seconds)

    @staticmethod
    def _diagnostics(
        audio: np.ndarray,
        *,
        start: float,
        end: float,
        began: float,
        local_speaker_count: int,
        tracked_speaker_count: int,
        profile_count: int,
    ) -> ProcessingDiagnostics:
        rms = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.max(np.abs(audio)))
        return ProcessingDiagnostics(
            window_start=start,
            window_end=end,
            processing_seconds=time.perf_counter() - began,
            input_rms_dbfs=20 * math.log10(max(rms, 1e-12)),
            input_peak_dbfs=20 * math.log10(max(peak, 1e-12)),
            local_speaker_count=local_speaker_count,
            tracked_speaker_count=tracked_speaker_count,
            profile_count=profile_count,
        )

    def _flush_diagnostics(self, began: float) -> ProcessingDiagnostics:
        if self._last_diagnostics is None:
            return ProcessingDiagnostics(
                window_start=0,
                window_end=0,
                processing_seconds=time.perf_counter() - began,
                input_rms_dbfs=-240,
                input_peak_dbfs=-240,
                local_speaker_count=0,
                tracked_speaker_count=0,
                profile_count=self.profile_count,
            )
        return replace(
            self._last_diagnostics,
            processing_seconds=time.perf_counter() - began,
        )


def _midpoint(word: RecognizedWord) -> float:
    return (word.start + word.end) / 2
