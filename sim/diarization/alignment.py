"""Pure timestamp conversion and word-to-speaker alignment functions.

Providers describe time relative to the beginning of the current audio
window.  The rest of the application uses one monotonically increasing
session clock.  This module performs that conversion and then combines the
independent answers to "who spoke when?" and "what words were heard?".
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .models import RecognizedWord, SpeakerTurn, Utterance, UtteranceStatus
from .providers import LocalSpeakerTurn
from .tracking import UNKNOWN_SPEAKER_ID


def to_session_speaker_turns(
    turns: Iterable[LocalSpeakerTurn],
    speaker_mapping: Mapping[str, str],
    *,
    window_start: float,
    window_end: float,
) -> tuple[SpeakerTurn, ...]:
    """Shift window-local turns onto the session clock and clip the window."""

    result: list[SpeakerTurn] = []
    for turn in turns:
        # Defensive clipping prevents a provider's small rounding error from
        # creating timestamps outside the audio that was actually analyzed.
        start = max(window_start, window_start + turn.start)
        end = min(window_end, window_start + turn.end)
        if end <= start:
            continue
        result.append(
            SpeakerTurn(
                start,
                end,
                speaker_mapping.get(turn.speaker_id, UNKNOWN_SPEAKER_ID),
                turn.speaker_id,
            )
        )
    return tuple(result)


def to_session_words(
    words: Iterable[RecognizedWord],
    *,
    window_start: float,
    window_end: float,
) -> tuple[RecognizedWord, ...]:
    """Shift window-local recognized words onto the session clock."""

    result: list[RecognizedWord] = []
    for word in words:
        start = max(window_start, window_start + word.start)
        end = min(window_end, window_start + word.end)
        if end > start:
            result.append(RecognizedWord(start, end, word.text))
    return tuple(result)


def align_words_to_speakers(
    words: Iterable[RecognizedWord],
    speaker_turns: Iterable[SpeakerTurn],
    *,
    status: UtteranceStatus,
) -> tuple[Utterance, ...]:
    """Assign words to their most-overlapping acoustic turn and group speakers.

    Pyannote and Whisper run independently, so their boundaries rarely match
    exactly.  Duration of timestamp overlap is a simple deterministic rule
    that does not invent role semantics.  Words with no overlap are omitted
    from utterances but remain available in ``ProcessingResult.recognized_words``
    for inspection and debugging.
    """
    turns = tuple(speaker_turns)
    if not turns:
        return ()

    utterances: list[Utterance] = []
    for word in words:
        # ``max`` is safe because the empty-turn case returned above.
        turn = max(turns, key=lambda item: _overlap(word, item))
        if _overlap(word, turn) <= 0:
            continue
        # Consecutive words attributed to the same acoustic speaker become one
        # readable utterance.  A speaker change starts a new utterance.
        if utterances and utterances[-1].speaker_id == turn.speaker_id:
            previous = utterances[-1]
            utterances[-1] = Utterance(
                previous.start,
                max(previous.end, word.end),
                previous.speaker_id,
                previous.text + " " + word.text,
                status,
            )
        else:
            utterances.append(
                Utterance(word.start, word.end, turn.speaker_id, word.text, status)
            )
    return tuple(utterances)


def _overlap(word: RecognizedWord, turn: SpeakerTurn) -> float:
    return max(0.0, min(word.end, turn.end) - max(word.start, turn.start))
