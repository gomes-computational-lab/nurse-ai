from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Callable, Literal


Emotion = Literal[
    "neutral",
    "anxious",
    "fearful",
    "frustrated",
    "confused",
    "relieved",
    "sad",
    "in_pain",
    "tired",
]
Pace = Literal["slow", "normal", "fast"]

DELIVERY_PREFIX = "[[delivery]]"
DELIVERY_SUFFIX = "[[/delivery]]"
MAX_DELIVERY_LEADING_CHARS = 64
MAX_DELIVERY_HEADER_CHARS = 4096
VALID_EMOTIONS = {
    "neutral",
    "anxious",
    "fearful",
    "frustrated",
    "confused",
    "relieved",
    "sad",
    "in_pain",
    "tired",
}
VALID_PACES = {"slow", "normal", "fast"}
DELIVERY_KEYS = {"emotion", "intensity", "pace"}
_OUTER_CODE_FENCE_PATTERN = re.compile(
    r"^```(?:json|text)?\s*\n(?P<body>.*)\n```$",
    re.IGNORECASE | re.DOTALL,
)
_HEADER_CODE_FENCE_PATTERN = re.compile(
    rf"^```(?:json|text)?\s*(?:\r?\n)?"
    rf"(?P<header>{re.escape(DELIVERY_PREFIX)}.*?{re.escape(DELIVERY_SUFFIX)})"
    r"\s*```\s*(?P<body>.*)$",
    re.IGNORECASE | re.DOTALL,
)
_DELIVERY_TAG_PATTERN = re.compile(
    rf"{re.escape(DELIVERY_PREFIX)}|{re.escape(DELIVERY_SUFFIX)}",
    re.IGNORECASE,
)
_METADATA_JSON_KEY_PATTERN = re.compile(
    r'"[A-Za-z_][A-Za-z0-9_ -]*"\s*:',
    re.IGNORECASE,
)
_METADATA_LABEL_PATTERN = re.compile(
    r"^\s*(?:delivery|emotion|intensity|pace|metadata|pain\s*level)\s*:",
    re.IGNORECASE | re.MULTILINE,
)


class UnsafeVoiceResponseError(ValueError):
    """The model response could expose internal metadata as patient speech."""


@dataclass(frozen=True)
class DeliveryStyle:
    emotion: Emotion = "neutral"
    intensity: int = 1
    pace: Pace = "normal"

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


DEFAULT_DELIVERY = DeliveryStyle()


def parse_delivery_response(response: str) -> tuple[str, DeliveryStyle]:
    """Return clean patient wording and a validated, bounded delivery style."""
    stripped = response.strip().lstrip("\ufeff").lstrip()
    stripped = _unwrap_outer_code_fence(stripped)
    if not stripped.startswith(DELIVERY_PREFIX):
        return validate_spoken_text(stripped), DEFAULT_DELIVERY

    end = stripped.find(DELIVERY_SUFFIX, len(DELIVERY_PREFIX))
    if end < 0:
        raise UnsafeVoiceResponseError("delivery metadata was not closed")

    raw_metadata = stripped[len(DELIVERY_PREFIX) : end].strip()
    spoken_text = stripped[end + len(DELIVERY_SUFFIX) :].strip()
    if spoken_text.endswith(DELIVERY_SUFFIX):
        spoken_text = spoken_text[: -len(DELIVERY_SUFFIX)].rstrip()
    try:
        payload = json.loads(raw_metadata)
        if not isinstance(payload, dict) or set(payload) != DELIVERY_KEYS:
            raise ValueError("delivery metadata must contain only approved fields")
        emotion = payload["emotion"]
        intensity = payload["intensity"]
        pace = payload["pace"]
        if emotion not in VALID_EMOTIONS or pace not in VALID_PACES:
            raise ValueError("unsupported delivery value")
        if (
            isinstance(intensity, bool)
            or not isinstance(intensity, int)
            or not 1 <= intensity <= 3
        ):
            raise ValueError("intensity must be 1-3")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        # The delivery header is never spoken. If its boundaries are intact, keep
        # the separately validated patient wording and use a neutral voice instead
        # of failing the whole conversational turn.
        return validate_spoken_text(spoken_text), DEFAULT_DELIVERY

    return validate_spoken_text(spoken_text), DeliveryStyle(
        emotion=emotion,
        intensity=intensity,
        pace=pace,
    )


def validate_spoken_text(text: str) -> str:
    """Return safe patient wording or reject text containing structural metadata."""
    spoken_text = text.strip()
    if not spoken_text:
        raise UnsafeVoiceResponseError("patient response was empty")
    if _DELIVERY_TAG_PATTERN.search(spoken_text):
        raise UnsafeVoiceResponseError("delivery markers remained in patient speech")
    if "```" in spoken_text:
        raise UnsafeVoiceResponseError("a code fence remained in patient speech")
    if _METADATA_JSON_KEY_PATTERN.search(spoken_text):
        raise UnsafeVoiceResponseError("JSON metadata remained in patient speech")
    if _METADATA_LABEL_PATTERN.search(spoken_text):
        raise UnsafeVoiceResponseError("a metadata label remained in patient speech")
    return spoken_text


def _unwrap_outer_code_fence(text: str) -> str:
    outer_fence = _OUTER_CODE_FENCE_PATTERN.fullmatch(text)
    if outer_fence:
        return outer_fence.group("body").strip()
    header_fence = _HEADER_CODE_FENCE_PATTERN.fullmatch(text)
    if header_fence:
        return f"{header_fence.group('header')}\n{header_fence.group('body')}".strip()
    return text


class DeliveryStreamParser:
    """Strip a fragmented metadata header before forwarding visible response text."""

    def __init__(self, on_text: Callable[[str], None] | None):
        self._on_text = on_text
        self._buffer = ""
        self._raw_chunks: list[str] = []
        self._header_resolved = False
        self._discarding_oversized_header = False
        self._visible_chunks: list[str] = []
        self.received_input = False
        self.style = DEFAULT_DELIVERY

    def add_chunk(self, chunk: str) -> None:
        if not chunk:
            return
        self.received_input = True
        self._raw_chunks.append(chunk)
        if self._header_resolved:
            self._emit(chunk)
            return

        self._buffer += chunk
        if self._discarding_oversized_header:
            self._discard_oversized_header()
            return
        if self._could_be_markdown_wrapped():
            return

        header_start = self._header_start()
        if header_start is None:
            buffered = self._buffer
            self._buffer = ""
            self._header_resolved = True
            self._emit(buffered)
            return

        candidate = self._buffer[header_start:]
        if DELIVERY_PREFIX.startswith(candidate):
            return

        end = candidate.find(DELIVERY_SUFFIX, len(DELIVERY_PREFIX))
        if end < 0:
            if len(candidate) > MAX_DELIVERY_HEADER_CHARS:
                self._discarding_oversized_header = True
                self.style = DEFAULT_DELIVERY
                self._discard_oversized_header()
            return

        header_end = end + len(DELIVERY_SUFFIX)
        header = candidate[:header_end]
        remainder = candidate[header_end:]
        if len(header) <= MAX_DELIVERY_HEADER_CHARS:
            try:
                _, self.style = parse_delivery_response(f"{header}\nplaceholder")
            except UnsafeVoiceResponseError:
                self.style = DEFAULT_DELIVERY
        self._buffer = ""
        self._header_resolved = True
        if remainder:
            self._emit(remainder.lstrip("\r\n"))

    def finish(self) -> tuple[str, DeliveryStyle]:
        raw_response = "".join(self._raw_chunks)
        text, self.style = parse_delivery_response(raw_response)
        self._buffer = ""
        self._header_resolved = True
        return text, self.style

    def _could_be_markdown_wrapped(self) -> bool:
        leading = 0
        while leading < len(self._buffer):
            char = self._buffer[leading]
            if char != "\ufeff" and not char.isspace():
                break
            leading += 1
        candidate = self._buffer[leading:]
        return "```".startswith(candidate) or candidate.startswith("```")

    def _header_start(self) -> int | None:
        """Return the possible header offset, or None once plain text is certain."""
        leading = 0
        while leading < len(self._buffer):
            char = self._buffer[leading]
            if char != "\ufeff" and not char.isspace():
                break
            leading += 1
            if leading > MAX_DELIVERY_LEADING_CHARS:
                return None

        candidate = self._buffer[leading:]
        if not candidate or DELIVERY_PREFIX.startswith(candidate):
            return leading
        if candidate.startswith(DELIVERY_PREFIX):
            return leading
        return None

    def _discard_oversized_header(self) -> None:
        """Hide an overlong header while retaining only enough to find its suffix."""
        end = self._buffer.find(DELIVERY_SUFFIX)
        if end >= 0:
            remainder = self._buffer[end + len(DELIVERY_SUFFIX) :]
            self._buffer = ""
            self._discarding_oversized_header = False
            self._header_resolved = True
            if remainder:
                self._emit(remainder.lstrip("\r\n"))
            return

        overlap = len(DELIVERY_SUFFIX) - 1
        self._buffer = self._buffer[-overlap:]

    def _emit(self, text: str) -> None:
        if not text:
            return
        self._visible_chunks.append(text)
        if self._on_text is not None:
            self._on_text(text)
