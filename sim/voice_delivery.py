from __future__ import annotations

import json
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
    stripped = response.strip()
    if not stripped.startswith(DELIVERY_PREFIX):
        return stripped, DEFAULT_DELIVERY

    end = stripped.find(DELIVERY_SUFFIX, len(DELIVERY_PREFIX))
    if end < 0:
        lines = stripped.splitlines()
        return (
            "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
        ), DEFAULT_DELIVERY

    raw_metadata = stripped[len(DELIVERY_PREFIX) : end].strip()
    spoken_text = stripped[end + len(DELIVERY_SUFFIX) :].strip()
    try:
        payload = json.loads(raw_metadata)
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
        if not spoken_text:
            raise ValueError("patient response is empty")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return spoken_text, DEFAULT_DELIVERY

    return spoken_text, DeliveryStyle(emotion=emotion, intensity=intensity, pace=pace)


class DeliveryStreamParser:
    """Strip a fragmented metadata header before forwarding visible response text."""

    def __init__(self, on_text: Callable[[str], None] | None):
        self._on_text = on_text
        self._buffer = ""
        self._header_resolved = False
        self._discarding_oversized_header = False
        self._visible_chunks: list[str] = []
        self.received_input = False
        self.style = DEFAULT_DELIVERY

    def add_chunk(self, chunk: str) -> None:
        if not chunk:
            return
        self.received_input = True
        if self._header_resolved:
            self._emit(chunk)
            return

        self._buffer += chunk
        if self._discarding_oversized_header:
            self._discard_oversized_header()
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
            _, self.style = parse_delivery_response(f"{header}\nplaceholder")
        self._buffer = ""
        self._header_resolved = True
        if remainder:
            self._emit(remainder.lstrip("\r\n"))

    def finish(self) -> tuple[str, DeliveryStyle]:
        if not self._header_resolved and self._buffer:
            text, self.style = parse_delivery_response(self._buffer)
            self._buffer = ""
            self._header_resolved = True
            self._emit(text)
        return "".join(self._visible_chunks).strip(), self.style

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
