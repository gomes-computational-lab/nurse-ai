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
        if DELIVERY_PREFIX.startswith(self._buffer):
            return
        if not self._buffer.startswith(DELIVERY_PREFIX):
            buffered = self._buffer
            self._buffer = ""
            self._header_resolved = True
            self._emit(buffered)
            return

        end = self._buffer.find(DELIVERY_SUFFIX, len(DELIVERY_PREFIX))
        if end < 0:
            return

        header = self._buffer[: end + len(DELIVERY_SUFFIX)]
        remainder = self._buffer[end + len(DELIVERY_SUFFIX) :]
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

    def _emit(self, text: str) -> None:
        if not text:
            return
        self._visible_chunks.append(text)
        if self._on_text is not None:
            self._on_text(text)
