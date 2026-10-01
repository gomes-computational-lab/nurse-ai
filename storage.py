from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from sim.models import Message, Scenario


TRANSCRIPT_DIR = Path(__file__).resolve().parent.parent / "transcripts"


def save_result(
    scenario: Scenario,
    transcript: list[Message],
    feedback: dict[str, Any] | None = None,
) -> Path:
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = TRANSCRIPT_DIR / f"{timestamp}_{scenario.id}.json"

    payload = {
        "scenario": {
            "id": scenario.id,
            "title": scenario.title,
            "setting": scenario.setting,
        },
        "transcript": [message.__dict__ for message in transcript],
        "feedback": feedback,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path

