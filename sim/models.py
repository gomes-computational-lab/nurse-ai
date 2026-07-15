from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    role: str
    setting: str
    patient_profile: dict[str, Any]
    clinical_context: dict[str, Any]
    behavior_guidelines: list[str]
    opening_prompt: str
    learning_objectives: list[str]
    evaluation_rubric: dict[str, str]


@dataclass
class Message:
    role: str
    content: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


@dataclass
class SimulationResult:
    scenario_id: str
    transcript: list[Message]
    feedback: dict[str, Any] | None = None

