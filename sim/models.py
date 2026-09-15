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
    agent_role: str = "patient"
    relationship: str | None = None
    character_name: str | None = None
    patient_name: str | None = None
    scenario_phase: int | None = None
    phases: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def active_phase(self) -> dict[str, Any]:
        if self.scenario_phase is None:
            return {}
        return self.phases.get(str(self.scenario_phase), {})

    @property
    def emotional_state(self) -> dict[str, float]:
        state = self.active_phase.get("emotional_state", {})
        return state if isinstance(state, dict) else {}

    @property
    def phase_behavior_guidelines(self) -> list[str]:
        guidelines = self.active_phase.get("behavior_guidelines", [])
        return guidelines if isinstance(guidelines, list) else []


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
