from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any


@dataclass(frozen=True)
class Participant:
    id: str
    participant_type: str
    role: str | None = None
    display_name: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("Participant id must not be empty.")
        if not isinstance(self.participant_type, str) or not self.participant_type.strip():
            raise ValueError(f"Participant '{self.id}' must define a participant type.")
        if self.role is not None and (not isinstance(self.role, str) or not self.role.strip()):
            raise ValueError(f"Participant '{self.id}' role must be a non-empty string.")
        if not isinstance(self.attributes, dict):
            raise ValueError(f"Participant '{self.id}' attributes must be an object.")


@dataclass(frozen=True)
class LearnerConfiguration:
    min_nurses: int = 1
    max_nurses: int = 1
    supported_roles: tuple[str, ...] = ("nurse_primary",)

    def __post_init__(self) -> None:
        if isinstance(self.min_nurses, bool) or not isinstance(self.min_nurses, int):
            raise ValueError("Learner minimum must be an integer.")
        if isinstance(self.max_nurses, bool) or not isinstance(self.max_nurses, int):
            raise ValueError("Learner maximum must be an integer.")
        if self.min_nurses < 1 or self.max_nurses < self.min_nurses:
            raise ValueError("Learner nurse limits must satisfy 1 <= min_nurses <= max_nurses.")
        if len(self.supported_roles) < self.max_nurses:
            raise ValueError("Learner configuration must define a role for every supported nurse.")
        if len(set(self.supported_roles)) != len(self.supported_roles):
            raise ValueError("Learner roles must be unique.")
        if any(not isinstance(role, str) or not role.strip() for role in self.supported_roles):
            raise ValueError("Learner roles must be non-empty strings.")

    def roles_for_count(self, nurse_count: int) -> tuple[str, ...]:
        if (
            isinstance(nurse_count, bool)
            or not isinstance(nurse_count, int)
            or nurse_count < self.min_nurses
            or nurse_count > self.max_nurses
        ):
            raise ValueError(
                f"Nurse count must be from {self.min_nurses} to {self.max_nurses}."
            )
        return self.supported_roles[:nurse_count]


@dataclass(frozen=True)
class SimulationTimeline:
    simulation_date: str | None = None
    clinical_day: int | None = None
    time_of_day: str | None = None

    def __post_init__(self) -> None:
        if self.simulation_date is not None:
            try:
                date.fromisoformat(self.simulation_date)
            except (TypeError, ValueError) as exc:
                raise ValueError("Simulation date must use ISO format YYYY-MM-DD.") from exc
        if self.clinical_day is not None and (
            isinstance(self.clinical_day, bool)
            or not isinstance(self.clinical_day, int)
            or self.clinical_day < 1
        ):
            raise ValueError("Clinical day must be a positive integer.")
        if self.time_of_day is not None and (
            not isinstance(self.time_of_day, str) or not self.time_of_day.strip()
        ):
            raise ValueError("Simulation time of day must be a non-empty string.")


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
    participants: tuple[Participant, ...] = ()
    learner_configuration: LearnerConfiguration = field(default_factory=LearnerConfiguration)
    simulation_timeline: SimulationTimeline | None = None

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
    speaker_role: str | None = None


@dataclass
class SimulationResult:
    scenario_id: str
    transcript: list[Message]
    feedback: dict[str, Any] | None = None
