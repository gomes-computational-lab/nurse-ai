from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from sim.models import LearnerConfiguration, Participant, Scenario, SimulationTimeline


SCENARIO_DIR = Path(__file__).resolve().parent.parent / "scenarios"


def list_scenarios() -> list[Scenario]:
    return [load_scenario(path.stem) for path in sorted(SCENARIO_DIR.glob("*.json"))]


def load_scenario(scenario_id: str, *, phase: int | None = None) -> Scenario:
    path = SCENARIO_DIR / f"{scenario_id}.json"
    if not path.exists():
        available = ", ".join(s.id for s in list_scenarios()) or "none"
        raise FileNotFoundError(f"Unknown scenario '{scenario_id}'. Available scenarios: {available}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    scenario = Scenario(**_typed_scenario_data(data))
    _validate_scenario(scenario)
    return select_scenario_phase(scenario, phase)


def select_scenario_phase(scenario: Scenario, phase: int | None) -> Scenario:
    if phase is None or phase == scenario.scenario_phase:
        return scenario
    if str(phase) not in scenario.phases:
        available = ", ".join(sorted(scenario.phases)) or "none"
        raise ValueError(
            f"Scenario '{scenario.id}' does not define phase {phase}. Available phases: {available}"
        )
    return replace(scenario, scenario_phase=phase)


def _typed_scenario_data(data: dict) -> dict:
    typed = dict(data)
    typed["participants"] = tuple(
        Participant(**participant) for participant in typed.get("participants", [])
    )

    learner_data = typed.get("learner_configuration")
    if learner_data is None:
        typed["learner_configuration"] = LearnerConfiguration()
    else:
        learner_data = dict(learner_data)
        learner_data["supported_roles"] = tuple(learner_data.get("supported_roles", ()))
        typed["learner_configuration"] = LearnerConfiguration(**learner_data)

    timeline_data = typed.get("simulation_timeline")
    typed["simulation_timeline"] = (
        SimulationTimeline(**timeline_data) if timeline_data is not None else None
    )
    return typed


def _validate_scenario(scenario: Scenario) -> None:
    if scenario.agent_role not in {"patient", "family_member"}:
        raise ValueError(
            f"Scenario '{scenario.id}' has unsupported agent role '{scenario.agent_role}'."
        )
    if scenario.scenario_phase is not None and str(scenario.scenario_phase) not in scenario.phases:
        raise ValueError(
            f"Scenario '{scenario.id}' does not define its selected phase {scenario.scenario_phase}."
        )

    participant_ids = [participant.id for participant in scenario.participants]
    if len(participant_ids) != len(set(participant_ids)):
        raise ValueError(f"Scenario '{scenario.id}' participant ids must be unique.")

    for phase_name, phase in scenario.phases.items():
        if not isinstance(phase, dict):
            raise ValueError(f"Scenario '{scenario.id}' phase {phase_name} must be an object.")
        emotional_state = phase.get("emotional_state", {})
        if not isinstance(emotional_state, dict) or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= value <= 1
            for value in emotional_state.values()
        ):
            raise ValueError(
                f"Scenario '{scenario.id}' phase {phase_name} emotional values must be numbers from 0 to 1."
            )
