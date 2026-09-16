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
    latency: dict[str, Any] | None = None,
    learner_roles: tuple[str, ...] | list[str] | None = None,
) -> Path:
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    saved_at = datetime.now()
    timestamp = saved_at.strftime("%Y%m%d_%H%M%S")
    path = TRANSCRIPT_DIR / f"{timestamp}_{scenario.id}.json"
    selected_roles = tuple(learner_roles) if learner_roles is not None else (
        scenario.learner_configuration.roles_for_count(
            scenario.learner_configuration.min_nurses
        )
    )

    payload = {
        "metadata": {
            "agent_role": scenario.agent_role,
            "scenario_id": scenario.id,
            "scenario_phase": scenario.scenario_phase,
            "learner_configuration": {
                "nurse_count": len(selected_roles),
                "roles": list(selected_roles),
            },
            "simulation_timeline": _timeline_payload(scenario),
            "execution_saved_at": saved_at.isoformat(timespec="seconds"),
        },
        "scenario": {
            "id": scenario.id,
            "title": scenario.title,
            "setting": scenario.setting,
            "participants": [
                {
                    "id": participant.id,
                    "participant_type": participant.participant_type,
                    "role": participant.role,
                    "display_name": participant.display_name,
                    "attributes": participant.attributes,
                }
                for participant in scenario.participants
            ],
            "learner_configuration": {
                "min_nurses": scenario.learner_configuration.min_nurses,
                "max_nurses": scenario.learner_configuration.max_nurses,
                "supported_roles": list(scenario.learner_configuration.supported_roles),
            },
        },
        "transcript": [_message_payload(message) for message in transcript],
        "feedback": feedback,
    }
    if latency is not None:
        payload["latency"] = latency
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _message_payload(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": message.role,
        "content": message.content,
        "created_at": message.created_at,
    }
    if message.speaker_role is not None:
        payload["speaker_role"] = message.speaker_role
    return payload


def _timeline_payload(scenario: Scenario) -> dict[str, Any] | None:
    timeline = scenario.simulation_timeline
    if timeline is None:
        return None
    return {
        "simulation_date": timeline.simulation_date,
        "clinical_day": timeline.clinical_day,
        "time_of_day": timeline.time_of_day,
    }
