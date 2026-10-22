from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sim.evaluator import evaluate_transcript
from sim.models import Message, Scenario
from sim.ollama_client import (
    OllamaClient,
    OllamaError,
    configured_base_url,
    configured_model,
)
from sim.scenarios import load_scenario
from sim.terminal_ui import print_feedback


TRANSCRIPT_ROLES = {"student", "patient", "family_member", "facilitator"}


def load_transcript_file(
    path: Path,
    *,
    scenario_override: str | None = None,
) -> tuple[Scenario, list[Message]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Transcript file '{path}' does not contain valid JSON.") from exc

    transcript_data, scenario_id, phase = _transcript_data(payload, scenario_override)
    scenario = load_scenario(scenario_id, phase=phase)
    transcript = [_message_from_data(item, index) for index, item in enumerate(transcript_data)]
    return scenario, transcript


def _transcript_data(
    payload: Any,
    scenario_override: str | None,
) -> tuple[list[Any], str, int | None]:
    metadata: dict[str, Any] = {}
    embedded_scenario: dict[str, Any] = {}

    if isinstance(payload, list):
        transcript_data = payload
    elif isinstance(payload, dict):
        transcript_data = payload.get("transcript")
        raw_metadata = payload.get("metadata", {})
        raw_scenario = payload.get("scenario", {})
        if not isinstance(raw_metadata, dict):
            raise ValueError("Transcript metadata must be a JSON object.")
        if not isinstance(raw_scenario, dict):
            raise ValueError("Transcript scenario must be a JSON object.")
        metadata = raw_metadata
        embedded_scenario = raw_scenario
    else:
        raise ValueError("Transcript JSON must be a saved result object or a message array.")

    if not isinstance(transcript_data, list) or not transcript_data:
        raise ValueError("Transcript JSON must contain a non-empty transcript message array.")

    scenario_id = scenario_override or metadata.get("scenario_id") or embedded_scenario.get("id")
    if not isinstance(scenario_id, str) or not scenario_id.strip():
        raise ValueError(
            "The transcript does not identify a scenario. Provide one with --scenario."
        )

    phase = metadata.get("scenario_phase")
    if phase is not None and (isinstance(phase, bool) or not isinstance(phase, int)):
        raise ValueError("Transcript scenario phase must be an integer or null.")
    return transcript_data, scenario_id.strip(), phase


def _message_from_data(data: Any, index: int) -> Message:
    if not isinstance(data, dict):
        raise ValueError(f"Transcript message {index + 1} must be a JSON object.")

    role = data.get("role")
    content = data.get("content")
    speaker_role = data.get("speaker_role")
    created_at = data.get("created_at")

    if role not in TRANSCRIPT_ROLES:
        roles = ", ".join(sorted(TRANSCRIPT_ROLES))
        raise ValueError(
            f"Transcript message {index + 1} has an unsupported role. Expected: {roles}."
        )
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"Transcript message {index + 1} content must be a non-empty string.")
    if speaker_role is not None and (
        not isinstance(speaker_role, str) or not speaker_role.strip()
    ):
        raise ValueError(f"Transcript message {index + 1} speaker_role must be a non-empty string.")
    if created_at is not None and not isinstance(created_at, str):
        raise ValueError(f"Transcript message {index + 1} created_at must be a string.")

    values: dict[str, Any] = {
        "role": role,
        "content": content,
        "speaker_role": speaker_role,
    }
    if created_at is not None:
        values["created_at"] = created_at
    return Message(**values)


def main(arguments: list[str] | None = None) -> None:
    default_model = configured_model()
    default_host = configured_base_url()
    parser = argparse.ArgumentParser(description="Evaluate a saved nursing simulation transcript.")
    parser.add_argument("transcript", type=Path, help="Saved transcript JSON file to evaluate.")
    parser.add_argument(
        "--scenario",
        default=None,
        help="Scenario ID, required only when the JSON file does not contain one.",
    )
    parser.add_argument(
        "--model",
        default=default_model,
        help=f"Ollama model. Default: {default_model}",
    )
    parser.add_argument(
        "--host",
        default=default_host,
        help=f"Ollama base URL. Default: {default_host}",
    )
    args = parser.parse_args(arguments)

    try:
        scenario, transcript = load_transcript_file(
            args.transcript,
            scenario_override=args.scenario,
        )
        client = OllamaClient(model=args.model, host=args.host)
        client.health_check()
        feedback = evaluate_transcript(scenario, transcript, client)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except OllamaError as exc:
        print(f"Ollama error: {exc}", file=sys.stderr)
        raise SystemExit(1)

    print_feedback(feedback)


if __name__ == "__main__":
    main()
