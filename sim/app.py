from __future__ import annotations

import argparse
import sys
import time

from sim.evaluator import evaluate_transcript
from sim.ollama_client import (
    OllamaClient,
    OllamaError,
    configured_base_url,
    configured_model,
)
from sim.scenarios import load_scenario, select_scenario_phase
from sim.session import SimulationSession
from sim.storage import save_result
from sim.terminal_ui import (
    choose_learner_count,
    choose_scenario,
    print_feedback,
    print_scenario_header,
    print_scenarios,
)


def main() -> None:
    default_model = configured_model()
    default_host = configured_base_url()
    parser = argparse.ArgumentParser(description="Terminal nursing simulation using local Ollama models.")
    parser.add_argument("--model", default=default_model, help=f"Ollama model to use. Default: {default_model}")
    parser.add_argument("--host", default=default_host, help=f"Ollama base URL. Default: {default_host}")
    parser.add_argument("--scenario", default=None, help="Scenario ID to run.")
    parser.add_argument("--phase", type=int, default=None, help="Scenario phase to run, when supported.")
    parser.add_argument("--list", action="store_true", help="List available scenarios and exit.")
    parser.add_argument(
        "--check-llm",
        action="store_true",
        help="Check Ollama health, model availability, and inference, then exit.",
    )
    args = parser.parse_args()

    if args.list:
        print_scenarios()
        return

    client = OllamaClient(model=args.model, host=args.host)
    if args.check_llm:
        _check_llm(client)
        return

    try:
        scenario = load_scenario(args.scenario) if args.scenario else choose_scenario()
        scenario = select_scenario_phase(scenario, args.phase)
        learner_count = choose_learner_count(scenario)
        client.health_check()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except OllamaError as exc:
        print(f"Ollama error: {exc}", file=sys.stderr)
        raise SystemExit(1)

    session = SimulationSession(scenario, client, learner_count=learner_count)

    print_scenario_header(scenario, session.selected_learner_roles)
    if len(session.selected_learner_roles) > 1:
        print(f"Active Learner: {_learner_label(session.active_learner_role)}")
    print("Type /help for commands.\n")

    try:
        _print_role_prefix(scenario.role)
        patient_text = session.opening(on_chunk=_print_stream_chunk)
        _finish_streamed_response()

        while True:
            student_text = input(_student_prompt(session)).strip()
            if not student_text:
                continue
            if _switch_learner(student_text, session):
                continue
            if student_text == "/help":
                _print_help(len(session.selected_learner_roles) > 1)
                continue
            if student_text == "/quit":
                path = save_result(
                    scenario,
                    session.transcript,
                    learner_roles=session.selected_learner_roles,
                )
                print(f"\nSaved transcript without feedback: {path}")
                return
            if student_text == "/end":
                break

            _print_role_prefix(scenario.role)
            patient_text = session.respond(student_text, on_chunk=_print_stream_chunk)
            _finish_streamed_response()

        print("\nEvaluating student performance...\n")
        feedback = evaluate_transcript(scenario, session.transcript, client)
        path = save_result(
            scenario,
            session.transcript,
            feedback,
            learner_roles=session.selected_learner_roles,
        )
        print_feedback(feedback)
        print(f"\nSaved transcript and feedback: {path}")
    except KeyboardInterrupt:
        path = save_result(
            scenario,
            session.transcript,
            learner_roles=session.selected_learner_roles,
        )
        print(f"\nInterrupted. Saved transcript without feedback: {path}")
    except OllamaError as exc:
        print(f"\nOllama error: {exc}", file=sys.stderr)
        raise SystemExit(1)


def _check_llm(client: OllamaClient) -> None:
    started = time.perf_counter()
    try:
        health_started = time.perf_counter()
        health = client.health_check()
        health_seconds = time.perf_counter() - health_started

        inference_started = time.perf_counter()
        response = client.warmup()
        inference_seconds = time.perf_counter() - inference_started
    except OllamaError as exc:
        print(f"Ollama check failed: {exc}", file=sys.stderr)
        raise SystemExit(1)

    print("Ollama check passed")
    print(f"Base URL: {health['base_url']}")
    print(f"Ollama version: {health['version']}")
    print(f"Model: {health['model']}")
    print(f"Health/model validation: {health_seconds:.3f} seconds")
    print(f"Minimal inference: {inference_seconds:.3f} seconds")
    print(f"Total: {time.perf_counter() - started:.3f} seconds")
    print(f"Response: {response or '(empty response)'}")


def _print_help(show_learner_switches: bool = False) -> None:
    print("\nCommands:")
    print("  /end   End simulation and generate feedback")
    print("  /quit  Exit and save transcript without feedback")
    print("  /help  Show this help")
    if show_learner_switches:
        print("  /primary    Switch text entry to the primary nurse")
        print("  /secondary  Switch text entry to the secondary nurse")
    print()


def _switch_learner(command: str, session: SimulationSession) -> bool:
    role_by_command = {
        "/primary": "nurse_primary",
        "/secondary": "nurse_secondary",
    }
    role = role_by_command.get(command)
    if role is None:
        return False
    if role not in session.selected_learner_roles:
        print(f"{_learner_label(role)} is not active in this simulation run.\n")
        return True
    session.select_learner_role(role)
    print(f"Active learner: {_learner_label(role)}\n")
    return True


def _student_prompt(session: SimulationSession) -> str:
    if len(session.selected_learner_roles) == 1:
        return "Student: "
    return f"{_learner_label(session.active_learner_role)}: "


def _learner_label(role: str) -> str:
    return role.replace("_", " ").title()


def _print_role_prefix(role: str) -> None:
    print(f"{role}: ", end="", flush=True)


def _print_stream_chunk(chunk: str) -> None:
    print(chunk, end="", flush=True)


def _finish_streamed_response() -> None:
    print("\n")
