from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from sim.evaluator import evaluate_transcript
from sim.ollama_client import OllamaClient, OllamaError
from sim.scenarios import list_scenarios, load_scenario
from sim.session import SimulationSession
from sim.storage import save_result

# Yegeon: Use a smaller local Ollama model as the default for testing.
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:0.8b") 


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminal nursing simulation using local Ollama models.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model to use. Default: {DEFAULT_MODEL}")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host URL.")
    parser.add_argument("--scenario", default=None, help="Scenario ID to run.")
    parser.add_argument("--list", action="store_true", help="List available scenarios and exit.")
    args = parser.parse_args()

    if args.list:
        _print_scenarios()
        return

    try:
        scenario = load_scenario(args.scenario) if args.scenario else _choose_scenario()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)

    client = OllamaClient(model=args.model, host=args.host)
    session = SimulationSession(scenario, client)

    print(f"\nScenario: {scenario.title}")
    print(f"Setting: {scenario.setting}")
    print("Type /help for commands.\n")

    try:
        patient_text = session.opening()
        print(f"{scenario.role}: {patient_text}\n")

        while True:
            student_text = input("Student: ").strip()
            if not student_text:
                continue
            if student_text == "/help":
                _print_help()
                continue
            if student_text == "/quit":
                path = save_result(scenario, session.transcript)
                print(f"\nSaved transcript without feedback: {path}")
                return
            if student_text == "/end":
                break

            patient_text = session.respond(student_text)
            print(f"\n{scenario.role}: {patient_text}\n")

        print("\nEvaluating student performance...\n")
        feedback = evaluate_transcript(scenario, session.transcript, client)
        path = save_result(scenario, session.transcript, feedback)
        _print_feedback(feedback)
        print(f"\nSaved transcript and feedback: {path}")
    except KeyboardInterrupt:
        path = save_result(scenario, session.transcript)
        print(f"\nInterrupted. Saved transcript without feedback: {path}")
    except OllamaError as exc:
        print(f"\nOllama error: {exc}", file=sys.stderr)
        raise SystemExit(1)


def _choose_scenario():
    scenarios = list_scenarios()
    if not scenarios:
        raise ValueError("No scenarios found in scenarios/.")

    print("Available scenarios:")
    for index, scenario in enumerate(scenarios, start=1):
        print(f"{index}. {scenario.id} - {scenario.title}")

    while True:
        choice = input("\nChoose scenario number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(scenarios):
            return scenarios[int(choice) - 1]
        print("Please enter a valid scenario number.")


def _print_scenarios() -> None:
    for scenario in list_scenarios():
        print(f"{scenario.id}: {scenario.title}")


def _print_help() -> None:
    print("\nCommands:")
    print("  /end   End simulation and generate feedback")
    print("  /quit  Exit and save transcript without feedback")
    print("  /help  Show this help\n")


def _print_feedback(feedback: dict[str, Any]) -> None:
    print("Feedback")
    print("--------")
    print(f"Overall score: {feedback.get('overall_score')}")
    print(f"Summary: {feedback.get('summary')}\n")

    criteria = feedback.get("criteria", {})

    if isinstance(criteria, dict) and criteria:
        print("Criteria:")
        for name, result in criteria.items():
            if isinstance(result, dict):
                print(f"- {name}: {result.get('score')}/5")
                print(f"  Evidence: {result.get('evidence')}")
                print(f"  Coaching: {result.get('coaching')}")

    # Yegeon: Support the new evaluator output format,
    # where criteria are returned as a list of rubric results.
    elif isinstance(criteria, list) and criteria:
        print("Criteria:")
        for result in criteria:
            if isinstance(result, dict):
                print(f"- {result.get('name')}: {result.get('score')}/5")
                print(f"  Evidence: {result.get('evidence')}")
                print(f"  Coaching: {result.get('coaching')}")

    _print_list("Strengths", feedback.get("strengths"))
    _print_list("Improvements", feedback.get("improvements"))
    _print_list("Safety concerns", feedback.get("safety_concerns"))


def _print_list(title: str, values: Any) -> None:
    if not values:
        return

    print(f"\n{title}:")
    if isinstance(values, list):
        for value in values:
            print(f"- {value}")
    else:
        print(values)

if __name__ == "__main__": 
    main()