from __future__ import annotations

import argparse
import os
import sys

from sim.evaluator import evaluate_transcript
from sim.ollama_client import OllamaClient, OllamaError
from sim.scenarios import load_scenario
from sim.session import SimulationSession
from sim.storage import save_result
from sim.terminal_ui import choose_scenario, print_feedback, print_scenarios


DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminal nursing simulation using local Ollama models.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model to use. Default: {DEFAULT_MODEL}")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host URL.")
    parser.add_argument("--scenario", default=None, help="Scenario ID to run.")
    parser.add_argument("--list", action="store_true", help="List available scenarios and exit.")
    args = parser.parse_args()

    if args.list:
        print_scenarios()
        return

    try:
        scenario = load_scenario(args.scenario) if args.scenario else choose_scenario()
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
        print_feedback(feedback)
        print(f"\nSaved transcript and feedback: {path}")
    except KeyboardInterrupt:
        path = save_result(scenario, session.transcript)
        print(f"\nInterrupted. Saved transcript without feedback: {path}")
    except OllamaError as exc:
        print(f"\nOllama error: {exc}", file=sys.stderr)
        raise SystemExit(1)


def _print_help() -> None:
    print("\nCommands:")
    print("  /end   End simulation and generate feedback")
    print("  /quit  Exit and save transcript without feedback")
    print("  /help  Show this help\n")
