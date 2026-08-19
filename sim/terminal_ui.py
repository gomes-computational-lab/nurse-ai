from __future__ import annotations

from typing import Any

from sim.scenarios import list_scenarios


def choose_scenario():
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


def print_scenarios() -> None:
    for scenario in list_scenarios():
        print(f"{scenario.id}: {scenario.title}")


def print_feedback(feedback: dict[str, Any]) -> None:
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
