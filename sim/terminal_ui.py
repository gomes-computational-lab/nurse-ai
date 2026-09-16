from __future__ import annotations

from typing import Any

from sim.models import Scenario
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


def choose_learner_count(scenario: Scenario) -> int:
    configuration = scenario.learner_configuration
    if configuration.min_nurses == configuration.max_nurses:
        return configuration.min_nurses

    choices = list(range(configuration.min_nurses, configuration.max_nurses + 1))
    labels = {1: "One", 2: "Two"}
    print("Number of student nurses:")
    for index, nurse_count in enumerate(choices, start=1):
        print(f"{index}. {labels.get(nurse_count, str(nurse_count))}")

    while True:
        choice = input("\nChoose number of student nurses: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(choices):
            return choices[int(choice) - 1]
        print("Please enter a valid option number.")


def print_scenario_header(
    scenario: Scenario,
    selected_learner_roles: tuple[str, ...] | None = None,
) -> None:
    print(f"\nScenario: {scenario.title}")
    print(f"Setting: {scenario.setting}")
    print(f"AI Role: {scenario.agent_role.replace('_', ' ').title()}")
    if scenario.character_name:
        character = scenario.character_name
        if scenario.relationship and scenario.patient_name:
            character += f" — {scenario.patient_name}'s {scenario.relationship}"
        print(f"Character: {character}")
    if scenario.scenario_phase is not None:
        print(f"Scenario Phase: {scenario.scenario_phase}")
    if selected_learner_roles:
        roles = ", ".join(role.replace("_", " ").title() for role in selected_learner_roles)
        print(f"Student Nurse Roles: {roles}")


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
