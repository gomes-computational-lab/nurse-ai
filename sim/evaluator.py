from __future__ import annotations

import json
import re
from typing import Any

from sim.models import Message, Scenario
from sim.ollama_client import OllamaClient

# Yegeon: Define fixed K-HCAT and DARE2-informed criteria
# for the final nursing simulation evaluator.
CRITERIA = [
    {
        "name": "Empathy and rapport",
        "source": "K-HCAT",
        "description": (
            "Evaluate whether the student acknowledges the patient's pain, fear, worry, "
            "or anxiety in a supportive and respectful way."
        ),
        "high_score": (
            "High score if the student validates pain or anxiety, shows concern, "
            "and uses supportive patient-centered language."
        ),
        "low_score": (
            "Low score if the student ignores feelings, only asks task-focused questions, "
            "or sounds dismissive, rushed, or disrespectful."
        ),
    },
    {
        "name": "Relationship building and patient involvement",
        "source": "K-HCAT",
        "description": (
            "Evaluate whether the student builds trust, explains their role or actions, "
            "and involves the patient in the care plan."
        ),
        "high_score": (
            "High score if the student introduces themself, explains what they are doing, "
            "speaks respectfully, and encourages patient participation."
        ),
        "low_score": (
            "Low score if the student gives commands without building trust, explaining, "
            "or involving the patient."
        ),
    },
    {
        "name": "Patient assessment",
        "source": "DARE2",
        "description": (
            "Evaluate whether the student asks relevant assessment questions about the "
            "patient's condition."
        ),
        "high_score": (
            "High score if the student asks several relevant assessment questions, such as "
            "pain level, location, quality, triggers, related symptoms, breathing, coughing, "
            "movement, incision concerns, or changes in condition."
        ),
        "low_score": (
            "Low score if the student asks no assessment questions or only asks one very "
            "limited question."
        ),
    },
    {
        "name": "Clinical response",
        "source": "DARE2",
        "description": (
            "Evaluate whether the student responds appropriately to the patient's condition "
            "with relevant nursing actions."
        ),
        "high_score": (
            "High score if the student mentions appropriate actions such as checking the incision, "
            "vital signs, medication orders, pain management options, or notifying the nurse/provider."
        ),
        "low_score": (
            "Low score if the student gives no clinical response, gives unsupported advice, "
            "or does not connect assessment findings to care."
        ),
    },
    {
        "name": "Patient education and explanation",
        "source": "K-HCAT",
        "description": (
            "Evaluate whether the student explains care steps clearly in patient-friendly language."
        ),
        "high_score": (
            "High score if the student explains care steps or explains why an action helps recovery. "
            "Examples include explaining pain management, deep breathing, coughing, movement, "
            "incentive spirometer use, keeping lungs clear, or safety steps."
        ),
        "low_score": (
            "Low score if the student gives no education, gives unclear education, "
            "or only gives instructions without explaining the purpose."
        ),
    },
    {
        "name": "Communication clarity",
        "source": "K-HCAT and DARE2",
        "description": (
            "Evaluate whether the student communicates clearly, professionally, respectfully, "
            "and in a patient-centered way."
        ),
        "high_score": (
            "High score if the response is clear, organized, respectful, specific, "
            "and easy for the patient to understand."
        ),
        "low_score": (
            "Low score if the response is vague, confusing, abrupt, disrespectful, "
            "too technical, or hard to understand."
        ),
    },
    {
        "name": "Safety and escalation",
        "source": "DARE2",
        "description": (
            "Evaluate whether the student addresses patient safety risks and escalation."
        ),
        "high_score": (
            "High score if the student mentions safety actions such as call light use, "
            "not getting up alone, fall prevention, staying with the patient, worsening symptoms, "
            "or notifying the nurse/provider."
        ),
        "low_score": (
            "Low score if the student gives no safety guidance, ignores safety risks, "
            "or gives unsafe advice."
        ),
    },
]

# Yegeon: Evaluate each criterion separately because the small local model
# was inconsistent when scoring all categories in one long prompt.
def evaluate_transcript(
    scenario: Scenario,
    transcript: list[Message],
    client: OllamaClient,
) -> dict[str, Any]:
    student_text = _student_text(transcript)

    criteria_results = [
        _evaluate_one_criterion(
            scenario=scenario,
            student_text=student_text,
            criterion=criterion,
            client=client,
        )
        for criterion in CRITERIA
    ]

    feedback = {
        "overall_score": _calculate_overall_score(criteria_results),
        "summary": _build_summary(criteria_results),
        "criteria": criteria_results,
        "strengths": _build_strengths(criteria_results),
        "improvements": _build_improvements(criteria_results),
        "safety_concerns": _build_safety_concerns(criteria_results),
    }

    return feedback


def _student_text(transcript: list[Message]) -> str:
    return "\n".join(
        f"STUDENT: {message.content}"
        for message in transcript
        if message.role == "student"
    )

# Yegeon: Ask the model to score one rubric criterion at a time
# to make feedback more focused and consistent.
def _evaluate_one_criterion(
    scenario: Scenario,
    student_text: str,
    criterion: dict[str, str],
    client: OllamaClient,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": _system_prompt()},
        {
            "role": "user",
            "content": _criterion_prompt(
                scenario=scenario,
                student_text=student_text,
                criterion=criterion,
            ),
        },
    ]

    raw = client.chat(messages, temperature=0.1, format_json=True)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = _recover_criterion_result(raw)

    return _normalize_criterion_result(result, criterion["name"])


def _system_prompt() -> str:
    return """
You are a nursing simulation evaluator.
Evaluate only the nursing student's words.
Do not evaluate the patient.
Do not use patient statements as evidence.
Use only the provided STUDENT lines as evidence.
Return valid JSON only.
""".strip()


def _criterion_prompt(
    scenario: Scenario,
    student_text: str,
    criterion: dict[str, str],
) -> str:
    objectives = "\n".join(f"- {item}" for item in scenario.learning_objectives)
    rubric = "\n".join(
        f"- {name}: {description}"
        for name, description in scenario.evaluation_rubric.items()
    )

    return f"""
Scenario: {scenario.title}
Setting: {scenario.setting}

Learning objectives:
{objectives}

Scenario rubric:
{rubric}

Criterion to evaluate:
Name: {criterion["name"]}
Research source: {criterion["source"]}
Description: {criterion["description"]}
High-score guidance: {criterion["high_score"]}
Low-score guidance: {criterion["low_score"]}

Scoring scale:
1 = absent, unsafe, disrespectful, or not demonstrated.
2 = very limited; one small relevant action but mostly incomplete.
3 = partial; some relevant response but missing important parts.
4 = good; mostly complete with minor gaps.
5 = excellent; clear, specific, patient-centered, and complete.

Student responses to evaluate:
{student_text}

Rules:
- Evaluate only this one criterion: {criterion["name"]}.
- Use only the STUDENT lines above as evidence.
- Do not quote or use patient statements.
- Do not give credit for anything the student did not say.
- If the student did not demonstrate this criterion, give a low score.
- Evidence should explain why the score was given.
- Coaching should give one concrete suggestion for improvement.

Return valid JSON only using exactly this shape:
{{
  "name": "{criterion["name"]}",
  "score": 1,
  "evidence": "Specific evidence from the student's words, or no evidence if missing.",
  "coaching": "Concrete suggestion."
}}

Do not include markdown.
Do not include explanations outside the JSON.
""".strip()


def _normalize_criterion_result(
    result: dict[str, Any],
    expected_name: str,
) -> dict[str, Any]:
    score = _clamp_score(result.get("score"))

    evidence = result.get("evidence") or "No student evidence was provided for this criterion."

    if not isinstance(evidence, str) or len(evidence.strip()) < 5:
        evidence = "No specific student evidence provided."

    coaching = result.get("coaching") or "Address this area more clearly in the student response."

    if score <= 1:
        evidence = "No student evidence was provided for this criterion."

    return {
        "name": expected_name,
        "score": score,
        "evidence": evidence,
        "coaching": coaching,
    }


def _recover_criterion_result(raw: str) -> dict[str, Any]:
    return {
        "score": _recover_score(raw),
        "evidence": _recover_string_field(raw, "evidence"),
        "coaching": _recover_string_field(raw, "coaching"),
    }

# Yegeon: Calculate the overall score from the criterion scores
# instead of relying on the model to generate a separate score.
def _calculate_overall_score(criteria: list[dict[str, Any]]) -> float | None:
    scores = [_clamp_score(item.get("score")) for item in criteria]

    if not scores:
        return None

    return round(sum(scores) / len(scores), 2)


def _build_summary(criteria: list[dict[str, Any]]) -> str:
    overall = _calculate_overall_score(criteria)

    if overall is None:
        return "The evaluator could not calculate an overall score."

    if overall < 2:
        return (
            "The student provided very limited care. The response showed major gaps in "
            "communication, assessment, clinical response, education, and safety."
        )

    if overall < 3:
        return (
            "The student showed some relevant actions, but the response was incomplete "
            "across several communication, assessment, clinical response, education, or safety areas."
        )

    if overall < 4:
        return (
            "The student demonstrated a partially effective response with some strengths, "
            "but several areas still need improvement."
        )

    return (
        "The student demonstrated a strong response using the K-HCAT and DARE2-informed criteria."
    )

# Yegeon: Build strengths only from high-scoring criteria
# that include usable evidence.
def _build_strengths(criteria: list[dict[str, Any]]) -> list[str]:
    strengths = []

    for item in criteria:
        score = _clamp_score(item.get("score"))
        evidence = str(item.get("evidence", "")).strip()

        if score >= 4 and evidence and not evidence.startswith("No specific") and not evidence.startswith("No student"):
            strengths.append(f"{item['name']}: {evidence}")

    return strengths[:3]

def _build_improvements(criteria: list[dict[str, Any]]) -> list[str]:
    improvements = [
        f"{item['name']}: {item['coaching']}"
        for item in criteria
        if _clamp_score(item.get("score")) <= 3
    ]

    return improvements[:3]


def _build_safety_concerns(criteria: list[dict[str, Any]]) -> list[str]:
    for item in criteria:
        if item["name"] == "Safety and escalation" and _clamp_score(item.get("score")) <= 1:
            return ["The student did not clearly address patient safety or escalation."]

    return []


def _clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 1.0

    if score < 1:
        return 1.0
    if score > 5:
        return 5.0
    return score


def _recover_score(raw: str) -> float | None:
    match = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', raw)
    if not match:
        return None

    return _clamp_score(match.group(1))


def _recover_string_field(raw: str, field_name: str) -> str | None:
    pattern = rf'"{field_name}"\s*:\s*"([^"]*)"'
    match = re.search(pattern, raw)
    if match:
        return match.group(1)

    return None