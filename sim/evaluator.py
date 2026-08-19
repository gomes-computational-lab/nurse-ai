from __future__ import annotations

import json
from typing import Any

from sim.models import Message, Scenario
from sim.ollama_client import OllamaClient


def evaluate_transcript(
    scenario: Scenario,
    transcript: list[Message],
    client: OllamaClient,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": _evaluation_prompt(scenario, transcript)},
    ]
    raw = client.chat(messages, temperature=0.1, format_json=True)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "overall_score": None,
            "summary": raw,
            "criteria": {},
            "strengths": [],
            "improvements": ["Evaluator returned non-JSON feedback."],
            "safety_concerns": [],
        }


def _system_prompt() -> str:
    return """
You are a nursing simulation evaluator. Assess only the nursing student's responses.
Be specific, fair, and grounded in the transcript. Do not invent actions the student did not take.
Return valid JSON only.
""".strip()


def _evaluation_prompt(scenario: Scenario, transcript: list[Message]) -> str:
    transcript_text = "\n".join(f"{m.role.upper()}: {m.content}" for m in transcript)
    rubric = "\n".join(f"- {name}: {description}" for name, description in scenario.evaluation_rubric.items())
    objectives = "\n".join(f"- {item}" for item in scenario.learning_objectives)

    return f"""
Scenario: {scenario.title}
Setting: {scenario.setting}

Learning objectives:
{objectives}

Rubric:
{rubric}

Transcript:
{transcript_text}

Return this JSON shape:
{{
  "overall_score": 1,
  "summary": "Brief overall assessment.",
  "criteria": {{
    "criterion name": {{
      "score": 1,
      "evidence": "Specific evidence from student responses.",
      "coaching": "Concrete suggestion."
    }}
  }},
  "strengths": ["Specific strength"],
  "improvements": ["Specific next step"],
  "safety_concerns": ["Any safety concern, or empty list"]
}}

Use scores from 1 to 5, where 1 is unsafe or absent and 5 is excellent.
""".strip()

