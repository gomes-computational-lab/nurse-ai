from __future__ import annotations

import json
import re
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
        match = re.search(r"\{.*\}", raw, re.DOTALL)

        if not match:
            raise json.JSONDecodeError("No JSON found", raw, 0)

        result = json.loads(match.group())

        return {
            "overall_score": result.get("overall_score"),
            "pain_evaluation": {
                "pain_intensity": result["pain_evaluation"]["pain_intensity"],
                "functional_impact": result["pain_evaluation"]["functional_impact"],
                "pain_distress": result["pain_evaluation"]["pain_distress"],
                "movement_impact": result["pain_evaluation"]["movement_impact"],
                "persistence": result["pain_evaluation"]["persistence"],
            },
        }

    except (json.JSONDecodeError, KeyError, TypeError):
        return {
            "overall_score": None,
            "pain_evaluation": {
                "pain_intensity": None,
                "functional_impact": None,
                "pain_distress": None,
                "movement_impact": None,
                "persistence": None,
            },
        }


def _system_prompt() -> str:
    return """
You are a nursing simulation evaluator.

Evaluate:
1. The student's overall nursing performance.
2. The patient's pain presentation.

Use only information that is actually present in the transcript.

Do not invent symptoms, actions, or pain information.

Return ONLY valid JSON.
""".strip()


def _evaluation_prompt(
    scenario: Scenario,
    transcript: list[Message],
) -> str:

    transcript_text = "\n".join(
        f"{m.role.upper()}: {m.content}"
        for m in transcript
    )

    return f"""
Evaluate this nursing simulation.

Scenario: {scenario.title}

Setting: {scenario.setting}

Transcript:
{transcript_text}

Return ONLY this JSON structure:

{{
  "overall_score": 1,
  "pain_evaluation": {{
    "pain_intensity": 1,
    "functional_impact": 1,
    "pain_distress": 1,
    "movement_impact": 1,
    "persistence": 1
  }}
}}

Use 1 through 5 for all scores.

overall_score:
Evaluate the student's overall nursing performance.

pain_intensity:
Severity of the patient's pain.

functional_impact:
How much the pain interferes with normal activities.

pain_distress:
How much distress the patient shows because of the pain.

movement_impact:
How much the pain affects movement, coughing, breathing, or repositioning.

persistence:
How ongoing or persistent the pain appears to be.

Use ONLY information from the transcript.

""".strip()