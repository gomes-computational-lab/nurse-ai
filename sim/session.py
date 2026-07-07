from __future__ import annotations

from typing import Callable

from sim.models import Message, Scenario
from sim.ollama_client import OllamaClient


class SimulationSession:
    def __init__(self, scenario: Scenario, client: OllamaClient):
        self.scenario = scenario
        self.client = client
        self.transcript: list[Message] = []

    def opening(self, on_chunk: Callable[[str], None] | None = None) -> str:
        response = self.client.chat(self._patient_messages(), temperature=0.75, on_chunk=on_chunk)
        self.transcript.append(Message(role="patient", content=response))
        return response

    def respond(self, student_response: str, on_chunk: Callable[[str], None] | None = None) -> str:
        self.transcript.append(Message(role="student", content=student_response))
        response = self.client.chat(self._patient_messages(), temperature=0.75, on_chunk=on_chunk)
        self.transcript.append(Message(role="patient", content=response))
        return response

    def _patient_messages(self) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self._system_prompt()}]

        if not self.transcript:
            messages.append({"role": "user", "content": self.scenario.opening_prompt})
            return messages

        for message in self.transcript:
            if message.role == "student":
                messages.append({"role": "user", "content": message.content})
            else:
                messages.append({"role": "assistant", "content": message.content})

        messages.append(
            {
                "role": "user",
                "content": (
                    "Continue the simulation as the patient or family member. "
                    "Respond naturally to the student's last statement. Do not evaluate the student."
                ),
            }
        )
        return messages

    def _system_prompt(self) -> str:
        behavior = "\n".join(f"- {item}" for item in self.scenario.behavior_guidelines)
        objectives = "\n".join(f"- {item}" for item in self.scenario.learning_objectives)
        return f"""
You are role-playing in a nursing education simulation.

Role: {self.scenario.role}
Setting: {self.scenario.setting}

Patient profile:
{self.scenario.patient_profile}

Clinical context:
{self.scenario.clinical_context}

Behavior guidelines:
{behavior}

Learning objectives the student should be able to demonstrate:
{objectives}

Stay in character as the patient or family member. Give short, realistic conversational responses.
Reveal clinical information only if the student asks appropriate questions or provides appropriate care.
Do not provide coaching, scoring, or meta-commentary during the conversation.
If the student says something unsafe, react realistically with concern, confusion, fear, or resistance.
""".strip()
