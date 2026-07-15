from __future__ import annotations

from typing import Callable, Literal

from sim.models import Message, Scenario
from sim.ollama_client import OllamaClient


ResponseMode = Literal["text", "voice"]
VOICE_MAX_TOKENS = 80


class SimulationSession:
    def __init__(self, scenario: Scenario, client: OllamaClient):
        self.scenario = scenario
        self.client = client
        self.transcript: list[Message] = []

    def opening(
        self,
        on_chunk: Callable[[str], None] | None = None,
        *,
        response_mode: ResponseMode = "text",
    ) -> str:
        response = self.client.chat(
            self._patient_messages(response_mode=response_mode),
            temperature=0.75,
            max_tokens=VOICE_MAX_TOKENS if response_mode == "voice" else None,
            on_chunk=on_chunk,
        )
        self.transcript.append(Message(role="patient", content=response))
        return response

    def respond(
        self,
        student_response: str,
        on_chunk: Callable[[str], None] | None = None,
        *,
        response_mode: ResponseMode = "text",
    ) -> str:
        messages = self._patient_messages(
            student_response=student_response,
            response_mode=response_mode,
        )
        self.transcript.append(Message(role="student", content=student_response))
        response = self.client.chat(
            messages,
            temperature=0.75,
            max_tokens=VOICE_MAX_TOKENS if response_mode == "voice" else None,
            on_chunk=on_chunk,
        )
        self.transcript.append(Message(role="patient", content=response))
        return response

    def opening_prompt_char_count(self, *, response_mode: ResponseMode = "text") -> int:
        return self._prompt_char_count(self._patient_messages(response_mode=response_mode))

    def response_prompt_char_count(
        self,
        student_response: str,
        *,
        response_mode: ResponseMode = "text",
    ) -> int:
        return self._prompt_char_count(
            self._patient_messages(
                student_response=student_response,
                response_mode=response_mode,
            )
        )

    def _patient_messages(
        self,
        student_response: str | None = None,
        *,
        response_mode: ResponseMode = "text",
    ) -> list[dict[str, str]]:
        system_prompt = self._system_prompt()
        if response_mode == "voice":
            system_prompt += (
                "\n\nVoice response style:\n"
                "- Respond in one to three short, naturally punctuated spoken sentences.\n"
                "- Do not use Markdown, lists, stage directions, or parenthetical actions."
            )
        messages = [{"role": "system", "content": system_prompt}]

        if not self.transcript:
            messages.append({"role": "user", "content": self.scenario.opening_prompt})
            return messages

        for message in self.transcript:
            if message.role == "student":
                messages.append({"role": "user", "content": message.content})
            else:
                messages.append({"role": "assistant", "content": message.content})

        if student_response is not None:
            messages.append({"role": "user", "content": student_response})

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

    def _prompt_char_count(self, messages: list[dict[str, str]]) -> int:
        return sum(len(message["content"]) for message in messages)
