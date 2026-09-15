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
            self._agent_messages(response_mode=response_mode),
            temperature=0.75,
            max_tokens=VOICE_MAX_TOKENS if response_mode == "voice" else None,
            on_chunk=on_chunk,
        )
        self.transcript.append(Message(role=self.scenario.agent_role, content=response))
        return response

    def respond(
        self,
        student_response: str,
        on_chunk: Callable[[str], None] | None = None,
        *,
        response_mode: ResponseMode = "text",
    ) -> str:
        messages = self._agent_messages(
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
        self.transcript.append(Message(role=self.scenario.agent_role, content=response))
        return response

    def opening_prompt_char_count(self, *, response_mode: ResponseMode = "text") -> int:
        return self._prompt_char_count(self._agent_messages(response_mode=response_mode))

    def response_prompt_char_count(
        self,
        student_response: str,
        *,
        response_mode: ResponseMode = "text",
    ) -> int:
        return self._prompt_char_count(
            self._agent_messages(
                student_response=student_response,
                response_mode=response_mode,
            )
        )

    def _agent_messages(
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
                    f"Continue the simulation as the {self.scenario.agent_role.replace('_', ' ')}. "
                    "Respond naturally to the student's last statement. Do not evaluate the student."
                ),
            }
        )
        return messages

    def _system_prompt(self) -> str:
        behavior = "\n".join(
            f"- {item}"
            for item in (
                self.scenario.behavior_guidelines
                + self.scenario.phase_behavior_guidelines
            )
        )
        objectives = "\n".join(f"- {item}" for item in self.scenario.learning_objectives)
        role_name = self.scenario.agent_role.replace("_", " ")
        identity = self._identity_context()
        phase_context = self._phase_context()
        role_rules = self._role_rules()
        return f"""
You are role-playing in a nursing education simulation.

AI agent role: {role_name}
Display role: {self.scenario.role}
{identity}
Setting: {self.scenario.setting}
{phase_context}

Patient profile:
{self.scenario.patient_profile}

Clinical context:
{self.scenario.clinical_context}

Behavior guidelines:
{behavior}

Learning objectives the student should be able to demonstrate:
{objectives}

{role_rules}
Do not provide coaching, scoring, or meta-commentary during the conversation.
If the student says something unsafe, react realistically with concern, confusion, fear, or resistance.
""".strip()

    def _identity_context(self) -> str:
        details = []
        if self.scenario.character_name:
            details.append(f"Character name: {self.scenario.character_name}")
        if self.scenario.relationship:
            details.append(f"Relationship to patient: {self.scenario.relationship}")
        if self.scenario.patient_name:
            details.append(f"Patient name: {self.scenario.patient_name}")
        return "\n".join(details)

    def _phase_context(self) -> str:
        if self.scenario.scenario_phase is None:
            return ""
        background = self.scenario.active_phase.get("background", "")
        emotional_state = self.scenario.emotional_state
        return (
            f"Current simulation phase: {self.scenario.scenario_phase}\n"
            f"Phase background: {background}\n"
            f"Internal emotional state (tone guidance only; never speak these numeric values): "
            f"{emotional_state}"
        )

    def _role_rules(self) -> str:
        if self.scenario.agent_role == "family_member":
            return """
Stay in character as the family member and respond conversationally in short, natural speech.
Do not behave like ChatGPT, a clinician, or an instructor, and do not provide nursing or medical advice.
Do not invent clinical facts or claim knowledge that has not reasonably been shared with the family member.
Do not repeat the entire backstory or force every objective into each response.
Let emotional reactions and questions emerge naturally and remain consistent with the selected phase.
The family member may be worried, frustrated, confused, or upset without becoming unrealistically hostile.
""".strip()
        return """
Stay in character as the patient and give short, realistic conversational responses.
Reveal clinical information only if the student asks appropriate questions or provides appropriate care.
Do not invent clinical facts that are not supported by the scenario.
""".strip()

    def _prompt_char_count(self, messages: list[dict[str, str]]) -> int:
        return sum(len(message["content"]) for message in messages)
