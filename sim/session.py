from __future__ import annotations

from typing import Callable, Literal

from sim.models import Message, Scenario
from sim.ollama_client import OllamaClient


ResponseMode = Literal["text", "voice"]
VOICE_MAX_TOKENS = 80


class SimulationSession:
    def __init__(
        self,
        scenario: Scenario,
        client: OllamaClient,
        *,
        learner_count: int | None = None,
    ):
        self.scenario = scenario
        self.client = client
        self.transcript: list[Message] = []
        selected_count = (
            scenario.learner_configuration.min_nurses
            if learner_count is None
            else learner_count
        )
        self.selected_learner_roles = scenario.learner_configuration.roles_for_count(
            selected_count
        )
        self.active_learner_role = self.selected_learner_roles[0]

    def select_learner_role(self, speaker_role: str) -> None:
        if speaker_role not in self.selected_learner_roles:
            available = ", ".join(self.selected_learner_roles)
            raise ValueError(
                f"Learner role '{speaker_role}' is not active in this session. "
                f"Active roles: {available}."
            )
        self.active_learner_role = speaker_role

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
        speaker_role: str | None = None,
    ) -> str:
        selected_speaker = speaker_role or self.active_learner_role
        if selected_speaker not in self.selected_learner_roles:
            raise ValueError(f"Learner role '{selected_speaker}' is not active in this session.")
        messages = self._agent_messages(
            student_response=student_response,
            student_speaker_role=selected_speaker,
            response_mode=response_mode,
        )
        self.transcript.append(
            Message(
                role="student",
                content=student_response,
                speaker_role=selected_speaker,
            )
        )
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
        speaker_role: str | None = None,
    ) -> int:
        return self._prompt_char_count(
            self._agent_messages(
                student_response=student_response,
                student_speaker_role=speaker_role or self.active_learner_role,
                response_mode=response_mode,
            )
        )

    def _agent_messages(
        self,
        student_response: str | None = None,
        *,
        student_speaker_role: str | None = None,
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
                messages.append(
                    {
                        "role": "user",
                        "content": self._learner_message_content(
                            message.content,
                            message.speaker_role,
                        ),
                    }
                )
            else:
                messages.append({"role": "assistant", "content": message.content})

        if student_response is not None:
            messages.append(
                {
                    "role": "user",
                    "content": self._learner_message_content(
                        student_response,
                        student_speaker_role,
                    ),
                }
            )

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
        participant_context = self._participant_context()
        learner_context = self._learner_context()
        timeline_context = self._simulation_timeline_context()
        role_rules = self._role_rules()
        return f"""
You are role-playing in a nursing education simulation.

AI agent role: {role_name}
Display role: {self.scenario.role}
{identity}
Setting: {self.scenario.setting}
{phase_context}
{timeline_context}

{participant_context}
{learner_context}

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

    def _participant_context(self) -> str:
        if not self.scenario.participants:
            return ""
        participants = []
        for participant in self.scenario.participants:
            label = participant.display_name or participant.id
            details = [participant.participant_type]
            if participant.role:
                details.append(f"role={participant.role}")
            participants.append(f"- {label}: {', '.join(details)}")
        return "Participants:\n" + "\n".join(participants)

    def _learner_context(self) -> str:
        roles = ", ".join(role.replace("_", " ") for role in self.selected_learner_roles)
        return f"Active student nurse roles for this simulation run: {roles}."

    def _simulation_timeline_context(self) -> str:
        timeline = self.scenario.simulation_timeline
        if timeline is None:
            return ""

        details = []
        if timeline.simulation_date:
            details.append(f"Simulation date (the fictional 'today'): {timeline.simulation_date}")
        if timeline.clinical_day is not None:
            details.append(f"Clinical day: {timeline.clinical_day}")
        if timeline.time_of_day:
            details.append(f"Time of day: {timeline.time_of_day}")
        rendered = "\n".join(f"- {detail}" for detail in details)
        return (
            "Simulation timeline (fictional clinical time; authoritative):\n"
            f"{rendered}\n"
            "Interpret today, yesterday, tomorrow, length of stay, and clinical day relative "
            "to this simulation timeline. Never substitute the host computer's current date."
        )

    def _learner_message_content(self, content: str, speaker_role: str | None) -> str:
        if len(self.selected_learner_roles) == 1 or speaker_role is None:
            return content
        label = speaker_role.replace("_", " ").title()
        return f"{label}: {content}"

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
