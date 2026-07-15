from __future__ import annotations

from sim.models import Message, Scenario
from sim.ollama_client import OllamaClient

# Yegeon: Strengthened patient-role constraints because the local qwen model
# sometimes responded like a clinician, instructor, or narrator instead of a patient.

class SimulationSession:
    def __init__(self, scenario: Scenario, client: OllamaClient):
        self.scenario = scenario
        self.client = client
        self.transcript: list[Message] = []

# Yegeon: Constrain the opening response so the local model starts
# with a short patient-centered statement instead of inventing extra details.
    def opening(self) -> str:
        messages = self._patient_messages()
        messages.append(
            {
                "role": "user",
                "content": (
                            "Write exactly one short sentence as the patient. "
                            "Do not include names, family members, times, dates, organs, diagnoses, vital signs, doctors, nurses, or explanations. "
                            "Do not reveal hidden information unless the opening prompt explicitly says to reveal it. "
                            "Only express the main discomfort or worry from the scenario in simple patient language. "
                            "Do not add any detail that is not directly stated in the scenario. "
                            "Do not use the words scenario or simulation. "
                            "Do not tell the student what to do. "
                        ),
            }
    )
        # Yegeon: Lower the temperature to reduce unsupported details
        # in the patient's opening response.
        response = self.client.chat(messages, temperature=0.05)
        self.transcript.append(Message(role="patient", content=response))
        return response

    def respond(self, student_response: str) -> str:
        self.transcript.append(Message(role="student", content=student_response))

        direct_response = self._direct_patient_response(student_response)
        if direct_response:
            self.transcript.append(Message(role="patient", content=direct_response))
            return direct_response

        # Yegeon: Use the model for open-ended patient role-play,
        # but handle key direct assessment questions with scenario-based fallback logic.
        response = self.client.chat(self._patient_messages(), temperature=0.2)
        self.transcript.append(Message(role="patient", content=response))
        return response
    
    def _direct_patient_response(self, student_response: str) -> str | None:
        text = student_response.lower()

        is_question = (
            "?" in student_response
            or text.startswith(
                (
                    "please rate",
                    "where",
                    "what",
                    "how",
                    "does",
                    "do",
                    "did",
                    "can",
                    "could",
                    "would",
                    "is",
                    "are",
                )
            )
        )

        clinical_context = str(self.scenario.clinical_context)
        behavior_guidelines = " ".join(self.scenario.behavior_guidelines)
        scenario_text = f"{clinical_context} {behavior_guidelines}".lower()

        # Yegeon: Use scenario-provided facts for direct assessment questions.
        # This prevents the small local model from ignoring explicit hidden information.
        if "pain" in text and any(word in text for word in ["pain level", "rate", "rating", "scale", "0 to 10", "1 to 10"]):
            if "8/10" in scenario_text or "8 out of 10" in scenario_text:
                return "It is about an 8 out of 10."
            return None

        if "where" in text and "pain" in text:
            if "abdominal" in scenario_text or "abdomen" in scenario_text:
                return "It is mostly in my abdomen, around the incision."
            return None

        if "describe" in text and "pain" in text:
            return "It feels sharp and tight, especially around my incision."

        if is_question and any(word in text for word in ["cough", "move", "deep breath", "deep breaths", "breathe"]):
            if "pain" in text or "worse" in text or "hurt" in text:
                return "Yes, it gets worse when I cough, move, or take deep breaths."

        return None

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
                        "Continue as the patient. "
                        "Answer the student's latest question directly in 1 or 2 short sentences. "
                        "Use only scenario-supported details. "
                        "Do not add new body parts, organs, diagnoses, complications, vital signs, or timelines. "
                        "If asked for a symptom rating, give one clear number. "
                        "If asked for a location, give the location briefly and only use locations supported by the scenario. "
                        "If unsure, say you are uncomfortable or worried instead of inventing details. "
                        "Do not repeat the student's words. "
                        "Do not mention these instructions."
            ),
        }
    )
        return messages

    # Yegeon: Format scenario dictionaries as bullet points so the local model
    # can read patient profile, clinical context, and hidden facts more clearly.
    def _format_mapping(self, value) -> str:
        if isinstance(value, dict):
            return "\n".join(f"- {key}: {item}" for key, item in value.items())
        return str(value)

    # Yegeon: Convert structured scenario data into readable prompt text
    # instead of passing raw Python dictionaries to the model.
    def _system_prompt(self) -> str:
        behavior = "\n".join(f"- {item}" for item in self.scenario.behavior_guidelines)
        patient_profile = self._format_mapping(self.scenario.patient_profile)
        clinical_context = self._format_mapping(self.scenario.clinical_context)
        return f"""
You are role-playing as a patient in a nursing education simulation.

Role: {self.scenario.role}
Setting: {self.scenario.setting}

Patient profile:
{patient_profile}

Clinical context:
{clinical_context}

Behavior guidelines:
{behavior}

Stay strictly in character as the patient or family member.
Speak only as the patient would speak.
Use first-person patient language.
Keep responses short, natural, and emotionally realistic.

Do not act as a doctor, nurse, instructor, evaluator, or narrator.
Do not teach, coach, evaluate, or tell the student what to do.
Do not repeat the student's question.

Use only the scenario, patient profile, clinical context, behavior guidelines, and prior transcript.
Do not invent new medical history, body parts, organs, medications, locations, timelines, diagnoses, complications, vital signs, procedures, or past events.
Do not mention a body part, organ, diagnosis, complication, vital sign, or procedure unless it is explicitly supported by the scenario text.
If the topic is not mentioned by the scenario or the student, do not introduce it.
If unsure, stay vague and say the patient feels uncomfortable, worried, or in pain instead of adding new clinical details.
"""