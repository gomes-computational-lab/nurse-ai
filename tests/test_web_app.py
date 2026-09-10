from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from sim.models import Message, Scenario


APP_PATH = Path(__file__).resolve().parent.parent / "streamlit_app.py"


def scenario() -> Scenario:
    return Scenario(
        id="browser_test",
        title="Browser Test",
        role="Patient",
        setting="Test room",
        patient_profile={"name": "Pat"},
        clinical_context={"symptom": "pain"},
        behavior_guidelines=["Answer naturally."],
        opening_prompt="Ask for help.",
        learning_objectives=["Communicate clearly."],
        evaluation_rubric={"Communication": "Communicates clearly."},
    )


class FakeSession:
    def __init__(self, selected_scenario, client):
        del client
        self.scenario = selected_scenario
        self.transcript: list[Message] = []

    def opening(self, on_chunk, *, response_mode):
        self._assert_voice_mode(response_mode)
        response = "Can you help me with this pain?"
        on_chunk(response)
        self.transcript.append(Message(role="patient", content=response))
        return response

    def respond(self, student_response, on_chunk, *, response_mode):
        self._assert_voice_mode(response_mode)
        self.transcript.append(Message(role="student", content=student_response))
        response = "It is an eight out of ten."
        on_chunk(response)
        self.transcript.append(Message(role="patient", content=response))
        return response

    @staticmethod
    def _assert_voice_mode(response_mode):
        if response_mode != "voice":
            raise AssertionError("Browser responses should use voice mode.")


class StreamlitAppTests(unittest.TestCase):
    def _patch_dependencies(self):
        return (
            patch("sim.web_app.list_scenarios", return_value=[scenario()]),
            patch("sim.web_app.OllamaClient", return_value=object()),
            patch("sim.web_app.SimulationSession", FakeSession),
        )

    def test_initial_page_renders_scenario_and_start_control(self) -> None:
        patches = self._patch_dependencies()
        with patches[0], patches[1], patches[2]:
            app = AppTest.from_file(str(APP_PATH)).run()

        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "Nursing AI Simulation")
        self.assertIn("Browser Test", [item.value for item in app.subheader])
        self.assertIn("Start simulation", [button.label for button in app.button])

    def test_typed_turn_uses_session_and_renders_both_messages(self) -> None:
        patches = self._patch_dependencies()
        with patches[0], patches[1], patches[2]:
            app = AppTest.from_file(str(APP_PATH)).run()
            app.toggle[0].set_value(False).run()
            self._button(app, "Start simulation").click().run()

            self.assertFalse(app.exception)
            self.assertEqual(len(app.get("audio_input")), 0)
            self.assertTrue(all(item.disabled for item in app.sidebar.text_input))
            self.assertIn(
                "Can you help me with this pain?",
                [markdown.value for markdown in app.markdown],
            )

            app.text_area[0].input("I will assess your pain now.").run()
            self._button(app, "Send response").click().run()

        self.assertFalse(app.exception)
        rendered = [markdown.value for markdown in app.markdown]
        self.assertIn("I will assess your pain now.", rendered)
        self.assertIn("It is an eight out of ten.", rendered)

    def test_end_saves_feedback_and_new_simulation_resets_state(self) -> None:
        feedback = {
            "overall_score": 4,
            "summary": "A clear assessment.",
            "criteria": {
                "Communication": {
                    "score": 4,
                    "evidence": "Asked about pain.",
                    "coaching": "Explore associated symptoms.",
                }
            },
            "strengths": ["Used clear language."],
            "improvements": ["Ask about pain quality."],
            "safety_concerns": [],
        }
        patches = self._patch_dependencies()
        with (
            patches[0],
            patches[1],
            patches[2],
            patch("sim.web_app.evaluate_transcript", return_value=feedback) as evaluate,
            patch(
                "sim.web_app.save_result",
                return_value=Path("transcripts/browser_test.json"),
            ) as save,
        ):
            app = AppTest.from_file(str(APP_PATH)).run()
            app.toggle[0].set_value(False).run()
            self._button(app, "Start simulation").click().run()
            self._button(app, "End simulation").click().run()

            self.assertFalse(app.exception)
            self.assertIn("Feedback", [header.value for header in app.header])
            self.assertIn("A clear assessment.", [item.value for item in app.markdown])
            self.assertEqual(app.metric[0].value, "4/5")
            evaluate.assert_called_once()
            save.assert_called_once()

            self._button(app, "New simulation").click().run()

        self.assertFalse(app.exception)
        self.assertIn("Start simulation", [button.label for button in app.button])
        self.assertNotIn("Feedback", [header.value for header in app.header])

    @staticmethod
    def _button(app: AppTest, label: str):
        return next(button for button in app.button if button.label == label)


if __name__ == "__main__":
    unittest.main()
