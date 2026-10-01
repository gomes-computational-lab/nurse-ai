from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import ANY, patch

from streamlit.testing.v1 import AppTest

from sim.expressive_tts import SynthesizedAudio
from sim.models import Message, Scenario
from sim.ollama_client import OllamaError


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
            patch("sim.web_app._preload_local_speech_models", return_value=True),
            patch(
                "sim.web_app.list_approved_voices",
                return_value={"child_female_8yo": Path("voices/child_female_8yo.wav")},
            ),
        )

    def test_initial_page_renders_scenario_and_start_control(self) -> None:
        patches = self._patch_dependencies()
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            app = AppTest.from_file(str(APP_PATH)).run()

        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "Nursing AI Simulation")
        self.assertIn("Browser Test", [item.value for item in app.subheader])
        self.assertIn("How it works", [item.value for item in app.subheader])
        self.assertTrue(
            any(
                "Leave **Patient speaks aloud** on to hear Ruth" in item.value
                for item in app.sidebar.markdown
            )
        )
        self.assertIn("Start simulation", [button.label for button in app.button])
        self.assertIn(
            "Load voice and recording", [button.label for button in app.button]
        )
        self.assertNotIn(
            "Prepare local speech models", [button.label for button in app.button]
        )
        self.assertNotIn("Check voice engine", [button.label for button in app.button])
        self.assertEqual(
            [selectbox.label for selectbox in app.selectbox],
            ["Scenario", "Patient voice"],
        )
        self.assertIn(
            "Ruth — child voice",
            app.selectbox[1].options,
        )
        self.assertEqual(
            [toggle.label for toggle in app.toggle], ["Patient speaks aloud"]
        )
        self.assertEqual(len(app.sidebar.text_input), 0)
        self.assertTrue(any("AI-generated" in item.value for item in app.caption))

    def test_typed_turn_uses_session_and_renders_both_messages(self) -> None:
        patches = self._patch_dependencies()
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            app = AppTest.from_file(str(APP_PATH)).run()
            app.toggle[0].set_value(False).run()
            self._button(app, "Start simulation").click().run()

            self.assertFalse(app.exception)
            self.assertEqual(len(app.get("audio_input")), 0)
            self.assertTrue(all(item.disabled for item in app.sidebar.text_input))
            self.assertTrue(
                any(
                    "A pulsing red dot means recording is active" in item.value
                    for item in app.markdown
                )
            )
            self.assertTrue(
                any(
                    "Begin speaking within five seconds" in item.value
                    and "Three seconds of silence" in item.value
                    for item in app.markdown
                )
            )
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
            patches[3],
            patches[4],
            patch("sim.web_app.evaluate_transcript", return_value=feedback) as evaluate,
            patch(
                "sim.web_app.save_result",
                return_value=Path("transcripts/browser_test.json"),
            ) as save,
        ):
            app = AppTest.from_file(str(APP_PATH)).run()
            app.toggle[0].set_value(False).run()
            self._button(app, "Start simulation").click().run()
            self._button(app, "End and score simulation").click().run()

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

    def test_end_without_scoring_saves_transcript_and_skips_evaluation(self) -> None:
        patches = self._patch_dependencies()
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patch("sim.web_app.evaluate_transcript") as evaluate,
            patch(
                "sim.web_app.save_result",
                return_value=Path("transcripts/browser_test.json"),
            ) as save,
        ):
            app = AppTest.from_file(str(APP_PATH)).run()
            app.toggle[0].set_value(False).run()
            self._button(app, "Start simulation").click().run()
            self._button(app, "End without scoring").click().run()

        self.assertFalse(app.exception)
        self.assertTrue(
            any(
                "Simulation ended without scoring" in message.value
                for message in app.info
            )
        )
        self.assertNotIn("Feedback", [header.value for header in app.header])
        evaluate.assert_not_called()
        save.assert_called_once()
        self.assertEqual(len(save.call_args.args), 2)

    def test_start_prepares_local_models_once_with_ruth(self) -> None:
        patches = self._patch_dependencies()
        with (
            patches[0],
            patches[1] as client,
            patches[2],
            patches[3] as preload,
            patches[4],
            patch(
                "sim.web_app.synthesize_patient_audio",
                return_value=SynthesizedAudio(b"wav", "audio/wav", "chatterbox_nano"),
            ),
        ):
            app = AppTest.from_file(str(APP_PATH)).run()
            self._button(app, "Start simulation").click().run()

        self.assertFalse(app.exception)
        client.assert_called_once_with(
            model="llama3.1", host="http://localhost:11434"
        )
        preload.assert_called_once_with(
            stt_model="tiny.en",
            include_chatterbox=True,
            chatterbox_voice="child_female_8yo",
            voice_catalog_dir="voices",
            _on_stage=ANY,
        )
        self.assertEqual(
            app.session_state.simulation_config["tts_provider"],
            "chatterbox_nano",
        )
        self.assertFalse(
            app.session_state.simulation_config["allow_online_edge_fallback"]
        )

    def test_voice_and_recording_can_be_loaded_before_start(self) -> None:
        patches = self._patch_dependencies()
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3] as preload,
            patches[4],
        ):
            app = AppTest.from_file(str(APP_PATH)).run()
            self._button(app, "Load voice and recording").click().run()

        self.assertFalse(app.exception)
        self.assertTrue(self._button(app, "Load voice and recording").disabled)
        self.assertIsNone(app.session_state.simulation_session)
        self.assertEqual(
            app.session_state.prepared_speech_models,
            ("tiny.en", True, "child_female_8yo", "voices"),
        )
        preload.assert_called_once_with(
            stt_model="tiny.en",
            include_chatterbox=True,
            chatterbox_voice="child_female_8yo",
            voice_catalog_dir="voices",
            _on_stage=ANY,
        )

    def test_speech_preload_reports_each_stage_in_order(self) -> None:
        from sim.web_app import _preload_local_speech_models

        stages: list[str] = []
        with (
            patch("sim.web_app._preload_speech_to_text_resource") as transcription,
            patch("sim.web_app._preload_chatterbox_runtime_resource") as voice,
            patch("sim.web_app._prepare_chatterbox_voice_resource") as prepare,
        ):
            _preload_local_speech_models(
                stt_model="tiny.en",
                include_chatterbox=True,
                chatterbox_voice="child_female_8yo",
                voice_catalog_dir="voices",
                _on_stage=stages.append,
            )

        self.assertEqual(
            stages,
            [
                "Loading transcription model",
                "Loading patient voice",
                "Preparing Ruth’s voice",
            ],
        )
        transcription.assert_called_once_with("tiny.en")
        voice.assert_called_once_with()
        prepare.assert_called_once_with("child_female_8yo", "voices")

    def test_missing_ruth_voice_has_clear_nontechnical_recovery(self) -> None:
        patches = self._patch_dependencies()
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patch("sim.web_app.list_approved_voices", return_value={}),
        ):
            app = AppTest.from_file(str(APP_PATH)).run()

            self.assertTrue(self._button(app, "Start simulation").disabled)
            self.assertTrue(
                any(
                    "Ruth's voice is not installed" in warning.value
                    for warning in app.sidebar.warning
                )
            )

            app.toggle[0].set_value(False).run()

        self.assertFalse(self._button(app, "Start simulation").disabled)
        self.assertEqual(
            [selectbox.label for selectbox in app.selectbox],
            ["Scenario"],
        )

    def test_failed_safe_response_preserves_draft_and_clears_partial_output(
        self,
    ) -> None:
        class FailingSession(FakeSession):
            def respond(self, student_response, on_chunk, *, response_mode):
                del student_response
                self._assert_voice_mode(response_mode)
                on_chunk("Pain level: 8/10")
                raise OllamaError(
                    "The patient response could not be prepared safely. Please try again."
                )

        patches = self._patch_dependencies()
        with (
            patches[0],
            patches[1],
            patch("sim.web_app.SimulationSession", FailingSession),
            patches[3],
            patches[4],
            patch(
                "sim.web_app.synthesize_patient_audio",
                return_value=SynthesizedAudio(b"wav", "audio/wav", "chatterbox_nano"),
            ) as synthesize,
        ):
            app = AppTest.from_file(str(APP_PATH)).run()
            self._button(app, "Start simulation").click().run()
            app.text_area[0].input("I will assess your pain now.").run()
            self._button(app, "Send response").click().run()

        self.assertFalse(app.exception)
        self.assertEqual(app.text_area[0].value, "I will assess your pain now.")
        self.assertEqual(app.session_state.draft_text, "I will assess your pain now.")
        self.assertEqual(len(app.session_state.simulation_session.transcript), 1)
        self.assertEqual(synthesize.call_count, 1)
        self.assertFalse(any("Pain level: 8/10" in item.value for item in app.markdown))
        self.assertTrue(any("prepared safely" in error.value for error in app.error))

    @staticmethod
    def _button(app: AppTest, label: str):
        return next(button for button in app.button if button.label == label)


if __name__ == "__main__":
    unittest.main()
