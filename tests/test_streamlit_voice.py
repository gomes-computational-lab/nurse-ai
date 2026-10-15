from __future__ import annotations

import base64
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from app.voice_turn import VOICE_TURN_JS
from sim.models import Message


class StreamlitVoiceTests(unittest.TestCase):
    def setUp(self):
        self.simulation = SimpleNamespace(
            scenario=SimpleNamespace(title="Test"),
            transcript=[Message("patient", "Hello.")],
        )
        self.app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app/streamlit_app.py"))
        for key, value in {
            "simulation": self.simulation,
            "recording_number": 0,
            "voice_reply": "Hello.",
            "voice_pause_ms": 700,
            "voice_threshold": 0.015,
        }.items():
            self.app.session_state[key] = value
        self.voice = patch("app.voice_turn.voice_turn").start()
        self.addCleanup(patch.stopall)

    def queue_audio(self):
        self.app.session_state["pending_audio"] = {
            "base64": base64.b64encode(b"audio").decode(),
            "mime_type": "audio/webm",
        }

    def test_start_sets_voice_options_and_plays_opening(self):
        self.app.session_state["simulation"] = None
        self.simulation.opening = lambda **kwargs: "Welcome."
        with patch("sim.scenarios.list_scenarios", return_value=[self.simulation.scenario]), patch(
            "sim.session.SimulationSession", return_value=self.simulation
        ):
            self.app.run()
            self.app.button[0].click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.voice.call_args.kwargs["reply"], "Welcome.")
        self.assertEqual(self.voice.call_args.kwargs["pause_ms"], 700)

    def test_pause_processes_turn_and_rearms_without_send_button(self):
        def respond(text, **kwargs):
            self.simulation.transcript.extend([
                Message("student", text), Message("patient", "Thank you."),
            ])
            return "Thank you."

        self.simulation.respond = respond
        self.queue_audio()
        with patch("sim.speech_to_text.transcribe_audio", return_value="Hello nurse.") as transcribe:
            self.app.run()
            self.assertFalse(self.app.exception)
            self.assertEqual(self.app.session_state["recording_number"], 1)
            self.assertIsNone(self.app.session_state["pending_audio"])
            self.assertEqual(self.voice.call_args.kwargs["reply"], "Thank you.")
            self.assertEqual(self.voice.call_args.kwargs["key"], "recording_1")
            self.assertEqual([b.label for b in self.app.button], ["End and save"])
            self.app.run()
            transcribe.assert_called_once()
            self.assertEqual(len(self.simulation.transcript), 3)

    def test_empty_transcription_rearms_without_ai_reply(self):
        self.queue_audio()
        with patch("sim.speech_to_text.transcribe_audio", return_value=""):
            self.app.run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.voice.call_args.kwargs["reply"], "")
        self.assertEqual(len(self.simulation.transcript), 1)

    def test_failed_reply_can_retry_without_duplicating_student_message(self):
        calls = []

        def respond(text, **kwargs):
            calls.append(text)
            self.simulation.transcript.append(Message("student", text))
            if len(calls) == 1:
                raise RuntimeError("Model unavailable")
            self.simulation.transcript.append(Message("patient", "Reply"))
            return "Reply"

        self.simulation.respond = respond
        self.queue_audio()
        with patch("sim.speech_to_text.transcribe_audio", return_value="Hello") as transcribe:
            self.app.run()
            self.assertFalse(self.app.exception)
            self.assertEqual(len(self.simulation.transcript), 1)
            self.voice.assert_not_called()
            self.app.button[1].click().run()
            self.assertFalse(self.app.exception)
            self.assertEqual(len(self.simulation.transcript), 3)
            transcribe.assert_called_once()

    def test_end_saves_and_unmounts_microphone(self):
        self.app.run()
        self.voice.reset_mock()
        with patch("sim.storage.save_result", return_value=Path("saved.json")) as save, patch(
            "sim.scenarios.list_scenarios", return_value=[self.simulation.scenario]
        ):
            self.app.button[0].click().run()
        self.assertFalse(self.app.exception)
        save.assert_called_once_with(self.simulation.scenario, self.simulation.transcript)
        self.assertIsNone(self.app.session_state["simulation"])
        self.voice.assert_not_called()

    @unittest.skipUnless(shutil.which("node"), "Node is needed for browser API mocks")
    def test_browser_pause_and_cleanup(self):
        with tempfile.TemporaryDirectory() as folder:
            module = Path(folder) / "voice.mjs"
            module.write_text(VOICE_TURN_JS)
            result = subprocess.run(
                ["node", str(Path(__file__).with_name("voice_turn_harness.mjs")), str(module)],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
