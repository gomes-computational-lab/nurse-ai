from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from sim import app
from sim.models import Message
from sim.scenarios import load_scenario
from sim.session import SimulationSession
from sim.storage import save_result
from sim.terminal_ui import choose_learner_count


class FakeClient:
    def __init__(self):
        self.calls: list[list[dict[str, str]]] = []
        self.health_checks = 0

    def health_check(self):
        self.health_checks += 1

    def chat(self, messages, **options):
        self.calls.append(messages)
        callback = options.get("on_chunk")
        if callback is not None:
            callback("Hello.")
        return "Hello."


class SimulationModelTests(unittest.TestCase):
    def test_existing_scenario_json_uses_backward_compatible_defaults(self) -> None:
        scenario = load_scenario("post_op_pain")

        self.assertEqual(scenario.participants, ())
        self.assertEqual(scenario.learner_configuration.min_nurses, 1)
        self.assertEqual(scenario.learner_configuration.max_nurses, 1)
        self.assertEqual(scenario.learner_configuration.supported_roles, ("nurse_primary",))
        self.assertIsNone(scenario.simulation_timeline)

    def test_ruth_defines_human_ai_learner_and_facilitator_participants(self) -> None:
        scenario = load_scenario("ruth_family_member")
        participants = {participant.id: participant for participant in scenario.participants}

        self.assertEqual(participants["patient_ruth_lawson"].participant_type, "human")
        self.assertEqual(participants["patient_ruth_lawson"].role, "patient")
        self.assertEqual(participants["family_member_alex"].participant_type, "ai")
        self.assertEqual(participants["family_member_alex"].role, "family_member")
        self.assertEqual(participants["student_nurses"].participant_type, "learner")
        self.assertEqual(participants["simulation_facilitator"].role, "facilitator")
        self.assertNotEqual(participants["patient_ruth_lawson"].participant_type, "ai")

    def test_one_nurse_session_uses_primary_role(self) -> None:
        session = SimulationSession(load_scenario("ruth_family_member"), FakeClient(), learner_count=1)

        session.respond("How are you feeling?")

        self.assertEqual(session.selected_learner_roles, ("nurse_primary",))
        self.assertEqual(session.active_learner_role, "nurse_primary")
        self.assertEqual(session.transcript[0].speaker_role, "nurse_primary")

    def test_two_nurse_session_tracks_primary_and_secondary_speakers(self) -> None:
        client = FakeClient()
        session = SimulationSession(load_scenario("ruth_family_member"), client, learner_count=2)

        session.respond("I will begin the assessment.")
        session.select_learner_role("nurse_secondary")
        session.respond("I will review the safety plan.")

        self.assertEqual(
            session.selected_learner_roles,
            ("nurse_primary", "nurse_secondary"),
        )
        student_messages = [message for message in session.transcript if message.role == "student"]
        self.assertEqual(
            [message.speaker_role for message in student_messages],
            ["nurse_primary", "nurse_secondary"],
        )
        self.assertEqual(client.calls[-1][-2]["content"], "Nurse Secondary: I will review the safety plan.")

    def test_transcript_serialization_includes_optional_speaker_role(self) -> None:
        scenario = load_scenario("ruth_family_member")
        transcript = [
            Message(role="family_member", content="I am worried."),
            Message(role="student", content="I hear your concern.", speaker_role="nurse_secondary"),
        ]

        with tempfile.TemporaryDirectory() as directory:
            with patch("sim.storage.TRANSCRIPT_DIR", Path(directory)):
                path = save_result(
                    scenario,
                    transcript,
                    learner_roles=("nurse_primary", "nurse_secondary"),
                )
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertNotIn("speaker_role", payload["transcript"][0])
        self.assertEqual(payload["transcript"][1]["speaker_role"], "nurse_secondary")
        self.assertEqual(payload["metadata"]["learner_configuration"], {
            "nurse_count": 2,
            "roles": ["nurse_primary", "nurse_secondary"],
        })

    def test_simulation_timeline_is_authoritative_in_prompt_and_storage(self) -> None:
        scenario = load_scenario("ruth_family_member")
        prompt = SimulationSession(scenario, FakeClient())._system_prompt()

        self.assertIn("Simulation date (the fictional 'today'): 2025-01-15", prompt)
        self.assertIn("Clinical day: 7", prompt)
        self.assertIn("Time of day: morning", prompt)
        self.assertIn("Never substitute the host computer's current date", prompt)

        with tempfile.TemporaryDirectory() as directory:
            with patch("sim.storage.TRANSCRIPT_DIR", Path(directory)):
                path = save_result(scenario, [])
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["metadata"]["simulation_timeline"], {
            "simulation_date": "2025-01-15",
            "clinical_day": 7,
            "time_of_day": "morning",
        })

    def test_learner_count_prompt_offers_one_or_two_nurses(self) -> None:
        scenario = load_scenario("ruth_family_member")
        output = io.StringIO()

        with patch("builtins.input", return_value="2"), redirect_stdout(output):
            selected = choose_learner_count(scenario)

        self.assertEqual(selected, 2)
        self.assertIn("1. One", output.getvalue())
        self.assertIn("2. Two", output.getvalue())

    def test_text_cli_switches_to_secondary_nurse(self) -> None:
        scenario = load_scenario("ruth_family_member")
        client = FakeClient()

        with (
            patch.object(sys, "argv", ["main.py", "--scenario", "ruth_family_member"]),
            patch("sim.app.load_scenario", return_value=scenario),
            patch("sim.app.OllamaClient", return_value=client),
            patch("sim.app.save_result", return_value="transcript.json") as save_result_mock,
            patch("builtins.input", side_effect=["2", "/secondary", "Hello", "/quit"]),
            redirect_stdout(io.StringIO()),
        ):
            app.main()

        saved_transcript = save_result_mock.call_args.args[1]
        student_messages = [message for message in saved_transcript if message.role == "student"]
        self.assertEqual(student_messages[0].speaker_role, "nurse_secondary")
        self.assertEqual(
            save_result_mock.call_args.kwargs["learner_roles"],
            ("nurse_primary", "nurse_secondary"),
        )


if __name__ == "__main__":
    unittest.main()
