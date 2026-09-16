from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from sim.models import Scenario
from sim.scenarios import load_scenario, select_scenario_phase
from sim.session import SimulationSession
from sim.storage import save_result
from sim.terminal_ui import print_scenario_header


class FakeClient:
    def chat(self, messages, **options):
        del messages, options
        return "Hello."


class ScenarioRoleTests(unittest.TestCase):
    def test_existing_patient_scenario_uses_compatible_default_role(self) -> None:
        scenario = load_scenario("post_op_pain")

        self.assertEqual(scenario.agent_role, "patient")
        self.assertIsNone(scenario.scenario_phase)

    def test_ruth_family_member_and_both_phases_load(self) -> None:
        phase_one = load_scenario("ruth_family_member")
        phase_two = load_scenario("ruth_family_member", phase=2)

        self.assertEqual(phase_one.agent_role, "family_member")
        self.assertEqual(phase_one.relationship, "grandchild")
        self.assertEqual(phase_one.scenario_phase, 1)
        self.assertEqual(phase_two.scenario_phase, 2)
        self.assertEqual(phase_one.emotional_state["anxiety"], 0.7)
        self.assertEqual(phase_two.emotional_state["fatigue"], 0.85)
        self.assertIn("Foley catheter", " ".join(phase_one.phase_behavior_guidelines))
        self.assertIn("nothing seems to be going right", " ".join(phase_two.phase_behavior_guidelines))

    def test_undefined_phase_is_rejected(self) -> None:
        scenario = load_scenario("ruth_family_member")

        with self.assertRaisesRegex(ValueError, "does not define phase 3"):
            select_scenario_phase(scenario, 3)

    def test_family_prompt_identifies_role_phase_and_internal_emotion(self) -> None:
        scenario = load_scenario("ruth_family_member", phase=2)
        session = SimulationSession(scenario, FakeClient())

        prompt = session._system_prompt()

        self.assertIn("AI agent role: family member", prompt)
        self.assertIn("Character name: Alex", prompt)
        self.assertIn("Relationship to patient: grandchild", prompt)
        self.assertIn("Current simulation phase: 2", prompt)
        self.assertIn("'fatigue': 0.85", prompt)
        self.assertIn("never speak these numeric values", prompt)
        self.assertIn("Do not behave like ChatGPT, a clinician, or an instructor", prompt)

    def test_family_response_is_logged_with_family_member_role(self) -> None:
        scenario = load_scenario("ruth_family_member")
        session = SimulationSession(scenario, FakeClient())

        session.opening()

        self.assertEqual(session.transcript[0].role, "family_member")

    def test_family_header_displays_role_character_and_phase(self) -> None:
        scenario = load_scenario("ruth_family_member", phase=2)
        output = io.StringIO()

        with redirect_stdout(output):
            print_scenario_header(scenario)

        rendered = output.getvalue()
        self.assertIn("AI Role: Family Member", rendered)
        self.assertIn("Character: Alex — Ruth Lawson's grandchild", rendered)
        self.assertIn("Scenario Phase: 2", rendered)

    def test_saved_result_includes_role_scenario_and_phase_metadata(self) -> None:
        scenario = load_scenario("ruth_family_member", phase=2)
        with tempfile.TemporaryDirectory() as directory:
            with patch("sim.storage.TRANSCRIPT_DIR", Path(directory)):
                path = save_result(scenario, [])
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual({key: payload["metadata"][key] for key in (
            "agent_role", "scenario_id", "scenario_phase"
        )}, {
            "agent_role": "family_member",
            "scenario_id": "ruth_family_member",
            "scenario_phase": 2,
        })
        self.assertEqual(payload["metadata"]["learner_configuration"], {
            "nurse_count": 1,
            "roles": ["nurse_primary"],
        })
        self.assertEqual(payload["metadata"]["simulation_timeline"]["clinical_day"], 7)
        self.assertIn("execution_saved_at", payload["metadata"])

    def test_family_member_is_an_accepted_configured_role(self) -> None:
        configured = Scenario(
            id="family",
            title="Family",
            role="Family Member",
            setting="Room",
            patient_profile={},
            clinical_context={},
            behavior_guidelines=[],
            opening_prompt="Begin.",
            learning_objectives=[],
            evaluation_rubric={},
            agent_role="family_member",
        )

        self.assertEqual(configured.agent_role, "family_member")


if __name__ == "__main__":
    unittest.main()
