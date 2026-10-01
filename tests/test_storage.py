from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sim.models import Message, Scenario
from sim.storage import save_result
from sim.voice_delivery import DeliveryStyle


class StorageTests(unittest.TestCase):
    def test_delivery_metadata_is_serialized_with_patient_message(self) -> None:
        scenario = Scenario(
            id="test",
            title="Test",
            role="Patient",
            setting="Room",
            patient_profile={},
            clinical_context={},
            behavior_guidelines=[],
            opening_prompt="Begin.",
            learning_objectives=[],
            evaluation_rubric={},
        )
        message = Message(
            role="patient",
            content="Please help me.",
            delivery=DeliveryStyle(emotion="anxious", intensity=2, pace="slow"),
        )

        with tempfile.TemporaryDirectory() as directory:
            with patch("sim.storage.TRANSCRIPT_DIR", Path(directory)):
                path = save_result(scenario, [message])
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(
            payload["transcript"][0]["delivery"],
            {"emotion": "anxious", "intensity": 2, "pace": "slow"},
        )
        self.assertEqual(payload["transcript"][0]["content"], "Please help me.")


if __name__ == "__main__":
    unittest.main()
