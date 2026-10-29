from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from sim.evaluator import CRITERIA, evaluate_transcript
from sim.models import Message
from sim.ollama_client import OllamaClient, configured_base_url, configured_model
from sim.scenarios import load_scenario


@unittest.skipUnless(
    os.environ.get("RUN_OLLAMA_INTEGRATION") == "1",
    "Set RUN_OLLAMA_INTEGRATION=1 to test the configured Ollama service.",
)
class OllamaIntegrationTests(unittest.TestCase):
    def test_configured_service_health_and_inference(self) -> None:
        client = OllamaClient(
            model=configured_model(),
            host=configured_base_url(),
        )

        health = client.health_check()
        response = client.warmup()

        self.assertEqual(health["model"], configured_model())
        self.assertTrue(response.strip())

    def test_configured_service_one_call_evaluator(self) -> None:
        client = OllamaClient(
            model=configured_model(),
            host=configured_base_url(),
        )
        scenario = load_scenario("post_op_pain")
        transcript = [
            Message(role="patient", content="My incision pain is 8 out of 10."),
            Message(
                role="student",
                content=(
                    "I am sorry you are hurting. I will assess the pain and incision, check your "
                    "vital signs and medication orders, help you reposition safely, and contact "
                    "your nurse or provider if the pain or your condition worsens."
                ),
                speaker_role="nurse_primary",
            ),
        ]

        client.health_check()
        with patch.object(client, "chat", wraps=client.chat) as chat:
            result = evaluate_transcript(scenario, transcript, client)

        self.assertEqual(chat.call_count, 1)
        self.assertIsNotNone(result["overall_score"], result["improvements"])
        self.assertEqual(list(result["criteria"]), [item["name"] for item in CRITERIA])


if __name__ == "__main__":
    unittest.main()
