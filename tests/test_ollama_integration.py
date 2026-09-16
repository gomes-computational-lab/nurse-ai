from __future__ import annotations

import os
import unittest

from sim.ollama_client import OllamaClient, configured_base_url, configured_model


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
        response = client.chat(
            [{"role": "user", "content": "Reply with exactly: OK"}],
            temperature=0.0,
            max_tokens=8,
        )

        self.assertEqual(health["model"], configured_model())
        self.assertTrue(response.strip())


if __name__ == "__main__":
    unittest.main()
