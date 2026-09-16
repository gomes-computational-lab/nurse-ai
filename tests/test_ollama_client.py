from __future__ import annotations

import io
import json
import os
import sys
import unittest
import urllib.error
from contextlib import redirect_stdout
from unittest.mock import patch

from sim import app
from sim.models import Scenario
from sim.ollama_client import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    OllamaAPIError,
    OllamaClient,
    OllamaConnectionError,
    OllamaModelUnavailableError,
    OllamaResponseError,
    configured_base_url,
    configured_model,
)


class FakeResponse:
    def __init__(self, body: str = "", lines: list[str] | None = None):
        self.body = body.encode("utf-8")
        self.lines = [line.encode("utf-8") for line in (lines or [])]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body

    def __iter__(self):
        return iter(self.lines)


class OllamaClientTests(unittest.TestCase):
    def test_environment_configuration_and_existing_defaults(self) -> None:
        self.assertEqual(configured_base_url({}), DEFAULT_OLLAMA_BASE_URL)
        self.assertEqual(configured_model({}), DEFAULT_OLLAMA_MODEL)
        self.assertEqual(
            configured_base_url({"OLLAMA_BASE_URL": " http://127.0.0.1:11435/ "}),
            "http://127.0.0.1:11435/",
        )
        self.assertEqual(configured_model({"OLLAMA_MODEL": " qwen3:4b "}), "qwen3:4b")

    def test_health_check_verifies_version_tags_and_latest_alias(self) -> None:
        responses = [
            FakeResponse('{"version":"0.12.0"}'),
            FakeResponse('{"models":[{"name":"llama3.1:latest"},{"model":"qwen3:4b"}]}'),
        ]
        requests: list[tuple[str, int]] = []

        def fake_urlopen(request, timeout):
            requests.append((request.full_url, timeout))
            return responses.pop(0)

        client = OllamaClient("llama3.1", host="http://127.0.0.1:11435/")
        with patch("sim.ollama_client.urllib.request.urlopen", side_effect=fake_urlopen):
            result = client.health_check()

        self.assertEqual(result["version"], "0.12.0")
        self.assertEqual(result["model"], "llama3.1")
        self.assertEqual(result["base_url"], "http://127.0.0.1:11435")
        self.assertEqual(
            [url for url, _timeout in requests],
            ["http://127.0.0.1:11435/api/version", "http://127.0.0.1:11435/api/tags"],
        )

    def test_health_check_reports_unavailable_model(self) -> None:
        responses = [
            FakeResponse('{"version":"0.12.0"}'),
            FakeResponse('{"models":[{"name":"qwen3:4b"}]}'),
        ]
        client = OllamaClient("qwen3:14b", host="http://127.0.0.1:11435")

        with (
            patch("sim.ollama_client.urllib.request.urlopen", side_effect=responses),
            self.assertRaises(OllamaModelUnavailableError) as raised,
        ):
            client.health_check()

        self.assertIn("qwen3:14b", str(raised.exception))
        self.assertIn("qwen3:4b", str(raised.exception))
        self.assertIn("ollama pull qwen3:14b", str(raised.exception))

    def test_connection_error_mentions_endpoint_and_tunnel(self) -> None:
        client = OllamaClient("qwen3:4b", host="http://127.0.0.1:11435")
        unavailable = urllib.error.URLError(ConnectionRefusedError("connection refused"))

        with (
            patch("sim.ollama_client.urllib.request.urlopen", side_effect=unavailable),
            self.assertRaises(OllamaConnectionError) as raised,
        ):
            client.health_check()

        message = str(raised.exception)
        self.assertIn("http://127.0.0.1:11435", message)
        self.assertIn("SSH tunnel", message)

    def test_http_error_includes_status_and_api_detail(self) -> None:
        error = urllib.error.HTTPError(
            "http://127.0.0.1:11435/api/version",
            503,
            "Service Unavailable",
            {},
            io.BytesIO(b'{"error":"GPU service unavailable"}'),
        )
        client = OllamaClient("qwen3:4b", host="http://127.0.0.1:11435")

        with (
            patch("sim.ollama_client.urllib.request.urlopen", side_effect=error),
            self.assertRaises(OllamaAPIError) as raised,
        ):
            client.health_check()

        self.assertIn("HTTP 503", str(raised.exception))
        self.assertIn("GPU service unavailable", str(raised.exception))

    def test_malformed_health_and_chat_responses_are_clear(self) -> None:
        client = OllamaClient("qwen3:4b")
        with (
            patch("sim.ollama_client.urllib.request.urlopen", return_value=FakeResponse("not json")),
            self.assertRaises(OllamaResponseError) as health_error,
        ):
            client.health_check()
        self.assertIn("malformed JSON", str(health_error.exception))

        with (
            patch("sim.ollama_client.urllib.request.urlopen", return_value=FakeResponse('{"message":{}}')),
            self.assertRaises(OllamaResponseError) as chat_error,
        ):
            client.chat([{"role": "user", "content": "Hello"}])
        self.assertIn("malformed response", str(chat_error.exception))

    def test_streaming_chat_is_preserved(self) -> None:
        response = FakeResponse(lines=[
            '{"message":{"content":"Hello"}}\n',
            '{"message":{"content":" there"}}\n',
            '{"message":{"content":"."},"done":true}\n',
        ])
        chunks: list[str] = []
        client = OllamaClient("qwen3:4b")

        with patch("sim.ollama_client.urllib.request.urlopen", return_value=response):
            result = client.chat(
                [{"role": "user", "content": "Hello"}],
                on_chunk=chunks.append,
            )

        self.assertEqual(result, "Hello there.")
        self.assertEqual(chunks, ["Hello", " there", "."])

    def test_check_llm_cli_options_override_environment(self) -> None:
        class FakeClient:
            def health_check(self):
                return {
                    "base_url": "http://cli-host:11434",
                    "version": "test-version",
                    "model": "cli-model",
                    "available_models": ["cli-model"],
                }

            def warmup(self):
                return "OK"

        output = io.StringIO()
        with (
            patch.dict(
                os.environ,
                {"OLLAMA_BASE_URL": "http://env-host:11434", "OLLAMA_MODEL": "env-model"},
            ),
            patch.object(
                sys,
                "argv",
                ["main.py", "--check-llm", "--host", "http://cli-host:11434", "--model", "cli-model"],
            ),
            patch("sim.app.OllamaClient", return_value=FakeClient()) as constructor,
            redirect_stdout(output),
        ):
            app.main()

        constructor.assert_called_once_with(model="cli-model", host="http://cli-host:11434")
        self.assertIn("Ollama check passed", output.getvalue())
        self.assertIn("Minimal inference:", output.getvalue())

    def test_text_app_uses_environment_and_validates_once_at_startup(self) -> None:
        configured_scenario = Scenario(
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

        class FakeClient:
            health_checks = 0

            def health_check(self):
                self.health_checks += 1

        class FakeSession:
            transcript = []
            selected_learner_roles = ("nurse_primary",)
            active_learner_role = "nurse_primary"

            def opening(self, on_chunk):
                on_chunk("Hello.")
                return "Hello."

        client = FakeClient()
        with (
            patch.dict(
                os.environ,
                {"OLLAMA_BASE_URL": "http://env-host:11435", "OLLAMA_MODEL": "env-model"},
            ),
            patch.object(sys, "argv", ["main.py", "--scenario", "test"]),
            patch("sim.app.OllamaClient", return_value=client) as constructor,
            patch("sim.app.load_scenario", return_value=configured_scenario),
            patch("sim.app.SimulationSession", return_value=FakeSession()),
            patch("sim.app.save_result", return_value="transcript.json"),
            patch("builtins.input", return_value="/quit"),
            redirect_stdout(io.StringIO()),
        ):
            app.main()

        constructor.assert_called_once_with(model="env-model", host="http://env-host:11435")
        self.assertEqual(client.health_checks, 1)

    def test_voice_preload_validates_before_warmup(self) -> None:
        from sim.voice_app import _await_ollama_preload, _start_ollama_preload

        class FakeClient:
            calls: list[str] = []

            def health_check(self):
                self.calls.append("health")

            def warmup(self):
                self.calls.append("warmup")

        client = FakeClient()
        preload = _start_ollama_preload(client)
        _await_ollama_preload(preload)

        self.assertEqual(client.calls, ["health", "warmup"])


if __name__ == "__main__":
    unittest.main()
