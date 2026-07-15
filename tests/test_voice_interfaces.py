from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import numpy as np

from sim.audio import RecordedAudio, VoiceActivityDetectionUnavailable
from sim.models import Scenario
from sim.ollama_client import OllamaClient
from sim.session import SimulationSession, VOICE_MAX_TOKENS
from sim.speech_to_text import transcribe_audio
from sim.text_to_speech import SpeechMetrics


class FakeWhisperModel:
    def __init__(self):
        self.audio = None
        self.options = None

    def transcribe(self, audio, **options):
        self.audio = audio
        self.options = options
        segments = [type("Segment", (), {"text": "  hello  "})()]
        return segments, object()


class FakeClient:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        callback = kwargs.get("on_chunk")
        if callback is not None:
            callback("Hello.")
        return "Hello."


def scenario() -> Scenario:
    return Scenario(
        id="test",
        title="Test",
        role="Patient",
        setting="Room",
        patient_profile={"name": "Pat"},
        clinical_context={"symptom": "pain"},
        behavior_guidelines=["Answer naturally."],
        opening_prompt="Call the nurse.",
        learning_objectives=["Communicate."],
        evaluation_rubric={"Communication": "Clear"},
    )


class VoiceInterfaceTests(unittest.TestCase):
    def test_transcription_accepts_samples_and_enables_residual_vad(self) -> None:
        model = FakeWhisperModel()
        samples = np.zeros(1600, dtype=np.float32)

        with patch("sim.speech_to_text._load_model", return_value=model):
            result = transcribe_audio(samples)

        self.assertEqual(result, "hello")
        self.assertIs(model.audio, samples)
        self.assertTrue(model.options["vad_filter"])
        self.assertEqual(model.options["beam_size"], 1)
        self.assertEqual(model.options["vad_parameters"]["speech_pad_ms"], 200)

    def test_voice_mode_adds_spoken_prompt_and_token_limit(self) -> None:
        client = FakeClient()
        session = SimulationSession(scenario(), client)

        session.opening(response_mode="voice")

        call = client.calls[0]
        self.assertEqual(call["max_tokens"], VOICE_MAX_TOKENS)
        system_prompt = call["messages"][0]["content"]
        self.assertIn("one to three short", system_prompt)
        self.assertIn("Do not use Markdown", system_prompt)

    def test_text_mode_keeps_unlimited_generation_and_original_prompt(self) -> None:
        client = FakeClient()
        session = SimulationSession(scenario(), client)

        session.opening()

        call = client.calls[0]
        self.assertIsNone(call["max_tokens"])
        self.assertNotIn("Voice response style", call["messages"][0]["content"])

    def test_ollama_token_limit_maps_to_num_predict(self) -> None:
        captured: dict[str, object] = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"message":{"content":"ok"}}'

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return Response()

        client = OllamaClient("model")
        with patch("sim.ollama_client.urllib.request.urlopen", side_effect=fake_urlopen):
            result = client.chat([{"role": "user", "content": "Hi"}], max_tokens=80)

        self.assertEqual(result, "ok")
        self.assertEqual(captured["payload"]["options"]["num_predict"], 80)

    def test_vad_unavailable_falls_back_to_manual_recording(self) -> None:
        from sim.voice_app import _record_voice_turn

        expected = RecordedAudio(
            samples=np.ones(160, dtype=np.float32),
            sample_rate=16000,
            captured_seconds=0.01,
            speech_seconds=0.01,
            endpoint_delay_seconds=0.0,
            stop_reason="manual",
        )
        with (
            patch(
                "sim.voice_app.record_microphone_until_silence",
                side_effect=VoiceActivityDetectionUnavailable("missing"),
            ),
            patch("sim.voice_app._record_until_enter", return_value=expected),
        ):
            audio, manual = _record_voice_turn(
                manual_recording=False,
                end_silence_ms=700,
                max_recording_seconds=60,
            )

        self.assertIs(audio, expected)
        self.assertTrue(manual)

    def test_audio_tracker_uses_speech_end_as_metric_origin(self) -> None:
        from sim.voice_app import _track_audio_completion

        class Stream:
            def wait_until_done(self):
                return True

            def metrics(self):
                return SpeechMetrics(
                    status="completed",
                    first_audio_started_at=12.0,
                    completed_at=15.0,
                    segments_started=2,
                    first_segment_submitted_at=11.0,
                )

        metrics: dict[str, object] = {}
        tracker = _track_audio_completion(
            Stream(),
            metrics,
            origin=10.0,
            metric_prefix="speech_end",
        )
        tracker.join(1.0)

        self.assertEqual(metrics["speech_end_to_first_audio_seconds"], 2.0)
        self.assertEqual(metrics["speech_end_to_first_tts_segment_seconds"], 1.0)
        self.assertEqual(metrics["first_tts_segment_to_first_audio_seconds"], 1.0)
        self.assertEqual(metrics["speech_end_to_audio_end_seconds"], 5.0)
        self.assertEqual(metrics["audio_status"], "completed")


if __name__ == "__main__":
    unittest.main()
