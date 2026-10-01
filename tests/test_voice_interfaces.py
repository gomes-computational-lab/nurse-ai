from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from sim.audio import RecordedAudio, VoiceActivityDetectionUnavailable
from sim.expressive_tts import SynthesizedAudio
from sim.models import Message, Scenario
from sim.ollama_client import OllamaClient, OllamaError
from sim.session import (
    SimulationSession,
    VOICE_MAX_TOKENS,
    VOICE_RETRY_TEMPERATURE,
)
from sim.speech_to_text import transcribe_audio, transcribe_audio_bytes
from sim.text_to_speech import SpeechMetrics
from sim.voice_delivery import DEFAULT_DELIVERY


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
    def test_voice_loop_waits_for_patient_audio_before_accepting_student_input(
        self,
    ) -> None:
        from sim.voice_app import main as voice_main

        class FakeStream:
            def __init__(self):
                self.finished = False
                self.wait_calls = 0
                self.chunks = []

            def add_chunk(self, chunk):
                self.chunks.append(chunk)

            def finish(self, delivery=None):
                del delivery
                self.finished = True

            def wait_until_done(self):
                self.wait_calls += 1
                return True

        class FakeSession:
            def __init__(self):
                self.transcript = []

            def opening_prompt_char_count(self, **kwargs):
                del kwargs
                return 10

            def opening(self, on_chunk, **kwargs):
                del kwargs
                on_chunk("Opening response.")
                self.transcript.append(
                    Message(role="patient", content="Opening response.")
                )
                return "Opening response."

            def response_prompt_char_count(self, *args, **kwargs):
                del args, kwargs
                return 20

            def respond(self, student_response, on_chunk, **kwargs):
                del student_response, kwargs
                on_chunk("Patient response.")
                self.transcript.append(
                    Message(role="patient", content="Patient response.")
                )
                return "Patient response."

        streams = [FakeStream(), FakeStream()]
        recorded_audio = RecordedAudio(
            samples=np.ones(160, dtype=np.float32),
            sample_rate=16000,
            captured_seconds=0.01,
            speech_seconds=0.01,
            endpoint_delay_seconds=0.0,
            stop_reason="silence",
        )

        with (
            patch("sys.argv", ["voice_demo.py", "--scenario", "test"]),
            patch("sim.voice_app.load_scenario", return_value=scenario()),
            patch("sim.voice_app.OllamaClient", return_value=object()),
            patch("sim.voice_app.SimulationSession", return_value=FakeSession()),
            patch("sim.voice_app._start_stt_preload", return_value={}),
            patch("sim.voice_app._start_ollama_preload", return_value={}),
            patch("sim.voice_app._await_stt_preload"),
            patch("sim.voice_app._await_ollama_preload"),
            patch("sim.voice_app.create_speech_stream", side_effect=streams),
            patch("sim.voice_app._track_audio_completion", return_value=object()),
            patch("sim.voice_app._read_voice_action", side_effect=["speak", "quit"]),
            patch(
                "sim.voice_app._record_voice_turn", return_value=(recorded_audio, False)
            ),
            patch("sim.voice_app.transcribe_audio", return_value="Hello"),
            patch("sim.voice_app.stop_speaking") as stop_speaking,
            patch("sim.voice_app._finalize_audio_metrics"),
            patch("sim.voice_app.refresh_latency_summary"),
            patch("sim.voice_app.save_result", return_value="transcript.json"),
        ):
            voice_main()

        self.assertTrue(all(stream.finished for stream in streams))
        self.assertTrue(all(stream.wait_calls == 1 for stream in streams))
        self.assertEqual(streams[0].chunks, ["Opening response."])
        self.assertEqual(streams[1].chunks, ["Patient response."])
        stop_speaking.assert_called_once_with()

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

    def test_browser_audio_bytes_use_a_temporary_file_and_clean_it_up(self) -> None:
        observed_path: Path | None = None

        def fake_transcribe(path, *, model_name):
            nonlocal observed_path
            observed_path = path
            self.assertTrue(path.exists())
            self.assertEqual(path.read_bytes(), b"RIFF fake wav")
            self.assertEqual(model_name, "tiny.en")
            return "browser transcript"

        with patch("sim.speech_to_text.transcribe_audio", side_effect=fake_transcribe):
            result = transcribe_audio_bytes(BytesIO(b"RIFF fake wav"))

        self.assertEqual(result, "browser transcript")
        self.assertIsNotNone(observed_path)
        self.assertFalse(observed_path.exists())

    def test_empty_browser_audio_does_not_load_the_stt_model(self) -> None:
        with patch("sim.speech_to_text.transcribe_audio") as transcribe:
            result = transcribe_audio_bytes(b"")

        self.assertEqual(result, "")
        transcribe.assert_not_called()

    def test_browser_audio_temp_file_is_cleaned_after_transcription_failure(
        self,
    ) -> None:
        observed_path: Path | None = None

        def fail(path, *, model_name):
            nonlocal observed_path
            del model_name
            observed_path = path
            raise RuntimeError("failed")

        with patch("sim.speech_to_text.transcribe_audio", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                transcribe_audio_bytes(b"RIFF fake wav")

        self.assertIsNotNone(observed_path)
        self.assertFalse(observed_path.exists())

    def test_browser_tts_returns_provider_audio_bytes(self) -> None:
        import sim.text_to_speech as tts

        class Service:
            def synthesize(self, text, delivery):
                self.options = (text, delivery)
                return SynthesizedAudio(b"audio", "audio/wav", "zonos2")

        result = tts.synthesize_speech_bytes(
            "  Patient says hello.  ", service=Service()
        )

        self.assertEqual(result, b"audio")

    def test_browser_tts_reports_provider_failure(self) -> None:
        import sim.text_to_speech as tts

        class Service:
            def synthesize(self, text, delivery):
                del text, delivery
                raise RuntimeError("missing local model")

        with self.assertRaisesRegex(tts.TextToSpeechError, "missing local model"):
            tts.synthesize_speech_bytes("Hello", service=Service())

    def test_browser_tts_rejects_empty_audio_response(self) -> None:
        import sim.text_to_speech as tts

        class Service:
            def synthesize(self, text, delivery):
                del text, delivery
                raise RuntimeError("no audio")

        with self.assertRaisesRegex(tts.TextToSpeechError, "no audio"):
            tts.synthesize_speech_bytes("Hello", service=Service())

    def test_voice_mode_adds_spoken_prompt_and_token_limit(self) -> None:
        client = FakeClient()
        session = SimulationSession(scenario(), client)

        session.opening(response_mode="voice")

        call = client.calls[0]
        self.assertEqual(call["max_tokens"], VOICE_MAX_TOKENS)
        system_prompt = call["messages"][0]["content"]
        self.assertIn("one to three short", system_prompt)
        self.assertIn("[[delivery]]", system_prompt)
        self.assertIn("Do not use Markdown", system_prompt)
        self.assertIn("appear exactly once", system_prompt)
        self.assertIn("Pain level", system_prompt)

    def test_voice_mode_strips_delivery_header_from_stream_and_transcript(self) -> None:
        class DeliveryClient:
            def chat(self, messages, **kwargs):
                del messages
                response = (
                    '[[delivery]]{"emotion":"in_pain","intensity":2,"pace":"slow"}'
                    "[[/delivery]]\nIt hurts when I breathe."
                )
                callback = kwargs.get("on_chunk")
                callback(response[:18])
                callback(response[18:])
                return response

        visible = []
        session = SimulationSession(scenario(), DeliveryClient())

        response = session.opening(visible.append, response_mode="voice")

        self.assertEqual(response, "It hurts when I breathe.")
        self.assertEqual("".join(visible), response)
        self.assertNotIn("delivery", session.transcript[0].content)
        self.assertEqual(session.transcript[0].delivery.emotion, "in_pain")

    def test_malformed_streamed_response_retries_once_without_streaming_retry(
        self,
    ) -> None:
        class RetryClient:
            def __init__(self):
                self.calls = []

            def chat(self, messages, **kwargs):
                self.calls.append({"messages": messages, **kwargs})
                if len(self.calls) == 1:
                    response = "Pain level: 8/10\nPlease help me."
                    kwargs["on_chunk"](response)
                    return response
                return (
                    '[[delivery]]{"emotion":"in_pain","intensity":2,'
                    '"pace":"slow"}[[/delivery]]\nMy pain is 8 out of 10.'
                )

        client = RetryClient()
        visible: list[str] = []
        session = SimulationSession(scenario(), client)

        response = session.opening(visible.append, response_mode="voice")

        self.assertEqual(response, "My pain is 8 out of 10.")
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[1]["temperature"], VOICE_RETRY_TEMPERATURE)
        self.assertNotIn("on_chunk", client.calls[1])
        self.assertIn(
            "Regenerate the same patient reply",
            client.calls[1]["messages"][-1]["content"],
        )
        self.assertEqual(session.transcript[0].content, response)
        self.assertEqual(session.transcript[0].delivery.emotion, "in_pain")

    def test_invalid_delivery_header_uses_neutral_voice_without_retry(self) -> None:
        class InvalidDeliveryClient:
            def __init__(self):
                self.calls = 0

            def chat(self, messages, **kwargs):
                del messages
                self.calls += 1
                response = (
                    '[[delivery]]{"emotion":"dramatic","intensity":9,'
                    '"pace":"slow"}[[/delivery]]\nPlease stay with me.'
                )
                callback = kwargs.get("on_chunk")
                if callback is not None:
                    callback(response)
                return response

        client = InvalidDeliveryClient()
        session = SimulationSession(scenario(), client)

        response = session.opening(lambda chunk: None, response_mode="voice")

        self.assertEqual(response, "Please stay with me.")
        self.assertEqual(client.calls, 1)
        self.assertEqual(session.transcript[0].delivery, DEFAULT_DELIVERY)

    def test_two_malformed_responses_leave_turn_uncommitted(self) -> None:
        class InvalidClient:
            def __init__(self):
                self.calls = 0

            def chat(self, messages, **kwargs):
                del messages
                self.calls += 1
                response = "Emotion: anxious\nPlease help me."
                callback = kwargs.get("on_chunk")
                if callback is not None:
                    callback(response)
                return response

        client = InvalidClient()
        session = SimulationSession(scenario(), client)
        session.transcript.append(Message(role="patient", content="How can you help?"))
        before = list(session.transcript)

        with self.assertRaisesRegex(OllamaError, "prepared safely"):
            session.respond(
                "I will assess your pain.",
                lambda chunk: None,
                response_mode="voice",
            )

        self.assertEqual(client.calls, 2)
        self.assertEqual(session.transcript, before)

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
        with patch(
            "sim.ollama_client.urllib.request.urlopen", side_effect=fake_urlopen
        ):
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
