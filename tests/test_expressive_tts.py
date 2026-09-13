from __future__ import annotations

import json
from pathlib import Path
import threading
import time
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from sim import expressive_tts
from sim.expressive_tts import (
    LocalTTSError,
    LocalTTSUnavailable,
    SynthesizedAudio,
    TTSConfig,
    TTSService,
    _validate_zonos2_pcm_response,
    chatterbox_delivery_parameters,
    sanitize_chatterbox_cues,
    zonos2_delivery_parameters,
)
from sim.voice_delivery import DeliveryStyle


class StubProvider:
    def __init__(self, name, result=None, error=None):
        self.name = name
        self.result = result
        self.error = error
        self.calls = []

    def synthesize(self, text, style, voice, cancel_event):
        self.calls.append((text, style, voice, cancel_event))
        if self.error:
            raise self.error
        return self.result

    def capability(self):
        return None


class ExpressiveTTSTests(unittest.TestCase):
    def test_invalid_provider_is_rejected_instead_of_routing_to_edge(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported TTS provider 'typo'"):
            TTSConfig(provider="typo")  # type: ignore[arg-type]

    def test_delivery_mappings_are_bounded_and_clinically_subtle(self) -> None:
        style = DeliveryStyle(emotion="fearful", intensity=3, pace="fast")

        zonos = zonos2_delivery_parameters(style)
        chatterbox = chatterbox_delivery_parameters(style)

        self.assertTrue(zonos["emotion_enabled"])
        self.assertLessEqual(abs(zonos["emotion_valence"]), 0.5)
        self.assertLessEqual(abs(zonos["emotion_arousal"]), 0.4)
        self.assertLessEqual(zonos["emotion_strength"], 1.0)
        self.assertLessEqual(chatterbox["exaggeration"], 0.65)

    def test_chatterbox_removes_unapproved_nonverbal_cues(self) -> None:
        text = sanitize_chatterbox_cues("[laugh] I feel worse [sigh] [SCREAM].")

        self.assertEqual(text, "I feel worse [sigh] .")

    def test_chatterbox_model_is_loaded_once_per_device(self) -> None:
        class FakeCuda:
            @staticmethod
            def is_available():
                return False

        class FakeTorch:
            cuda = FakeCuda()

        class FakeModelLoader:
            calls = 0

            @classmethod
            def from_pretrained(cls, *, device, nano):
                cls.calls += 1
                return (device, nano)

        class FakeChatterboxModule:
            ChatterboxTurboTTS = FakeModelLoader

        modules = {
            "torch": FakeTorch(),
            "torchaudio": object(),
            "chatterbox.tts_turbo": FakeChatterboxModule(),
        }

        with (
            patch.dict(expressive_tts._CHATTERBOX_RUNTIMES, {}, clear=True),
            patch(
                "sim.expressive_tts.importlib.import_module",
                side_effect=modules.__getitem__,
            ),
        ):
            first = expressive_tts._get_chatterbox_runtime()
            second = expressive_tts._get_chatterbox_runtime()

        self.assertIs(first, second)
        self.assertEqual(FakeModelLoader.calls, 1)

    def test_chatterbox_serializes_inference_across_provider_instances(self) -> None:
        active = 0
        maximum_active = 0
        activity_lock = threading.Lock()

        class FakeAudio:
            def cpu(self):
                return self

        class FakeModel:
            sr = 24000

            def generate(self, text, **params):
                nonlocal active, maximum_active
                del text, params
                with activity_lock:
                    active += 1
                    maximum_active = max(maximum_active, active)
                time.sleep(0.02)
                with activity_lock:
                    active -= 1
                return FakeAudio()

        class FakeTorchaudio:
            @staticmethod
            def save(path, audio, sample_rate):
                del audio, sample_rate
                Path(path).write_bytes(b"wav")

        runtime = expressive_tts._ChatterboxRuntime(
            model=FakeModel(),
            torchaudio=FakeTorchaudio(),
            inference_lock=threading.Lock(),
        )
        failures = []
        start = threading.Barrier(3)

        with TemporaryDirectory() as catalog_dir:
            Path(catalog_dir, "patient-a.wav").touch()
            config = TTSConfig(
                provider="chatterbox_nano",
                voice="patient-a",
                voice_catalog_dir=catalog_dir,
            )
            providers = [
                expressive_tts._ChatterboxNanoProvider(config),
                expressive_tts._ChatterboxNanoProvider(config),
            ]

            def synthesize(provider):
                try:
                    start.wait()
                    provider.synthesize(
                        "Please help me.",
                        DeliveryStyle(),
                        "patient-a",
                        threading.Event(),
                    )
                except Exception as exc:  # pragma: no cover - surfaced below
                    failures.append(exc)

            with patch(
                "sim.expressive_tts._get_chatterbox_runtime",
                return_value=runtime,
            ):
                workers = [
                    threading.Thread(target=synthesize, args=(provider,))
                    for provider in providers
                ]
                for worker in workers:
                    worker.start()
                start.wait()
                for worker in workers:
                    worker.join()

        self.assertEqual(failures, [])
        self.assertEqual(maximum_active, 1)

    def test_local_fallback_order_uses_chatterbox_after_zonos(self) -> None:
        service = TTSService(TTSConfig(provider="zonos2", voice="patient-a"))
        zonos = StubProvider("zonos2", error=LocalTTSUnavailable("offline"))
        nano = StubProvider(
            "chatterbox_nano",
            result=SynthesizedAudio(b"wav", "audio/wav", "chatterbox_nano"),
        )
        service._providers = {"zonos2": zonos, "chatterbox_nano": nano}

        with patch("sim.expressive_tts.urllib.request.urlopen") as urlopen:
            result = service.synthesize("Help me.", DeliveryStyle(emotion="in_pain"))

        self.assertEqual(result.provider, "chatterbox_nano")
        self.assertEqual(result.fallback_from, "zonos2")
        self.assertEqual(len(zonos.calls), 1)
        self.assertEqual(len(nano.calls), 1)
        urlopen.assert_not_called()

    def test_edge_fallback_is_disabled_by_default(self) -> None:
        service = TTSService(TTSConfig(provider="chatterbox_nano"))
        nano = StubProvider("chatterbox_nano", error=LocalTTSUnavailable("missing"))
        edge = StubProvider(
            "edge", result=SynthesizedAudio(b"mp3", "audio/mpeg", "edge")
        )
        service._providers = {"chatterbox_nano": nano, "edge": edge}

        with self.assertRaises(LocalTTSUnavailable):
            service.synthesize("Hello")

        self.assertEqual(edge.calls, [])

    def test_edge_fallback_requires_explicit_opt_in(self) -> None:
        service = TTSService(
            TTSConfig(provider="chatterbox_nano", allow_online_edge_fallback=True)
        )
        service._providers = {
            "chatterbox_nano": StubProvider(
                "chatterbox_nano", error=LocalTTSUnavailable("missing")
            ),
            "edge": StubProvider(
                "edge", result=SynthesizedAudio(b"mp3", "audio/mpeg", "edge")
            ),
        }

        result = service.synthesize("Hello")

        self.assertEqual(result.provider, "edge")
        self.assertEqual(result.fallback_from, "chatterbox_nano")

    def test_zonos_request_uses_full_expressive_contract(self) -> None:
        captured = []

        class Headers:
            def __init__(self, values=None):
                self.values = values or {}

            def get(self, name, default=None):
                return self.values.get(name, default)

        class Response:
            def __init__(self, body, headers=None):
                self.body = body
                self.headers = Headers(headers)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return self.body

        def urlopen(request, timeout):
            captured.append(
                {
                    "url": request.full_url,
                    "method": request.get_method(),
                    "payload": json.loads(request.data) if request.data else None,
                    "timeout": timeout,
                }
            )
            if request.full_url.endswith("/tts/speakers"):
                return Response(
                    json.dumps(
                        {
                            "speakers": [
                                {
                                    "id": "default_patient_a",
                                    "label": "Patient A",
                                    "scope": "default",
                                    "is_default": True,
                                    "original_name": "patient-a.wav",
                                }
                            ]
                        }
                    ).encode()
                )
            return Response(
                b"\x00\x00\x00\x00",
                {
                    "Content-Type": "audio/pcm",
                    "X-Audio-Sample-Rate": "44100",
                    "X-Audio-Channels": "1",
                    "X-Audio-Format": "float32",
                },
            )

        service = TTSService(
            TTSConfig(
                provider="zonos2", voice="patient-a", zonos2_url="http://127.0.0.1:1919"
            )
        )
        with patch("sim.expressive_tts.urllib.request.urlopen", side_effect=urlopen):
            result = service.synthesize(
                "I feel worried.",
                DeliveryStyle(emotion="anxious", intensity=2, pace="slow"),
            )

        self.assertEqual(
            [request["url"] for request in captured],
            [
                "http://127.0.0.1:1919/tts/speakers",
                "http://127.0.0.1:1919/tts/generate",
            ],
        )
        payload = captured[1]["payload"]
        self.assertEqual(payload["text"], "I feel worried.")
        self.assertEqual(payload["speaker_embedding_id"], "default_patient_a")
        self.assertEqual(payload["speed"], 0.92)
        self.assertFalse(payload["stream"])
        self.assertNotIn("input", payload)
        self.assertNotIn("voice", payload)
        self.assertNotIn("response_format", payload)
        self.assertEqual(result.data[:4], b"RIFF")

    def test_zonos_rejects_audio_that_is_not_documented_float32_pcm(self) -> None:
        headers = {
            "content_type": "audio/wav",
            "sample_rate": "44100",
            "channels": "1",
            "format": "float32",
        }

        with self.assertRaisesRegex(LocalTTSError, "unexpected content type"):
            _validate_zonos2_pcm_response(b"RIFF", headers)

    def test_cancelled_request_never_reaches_a_provider(self) -> None:
        service = TTSService(TTSConfig(provider="chatterbox_nano"))
        provider = StubProvider(
            "chatterbox_nano",
            result=SynthesizedAudio(b"wav", "audio/wav", "chatterbox_nano"),
        )
        service._providers = {"chatterbox_nano": provider}
        cancelled = threading.Event()
        cancelled.set()

        with self.assertRaisesRegex(Exception, "cancelled"):
            service.synthesize("Hello", cancel_event=cancelled)
        self.assertEqual(provider.calls, [])


if __name__ == "__main__":
    unittest.main()
