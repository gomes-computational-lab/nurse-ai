from __future__ import annotations

import json
import threading
import unittest
from unittest.mock import patch

from sim.expressive_tts import (
    LocalTTSUnavailable,
    SynthesizedAudio,
    TTSConfig,
    TTSService,
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

    def test_zonos_request_includes_voice_and_delivery_controls(self) -> None:
        captured = {}

        class Headers:
            @staticmethod
            def get(name, default=None):
                return "44100" if name == "X-Audio-Sample-Rate" else default

        class Response:
            headers = Headers()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b"\x00\x00\x00\x00"

        def urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["payload"] = json.loads(request.data)
            captured["timeout"] = timeout
            return Response()

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

        self.assertEqual(captured["url"], "http://127.0.0.1:1919/v1/audio/speech")
        self.assertEqual(captured["payload"]["voice"], "patient-a")
        self.assertEqual(captured["payload"]["speed"], 0.92)
        self.assertEqual(result.data[:4], b"RIFF")

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
