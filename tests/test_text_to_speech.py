from __future__ import annotations

import importlib
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch


class FakeBackend:
    def __init__(
        self,
        prepared_type,
        *,
        auto_finish: bool = True,
        fail_synthesis: bool = False,
        fail_load: bool = False,
    ):
        self.prepared_type = prepared_type
        self.auto_finish = auto_finish
        self.fail_synthesis = fail_synthesis
        self.fail_load = fail_load
        self.synthesized: list[str] = []
        self.paths: list[Path] = []
        self.started = threading.Event()
        self.queued = threading.Event()
        self.stop_calls = 0
        self._current = None
        self._queued = None
        self._busy = False
        self._polls = 0
        self._lock = threading.Lock()

    def synthesize(self, text: str, style, cancel_event: threading.Event):
        del style
        if self.fail_synthesis:
            raise RuntimeError("offline")
        if cancel_event.is_set():
            return None
        self.synthesized.append(text)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as file:
            file.write(text.encode("utf-8"))
            path = Path(file.name)
        self.paths.append(path)
        return self.prepared_type(path)

    def load(self, audio):
        if self.fail_load:
            raise RuntimeError("decode failed")
        return object()

    def play(self, playable: object) -> None:
        with self._lock:
            self._current = playable
            self._busy = True
            self._polls = 0
        self.started.set()

    def queue(self, playable: object) -> None:
        with self._lock:
            self._queued = playable
        self.queued.set()

    def current(self):
        with self._lock:
            return self._current

    def is_busy(self) -> bool:
        with self._lock:
            if not self._busy or not self.auto_finish:
                return self._busy
            self._polls += 1
            if self._polls >= 3:
                if self._queued is not None:
                    self._current = self._queued
                    self._queued = None
                    self._polls = 0
                else:
                    self._current = None
                    self._busy = False
            return self._busy

    def stop(self) -> None:
        with self._lock:
            self.stop_calls += 1
            self._busy = False
            self._current = None
            self._queued = None


class TextToSpeechTests(unittest.TestCase):
    def setUp(self) -> None:
        import sim.text_to_speech as tts

        self.tts = importlib.reload(tts)

    def test_speech_stream_submits_one_complete_utterance_after_finish(self) -> None:
        captured: list[tuple[int, str]] = []

        with patch.object(
            self.tts,
            "_enqueue_speech",
            side_effect=lambda session_id, text: captured.append((session_id, text)),
        ):
            stream = self.tts.SpeechStream(7, enabled=True)
            stream.add_chunk("Hello")
            self.assertEqual(captured, [])
            stream.add_chunk(" there.")
            self.assertEqual(captured, [])
            stream.add_chunk(" ")
            self.assertEqual(captured, [])
            stream.add_chunk(" How are you? Fine")
            stream.finish()

        self.assertEqual(captured, [(7, "Hello there.  How are you? Fine")])

    def test_long_unpunctuated_text_remains_one_utterance(self) -> None:
        captured: list[str] = []
        text = "word " * 50

        with patch.object(
            self.tts,
            "_enqueue_speech",
            side_effect=lambda session_id, segment: captured.append(segment),
        ):
            stream = self.tts.SpeechStream(3, enabled=True)
            stream.add_chunk(text)
            stream.finish()

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].split(), text.split())

    def test_stream_waits_for_finish_with_decimals_and_abbreviations(self) -> None:
        captured: list[str] = []

        with patch.object(
            self.tts,
            "_enqueue_speech",
            side_effect=lambda session_id, segment: captured.append(segment),
        ):
            stream = self.tts.SpeechStream(4, enabled=True)
            stream.add_chunk("The dose is 2.")
            stream.add_chunk("5 mg. Dr. ")
            self.assertEqual(captured, [])
            stream.add_chunk("Smith agrees. ")
            stream.finish()

        self.assertEqual(captured, ["The dose is 2.5 mg. Dr. Smith agrees."])

    def test_clause_is_not_emitted_before_response_finishes(self) -> None:
        captured: list[str] = []

        with patch.object(
            self.tts,
            "_enqueue_speech",
            side_effect=lambda session_id, segment: captured.append(segment),
        ):
            stream = self.tts.SpeechStream(5, enabled=True)
            stream.add_chunk("It feels like an 8 out of 10, ")

        self.assertEqual(captured, [])

    def test_unpunctuated_text_is_not_emitted_before_finish(self) -> None:
        captured: list[str] = []
        text = "I have been feeling a steady pressure across my abdomen and it keeps getting worse"

        with patch.object(
            self.tts,
            "_enqueue_speech",
            side_effect=lambda session_id, segment: captured.append(segment),
        ):
            stream = self.tts.SpeechStream(6, enabled=True)
            stream.add_chunk(text)

        self.assertEqual(captured, [])

    def test_normalization_preserves_prosody_punctuation(self) -> None:
        text = "Wait, please; I... need help: now."
        self.assertEqual(self.tts._normalize_speech_text(text), text)

    def test_create_speech_stream_falls_back_when_backend_is_unavailable(self) -> None:
        with patch.object(self.tts, "_create_tts_backend", return_value=None):
            stream = self.tts.create_speech_stream()

        self.assertFalse(stream._enabled)
        self.assertEqual(stream.metrics().status, "failed")

    def test_streamed_sentence_starts_only_after_response_finishes(self) -> None:
        backend = FakeBackend(self.tts._PreparedAudio, auto_finish=False)
        self.tts._backend = backend

        stream = self.tts.create_speech_stream()
        stream.add_chunk("Patient says hello. ")

        self.assertFalse(backend.started.wait(0.05))

        stream.add_chunk(" Another response.")
        stream.finish()
        self.assertTrue(backend.started.wait(1.0))
        self.assertEqual(backend.synthesized, ["Patient says hello. Another response."])
        self.tts.stop_speaking()
        self.assertTrue(stream.wait_until_done(1.0))
        self.assertEqual(stream.metrics().status, "interrupted")

    def test_first_clause_waits_for_complete_response(self) -> None:
        backend = FakeBackend(self.tts._PreparedAudio, auto_finish=False)
        self.tts._backend = backend

        stream = self.tts.create_speech_stream()
        stream.add_chunk("It feels like an 8 out of 10, ")

        self.assertFalse(backend.started.wait(0.05))
        self.assertFalse(stream._finished)

        stream.add_chunk("especially when I take a deep breath.")
        stream.finish()
        self.assertTrue(backend.started.wait(1.0))
        self.assertEqual(
            backend.synthesized,
            ["It feels like an 8 out of 10, especially when I take a deep breath."],
        )
        self.tts.stop_speaking()
        self.assertTrue(stream.wait_until_done(1.0))

    def test_complete_response_uses_one_synthesis_request(self) -> None:
        backend = FakeBackend(self.tts._PreparedAudio)
        self.tts._backend = backend

        stream = self.tts.create_speech_stream()
        stream.add_chunk("First sentence. Second sentence.")
        stream.finish()

        self.assertTrue(stream.wait_until_done(2.0))
        self.assertEqual(backend.synthesized, ["First sentence. Second sentence."])
        self.assertEqual(stream.metrics().status, "completed")
        self.assertEqual(stream.metrics().segments_started, 1)
        self.assertTrue(all(not path.exists() for path in backend.paths))

    def test_stop_speaking_cancels_playback_and_cleans_queues(self) -> None:
        backend = FakeBackend(self.tts._PreparedAudio, auto_finish=False)
        self.tts._backend = backend

        stream = self.tts.create_speech_stream()
        stream.add_chunk("First sentence. Second sentence.")
        stream.finish()
        self.assertTrue(backend.started.wait(1.0))

        self.tts.stop_speaking()

        self.assertTrue(stream.wait_until_done(1.0))
        self.assertEqual(stream.metrics().status, "interrupted")
        self.assertGreaterEqual(backend.stop_calls, 1)
        deadline = time.time() + 1.0
        while any(path.exists() for path in backend.paths) and time.time() < deadline:
            time.sleep(0.01)
        self.assertTrue(all(not path.exists() for path in backend.paths))

    def test_backend_failure_marks_stream_failed(self) -> None:
        backend = FakeBackend(self.tts._PreparedAudio, fail_synthesis=True)
        self.tts._backend = backend

        stream = self.tts.create_speech_stream()
        stream.add_chunk("This will fail.")
        stream.finish()

        self.assertTrue(stream.wait_until_done(1.0))
        self.assertEqual(stream.metrics().status, "failed")

    def test_playback_decode_failure_marks_stream_failed(self) -> None:
        backend = FakeBackend(self.tts._PreparedAudio, fail_load=True)
        self.tts._backend = backend

        stream = self.tts.create_speech_stream()
        stream.add_chunk("This cannot be decoded.")
        stream.finish()

        self.assertTrue(stream.wait_until_done(1.0))
        self.assertEqual(stream.metrics().status, "failed")
        self.assertTrue(all(not path.exists() for path in backend.paths))


if __name__ == "__main__":
    unittest.main()
