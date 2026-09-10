from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sim import browser_recorder
from sim.browser_recorder import BrowserRecording


class BrowserRecordingTests(unittest.TestCase):
    def test_file_suffix_matches_browser_mime_type(self) -> None:
        cases = {
            "audio/webm;codecs=opus": ".webm",
            "audio/ogg;codecs=opus": ".ogg",
            "audio/mp4": ".mp4",
            "": ".webm",
        }

        for mime_type, expected in cases.items():
            with self.subTest(mime_type=mime_type):
                recording = BrowserRecording(audio=b"audio", mime_type=mime_type)
                self.assertEqual(recording.file_suffix, expected)

    def test_recorder_uses_two_stage_silence_defaults(self) -> None:
        result = SimpleNamespace(recording=None, error=None)
        with patch.object(
            browser_recorder,
            "_AUTOMATIC_RECORDER",
            return_value=result,
        ) as component:
            recording, error = browser_recorder.automatic_silence_recorder(
                turn_id="turn-1",
                patient_audio=None,
                key="recorder-1",
            )

        self.assertIsNone(recording)
        self.assertIsNone(error)
        data = component.call_args.kwargs["data"]
        self.assertEqual(data["speech_start_timeout_seconds"], 5)
        self.assertEqual(data["trailing_silence_seconds"], 3)


if __name__ == "__main__":
    unittest.main()
