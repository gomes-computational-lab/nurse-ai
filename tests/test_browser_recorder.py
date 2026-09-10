from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
