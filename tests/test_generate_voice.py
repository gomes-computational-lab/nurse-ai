from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

from scripts import generate_voice


class GenerateVoiceTests(unittest.TestCase):
    def test_missing_ffmpeg_has_cross_platform_install_instructions(self) -> None:
        with patch("scripts.generate_voice.shutil.which", return_value=None):
            with self.assertRaisesRegex(
                generate_voice.VoiceGenerationError, "brew install ffmpeg"
            ):
                generate_voice.find_ffmpeg()

    def test_generation_converts_to_valid_wav_and_removes_intermediates(self) -> None:
        observed = {}

        class FakeCommunicate:
            def __init__(self, text, voice):
                observed["text"] = text
                observed["voice"] = voice

            async def save(self, path):
                Path(path).write_bytes(b"synthetic mp3")

        class FakeEdgeTTS:
            Communicate = FakeCommunicate

        def fake_run(command, **options):
            observed["command"] = command
            observed["options"] = options
            with wave.open(command[-1], "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(24_000)
                audio.writeframes(b"\x00\x00" * 24_000)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory, "child_female_8yo.wav")
            with (
                patch.object(
                    generate_voice.importlib,
                    "import_module",
                    return_value=FakeEdgeTTS(),
                ),
                patch.object(
                    generate_voice,
                    "find_ffmpeg",
                    return_value="/usr/bin/ffmpeg",
                ),
                patch.object(generate_voice.subprocess, "run", side_effect=fake_run),
            ):
                duration = asyncio.run(generate_voice.generate_voice(output))

            self.assertTrue(output.is_file())
            self.assertEqual(duration, 1.0)
            self.assertEqual(list(Path(directory).iterdir()), [output])

        self.assertEqual(observed["voice"], "en-US-AnaNeural")
        self.assertIn("when we can go home", observed["text"])
        self.assertIn("-ac", observed["command"])
        self.assertIn("24000", observed["command"])
        self.assertTrue(observed["options"]["check"])


if __name__ == "__main__":
    unittest.main()
