from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

import main as entrypoint
from sim import voice_app


class MainEntrypointTests(unittest.TestCase):
    def test_default_path_runs_original_typed_app(self) -> None:
        arguments_seen: list[str] = []

        with (
            patch.object(sys, "argv", ["main.py", "--scenario", "test"]),
            patch("sim.app.main", side_effect=lambda: arguments_seen.extend(sys.argv)),
        ):
            entrypoint.main()

        self.assertEqual(arguments_seen, ["main.py", "--scenario", "test"])

    def test_voice_flag_runs_voice_app_without_forwarding_dispatch_flag(self) -> None:
        arguments_seen: list[str] = []

        with (
            patch.object(
                sys,
                "argv",
                ["main.py", "--voice", "--scenario", "test", "--stt-model", "tiny.en"],
            ),
            patch.object(
                voice_app,
                "main",
                side_effect=lambda: arguments_seen.extend(sys.argv),
            ),
        ):
            entrypoint.main()

        self.assertEqual(
            arguments_seen,
            ["main.py", "--scenario", "test", "--stt-model", "tiny.en"],
        )

    def test_entrypoint_restores_arguments_after_dispatch(self) -> None:
        original_arguments = ["main.py", "--voice", "--list"]

        with (
            patch.object(sys, "argv", original_arguments),
            patch.object(voice_app, "main"),
        ):
            entrypoint.main()
            self.assertIs(sys.argv, original_arguments)


if __name__ == "__main__":
    unittest.main()
