from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

import diarization_demo
from sim.diarization import (
    AudioGap,
    AudioSnapshot,
    DiarizationConfig,
    InputDevice,
    ProcessingDiagnostics,
    ProcessingResult,
    Utterance,
    UtteranceStatus,
)
from sim.diarization import cli


def diagnostics(start: float = 0, end: float = 6) -> ProcessingDiagnostics:
    return ProcessingDiagnostics(start, end, 0.25, -20, -5, 2, 2, 2)


def result(
    *,
    final: tuple[Utterance, ...] = (),
    provisional: tuple[Utterance, ...] = (),
    gaps: tuple[AudioGap, ...] = (),
    start: float = 0,
    end: float = 6,
) -> ProcessingResult:
    return ProcessingResult((), (), final, provisional, gaps, diagnostics(start, end))


class FakeCapture:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.sample_rate = kwargs["sample_rate"]
        self.stopped = False
        self.snapshot_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def ensure_healthy(self):
        return None

    def stop(self):
        self.stopped = True

    def snapshot(self, minimum_total):
        del minimum_total
        self.snapshot_calls += 1
        if self.snapshot_calls == 1:
            return AudioSnapshot(
                np.ones(6 * self.sample_rate, dtype=np.float32),
                6 * self.sample_rate,
                self.sample_rate,
            )
        if self.snapshot_calls == 2:
            raise KeyboardInterrupt
        return AudioSnapshot(
            np.ones(6 * self.sample_rate, dtype=np.float32),
            7 * self.sample_rate,
            self.sample_rate,
        )


class DiarizationDemoTests(unittest.TestCase):
    def test_root_module_is_only_a_compatibility_entry_point(self) -> None:
        self.assertIs(diarization_demo.main, cli.main)
        self.assertEqual(diarization_demo.__all__, ["main"])

    def test_normal_presentation_labels_final_and_provisional_output(self) -> None:
        final = Utterance(1, 2, "SPEAKER_00", "Hello.", UtteranceStatus.FINALIZED)
        provisional = Utterance(4, 5, "SPEAKER_01", "Hi.", UtteranceStatus.PROVISIONAL)
        stdout, stderr = StringIO(), StringIO()

        cli.present_result(
            result(
                final=(final,),
                provisional=(provisional,),
                gaps=(AudioGap(2, 3, "overwritten"),),
            ),
            inspect=False,
            stdout=stdout,
            stderr=stderr,
        )

        self.assertEqual(stdout.getvalue().splitlines(), [
            "[FINAL 00:01.000-00:02.000] SPEAKER_00: Hello.",
            "[PROVISIONAL 00:04.000-00:05.000] SPEAKER_01: Hi.",
        ])
        self.assertIn("skipped 00:02.000-00:03.000", stderr.getvalue())
        self.assertIn("Analyzed 00:00.000-00:06.000 in 0.25s", stderr.getvalue())

    def test_ctrl_c_processes_new_capture_then_flushes_pending_tail(self) -> None:
        config = DiarizationConfig()
        service = Mock()
        service.process.side_effect = [
            result(start=0, end=6),
            result(
                provisional=(
                    Utterance(6, 7, "SPEAKER_00", "tail", UtteranceStatus.PROVISIONAL),
                ),
                start=1,
                end=7,
            ),
        ]
        service.flush.return_value = result(
            final=(Utterance(6, 7, "SPEAKER_00", "tail", UtteranceStatus.FINALIZED),),
            start=1,
            end=7,
        )
        captures: list[FakeCapture] = []

        def capture_factory(**kwargs):
            capture = FakeCapture(**kwargs)
            captures.append(capture)
            return capture

        stdout, stderr = StringIO(), StringIO()
        with (
            redirect_stdout(stdout),
            redirect_stderr(stderr),
            self.assertRaises(KeyboardInterrupt),
        ):
            cli.run_live(service, None, config, capture_factory=capture_factory)

        self.assertTrue(captures[0].stopped)
        self.assertEqual(service.process.call_count, 2)
        self.assertEqual(service.process.call_args_list[0].args[1], 0)
        self.assertEqual(service.process.call_args_list[1].args[1], 1)
        self.assertEqual(len(service.process.call_args_list[1].args[0]), 6 * config.sample_rate)
        service.flush.assert_called_once_with()
        self.assertIn("Finalizing newly captured audio...", stdout.getvalue())
        self.assertIn("[PROVISIONAL 00:06.000-00:07.000]", stdout.getvalue())
        self.assertIn("[FINAL 00:06.000-00:07.000]", stdout.getvalue())

    def test_main_builds_configuration_and_runs_modular_service(self) -> None:
        service = Mock()
        with (
            patch("sim.diarization.cli.validate_input_device", return_value="Test microphone"),
            patch("sim.diarization.cli.load_service", return_value=service) as load,
            patch("sim.diarization.cli.run_live") as live,
            redirect_stdout(StringIO()),
        ):
            status = cli.main([
                "--window", "15",
                "--step", "3",
                "--finalization-delay", "3",
                "--stt-model", "base.en",
            ])

        self.assertEqual(status, 0)
        config = load.call_args.args[0]
        self.assertEqual(config.window_seconds, 15)
        self.assertEqual(config.step_seconds, 3)
        self.assertEqual(config.finalization_delay_seconds, 3)
        self.assertEqual(config.whisper_model_name, "base.en")
        live.assert_called_once_with(service, None, config, inspect=False)

    def test_list_devices_and_mic_check_do_not_load_models(self) -> None:
        devices = (InputDevice(2, "Boom", "Core Audio", True),)
        with (
            patch("sim.diarization.cli.list_input_devices", return_value=devices),
            patch("sim.diarization.cli.load_service") as load,
            redirect_stdout(StringIO()) as stdout,
        ):
            self.assertEqual(cli.main(["--list-devices"]), 0)
        self.assertIn("* 2: Boom [Core Audio]", stdout.getvalue())
        load.assert_not_called()

        with (
            patch("sim.diarization.cli.validate_input_device", return_value="Boom"),
            patch("sim.diarization.cli.run_microphone_check") as check,
            patch("sim.diarization.cli.load_service") as load,
            redirect_stdout(StringIO()),
        ):
            self.assertEqual(cli.main(["--check-mic", "--device", "2"]), 0)
        check.assert_called_once_with(2, sample_rate=16_000)
        load.assert_not_called()

    def test_invalid_configuration_and_runtime_errors_have_cli_statuses(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
            cli.main(["--window", "2", "--step", "1", "--finalization-delay", "3"])
        self.assertEqual(error.exception.code, 2)

        with (
            patch("sim.diarization.cli.validate_input_device", side_effect=RuntimeError("no mic")),
            redirect_stderr(StringIO()) as stderr,
        ):
            self.assertEqual(cli.main([]), 1)
        self.assertIn("Error: no mic", stderr.getvalue())

    def test_main_treats_ctrl_c_as_a_clean_shutdown(self) -> None:
        with (
            patch("sim.diarization.cli.validate_input_device", return_value="Boom"),
            patch("sim.diarization.cli.load_service", return_value=Mock()),
            patch("sim.diarization.cli.run_live", side_effect=KeyboardInterrupt),
            redirect_stdout(StringIO()) as stdout,
        ):
            self.assertEqual(cli.main([]), 0)
        self.assertIn("Stopped. Microphone closed.", stdout.getvalue())

    def test_cli_import_does_not_load_optional_runtime_dependencies(self) -> None:
        script = """
import sys
import sim.diarization.cli
heavy = ('torch', 'pyannote', 'pyannote.audio', 'faster_whisper', 'sounddevice')
print([name for name in heavy if name in sys.modules])
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stdout.strip(), "[]")


if __name__ == "__main__":
    unittest.main()
