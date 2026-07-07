from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from sim.audio import AudioDependencyError, AudioRecordingError, record_microphone_until_stopped
from sim.evaluator import evaluate_transcript
from sim.ollama_client import OllamaClient, OllamaError
from sim.scenarios import load_scenario
from sim.session import SimulationSession
from sim.speech_to_text import DEFAULT_STT_MODEL, SpeechToTextDependencyError, SpeechToTextError, transcribe_audio
from sim.storage import save_result
from sim.terminal_ui import choose_scenario, print_feedback, print_scenarios
from sim.text_to_speech import speak_text


DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminal voice demo for the nursing simulation app.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model to use. Default: {DEFAULT_MODEL}")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host URL.")
    parser.add_argument("--scenario", default=None, help="Scenario ID to run.")
    parser.add_argument("--list", action="store_true", help="List available scenarios and exit.")
    parser.add_argument(
        "--stt-model",
        default=DEFAULT_STT_MODEL,
        help=f"Local faster-whisper model name. Default: {DEFAULT_STT_MODEL}",
    )
    args = parser.parse_args()

    if args.list:
        print_scenarios()
        return

    try:
        scenario = load_scenario(args.scenario) if args.scenario else choose_scenario()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)

    client = OllamaClient(model=args.model, host=args.host)
    session = SimulationSession(scenario, client)

    print(f"\nScenario: {scenario.title}")
    print(f"Setting: {scenario.setting}")
    print("Press Space to start recording and Space again to stop.")
    print("Type /end or /quit, then press Enter.\n")

    try:
        opening_latencies: dict[str, float] = {}
        saved_latency: dict[str, Any] = {"opening": opening_latencies, "turns": []}

        patient_text = _time_step("generation", opening_latencies, session.opening)
        print(f"{scenario.role}: {patient_text}\n")
        _time_step("speech", opening_latencies, speak_text, patient_text)
        saved_latency["opening"] = _serialize_latencies(opening_latencies)
        _print_latency_summary(opening_latencies)

        while True:
            action = _read_voice_action()
            if action == "quit":
                path = save_result(scenario, session.transcript, latency=saved_latency)
                print(f"\nSaved transcript without feedback: {path}")
                return
            if action == "end":
                break

            print("\nRecording... press Space to stop.\n")
            turn_latencies: dict[str, float] = {}

            try:
                audio_path = _time_step("recording", turn_latencies, _record_until_space)
            except AudioDependencyError as exc:
                print(f"\nError: {exc}", file=sys.stderr)
                raise SystemExit(1)
            except AudioRecordingError as exc:
                print(f"\nAudio error: {exc}\n")
                continue

            try:
                transcript = _time_step(
                    "transcription",
                    turn_latencies,
                    transcribe_audio,
                    audio_path,
                    model_name=args.stt_model,
                )
            except SpeechToTextDependencyError as exc:
                print(f"\nError: {exc}", file=sys.stderr)
                raise SystemExit(1)
            except SpeechToTextError as exc:
                print(f"\nTranscription error: {exc}\n")
                continue
            finally:
                _cleanup_temp_file(audio_path)

            if not transcript:
                print("\nNo speech detected. Try again.\n")
                _print_latency_summary(turn_latencies)
                continue

            print(f"\nStudent: {transcript}\n")

            patient_text = _time_step("generation", turn_latencies, session.respond, transcript)
            print(f"{scenario.role}: {patient_text}\n")
            _time_step("speech", turn_latencies, speak_text, patient_text)
            saved_latency["turns"].append(
                {
                    "turn": len(saved_latency["turns"]) + 1,
                    "latency": _serialize_latencies(turn_latencies),
                }
            )
            _print_latency_summary(turn_latencies)

        print("\nEvaluating student performance...\n")
        feedback = evaluate_transcript(scenario, session.transcript, client)
        path = save_result(scenario, session.transcript, feedback, latency=saved_latency)
        print_feedback(feedback)
        print(f"\nSaved transcript and feedback: {path}")
    except KeyboardInterrupt:
        path = save_result(scenario, session.transcript, latency=saved_latency)
        print(f"\nInterrupted. Saved transcript without feedback: {path}")
    except OllamaError as exc:
        print(f"\nOllama error: {exc}", file=sys.stderr)
        raise SystemExit(1)


def _time_step(name: str, latencies: dict[str, float], func, *args, **kwargs):
    started = time.perf_counter()
    result = func(*args, **kwargs)
    latencies[name] = time.perf_counter() - started
    return result


def _print_latency_summary(latencies: dict[str, float]) -> None:
    if not latencies:
        return

    ordered_names = ("recording", "transcription", "generation", "speech")
    parts = [f"{name}: {latencies[name]:.2f}s" for name in ordered_names if name in latencies]
    print(f"Latency | {' | '.join(parts)}\n")


def _serialize_latencies(latencies: dict[str, float]) -> dict[str, float]:
    return {name: round(value, 3) for name, value in latencies.items()}


def _cleanup_temp_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _record_until_space() -> Path:
    stop_event = threading.Event()
    result: dict[str, Path] = {}
    error: dict[str, BaseException] = {}

    def target() -> None:
        try:
            result["audio_path"] = record_microphone_until_stopped(stop_event)
        except BaseException as exc:
            error["exception"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()

    try:
        _wait_for_space()
    finally:
        stop_event.set()
        worker.join()

    if "exception" in error:
        raise error["exception"]

    return result["audio_path"]


def _read_voice_action() -> str:
    prompt = "Press Space to record, or type /end or /quit then Enter: "

    if os.name == "nt":
        return _read_voice_action_windows(prompt)

    if sys.stdin.isatty():
        return _read_voice_action_posix(prompt)

    command = input(prompt).strip()
    if command == "/quit":
        return "quit"
    if command == "/end":
        return "end"
    return "record"


def _wait_for_space() -> None:
    if os.name == "nt":
        _wait_for_space_windows()
        return

    if sys.stdin.isatty():
        _wait_for_space_posix()
        return

    input("Press Enter to stop recording: ")


def _read_voice_action_windows(prompt: str) -> str:
    import msvcrt

    sys.stdout.write(prompt)
    sys.stdout.flush()
    buffer: list[str] = []

    while True:
        char = msvcrt.getwch()
        if char == "\x03":
            raise KeyboardInterrupt
        if char == " " and not buffer:
            sys.stdout.write("\n")
            sys.stdout.flush()
            return "record"
        if char in ("\r", "\n"):
            command = "".join(buffer).strip()
            sys.stdout.write("\n")
            sys.stdout.flush()
            if command == "/quit":
                return "quit"
            if command == "/end":
                return "end"
            buffer.clear()
            sys.stdout.write(prompt)
            sys.stdout.flush()
            continue
        if char == "\x08":
            if buffer:
                buffer.pop()
                sys.stdout.write("\b \b")
                sys.stdout.flush()
            continue
        if char.isprintable():
            buffer.append(char)
            sys.stdout.write(char)
            sys.stdout.flush()


def _wait_for_space_windows() -> None:
    import msvcrt

    while True:
        char = msvcrt.getwch()
        if char == "\x03":
            raise KeyboardInterrupt
        if char == " ":
            return


def _read_voice_action_posix(prompt: str) -> str:
    import termios
    import tty

    fd = sys.stdin.fileno()
    original = termios.tcgetattr(fd)
    buffer: list[str] = []

    sys.stdout.write(prompt)
    sys.stdout.flush()

    try:
        tty.setraw(fd)
        while True:
            char = sys.stdin.read(1)
            if char == "\x03":
                raise KeyboardInterrupt
            if char == " " and not buffer:
                sys.stdout.write("\n")
                sys.stdout.flush()
                return "record"
            if char in ("\r", "\n"):
                command = "".join(buffer).strip()
                sys.stdout.write("\n")
                sys.stdout.flush()
                if command == "/quit":
                    return "quit"
                if command == "/end":
                    return "end"
                buffer.clear()
                sys.stdout.write(prompt)
                sys.stdout.flush()
                continue
            if char in ("\x7f", "\b"):
                if buffer:
                    buffer.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue
            if char.isprintable():
                buffer.append(char)
                sys.stdout.write(char)
                sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original)


def _wait_for_space_posix() -> None:
    import termios
    import tty

    fd = sys.stdin.fileno()
    original = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)
        while True:
            char = sys.stdin.read(1)
            if char == "\x03":
                raise KeyboardInterrupt
            if char == " ":
                return
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original)
