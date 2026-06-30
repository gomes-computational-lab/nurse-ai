from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from sim.audio import AudioDependencyError, AudioRecordingError, record_microphone_clip
from sim.evaluator import evaluate_transcript
from sim.ollama_client import OllamaClient, OllamaError
from sim.scenarios import load_scenario
from sim.session import SimulationSession
from sim.speech_to_text import DEFAULT_STT_MODEL, SpeechToTextDependencyError, SpeechToTextError, transcribe_audio
from sim.storage import save_result
from sim.terminal_ui import choose_scenario, print_feedback, print_scenarios
from sim.text_to_speech import speak_text


DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")
DEFAULT_RECORD_SECONDS = 5.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminal voice demo for the nursing simulation app.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model to use. Default: {DEFAULT_MODEL}")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host URL.")
    parser.add_argument("--scenario", default=None, help="Scenario ID to run.")
    parser.add_argument("--list", action="store_true", help="List available scenarios and exit.")
    parser.add_argument(
        "--record-seconds",
        type=float,
        default=DEFAULT_RECORD_SECONDS,
        help=f"Fixed microphone recording duration in seconds. Default: {DEFAULT_RECORD_SECONDS}",
    )
    parser.add_argument(
        "--stt-model",
        default=DEFAULT_STT_MODEL,
        help=f"Local faster-whisper model name. Default: {DEFAULT_STT_MODEL}",
    )
    args = parser.parse_args()

    if args.list:
        print_scenarios()
        return

    if args.record_seconds <= 0:
        print("Error: --record-seconds must be greater than zero.", file=sys.stderr)
        raise SystemExit(1)

    try:
        scenario = load_scenario(args.scenario) if args.scenario else choose_scenario()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)

    client = OllamaClient(model=args.model, host=args.host)
    session = SimulationSession(scenario, client)

    print(f"\nScenario: {scenario.title}")
    print(f"Setting: {scenario.setting}")
    print(f"Voice recording length: {args.record_seconds:.1f} seconds")
    print("Press Enter to record, or type /end or /quit.\n")

    try:
        latencies: dict[str, float] = {}
        patient_text = _time_step("generation", latencies, session.opening)
        print(f"{scenario.role}: {patient_text}\n")
        _time_step("speech", latencies, speak_text, patient_text)
        _print_latency_summary(latencies)

        while True:
            command = input("Command: ").strip()
            if command == "/quit":
                path = save_result(scenario, session.transcript)
                print(f"\nSaved transcript without feedback: {path}")
                return
            if command == "/end":
                break
            if command:
                print("Press Enter to record, or type /end or /quit.")
                continue

            print(f"\nRecording for {args.record_seconds:.1f} seconds...")
            turn_latencies: dict[str, float] = {}

            try:
                audio_path = _time_step("recording", turn_latencies, record_microphone_clip, args.record_seconds)
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
            _print_latency_summary(turn_latencies)

        print("\nEvaluating student performance...\n")
        feedback = evaluate_transcript(scenario, session.transcript, client)
        path = save_result(scenario, session.transcript, feedback)
        print_feedback(feedback)
        print(f"\nSaved transcript and feedback: {path}")
    except KeyboardInterrupt:
        path = save_result(scenario, session.transcript)
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


def _cleanup_temp_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
