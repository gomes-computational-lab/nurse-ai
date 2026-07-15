from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from typing import Any

from sim.audio import (
    DEFAULT_END_SILENCE_MS,
    DEFAULT_MAX_RECORDING_SECONDS,
    AudioDependencyError,
    AudioRecordingError,
    RecordedAudio,
    VoiceActivityDetectionUnavailable,
    record_microphone_until_silence,
    record_microphone_until_stopped,
)
from sim.evaluator import evaluate_transcript
from sim.ollama_client import OllamaClient, OllamaError
from sim.scenarios import load_scenario
from sim.session import SimulationSession
from sim.speech_to_text import (
    DEFAULT_STT_MODEL,
    SpeechToTextDependencyError,
    SpeechToTextError,
    preload_speech_to_text_model,
    transcribe_audio,
)
from sim.storage import save_result
from sim.terminal_ui import choose_scenario, print_feedback, print_scenarios
from sim.text_to_speech import create_speech_stream, stop_speaking


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
    parser.add_argument(
        "--manual-stop",
        action="store_true",
        help="Require Enter to stop recording instead of detecting end-of-speech.",
    )
    parser.add_argument(
        "--end-silence-ms",
        type=int,
        default=DEFAULT_END_SILENCE_MS,
        help=f"Silence that ends an automatic recording. Default: {DEFAULT_END_SILENCE_MS} ms.",
    )
    parser.add_argument(
        "--max-recording-seconds",
        type=float,
        default=DEFAULT_MAX_RECORDING_SECONDS,
        help=f"Maximum automatic recording length. Default: {DEFAULT_MAX_RECORDING_SECONDS:g} seconds.",
    )
    args = parser.parse_args()

    if args.end_silence_ms <= 0:
        parser.error("--end-silence-ms must be greater than zero")
    if args.max_recording_seconds <= 0:
        parser.error("--max-recording-seconds must be greater than zero")

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
    stt_preload = _start_stt_preload(args.stt_model)
    ollama_preload = _start_ollama_preload(client)
    manual_recording = bool(args.manual_stop)

    print(f"\nScenario: {scenario.title}")
    print(f"Setting: {scenario.setting}")
    if manual_recording:
        print("Press Enter to start recording and Enter again to stop.")
    else:
        print("Press Enter to speak. Recording stops automatically when you finish.")
    print("Pressing Enter also interrupts any patient audio still playing.")
    print("Type /end or /quit, then press Enter.\n")

    pending_audio_trackers: list[threading.Thread] = []
    saved_latency: dict[str, Any] = {"opening": {}, "turns": []}

    try:
        _await_ollama_preload(ollama_preload)
        opening_started = time.perf_counter()
        opening_prompt_char_count = session.opening_prompt_char_count(response_mode="voice")
        opening_chunk_timing: dict[str, float] = {}
        _print_role_prefix(scenario.role)
        opening_speech_stream = create_speech_stream()
        generation_started = time.perf_counter()
        session.opening(
            _stream_patient_chunk(opening_speech_stream, opening_chunk_timing),
            response_mode="voice",
        )
        generation_finished = time.perf_counter()
        _finish_streamed_response()
        opening_speech_stream.finish()
        opening_metrics: dict[str, Any] = {
            "generation_seconds": _round_seconds(generation_finished - generation_started),
            "llm_first_token_seconds": _duration_from_timestamp(
                opening_chunk_timing.get("first_token_at"), generation_started
            ),
            "prompt_char_count": opening_prompt_char_count,
            "turn_total_until_text": _round_seconds(generation_finished - opening_started),
        }
        pending_audio_trackers.append(
            _track_audio_completion(
                opening_speech_stream,
                opening_metrics,
                origin=opening_started,
                metric_prefix="opening",
            )
        )
        saved_latency["opening"] = opening_metrics

        while True:
            action = _read_voice_action()
            if action == "quit":
                _finalize_audio_metrics(pending_audio_trackers)
                path = save_result(scenario, session.transcript, latency=saved_latency)
                print(f"\nSaved transcript without feedback: {path}")
                return
            if action == "end":
                break

            stop_speaking()
            turn_started = time.perf_counter()

            try:
                recorded_audio, manual_recording = _record_voice_turn(
                    manual_recording=manual_recording,
                    end_silence_ms=args.end_silence_ms,
                    max_recording_seconds=args.max_recording_seconds,
                )
            except AudioDependencyError as exc:
                print(f"\nError: {exc}", file=sys.stderr)
                raise SystemExit(1)
            except AudioRecordingError as exc:
                print(f"\nAudio error: {exc}\n")
                continue

            recording_finished = time.perf_counter()
            recording_seconds = recording_finished - turn_started
            endpoint_delay = recorded_audio.endpoint_delay_seconds or 0.0
            speech_ended_at = recording_finished - endpoint_delay

            if not recorded_audio.has_speech:
                print("\nNo speech detected. Try again.\n")
                continue

            try:
                _await_stt_preload(stt_preload)
                transcription_started = time.perf_counter()
                transcript = transcribe_audio(
                    recorded_audio.samples,
                    model_name=args.stt_model,
                )
                transcription_finished = time.perf_counter()
            except SpeechToTextDependencyError as exc:
                print(f"\nError: {exc}", file=sys.stderr)
                raise SystemExit(1)
            except SpeechToTextError as exc:
                print(f"\nTranscription error: {exc}\n")
                continue

            if not transcript:
                print("\nNo speech detected. Try again.\n")
                continue

            print(f"\nStudent: {transcript}\n")

            prompt_char_count = session.response_prompt_char_count(
                transcript,
                response_mode="voice",
            )
            _print_role_prefix(scenario.role)
            speech_stream = create_speech_stream()
            chunk_timing: dict[str, float] = {}
            generation_started = time.perf_counter()
            session.respond(
                transcript,
                _stream_patient_chunk(speech_stream, chunk_timing),
                response_mode="voice",
            )
            generation_finished = time.perf_counter()
            _finish_streamed_response()
            speech_stream.finish()

            first_token_at = chunk_timing.get("first_token_at")
            turn_metrics: dict[str, Any] = {
                "recording_seconds": _round_seconds(recording_seconds),
                "captured_audio_seconds": _round_seconds(recorded_audio.captured_seconds),
                "speech_seconds": _round_seconds(recorded_audio.speech_seconds),
                "endpoint_delay_seconds": _round_seconds(endpoint_delay),
                "recording_stop_reason": recorded_audio.stop_reason,
                "transcription_seconds": _round_seconds(
                    transcription_finished - transcription_started
                ),
                "speech_end_to_transcript_seconds": _round_seconds(
                    transcription_finished - speech_ended_at
                ),
                "llm_first_token_seconds": _duration_from_timestamp(
                    first_token_at, generation_started
                ),
                "speech_end_to_first_token_seconds": _duration_from_timestamp(
                    first_token_at, speech_ended_at
                ),
                "generation_seconds": _round_seconds(generation_finished - generation_started),
                "prompt_char_count": prompt_char_count,
                "turn_total_until_text": _round_seconds(generation_finished - turn_started),
            }
            pending_audio_trackers.append(
                _track_audio_completion(
                    speech_stream,
                    turn_metrics,
                    origin=speech_ended_at,
                    metric_prefix="speech_end",
                )
            )
            saved_latency["turns"].append(
                {
                    "turn": len(saved_latency["turns"]) + 1,
                    "metrics": turn_metrics,
                }
            )

        print("\nEvaluating student performance...\n")
        _finalize_audio_metrics(pending_audio_trackers)
        feedback = evaluate_transcript(scenario, session.transcript, client)
        path = save_result(scenario, session.transcript, feedback, latency=saved_latency)
        print_feedback(feedback)
        print(f"\nSaved transcript and feedback: {path}")
    except KeyboardInterrupt:
        _finalize_audio_metrics(pending_audio_trackers)
        path = save_result(scenario, session.transcript, latency=saved_latency)
        print(f"\nInterrupted. Saved transcript without feedback: {path}")
    except OllamaError as exc:
        print(f"\nOllama error: {exc}", file=sys.stderr)
        raise SystemExit(1)


def _round_seconds(value: float) -> float:
    return round(max(0.0, value), 3)


def _duration_from_timestamp(timestamp: float | None, origin: float) -> float | None:
    return _round_seconds(timestamp - origin) if timestamp is not None else None


def _print_role_prefix(role: str) -> None:
    print(f"{role}: ", end="", flush=True)


def _print_stream_chunk(chunk: str) -> None:
    print(chunk, end="", flush=True)


def _stream_patient_chunk(speech_stream, timing: dict[str, float]):
    def callback(chunk: str) -> None:
        if chunk and "first_token_at" not in timing:
            timing["first_token_at"] = time.perf_counter()
        _print_stream_chunk(chunk)
        speech_stream.add_chunk(chunk)

    return callback


def _finish_streamed_response() -> None:
    print("\n")


def _track_audio_completion(
    speech_stream,
    metrics: dict[str, Any],
    *,
    origin: float,
    metric_prefix: str,
) -> threading.Thread:
    def target() -> None:
        speech_stream.wait_until_done()
        speech_metrics = speech_stream.metrics()
        metrics["audio_status"] = speech_metrics.status
        metrics["audio_segments_started"] = speech_metrics.segments_started
        metrics[f"{metric_prefix}_to_first_audio_seconds"] = _duration_from_timestamp(
            speech_metrics.first_audio_started_at,
            origin,
        )
        metrics[f"{metric_prefix}_to_audio_end_seconds"] = _duration_from_timestamp(
            speech_metrics.completed_at,
            origin,
        )

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    return worker


def _finalize_audio_metrics(trackers: list[threading.Thread]) -> None:
    stop_speaking()
    for tracker in trackers:
        tracker.join(timeout=5.0)


def _start_ollama_preload(client: OllamaClient) -> dict[str, Any]:
    preload_state: dict[str, Any] = {"error": None}

    def target() -> None:
        try:
            client.warmup()
        except BaseException as exc:
            preload_state["error"] = exc

    preload_state["thread"] = threading.Thread(target=target, daemon=True)
    preload_state["thread"].start()
    return preload_state


def _await_ollama_preload(preload_state: dict[str, Any]) -> None:
    thread = preload_state.get("thread")
    if thread is not None:
        thread.join()
        preload_state["thread"] = None

    error = preload_state.get("error")
    if error is not None:
        raise error


def _start_stt_preload(model_name: str) -> dict[str, Any]:
    preload_state: dict[str, Any] = {"error": None}

    def target() -> None:
        try:
            preload_speech_to_text_model(model_name=model_name)
        except BaseException as exc:
            preload_state["error"] = exc

    preload_state["thread"] = threading.Thread(target=target, daemon=True)
    preload_state["thread"].start()
    return preload_state


def _await_stt_preload(preload_state: dict[str, Any]) -> None:
    thread = preload_state.get("thread")
    if thread is not None:
        thread.join()
        preload_state["thread"] = None

    error = preload_state.get("error")
    if error is not None:
        raise error


def _record_voice_turn(
    *,
    manual_recording: bool,
    end_silence_ms: int,
    max_recording_seconds: float,
) -> tuple[RecordedAudio, bool]:
    if manual_recording:
        print("\nRecording... press Enter to stop.\n")
        return _record_until_enter(), True

    print("\nListening... speak now. Recording will stop automatically.\n")
    try:
        return (
            record_microphone_until_silence(
                end_silence_ms=end_silence_ms,
                max_recording_seconds=max_recording_seconds,
            ),
            False,
        )
    except VoiceActivityDetectionUnavailable as exc:
        print(f"Voice activity detection unavailable ({exc}). Falling back to manual stop.")
        print("\nRecording... press Enter to stop.\n")
        return _record_until_enter(), True


def _record_until_enter() -> RecordedAudio:
    stop_event = threading.Event()
    result: dict[str, RecordedAudio] = {}
    error: dict[str, BaseException] = {}

    def target() -> None:
        try:
            result["audio"] = record_microphone_until_stopped(stop_event)
        except BaseException as exc:
            error["exception"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()

    try:
        _prompt_line("Press Enter to stop recording: ")
    finally:
        stop_event.set()
        worker.join(timeout=5.0)

    if worker.is_alive():
        raise AudioRecordingError("Microphone recording did not stop cleanly. Try again.")
    if "exception" in error:
        raise error["exception"]
    if "audio" not in result:
        raise AudioRecordingError("Microphone recording ended without audio.")
    return result["audio"]


def _read_voice_action() -> str:
    prompt = "Press Enter to speak, or type /end or /quit then Enter: "
    command = _prompt_line(prompt).strip()
    if command == "/quit":
        return "quit"
    if command == "/end":
        return "end"
    return "record"


def _prompt_line(prompt: str) -> str:
    sys.stdout.write(prompt)
    sys.stdout.flush()
    line = sys.stdin.readline()
    if line == "":
        raise EOFError("Standard input closed.")
    return line
