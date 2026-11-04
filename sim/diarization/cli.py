"""Command-line presentation for the standalone live diarization service."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from collections.abc import Callable
from typing import TextIO

import numpy as np

from .capture import (
    AudioSnapshot,
    InputDevice,
    MicrophoneCapture,
    list_input_devices,
    validate_input_device,
)
from .config import DiarizationConfig
from .models import ProcessingResult, Utterance
from .providers import load_pyannote_provider, load_whisper_provider
from .service import DiarizationService


CaptureFactory = Callable[..., MicrophoneCapture]


def timestamp(seconds: float) -> str:
    milliseconds = round(max(0.0, seconds) * 1000)
    minutes, remainder = divmod(milliseconds, 60_000)
    return f"{minutes:02d}:{remainder // 1000:02d}.{remainder % 1000:03d}"


def present_result(
    result: ProcessingResult,
    *,
    inspect: bool,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    include_diagnostics: bool = True,
) -> None:
    """Format one typed service result; no lower layer performs presentation."""
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    for gap in result.audio_gaps:
        print(
            f"Warning: inference fell behind; skipped "
            f"{timestamp(gap.start)}-{timestamp(gap.end)} ({gap.reason}).",
            file=stderr,
            flush=True,
        )

    if inspect:
        for turn in result.speaker_turns:
            print(
                f"[TURN {timestamp(turn.start)}-{timestamp(turn.end)}] "
                f"{turn.speaker_id} (model={turn.source_speaker_id})",
                file=stdout,
                flush=True,
            )
        if result.recognized_words:
            print(
                f"[STT {timestamp(result.recognized_words[0].start)}-"
                f"{timestamp(result.recognized_words[-1].end)}] "
                + " ".join(word.text for word in result.recognized_words),
                file=stdout,
                flush=True,
            )
        elif include_diagnostics and result.diagnostics.window_end > result.diagnostics.window_start:
            print(
                f"[STT {timestamp(result.diagnostics.window_start)}-"
                f"{timestamp(result.diagnostics.window_end)}] (no text)",
                file=stdout,
                flush=True,
            )
    else:
        for utterance in result.finalized_utterances:
            _print_utterance(utterance, "FINAL", stdout)
        for utterance in result.provisional_utterances:
            _print_utterance(utterance, "PROVISIONAL", stdout)
        if (
            result.speaker_turns
            and not result.finalized_utterances
            and not result.provisional_utterances
        ):
            print(
                "Speech detected, but no aligned words were emitted in this window.",
                file=stderr,
                flush=True,
            )

    if include_diagnostics:
        diagnostics = result.diagnostics
        details = (
            f"; input RMS {diagnostics.input_rms_dbfs:.0f} dBFS, "
            f"peak {diagnostics.input_peak_dbfs:.0f} dBFS; "
            f"model={diagnostics.local_speaker_count}, "
            f"tracked={diagnostics.tracked_speaker_count}, "
            f"profiles={diagnostics.profile_count}"
        )
        print(
            f"Analyzed {timestamp(diagnostics.window_start)}-"
            f"{timestamp(diagnostics.window_end)} in "
            f"{diagnostics.processing_seconds:.2f}s{details}",
            file=stdout if inspect else stderr,
            flush=True,
        )


def _print_utterance(utterance: Utterance, label: str, output: TextIO) -> None:
    print(
        f"[{label} {timestamp(utterance.start)}-{timestamp(utterance.end)}] "
        f"{utterance.speaker_id}: {utterance.text}",
        file=output,
        flush=True,
    )


def load_service(config: DiarizationConfig) -> DiarizationService:
    token = _read_hf_token()
    print(f"Loading {config.pyannote_model_name}...", flush=True)
    diarization_provider = load_pyannote_provider(config, token=token)
    print(f"Diarization device: {diarization_provider.device_name}", flush=True)
    print(
        f"Loading Whisper {config.whisper_model_name}, CPU int8, "
        f"language={config.language}...",
        flush=True,
    )
    transcription_provider = load_whisper_provider(config)
    return DiarizationService(diarization_provider, transcription_provider, config)


def _read_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if not env_file.exists():
        return None
    for line in env_file.read_text(encoding="utf-8-sig").splitlines():
        key, separator, value = line.strip().partition("=")
        if separator and key.strip() == "HF_TOKEN":
            return value.strip().strip("\"'")
    return None


def run_live(
    service: DiarizationService,
    device: int | None,
    config: DiarizationConfig,
    *,
    inspect: bool = False,
    capture_factory: CaptureFactory = MicrophoneCapture,
) -> None:
    window = round(config.window_seconds * config.sample_rate)
    step = round(config.step_seconds * config.sample_rate)
    last_processed_total = 0
    with capture_factory(
        sample_rate=config.sample_rate,
        capacity_samples=window,
        device=device,
    ) as capture:
        print(
            f"Listening: {config.sample_rate / 1000:g} kHz mono; "
            f"{config.window_seconds:g}s window, {config.step_seconds:g}s interval; "
            f"up to {config.max_speakers} speakers. Ctrl+C to stop.",
            flush=True,
        )
        next_sample = window
        try:
            while True:
                capture.ensure_healthy()
                snapshot = capture.snapshot(next_sample)
                if snapshot is None:
                    time.sleep(0.05)
                    continue
                result = service.process(snapshot.samples, snapshot.start_seconds)
                present_result(result, inspect=inspect)
                last_processed_total = snapshot.total_samples
                next_sample = snapshot.total_samples + step
        except KeyboardInterrupt:
            capture.stop()
            final_error: Exception | None = None
            snapshot = capture.snapshot(1)
            if snapshot is not None and snapshot.total_samples > last_processed_total:
                print("Finalizing newly captured audio...", flush=True)
                try:
                    result = service.process(snapshot.samples, snapshot.start_seconds)
                    present_result(result, inspect=inspect)
                except Exception as exc:
                    final_error = exc
            flush_result = service.flush()
            present_result(
                flush_result,
                inspect=inspect,
                include_diagnostics=False,
            )
            if final_error is not None:
                raise final_error
            raise


def run_microphone_check(
    device: int | None,
    *,
    sample_rate: int,
    capture_factory: CaptureFactory = MicrophoneCapture,
) -> None:
    with capture_factory(
        sample_rate=sample_rate,
        capacity_samples=sample_rate,
        device=device,
    ) as capture:
        print(
            f"Microphone check: 10 seconds, {sample_rate / 1000:g} kHz mono. "
            "No models/token needed. Ctrl+C to stop.",
            flush=True,
        )
        next_sample = sample_rate
        while True:
            capture.ensure_healthy()
            snapshot = capture.snapshot(next_sample)
            if snapshot is None:
                time.sleep(0.05)
                continue
            level, peak = _levels(snapshot)
            hint = (
                " -- nearly silent; check headset mute, boom, and input volume"
                if peak < -65
                else ""
            )
            print(
                f"Input {snapshot.end_seconds:.1f}s: RMS {level:.0f} dBFS, "
                f"peak {peak:.0f} dBFS{hint}",
                flush=True,
            )
            if snapshot.total_samples >= 10 * sample_rate:
                print("Microphone check complete. Microphone closed.", flush=True)
                return
            next_sample = snapshot.total_samples + sample_rate


def _levels(snapshot: AudioSnapshot) -> tuple[float, float]:
    rms = float(np.sqrt(np.mean(snapshot.samples ** 2)))
    peak = float(np.max(np.abs(snapshot.samples)))
    return 20 * np.log10(max(rms, 1e-12)), 20 * np.log10(max(peak, 1e-12))


def _print_devices(devices: tuple[InputDevice, ...]) -> None:
    print("Microphone IDs (* = default input):")
    for device in devices:
        print(
            f"{'*' if device.is_default else ' '} {device.id}: "
            f"{device.name} [{device.host_api}]"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-devices", action="store_true", help="List microphone IDs; no models needed.")
    parser.add_argument("--device", type=int, help="Input device ID (default: system microphone).")
    parser.add_argument("--check-mic", action="store_true", help="Show microphone levels for 10 seconds.")
    parser.add_argument("--window", type=float, default=6.0, help="Analysis window seconds (default: 6).")
    parser.add_argument("--step", type=float, default=3.0, help="Analysis interval seconds (default: 3).")
    parser.add_argument(
        "--finalization-delay",
        type=float,
        default=3.0,
        help="Provisional tail retained for overlapping reanalysis (default: 3).",
    )
    parser.add_argument(
        "--diarization-model",
        default="pyannote/speaker-diarization-community-1",
        help="Pyannote model name.",
    )
    parser.add_argument("--stt-model", default="tiny.en", help="Whisper model name (default: tiny.en).")
    parser.add_argument("--language", default="en", help="Speech language code (default: en).")
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print typed speaker turns, independent STT words, and diagnostics.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = DiarizationConfig(
            pyannote_model_name=args.diarization_model,
            whisper_model_name=args.stt_model,
            language=args.language,
            window_seconds=args.window,
            step_seconds=args.step,
            finalization_delay_seconds=args.finalization_delay,
        )
    except ValueError as exc:
        parser.error(str(exc))

    try:
        if args.list_devices:
            _print_devices(list_input_devices())
            return 0
        microphone_name = validate_input_device(args.device, sample_rate=config.sample_rate)
        print(f"Microphone: {microphone_name}", flush=True)
        if args.check_mic:
            run_microphone_check(args.device, sample_rate=config.sample_rate)
            return 0
        service = load_service(config)
        if args.inspect:
            print(
                "Inspection: TURN shows tracked and local acoustic IDs; "
                "STT shows recognized words independently.",
                flush=True,
            )
        run_live(service, args.device, config, inspect=args.inspect)
        return 0
    except KeyboardInterrupt:
        print("\nStopped. Microphone closed.", flush=True)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
