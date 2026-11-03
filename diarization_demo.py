"""Live two-speaker Pyannote demo; see demos/diarization/README.md."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys
import threading
import time

import numpy as np

from sim.audio import DEFAULT_SAMPLE_RATE
from sim.speech_to_text import DEFAULT_STT_MODEL

MODEL = "pyannote/speaker-diarization-community-1"


class VoiceTracker:
    """Learn at most two session voice profiles, without a setup recording."""

    def __init__(self):
        self.profiles: list[np.ndarray] = []

    def assign(self, output, turns) -> dict[str, str]:
        # Pyannote documents embeddings in speaker_diarization.labels() order.
        labels = output.speaker_diarization.labels()
        mapping = {label: "SPEAKER_UNKNOWN" for label in labels}
        embeddings = output.speaker_embeddings
        if embeddings is None or embeddings.ndim != 2 or len(embeddings) != len(labels):
            return mapping
        durations = {label: sum(b - a for a, b, speaker in turns if speaker == label)
                     for label in labels}
        # Start with the longest speech so brief interjections do not seed a voice.
        for index in sorted(range(len(labels)), key=lambda i: -durations[labels[i]]):
            label = labels[index]
            vector = np.asarray(embeddings[index], dtype=np.float64)
            norm = np.linalg.norm(vector)
            if not np.isfinite(vector).all() or norm < 1e-8:
                continue
            vector = vector / norm
            scores = np.array([np.dot(vector, profile) for profile in self.profiles])
            # These cosine thresholds are demo heuristics, not calibrated probabilities.
            if len(self.profiles) < 2 and durations[label] >= 1.0 and (
                not len(scores) or scores.max() < 0.50
            ):
                match = len(self.profiles)
                self.profiles.append(vector.copy())
            elif len(scores):
                match = int(scores.argmax())
                margin = float(scores[match] - np.min(scores)) if len(scores) == 2 else 1.0
                if scores[match] < 0.60 or margin < 0.10:
                    continue
                # Adapt slowly, only on stronger matches with enough speech.
                if scores[match] >= 0.75 and margin >= 0.20 and durations[label] >= 1.0:
                    profile = 0.95 * self.profiles[match] + 0.05 * vector
                    self.profiles[match] = profile / np.linalg.norm(profile)
            else:
                continue
            # Multiple local clusters may match the same actual person.
            mapping[label] = f"SPEAKER_{match:02d}"
        return mapping


class ConversationRoles:
    """Conservative English phrase rules; roles remain explicit estimates."""

    NURSE_CUES = (
        r"\b(?:i(?:'ll| will)|let me|we(?:'ll| will)) (?:need to )?(?:check|take|measure|examine)\b",
        r"\b(?:have you|are you|do you|did you|how long have you|when did you)\b[^.!?]{0,90}\b(?:symptoms|fever|pain|cough|medication|allergies|feeling|experienced|taken)\b",
        r"\b(?:your (?:blood pressure|temperature|pulse|vital signs)|roll up your sleeve)\b",
        r"\b(?:you (?:should|need to|must)|make sure you|try to)\b[^.!?]{0,50}\b(?:rest|drink|stay hydrated|take|avoid|monitor)\b",
        r"\bi(?:'ll| will) (?:inform|notify) (?:the |your )?(?:doctor|physician)\b",
    )
    PATIENT_CUES = (
        r"\bi(?:'m| am) (?:not feeling well|feeling (?:sick|unwell|tired)|in pain)\b",
        r"\bi(?:'ve| have) (?:also )?(?:had|been)\b[^.!?]{0,50}\b(?:headache|fever|cough|coughing|vomiting|pain|dizzy)\b",
        r"\bmy (?:head|stomach|chest|back|throat|shoulder) (?:hurts|aches|is sore)\b",
        r"\b(?:i feel (?:very )?(?:tired|dizzy|sick|weak|unwell)|i(?:'m| am) (?:coughing|vomiting|nauseous))\b",
        r"\bi (?:took|have taken)\b[^.!?]{0,40}\b(?:painkillers|medication|medicine|ibuprofen|paracetamol|aspirin)\b",
        r"\b(?:is it serious|should i be worried|do i need to (?:stay|take)|do you think i need)\b",
    )

    def __init__(self):
        # Two small sets of cue IDs per voice, plus at most 160 trailing characters.
        self.evidence = {label: (set(), set()) for label in ("SPEAKER_00", "SPEAKER_01")}
        self.tails: dict[str, tuple[float, str]] = {}
        self.roles: dict[str, str] = {}

    def observe(self, lines) -> bool:
        for a, b, label, text in lines:
            if label not in self.evidence:
                continue
            old_end, tail = self.tails.get(label, (-float("inf"), ""))
            text = ((tail + " ") if a - old_end <= 1.5 else "") + text.lower().replace("’", "'")
            self.tails[label] = (b, re.split(r"[.!?]", text)[-1][-160:])
            nurse = {i for i, rule in enumerate(self.NURSE_CUES) if re.search(rule, text)}
            patient = {i for i, rule in enumerate(self.PATIENT_CUES) if re.search(rule, text)}
            if nurse and patient:
                continue  # A merged/mixed-role line is unsafe evidence for either role.
            self.evidence[label][0].update(nurse)
            self.evidence[label][1].update(patient)
        scores = {label: len(nurse) - len(patient)
                  for label, (nurse, patient) in self.evidence.items()}
        roles = {}
        if scores["SPEAKER_00"] >= 2 and scores["SPEAKER_01"] <= -2:
            roles = {"SPEAKER_00": "NURSE", "SPEAKER_01": "PATIENT"}
        elif scores["SPEAKER_01"] >= 2 and scores["SPEAKER_00"] <= -2:
            roles = {"SPEAKER_00": "PATIENT", "SPEAKER_01": "NURSE"}
        changed = roles != self.roles
        self.roles = roles
        return changed

    def describe(self, label: str) -> str:
        role = self.roles.get(label)
        return f"{label} [{role}, estimated]" if role else f"{label} [ROLE_UNKNOWN]"


def timestamp(seconds: float) -> str:
    milliseconds = round(max(0.0, seconds) * 1000)
    minutes, remainder = divmod(milliseconds, 60_000)
    return f"{minutes:02d}:{remainder // 1000:02d}.{remainder % 1000:03d}"


class Diarizer:
    """Process rolling microphone windows and emit speaker turns and text."""

    def __init__(self, pipeline, torch, whisper=None, language: str | None = "en",
                 nurse_patient: bool = False, holdback: float = 0.0,
                 track_speakers: bool = False, inspect: bool = False):
        self.pipeline = pipeline
        self.torch = torch
        self.whisper = whisper
        self.language = language
        self.holdback = holdback
        self.inspect = inspect
        self.previous: list[tuple[float, float, str]] = []
        self.emitted_until = 0.0
        self.voices = VoiceTracker() if nurse_patient or track_speakers else None
        self.roles = ConversationRoles() if nurse_patient and not inspect else None

    def process(self, samples: np.ndarray, start: float, *, final: bool = False) -> None:
        end = start + len(samples) / DEFAULT_SAMPLE_RATE
        # Revisit the newest speech with more context before printing it.
        emit_until = end if final else max(self.emitted_until, end - self.holdback)
        began = time.perf_counter()
        with self.torch.inference_mode():
            output = self.pipeline(
                {"waveform": self.torch.from_numpy(samples).unsqueeze(0),
                 "sample_rate": DEFAULT_SAMPLE_RATE},
                min_speakers=2,
                max_speakers=2,
            )
        turns = [
            (max(start, start + turn.start), min(end, start + turn.end), speaker)
            for turn, _, speaker in output.speaker_diarization.itertracks(yield_label=True)
        ]
        local_turns = turns
        # Local cluster IDs can swap. Match the two possible assignments using
        # speech in the overlapping window. No enrollment or history of audio.
        def agreement(swap: bool) -> float:
            return sum(
                max(0.0, min(b, old_b) - max(a, old_a))
                for a, b, label in turns
                for old_a, old_b, old_label in self.previous
                if (self.swap(label) if swap else label) == old_label
            )

        if self.voices is not None:
            mapping = self.voices.assign(output, turns)
            # Community-1's exclusive turns avoid choosing arbitrarily between
            # simultaneous speakers when aligning a single recognized word.
            local_turns = [
                (max(start, start + turn.start), min(end, start + turn.end), speaker)
                for turn, _, speaker in output.exclusive_speaker_diarization.itertracks(yield_label=True)
            ]
            turns = [(a, b, mapping.get(speaker, "SPEAKER_UNKNOWN"))
                     for a, b, speaker in local_turns]
        elif agreement(True) > agreement(False):
            turns = [(a, b, self.swap(label)) for a, b, label in turns]
        if start > self.emitted_until + 0.001:
            print(f"Warning: inference fell behind; skipped "
                  f"{timestamp(self.emitted_until)}-{timestamp(start)}.", file=sys.stderr)
        if self.inspect:
            # Print acoustic turns even if Whisper returns no words. The local
            # ID exposes model groups before cross-window voice matching.
            for (a, b, label), (_, _, local) in zip(turns, local_turns):
                a, b = max(a, self.emitted_until), min(b, emit_until)
                if b > a:
                    print(f"[TURN {timestamp(a)}-{timestamp(b)}] {label} (model={local})", flush=True)
            if self.whisper is not None:
                words = self.recognize(samples, start, emit_until=emit_until)
                if words:
                    a = max(words[0][0], self.emitted_until)
                    print(f"[STT {timestamp(a)}-{timestamp(words[-1][1])}] "
                          + " ".join(text for _, _, text in words), flush=True)
                elif emit_until > max(start, self.emitted_until):
                    print(f"[STT {timestamp(max(start, self.emitted_until))}-{timestamp(emit_until)}] "
                          "(no text)", flush=True)
        elif self.whisper is not None and turns:
            lines = self.transcribe(samples, start, turns, emit_until=emit_until)
            if self.roles is not None and self.roles.observe(lines):
                description = "; ".join(f"{label}={role}" for label, role in self.roles.roles.items())
                print(f"Role estimate updated: {description or 'roles uncertain'}. "
                      "Earlier lines are unchanged.", flush=True)
            for a, b, label, text in lines:
                displayed = self.roles.describe(label) if self.roles is not None else label
                print(f"[{timestamp(a)}-{timestamp(b)}] {displayed}: {text}", flush=True)
            if not lines and any(min(b, emit_until) > max(a, self.emitted_until)
                                 for a, b, _ in turns):
                print("Speech detected, but no new words recognized in this window.",
                      file=sys.stderr, flush=True)
        else:
            for a, b, label in turns:
                a = max(a, self.emitted_until)
                b = min(b, emit_until)
                if b > a:
                    print(f"[{timestamp(a)}-{timestamp(b)}] {label}", flush=True)
        elapsed = time.perf_counter() - began
        rms = float(np.sqrt(np.mean(samples ** 2)))
        level = 20 * np.log10(max(rms, 1e-12))
        if self.inspect:
            peak = 20 * np.log10(max(float(np.max(np.abs(samples))), 1e-12))
            details = f"; input RMS {level:.0f} dBFS, peak {peak:.0f} dBFS"
            details += f"; model={len({speaker for _, _, speaker in local_turns})}"
            if self.voices is not None:
                known = {speaker for _, _, speaker in turns} - {"SPEAKER_UNKNOWN"}
                details += f", tracked={len(known)}, profiles={len(self.voices.profiles)}/2"
        else:
            details = f" (no speech; input {level:.0f} dBFS)" if not turns else ""
        print(f"Analyzed {timestamp(start)}-{timestamp(end)} in {elapsed:.2f}s"
              + details, file=sys.stdout if self.inspect else sys.stderr, flush=True)
        self.previous = turns
        self.emitted_until = emit_until

    def recognize(self, samples: np.ndarray, start: float, *, emit_until: float | None = None):
        """Return timed words without using diarization to filter recognition."""
        segments, _ = self.whisper.transcribe(
            samples,
            language=self.language,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300, "speech_pad_ms": 200},
            word_timestamps=True,
        )
        words: list[tuple[float, float, str]] = []
        window_end = start + len(samples) / DEFAULT_SAMPLE_RATE
        for segment in segments:
            for word in segment.words or []:
                a, b = max(start, start + word.start), min(window_end, start + word.end)
                # A word straddling an update boundary belongs to the window
                # containing its midpoint. Avoid repeating the overlapping half.
                if b <= a or (a + b) / 2 < self.emitted_until:
                    continue
                if emit_until is not None and (a + b) / 2 >= emit_until:
                    continue
                text = word.word.strip()
                if text:
                    words.append((a, b, text))
        return words

    def transcribe(self, samples: np.ndarray, start: float, turns, *, emit_until: float | None = None):
        """Match each recognized word to its most-overlapping speaker turn."""
        lines: list[tuple[float, float, str, str]] = []
        for a, b, text in self.recognize(samples, start, emit_until=emit_until):
            turn = max(turns, key=lambda turn: max(0.0, min(b, turn[1]) - max(a, turn[0])))
            if min(b, turn[1]) <= max(a, turn[0]):
                continue  # Do not label ASR hallucinations outside detected speech.
            a = max(a, self.emitted_until)
            label = turn[2]
            if lines and lines[-1][2] == label:
                old_a, _, _, old_text = lines[-1]
                lines[-1] = (old_a, max(lines[-1][1], b), label, old_text + " " + text)
            else:
                lines.append((a, b, label, text))
        return lines

    @staticmethod
    def swap(label: str) -> str:
        return "SPEAKER_01" if label == "SPEAKER_00" else "SPEAKER_00"


def load_diarizer(stt_model: str | None = DEFAULT_STT_MODEL, language: str | None = "en",
                 nurse_patient: bool = False, holdback: float = 0.0,
                 track_speakers: bool = False, inspect: bool = False) -> Diarizer:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""),
          flush=True)
    print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}", flush=True)
    if device.type == "cuda":
        # Detect unsupported GPU wheels before downloading a model.
        (torch.ones(8, device=device) * 2).sum().item()
        torch.cuda.synchronize()
        print(f"CUDA kernel check: passed; capability {torch.cuda.get_device_capability(0)}", flush=True)
    # Optional local .env convenience; an explicitly set environment wins.
    if not os.environ.get("HF_TOKEN"):
        env_file = Path(__file__).with_name(".env")
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8-sig").splitlines():
                key, separator, value = line.strip().partition("=")
                if separator and key.strip() == "HF_TOKEN":
                    os.environ["HF_TOKEN"] = value.strip().strip("\"'")
                    break
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is not set. Set $env:HF_TOKEN or HF_TOKEN in the local .env file; "
                           "accept the model conditions at https://huggingface.co/" + MODEL)
    if not token.startswith("hf_") or any(char.isspace() for char in token):
        raise RuntimeError("HF_TOKEN is malformed. Copy the complete Hugging Face Read token "
                           "beginning with hf_, without quotes or whitespace.")
    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")
    print(f"Loading {MODEL} (first run imports dependencies and downloads model files)...", flush=True)
    # Microphone input is a tensor: the optional TorchCodec/FFmpeg decoder is unused.
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"\s*torchcodec is not installed correctly.*")
        from pyannote.audio import Pipeline

    try:
        pipeline = Pipeline.from_pretrained(MODEL, token=token)
    except Exception as exc:
        detail = str(exc).replace(token, "[REDACTED]")
        raise RuntimeError(f"Model loading failed ({type(exc).__name__}): {detail}\n"
                           "Check the demo requirements, internet access, "
                           "HF_TOKEN read access, and accepted conditions at "
                           "https://huggingface.co/" + MODEL) from None
    if pipeline is None:
        raise RuntimeError("Model access denied. Check HF_TOKEN and the model access agreement.")
    pipeline.to(device)
    whisper = None
    if stt_model is not None:
        # Reuse the app's cached CPU/int8 loader. Timestamp alignment stays
        # isolated here because the app's transcribe_audio returns plain text.
        from sim.speech_to_text import _load_model

        print(f"Transcription: {stt_model}, CPU int8, language={language or 'auto'} "
              "(first run downloads Whisper weights)...", flush=True)
        whisper = _load_model(stt_model)
    return Diarizer(pipeline, torch, whisper, language, nurse_patient=nurse_patient, holdback=holdback,
                    track_speakers=track_speakers, inspect=inspect)


class RollingAudio:
    """Fixed storage: capture overwrites old audio while inference is busy."""

    def __init__(self, capacity: int):
        self.data = np.empty(capacity, dtype=np.float32)
        self.position = 0
        self.total = 0
        self.lock = threading.Lock()

    def append(self, samples: np.ndarray) -> None:
        with self.lock:
            count = len(samples)
            self.total += count
            if count >= len(self.data):
                self.data[:] = samples[-len(self.data):]
                self.position = 0
                return
            first = min(count, len(self.data) - self.position)
            self.data[self.position:self.position + first] = samples[:first]
            self.data[:count - first] = samples[first:]
            self.position = (self.position + count) % len(self.data)

    def snapshot(self, minimum_total: int) -> tuple[np.ndarray, int] | None:
        with self.lock:
            if self.total < minimum_total:
                return None
            if self.total < len(self.data):
                samples = self.data[:self.total].copy()
            else:
                samples = np.concatenate((self.data[self.position:], self.data[:self.position]))
            return samples, self.total


def list_devices() -> None:
    import sounddevice as sd

    default = sd.default.device[0]
    print("Microphone IDs (* = default input):")
    for index, device in enumerate(sd.query_devices()):
        if device["max_input_channels"]:
            api = sd.query_hostapis(device["hostapi"])["name"]
            print(f"{'*' if index == default else ' '} {index}: {device['name']} [{api}]")


def run_live(diarizer: Diarizer | None, device: int | None, window: int, step: int) -> None:
    import sounddevice as sd

    audio = RollingAudio(window)
    capture_error: str | None = None
    last_capture = time.monotonic()

    def capture(indata, frames, time_info, status):
        nonlocal capture_error, last_capture
        del frames, time_info
        if status:
            capture_error = str(status)
            raise sd.CallbackAbort
        audio.append(indata[:, 0])
        last_capture = time.monotonic()

    with sd.InputStream(
        samplerate=DEFAULT_SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=DEFAULT_SAMPLE_RATE // 10,
        device=device,
        callback=capture,
    ) as stream:
        if diarizer is None:
            print("Microphone check: 10 seconds, 16 kHz mono; speak or play audio near the boom. "
                  "No models/token needed. Ctrl+C to stop.", flush=True)
        else:
            print(f"Listening: 16 kHz mono; {window / DEFAULT_SAMPLE_RATE:g}s window, "
                  f"{step / DEFAULT_SAMPLE_RATE:g}s interval; exactly 2 speakers. Ctrl+C to stop.", flush=True)
            if diarizer.holdback:
                print(f"Finalization holdback: {diarizer.holdback:g}s; "
                      "Ctrl+C finalizes buffered audio before closing.", flush=True)
        next_sample = window
        try:
            while True:
                if capture_error or not stream.active:
                    raise RuntimeError(f"Microphone stream stopped: {capture_error or 'device disconnected'}")
                if time.monotonic() - last_capture > 5:
                    raise RuntimeError("Microphone stopped delivering audio. Check the selected input device.")
                snapshot = audio.snapshot(next_sample)
                if snapshot is None:
                    time.sleep(0.05)
                    continue
                samples, total = snapshot
                if diarizer is None:
                    rms = float(np.sqrt(np.mean(samples ** 2)))
                    peak = float(np.max(np.abs(samples)))
                    level = 20 * np.log10(max(rms, 1e-12))
                    peak_level = 20 * np.log10(max(peak, 1e-12))
                    hint = " -- nearly silent; check headset mute, boom, and input volume" if peak_level < -65 else ""
                    print(f"Input {total / DEFAULT_SAMPLE_RATE:.1f}s: RMS {level:.0f} dBFS, "
                          f"peak {peak_level:.0f} dBFS{hint}", flush=True)
                    if total >= 10 * DEFAULT_SAMPLE_RATE:
                        print("Microphone check complete. Microphone closed.", flush=True)
                        return
                else:
                    diarizer.process(samples, (total - len(samples)) / DEFAULT_SAMPLE_RATE)
                next_sample = total + step
        except KeyboardInterrupt:
            if diarizer is not None and diarizer.holdback:
                stream.stop()
                snapshot = audio.snapshot(1)
                if snapshot is not None:
                    samples, total = snapshot
                    print("Finalizing buffered audio...", flush=True)
                    diarizer.process(samples, (total - len(samples)) / DEFAULT_SAMPLE_RATE, final=True)
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-devices", action="store_true", help="List microphone IDs; no model/token needed.")
    parser.add_argument("--device", type=int, help="Input device ID (default: system microphone).")
    parser.add_argument("--check-mic", action="store_true",
                        help="Show microphone levels for 10 seconds without loading models or using HF_TOKEN.")
    parser.add_argument("--window", type=float, default=6.0, help="Analysis window seconds (default: 6).")
    parser.add_argument("--step", type=float, default=3.0, help="Analysis interval seconds (default: 3).")
    parser.add_argument("--holdback", type=float, default=0.0,
                        help="Wait for more context before finalizing labels, in seconds (default: 0).")
    parser.add_argument("--no-transcript", action="store_true", help="Print speaker labels only; do not load Whisper.")
    parser.add_argument("--stt-model", default=DEFAULT_STT_MODEL, help="Whisper model (default: tiny.en; base.en for better English accuracy).")
    parser.add_argument("--language", default="en", help="Speech language code (default: en; use auto to detect).")
    parser.add_argument("--nurse-patient", action="store_true",
                        help="Learn two voices without setup; estimate English nurse/patient roles.")
    parser.add_argument("--track-speakers", action="store_true",
                        help="Keep session voice IDs across windows without estimating roles.")
    parser.add_argument("--inspect", action="store_true",
                        help="Print speaker turns, independent STT text, and input levels; omit roles.")
    args = parser.parse_args(argv)
    if not (0 < args.step <= args.window < float("inf")):
        parser.error("Require 0 < --step <= --window, with finite durations.")
    if not 0 <= args.holdback <= args.window - args.step:
        parser.error("Require 0 <= --holdback <= --window minus --step so deferred audio stays in the buffer.")
    window, step = (round(value * DEFAULT_SAMPLE_RATE) for value in (args.window, args.step))
    if not step:
        parser.error("--step must be at least one sample.")
    if not args.no_transcript and args.stt_model.endswith(".en") and args.language not in ("en", "auto"):
        parser.error("For other languages, use a multilingual model, e.g. --stt-model tiny --language pt.")
    if args.nurse_patient and (args.no_transcript or args.language != "en"):
        parser.error("--nurse-patient requires English transcription (--language en, without --no-transcript).")
    try:
        if args.list_devices:
            list_devices()
            return 0
        import sounddevice as sd
        # Reject an invalid microphone before importing/loading the model.
        sd.check_input_settings(device=args.device, channels=1, dtype="float32", samplerate=DEFAULT_SAMPLE_RATE)
        info = sd.query_devices(args.device, "input")
        print(f"Microphone: {info['name']}", flush=True)
        if args.check_mic:
            run_live(None, args.device, DEFAULT_SAMPLE_RATE, DEFAULT_SAMPLE_RATE)
            return 0
        diarizer = load_diarizer(None if args.no_transcript else args.stt_model,
                                None if args.language == "auto" else args.language,
                                nurse_patient=args.nurse_patient, holdback=args.holdback,
                                track_speakers=args.track_speakers, inspect=args.inspect)
        if args.inspect:
            print("Inspection: TURN shows speaker intervals and local model IDs; "
                  "STT shows recognized text independently. Roles are disabled.", flush=True)
        elif args.nurse_patient:
            print("Nurse/patient mode: learning two voices during the conversation; no setup. "
                  "Roles use English phrase rules and remain estimates; unclear voices/roles are UNKNOWN.",
                  flush=True)
        run_live(diarizer, args.device, window, step)
        return 0
    except KeyboardInterrupt:
        print("\nStopped. Microphone closed.", flush=True)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
