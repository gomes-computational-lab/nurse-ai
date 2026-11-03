"""Live two-speaker Pyannote demo; see demos/diarization/README.md."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np

from sim.audio import DEFAULT_SAMPLE_RATE
from sim.diarization import (
    DiarizationConfig,
    PyannoteDiarizationProvider,
    RecognizedWord,
    UtteranceStatus,
    VoiceTracker,
    align_words_to_speakers,
    to_session_speaker_turns,
)
from sim.speech_to_text import DEFAULT_STT_MODEL

# Reading guide: main() selects options and loads models, then run_live() captures
# microphone samples into RollingAudio. Diarizer.process() analyzes a snapshot,
# matches speaker IDs, and joins speaker turns with Whisper word timestamps.
#
# Three separate questions are handled here:
#   1. Pyannote: when is each voice speaking in this audio window?
#   2. VoiceTracker: which session voice does each window-local voice resemble?
#   3. Whisper: what words were spoken, and when?
# No LLM runs here, and no decision about when an assistant should speak is made.
#
# Audio is mono float32 at DEFAULT_SAMPLE_RATE (16,000 samples per second).
# Model timestamps start at zero within each window; printed timestamps refer to
# the microphone session. A turn is (start_seconds, end_seconds, speaker_label).

# This is a pretrained pipeline, not a model trained or fine-tuned by this demo.
MODEL = "pyannote/speaker-diarization-community-1"


def timestamp(seconds: float) -> str:
    """Format session-relative seconds as MM:SS.mmm; clamp negatives to zero."""
    milliseconds = round(max(0.0, seconds) * 1000)
    minutes, remainder = divmod(milliseconds, 60_000)
    return f"{minutes:02d}:{remainder // 1000:02d}.{remainder % 1000:03d}"


class Diarizer:
    """Analyze snapshots while retaining only the state needed across windows.

    The pipeline and Whisper model are loaded once and reused. Voice
    profiles retain identity; emitted_until marks the finalized output time.
    Overlapping audio is analyzed again, but finalized output is not rewritten.
    """

    def __init__(self, pipeline, torch, whisper, inspect: bool = False):
        self.pipeline = pipeline
        self.torch = torch
        self.whisper = whisper
        self.inspect = inspect
        self.config = DiarizationConfig(finalization_delay_seconds=0)
        self.diarization_provider = PyannoteDiarizationProvider(
            pipeline,
            torch,
            model_name=self.config.pyannote_model_name,
            device_name="legacy-demo",
        )
        # Times below are session-relative seconds, unlike the sample counters
        # in RollingAudio. No full conversation audio is accumulated here.
        self.emitted_until = 0.0
        self.voices = VoiceTracker(self.config)

    def process(self, samples: np.ndarray, start: float) -> None:
        """Analyze one mono snapshot starting at session time ``start``."""
        end = start + len(samples) / DEFAULT_SAMPLE_RATE
        began = time.perf_counter()
        frame = self.diarization_provider.diarize(
            samples,
            sample_rate=self.config.sample_rate,
            min_speakers=self.config.min_speakers,
            max_speakers=self.config.max_speakers,
        )
        mapping = self.voices.assign(frame)
        # Exclusive turns avoid arbitrary word attribution during overlapping speech.
        turns = to_session_speaker_turns(
            frame.exclusive_speaker_turns,
            mapping,
            window_start=start,
            window_end=end,
        )
        if start > self.emitted_until + 0.001:
            # The ring buffer may overwrite audio while a slow inference runs.
            # Make any resulting gap in finalized output explicit to the user.
            print(f"Warning: inference fell behind; skipped "
                  f"{timestamp(self.emitted_until)}-{timestamp(start)}.", file=sys.stderr)
        if self.inspect:
            # Print acoustic turns even if Whisper returns no words. The local
            # ID exposes model groups before cross-window voice matching.
            for turn in turns:
                a, b = max(turn.start, self.emitted_until), min(turn.end, end)
                if b > a:
                    print(f"[TURN {timestamp(a)}-{timestamp(b)}] {turn.speaker_id} "
                          f"(model={turn.source_speaker_id})", flush=True)
            words = self.recognize(samples, start)
            if words:
                a = max(words[0][0], self.emitted_until)
                print(f"[STT {timestamp(a)}-{timestamp(words[-1][1])}] "
                      + " ".join(text for _, _, text in words), flush=True)
            elif end > max(start, self.emitted_until):
                print(f"[STT {timestamp(max(start, self.emitted_until))}-{timestamp(end)}] "
                      "(no text)", flush=True)
        elif turns:
            # Normal view joins recognized words to tracked session speakers.
            lines = self.transcribe(samples, start, turns)
            for line in lines:
                print(f"[{timestamp(line.start)}-{timestamp(line.end)}] "
                      f"{line.speaker_id}: {line.text}", flush=True)
            if not lines and any(min(turn.end, end) > max(turn.start, self.emitted_until)
                                 for turn in turns):
                # Speech turns and recognized words come from different models.
                # This can mean empty STT or no words passing time/turn alignment;
                # it does not mean the vocabulary has already appeared before.
                print("Speech detected, but no new words recognized in this window.",
                      file=sys.stderr, flush=True)
        elapsed = time.perf_counter() - began
        rms = float(np.sqrt(np.mean(samples ** 2)))
        # dBFS expresses signal level relative to full-scale amplitude 1.0.
        # RMS measures average energy; peak (inspection only) measures the largest
        # sample. The floor avoids log10(0) for a completely silent snapshot.
        level = 20 * np.log10(max(rms, 1e-12))
        if self.inspect:
            peak = 20 * np.log10(max(float(np.max(np.abs(samples))), 1e-12))
            details = f"; input RMS {level:.0f} dBFS, peak {peak:.0f} dBFS"
            details += f"; model={len({turn.speaker_id for turn in frame.exclusive_speaker_turns})}"
            known = {turn.speaker_id for turn in turns} - {"SPEAKER_UNKNOWN"}
            details += f", tracked={len(known)}, profiles={len(self.voices.profiles)}/2"
        else:
            details = f" (no speech; input {level:.0f} dBFS)" if not turns else ""
        print(f"Analyzed {timestamp(start)}-{timestamp(end)} in {elapsed:.2f}s"
              + details, file=sys.stdout if self.inspect else sys.stderr, flush=True)
        # Normal diagnostics use stderr so stdout can be saved as a transcript.
        # Keep the finalized time monotonic when snapshots overlap.
        self.emitted_until = max(self.emitted_until, end)

    def recognize(self, samples: np.ndarray, start: float):
        """Return (session_start, session_end, text) for newly eligible words.

        This path does not require speaker turns, so inspection can show words
        even where Pyannote misses speech. Whisper's own VAD and time filters
        still apply; independent STT is not unfiltered recognition.
        """
        # Fast decoding and no previous-text conditioning keep each audio window
        # independent. Whisper's VAD suppresses non-speech; word timestamps let us
        # attach words to speakers later instead of guessing from whole sentences.
        segments, _ = self.whisper.transcribe(
            samples,
            language="en",
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
                text = word.word.strip()
                if text:
                    words.append((a, b, text))
        return words

    def transcribe(self, samples: np.ndarray, start: float, turns):
        """Assign each eligible word to its most-overlapping speaker turn.

        Called only with nonempty turns. Consecutive words assigned to the same
        speaker are grouped for display, even across pauses. These printed lines
        are convenient output groups, not guaranteed complete sentences or turns.
        """
        words = tuple(
            RecognizedWord(max(a, self.emitted_until), b, text)
            for a, b, text in self.recognize(samples, start)
            if b > max(a, self.emitted_until)
        )
        return align_words_to_speakers(words, turns, status=UtteranceStatus.FINALIZED)


def load_diarizer(stt_model: str = DEFAULT_STT_MODEL, inspect: bool = False) -> Diarizer:
    """Load pretrained models once and assemble their per-session coordinator.

    Heavy dependencies are imported here so --help, --list-devices, and --check-mic
    do not load the neural models. Listening always includes English transcription.
    """
    import torch

    # Only the diarization pipeline uses CUDA here. Without CUDA, including on
    # macOS, it uses CPU; this demo does not select Apple's MPS GPU backend.
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
    # Read only HF_TOKEN, not the simulation's Ollama/TTS configuration. This is
    # a small KEY=VALUE reader rather than a complete dotenv parser.
    if not os.environ.get("HF_TOKEN"):
        env_file = Path(__file__).with_name(".env")
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8-sig").splitlines():
                key, separator, value = line.strip().partition("=")
                if separator and key.strip() == "HF_TOKEN":
                    os.environ["HF_TOKEN"] = value.strip().strip("\"'")
                    break
    token = os.environ.get("HF_TOKEN", "").strip()
    # These checks catch common setup mistakes locally. Actual access still
    # depends on Hugging Face permissions and acceptance of the model conditions.
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
        # Suppress only the optional decoder warning during import. Other model
        # warnings remain visible because they can help explain accuracy problems.
        warnings.filterwarnings("ignore", message=r"\s*torchcodec is not installed correctly.*")
        from pyannote.audio import Pipeline

    try:
        pipeline = Pipeline.from_pretrained(MODEL, token=token)
    except Exception as exc:
        # Loading errors may contain request details. Redact the supplied token
        # before putting the error into terminal output or a captured log.
        detail = str(exc).replace(token, "[REDACTED]")
        raise RuntimeError(f"Model loading failed ({type(exc).__name__}): {detail}\n"
                           "Check the demo requirements, internet access, "
                           "HF_TOKEN read access, and accepted conditions at "
                           "https://huggingface.co/" + MODEL) from None
    if pipeline is None:
        raise RuntimeError("Model access denied. Check HF_TOKEN and the model access agreement.")
    pipeline.to(device)
    # Reuse the app's cached CPU/int8 loader. Timestamp alignment stays
    # isolated here because the app's transcribe_audio returns plain text.
    from sim.speech_to_text import _load_model

    print(f"Transcription: {stt_model}, CPU int8, language=en "
          "(first run downloads Whisper weights)...", flush=True)
    whisper = _load_model(stt_model)
    # Reuse the application's loader, but keep timestamped recognition and
    # speaker alignment in this demo; the app's plain-text STT path is unchanged.
    return Diarizer(pipeline, torch, whisper, inspect=inspect)


class RollingAudio:
    """A bounded circular buffer shared by capture and the processing loop.

    The microphone callback writes new samples while model inference runs in the
    main thread. Only the most recent ``capacity`` samples are retained. A lock
    keeps writes and snapshot copies from observing partially updated storage.
    """

    def __init__(self, capacity: int):
        # capacity and position are sample counts, not seconds. position is the
        # next write location; total counts ALL samples captured since startup,
        # including samples already overwritten, so session time stays continuous.
        self.data = np.empty(capacity, dtype=np.float32)
        self.position = 0
        self.total = 0
        self.lock = threading.Lock()

    def append(self, samples: np.ndarray) -> None:
        """Write a capture block, replacing the oldest audio when storage fills."""
        with self.lock:
            count = len(samples)
            self.total += count
            if count >= len(self.data):
                # An unusually large callback can replace the whole buffer.
                # Its newest capacity samples are sufficient for the next window.
                self.data[:] = samples[-len(self.data):]
                self.position = 0
                return
            first = min(count, len(self.data) - self.position)
            # A write crossing the physical end continues at index zero. Modulo
            # moves the next write position back into the fixed storage range.
            self.data[self.position:self.position + first] = samples[:first]
            self.data[:count - first] = samples[first:]
            self.position = (self.position + count) % len(self.data)

    def snapshot(self, minimum_total: int) -> tuple[np.ndarray, int] | None:
        """Copy retained audio in time order once enough samples have arrived.

        Return (samples, total_captured) or None while waiting. The independent
        copy lets capture overwrite the ring safely while inference uses it.
        """
        with self.lock:
            if self.total < minimum_total:
                return None
            if self.total < len(self.data):
                # Before the buffer fills, its unused tail is uninitialized.
                samples = self.data[:self.total].copy()
            else:
                # Once full, the next write location is also the oldest sample.
                # Join the two physical slices to restore chronological order.
                samples = np.concatenate((self.data[self.position:], self.data[:self.position]))
            return samples, self.total


def list_devices() -> None:
    """List available input devices; output-only devices cannot be microphones."""
    import sounddevice as sd

    default = sd.default.device[0]
    print("Microphone IDs (* = default input):")
    for index, device in enumerate(sd.query_devices()):
        if device["max_input_channels"]:
            api = sd.query_hostapis(device["hostapi"])["name"]
            print(f"{'*' if index == default else ' '} {index}: {device['name']} [{api}]")


def run_live(diarizer: Diarizer | None, device: int | None, window: int, step: int) -> None:
    """Capture continuously and analyze snapshots, or run a ten-second mic check.

    window and step are integer SAMPLE counts, converted from CLI seconds in
    main(). diarizer=None selects the levels-only check without loading models.
    Ctrl+C exits the stream context and closes capture without a final model pass.
    """
    import sounddevice as sd

    audio = RollingAudio(window)
    capture_error: str | None = None
    last_capture = time.monotonic()

    def capture(indata, frames, time_info, status):
        # sounddevice invokes this callback on its capture thread. Keep model
        # inference and printing in the main loop so they do not block capture.
        # indata has shape [frames, channels]; only the requested mono channel
        # is stored. Callback failures are reported to the main loop via state.
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
        blocksize=DEFAULT_SAMPLE_RATE // 10,  # About 100 ms per callback.
        device=device,
        callback=capture,
    ) as stream:
        if diarizer is None:
            print("Microphone check: 10 seconds, 16 kHz mono; speak or play audio near the boom. "
                  "No models/token needed. Ctrl+C to stop.", flush=True)
        else:
            print(f"Listening: 16 kHz mono; {window / DEFAULT_SAMPLE_RATE:g}s window, "
                  f"{step / DEFAULT_SAMPLE_RATE:g}s interval; up to 2 speakers. Ctrl+C to stop.", flush=True)
        # Wait for the first full window before inference. In microphone-check
        # mode both window and step are one second, giving one levels update/sec.
        next_sample = window
        while True:
            if capture_error or not stream.active:
                raise RuntimeError(f"Microphone stream stopped: {capture_error or 'device disconnected'}")
            if time.monotonic() - last_capture > 5:
                # Distinguish a stalled stream from valid but silent audio.
                raise RuntimeError("Microphone stopped delivering audio. Check the selected input device.")
            snapshot = audio.snapshot(next_sample)
            if snapshot is None:
                # Short waits keep Ctrl+C responsive without a busy loop.
                time.sleep(0.05)
                continue
            samples, total = snapshot
            if diarizer is None:
                # Levels diagnose whether the selected input actually carries
                # audio. They do not establish whether speech is intelligible.
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
                # Derive the snapshot's session start from captured samples,
                # not wall-clock time spent loading models or doing inference.
                diarizer.process(samples, (total - len(samples)) / DEFAULT_SAMPLE_RATE)
            # Request at least step NEW samples beyond this snapshot. Capture
            # continues during inference, so updates may cover more than step
            # seconds and are not guaranteed to arrive on an exact schedule.
            next_sample = total + step


def main(argv: list[str] | None = None) -> int:
    """Validate CLI options, prepare the input/models, and run until stopped.

    main.py --diarization dispatches here after removing its mode-selection flag.
    Return 0 for completion/user stop and 1 for a runtime failure. argparse handles
    --help and invalid arguments with its own exit statuses before capture starts.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    # Device diagnostics come first so setup can be checked without model access.
    parser.add_argument("--list-devices", action="store_true", help="List microphone IDs; no model/token needed.")
    parser.add_argument("--device", type=int, help="Input device ID (default: system microphone).")
    parser.add_argument("--check-mic", action="store_true",
                        help="Show microphone levels for 10 seconds without loading models or using HF_TOKEN.")
    # window controls recent context; step controls new audio before each snapshot.
    parser.add_argument("--window", type=float, default=6.0, help="Analysis window seconds (default: 6).")
    parser.add_argument("--step", type=float, default=3.0, help="Analysis interval seconds (default: 3).")
    # English STT is always enabled; the model can still be selected for comparison.
    parser.add_argument("--stt-model", default=DEFAULT_STT_MODEL, help="Whisper model (default: tiny.en; base.en for better English accuracy).")
    # Inspection separates model outputs so missing STT is not mistaken for
    # missing speaker turns. Session voice tracking is always enabled.
    parser.add_argument("--inspect", action="store_true",
                        help="Print speaker turns, independent STT text, and input levels.")
    args = parser.parse_args(argv)
    if not (0 < args.step <= args.window < float("inf")):
        parser.error("Require 0 < --step <= --window, with finite durations.")
    window, step = (round(value * DEFAULT_SAMPLE_RATE) for value in (args.window, args.step))
    # The CLI uses seconds; the stream/buffer operate on whole samples.
    if not step:
        parser.error("--step must be at least one sample.")
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
            # One-second snapshots for ten seconds, without touching either model.
            run_live(None, args.device, DEFAULT_SAMPLE_RATE, DEFAULT_SAMPLE_RATE)
            return 0
        diarizer = load_diarizer(args.stt_model, inspect=args.inspect)
        if args.inspect:
            print("Inspection: TURN shows speaker intervals and local model IDs; "
                  "STT shows recognized text independently.", flush=True)
        # Models are ready before opening the long-running microphone stream.
        run_live(diarizer, args.device, window, step)
        return 0
    except KeyboardInterrupt:
        print("\nStopped. Microphone closed.", flush=True)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    # Preserve the same process exit status for direct and main.py launches.
    raise SystemExit(main())
