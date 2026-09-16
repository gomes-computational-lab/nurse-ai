from __future__ import annotations

import argparse
import asyncio
import importlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import wave


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VOICE_DIR = PROJECT_ROOT / "voices"
OUTPUT_PATH = VOICE_DIR / "child_female_8yo.wav"
EDGE_VOICE = "en-US-AnaNeural"
REFERENCE_TEXT = (
    "Mom, I don't really understand what's happening. Is everything going to be "
    "okay? I'm a little scared, and I just want to know when we can go home."
)
SAMPLE_RATE = 24_000


class VoiceGenerationError(RuntimeError):
    pass


def find_ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    raise VoiceGenerationError(
        "FFmpeg is required to create the Chatterbox reference WAV. Install it "
        "with `brew install ffmpeg` on macOS, `sudo apt install ffmpeg` on "
        "Ubuntu/Debian, or `winget install Gyan.FFmpeg` on Windows."
    )


async def generate_voice(output_path: Path = OUTPUT_PATH) -> float:
    try:
        edge_tts = importlib.import_module("edge_tts")
    except ImportError as exc:
        raise VoiceGenerationError(
            "edge-tts is not installed. Run `python -m pip install -r requirements.txt`."
        ) from exc

    ffmpeg = find_ffmpeg()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=".ruth_voice_", dir=output_path.parent
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        mp3_path = temporary_root / "ruth_edge.mp3"
        wav_path = temporary_root / output_path.name

        try:
            communicator = edge_tts.Communicate(REFERENCE_TEXT, EDGE_VOICE)
            await communicator.save(str(mp3_path))
        except Exception as exc:
            raise VoiceGenerationError(
                f"Edge TTS could not generate Ruth's reference audio: {exc}"
            ) from exc

        try:
            subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(mp3_path),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    str(SAMPLE_RATE),
                    "-c:a",
                    "pcm_s16le",
                    str(wav_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() or str(exc)
            raise VoiceGenerationError(
                f"FFmpeg could not convert Ruth's reference audio: {detail}"
            ) from exc

        duration = validate_reference_wav(wav_path)
        wav_path.replace(output_path)

    return duration


def validate_reference_wav(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as audio:
            channels = audio.getnchannels()
            sample_rate = audio.getframerate()
            sample_width = audio.getsampwidth()
            frame_count = audio.getnframes()
    except (OSError, wave.Error) as exc:
        raise VoiceGenerationError(f"Generated WAV is invalid: {exc}") from exc

    if channels != 1:
        raise VoiceGenerationError(
            "Generated WAV must contain exactly one audio channel."
        )
    if sample_rate != SAMPLE_RATE:
        raise VoiceGenerationError(
            f"Generated WAV must use {SAMPLE_RATE} Hz audio, not {sample_rate} Hz."
        )
    if sample_width != 2:
        raise VoiceGenerationError("Generated WAV must use 16-bit PCM audio.")
    if frame_count <= 0:
        raise VoiceGenerationError("Generated WAV contains no audio frames.")
    return frame_count / sample_rate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Ruth's synthetic Chatterbox reference voice."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Output WAV path. Default: {OUTPUT_PATH}",
    )
    args = parser.parse_args()

    try:
        duration = asyncio.run(generate_voice(args.output))
    except VoiceGenerationError as exc:
        raise SystemExit(f"Error: {exc}") from exc

    print(f"Created Ruth's reference voice: {args.output.resolve()}")
    print(f"Format: mono, {SAMPLE_RATE} Hz, 16-bit PCM WAV ({duration:.1f} seconds)")
    print("Use with Chatterbox voice ID: child_female_8yo")


if __name__ == "__main__":
    main()
