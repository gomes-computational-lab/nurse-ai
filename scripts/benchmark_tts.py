from __future__ import annotations

import argparse
import json
import resource
import sys
import time
import wave
from pathlib import Path

from sim.expressive_tts import TTSConfig, TTSService
from sim.voice_delivery import DeliveryStyle


SAMPLES = (
    ("neutral", "Could you help me sit up a little?"),
    ("anxious", "I'm worried this pain means something went wrong."),
    ("fearful", "Please don't leave yet; I'm afraid I can't breathe deeply."),
    ("confused", "I'm not sure which breathing device you mean."),
    ("relieved", "That makes sense, and I feel a little better now."),
    ("tired", "I can try again, but I feel very tired."),
    ("in_pain", "It hurts badly when I cough or take a deep breath."),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark and prepare blinded local TTS samples."
    )
    parser.add_argument(
        "--provider", choices=("zonos2", "chatterbox_nano"), required=True
    )
    parser.add_argument("--voice", default="default")
    parser.add_argument("--zonos2-url", default="http://localhost:1919")
    parser.add_argument("--voice-catalog", default="voices")
    parser.add_argument("--output", type=Path, default=Path("output/tts_benchmark"))
    args = parser.parse_args()

    service = TTSService(
        TTSConfig(
            provider=args.provider,
            voice=args.voice,
            zonos2_url=args.zonos2_url,
            voice_catalog_dir=args.voice_catalog,
        )
    )
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    key = {}

    for index, (emotion, text) in enumerate(SAMPLES, start=1):
        started = time.perf_counter()
        audio = service.synthesize(
            text, DeliveryStyle(emotion=emotion, intensity=2, pace="normal")
        )
        elapsed = time.perf_counter() - started
        path = args.output / f"sample-{index:02d}.wav"
        path.write_bytes(audio.data)
        duration = _wav_duration(path)
        results.append(
            {
                "sample": path.name,
                "provider": audio.provider,
                "synthesis_seconds": round(elapsed, 3),
                "audio_seconds": round(duration, 3),
                "real_time_factor": round(elapsed / duration, 3) if duration else None,
            }
        )
        key[path.name] = {"emotion": emotion, "text": text}

    report = {
        "first_response_seconds": results[0]["synthesis_seconds"],
        "subsequent_response_median_seconds": _median(
            [item["synthesis_seconds"] for item in results[1:]]
        ),
        "maximum_resident_memory_mb": round(_maximum_resident_memory_mb(), 1),
        "samples": results,
    }
    (args.output / "benchmark.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (args.output / "listening-key.json").write_text(
        json.dumps(key, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[middle], 3)
    return round((ordered[middle - 1] + ordered[middle]) / 2, 3)


def _maximum_resident_memory_mb() -> float:
    resident = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return resident / (1024 * 1024) if sys.platform == "darwin" else resident / 1024


if __name__ == "__main__":
    main()
