from __future__ import annotations

import statistics
from typing import Any


LATENCY_SCHEMA_VERSION = 2

METRIC_DEFINITIONS: dict[str, dict[str, str]] = {
    "recording_wall_time_seconds": {
        "unit": "seconds",
        "meaning": (
            "Time from starting the recorder until it stops. This includes leading silence, "
            "speech, pauses, and the end-of-speech wait."
        ),
    },
    "captured_audio_duration_seconds": {
        "unit": "seconds",
        "meaning": (
            "Length of the audio retained for transcription, including pre-roll and trailing silence. "
            "Leading silence that was discarded is not included."
        ),
    },
    "detected_speech_duration_seconds": {
        "unit": "seconds",
        "meaning": "Total duration of audio frames classified as speech by voice activity detection.",
    },
    "end_of_speech_detection_delay_seconds": {
        "unit": "seconds",
        "meaning": (
            "Time between the last detected speech frame and automatic recording stop. "
            "This is normally near 0.7 seconds and is zero in manual-stop mode."
        ),
    },
    "recording_stop_reason": {
        "unit": "category",
        "meaning": "Why recording stopped: silence, manual, no_speech, or max_duration.",
    },
    "speech_to_text_processing_seconds": {
        "unit": "seconds",
        "meaning": "Time faster-whisper spent converting the captured audio into text.",
    },
    "speech_end_to_transcript_ready_seconds": {
        "unit": "seconds",
        "meaning": (
            "Time from the last detected speech frame until the transcript is ready. "
            "Includes endpoint detection, model preload waiting, and transcription."
        ),
    },
    "llm_time_to_first_text_seconds": {
        "unit": "seconds",
        "meaning": "Time from sending the Ollama request until its first non-empty text chunk arrives.",
    },
    "speech_end_to_first_text_seconds": {
        "unit": "seconds",
        "meaning": (
            "Time from the student's last detected speech frame until the first patient text arrives. "
            "Includes endpoint detection, transcription, and initial Ollama latency."
        ),
    },
    "llm_response_generation_seconds": {
        "unit": "seconds",
        "meaning": "Time Ollama takes to generate the complete patient response.",
    },
    "llm_prompt_characters": {
        "unit": "characters",
        "meaning": "Number of characters sent to Ollama, including scenario instructions and conversation history.",
    },
    "turn_start_to_text_complete_seconds": {
        "unit": "seconds",
        "meaning": "Time from starting the student's recording until the complete patient text is generated.",
    },
    "opening_start_to_text_complete_seconds": {
        "unit": "seconds",
        "meaning": "Time from starting the opening response until its complete text is generated.",
    },
    "speech_end_to_first_audio_seconds": {
        "unit": "seconds",
        "meaning": (
            "Main conversational latency: time from the student's last detected speech frame until "
            "patient audio begins. Lower is better."
        ),
    },
    "speech_end_to_first_tts_segment_seconds": {
        "unit": "seconds",
        "meaning": (
            "Time from the student's last detected speech frame until enough patient text is ready "
            "to start the first TTS request."
        ),
    },
    "opening_start_to_first_tts_segment_seconds": {
        "unit": "seconds",
        "meaning": "Time from starting the opening response until its first text segment is sent to TTS.",
    },
    "first_tts_segment_to_first_audio_seconds": {
        "unit": "seconds",
        "meaning": (
            "Time from submitting the first text segment to TTS until audio playback begins. "
            "This isolates Edge TTS network synthesis and audio decoding delay."
        ),
    },
    "speech_end_to_audio_end_seconds": {
        "unit": "seconds",
        "meaning": "Time from the student's last detected speech frame until patient audio finishes or is interrupted.",
    },
    "opening_start_to_first_audio_seconds": {
        "unit": "seconds",
        "meaning": "Time from starting the opening response until opening audio begins.",
    },
    "opening_start_to_audio_end_seconds": {
        "unit": "seconds",
        "meaning": "Time from starting the opening response until opening audio finishes or is interrupted.",
    },
    "audio_status": {
        "unit": "category",
        "meaning": "Audio outcome: completed, interrupted by the next turn, or failed/unavailable.",
    },
    "audio_segments_started": {
        "unit": "count",
        "meaning": "Number of synthesized speech segments that started playing.",
    },
}


def create_latency_report() -> dict[str, Any]:
    return {
        "schema_version": LATENCY_SCHEMA_VERSION,
        "about": (
            "Durations are in seconds. Student-turn response metrics use the last detected speech "
            "frame as their starting point unless their name says otherwise."
        ),
        "metric_definitions": METRIC_DEFINITIONS,
        "summary": {},
        "opening": {},
        "turns": [],
    }


def refresh_latency_summary(report: dict[str, Any]) -> None:
    turns = report.get("turns", [])
    turn_metrics = [turn.get("metrics", {}) for turn in turns]
    audio_statuses = [metrics.get("audio_status") for metrics in turn_metrics]

    report["summary"] = {
        "measured_student_turns": len(turn_metrics),
        "turns_where_audio_started": sum(
            metrics.get("speech_end_to_first_audio_seconds") is not None
            for metrics in turn_metrics
        ),
        "median_speech_end_to_first_audio_seconds": _median_metric(
            turn_metrics,
            "speech_end_to_first_audio_seconds",
        ),
        "median_first_tts_segment_to_first_audio_seconds": _median_metric(
            turn_metrics,
            "first_tts_segment_to_first_audio_seconds",
        ),
        "median_speech_end_to_first_text_seconds": _median_metric(
            turn_metrics,
            "speech_end_to_first_text_seconds",
        ),
        "median_speech_to_text_processing_seconds": _median_metric(
            turn_metrics,
            "speech_to_text_processing_seconds",
        ),
        "audio_outcomes": {
            "completed": audio_statuses.count("completed"),
            "interrupted": audio_statuses.count("interrupted"),
            "failed": audio_statuses.count("failed"),
        },
    }


def _median_metric(turn_metrics: list[dict[str, Any]], name: str) -> float | None:
    values = [
        float(metrics[name])
        for metrics in turn_metrics
        if isinstance(metrics.get(name), (int, float))
    ]
    return round(statistics.median(values), 3) if values else None
