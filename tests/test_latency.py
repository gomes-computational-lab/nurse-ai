from __future__ import annotations

import unittest

from sim.latency import METRIC_DEFINITIONS, create_latency_report, refresh_latency_summary


class LatencyReportTests(unittest.TestCase):
    def test_report_contains_units_and_plain_language_definitions(self) -> None:
        report = create_latency_report()

        self.assertEqual(report["schema_version"], 2)
        self.assertIn("seconds", report["about"])
        self.assertEqual(
            report["metric_definitions"]["speech_end_to_first_audio_seconds"]["unit"],
            "seconds",
        )
        self.assertIn(
            "Main conversational latency",
            report["metric_definitions"]["speech_end_to_first_audio_seconds"]["meaning"],
        )

    def test_every_saved_turn_metric_has_a_definition(self) -> None:
        expected_metrics = {
            "recording_wall_time_seconds",
            "captured_audio_duration_seconds",
            "detected_speech_duration_seconds",
            "end_of_speech_detection_delay_seconds",
            "recording_stop_reason",
            "speech_to_text_processing_seconds",
            "speech_end_to_transcript_ready_seconds",
            "llm_time_to_first_text_seconds",
            "speech_end_to_first_text_seconds",
            "llm_response_generation_seconds",
            "llm_prompt_characters",
            "turn_start_to_text_complete_seconds",
            "speech_end_to_first_audio_seconds",
            "speech_end_to_first_tts_segment_seconds",
            "first_tts_segment_to_first_audio_seconds",
            "speech_end_to_audio_end_seconds",
            "audio_status",
            "audio_segments_started",
        }

        self.assertTrue(expected_metrics.issubset(METRIC_DEFINITIONS))

    def test_summary_reports_medians_and_audio_outcomes(self) -> None:
        report = create_latency_report()
        report["turns"] = [
            {
                "turn": 1,
                "metrics": {
                    "speech_end_to_first_audio_seconds": 2.0,
                    "first_tts_segment_to_first_audio_seconds": 0.8,
                    "speech_end_to_first_text_seconds": 1.0,
                    "speech_to_text_processing_seconds": 0.4,
                    "audio_status": "completed",
                },
            },
            {
                "turn": 2,
                "metrics": {
                    "speech_end_to_first_audio_seconds": 4.0,
                    "first_tts_segment_to_first_audio_seconds": 1.2,
                    "speech_end_to_first_text_seconds": 3.0,
                    "speech_to_text_processing_seconds": 0.8,
                    "audio_status": "interrupted",
                },
            },
            {
                "turn": 3,
                "metrics": {
                    "speech_end_to_first_audio_seconds": None,
                    "audio_status": "failed",
                },
            },
        ]

        refresh_latency_summary(report)

        self.assertEqual(report["summary"]["measured_student_turns"], 3)
        self.assertEqual(report["summary"]["turns_where_audio_started"], 2)
        self.assertEqual(
            report["summary"]["median_speech_end_to_first_audio_seconds"],
            3.0,
        )
        self.assertEqual(
            report["summary"]["median_speech_end_to_first_text_seconds"],
            2.0,
        )
        self.assertEqual(
            report["summary"]["median_first_tts_segment_to_first_audio_seconds"],
            1.0,
        )
        self.assertEqual(
            report["summary"]["median_speech_to_text_processing_seconds"],
            0.6,
        )
        self.assertEqual(
            report["summary"]["audio_outcomes"],
            {"completed": 1, "interrupted": 1, "failed": 1},
        )


if __name__ == "__main__":
    unittest.main()
