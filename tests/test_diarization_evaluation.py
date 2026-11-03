from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from demos.diarization.evaluate_transcript import evaluate, read_text


class DiarizationEvaluationTests(unittest.TestCase):
    def test_anonymous_ids_are_mapped_once_for_the_whole_transcript(self):
        reference = "Nurse: Hello there.\nPatient: Yes thanks.\nNurse: Goodbye."
        prediction = ("[00:00-00:01] SPEAKER_01 [ROLE_UNKNOWN]: Hello there.\n"
                      "[00:01-00:02] SPEAKER_00 [ROLE_UNKNOWN]: Yes thanks.\n"
                      "[00:02-00:03] SPEAKER_01 [ROLE_UNKNOWN]: Goodbye.")
        report = evaluate(reference, prediction)
        self.assertEqual(report["speaker_mapping"], {"SPEAKER_00": "PATIENT", "SPEAKER_01": "NURSE"})
        self.assertEqual(report["matched_word_speaker_accuracy"], 1)

    def test_a_merged_patient_is_not_hidden_by_the_dominant_nurse(self):
        report = evaluate("Nurse: Hello there today.\nPatient: Yes thanks.",
                          "[00:00-00:03] SPEAKER_00: Hello there today yes thanks.")
        self.assertEqual(report["by_speaker"]["NURSE"]["matched_word_speaker_accuracy"], 1)
        self.assertEqual(report["by_speaker"]["PATIENT"]["matched_word_speaker_accuracy"], 0)

    def test_unknown_words_count_against_speaker_accuracy(self):
        report = evaluate("Nurse: Hello there.\nPatient: Yes thanks.",
                          "[00:00-00:01] SPEAKER_00: Hello there.\n"
                          "[00:01-00:02] SPEAKER_UNKNOWN: Yes thanks.")
        self.assertEqual(report["by_speaker"]["PATIENT"]["unknown_speaker_words"], 2)
        self.assertEqual(report["matched_word_speaker_accuracy"], .5)

    def test_missing_patient_speech_reports_zero_coverage_and_no_accuracy(self):
        report = evaluate("Nurse: Hello there.\nPatient: Yes thanks.",
                          "[00:00-00:01] SPEAKER_00: Hello there.")
        self.assertEqual(report["by_speaker"]["PATIENT"]["lexical_coverage"], 0)
        self.assertIsNone(report["by_speaker"]["PATIENT"]["matched_word_speaker_accuracy"])

    def test_word_substitutions_reduce_coverage_without_relabelling_speakers(self):
        report = evaluate("Nurse: Hello there.\nPatient: Yes thanks.",
                          "[00:00-00:01] SPEAKER_01: Hello here.\n"
                          "[00:01-00:02] SPEAKER_00: Yes thanks.")
        self.assertEqual(report["lexical_coverage"], .75)
        self.assertEqual(report["matched_word_speaker_accuracy"], 1)

    def test_reads_windows_powershell_utf16_logs(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "output.txt"
            path.write_text("[00:00-00:01] SPEAKER_00: Hello.", encoding="utf-16")
            self.assertEqual(read_text(path), "[00:00-00:01] SPEAKER_00: Hello.")


if __name__ == "__main__":
    unittest.main()
