from __future__ import annotations

import unittest

import numpy as np

from sim.diarization import (
    DiarizationConfig,
    DiarizationFrame,
    LocalSpeakerTurn,
    RecognizedWord,
    SpeakerTurn,
    UtteranceStatus,
    VoiceTracker,
    align_words_to_speakers,
    to_session_speaker_turns,
    to_session_words,
)


def frame(
    turns: list[tuple[float, float, str]],
    embeddings: tuple[tuple[float, ...], ...] | None,
    *,
    labels: tuple[str, ...] | None = None,
) -> DiarizationFrame:
    local_turns = tuple(LocalSpeakerTurn(start, end, speaker) for start, end, speaker in turns)
    return DiarizationFrame(
        speaker_turns=local_turns,
        exclusive_speaker_turns=local_turns,
        speaker_labels=labels or tuple(sorted({speaker for _, _, speaker in turns})),
        speaker_embeddings=embeddings,
    )


class VoiceTrackerTests(unittest.TestCase):
    def test_tracks_swapped_clusters_and_keeps_ambiguous_voices_unknown(self) -> None:
        tracker = VoiceTracker()
        duplicate = frame(
            [(0, 3, "LOCAL_A"), (3, 6, "LOCAL_B")],
            ((1, 0, 0), (1, 0.01, 0)),
        )
        self.assertEqual(
            tracker.assign(duplicate),
            {"LOCAL_A": "SPEAKER_00", "LOCAL_B": "SPEAKER_00"},
        )

        second_voice = frame([(0, 6, "LOCAL_A")], ((0, 1, 0),))
        self.assertEqual(tracker.assign(second_voice), {"LOCAL_A": "SPEAKER_01"})

        swapped = frame(
            [(0, 3, "LOCAL_A"), (3, 6, "LOCAL_B")],
            ((0, 1, 0), (1, 0, 0)),
        )
        self.assertEqual(
            tracker.assign(swapped),
            {"LOCAL_A": "SPEAKER_01", "LOCAL_B": "SPEAKER_00"},
        )
        ambiguous = frame([(0, 6, "LOCAL_A")], ((1, 1, 0),))
        self.assertEqual(tracker.assign(ambiguous), {"LOCAL_A": "SPEAKER_UNKNOWN"})
        self.assertEqual(tracker.profile_count, 2)

    def test_rejects_missing_malformed_and_invalid_embeddings(self) -> None:
        tracker = VoiceTracker()
        missing = frame([(0, 2, "LOCAL_A")], None)
        mismatched = frame(
            [(0, 2, "LOCAL_A"), (2, 4, "LOCAL_B")],
            ((1, 0),),
        )
        ragged = frame(
            [(0, 2, "LOCAL_A"), (2, 4, "LOCAL_B")],
            ((1, 0), (0,)),
        )
        invalid = frame([(0, 2, "LOCAL_A")], ((float("nan"), 0),))
        zero = frame([(0, 2, "LOCAL_A")], ((0, 0),))

        for value in (missing, mismatched, ragged, invalid, zero):
            with self.subTest(value=value):
                self.assertTrue(
                    all(speaker == "SPEAKER_UNKNOWN" for speaker in tracker.assign(value).values())
                )
        self.assertEqual(tracker.profile_count, 0)

    def test_uses_configuration_for_profile_seeding_capacity_and_updates(self) -> None:
        config = DiarizationConfig(
            max_voice_profiles=1,
            profile_seed_min_speech_seconds=2,
            profile_update_min_similarity=0.70,
            profile_update_weight=0.50,
        )
        tracker = VoiceTracker(config)

        brief = frame([(0, 1, "LOCAL_A")], ((1, 0),))
        self.assertEqual(tracker.assign(brief), {"LOCAL_A": "SPEAKER_UNKNOWN"})
        self.assertEqual(tracker.profile_count, 0)

        seed = frame([(0, 3, "LOCAL_A")], ((1, 0),))
        self.assertEqual(tracker.assign(seed), {"LOCAL_A": "SPEAKER_00"})
        before = tracker.profiles[0]

        strong_match = frame([(0, 3, "LOCAL_A")], ((0.8, 0.6),))
        self.assertEqual(tracker.assign(strong_match), {"LOCAL_A": "SPEAKER_00"})
        after = tracker.profiles[0]
        self.assertFalse(np.allclose(before, after))

        tracker.profiles[0][0] = -100
        self.assertFalse(np.isclose(tracker.profiles[0][0], -100))

        second_voice = frame([(0, 3, "LOCAL_A")], ((0, 1),))
        self.assertEqual(tracker.assign(second_voice), {"LOCAL_A": "SPEAKER_UNKNOWN"})
        self.assertEqual(tracker.profile_count, 1)


class AlignmentTests(unittest.TestCase):
    def test_converts_local_turns_and_words_to_the_session_clock(self) -> None:
        turns = to_session_speaker_turns(
            (
                LocalSpeakerTurn(0, 2, "LOCAL_A"),
                LocalSpeakerTurn(4, 8, "LOCAL_B"),
            ),
            {"LOCAL_A": "SPEAKER_00"},
            window_start=10,
            window_end=16,
        )
        words = to_session_words(
            (RecognizedWord(1, 2, "Hello"), RecognizedWord(5, 8, "tail")),
            window_start=10,
            window_end=16,
        )

        self.assertEqual(
            turns,
            (
                SpeakerTurn(10, 12, "SPEAKER_00", "LOCAL_A"),
                SpeakerTurn(14, 16, "SPEAKER_UNKNOWN", "LOCAL_B"),
            ),
        )
        self.assertEqual(words, (RecognizedWord(11, 12, "Hello"), RecognizedWord(15, 16, "tail")))

    def test_aligns_by_largest_overlap_groups_speakers_and_drops_unmatched_words(self) -> None:
        turns = (
            SpeakerTurn(0, 3, "SPEAKER_00", "LOCAL_A"),
            SpeakerTurn(3, 6, "SPEAKER_01", "LOCAL_B"),
        )
        words = (
            RecognizedWord(0.5, 1.0, "Good"),
            RecognizedWord(2.0, 2.8, "morning."),
            RecognizedWord(3.2, 4.0, "Hello."),
            RecognizedWord(6.2, 7.0, "unmatched"),
        )

        utterances = align_words_to_speakers(
            words,
            turns,
            status=UtteranceStatus.PROVISIONAL,
        )

        self.assertEqual(len(utterances), 2)
        self.assertEqual(
            (
                utterances[0].start,
                utterances[0].end,
                utterances[0].speaker_id,
                utterances[0].text,
                utterances[0].status,
            ),
            (0.5, 2.8, "SPEAKER_00", "Good morning.", UtteranceStatus.PROVISIONAL),
        )
        self.assertEqual(utterances[1].speaker_id, "SPEAKER_01")
        self.assertEqual(utterances[1].text, "Hello.")
        self.assertNotIn("unmatched", " ".join(item.text for item in utterances))

    def test_alignment_has_no_role_mapping_and_handles_no_turns(self) -> None:
        words = (RecognizedWord(0, 1, "Hello"),)
        self.assertEqual(
            align_words_to_speakers(words, (), status=UtteranceStatus.FINALIZED),
            (),
        )
        turn = SpeakerTurn(0, 1, "SPEAKER_00", "LOCAL_A")
        utterance = align_words_to_speakers(
            words,
            (turn,),
            status=UtteranceStatus.FINALIZED,
        )[0]
        self.assertEqual(utterance.speaker_id, "SPEAKER_00")
        self.assertFalse(hasattr(utterance, "role"))
        self.assertNotIn("NURSE", utterance.speaker_id)
        self.assertNotIn("PATIENT", utterance.speaker_id)


if __name__ == "__main__":
    unittest.main()
