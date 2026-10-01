from __future__ import annotations

import unittest

from sim.voice_delivery import (
    DEFAULT_DELIVERY,
    DeliveryStreamParser,
    MAX_DELIVERY_HEADER_CHARS,
    UnsafeVoiceResponseError,
    parse_delivery_response,
    validate_spoken_text,
)


class VoiceDeliveryTests(unittest.TestCase):
    def test_fragmented_header_is_hidden_and_validated(self) -> None:
        visible: list[str] = []
        parser = DeliveryStreamParser(visible.append)

        parser.add_chunk("[[deliv")
        parser.add_chunk('ery]]{"emotion":"anxious","intensity":2,')
        parser.add_chunk('"pace":"slow"}[[/delivery]]\nI am hurting')
        parser.add_chunk(" quite a bit.")
        text, style = parser.finish()

        self.assertEqual(text, "I am hurting quite a bit.")
        self.assertEqual("".join(visible), "I am hurting quite a bit.")
        self.assertEqual(style.emotion, "anxious")
        self.assertEqual(style.intensity, 2)
        self.assertEqual(style.pace, "slow")

    def test_leading_whitespace_and_bom_before_header_are_hidden(self) -> None:
        visible: list[str] = []
        parser = DeliveryStreamParser(visible.append)

        parser.add_chunk('\ufeff \r\n\t[[delivery]]{"emotion":"tired",')
        parser.add_chunk('"intensity":2,"pace":"slow"}[[/delivery]]\n')
        parser.add_chunk("I need to rest.")
        text, style = parser.finish()

        self.assertEqual(text, "I need to rest.")
        self.assertEqual("".join(visible), "I need to rest.")
        self.assertEqual(style.emotion, "tired")

    def test_fragmented_leading_whitespace_does_not_resolve_as_plain_text(self) -> None:
        visible: list[str] = []
        parser = DeliveryStreamParser(visible.append)

        for chunk in ("\n", " ", "\ufeff", "[[del", "ivery]]"):
            parser.add_chunk(chunk)
        parser.add_chunk(
            '{"emotion":"anxious","intensity":1,"pace":"normal"}'
            "[[/delivery]]Patient text."
        )
        text, style = parser.finish()

        self.assertEqual(text, "Patient text.")
        self.assertEqual("".join(visible), "Patient text.")
        self.assertEqual(style.emotion, "anxious")

    def test_malformed_metadata_falls_back_to_neutral_delivery(self) -> None:
        text, style = parse_delivery_response(
            '[[delivery]]{"emotion":"dramatic","intensity":9,"pace":"slow"}'
            "[[/delivery]]\nPlease help me."
        )

        self.assertEqual(text, "Please help me.")
        self.assertEqual(style, DEFAULT_DELIVERY)

    def test_streamed_malformed_header_is_hidden(self) -> None:
        visible: list[str] = []
        parser = DeliveryStreamParser(visible.append)

        parser.add_chunk("\n[[delivery]]{not-json")
        parser.add_chunk("}[[/delivery]]\nPlease help me.")
        text, style = parser.finish()

        self.assertEqual("".join(visible), "Please help me.")
        self.assertEqual(text, "Please help me.")
        self.assertEqual(style, DEFAULT_DELIVERY)

    def test_oversized_header_is_discarded_without_leaking_or_growing_buffer(
        self,
    ) -> None:
        visible: list[str] = []
        parser = DeliveryStreamParser(visible.append)

        parser.add_chunk("\ufeff[[delivery]]" + "x" * MAX_DELIVERY_HEADER_CHARS)
        self.assertEqual(visible, [])
        self.assertLess(len(parser._buffer), len("[[/delivery]]"))

        parser.add_chunk("more metadata[[/del")
        parser.add_chunk("ivery]]\nSafe patient wording.")
        text, style = parser.finish()

        self.assertEqual("".join(visible), "Safe patient wording.")
        self.assertEqual(text, "Safe patient wording.")
        self.assertEqual(style, DEFAULT_DELIVERY)

    def test_plain_response_remains_visible(self) -> None:
        self.assertEqual(
            parse_delivery_response("  Please help me.  "),
            ("Please help me.", DEFAULT_DELIVERY),
        )

    def test_duplicate_trailing_close_tag_is_removed(self) -> None:
        text, style = parse_delivery_response(
            '[[delivery]]{"emotion":"in_pain","intensity":2,"pace":"slow"}'
            "[[/delivery]]\nPlease help me.[[/delivery]]"
        )

        self.assertEqual(text, "Please help me.")
        self.assertEqual(style.emotion, "in_pain")

    def test_markdown_wrapped_delivery_header_is_accepted(self) -> None:
        responses = (
            "```text\n"
            '[[delivery]]{"emotion":"anxious","intensity":2,"pace":"slow"}'
            "[[/delivery]]\nPlease stay with me.\n```",
            "```json\n"
            '[[delivery]]{"emotion":"anxious","intensity":2,"pace":"slow"}'
            "[[/delivery]]\n```\nPlease stay with me.",
        )

        for response in responses:
            with self.subTest(response=response):
                text, style = parse_delivery_response(response)
                self.assertEqual(text, "Please stay with me.")
                self.assertEqual(style.emotion, "anxious")

    def test_streamed_markdown_wrapper_does_not_expose_header(self) -> None:
        visible: list[str] = []
        parser = DeliveryStreamParser(visible.append)

        parser.add_chunk("```json\n[[deliv")
        parser.add_chunk(
            'ery]]{"emotion":"anxious","intensity":2,"pace":"slow"}'
            "[[/delivery]]\n```\nPlease stay with me."
        )
        text, style = parser.finish()

        self.assertEqual(visible, [])
        self.assertEqual(text, "Please stay with me.")
        self.assertEqual(style.emotion, "anxious")

    def test_structured_metadata_is_rejected_from_patient_wording(self) -> None:
        unsafe_responses = (
            '{"emotion":"anxious","intensity":2,"pace":"slow"}',
            "Emotion: anxious\nPlease help me.",
            "Intensity: 2\nPlease help me.",
            "Pace: slow\nPlease help me.",
            "Delivery: anxious\nPlease help me.",
            "Pain level: 8/10\nPlease help me.",
            "```json\n{}",
            "Please help me.[[/delivery]]",
        )

        for response in unsafe_responses:
            with self.subTest(response=response):
                with self.assertRaises(UnsafeVoiceResponseError):
                    parse_delivery_response(response)

    def test_repeated_or_unmatched_delivery_tags_are_rejected(self) -> None:
        responses = (
            '[[delivery]]{"emotion":"neutral","intensity":1,"pace":"normal"}'
            "[[/delivery]]\nPlease help.[[/delivery]][[/delivery]]",
            '[[delivery]]{"emotion":"neutral","intensity":1,"pace":"normal"}'
            "[[/delivery]]\n[[delivery]]Please help.",
        )

        for response in responses:
            with self.subTest(response=response):
                with self.assertRaises(UnsafeVoiceResponseError):
                    parse_delivery_response(response)

    def test_extra_delivery_metadata_key_falls_back_to_neutral(self) -> None:
        text, style = parse_delivery_response(
            '[[delivery]]{"emotion":"anxious","intensity":2,"pace":"slow",'
            '"pain_level":"8/10"}[[/delivery]]\nPlease help me.'
        )

        self.assertEqual(text, "Please help me.")
        self.assertEqual(style, DEFAULT_DELIVERY)

    def test_natural_pain_wording_remains_speakable(self) -> None:
        self.assertEqual(
            validate_spoken_text("My pain is 8 out of 10."),
            "My pain is 8 out of 10.",
        )

    def test_unclosed_header_is_rejected(self) -> None:
        with self.assertRaises(UnsafeVoiceResponseError):
            parse_delivery_response(
                '[[delivery]]{"emotion":"anxious"}\nPlease help me.'
            )


if __name__ == "__main__":
    unittest.main()
