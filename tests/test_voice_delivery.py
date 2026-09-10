from __future__ import annotations

import unittest

from sim.voice_delivery import (
    DEFAULT_DELIVERY,
    DeliveryStreamParser,
    parse_delivery_response,
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

    def test_malformed_metadata_defaults_without_exposing_header(self) -> None:
        text, style = parse_delivery_response(
            '[[delivery]]{"emotion":"dramatic","intensity":9,"pace":"slow"}'
            "[[/delivery]]\nPlease help me."
        )

        self.assertEqual(text, "Please help me.")
        self.assertEqual(style, DEFAULT_DELIVERY)

    def test_plain_response_remains_visible(self) -> None:
        self.assertEqual(
            parse_delivery_response("  Please help me.  "),
            ("Please help me.", DEFAULT_DELIVERY),
        )

    def test_unclosed_header_is_never_exposed(self) -> None:
        text, style = parse_delivery_response(
            '[[delivery]]{"emotion":"anxious"}\nPlease help me.'
        )

        self.assertEqual(text, "Please help me.")
        self.assertEqual(style, DEFAULT_DELIVERY)


if __name__ == "__main__":
    unittest.main()
