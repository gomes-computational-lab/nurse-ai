from __future__ import annotations

import unittest

from sim.voice_delivery import (
    DEFAULT_DELIVERY,
    DeliveryStreamParser,
    MAX_DELIVERY_HEADER_CHARS,
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

    def test_malformed_metadata_defaults_without_exposing_header(self) -> None:
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

        self.assertEqual(text, "Please help me.")
        self.assertEqual("".join(visible), "Please help me.")
        self.assertEqual(style, DEFAULT_DELIVERY)

    def test_oversized_header_is_discarded_without_leaking_or_growing_buffer(self) -> None:
        visible: list[str] = []
        parser = DeliveryStreamParser(visible.append)

        parser.add_chunk("\ufeff[[delivery]]" + "x" * MAX_DELIVERY_HEADER_CHARS)
        self.assertEqual(visible, [])
        self.assertLess(len(parser._buffer), len("[[/delivery]]"))

        parser.add_chunk("more metadata[[/del")
        parser.add_chunk("ivery]]\nSafe patient wording.")
        text, style = parser.finish()

        self.assertEqual(text, "Safe patient wording.")
        self.assertEqual("".join(visible), "Safe patient wording.")
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
