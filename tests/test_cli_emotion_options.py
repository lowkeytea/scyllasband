from __future__ import annotations

import argparse
import unittest

from scyllasband.cli import (
    _add_synthesis_args,
    _parse_cli_args,
    _parse_group_lines,
    build_parser,
)


class CliEmotionOptionsTest(unittest.TestCase):
    def parse(self, *arguments: str) -> argparse.Namespace:
        return _parse_cli_args(build_parser(), ["speak", *arguments])

    def test_new_emotion_names_feed_current_runtime_fields(self) -> None:
        args = self.parse(
            "--voice",
            "scylla",
            "--emotion",
            "joy=0.7,sarcasm=0.4",
            "--emotion-scale",
            "1.25",
            "Hello.",
        )
        self.assertIsNone(args.emotion)
        self.assertEqual(args.affect, "joy=0.7,sarcasm=0.4")
        self.assertEqual(args.affect_guidance_scale, 1.25)

    def test_old_affect_names_remain_compatible_aliases(self) -> None:
        args = self.parse(
            "--voice",
            "scylla",
            "--affect",
            "anger=0.75",
            "--affect-guidance-scale",
            "1.5",
            "Hello.",
        )
        self.assertEqual(args.affect, "anger=0.75")
        self.assertEqual(args.affect_guidance_scale, 1.5)

    def test_old_emotion_guidance_scale_spelling_no_longer_errors(self) -> None:
        args = self.parse(
            "--voice",
            "scylla",
            "--emotion",
            "anger",
            "--emotion-guidance-scale",
            "2.0",
            "Hello.",
        )
        self.assertEqual(args.affect, "anger")
        self.assertEqual(args.affect_guidance_scale, 2.0)

    def test_help_only_advertises_simplified_names(self) -> None:
        parser = argparse.ArgumentParser()
        _add_synthesis_args(parser)
        help_text = parser.format_help()
        self.assertIn("--emotion EMOTION", help_text)
        self.assertIn("--emotion-scale SCALE", help_text)
        self.assertNotIn("--affect", help_text)
        self.assertNotIn("--emotion-guidance", help_text)

    def test_group_tags_treat_bare_names_as_emotion_presets(self) -> None:
        rows = _parse_group_lines(
            "[scylla:en_us:joy] Bright line. [anger] Firm line.",
            default_voice=None,
            default_language=None,
        )
        self.assertEqual([row["affect"] for row in rows], ["joy", "anger"])
        self.assertEqual([row["emotion"] for row in rows], [None, None])


if __name__ == "__main__":
    unittest.main()
