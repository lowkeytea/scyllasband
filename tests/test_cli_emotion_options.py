from __future__ import annotations

import argparse
from types import SimpleNamespace
import unittest

from scyllasband.cli import (
    _add_synthesis_args,
    _parse_cli_args,
    _parse_group_lines,
    build_parser,
)
from scyllasband.planner import planner_options_from, prepare_render_chunks


class _PlanningRuntime:
    manifest = SimpleNamespace(
        audio=SimpleNamespace(sample_rate=24_000, latent_hop_length=512),
        controls={},
    )

    def resolve_language_for_voice(self, voice: str, language: str | None) -> str:
        return language or "en_us"

    def normalize_text(self, text: str, *, language: str, voice_id: str) -> str:
        return text


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

    def test_group_planning_preserves_line_affect_and_global_scale(self) -> None:
        args = _parse_cli_args(
            build_parser(),
            [
                "group-speak",
                "--emotion-scale",
                "2.5",
                "--text",
                "[ink:en_gb:calm=0.25,joy=0.5] Hello!",
            ],
        )
        rows = _parse_group_lines(
            args.text_flag,
            default_voice=args.voice,
            default_language=args.language,
            default_emotion=args.emotion,
            default_affect=args.affect,
            default_emotion_guidance=args.emotion_guidance,
        )
        opts = planner_options_from(args, no_preflight_chunks=True)

        chunks = prepare_render_chunks(_PlanningRuntime(), rows, opts)

        self.assertEqual(chunks[0]["affect"], "calm=0.25,joy=0.5")
        self.assertEqual(chunks[0]["affect_guidance_scale"], 2.5)


if __name__ == "__main__":
    unittest.main()
