from __future__ import annotations

import unittest

from scyllasband.litert import LiteRTRunner, _apply_punctuation_duration_floors
from scyllasband.runtime import SynthesisRequest


class ScyllasBandPunctuationRuntimeTest(unittest.TestCase):
    def _runner(self) -> LiteRTRunner:
        runner = object.__new__(LiteRTRunner)
        runner.g2p_segment_config = {
            "input_granularity": "phrase",
            "chunk_max_chars": 140,
            "drop_bracketed_notes": True,
        }
        runner.phone_to_id = {
            "<sil>": 0,
            "HH": 1,
            "<pause_comma>": 2,
            "<pause_semicolon>": 3,
            "<pause_colon>": 4,
            "<ellipsis>": 5,
            "<end_stmt>": 6,
            "<end_question>": 7,
            "<end_exclaim>": 8,
        }
        runner.g2p_config = {}
        runner.g2p_punctuation_token_remap = {
            "<ellipsis>": "<sil>",
            "<end_exclaim>": "<sil>",
            "<end_question>": "<sil>",
        }
        runner.g2p_punctuation_token_remap_scope = "continuation_only"
        runner.punctuation_silence_target = "explicit_silence"
        runner._g2p_language = lambda language: language
        runner._predict_g2p_segment = lambda segment, language: {
            "phones": ["HH"],
            "confidence": 1.0,
            "prediction_text": "HH",
        }
        return runner

    def test_continuation_remap_preserves_source_semantics(self) -> None:
        result = self._runner()._phonemize_uncached(
            "One? Two!",
            language="en_us",
        )

        self.assertEqual(
            result["phones"],
            ["<sil>", "HH", "<sil>", "HH", "<end_exclaim>", "<sil>"],
        )
        self.assertEqual(
            result["g2p_source_boundary_tokens"],
            ["<end_question>", "<end_exclaim>"],
        )
        self.assertEqual(
            result["g2p_punctuation_events"],
            [
                {
                    "segment_index": 0,
                    "phone_index": 2,
                    "source_phone": "<end_question>",
                    "emitted_phone": "<sil>",
                    "remapped": True,
                    "has_following_segment": True,
                },
                {
                    "segment_index": 1,
                    "phone_index": 4,
                    "source_phone": "<end_exclaim>",
                    "emitted_phone": "<end_exclaim>",
                    "remapped": False,
                    "has_following_segment": False,
                },
            ],
        )

    def test_colon_and_semicolon_keep_typed_tokens(self) -> None:
        result = self._runner()._phonemize_uncached(
            "One: Two;",
            language="en_us",
        )

        events = result["g2p_punctuation_events"]
        self.assertEqual([event["source_phone"] for event in events], [
            "<pause_colon>",
            "<pause_semicolon>",
        ])
        self.assertEqual([event["emitted_phone"] for event in events], [
            "<pause_colon>",
            "<pause_semicolon>",
        ])

    def test_remapped_silence_uses_source_sentence_floor(self) -> None:
        audit: list[dict[str, object]] = []
        durations = _apply_punctuation_duration_floors(
            [2, 3, 2],
            phones=["HH", "<sil>", "W"],
            semantic_phones={1: "<end_question>"},
            sentence_frames=15,
            clause_frames=8,
            audit=audit,
        )

        self.assertEqual(durations, [2, 15, 2])
        self.assertEqual(audit[0]["source_phone"], "<end_question>")
        self.assertEqual(audit[0]["emitted_phone"], "<sil>")
        self.assertEqual(audit[0]["applied_frames"], 15)

    def test_terminal_floor_requires_a_following_chunk(self) -> None:
        common = {
            "phones": ["HH", "<sil>"],
            "semantic_phones": {1: "<end_question>"},
            "sentence_frames": 15,
            "clause_frames": 8,
        }
        self.assertEqual(
            _apply_punctuation_duration_floors([2, 3], **common),
            [2, 3],
        )
        self.assertEqual(
            _apply_punctuation_duration_floors([2, 3], floor_terminal=True, **common),
            [2, 15],
        )

    def test_direct_request_uses_public_pause_defaults(self) -> None:
        request = SynthesisRequest(text="One? Two.", voice_id="ariadne")

        self.assertEqual(request.min_sentence_pause_ms, 107.0)
        self.assertEqual(request.min_clause_pause_ms, 160.0)


if __name__ == "__main__":
    unittest.main()
