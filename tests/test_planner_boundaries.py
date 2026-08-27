from __future__ import annotations

import unittest

from scyllasband.planner import split_text_chunk_records


class PlannerBoundaryTest(unittest.TestCase):
    def test_oversized_sentence_prefers_existing_clause_pause(self) -> None:
        text = (
            "His band, Acorn Apocalypse, had been selling out shows since March "
            "fifteenth, two thousand and twenty four, when they headlined the "
            "infamous Hollow Log Festival and grossed forty seven thousand five "
            "hundred dollars in merch alone."
        )

        records = split_text_chunk_records(
            text,
            max_chars=220,
            min_chars=48,
        )

        self.assertEqual(len(records), 2)
        self.assertTrue(records[0]["text"].endswith("twenty four,"))
        self.assertEqual(records[0]["boundary_after"], "clause_continue")
        self.assertEqual(records[1]["boundary_before"], "clause_continue")
        self.assertEqual(
            " ".join(str(record["text"]) for record in records),
            text,
        )

    def test_unpunctuated_oversized_sentence_retains_artificial_boundary(self) -> None:
        records = split_text_chunk_records(
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda",
            max_chars=32,
            min_chars=8,
        )

        self.assertGreater(len(records), 1)
        self.assertEqual(records[0]["boundary_after"], "chunk_continue")
        self.assertEqual(records[1]["boundary_before"], "chunk_continue")


if __name__ == "__main__":
    unittest.main()
