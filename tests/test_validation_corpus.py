import json
from pathlib import Path
import tempfile
import unittest

from scyllasband.validation.corpus import DEFAULT_SUITE_PATH, ValidationCorpusError, load_suite, validate_suite


class ValidationCorpusTests(unittest.TestCase):
    def test_public_suite_has_three_documents_per_language(self) -> None:
        summary = validate_suite(DEFAULT_SUITE_PATH)
        self.assertEqual(summary["documents"], 18)
        self.assertEqual(summary["document_kinds"], {
            "challenge": 6,
            "dialogue_punctuation": 6,
            "narrative": 6,
        })
        self.assertEqual(
            summary["languages"],
            ["de", "en_gb", "en_us", "es", "fr", "it", "vi"],
        )

    def test_missing_challenge_surface_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_text("A valid line.", encoding="utf-8")
            (root / "document.json").write_text(json.dumps({
                "schema_version": 1,
                "id": "bad",
                "kind": "challenge",
                "language": "en_us",
                "input_path": "input.txt",
                "training_overlap": "fully_held_out",
                "provenance": {"redistribution": "approved"},
                "challenge_spans": [{"surface": "missing", "tags": ["test"]}],
            }), encoding="utf-8")
            (root / "suite.json").write_text(json.dumps({
                "schema_version": 1,
                "id": "bad-suite",
                "conditions": [{"id": "neutral_0", "ratings": {}}],
                "documents": ["document.json"],
            }), encoding="utf-8")
            with self.assertRaises(ValidationCorpusError):
                load_suite(root / "suite.json")


if __name__ == "__main__":
    unittest.main()

