import unittest

from scyllasband.validation.asr import _chunk_reference


class ValidationASRReferenceTests(unittest.TestCase):
    def test_chunk_reference_prefers_local_spoken_text(self) -> None:
        job = {
            "spoken_replacements": [
                {"surface": "7:45", "spoken": "seven forty five"},
            ]
        }
        chunk = {
            "source_text": "The entire source contains several unrelated sentences.",
            "text": "At 7:45, begin.",
        }
        self.assertEqual(_chunk_reference(job, chunk), "At seven forty five, begin.")


if __name__ == "__main__":
    unittest.main()

