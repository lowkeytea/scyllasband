import unittest

from scyllasband.validation.alignment import align_words, normalize_for_wer, word_error_rate


class ValidationAlignmentTests(unittest.TestCase):
    def test_punctuation_case_and_compound_boundaries_do_not_false_fail(self) -> None:
        self.assertEqual(
            normalize_for_wer("Read-out—TEST", "en_us"),
            "read out test",
        )
        self.assertEqual(word_error_rate("read-out", "read out", "en_us"), 0.0)
        alignment = align_words("readout", "read out", "en_us")
        self.assertGreater(alignment["strict_wer"], 0.0)
        self.assertEqual(alignment["format_tolerant_wer"], 0.0)

    def test_digit_and_spoken_number_formatting_match(self) -> None:
        self.assertEqual(word_error_rate("There are 5 parts.", "there are five parts", "en_us"), 0.0)
        self.assertEqual(word_error_rate("Hay 22 piezas.", "hay veintidós piezas", "es"), 0.0)

    def test_accents_and_vietnamese_tones_are_not_erased(self) -> None:
        self.assertGreater(word_error_rate("má", "ma", "vi"), 0.0)
        self.assertGreater(word_error_rate("él", "el", "es"), 0.0)


if __name__ == "__main__":
    unittest.main()

