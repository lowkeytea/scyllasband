from pathlib import Path
import tempfile
import unittest

from scyllasband.validation.corpus import DEFAULT_SUITE_PATH
from scyllasband.validation.jobs import ValidationJobError, build_run

from test_validation_jobs import write_bundle


class ValidationResumeTests(unittest.TestCase):
    def test_identical_plan_is_reused_but_changed_plan_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = write_bundle(root / "bundle", 2)
            output = root / "run"
            arguments = {
                "suite_path": DEFAULT_SUITE_PATH,
                "bundle_dir": bundle,
                "output_dir": output,
                "tier": "smoke",
                "voices": ["us_voice"],
                "languages": ["en_us"],
                "documents": ["en_challenge_01"],
            }
            first = build_run(**arguments)
            second = build_run(**arguments)
            self.assertEqual(first["run_contract_sha256"], second["run_contract_sha256"])
            self.assertTrue(second["reused"])
            with self.assertRaises(ValidationJobError):
                build_run(**arguments, steps=16)


if __name__ == "__main__":
    unittest.main()

