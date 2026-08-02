import json
from pathlib import Path
import tempfile
import unittest

from scyllasband.contract import AFFECT_AXIS_ORDERS
from scyllasband.validation.corpus import DEFAULT_SUITE_PATH
from scyllasband.validation.jobs import build_run, load_run


DURATION_INPUTS = [
    "phone_ids", "voice_id", "language_id", "affect_values",
    "boundary_before_id", "boundary_after_id", "phone_mask",
    "affect_condition_mask", "reference_style", "reference_prosody",
    "reference_mask", "reference_condition_mask",
]
VECTOR_INPUTS = [
    "noise", "time", "expanded_phone_ids", "voice_id", "language_id",
    "affect_values", "boundary_before_id", "boundary_after_id", "latent_mask",
    "affect_condition_mask", "reference_style", "reference_prosody",
    "reference_mask", "reference_condition_mask", "prefix_latents", "prefix_mask",
]


def write_bundle(root: Path, axis_version: int, *, model_version: str = "99") -> Path:
    root.mkdir(parents=True)
    for name in ("g2p", "duration", "vector", "vocoder"):
        (root / f"{name}.bin").write_bytes(name.encode("ascii"))
    axes = list(AFFECT_AXIS_ORDERS[axis_version])
    manifest = {
        "contract_version": "1.0.0",
        "model_name": "validation-test",
        "model_version": model_version,
        "architecture": "scyllasband-duration-flow",
        "audio": {
            "sample_rate": 24000, "hop_length": 256, "n_mels": 100,
            "latent_dim": 24, "latent_hop_length": 256,
        },
        "languages": ["en_us", "en_gb", "es", "it", "de", "fr", "vi"],
        "default_language": "en_us",
        "preferred_backends": ["onnx"],
        "voices": [
            {"id": "us_voice", "languages": ["en_us", "en_gb", "es", "it", "de", "fr", "vi"], "default_language": "en_us"},
            {"id": "gb_voice", "languages": ["en_us", "en_gb", "es", "it", "de", "fr", "vi"], "default_language": "en_gb"},
        ],
        "components": {
            "g2p": {"path": "g2p.bin", "inputs": [], "outputs": []},
            "duration_predictor": {"path": "duration.bin", "inputs": DURATION_INPUTS, "outputs": []},
            "vector_estimator": {"path": "vector.bin", "inputs": VECTOR_INPUTS, "outputs": []},
            "vocoder": {"path": "vocoder.bin", "inputs": [], "outputs": []},
        },
        "controls": {
            "graph_input_contract": f"scyllasband_affect_v{axis_version}",
            "affect": {
                "enabled": True,
                "axis_order_version": axis_version,
                "axes": axes,
                "dimension": 6,
                "range": [0.0, 1.0],
                "condition_mask_input": True,
                "all_zero_is_explicit": True,
                "presets": {"neutral": [0.0] * 6},
                "default_preset": "neutral",
            },
        },
        "assets": {},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


class ValidationJobTests(unittest.TestCase):
    def _run(self, axis_version: int) -> tuple[dict, list[dict]]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        bundle = write_bundle(root / "bundle", axis_version)
        output = root / "run"
        build_run(
            suite_path=DEFAULT_SUITE_PATH,
            bundle_dir=bundle,
            output_dir=output,
            tier="smoke",
        )
        return load_run(output)

    def test_axis_v1_runs_questioning_and_never_whisper(self) -> None:
        run, jobs = self._run(1)
        conditions = {job["condition"] for job in jobs}
        self.assertIn("questioning_2", conditions)
        self.assertNotIn("whisper_2", conditions)
        self.assertEqual(run["model_version"], "99")
        self.assertIn({"condition": "whisper_2", "reason": "unsupported_affect_axes", "axes": ["whisper"]}, run["skipped_conditions"])

    def test_axis_v2_runs_whisper_and_never_questioning(self) -> None:
        run, jobs = self._run(2)
        conditions = {job["condition"] for job in jobs}
        self.assertIn("whisper_2", conditions)
        self.assertNotIn("questioning_2", conditions)
        neutral = next(job for job in jobs if job["condition"] == "neutral_0")
        self.assertEqual(neutral["affect"], {axis: 0.0 for axis in AFFECT_AXIS_ORDERS[2]})

    def test_smoke_selects_both_english_dialects_without_crossing_them(self) -> None:
        _, jobs = self._run(2)
        pairs = {(job["voice"], job["language"]) for job in jobs}
        self.assertIn(("us_voice", "en_us"), pairs)
        self.assertNotIn(("us_voice", "en_gb"), pairs)
        self.assertIn(("gb_voice", "en_gb"), pairs)
        self.assertNotIn(("gb_voice", "en_us"), pairs)


if __name__ == "__main__":
    unittest.main()

