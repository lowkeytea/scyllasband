import json
from pathlib import Path
import tempfile
import unittest

from scyllasband.validation.audio import file_sha256, write_wav
from scyllasband.validation.corpus import canonical_json_bytes, sha256_bytes
from scyllasband.validation.report import build_report, publish_benchmark


class ValidationReportTests(unittest.TestCase):
    def test_static_report_contains_relative_audio_and_visible_asr_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job_id = "fixture__voice__en_us__neutral_0__abc"
            job = {
                "job_id": job_id,
                "voice": "voice",
                "language": "en_us",
                "condition": "neutral_0",
                "ratings": {},
                "affect": {"calm": 0.0},
                "document_id": "fixture",
                "document_kind": "challenge",
                "training_overlap": "fully_held_out",
                "text": "Five clear words are spoken.",
                "asr_reference_text": "Five clear words are spoken.",
                "challenge_spans": [],
            }
            job["job_contract_sha256"] = sha256_bytes(canonical_json_bytes({
                key: value for key, value in job.items()
                if key not in {"job_id", "job_contract_sha256"}
            }))
            run = {
                "schema_version": 1,
                "model_name": "model",
                "model_version": "1",
                "job_count": 1,
            }
            run["run_contract_sha256"] = sha256_bytes(canonical_json_bytes(run))
            (root / "run.json").write_text(json.dumps(run), encoding="utf-8")
            (root / "jobs.jsonl").write_text(json.dumps(job) + "\n", encoding="utf-8")
            render_root = root / "renders" / "onnx" / job_id
            audio_path = write_wav(render_root / "audio.wav", [0.0, 0.1, -0.1] * 100, 24000)
            (render_root / "metadata.json").write_text(json.dumps({
                "state": "complete",
                "job_id": job_id,
                "job_contract_sha256": job["job_contract_sha256"],
                "audio_sha256": file_sha256(audio_path),
                "audio_metrics": {"finite": True, "hard_clipped_sample_count": 0},
                "duration_seconds": 0.0125,
                "elapsed_seconds": 0.01,
                "realtime_factor": 0.8,
                "chunks": [{"index": 0}],
            }), encoding="utf-8")
            asr_root = root / "asr" / "onnx"
            asr_root.mkdir(parents=True)
            (asr_root / f"{job_id}.json").write_text(json.dumps({
                "final": {
                    "hypothesis": "Five clear words are spoken.",
                    "cer": 0.0,
                    "alignment": {
                        "strict_wer": 0.0,
                        "format_tolerant_wer": 0.0,
                        "reference_words": 5,
                        "errors": 0,
                        "operations": [],
                    },
                },
                "chunk_aggregate": {"strict_wer": 0.0},
            }), encoding="utf-8")

            result = build_report(root)
            self.assertEqual(result["rows"], 1)
            html = (root / "report" / "index.html").read_text(encoding="utf-8")
            self.assertIn(f"../renders/onnx/{job_id}/audio.wav", html)
            self.assertNotIn("<select", html)
            self.assertNotIn("scyllasband_validation_review_labels.json", html)
            self.assertIn('<p class="field-label">Script</p>', html)
            self.assertIn('<p class="field-label">ASR transcript</p>', html)
            self.assertIn('<p class="field-label">ASR diff</p>', html)
            self.assertIn("Scylla's Band<br>Voice Benchmark", html)
            self.assertIn("WER by voice", html)
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["aggregates"]["backend"][0]["micro_wer"], 0.0)

            published = root / "public" / "v1"
            pages = root / "pages" / "benchmark" / "v1"
            result = publish_benchmark(
                root,
                published,
                pages_dir=pages,
                audio_format="wav",
            )
            self.assertTrue(result["complete"])
            self.assertEqual(result["published_jobs"], 1)
            public_html = (published / "index.html").read_text(encoding="utf-8")
            self.assertIn(f"audio/onnx/{job_id}.wav", public_html)
            self.assertNotIn(str(root), public_html)
            self.assertNotIn("<select", public_html)
            self.assertTrue((pages / "audio" / "onnx" / f"{job_id}.wav").is_file())
            benchmark = json.loads((published / "benchmark.json").read_text(encoding="utf-8"))
            self.assertEqual(benchmark["model_version"], "1")
            self.assertTrue(benchmark["complete"])


if __name__ == "__main__":
    unittest.main()
