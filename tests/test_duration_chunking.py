"""Long-form duration targets must preserve text, controls and v1 defaults."""
from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import contextlib
import io
import json
import tempfile
import unittest
from unittest.mock import patch

from scyllasband.cli import build_parser, main as cli_main
from scyllasband.delivery import DELIVERY_SCHEMA
from scyllasband.planner import (
    PlannerOptions, plan_text, planner_options_from, planner_options_for_runtime,
    preflight_split_overlong_chunks,
)
from scyllasband.runtime import SynthesisRequest
from scyllasband.streaming import render_text_records


class DurationChunkingTest(unittest.TestCase):
    def test_duration_budget_keeps_fitting_sentence_whole(self) -> None:
        sentence = (
            "Let the toxic pollen and wild spores claw down your windpipe and "
            "cast Hideous Laughter on your lungs."
        )
        text = sentence + " Stomp the mud. You are merely a haunted puddle of soup piloting a fragile skin suit."
        runtime = _DurationRuntime()
        baseline = plan_text(runtime, text, voice="scylla")
        limited = plan_text(
            runtime, text, voice="scylla", options=PlannerOptions(max_chunk_seconds=8),
        )
        self.assertEqual(len(baseline.chunks), 1)
        self.assertGreater(len(limited.chunks), 1)
        self.assertEqual(limited.chunks[0].text, sentence)
        self.assertEqual(" ".join(c.text for c in limited.chunks), text)
        self.assertTrue(all(c.target_duration_seconds <= 8 for c in limited.chunks))
        self.assertTrue(all(c.ends_sentence for c in limited.chunks))
        self.assertEqual(limited.chunks[1].boundary_before, "sentence_start")

    def test_duration_budget_uses_requested_delivery_and_speed(self) -> None:
        text = "We can wait for you here. We can leave again tomorrow."
        runtime = _DurationRuntime()
        def planned(**overrides):
            return plan_text(
                runtime, text, voice="scylla",
                options=PlannerOptions(max_chunk_seconds=3, **overrides),
            )
        self.assertGreater(len(planned().chunks), 1)
        self.assertEqual(len(planned(speed=2).chunks), 1)
        self.assertEqual(len(planned(delivery={"energy": 3}).chunks), 1)
        self.assertTrue(all(c.metadata["delivery"] == {"energy": 3}
                            for c in planned(delivery={"energy": 3}).chunks))

    def test_duration_budget_fallback_preserves_long_sentence_words(self) -> None:
        text = " ".join(["a longer sentence without punctuation"] * 5) + "."
        plan = plan_text(
            _DurationRuntime(), text, voice="scylla",
            options=PlannerOptions(max_chunk_seconds=4, min_chunk_chars=1),
        )
        self.assertEqual(" ".join(c.text for c in plan.chunks), text)
        self.assertGreater(len(plan.chunks), 1)
        self.assertTrue(all(c.target_duration_seconds <= 4 for c in plan.chunks))
        self.assertTrue(any(c.boundary_after == "chunk_continue" for c in plan.chunks))

    def test_duration_budget_unsplittable_text_is_reported(self) -> None:
        plan = plan_text(
            _DurationRuntime(), "A", voice="scylla",
            options=PlannerOptions(max_chunk_seconds=0.001),
        )
        self.assertEqual([c.text for c in plan.chunks], ["A"])
        self.assertTrue(plan.chunks[0].metadata["duration_budget_unresolved"])

    def test_duration_budget_streaming_matches_eager_and_adaptive(self) -> None:
        text = "We can wait for you here. We can leave again tomorrow."
        rendered = []
        for overrides in (
            {"lazy_preflight": False},
            {"lazy_preflight": True, "pipeline_preflight": False},
            {"lazy_preflight": True, "pipeline_preflight": True},
            {"adaptive_chunking": True, "adaptive_chunk_schedule": (220,)},
        ):
            runtime = _DurationRuntime()
            result = render_text_records(
                runtime, [{"voice": "scylla", "text": text}],
                options=PlannerOptions(max_chunk_seconds=3, **overrides),
                progress_stream=None,
            )
            rendered.append([r.text for r in runtime.requests])
            self.assertEqual(result[2]["chunk_max_seconds"], 3)
            self.assertTrue(all(c["preflight_predicted_latent_frames"] * 512 / 24000 <= 3
                                for c in result[2]["chunks"]))
        self.assertGreater(len(rendered[0]), 1)
        self.assertTrue(all(texts == rendered[0] for texts in rendered))

    def test_duration_budget_can_be_disabled_and_rejects_invalid_limits(self) -> None:
        text = "We can wait for you here. We can leave again tomorrow."
        for overrides in ({"max_chunk_seconds": 0}, {"no_preflight_chunks": True},
                          {"no_auto_split_overlong": True}):
            values = {"max_chunk_seconds": 3, **overrides}
            plan = plan_text(_DurationRuntime(), text, voice="scylla", options=values)
            self.assertEqual(len(plan.chunks), 1)
        for seconds in (-1, float("inf"), float("nan")):
            with self.assertRaisesRegex(ValueError, "max-chunk-seconds"):
                planner_options_from({"max_chunk_seconds": seconds})
        args = build_parser().parse_args([
            "group-speak", "--file", "input.txt", "--voice", "scylla",
            "--max-chunk-seconds", "8", "-o", "output.wav",
        ])
        self.assertEqual(planner_options_from(args).max_chunk_seconds, 8)

    def test_duration_budget_does_not_disable_hard_frame_guard(self) -> None:
        runtime = _DurationRuntime()
        text = " ".join(["A lengthy utterance continues."] * 10)
        chunks = preflight_split_overlong_chunks(
            runtime, [{"text": text, "voice": "scylla", "language": "en_us",
                       "boundary_before": "paragraph_start", "boundary_after": "paragraph_end"}],
            PlannerOptions(max_chunk_chars=500, min_chunk_chars=1, max_chunk_seconds=60),
        )
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(c["preflight_predicted_latent_frames"] <= 1024 for c in chunks))
        self.assertEqual(" ".join(c["text"] for c in chunks), text)

    def test_bundle_default_and_explicit_overrides(self) -> None:
        runtime = _DurationRuntime()
        self.assertIsNone(planner_options_from().max_chunk_seconds)
        self.assertEqual(planner_options_for_runtime(runtime).max_chunk_seconds, 0)
        runtime.manifest.controls["graph_input_contract"] = DELIVERY_SCHEMA
        self.assertEqual(planner_options_for_runtime(runtime).max_chunk_seconds, 8)
        for seconds in (0, 3, 12):
            self.assertEqual(
                planner_options_for_runtime(runtime, {"max_chunk_seconds": seconds}).max_chunk_seconds,
                seconds,
            )
        args = build_parser().parse_args([
            "group-speak", "--file", "input.txt", "--voice", "scylla", "-o", "out.wav",
        ])
        self.assertIsNone(args.max_chunk_seconds)
        self.assertEqual(planner_options_for_runtime(runtime, args).max_chunk_seconds, 8)
        args.max_chunk_seconds = 0
        self.assertEqual(planner_options_for_runtime(runtime, args).max_chunk_seconds, 0)

    def test_default_targets_measured_schema_only(self) -> None:
        text = "A reasonably long sentence. " * 6
        for schema in ("scyllasband_affect_v1", "scyllasband_affect_v3", DELIVERY_SCHEMA):
            runtime = _DurationRuntime()
            runtime.manifest.controls["graph_input_contract"] = schema
            default = plan_text(runtime, text, voice="scylla")
            off = plan_text(runtime, text, voice="scylla", max_chunk_seconds=0)
            if schema == DELIVERY_SCHEMA:
                self.assertGreater(len(default.chunks), len(off.chunks))
                self.assertTrue(all(c.target_duration_seconds <= 8 for c in default.chunks))
            else:
                self.assertEqual(default.to_dict(), off.to_dict())

    def test_adaptive_merges_recheck_duration_in_eager_and_lazy_modes(self) -> None:
        text = "We can wait for you here. We can leave again tomorrow. " * 8
        for lazy in (False, True):
            with self.subTest(lazy=lazy):
                runtime = _DurationRuntime()
                runtime.manifest.controls["graph_input_contract"] = DELIVERY_SCHEMA
                audio, rate, metadata = render_text_records(
                    runtime, [{"voice": "scylla", "text": text}],
                    options=PlannerOptions(
                        lazy_preflight=lazy, adaptive_chunking=True,
                        adaptive_chunk_schedule=(60, 220), adaptive_min_chunk_chars=1,
                        adaptive_min_chunks_per_stage=1, adaptive_buffer_ms=0,
                        adaptive_realtime_factor=0,
                    ), progress_stream=None,
                )
                self.assertEqual(metadata["chunk_max_seconds"], 8)
                self.assertTrue(any(c.get("adaptive_stage") == 1 for c in metadata["chunks"]))
                self.assertTrue(all(c["preflight_predicted_latent_frames"] * 512 / rate <= 8
                                    for c in metadata["chunks"]))
                self.assertEqual(" ".join(r.text for r in runtime.requests), text.strip())

    def test_cli_stream_resolves_model_default_for_final_metadata(self) -> None:
        for measured in (False, True):
            with self.subTest(measured=measured), tempfile.TemporaryDirectory() as tmp:
                runtime = _DurationRuntime()
                if measured:
                    runtime.manifest.controls["graph_input_contract"] = DELIVERY_SCHEMA
                metadata_path = Path(tmp) / "metadata.json"
                with (
                    patch("scyllasband.cli.ScyllasBandRuntime.from_bundle", return_value=runtime),
                    patch("scyllasband.cli.replacement_notice", return_value=None),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    code = cli_main([
                        "speak", "--stream", "--voice", "scylla", "--no-progress",
                        "--text", "We can wait for you here. " * 6,
                        "--metadata", str(metadata_path), "-o", str(Path(tmp) / "out.wav"),
                    ])
                self.assertEqual(code, 0)
                metadata = json.loads(metadata_path.read_text())
                self.assertEqual(metadata["chunk_max_seconds"], 8 if measured else 0)
                self.assertEqual(metadata["chunk_count"], len(runtime.requests))


class _FakeRuntime:
    def __init__(self) -> None:
        self.manifest = SimpleNamespace(
            audio=SimpleNamespace(
                sample_rate=24000,
                latent_hop_length=512,
            ),
            controls={
                "fixed_shapes": {
                    "g2p_text_tokens": 512,
                    "phone_frames": 512,
                    "latent_frames": 640,
                }
            },
        )
        self.requests: list[SynthesisRequest] = []

    def resolve_language_for_voice(self, voice_id: str, language: str | None = None) -> str:
        return language or "en_us"

    def normalize_text(self, text: str, *, language: str | None = None, voice_id: str | None = None) -> str:
        return " ".join(str(text).split())

    def estimate_latent_frames(self, request: SynthesisRequest) -> dict[str, int]:
        predicted = max(1, len(str(request.text or "")))
        return {
            "predicted_latent_frames": predicted,
            "fixed_latent_frames": 640,
        }

    def plan_records(self, records: list[dict[str, object]], *, options: object | None = None, source_kind: str = "records") -> object:
        from scyllasband.planner import plan_records

        return plan_records(self, records, options=options, source_kind=source_kind)

    def synthesize_stream(self, records: list[dict[str, object]], *, options: object | None = None, source_kind: str = "records", progress_stream: object | None = None) -> object:
        from scyllasband.streaming import synthesize_records_stream

        return synthesize_records_stream(
            self,
            records,
            options=options,
            source_kind=source_kind,
            progress_stream=progress_stream,
        )

    def synthesize(self, request: SynthesisRequest) -> SimpleNamespace:
        self.requests.append(request)
        index = len(self.requests)
        audio = [float(index) / 10.0] * 8
        return SimpleNamespace(
            audio=audio,
            sample_rate=24000,
            duration_seconds=len(audio) / 24000.0,
            metadata={"text": request.text},
            latents=f"latents-{index}",
        )


class _DurationRuntime(_FakeRuntime):
    def estimate_latent_frames(self, request: SynthesisRequest) -> dict[str, int]:
        # Model a slower neutral delivery, with independent speed and delivery inputs.
        delivery_rate = 2 if request.delivery and request.delivery.get("energy") == 3 else 1
        return {
            "predicted_latent_frames": max(1, round(len(request.text) * 3.7 / request.speed / delivery_rate)),
            "fixed_latent_frames": 1024,
        }


if __name__ == "__main__":
    unittest.main()
