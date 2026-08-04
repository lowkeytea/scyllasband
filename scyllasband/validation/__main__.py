"""Command line entry point for public long-form model validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .asr import run_asr
from .corpus import DEFAULT_SUITE_PATH, validate_suite
from .jobs import build_run
from .render import import_backend_results, render_backend
from .report import build_report, publish_benchmark


def _values(items: Iterable[str] | None) -> list[str] | None:
    values = [
        part.strip()
        for item in items or ()
        for part in str(item).split(",")
        if part.strip()
    ]
    return values or None


def _filters(parser: argparse.ArgumentParser, *, include_jobs: bool = False) -> None:
    parser.add_argument("--voice", action="append", help="Voice ID; repeat or comma-separate")
    parser.add_argument("--language", action="append", help="Exact language ID")
    parser.add_argument("--condition", action="append", help="Condition ID")
    parser.add_argument("--document", action="append", help="Document ID")
    if include_jobs:
        parser.add_argument("--job-id", action="append", help="Exact frozen job ID")


def _selected(args: argparse.Namespace, name: str) -> list[str] | None:
    return _values(getattr(args, name, None))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scyllasband-validate",
        description="Manifest-driven multilingual long-form validation",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    corpus = commands.add_parser("corpus", help="Validate the frozen public text suite")
    corpus.add_argument("--suite", type=Path, default=DEFAULT_SUITE_PATH)

    plan = commands.add_parser("plan", help="Freeze jobs from a suite and bundle manifest")
    plan.add_argument("--suite", type=Path, default=DEFAULT_SUITE_PATH)
    plan.add_argument("--bundle", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--tier", choices=("smoke", "core", "full"), default="smoke")
    plan.add_argument("--steps", type=int, default=8)
    plan.add_argument("--sampler", default="heun")
    plan.add_argument("--seed", type=int, default=2027)
    plan.add_argument("--affect-guidance-scale", type=float, default=1.0)
    plan.add_argument("--max-chunk-chars", type=int, default=220)
    plan.add_argument("--min-chunk-chars", type=int, default=48)
    plan.add_argument("--min-sentence-pause-ms", type=float, default=320.0)
    plan.add_argument("--min-clause-pause-ms", type=float, default=160.0)
    _filters(plan)

    render = commands.add_parser("render", help="Render frozen jobs with a public backend")
    render.add_argument("--run", type=Path, required=True)
    render.add_argument("--backend", choices=("onnx", "litert", "coreml"), required=True)
    render.add_argument("--bundle", type=Path, required=True)
    render.add_argument("--overwrite-failed", action="store_true")
    render.add_argument("--fail-fast", action="store_true")
    render.add_argument("--no-hash-components", action="store_true")
    render.add_argument("--litert-accelerator", choices=("cpu", "gpu", "auto"), default="cpu")
    render.add_argument("--onnx-provider", action="append")
    render.add_argument("--onnx-intra-op-threads", type=int, default=0)
    render.add_argument("--onnx-inter-op-threads", type=int, default=0)
    _filters(render, include_jobs=True)

    importer = commands.add_parser(
        "import-results", help="Import matching external or PyTorch render artifacts"
    )
    importer.add_argument("--run", type=Path, required=True)
    importer.add_argument("--backend", required=True)
    importer.add_argument("--source", type=Path, required=True)
    importer.add_argument("--overwrite", action="store_true")

    asr = commands.add_parser("asr", help="Score completed renders with faster-whisper")
    asr.add_argument("--run", type=Path, required=True)
    asr.add_argument("--backend", action="append", help="Default: every rendered backend")
    asr.add_argument("--profile", choices=("smoke", "release"), default="smoke")
    asr.add_argument("--model", help="Override turbo (smoke) or large-v3 (release)")
    asr.add_argument("--device", default="cuda")
    asr.add_argument("--compute-type", default="float16")
    asr.add_argument("--beam-size", type=int, default=1)
    asr.add_argument("--workers", type=int, default=1)
    asr.add_argument("--scope", choices=("final", "chunks", "both"), default="both")
    asr.add_argument("--no-vad", action="store_true")
    asr.add_argument("--overwrite", action="store_true")
    _filters(asr, include_jobs=True)

    report = commands.add_parser("report", help="Build JSON, CSV, and static HTML results")
    report.add_argument("--run", type=Path, required=True)

    publish = commands.add_parser(
        "publish",
        help="Publish a compact static benchmark from a validation run",
    )
    publish.add_argument("--run", type=Path, required=True)
    publish.add_argument("--output", type=Path, required=True)
    publish.add_argument(
        "--pages-output",
        type=Path,
        help="Optional GitHub Pages destination, such as docs/benchmark/v1",
    )
    publish.add_argument("--audio-format", choices=("mp3", "wav"), default="mp3")
    publish.add_argument("--overwrite", action="store_true")
    return parser


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "corpus":
        return validate_suite(args.suite)
    if args.command == "plan":
        return build_run(
            suite_path=args.suite,
            bundle_dir=args.bundle,
            output_dir=args.output,
            tier=args.tier,
            voices=_selected(args, "voice"),
            languages=_selected(args, "language"),
            conditions=_selected(args, "condition"),
            documents=_selected(args, "document"),
            steps=args.steps,
            sampler=args.sampler,
            seed=args.seed,
            affect_guidance_scale=args.affect_guidance_scale,
            max_chunk_chars=args.max_chunk_chars,
            min_chunk_chars=args.min_chunk_chars,
            min_sentence_pause_ms=args.min_sentence_pause_ms,
            min_clause_pause_ms=args.min_clause_pause_ms,
        )
    if args.command == "render":
        return render_backend(
            run_dir=args.run,
            backend=args.backend,
            bundle_dir=args.bundle,
            voices=_selected(args, "voice"),
            languages=_selected(args, "language"),
            conditions=_selected(args, "condition"),
            documents=_selected(args, "document"),
            job_ids=_selected(args, "job_id"),
            overwrite_failed=args.overwrite_failed,
            fail_fast=args.fail_fast,
            hash_components=not args.no_hash_components,
            litert_accelerator=args.litert_accelerator,
            onnx_providers=_selected(args, "onnx_provider"),
            onnx_intra_op_num_threads=args.onnx_intra_op_threads,
            onnx_inter_op_num_threads=args.onnx_inter_op_threads,
        )
    if args.command == "import-results":
        return import_backend_results(
            run_dir=args.run,
            backend=args.backend,
            source_dir=args.source,
            overwrite=args.overwrite,
        )
    if args.command == "asr":
        model = args.model or ("turbo" if args.profile == "smoke" else "large-v3")
        return run_asr(
            run_dir=args.run,
            backends=_selected(args, "backend"),
            model=model,
            device=args.device,
            compute_type=args.compute_type,
            beam_size=args.beam_size,
            vad_filter=not args.no_vad,
            workers=args.workers,
            scope=args.scope,
            voices=_selected(args, "voice"),
            languages=_selected(args, "language"),
            conditions=_selected(args, "condition"),
            documents=_selected(args, "document"),
            job_ids=_selected(args, "job_id"),
            overwrite=args.overwrite,
        )
    if args.command == "report":
        return build_report(args.run)
    if args.command == "publish":
        return publish_benchmark(
            args.run,
            args.output,
            pages_dir=args.pages_output,
            audio_format=args.audio_format,
            overwrite=args.overwrite,
        )
    raise AssertionError(args.command)


def main(argv: list[str] | None = None) -> int:
    result = _run(_parser().parse_args(argv))
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 1 if int(result.get("failed", 0) or 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
