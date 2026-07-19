#!/usr/bin/env python3
"""Run the Scylla's Band native LiteRT CPU/GPU validation gate.

The gate combines the release-style native synthesis benchmark with lower-level
split-vector component probes. It is meant for local pre-commit or pre-upload
checks after LiteRT export/runtime changes:

1. Verify staged LiteRT runtime and GPU accelerator libraries are present.
2. Run one CPU and one GPU native synthesis and apply latency/policy gates.
3. Run the split vector prefix graph on CPU and GPU and compare output drift.
4. Run the split vector tail graph on CPU, matching the current runtime policy.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent


def _native_library_filename(platform_name: str | None = None, os_name: str | None = None) -> str:
    platform_name = sys.platform if platform_name is None else platform_name
    os_name = os.name if os_name is None else os_name
    if platform_name == "darwin":
        return "libscyllasband_native.dylib"
    if os_name == "nt" or platform_name.startswith("win"):
        return "scyllasband_native.dll"
    return "libscyllasband_native.so"


def _native_tool_filename(platform_name: str | None = None) -> str:
    platform_name = sys.platform if platform_name is None else platform_name
    return "scyllasband_native_speak.exe" if platform_name.startswith("win") else "scyllasband_native_speak"


DEFAULT_TOOL = SCRIPT_DIR.parents[0] / "build" / _native_tool_filename()
DEFAULT_LIBRARY = SCRIPT_DIR.parents[0] / "build" / _native_library_filename()
DEFAULT_BUNDLE = SCRIPT_DIR.parents[1] / "models" / "litert"
DEFAULT_LITERT_STAGE_ROOT = SCRIPT_DIR.parents[0] / "third_party" / "litert"
DEFAULT_TEXT = "This is a Scylla's Band LiteRT split vector benchmark for CPU and GPU."
DEFAULT_SHORT_TEXT = "Hello."


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", type=Path, default=DEFAULT_TOOL)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--short-text", default=DEFAULT_SHORT_TEXT)
    parser.add_argument("--voice", default="scylla")
    parser.add_argument("--language", default="en_us")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--active-frames", type=int, default=176)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/scyllasband_native_litert_gpu_gate"))
    stage_group = parser.add_mutually_exclusive_group()
    stage_group.add_argument("--litert-lib-dir", type=Path, help="Explicit staged LiteRT lib directory")
    stage_group.add_argument("--litert-stage-root", type=Path, default=DEFAULT_LITERT_STAGE_ROOT, help="LiteRT staging root containing lib/<platform>")
    parser.add_argument("--litert-platform", help="Optional staged platform name or alias for --litert-stage-root")
    parser.add_argument("--preflight-only", action="store_true", help="Validate staged LiteRT runtime/GPU libraries and skip native synthesis/probes")
    parser.add_argument("--baseline-json", type=Path, help="Existing compact benchmark baseline for CPU regression checks")
    parser.add_argument("--baseline-out", type=Path, help="Write a compact benchmark baseline from this run")
    parser.add_argument("--max-cpu-regression", type=float, default=1.05)
    parser.add_argument("--min-gpu-speedup", type=float, default=1.5)
    parser.add_argument("--require-gpu-vector-policy", default="gpu_prefix_cpu_tail_split_vector")
    parser.add_argument("--max-prefix-relative-diff-rms", type=float, default=0.02)
    parser.add_argument("--min-prefix-cosine-similarity", type=float, default=0.999)
    parser.add_argument(
        "--require-short-gpu-vector-policy",
        default="cpu_fallback_for_short_split_vector_request",
        help="Required vector policy for a short GPU synthesis request.",
    )
    return parser.parse_args()


def _stage_args(args: argparse.Namespace) -> list[str]:
    if args.litert_lib_dir is not None:
        return ["--litert-lib-dir", str(args.litert_lib_dir)]
    values = ["--litert-stage-root", str(args.litert_stage_root)]
    if args.litert_platform:
        values.extend(["--litert-platform", str(args.litert_platform)])
    return values


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _run_step(name: str, cmd: list[str], json_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    elapsed = time.perf_counter() - started
    payload = _load_json(json_path)
    return {
        "name": name,
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "command": cmd,
        "json": str(json_path),
        "payload": payload,
        "stdout_tail": proc.stdout.strip().splitlines()[-20:],
        "stderr_tail": proc.stderr.strip().splitlines()[-40:],
    }


def _step_ok(step: dict[str, Any]) -> bool:
    if int(step.get("returncode", 1)) != 0:
        return False
    payload = step.get("payload") if isinstance(step.get("payload"), dict) else {}
    if payload.get("status") == "failed":
        return False
    threshold = payload.get("threshold_check")
    if isinstance(threshold, dict) and threshold.get("status") == "failed":
        return False
    return True


def _summary_checks(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks = []
    for step in steps:
        checks.append({
            "name": step["name"],
            "passed": _step_ok(step),
            "returncode": step.get("returncode"),
        })
    return checks


def _step_payload(steps: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for step in steps:
        if step.get("name") == name and isinstance(step.get("payload"), dict):
            return step["payload"]
    return {}


def _first_probe_result(payload: dict[str, Any], accelerator: str | None = None) -> dict[str, Any]:
    results = payload.get("results")
    if not isinstance(results, list):
        return {}
    for item in results:
        if not isinstance(item, dict):
            continue
        if accelerator is None or item.get("accelerator") == accelerator:
            return item
    return {}


def _first_output(payload: dict[str, Any], accelerator: str | None = None) -> dict[str, Any]:
    result = _first_probe_result(payload, accelerator)
    outputs = result.get("outputs")
    if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
        return outputs[0]
    return {}


def _first_benchmark_result(payload: dict[str, Any], accelerator: str | None = None) -> dict[str, Any]:
    results = payload.get("results")
    if not isinstance(results, list):
        return {}
    for item in results:
        if not isinstance(item, dict):
            continue
        if accelerator is None or item.get("accelerator") == accelerator:
            return item
    return {}


def _first_comparison(payload: dict[str, Any]) -> dict[str, Any]:
    comparison = payload.get("output_comparison")
    if not isinstance(comparison, dict):
        return {}
    comparisons = comparison.get("comparisons")
    if isinstance(comparisons, list) and comparisons and isinstance(comparisons[0], dict):
        return comparisons[0]
    return {}


def _benchmark_metric(payload: dict[str, Any], accelerator: str, key: str) -> Any:
    summary = payload.get("accelerator_summary")
    if not isinstance(summary, dict):
        return None
    accelerator_summary = summary.get(accelerator)
    if not isinstance(accelerator_summary, dict):
        return None
    return accelerator_summary.get(key)


def _cpu_regression_ratio(payload: dict[str, Any]) -> float | None:
    threshold = payload.get("threshold_check")
    checks = threshold.get("checks") if isinstance(threshold, dict) else None
    if not isinstance(checks, list):
        return None
    for check in checks:
        if isinstance(check, dict) and check.get("name") == "cpu_regression":
            value = check.get("ratio")
            return float(value) if isinstance(value, (int, float)) else None
    return None


def _extract_metrics(steps: list[dict[str, Any]]) -> dict[str, Any]:
    preflight = _step_payload(steps, "preflight")
    benchmark = _step_payload(steps, "benchmark")
    short_gpu = _step_payload(steps, "short_gpu_fallback_benchmark")
    prefix = _step_payload(steps, "vector_prefix_cpu_gpu_probe")
    tail = _step_payload(steps, "vector_tail_cpu_probe")
    prefix_comparison = _first_comparison(prefix)
    short_gpu_result = _first_benchmark_result(short_gpu, "gpu")
    metrics: dict[str, Any] = {
        "litert_platform": preflight.get("litert_platform"),
        "litert_lib_dir": preflight.get("litert_lib_dir"),
        "runtime_library": preflight.get("runtime_library"),
        "gpu_accelerator_library": preflight.get("gpu_accelerator_library"),
        "library_path_env": preflight.get("library_path_env"),
        "host_library_path_env": preflight.get("host_library_path_env") or preflight.get("library_path_env"),
        "target_library_path_env": preflight.get("target_library_path_env") or preflight.get("library_path_env"),
    }
    if benchmark:
        metrics.update({
            "cpu_wall_seconds_mean": _benchmark_metric(benchmark, "cpu", "wall_seconds_mean"),
            "gpu_wall_seconds_mean": _benchmark_metric(benchmark, "gpu", "wall_seconds_mean"),
            "gpu_speedup_vs_cpu_wall_mean": _benchmark_metric(benchmark, "gpu", "speedup_vs_cpu_wall_mean"),
            "cpu_regression_ratio": _cpu_regression_ratio(benchmark),
            "gpu_vector_accelerator_policy": _benchmark_metric(benchmark, "gpu", "litert_vector_accelerator_policy"),
            "gpu_vocoder_accelerator_policy": _benchmark_metric(benchmark, "gpu", "litert_vocoder_accelerator_policy"),
        })
    if short_gpu:
        metrics.update({
            "short_gpu_wall_seconds_mean": _benchmark_metric(short_gpu, "gpu", "wall_seconds_mean"),
            "short_gpu_latent_frames": short_gpu_result.get("latent_frames"),
            "short_gpu_vector_execution": short_gpu_result.get("litert_vector_execution"),
            "short_gpu_vector_accelerator": short_gpu_result.get("litert_vector_accelerator"),
            "short_gpu_vector_accelerator_policy": short_gpu_result.get("litert_vector_accelerator_policy"),
            "short_gpu_min_split_latent_frames": short_gpu_result.get(
                "litert_min_gpu_split_vector_latent_frames"
            ),
        })
    if prefix:
        metrics.update({
            "vector_prefix_cpu_run_seconds": _first_probe_result(prefix, "cpu").get("run_seconds"),
            "vector_prefix_gpu_run_seconds": _first_probe_result(prefix, "gpu").get("run_seconds"),
            "vector_prefix_relative_diff_rms": prefix_comparison.get("relative_diff_rms"),
            "vector_prefix_cosine_similarity": prefix_comparison.get("cosine_similarity"),
            "vector_prefix_non_finite_count": _first_output(prefix, "gpu").get("non_finite_count"),
        })
    if tail:
        metrics.update({
            "vector_tail_cpu_run_seconds": _first_probe_result(tail, "cpu").get("run_seconds"),
            "vector_tail_non_finite_count": _first_output(tail, "cpu").get("non_finite_count"),
        })
    return metrics


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_script = SCRIPT_DIR / "benchmark_native_litert.py"
    probe_script = SCRIPT_DIR / "probe_litert_component.py"
    stage_args = _stage_args(args)

    preflight_json = args.output_dir / "preflight.json"
    benchmark_json = args.output_dir / "benchmark.json"
    short_gpu_json = args.output_dir / "short_gpu_fallback_benchmark.json"
    prefix_json = args.output_dir / "vector_prefix_probe.json"
    tail_json = args.output_dir / "vector_tail_cpu_probe.json"
    prefix_dump_dir = args.output_dir / "vector_prefix_dumps"

    preflight_step = _run_step(
        "preflight",
        [
            sys.executable,
            str(benchmark_script),
            "--preflight",
            "--tool",
            str(args.tool),
            "--bundle",
            str(args.bundle),
            *stage_args,
            "--json",
            str(preflight_json),
        ],
        preflight_json,
    )

    steps = [preflight_step]
    if not args.preflight_only:
        benchmark_cmd = [
            sys.executable,
            str(benchmark_script),
            "--tool",
            str(args.tool),
            "--bundle",
            str(args.bundle),
            "--text",
            args.text,
            "--voice",
            args.voice,
            "--language",
            args.language,
            "--seed",
            str(args.seed),
            "--accelerator",
            "cpu",
            "--accelerator",
            "gpu",
            *stage_args,
            "--json",
            str(benchmark_json),
            "--min-gpu-speedup",
            str(args.min_gpu_speedup),
            "--require-gpu-vector-policy",
            args.require_gpu_vector_policy,
        ]
        if args.baseline_json is not None:
            benchmark_cmd.extend([
                "--baseline-json",
                str(args.baseline_json),
                "--max-cpu-regression",
                str(args.max_cpu_regression),
            ])
        if args.baseline_out is not None:
            benchmark_cmd.extend(["--baseline-out", str(args.baseline_out)])

        steps.extend([
            _run_step("benchmark", benchmark_cmd, benchmark_json),
            _run_step(
                "short_gpu_fallback_benchmark",
                [
                    sys.executable,
                    str(benchmark_script),
                    "--tool",
                    str(args.tool),
                    "--bundle",
                    str(args.bundle),
                    "--text",
                    args.short_text,
                    "--voice",
                    args.voice,
                    "--language",
                    args.language,
                    "--seed",
                    str(args.seed),
                    "--accelerator",
                    "gpu",
                    *stage_args,
                    "--json",
                    str(short_gpu_json),
                    "--require-gpu-vector-policy",
                    args.require_short_gpu_vector_policy,
                ],
                short_gpu_json,
            ),
            _run_step(
                "vector_prefix_cpu_gpu_probe",
                [
                    sys.executable,
                    str(probe_script),
                    "--library",
                    str(args.library),
                    "--bundle",
                    str(args.bundle),
                    "--component",
                    "vector_estimator_prefix",
                    "--accelerator",
                    "cpu",
                    "--accelerator",
                    "gpu",
                    *stage_args,
                    "--input-profile",
                    "random",
                    "--seed",
                    str(args.seed),
                    "--active-frames",
                    str(args.active_frames),
                    "--dump-output-dir",
                    str(prefix_dump_dir),
                    "--max-relative-diff-rms",
                    str(args.max_prefix_relative_diff_rms),
                    "--min-cosine-similarity",
                    str(args.min_prefix_cosine_similarity),
                    "--json",
                    str(prefix_json),
                ],
                prefix_json,
            ),
            _run_step(
                "vector_tail_cpu_probe",
                [
                    sys.executable,
                    str(probe_script),
                    "--library",
                    str(args.library),
                    "--bundle",
                    str(args.bundle),
                    "--component",
                    "vector_estimator_tail",
                    "--accelerator",
                    "cpu",
                    *stage_args,
                    "--input-profile",
                    "random",
                    "--seed",
                    str(args.seed),
                    "--active-frames",
                    str(args.active_frames),
                    "--json",
                    str(tail_json),
                ],
                tail_json,
            ),
        ])

    checks = _summary_checks(steps)
    failed = [check for check in checks if not check.get("passed")]
    summary = {
        "schema": "scyllasband.native_litert_gpu_gate.v1",
        "status": "ok" if not failed else "failed",
        "failure_count": len(failed),
        "output_dir": str(args.output_dir),
        "bundle": str(args.bundle),
        "tool": str(args.tool),
        "library": str(args.library),
        "litert_lib_dir": str(args.litert_lib_dir) if args.litert_lib_dir is not None else None,
        "litert_stage_root": str(args.litert_stage_root) if args.litert_lib_dir is None else None,
        "litert_platform": args.litert_platform,
        "preflight_only": bool(args.preflight_only),
        "metrics": _extract_metrics(steps),
        "thresholds": {
            "max_cpu_regression": args.max_cpu_regression if args.baseline_json is not None else None,
            "min_gpu_speedup": args.min_gpu_speedup,
            "require_gpu_vector_policy": args.require_gpu_vector_policy,
            "require_short_gpu_vector_policy": args.require_short_gpu_vector_policy,
            "max_prefix_relative_diff_rms": args.max_prefix_relative_diff_rms,
            "min_prefix_cosine_similarity": args.min_prefix_cosine_similarity,
        },
        "checks": checks,
        "steps": steps,
    }
    summary_path = args.output_dir / "gate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "summary": str(summary_path), "checks": checks}, ensure_ascii=False), flush=True)
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
