#!/usr/bin/env python3
"""Benchmark Scylla's Band native LiteRT synthesis across accelerators.

This helper intentionally shells out to scyllasband_native_speak so it measures the
same native entry point used for release smoke tests. It writes compact JSON so
CPU/GPU profile comparisons are easy to diff and archive, and can also validate
an existing summary against simple CPU/GPU latency gates.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any


POLICY_KEYS = (
    "litert_vector_execution",
    "litert_vector_accelerator",
    "litert_vector_tail_accelerator",
    "litert_vector_accelerator_policy",
    "litert_vocoder_accelerator",
    "litert_vocoder_accelerator_policy",
)

LITERT_STAGE_PLATFORMS = {
    "android-arm64",
    "android-x86_64",
    "linux-arm64",
    "linux-x86_64",
    "macos-arm64",
    "ios-arm64",
    "ios-sim-arm64",
    "windows-x86_64",
}

LITERT_PLATFORM_ALIASES = {
    "android_aarch64": "android-arm64",
    "android-aarch64": "android-arm64",
    "android_arm64": "android-arm64",
    "android-arm64-v8a": "android-arm64",
    "android_arm64_v8a": "android-arm64",
    "arm64-v8a": "android-arm64",
    "android_x86_64": "android-x86_64",
    "linux_aarch64": "linux-arm64",
    "linux-aarch64": "linux-arm64",
    "linux_arm64": "linux-arm64",
    "linux_x86_64": "linux-x86_64",
    "linux_amd64": "linux-x86_64",
    "linux-amd64": "linux-x86_64",
    "macos_arm64": "macos-arm64",
    "macos-aarch64": "macos-arm64",
    "darwin-arm64": "macos-arm64",
    "darwin-aarch64": "macos-arm64",
    "ios_arm64": "ios-arm64",
    "ios_sim_arm64": "ios-sim-arm64",
    "ios-simulator-arm64": "ios-sim-arm64",
    "iphonesimulator-arm64": "ios-sim-arm64",
    "windows_x86_64": "windows-x86_64",
    "windows-amd64": "windows-x86_64",
    "windows_amd64": "windows-x86_64",
    "win32-x86_64": "windows-x86_64",
    "win32-amd64": "windows-x86_64",
}

LITERT_PLATFORM_LIBS = {
    "android-arm64": ("libLiteRt.so", "libLiteRtClGlAccelerator.so"),
    "android-x86_64": ("libLiteRt.so", "libLiteRtClGlAccelerator.so"),
    "linux-arm64": ("libLiteRt.so", "libLiteRtWebGpuAccelerator.so"),
    "linux-x86_64": ("libLiteRt.so", "libLiteRtWebGpuAccelerator.so"),
    "macos-arm64": ("libLiteRt.dylib", "libLiteRtMetalAccelerator.dylib"),
    "ios-arm64": ("libLiteRt.dylib", "libLiteRtMetalAccelerator.dylib"),
    "ios-sim-arm64": ("libLiteRt.dylib", "libLiteRtMetalAccelerator.dylib"),
    "windows-x86_64": ("libLiteRt.dll", "libLiteRtWebGpuAccelerator.dll"),
}

DEFAULT_LITERT_STAGE_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "litert"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", type=Path, default=Path("build/scyllasband_native_speak"))
    parser.add_argument("--bundle", type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--text")
    group.add_argument("--file", type=Path)
    parser.add_argument("--check-json", type=Path, help="Validate an existing benchmark summary JSON instead of running synthesis")
    parser.add_argument("--baseline-json", type=Path, help="Optional baseline summary JSON for regression checks")
    parser.add_argument("--max-cpu-regression", type=float, help="Fail if current CPU wall mean exceeds baseline CPU wall mean by this ratio")
    parser.add_argument("--min-gpu-speedup", type=float, help="Fail if GPU wall-time speedup versus CPU is below this value")
    parser.add_argument("--require-gpu-vector-policy", help="Fail if the GPU summary does not include this vector accelerator policy")
    parser.add_argument("--voice", default="scylla")
    parser.add_argument("--language", default="en_us")
    parser.add_argument("--emotion", default="neutral")
    parser.add_argument("--seed", type=int, help="Deterministic synthesis seed passed to scyllasband_native_speak")
    parser.add_argument("--accelerator", action="append", choices=["cpu", "auto", "gpu", "npu"], default=None)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--long-form", action="store_true", help="Benchmark native long-form synthesis instead of a single synthesis call")
    mode_group.add_argument("--stream", action="store_true", help="Benchmark native long-form streaming callbacks and first-audio latency")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/scyllasband_native_bench"))
    parser.add_argument("--litert-lib-dir", type=Path, help="Explicit directory prepended to the platform shared-library path for GPU accelerator runs")
    parser.add_argument("--litert-stage-root", type=Path, help="LiteRT staging root containing lib/<platform>; resolves the current host by default")
    parser.add_argument("--litert-platform", help="LiteRT staged platform name or alias for --litert-stage-root, such as linux-x86_64 or macos-arm64")
    parser.add_argument("--json", type=Path, help="Write benchmark summary JSON")
    parser.add_argument("--baseline-out", type=Path, help="Write compact benchmark baseline JSON without per-run logs or paths")
    parser.add_argument("--preflight", action="store_true", help="Resolve and validate staged LiteRT libraries without running synthesis")
    parser.add_argument("--keep-audio", action="store_true")
    return parser.parse_args()


def _load_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _load_summary_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"benchmark summary does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"benchmark summary is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"benchmark summary must be a JSON object: {path}")
    return payload


def _baseline_payload(summary: dict[str, Any]) -> dict[str, Any]:
    benchmark_keys = (
        "bundle",
        "tool",
        "library_path_env",
        "host_library_path_env",
        "target_library_path_env",
        "litert_lib_dir",
        "litert_stage_root",
        "litert_platform",
        "voice",
        "language",
        "emotion",
        "seed",
        "repeats",
        "mode",
    )
    benchmark = {key: summary.get(key) for key in benchmark_keys if key in summary}
    accelerator_summary = summary.get("accelerator_summary")
    if not isinstance(accelerator_summary, dict):
        accelerator_summary = {}
    return {
        "schema": "scyllasband.native_litert_benchmark.baseline.v1",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "benchmark": benchmark,
        "accelerator_summary": accelerator_summary,
    }


def _preflight_payload(args: argparse.Namespace) -> dict[str, Any]:
    runtime_lib, gpu_lib = LITERT_PLATFORM_LIBS.get(str(args.resolved_litert_platform or ""), (None, None))

    def path_state(path: Path | None) -> dict[str, Any] | None:
        if path is None:
            return None
        return {
            "path": str(path),
            "exists": path.exists(),
            "is_file": path.is_file(),
            "is_dir": path.is_dir(),
        }

    runtime_path = args.resolved_litert_lib_dir / runtime_lib if args.resolved_litert_lib_dir is not None and runtime_lib else None
    gpu_path = args.resolved_litert_lib_dir / gpu_lib if args.resolved_litert_lib_dir is not None and gpu_lib else None
    payload = {
        "schema": "scyllasband.native_litert_benchmark.preflight.v1",
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "library_path_env": _library_path_env_name(),
        "host_library_path_env": _library_path_env_name(),
        "target_library_path_env": _target_library_path_env_name(args.resolved_litert_platform),
        "litert_stage_root": str(args.resolved_litert_stage_root) if args.resolved_litert_stage_root is not None else None,
        "litert_platform": args.resolved_litert_platform,
        "litert_lib_dir": str(args.resolved_litert_lib_dir) if args.resolved_litert_lib_dir is not None else None,
        "tool": path_state(args.tool),
        "bundle": path_state(args.bundle) if args.bundle is not None else None,
        "runtime_library": {"name": runtime_lib, **(path_state(runtime_path) or {})} if runtime_lib else None,
        "gpu_accelerator_library": {"name": gpu_lib, **(path_state(gpu_path) or {})} if gpu_lib else None,
    }
    required = [payload.get("runtime_library"), payload.get("gpu_accelerator_library")]
    payload["status"] = "ok" if all(item and item.get("is_file") for item in required) else "failed"
    return payload


def _last_status_json(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _load_events_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    counts: collections.Counter[str] = collections.Counter()
    audio_chunk_samples: list[int] = []
    policy_counts: dict[str, collections.Counter[str]] = {key: collections.Counter() for key in POLICY_KEYS}
    timing_ms: collections.Counter[str] = collections.Counter()
    first_audio_ms = None
    last_audio_ms = None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        event_type = event.get("type")
        if event_type is not None:
            counts[str(event_type)] += 1
        elapsed_ms = event.get("elapsed_ms")
        if event_type != "audio_chunk":
            continue
        if first_audio_ms is None and elapsed_ms is not None:
            first_audio_ms = int(elapsed_ms)
        if elapsed_ms is not None:
            last_audio_ms = int(elapsed_ms)
        if event.get("sample_count") is not None:
            audio_chunk_samples.append(int(event["sample_count"]))
        chunk_metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
        runtime_metadata = chunk_metadata.get("metadata") if isinstance(chunk_metadata.get("metadata"), dict) else {}
        for key in POLICY_KEYS:
            value = runtime_metadata.get(key)
            if value is not None:
                policy_counts[key][str(value)] += 1
        timing = runtime_metadata.get("timing_ms")
        if isinstance(timing, dict):
            for key, value in timing.items():
                if isinstance(value, (int, float)):
                    timing_ms[str(key)] += float(value)
    summary = {
        "event_counts": dict(sorted(counts.items())),
        "audio_chunk_count": int(counts.get("audio_chunk", 0)),
        "first_audio_ms": first_audio_ms,
        "last_audio_ms": last_audio_ms,
        "audio_chunk_sample_count_total": sum(audio_chunk_samples),
        "audio_chunk_sample_count_min": min(audio_chunk_samples) if audio_chunk_samples else None,
        "audio_chunk_sample_count_max": max(audio_chunk_samples) if audio_chunk_samples else None,
        "timing_ms": dict(sorted(timing_ms.items())),
    }
    for key, values in policy_counts.items():
        if values:
            summary[key] = next(iter(values)) if len(values) == 1 else "mixed"
            summary[f"{key}_counts"] = dict(sorted(values.items()))
    return summary


def _library_path_env_name() -> str:
    if sys.platform == "darwin":
        return "DYLD_LIBRARY_PATH"
    if os.name == "nt":
        return "PATH"
    return "LD_LIBRARY_PATH"


def _target_library_path_env_name(platform_name: str | None) -> str:
    platform_name = str(platform_name or "")
    if platform_name.startswith("macos-") or platform_name.startswith("ios-"):
        return "DYLD_LIBRARY_PATH"
    if platform_name.startswith("windows-"):
        return "PATH"
    return "LD_LIBRARY_PATH"


def _normalize_litert_platform(value: str) -> str:
    normalized = value.strip().lower().replace("/", "-")
    normalized = LITERT_PLATFORM_ALIASES.get(normalized, normalized)
    if normalized not in LITERT_STAGE_PLATFORMS:
        supported = ", ".join(sorted(LITERT_STAGE_PLATFORMS))
        raise SystemExit(f"Unsupported LiteRT platform {value!r}. Supported: {supported}")
    return normalized


def _host_litert_platform_name() -> str:
    machine = platform.machine().lower()
    if sys.platform == "darwin":
        if machine not in {"arm64", "aarch64"}:
            raise SystemExit("LiteRT prebuilts currently publish macos-arm64 for macOS benchmark hosts.")
        return "macos-arm64"
    if sys.platform.startswith("linux"):
        if machine in {"aarch64", "arm64"}:
            return "linux-arm64"
        return "linux-x86_64"
    if sys.platform.startswith("win"):
        return "windows-x86_64"
    raise SystemExit(f"Unsupported host platform for LiteRT staging: {sys.platform}")


def _resolve_litert_lib_dir(args: argparse.Namespace) -> tuple[Path | None, str | None, Path | None]:
    if args.litert_lib_dir is not None:
        if args.litert_stage_root is not None or args.litert_platform is not None:
            raise SystemExit("Use either --litert-lib-dir or --litert-stage-root/--litert-platform, not both.")
        return args.litert_lib_dir.expanduser().resolve(), _host_litert_platform_name(), None
    if args.litert_stage_root is None and args.litert_platform is None:
        return None, None, None
    stage_root = (args.litert_stage_root or DEFAULT_LITERT_STAGE_ROOT).expanduser().resolve()
    platform_name = _normalize_litert_platform(args.litert_platform) if args.litert_platform else _host_litert_platform_name()
    lib_dir = stage_root / "lib" / platform_name
    if not lib_dir.is_dir():
        raise SystemExit(f"Resolved LiteRT lib directory does not exist: {lib_dir}")
    return lib_dir, platform_name, stage_root


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _min(values: list[float]) -> float | None:
    return min(values) if values else None


def _max(values: list[float]) -> float | None:
    return max(values) if values else None


def _metadata_value_counts(results: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: collections.Counter[str] = collections.Counter()
    for result in results:
        value = result.get(key)
        if value is not None:
            counts[str(value)] += 1
    return dict(sorted(counts.items()))


def _require_run_inputs(args: argparse.Namespace) -> None:
    if args.check_json is not None or args.preflight:
        return
    if args.bundle is None:
        raise SystemExit("--bundle is required unless --check-json is used")
    if args.text is None and args.file is None:
        raise SystemExit("pass --text or --file unless --check-json is used")


def _accelerator_summary(summary: dict[str, Any], accelerator: str) -> dict[str, Any]:
    value = summary.get("accelerator_summary", {}).get(accelerator)
    return value if isinstance(value, dict) else {}


def _float_metric(summary: dict[str, Any], accelerator: str, key: str) -> float | None:
    value = _accelerator_summary(summary, accelerator).get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _check_summary(
    summary: dict[str, Any],
    *,
    baseline: dict[str, Any] | None = None,
    max_cpu_regression: float | None = None,
    min_gpu_speedup: float | None = None,
    require_gpu_vector_policy: str | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add_check(name: str, passed: bool, **extra: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), **extra})

    if max_cpu_regression is not None:
        if baseline is None:
            add_check("cpu_regression", False, reason="--max-cpu-regression requires --baseline-json")
        else:
            current = _float_metric(summary, "cpu", "wall_seconds_mean")
            reference = _float_metric(baseline, "cpu", "wall_seconds_mean")
            ratio = (current / reference) if current is not None and reference else None
            add_check(
                "cpu_regression",
                ratio is not None and ratio <= float(max_cpu_regression),
                current_wall_seconds_mean=current,
                baseline_wall_seconds_mean=reference,
                ratio=ratio,
                max_ratio=float(max_cpu_regression),
            )

    if min_gpu_speedup is not None:
        speedup = _float_metric(summary, "gpu", "speedup_vs_cpu_wall_mean")
        add_check(
            "gpu_speedup",
            speedup is not None and speedup >= float(min_gpu_speedup),
            speedup=speedup,
            min_speedup=float(min_gpu_speedup),
        )

    if require_gpu_vector_policy:
        policies = _accelerator_summary(summary, "gpu").get("litert_vector_accelerator_policy")
        present = isinstance(policies, dict) and int(policies.get(require_gpu_vector_policy, 0)) > 0
        add_check(
            "gpu_vector_policy",
            present,
            required=require_gpu_vector_policy,
            observed=policies,
        )

    failures = [check for check in checks if not check.get("passed")]
    return {
        "status": "ok" if not failures else "failed",
        "checks": checks,
        "failure_count": len(failures),
    }


def _run_once(args: argparse.Namespace, accelerator: str, repeat: int) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{accelerator}_{repeat:02d}"
    wav_path = args.output_dir / f"{stem}.wav"
    metadata_path = args.output_dir / f"{stem}.json"
    events_path = args.output_dir / f"{stem}.events.jsonl"
    cmd = [
        str(args.tool),
        "--bundle",
        str(args.bundle),
        "--voice",
        str(args.voice),
        "--language",
        str(args.language),
        "--emotion",
        str(args.emotion),
        "--litert-accelerator",
        accelerator,
        "--output",
        str(wav_path),
        "--metadata",
        str(metadata_path),
        "--no-progress",
    ]
    if args.stream:
        cmd.extend(["--stream", "--events", str(events_path)])
    elif args.long_form:
        cmd.append("--long-form")
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.text is not None:
        cmd.extend(["--text", args.text])
    else:
        cmd.extend(["--file", str(args.file)])

    env = os.environ.copy()
    litert_lib_dir = getattr(args, "resolved_litert_lib_dir", None)
    if litert_lib_dir is not None:
        env_name = _library_path_env_name()
        current = env.get(env_name, "")
        env[env_name] = f"{litert_lib_dir}{os.pathsep}{current}" if current else str(litert_lib_dir)

    start = time.perf_counter()
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    elapsed = time.perf_counter() - start
    metadata = _load_metadata(metadata_path)
    status_json = _last_status_json(proc.stdout)
    events_summary = _load_events_summary(events_path) if args.stream else {}
    sample_rate = status_json.get("sample_rate") or metadata.get("sample_rate") or metadata.get("bundle", {}).get("sample_rate")
    sample_count = status_json.get("sample_count") or metadata.get("sample_count")
    audio_seconds = None
    if sample_rate and sample_count:
        audio_seconds = float(sample_count) / float(sample_rate)
    realtime_factor = None
    audio_seconds_per_wall_second = None
    if audio_seconds is not None and elapsed > 0.0:
        realtime_factor = elapsed / audio_seconds
        audio_seconds_per_wall_second = audio_seconds / elapsed
    if proc.returncode == 0 and not args.keep_audio:
        wav_path.unlink(missing_ok=True)

    return {
        "accelerator": accelerator,
        "repeat": repeat,
        "returncode": proc.returncode,
        "wall_seconds": elapsed,
        "sample_rate": sample_rate,
        "sample_count": sample_count,
        "audio_seconds": audio_seconds,
        "realtime_factor": realtime_factor,
        "audio_seconds_per_wall_second": audio_seconds_per_wall_second,
        "latent_frames": status_json.get("latent_frames") or metadata.get("latent_frames"),
        "first_audio_ms": status_json.get("first_audio_ms") or events_summary.get("first_audio_ms"),
        "last_audio_ms": events_summary.get("last_audio_ms"),
        "mode": status_json.get("mode") or metadata.get("mode") or ("stream" if args.stream else "long_form" if args.long_form else "single"),
        "chunk_count": status_json.get("chunk_count") or metadata.get("chunk_count"),
        "retry_split_count": metadata.get("retry_split_count"),
        "latent_retry_split_count": metadata.get("latent_retry_split_count"),
        "g2p_retry_split_count": metadata.get("g2p_retry_split_count"),
        "phone_retry_split_count": metadata.get("phone_retry_split_count"),
        "duration_preflight_split_count": metadata.get("duration_preflight_split_count"),
        "event_counts": events_summary.get("event_counts"),
        "audio_chunk_count": events_summary.get("audio_chunk_count"),
        "audio_chunk_sample_count_total": events_summary.get("audio_chunk_sample_count_total"),
        "timing_ms": metadata.get("timing_ms") or events_summary.get("timing_ms"),
        "litert_vector_execution": metadata.get("litert_vector_execution") or events_summary.get("litert_vector_execution"),
        "litert_vector_accelerator": metadata.get("litert_vector_accelerator") or events_summary.get("litert_vector_accelerator"),
        "litert_vector_tail_accelerator": metadata.get("litert_vector_tail_accelerator") or events_summary.get("litert_vector_tail_accelerator"),
        "litert_vector_accelerator_policy": metadata.get("litert_vector_accelerator_policy") or events_summary.get("litert_vector_accelerator_policy"),
        "litert_min_gpu_split_vector_latent_frames": metadata.get("litert_min_gpu_split_vector_latent_frames"),
        "litert_vocoder_accelerator": metadata.get("litert_vocoder_accelerator") or events_summary.get("litert_vocoder_accelerator"),
        "litert_vocoder_accelerator_policy": metadata.get("litert_vocoder_accelerator_policy") or events_summary.get("litert_vocoder_accelerator_policy"),
        "output": str(wav_path) if args.keep_audio else None,
        "metadata": str(metadata_path),
        "events": str(events_path) if args.stream else None,
        "stdout_tail": proc.stdout.strip().splitlines()[-5:],
        "stderr_unsupported_count": proc.stderr.count("Following operations are not supported by GPU delegate"),
        "stderr_validation_error_count": proc.stderr.count("Validation error"),
        "stderr_error_count": proc.stderr.count("ERROR:"),
        "stderr_tail": proc.stderr.strip().splitlines()[-20:],
    }


def _summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_accelerator: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for result in results:
        by_accelerator[str(result.get("accelerator"))].append(result)

    summaries: dict[str, dict[str, Any]] = {}
    for accelerator, accelerator_results in sorted(by_accelerator.items()):
        successful = [item for item in accelerator_results if int(item.get("returncode", 1)) == 0]
        wall_seconds = [float(item["wall_seconds"]) for item in successful if item.get("wall_seconds") is not None]
        realtime_factors = [float(item["realtime_factor"]) for item in successful if item.get("realtime_factor") is not None]
        audio_rates = [
            float(item["audio_seconds_per_wall_second"])
            for item in successful
            if item.get("audio_seconds_per_wall_second") is not None
        ]
        first_audio_ms_values = [
            float(item["first_audio_ms"])
            for item in successful
            if item.get("first_audio_ms") is not None
        ]
        chunk_counts = [
            float(item["chunk_count"])
            for item in successful
            if item.get("chunk_count") is not None
        ]
        audio_chunk_counts = [
            float(item["audio_chunk_count"])
            for item in successful
            if item.get("audio_chunk_count") is not None
        ]
        timing_keys: set[str] = set()
        for item in successful:
            timing = item.get("timing_ms")
            if isinstance(timing, dict):
                timing_keys.update(str(key) for key in timing)
        timing_ms_mean = {}
        for key in sorted(timing_keys):
            values = [
                float(item["timing_ms"][key])
                for item in successful
                if isinstance(item.get("timing_ms"), dict) and key in item["timing_ms"]
            ]
            timing_ms_mean[key] = _mean(values)
        summaries[accelerator] = {
            "runs": len(accelerator_results),
            "successful_runs": len(successful),
            "failed_runs": len(accelerator_results) - len(successful),
            "wall_seconds_mean": _mean(wall_seconds),
            "wall_seconds_min": _min(wall_seconds),
            "wall_seconds_max": _max(wall_seconds),
            "realtime_factor_mean": _mean(realtime_factors),
            "audio_seconds_per_wall_second_mean": _mean(audio_rates),
            "first_audio_ms_mean": _mean(first_audio_ms_values),
            "first_audio_ms_min": _min(first_audio_ms_values),
            "first_audio_ms_max": _max(first_audio_ms_values),
            "chunk_count_mean": _mean(chunk_counts),
            "audio_chunk_count_mean": _mean(audio_chunk_counts),
            "retry_split_count_total": sum(int(item.get("retry_split_count") or 0) for item in successful),
            "latent_retry_split_count_total": sum(int(item.get("latent_retry_split_count") or 0) for item in successful),
            "g2p_retry_split_count_total": sum(int(item.get("g2p_retry_split_count") or 0) for item in successful),
            "phone_retry_split_count_total": sum(int(item.get("phone_retry_split_count") or 0) for item in successful),
            "duration_preflight_split_count_total": sum(int(item.get("duration_preflight_split_count") or 0) for item in successful),
            "timing_ms_mean": timing_ms_mean,
            "stderr_unsupported_count_total": sum(int(item.get("stderr_unsupported_count") or 0) for item in accelerator_results),
            "stderr_validation_error_count_total": sum(int(item.get("stderr_validation_error_count") or 0) for item in accelerator_results),
            "stderr_error_count_total": sum(int(item.get("stderr_error_count") or 0) for item in accelerator_results),
            "litert_vector_execution": _metadata_value_counts(successful, "litert_vector_execution"),
            "litert_vector_accelerator": _metadata_value_counts(successful, "litert_vector_accelerator"),
            "litert_vector_tail_accelerator": _metadata_value_counts(successful, "litert_vector_tail_accelerator"),
            "litert_vector_accelerator_policy": _metadata_value_counts(successful, "litert_vector_accelerator_policy"),
            "litert_min_gpu_split_vector_latent_frames": _metadata_value_counts(
                successful,
                "litert_min_gpu_split_vector_latent_frames",
            ),
            "litert_vocoder_accelerator": _metadata_value_counts(successful, "litert_vocoder_accelerator"),
            "litert_vocoder_accelerator_policy": _metadata_value_counts(successful, "litert_vocoder_accelerator_policy"),
        }

    cpu_mean = summaries.get("cpu", {}).get("wall_seconds_mean")
    if cpu_mean:
        for summary in summaries.values():
            wall_mean = summary.get("wall_seconds_mean")
            summary["speedup_vs_cpu_wall_mean"] = float(cpu_mean) / float(wall_mean) if wall_mean else None
    return summaries


def main() -> int:
    args = _parse_args()
    if args.preflight and args.litert_lib_dir is None and args.litert_stage_root is None:
        args.litert_stage_root = DEFAULT_LITERT_STAGE_ROOT
    args.resolved_litert_lib_dir, args.resolved_litert_platform, args.resolved_litert_stage_root = _resolve_litert_lib_dir(args)
    _require_run_inputs(args)
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    if args.preflight:
        payload = _preflight_payload(args)
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        if args.json is not None:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return 0 if payload.get("status") == "ok" else 1

    if args.check_json is not None:
        summary = _load_summary_json(args.check_json)
        results = list(summary.get("results") or [])
    else:
        accelerators = args.accelerator or ["cpu", "auto"]
        results = []
        for accelerator in accelerators:
            for repeat in range(args.repeats):
                result = _run_once(args, accelerator, repeat)
                results.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
                if result["returncode"] != 0:
                    break
        summary = {
            "bundle": str(args.bundle),
            "tool": str(args.tool),
            "library_path_env": _library_path_env_name(),
            "host_library_path_env": _library_path_env_name(),
            "target_library_path_env": _target_library_path_env_name(args.resolved_litert_platform),
            "litert_lib_dir": str(args.resolved_litert_lib_dir) if args.resolved_litert_lib_dir is not None else None,
            "litert_stage_root": str(args.resolved_litert_stage_root) if args.resolved_litert_stage_root is not None else None,
            "litert_platform": args.resolved_litert_platform,
            "voice": args.voice,
            "language": args.language,
            "emotion": args.emotion,
            "seed": args.seed,
            "repeats": args.repeats,
            "mode": "stream" if args.stream else "long_form" if args.long_form else "single",
            "accelerator_summary": _summarize_results(results),
            "results": results,
        }

    baseline = _load_summary_json(args.baseline_json) if args.baseline_json is not None else None
    threshold_check = _check_summary(
        summary,
        baseline=baseline,
        max_cpu_regression=args.max_cpu_regression,
        min_gpu_speedup=args.min_gpu_speedup,
        require_gpu_vector_policy=args.require_gpu_vector_policy,
    )
    if threshold_check["checks"]:
        summary["threshold_check"] = threshold_check
        print(json.dumps(threshold_check, ensure_ascii=False), flush=True)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
        args.json.write_text(payload, encoding="utf-8")

    if args.baseline_out is not None:
        args.baseline_out.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(_baseline_payload(summary), indent=2, ensure_ascii=False) + "\n"
        args.baseline_out.write_text(payload, encoding="utf-8")

    runs_ok = all(int(item.get("returncode", 1)) == 0 for item in results)
    checks_ok = threshold_check["status"] == "ok"
    return 0 if runs_ok and checks_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
