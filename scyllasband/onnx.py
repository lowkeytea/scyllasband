"""ONNX Runtime graph runner for Scylla's Band duration-flow bundles."""

from __future__ import annotations

import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .contract import ScyllasBandBundleManifest
from .litert import LiteRTRunner


class ONNXRunner(LiteRTRunner):
    """Execute the Scylla's Band fixed-shape graph chain with ONNX Runtime.

    The request preparation, reference conditioning, duration post-processing,
    flow sampling, and bucket selection are shared with the LiteRT runner. This
    adapter only changes component artifact lookup and session invocation.
    """

    backend_name = "onnx"

    def __init__(
        self,
        bundle_dir: str | Path,
        manifest: ScyllasBandBundleManifest,
        *,
        providers: Sequence[str] | None = None,
        sessions: Mapping[str, Any] | None = None,
        intra_op_num_threads: int = 0,
        inter_op_num_threads: int = 0,
        autotune_intra_op_threads: bool = False,
        autotune_candidates: Sequence[int] | None = None,
        autotune_cache_path: str | Path | None = None,
        use_cached_thread_tuning: bool = True,
    ) -> None:
        super().__init__(bundle_dir, manifest, sessions=sessions)
        self.providers = tuple(str(item) for item in (providers or ("CPUExecutionProvider",)))
        configured_threads = int(intra_op_num_threads)
        if configured_threads < 0:
            raise ValueError("intra_op_num_threads must be >= 0")
        if int(inter_op_num_threads) < 0:
            raise ValueError("inter_op_num_threads must be >= 0")
        if configured_threads > 0 and autotune_intra_op_threads:
            raise ValueError(
                "autotune_intra_op_threads cannot be combined with an explicit "
                "intra_op_num_threads value"
            )
        if autotune_intra_op_threads and not _cpu_provider_is_primary(self.providers):
            raise ValueError(
                "ONNX thread autotuning is only supported when CPUExecutionProvider "
                "is the primary provider"
            )
        self.configured_intra_op_num_threads = configured_threads
        self.intra_op_num_threads = configured_threads
        self.inter_op_num_threads = int(inter_op_num_threads)
        self.autotune_candidates = _normalize_thread_candidates(autotune_candidates)
        self.autotune_cache_path = Path(
            autotune_cache_path or _default_onnx_thread_cache_path()
        ).expanduser()
        self._thread_tuning_lock = threading.RLock()
        self._thread_tuning_cache_key = _onnx_thread_cache_key(
            self.bundle_dir,
            providers=self.providers,
            inter_op_num_threads=self.inter_op_num_threads,
        )
        self._thread_tuning: dict[str, Any] = {
            "enabled": bool(autotune_intra_op_threads),
            "pending": False,
            "cache_hit": False,
            "configured_intra_op_threads": configured_threads,
            "selected_intra_op_threads": configured_threads,
            "candidates": list(self.autotune_candidates),
            "timings_ms": {},
            "cache_path": str(self.autotune_cache_path),
        }
        cached = None
        if (
            configured_threads == 0
            and use_cached_thread_tuning
            and sessions is None
            and _cpu_provider_is_primary(self.providers)
        ):
            cached = _load_cached_thread_tuning(
                self.autotune_cache_path,
                self._thread_tuning_cache_key,
            )
        if (
            cached is not None
            and autotune_intra_op_threads
            and autotune_candidates is not None
            and int(cached["selected_intra_op_threads"]) not in self.autotune_candidates
        ):
            cached = None
        if cached is not None:
            self.intra_op_num_threads = int(cached["selected_intra_op_threads"])
            self._thread_tuning.update(cached)
            self._thread_tuning["cache_hit"] = True
            self._thread_tuning["pending"] = False
        elif configured_threads == 0 and autotune_intra_op_threads:
            self._thread_tuning["pending"] = True
        self._autotune_component: str | None = None

    def enable_thread_autotune(
        self,
        *,
        candidates: Sequence[int] | None = None,
        cache_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Enable one-time vector-session tuning before the first vector invocation."""
        with self._thread_tuning_lock:
            if self.configured_intra_op_num_threads > 0:
                raise ValueError(
                    "Cannot autotune ONNX threads when an explicit intra-op thread count is set"
                )
            if not _cpu_provider_is_primary(self.providers):
                raise ValueError(
                    "ONNX thread autotuning is only supported when CPUExecutionProvider "
                    "is the primary provider"
                )
            if cache_path is not None:
                self.autotune_cache_path = Path(cache_path).expanduser()
                self._thread_tuning["cache_path"] = str(self.autotune_cache_path)
            if candidates is not None:
                self.autotune_candidates = _normalize_thread_candidates(candidates)
                self._thread_tuning["candidates"] = list(self.autotune_candidates)
            self._thread_tuning["enabled"] = True
            if self._thread_tuning.get("cache_hit"):
                return self.thread_tuning_status()
            if (
                not self._thread_tuning.get("pending")
                and self._thread_tuning.get("component")
                and self._thread_tuning.get("timings_ms")
            ):
                return self.thread_tuning_status()
            if any(_is_vector_component(name) for name in self._sessions):
                raise RuntimeError(
                    "ONNX thread autotuning must be enabled before a vector session is created"
                )
            self._thread_tuning["pending"] = True
            return self.thread_tuning_status()

    def thread_tuning_status(self) -> dict[str, Any]:
        with self._thread_tuning_lock:
            return dict(self._thread_tuning)

    def session_status(self) -> dict[str, Any]:
        status = super().session_status()
        status.update(
            {
                "intra_op_num_threads": int(self.intra_op_num_threads),
                "inter_op_num_threads": int(self.inter_op_num_threads),
                "thread_tuning": self.thread_tuning_status(),
            }
        )
        return status

    def _supports_batched_guidance(
        self,
        component_name: str,
        branch_count: int,
    ) -> bool:
        del component_name
        controls = self.manifest.controls if isinstance(self.manifest.controls, dict) else {}
        onnx_controls = controls.get("onnx")
        max_batch = (
            max(1, int(onnx_controls.get("dynamic_vector_batch_max") or 4))
            if isinstance(onnx_controls, Mapping)
            else 1
        )
        return bool(
            1 < branch_count <= max_batch
            and isinstance(onnx_controls, Mapping)
            and onnx_controls.get("dynamic_vector_batch")
        )

    def _session(self, component_name: str) -> Any:
        if component_name not in self._sessions:
            model_path = self._component_path(component_name)
            should_tune = bool(
                self._thread_tuning.get("pending")
                and self._autotune_component is None
                and _is_vector_component(component_name)
            )
            if should_tune:
                self._autotune_component = component_name
                self._sessions[component_name] = _AutotuningONNXSession(
                    model_path,
                    providers=self.providers,
                    candidates=self.autotune_candidates,
                    inter_op_num_threads=self.inter_op_num_threads,
                    on_complete=self._complete_thread_autotune,
                )
            else:
                self._sessions[component_name] = _ONNXSession(
                    model_path,
                    providers=self.providers,
                    intra_op_num_threads=self.intra_op_num_threads,
                    inter_op_num_threads=self.inter_op_num_threads,
                )
        return self._sessions[component_name]

    def _complete_thread_autotune(
        self,
        selected_threads: int,
        timings_ms: Mapping[str, float],
    ) -> None:
        with self._thread_tuning_lock:
            self.intra_op_num_threads = int(selected_threads)
            self._thread_tuning.update(
                {
                    "enabled": True,
                    "pending": False,
                    "cache_hit": False,
                    "selected_intra_op_threads": int(selected_threads),
                    "timings_ms": {
                        str(key): float(value) for key, value in timings_ms.items()
                    },
                    "component": self._autotune_component,
                }
            )
            _save_cached_thread_tuning(
                self.autotune_cache_path,
                self._thread_tuning_cache_key,
                self._thread_tuning,
            )

    def _component_path(self, component_name: str) -> Path:
        component = self.manifest.components[component_name]
        artifact = component.artifacts.get("onnx")
        if artifact is None:
            raise RuntimeError(f"Bundle component {component_name!r} has no ONNX artifact")
        return self.bundle_dir / artifact.path


class _ONNXSession:
    def __init__(
        self,
        model_path: Path,
        *,
        providers: Sequence[str],
        intra_op_num_threads: int = 0,
        inter_op_num_threads: int = 0,
    ) -> None:
        ort = _load_onnxruntime()
        options = ort.SessionOptions()
        if intra_op_num_threads > 0:
            options.intra_op_num_threads = int(intra_op_num_threads)
        if inter_op_num_threads > 0:
            options.inter_op_num_threads = int(inter_op_num_threads)
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=list(providers),
        )
        self.inputs = list(self.session.get_inputs())
        self.outputs = list(self.session.get_outputs())
        self._inputs_by_arg = {index: input_info for index, input_info in enumerate(self.inputs)}

    def invoke(self, args: Mapping[int, np.ndarray]) -> np.ndarray:
        return self.invoke_all(args)[0]

    def invoke_all(self, args: Mapping[int, np.ndarray]) -> list[np.ndarray]:
        missing = sorted(set(self._inputs_by_arg) - {int(key) for key in args})
        if missing:
            raise RuntimeError(f"Missing ONNX input arg(s): {missing}")
        feeds: dict[str, np.ndarray] = {}
        for arg_index, value in args.items():
            input_info = self._inputs_by_arg[int(arg_index)]
            array = np.asarray(value, dtype=_onnx_input_dtype(input_info.type))
            _validate_onnx_shape(input_info.name, input_info.shape, array.shape)
            feeds[input_info.name] = array
        output_names = [output.name for output in self.outputs]
        return list(self.session.run(output_names, feeds))


class _AutotuningONNXSession:
    """Benchmark candidate thread counts once using the first real vector input."""

    def __init__(
        self,
        model_path: Path,
        *,
        providers: Sequence[str],
        candidates: Sequence[int],
        inter_op_num_threads: int,
        on_complete: Any,
        benchmark_repeats: int = 5,
        session_factory: Any | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.providers = tuple(str(item) for item in providers)
        self.candidates = tuple(int(item) for item in candidates)
        self.inter_op_num_threads = int(inter_op_num_threads)
        self.on_complete = on_complete
        self.benchmark_repeats = max(1, int(benchmark_repeats))
        self.session_factory = session_factory or _ONNXSession
        self._selected_session: Any | None = None
        self._lock = threading.RLock()

    def invoke(self, args: Mapping[int, np.ndarray]) -> np.ndarray:
        with self._lock:
            if self._selected_session is not None:
                return self._selected_session.invoke(args)
            return self._tune(args)

    def _tune(self, args: Mapping[int, np.ndarray]) -> np.ndarray:
        timings_ms: dict[str, float] = {}
        errors: list[Exception] = []
        selected_candidate: int | None = None
        selected_timing_ms = float("inf")
        for candidate in self.candidates:
            session: Any | None = None
            try:
                session = self.session_factory(
                    self.model_path,
                    providers=self.providers,
                    intra_op_num_threads=int(candidate),
                    inter_op_num_threads=self.inter_op_num_threads,
                )
                session.invoke(args)
                samples: list[float] = []
                for _ in range(self.benchmark_repeats):
                    started_at = time.perf_counter()
                    session.invoke(args)
                    samples.append((time.perf_counter() - started_at) * 1000.0)
                timing_ms = float(statistics.median(samples))
                timings_ms[str(candidate)] = timing_ms
                if timing_ms < selected_timing_ms:
                    selected_candidate = int(candidate)
                    selected_timing_ms = timing_ms
            except Exception as exc:  # pragma: no cover - depends on provider behavior
                errors.append(exc)
            finally:
                session = None
        if selected_candidate is None:
            if errors:
                raise RuntimeError(f"ONNX thread autotuning failed: {errors[0]}") from errors[0]
            raise RuntimeError("ONNX thread autotuning produced no candidate measurements")
        selected_session = self.session_factory(
            self.model_path,
            providers=self.providers,
            intra_op_num_threads=selected_candidate,
            inter_op_num_threads=self.inter_op_num_threads,
        )
        selected_output = selected_session.invoke(args)
        self._selected_session = selected_session
        self.on_complete(selected_candidate, timings_ms)
        return selected_output


def _is_vector_component(component_name: str) -> bool:
    name = str(component_name)
    return name == "vector_estimator" or (
        name.startswith("vector_estimator_")
        and "_prefix" not in name
        and "_tail" not in name
    )


def _cpu_provider_is_primary(providers: Sequence[str]) -> bool:
    return bool(providers and str(providers[0]) == "CPUExecutionProvider")


def _normalize_thread_candidates(values: Sequence[int] | None) -> tuple[int, ...]:
    if values is None:
        cpu_count = max(1, int(os.cpu_count() or 1))
        values = (
            0,
            min(cpu_count, 4),
            min(cpu_count, 6),
            min(cpu_count, 8),
            min(cpu_count, 12),
        )
    output: list[int] = []
    for value in values:
        candidate = int(value)
        if candidate < 0:
            raise ValueError("ONNX thread autotune candidates must be >= 0")
        if candidate not in output:
            output.append(candidate)
    if not output:
        raise ValueError("ONNX thread autotune requires at least one candidate")
    return tuple(output)


def _default_onnx_thread_cache_path() -> Path:
    override = os.environ.get("SCYLLASBAND_ONNX_THREAD_CACHE")
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        root = Path.home() / "Library" / "Caches"
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "scyllasband" / "onnx_thread_tuning.json"


def _onnx_thread_cache_key(
    bundle_dir: Path,
    *,
    providers: Sequence[str],
    inter_op_num_threads: int = 0,
) -> str:
    markers: list[dict[str, Any]] = []
    component_dir = Path(bundle_dir) / "onnx" / "components"
    relative_paths = [
        Path("manifest.json"),
        Path("onnx") / "components" / "shared_weights.bin",
    ]
    try:
        relative_paths.extend(
            path.relative_to(bundle_dir)
            for path in sorted(component_dir.glob("vector_estimator*.onnx"))
        )
    except OSError:
        pass
    for relative in relative_paths:
        path = Path(bundle_dir) / relative
        try:
            stat = path.stat()
        except OSError:
            continue
        markers.append(
            {
                "path": str(relative),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    payload = {
        "format": 1,
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": int(os.cpu_count() or 1),
        "providers": [str(item) for item in providers],
        "inter_op_num_threads": int(inter_op_num_threads),
        "onnxruntime": _installed_onnxruntime_version(),
        "bundle": markers,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _installed_onnxruntime_version() -> str:
    for distribution in ("onnxruntime", "onnxruntime-gpu", "onnxruntime-silicon"):
        try:
            return str(importlib_metadata.version(distribution))
        except importlib_metadata.PackageNotFoundError:
            continue
    return "unavailable"


def _load_cached_thread_tuning(path: Path, cache_key: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    entries = payload.get("entries") if isinstance(payload, dict) else None
    entry = entries.get(cache_key) if isinstance(entries, dict) else None
    if not isinstance(entry, dict):
        return None
    try:
        selected = int(entry["selected_intra_op_threads"])
    except (KeyError, TypeError, ValueError):
        return None
    if selected < 0:
        return None
    return {
        "selected_intra_op_threads": selected,
        "timings_ms": dict(entry.get("timings_ms") or {}),
        "component": entry.get("component"),
    }


def _save_cached_thread_tuning(
    path: Path,
    cache_key: str,
    tuning: Mapping[str, Any],
) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        entries = {}
        payload["entries"] = entries
    entries[cache_key] = {
        "selected_intra_op_threads": int(tuning["selected_intra_op_threads"]),
        "timings_ms": {
            str(key): float(value)
            for key, value in dict(tuning.get("timings_ms") or {}).items()
        },
        "component": tuning.get("component"),
        "updated_at": time.time(),
    }
    payload["format"] = 1
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError:
        return


def _load_onnxruntime() -> Any:
    try:
        import onnxruntime as ort  # type: ignore[import]
    except Exception as exc:  # pragma: no cover - depends on optional runtime install
        raise RuntimeError(
            "ONNX backend requires onnxruntime. Install onnxruntime or select --backend litert."
        ) from exc
    return ort


def _onnx_input_dtype(type_name: str) -> np.dtype:
    normalized = str(type_name).strip().lower()
    return np.dtype(
        {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(double)": np.float64,
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
            "tensor(int16)": np.int16,
            "tensor(int8)": np.int8,
            "tensor(uint8)": np.uint8,
            "tensor(bool)": np.bool_,
        }.get(normalized, np.float32)
    )


def _validate_onnx_shape(name: str, expected: Sequence[Any], actual: Sequence[int]) -> None:
    if len(expected) != len(actual):
        raise RuntimeError(f"ONNX input {name!r} expected rank {len(expected)}, got {tuple(actual)}")
    for expected_dim, actual_dim in zip(expected, actual):
        if isinstance(expected_dim, int) and expected_dim > 0 and int(actual_dim) != expected_dim:
            raise RuntimeError(
                f"ONNX input {name!r} expected shape {tuple(expected)}, got {tuple(actual)}"
            )
