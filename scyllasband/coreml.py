"""Core ML sessions (macOS 15 or later): one compiled model per graph function.

An asset is a multifunction ``.mlmodelc`` whose functions (size buckets, fused flow) share one copy of the
weights; ``artifact["function"]`` names the function that implements a component. Inputs are int32 ids and
masks and float32 values.
"""

from __future__ import annotations

from pathlib import Path
import threading
from typing import Any, Callable, Mapping

import numpy as np

COMPUTE_UNITS = {"cpu": "CPU_ONLY", "gpu": "CPU_AND_GPU", "ane": "CPU_AND_NE", "all": "ALL"}


def parse_compute_units(value: str | Mapping[str, str] | None, default: str = "gpu") -> Callable[[str], str]:
    """``"gpu"`` for every asset, or ``"gpu,g2p=cpu,duration_predictor=cpu"``: a default plus per-asset choices."""
    if isinstance(value, Mapping):
        table, fallback = {str(k): str(v) for k, v in value.items()}, default
    else:
        table, fallback = {}, default
        for part in filter(None, (p.strip() for p in str(value or "").split(","))):
            key, sep, unit = part.partition("=")
            if sep:
                table[key.strip()] = unit.strip()
            else:
                fallback = key
    for unit in (fallback, *table.values()):
        if unit not in COMPUTE_UNITS:
            raise ValueError(f"Unknown compute unit {unit!r}; choose one of {', '.join(COMPUTE_UNITS)}")
    return lambda asset: table.get(asset, fallback)


def feeds_for(inputs: list[str], feeds: Mapping[str, np.ndarray], *, floats=np.float32) -> dict[str, np.ndarray]:
    out = {}
    for name in inputs:
        value = np.asarray(feeds[name])
        integral = value.dtype == bool or np.issubdtype(value.dtype, np.integer)
        out[name] = np.ascontiguousarray(value, dtype=np.int32 if integral else floats)
    return out


class CoreMLSession:
    def __init__(self, model: Any, inputs: list[str]):
        self.model, self.inputs = model, inputs

    def run(self, feeds: Mapping[str, np.ndarray]) -> np.ndarray:
        result = self.model.predict(feeds_for(self.inputs, feeds))
        return np.asarray(next(iter(result.values())), dtype=np.float32)


def coreml_session_factory(bundle_dir: Path, *, compute_units: str | Mapping[str, str] | None = None
                           ) -> Callable[[str, Mapping[str, Any]], CoreMLSession]:
    try:
        import coremltools as ct
    except ImportError as exc:   # pragma: no cover - depends on the environment
        raise RuntimeError("The Core ML backend needs coremltools: pip install coremltools") from exc
    unit_for = parse_compute_units(compute_units)
    models: dict[tuple[str, str | None], Any] = {}
    lock = threading.Lock()

    def factory(name: str, spec: Mapping[str, Any]) -> CoreMLSession:
        artifact = spec["artifacts"]["coreml"]
        path = Path(bundle_dir) / artifact["path"]
        key = (str(path), artifact.get("function"))
        with lock:
            if key not in models:
                unit = getattr(ct.ComputeUnit, COMPUTE_UNITS[unit_for(path.stem)])
                models[key] = ct.models.CompiledMLModel(str(path), compute_units=unit, function_name=artifact.get("function"))
        return CoreMLSession(models[key], list(spec["inputs"]))

    return factory
