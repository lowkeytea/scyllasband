"""Core AI sessions (macOS 27 or later): one model per ``.aimodel`` asset, one inference function per graph.

Each asset is loaded (and specialized for the chosen compute unit) once; its size buckets and the fused flow
are functions sharing that model's weights. Specialization is cached by the system across processes, so only
the first load after installing a bundle or updating macOS is slow. Inputs are int32 ids and masks and float32
values.
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
import threading
from typing import Any, Callable, Mapping

import numpy as np

from .coreml import parse_compute_units

KINDS = {"cpu": "cpu", "gpu": "gpu", "ane": "neural_engine", "all": None}


class CoreAISession:
    def __init__(self, function: Any, inputs: list[str], call: Callable[[Any], Any]):
        self.function, self.inputs, self._call = function, inputs, call
        # Feed each input in its declared type (an FP16 asset takes float16 values).
        self.dtypes = {name: np.dtype(str(function.desc.input_descriptor(name).dtype)) for name in inputs}

    def run(self, feeds: Mapping[str, np.ndarray]) -> np.ndarray:
        from coreai.runtime import NDArray
        arrays = {name: NDArray(np.ascontiguousarray(feeds[name], dtype=self.dtypes[name])) for name in self.inputs}
        result = self._call(self.function(arrays))
        # Copy out: the runtime may reuse (and stride) its output buffers.
        return np.array(next(iter(result.values())).numpy(), dtype=np.float32, copy=True)


def coreai_session_factory(bundle_dir: Path, *, compute_units: str | Mapping[str, str] | None = None
                           ) -> Callable[[str, Mapping[str, Any]], CoreAISession]:
    try:
        from coreai.runtime import AIModel, ComputeUnitKind, SpecializationOptions
    except ImportError as exc:   # pragma: no cover - depends on the environment
        raise RuntimeError("The Core AI backend needs macOS 27 and coreai-core: pip install --pre coreai-core") from exc
    unit_for = parse_compute_units(compute_units)
    loop = asyncio.new_event_loop()
    models: dict[str, Any] = {}
    functions: dict[tuple[str, str], Any] = {}
    lock = threading.Lock()

    def call(value: Any) -> Any:
        return loop.run_until_complete(value) if inspect.isawaitable(value) else value

    def options(asset: str):
        kind = KINDS[unit_for(asset)]
        return SpecializationOptions.default() if kind is None else \
            SpecializationOptions.from_preferred_compute_unit_kind(getattr(ComputeUnitKind, kind)())

    def factory(name: str, spec: Mapping[str, Any]) -> CoreAISession:
        artifact = spec["artifacts"]["coreai"]
        path = Path(bundle_dir) / artifact["path"]
        with lock:
            if str(path) not in models:
                models[str(path)] = call(AIModel.load(path, specialization_options=options(path.stem)))
            model = models[str(path)]
            names = list(model.function_names)
            function_name = artifact.get("function") or (names[0] if len(names) == 1 else "main")
            key = (str(path), function_name)
            if key not in functions:
                functions[key] = call(model.load_function(function_name))
        return CoreAISession(functions[key], list(spec["inputs"]), call)

    return factory
