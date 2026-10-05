"""LiteRT sessions: one interpreter per ``.tflite`` file, one signature runner per graph.

The flow and the vocoder hold every latent bucket as a signature of one model, so their weights are loaded
once; a bucket's runner (and its tensor arena) is created the first time that bucket is used.
"""

from __future__ import annotations

from pathlib import Path
import threading
from typing import Any, Callable, Mapping

import numpy as np


def _interpreter_class():
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError as exc:   # pragma: no cover - depends on the environment
        raise RuntimeError("The LiteRT backend needs ai-edge-litert: pip install ai-edge-litert") from exc
    return Interpreter


class LiteRTSession:
    def __init__(self, interpreter: Any, signature: str, inputs: list[str]):
        self.runner = interpreter.get_signature_runner(signature)
        details = self.runner.get_input_details()
        # LiteRT keeps traced argument names when it can, else args_0..N in contract order.
        self.keys = inputs if all(name in details for name in inputs) else sorted(details, key=_arg_index)
        if len(self.keys) != len(inputs):
            raise ValueError(f"Signature {signature!r} has inputs {list(details)}, expected {inputs}")
        self.dtypes = [details[key]["dtype"] for key in self.keys]
        self.inputs = inputs

    def run(self, feeds: Mapping[str, np.ndarray]) -> np.ndarray:
        args = {key: np.asarray(feeds[name], dtype=dtype) for key, name, dtype in zip(self.keys, self.inputs, self.dtypes)}
        return next(iter(self.runner(**args).values()))


def litert_session_factory(bundle_dir: Path, *, threads: int | None = None) -> Callable[[str, Mapping[str, Any]], LiteRTSession]:
    interpreters: dict[str, Any] = {}
    lock = threading.Lock()
    Interpreter = _interpreter_class()

    def factory(name: str, spec: Mapping[str, Any]) -> LiteRTSession:
        artifact = spec["artifacts"]["litert"]
        path = str(Path(bundle_dir) / artifact["path"])
        with lock:
            if path not in interpreters:
                interpreters[path] = Interpreter(model_path=path, num_threads=threads or None)
        return LiteRTSession(interpreters[path], str(artifact.get("signature") or "serving_default"), list(spec["inputs"]))

    return factory


def _arg_index(key: str) -> int:
    tail = key.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else 0
