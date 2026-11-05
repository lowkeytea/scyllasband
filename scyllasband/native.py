"""The native runtime (libscyllasband) from Python: one ctypes binding over its C API (libscyllasband/include/scyllasband.h).

libscyllasband is the runtime the iOS and Android apps use. ``NativeRuntime`` runs it with the surface of
:class:`scyllasband.runtime.ScyllasBandRuntime` that the CLI uses (records, streams, singing, warm-up, status), so speech
and singing go through the same code on every platform. The two engines plan and render identically (the parity harness,
libscyllasband/tests/parity_harness.py, checks it with the same noise injected), but each draws its own noise: the
native runtime from a seeded xoshiro256** generator, the Python engine from numpy, so a seed gives a different, equally
valid take on each. Metadata records the engine and the noise.

``open_runtime`` chooses the engine: ``engine="python"`` (or SCYLLASBAND_ENGINE=python) always runs the Python engine;
``"native"`` requires the native library; by default the native runtime runs where a library for the bundle's backend is
found (LiteRT; Core ML and Core AI on macOS) and the Python engine otherwise, with one notice line. ONNX bundles (and
CUDA) stay on the Python engine unless the native engine is asked for.

A library is found at SCYLLASBAND_NATIVE_LIB, else in scyllasband/_native/<backend>/ (``python -m scyllasband
build-native``), else in a checkout's libscyllasband/build/<backend>/. It must be ABI 3.1 or later and built for the
bundle's backend (litert, apple or onnx).
"""

from __future__ import annotations

import ctypes
import json
import os
import queue
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence, TextIO

import numpy as np

from .planner import PlanChunk, SynthesisPlan, records_from_text
from .runtime import SUPPORTED_SAMPLERS, ScyllasBandRuntime, SynthesisRequest, SynthesisResult, VoiceSpec, _bundle_backend
from .streaming import StreamingEvent

ABI_MAJOR, ABI_MINOR = 3, 1
ENGINE_ENV = "SCYLLASBAND_ENGINE"
LIBRARY_ENV = "SCYLLASBAND_NATIVE_LIB"
PACKAGE_DIR = Path(__file__).resolve().parent / "_native"
CHECKOUT_DIR = Path(__file__).resolve().parent.parent / "libscyllasband"
NOISE = {"native": "xoshiro256ss", "python": "numpy-pcg64"}
LIBRARY_BACKENDS = {"litert": "litert", "coreml": "apple", "coreai": "apple", "onnx": "onnx", "webgpu": "onnx"}
SAMPLERS = {None: 0, "heun": 1, "euler": 2}
ACCELERATORS = {"cpu": 0, "gpu": 1, "auto": 2, "ane": 3}
EVENT_TYPES = {1: "plan", 2: "chunk_started", 3: "audio_chunk", 4: "done"}
_DONE = object()


def library_name() -> str:
    if sys.platform == "darwin":
        return "libscyllasband_native.dylib"
    if sys.platform == "win32":
        return "scyllasband_native.dll"
    return "libscyllasband_native.so"


class NativeError(RuntimeError):
    """A native call failed (SCYLLASBAND_ERROR_RUNTIME)."""


class NativeInvalidArgument(NativeError, ValueError):
    """The request was refused (SCYLLASBAND_ERROR_INVALID_ARGUMENT); ``detail`` carries the C API's error JSON when it gives one."""

    def __init__(self, message: str, detail: dict | None = None):
        super().__init__(message)
        self.detail = detail or {}


class NativeCancelled(NativeError):
    """The call was cancelled (SCYLLASBAND_ERROR_CANCELLED)."""


class NativeUnavailable(RuntimeError):
    """No usable native library for this bundle."""


# --- the C structures ----------------------------------------------------------------------------------------------------
c = ctypes


class RuntimeOptions(c.Structure):
    _fields_ = [("bundle_dir", c.c_char_p), ("threads", c.c_int32), ("accelerator", c.c_int)]


class Request(c.Structure):
    _fields_ = [("text", c.c_char_p), ("voice_id", c.c_char_p), ("language", c.c_char_p), ("delivery", c.c_char_p), ("speed", c.c_float),
                ("steps", c.c_int32), ("sampler", c.c_int), ("seed", c.c_uint64), ("has_seed", c.c_int32), ("temperature", c.c_float),
                ("normalize_text", c.c_int32)]


class RecordsRequest(c.Structure):
    _fields_ = [("struct_size", c.c_uint32), ("records_json", c.c_char_p), ("speed", c.c_float), ("steps", c.c_int32), ("sampler", c.c_int),
                ("seed", c.c_uint64), ("has_seed", c.c_int32), ("temperature", c.c_float), ("normalize_text", c.c_int32), ("pause_ms", c.c_float)]


class SingRequest(c.Structure):
    _fields_ = [("struct_size", c.c_uint32), ("score_json", c.c_char_p), ("voice_id", c.c_char_p), ("language", c.c_char_p),
                ("transpose", c.c_int32), ("use_register", c.c_int32), ("steps", c.c_int32), ("sampler", c.c_int), ("seed", c.c_uint64),
                ("has_seed", c.c_int32), ("guidance", c.c_float), ("has_guidance", c.c_int32), ("decoder", c.c_char_p)]


class Audio(c.Structure):
    _fields_ = [("samples", c.POINTER(c.c_float)), ("sample_count", c.c_int64), ("sample_rate", c.c_int32), ("metadata_json", c.c_char_p)]


class Event(c.Structure):
    _fields_ = [("type", c.c_int), ("chunk_index", c.c_int32), ("chunk_count", c.c_int32), ("chunk_id", c.c_char_p), ("metadata_json", c.c_char_p),
                ("samples", c.POINTER(c.c_float)), ("sample_count", c.c_int64), ("sample_rate", c.c_int32)]


EventCallback = c.CFUNCTYPE(c.c_int32, c.POINTER(Event), c.c_void_p)
TestNoise = c.CFUNCTYPE(None, c.c_int32, c.c_uint64, c.c_int32, c.c_int32, c.c_int64, c.POINTER(c.c_float), c.c_void_p)


@dataclass
class NativeEvent:
    """An event copied out of the C callback (its pointers are only valid there)."""
    type: int
    chunk_index: int
    chunk_count: int
    chunk_id: str | None
    metadata: dict[str, Any]
    samples: np.ndarray | None
    sample_rate: int


def copy_event(event: Event) -> NativeEvent:
    samples = None
    if event.type == 3:
        samples = (np.ctypeslib.as_array(event.samples, shape=(event.sample_count,)).copy() if event.sample_count > 0
                   else np.zeros(0, np.float32))
    return NativeEvent(type=int(event.type), chunk_index=int(event.chunk_index), chunk_count=int(event.chunk_count),
                       chunk_id=event.chunk_id.decode("utf-8") if event.chunk_id else None,
                       metadata=json.loads(event.metadata_json.decode("utf-8")) if event.metadata_json else {},
                       samples=samples, sample_rate=int(event.sample_rate))


# --- the library ---------------------------------------------------------------------------------------------------------
class NativeLibrary:
    """A loaded libscyllasband with its C API declared. Load through :func:`load_library` (one per path and process)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        try:
            lib = c.CDLL(str(self.path))
        except OSError as exc:
            raise NativeUnavailable(f"cannot load {self.path}: {exc}") from exc
        self.lib = lib
        for name, restype, argtypes in (
            ("scyllasband_version", c.c_char_p, []), ("scyllasband_backend", c.c_char_p, []), ("scyllasband_last_error", c.c_char_p, []),
            ("scyllasband_request_init", None, [c.c_void_p]), ("scyllasband_runtime_create", c.c_int, [c.c_void_p, c.POINTER(c.c_void_p)]),
            ("scyllasband_runtime_destroy", None, [c.c_void_p]), ("scyllasband_voices_json", c.c_char_p, [c.c_void_p]),
            ("scyllasband_warmup", c.c_int, [c.c_void_p, c.c_char_p]),
            ("scyllasband_plan_json", c.c_int, [c.c_void_p, c.c_void_p, c.POINTER(c.c_void_p)]),
            ("scyllasband_synthesize", c.c_int, [c.c_void_p, c.c_void_p, c.c_void_p]),
            ("scyllasband_synthesize_stream", c.c_int, [c.c_void_p, c.c_void_p, EventCallback, c.c_void_p]),
            ("scyllasband_cancel", None, [c.c_void_p]), ("scyllasband_audio_free", None, [c.c_void_p]),
            ("scyllasband_string_free", None, [c.c_void_p]),
        ):
            self._declare(name, restype, argtypes)
        self.version = lib.scyllasband_version().decode()
        self.backend = lib.scyllasband_backend().decode()
        try:
            major, minor = (int(part) for part in self.version.split(".")[:2])
        except ValueError:
            raise NativeUnavailable(f"{self.path} reports ABI {self.version!r}") from None
        if major != ABI_MAJOR or minor < ABI_MINOR:
            raise NativeUnavailable(f"{self.path} is ABI {self.version}; this package needs {ABI_MAJOR}.{ABI_MINOR} or later")
        for name, restype, argtypes in (
            ("scyllasband_records_request_init", None, [c.c_void_p]),
            ("scyllasband_plan_records_json", c.c_int, [c.c_void_p, c.c_void_p, c.POINTER(c.c_void_p)]),
            ("scyllasband_synthesize_records_stream", c.c_int, [c.c_void_p, c.c_void_p, EventCallback, c.c_void_p]),
            ("scyllasband_sing_request_init", None, [c.c_void_p]),
            ("scyllasband_score_parse", c.c_int, [c.c_char_p, c.c_int32, c.POINTER(c.c_void_p)]),
            ("scyllasband_sing_fit_json", c.c_int, [c.c_void_p, c.c_void_p, c.POINTER(c.c_void_p)]),
            ("scyllasband_sing_plan_json", c.c_int, [c.c_void_p, c.c_void_p, c.POINTER(c.c_void_p)]),
            ("scyllasband_sing", c.c_int, [c.c_void_p, c.c_void_p, c.c_void_p]),
            ("scyllasband_sing_stream", c.c_int, [c.c_void_p, c.c_void_p, EventCallback, c.c_void_p]),
            ("scyllasband_sing_preview", c.c_int, [c.c_void_p, c.c_void_p, c.c_int32, c.c_void_p]),
            ("scyllasband_warmup_singing", c.c_int, [c.c_void_p, c.c_char_p, c.c_char_p]),
        ):
            self._declare(name, restype, argtypes)
        # Test entry points (src/scyllasband_test_api.h), used by the parity harness; not part of the public ABI.
        self.has_test_api = hasattr(lib, "scyllasband_test_set_noise")
        if self.has_test_api:
            self._declare("scyllasband_test_set_noise", None, [c.c_void_p, TestNoise, c.c_void_p])
            self._declare("scyllasband_test_free", None, [c.c_void_p])
        self.check_layouts()

    def _declare(self, name: str, restype, argtypes) -> None:
        try:
            function = getattr(self.lib, name)
        except AttributeError:
            raise NativeUnavailable(f"{self.path} has no {name}") from None
        function.restype, function.argtypes = restype, argtypes

    def check_layouts(self) -> None:
        """The request structures declared here must have the library's sizes (each init fills struct_size)."""
        for structure, init in ((RecordsRequest, self.lib.scyllasband_records_request_init), (SingRequest, self.lib.scyllasband_sing_request_init)):
            value = structure()
            init(c.byref(value))
            if value.struct_size != c.sizeof(structure):
                raise NativeUnavailable(f"{self.path}: {structure.__name__} is {value.struct_size} bytes in the library, "
                                        f"{c.sizeof(structure)} in scyllasband.native")

    def last_error(self) -> str:
        return (self.lib.scyllasband_last_error() or b"").decode("utf-8", "replace")

    def take_string(self, pointer: c.c_void_p) -> str | None:
        if not pointer.value:
            return None
        try:
            return c.cast(pointer, c.c_char_p).value.decode("utf-8")
        finally:
            self.lib.scyllasband_string_free(pointer)

    def check(self, status: int, what: str, detail: dict | None = None) -> None:
        if status == 0:
            return
        message = self.last_error() or f"{what} failed"
        if status == 1:
            raise NativeInvalidArgument(message, detail)
        if status == 3:
            raise NativeCancelled(message)
        raise NativeError(message)

    def score_parse(self, score: Mapping[str, Any] | str, transpose: int = 0) -> dict[str, Any]:
        """The score checked and normalized (no runtime needed); an invalid score raises NativeInvalidArgument whose
        ``detail["note_index"]`` names the note at fault."""
        out = c.c_void_p()
        status = self.lib.scyllasband_score_parse(_score_bytes(score), int(transpose), c.byref(out))
        text = self.take_string(out)
        data = json.loads(text) if text else {}
        self.check(status, "score_parse", data)
        return data


_LIBRARIES: dict[Path, NativeLibrary] = {}
_LOCK = threading.Lock()


def library_backend(flavor: str) -> str:
    """The native build that runs a bundle flavour: litert, apple (Core ML and Core AI) or onnx."""
    try:
        return LIBRARY_BACKENDS[flavor]
    except KeyError:
        raise ValueError(f"Unknown bundle flavour {flavor!r}") from None


def library_candidates(backend: str) -> list[Path]:
    """Where a library for ``backend`` may be, in order: SCYLLASBAND_NATIVE_LIB, the package's _native/<backend>/, a
    checkout's libscyllasband/build/<backend>/."""
    out = []
    if os.environ.get(LIBRARY_ENV):
        out.append(Path(os.environ[LIBRARY_ENV]).expanduser())
    out.append(PACKAGE_DIR / backend / library_name())
    out.append(CHECKOUT_DIR / "build" / backend / library_name())
    return out


def load_library(backend: str, path: str | Path | None = None) -> NativeLibrary:
    """The first usable library for ``backend`` (or the one at ``path``); NativeUnavailable says why there is none."""
    reasons = []
    for candidate in ([Path(path)] if path else library_candidates(backend)):
        if not candidate.is_file():
            continue
        key = candidate.resolve()
        with _LOCK:
            library = _LIBRARIES.get(key)
            if library is None:
                try:
                    library = NativeLibrary(candidate)
                except NativeUnavailable as exc:
                    reasons.append(str(exc))
                    continue
                _LIBRARIES[key] = library
        if library.backend != backend:
            reasons.append(f"{candidate} is built for {library.backend}, not {backend}")
            continue
        return library
    detail = "; ".join(reasons) if reasons else "none found"
    raise NativeUnavailable(f"no native {backend} library ({detail})")


# --- helpers -------------------------------------------------------------------------------------------------------------
def _text(value: Any, what: str) -> bytes:
    text = str(value)
    if "\x00" in text:
        raise ValueError(f"{what} contains a NUL character")
    return text.encode("utf-8")


def _optional_text(value: Any, what: str) -> bytes | None:
    return None if value is None else _text(value, what)


def _check_strings(value: Any, what: str) -> None:
    if isinstance(value, str):
        if "\x00" in value:
            raise ValueError(f"{what} contains a NUL character")
    elif isinstance(value, Mapping):
        for item in value.values():
            _check_strings(item, what)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _check_strings(item, what)


def _json_bytes(value: Any, what: str) -> bytes:
    _check_strings(value, what)
    return json.dumps(value, ensure_ascii=False, default=_json_default).encode("utf-8")


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _score_bytes(score: Mapping[str, Any] | str) -> bytes:
    if isinstance(score, str):
        return _text(score, "score")
    return _json_bytes(score, "score")


def check_seed(seed: Any) -> tuple[int, int]:
    """(seed, has_seed) for the C API; a seed must be an integer from 0 to 2**64 - 1."""
    if seed is None:
        return 0, 0
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"seed must be an integer; got {seed!r}")
    value = int(seed)
    if not 0 <= value < 2 ** 64:
        raise ValueError(f"seed must be between 0 and 2**64 - 1; got {value}")
    return value, 1


def _sampler(value: str | None) -> int:
    key = None if value is None else str(value).lower()
    if key not in SAMPLERS:
        raise ValueError(f"Unsupported sampler {value!r}; choose one of {', '.join(SUPPORTED_SAMPLERS)}")
    return SAMPLERS[key]


def _progress(stream: TextIO | None, message: str) -> None:
    if stream is not None:
        print(message, file=stream, flush=True)


# --- the runtime ---------------------------------------------------------------------------------------------------------
class NativeRuntime:
    """A bundle on libscyllasband, with the ScyllasBandRuntime surface the CLI uses."""

    engine_name = "native"

    def __init__(self, bundle_dir: str | Path, *, backend: str | None = None, threads: int | None = None,
                 compute_units: str | Mapping[str, str] | None = None, validate: bool = True, library: NativeLibrary | None = None):
        from .contract import validate_bundle_layout
        self.bundle_dir = Path(bundle_dir)
        if validate:
            validate_bundle_layout(self.bundle_dir)
        self.backend = backend or _bundle_backend(self.bundle_dir)
        wanted = library_backend(self.backend)
        self.library = library or load_library(wanted)
        if self.library.backend != wanted:
            raise NativeUnavailable(f"{self.library.path} is built for {self.library.backend}, not {wanted}")
        self.compute_units = compute_units
        accelerator = self._accelerator(compute_units)
        if threads is not None and int(threads) < 0:
            raise ValueError("threads must be zero or positive")
        options = RuntimeOptions(_text(self.bundle_dir, "bundle_dir"), int(threads or 0), accelerator)
        handle = c.c_void_p()
        self.library.check(self.library.lib.scyllasband_runtime_create(c.byref(options), c.byref(handle)), "runtime_create")
        self.handle = handle
        self.info = json.loads(self.library.lib.scyllasband_voices_json(self.handle).decode("utf-8"))
        self.sample_rate = int(self.info["sample_rate"])
        self.voices = {v["id"]: VoiceSpec(id=v["id"], languages=tuple(v["languages"]), default_language=v["default_language"])
                       for v in self.info["voices"]}
        manifest = json.loads((self.bundle_dir / "manifest.json").read_text(encoding="utf-8"))
        self.manifest = manifest
        controls = manifest["controls"]
        self.default_steps = int(controls.get("steps", {}).get("default", 8))
        self.default_sampler = str(controls.get("sampler", {}).get("default", "heun"))

    def _accelerator(self, compute_units: str | Mapping[str, str] | None) -> int:
        """Core ML / Core AI: auto (the bundle's recommendation), cpu, gpu, ane, or per graph ("gpu,g2p=cpu", set as
        SCYLLASBAND_COMPUTE_UNITS, which the native backend reads as each graph loads). Other backends run on the CPU."""
        if isinstance(compute_units, Mapping):
            compute_units = ",".join(f"{k}={v}" if k != "default" else str(v) for k, v in compute_units.items())
        value = str(compute_units or "").strip().lower()
        if library_backend(self.backend) != "apple":
            return ACCELERATORS["cpu"]
        if value in ("", "auto"):
            return ACCELERATORS["auto"]
        if value in ACCELERATORS:
            return ACCELERATORS[value]
        os.environ["SCYLLASBAND_COMPUTE_UNITS"] = value
        return ACCELERATORS["auto"]

    # --- lifecycle ---------------------------------------------------------------------------------------------------
    def close(self) -> None:
        handle, self.handle = getattr(self, "handle", None), None
        if handle is not None and handle.value:
            self.library.lib.scyllasband_runtime_destroy(handle)

    def __enter__(self) -> "NativeRuntime":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def cancel(self) -> None:
        """Stops the calls already issued on this runtime at their next graph call (safe from any thread)."""
        if self.handle is not None:
            self.library.lib.scyllasband_cancel(self.handle)

    # --- voices ------------------------------------------------------------------------------------------------------
    def available_voices(self) -> list[str]:
        return sorted(self.voices)

    def voice_for_id(self, voice_id: str) -> VoiceSpec:
        if voice_id not in self.voices:
            raise ValueError(f"Unknown voice {voice_id!r}; available: {', '.join(self.available_voices())}")
        return self.voices[voice_id]

    def resolve_language_for_voice(self, voice_id: str, language: str | None = None) -> str:
        voice = self.voice_for_id(voice_id)
        if language is None or not str(language).strip():
            return voice.default_language
        value = str(language).strip().lower().replace("-", "_")
        if value in ("en", "english"):
            return voice.default_language
        if value not in voice.languages:
            raise ValueError(f"Voice {voice_id!r} supports {', '.join(voice.languages)}; got {language!r}")
        return value

    @property
    def supports_singing(self) -> bool:
        return bool((self.info.get("singing") or {}).get("supported"))

    # --- calls -------------------------------------------------------------------------------------------------------
    def _events(self, start: Callable[[Any], int], what: str) -> Iterator[NativeEvent]:
        """Runs ``start(callback)`` on a worker thread and yields its events as they arrive (a bounded queue holds them back,
        so a slow consumer slows the native side). Closing the generator early, or an exception in the consumer
        (KeyboardInterrupt included), cancels the call and joins the thread; an exception while copying an event cancels
        the call and is raised after it returns."""
        handle = self.handle
        if handle is None:
            raise NativeError("This runtime is closed")
        events: queue.Queue = queue.Queue(maxsize=4)
        stop = threading.Event()
        failure: list[BaseException] = []
        result: dict[str, Any] = {}

        def deliver(item) -> bool:
            while not stop.is_set():
                try:
                    events.put(item, timeout=0.05)
                    return True
                except queue.Full:
                    continue
            return False

        def on_event(pointer, _user) -> int:
            if stop.is_set():
                return 1
            try:
                return 0 if deliver(copy_event(pointer.contents)) else 1
            except BaseException as exc:   # never let an exception unwind into C: cancel and raise it after the call
                failure.append(exc)
                return 1

        callback = EventCallback(on_event)

        def run() -> None:
            try:
                status = start(callback)
                result.update(status=status, message=self.library.last_error() if status else "")
            except BaseException as exc:
                failure.append(exc)
            deliver(_DONE)

        thread = threading.Thread(target=run, name=f"scyllasband-{what}", daemon=True)
        thread.start()
        finished = False
        try:
            while True:
                try:
                    item = events.get(timeout=0.1)
                except queue.Empty:
                    if not thread.is_alive() and events.empty():
                        break
                    continue
                if item is _DONE:
                    finished = True
                    break
                yield item
        finally:
            if not finished:
                stop.set()
                self.library.lib.scyllasband_cancel(handle)
            thread.join()
            del callback
        if failure:
            raise failure[0]
        status = result.get("status", 0)
        if status:
            message = result.get("message") or f"{what} failed"
            if status == 1:
                raise NativeInvalidArgument(message)
            if status == 3:
                raise NativeCancelled(message)
            raise NativeError(message)

    def _records_request(self, records: Sequence[Mapping[str, Any]], *, normalize: bool, steps: int | None = None, sampler: str | None = None,
                         speed: float = 1.0, seed: int | None = None, temperature: float = 1.0, pause_ms: float = 0.0) -> RecordsRequest:
        request = RecordsRequest()
        self.library.lib.scyllasband_records_request_init(c.byref(request))
        data = _json_bytes([dict(record) for record in records], "records")
        request._keep = data   # the bytes live as long as the request
        request.records_json = data
        if float(speed) <= 0:
            raise ValueError("speed must be positive")
        request.speed, request.steps, request.sampler = float(speed), int(steps or 0), _sampler(sampler)
        request.seed, request.has_seed = check_seed(seed)
        request.temperature, request.normalize_text, request.pause_ms = float(temperature), int(bool(normalize)), float(pause_ms)
        return request

    def plan_records(self, records: Sequence[Mapping[str, Any]], *, normalize: bool = True, source_kind: str = "text") -> SynthesisPlan:
        out = c.c_void_p()
        request = self._records_request(records, normalize=normalize)
        self.library.check(self.library.lib.scyllasband_plan_records_json(self.handle, c.byref(request), c.byref(out)), "plan_records")
        data = json.loads(self.library.take_string(out))
        return SynthesisPlan(chunks=[PlanChunk(**chunk) for chunk in data["chunks"]], source_kind=source_kind)

    def plan_text(self, text: str, *, voice_id: str, language: str | None = None, delivery: Any = None, intensity: float | None = None,
                  pace: float | None = None, pitch: float | None = None, normalize: bool = True) -> SynthesisPlan:
        return self.plan_records(records_from_text(text, voice=voice_id, language=language, delivery=delivery, intensity=intensity,
                                                   pace=pace, pitch=pitch), normalize=normalize)

    def synthesize_stream(self, records: Sequence[Mapping[str, Any]], *, normalize: bool = True, progress: TextIO | None = None,
                          **options: Any) -> Iterator[StreamingEvent]:
        """The records spoken target by target, as StreamingEvents (plan, chunk_started, audio_chunk, done)."""
        request = self._records_request(records, normalize=normalize, **options)
        lib = self.library.lib
        events = self._events(lambda callback: lib.scyllasband_synthesize_records_stream(self.handle, c.byref(request), callback, None),
                              "synthesize")
        return self._speech_events(events, request, progress)

    def _speech_events(self, events: Iterator[NativeEvent], request: RecordsRequest, progress: TextIO | None) -> Iterator[StreamingEvent]:
        try:
            for event in events:
                kind = EVENT_TYPES.get(event.type, "unknown")
                if kind == "chunk_started":
                    chunk = PlanChunk(**event.metadata)
                    _progress(progress, f"[scyllasband] {chunk.chunk_id} {chunk.voice}/{chunk.language}: {chunk.text[:80]}")
                    yield StreamingEvent(type=kind, chunk=chunk)
                elif kind == "audio_chunk":
                    metadata = dict(event.metadata)
                    chunk = PlanChunk(**metadata.pop("chunk"))
                    yield StreamingEvent(type=kind, chunk=chunk, audio=event.samples, sample_rate=event.sample_rate,
                                         metadata=dict(metadata, engine="native", noise=NOISE["native"]))
                elif kind == "done":
                    yield StreamingEvent(type=kind, sample_rate=self.sample_rate, metadata=dict(event.metadata, engine="native"))
                else:
                    yield StreamingEvent(type=kind, metadata=event.metadata)
        finally:
            events.close()
            del request

    def render_records(self, records: Sequence[Mapping[str, Any]], *, normalize: bool = True, progress: TextIO | None = None,
                       **options: Any) -> SynthesisResult:
        pieces, chunks, summary = [], [], {}
        stream = self.synthesize_stream(records, normalize=normalize, progress=progress, **options)
        try:
            for event in stream:
                if event.type == "audio_chunk" and event.audio is not None:
                    pieces.append(event.audio)
                    chunks.append(dict(chunk=event.chunk.to_dict() if event.chunk else None, **event.metadata))
                elif event.type == "done":
                    summary = event.metadata
        finally:
            stream.close()
        audio = np.concatenate(pieces) if pieces else np.zeros(0, np.float32)
        return SynthesisResult(audio=audio, sample_rate=self.sample_rate,
                               metadata=dict(summary, chunks=chunks, backend=self.backend, engine="native", noise=NOISE["native"]))

    def synthesize(self, request: SynthesisRequest, *, progress: TextIO | None = None) -> SynthesisResult:
        records = records_from_text(request.text, voice=request.voice_id, language=request.language, delivery=request.delivery,
                                    intensity=request.intensity, pace=request.pace, pitch=request.pitch)
        steps = request.steps or self.default_steps
        sampler = (request.sampler or self.default_sampler).lower()
        result = self.render_records(records, normalize=request.normalize_text, progress=progress, steps=steps, sampler=sampler,
                                     speed=request.speed, seed=request.seed, temperature=request.temperature)
        if not result.metadata["chunks"]:
            raise ValueError("Nothing to speak: the text has no words")
        first = result.metadata["chunks"][0]
        result.metadata.update(bundle_dir=str(self.bundle_dir), voice_id=request.voice_id, language=first["chunk"]["language"],
                               delivery=first.get("delivery"), steps=steps, sampler=sampler, speed=request.speed, seed=request.seed)
        return result

    # --- singing -----------------------------------------------------------------------------------------------------
    def _sing_request(self, score: Mapping[str, Any] | str, voice_id: str | None, language: str | None = None, *, seed: int | None = None,
                      steps: int | None = None, sampler: str | None = None, decoder: str | None = None, guidance: float | None = None,
                      register: bool = True, transpose: int = 0) -> SingRequest:
        request = SingRequest()
        self.library.lib.scyllasband_sing_request_init(c.byref(request))
        request._keep = [_score_bytes(score), _optional_text(voice_id, "voice"), _optional_text(language, "language"),
                         _optional_text(decoder, "decoder")]
        request.score_json, request.voice_id, request.language, request.decoder = request._keep
        request.transpose, request.use_register = int(transpose), int(bool(register))
        request.steps, request.sampler = int(steps or 0), _sampler(sampler)
        request.seed, request.has_seed = check_seed(seed)
        if guidance is not None:
            request.guidance, request.has_guidance = float(guidance), 1
        return request

    def sing_events(self, score: Mapping[str, Any] | str, voice_id: str, language: str | None = None, **options: Any) -> Iterator[NativeEvent]:
        """The sung stream: PLAN, a CHUNK_STARTED per phrase, per phrase the song audio it settles (contiguous), DONE."""
        request = self._sing_request(score, voice_id, language, **options)
        lib = self.library.lib
        events = self._events(lambda callback: lib.scyllasband_sing_stream(self.handle, c.byref(request), callback, None), "sing")
        try:
            yield from events
        finally:
            events.close()
            del request

    def sing(self, score: Mapping[str, Any], voice_id: str, language: str | None = None, *, seed: int | None = None,
             steps: int | None = None, sampler: str | None = None, decoder: str | None = None, guidance: float | None = None,
             register: bool = True, transpose: int = 0, progress: TextIO | None = None) -> SynthesisResult:
        if not self.supports_singing:
            raise ValueError("Singing needs a v4 bundle with delivery modes; this bundle is spoken only. Download a current model: "
                             "python -m scyllasband download --yes")
        language = self.resolve_language_for_voice(voice_id, language)
        pieces, phrases, plan, summary = [], [], {}, {}
        events = self.sing_events(score, voice_id, language, seed=seed, steps=steps, sampler=sampler, decoder=decoder, guidance=guidance,
                                  register=register, transpose=transpose)
        try:
            for event in events:
                if event.type == 1:
                    plan = event.metadata
                elif event.type == 2:
                    _progress(progress, f"[scyllasband] phrase {event.chunk_index + 1}/{event.chunk_count} {voice_id}/{language} "
                                        f"{plan.get('mode')}: {event.metadata.get('text', '')[:80]}")
                elif event.type == 3:
                    pieces.append(event.samples)
                    phrases.append({k: v for k, v in event.metadata.items()
                                    if k not in ("start_sample", "sample_count", "song_samples", "decoder", "first_audio_ms", "elapsed_ms")})
                elif event.type == 4:
                    summary = event.metadata
        finally:
            events.close()
        audio = np.concatenate(pieces) if pieces else np.zeros(0, np.float32)
        metadata = dict(plan, phrases=phrases, seed=seed, graph_contract=self.info.get("graph_contract"),
                        first_audio_ms=summary.get("first_audio_ms"), elapsed_ms=summary.get("elapsed_ms"), backend=self.backend,
                        bundle_dir=str(self.bundle_dir), voice_id=voice_id, engine="native", noise=NOISE["native"])
        return SynthesisResult(audio=audio, sample_rate=self.sample_rate, metadata=metadata)

    def score_parse(self, score: Mapping[str, Any] | str, transpose: int = 0) -> dict[str, Any]:
        return self.library.score_parse(score, transpose)

    def sing_plan(self, score: Mapping[str, Any] | str, voice_id: str, language: str | None = None, **options: Any) -> dict[str, Any]:
        out = c.c_void_p()
        request = self._sing_request(score, voice_id, language, **options)
        self.library.check(self.library.lib.scyllasband_sing_plan_json(self.handle, c.byref(request), c.byref(out)), "sing_plan")
        return json.loads(self.library.take_string(out))

    def sing_fit(self, score: Mapping[str, Any] | str, voice_id: str, *, transpose: int = 0, register: bool = True) -> dict[str, Any]:
        """How the score sits in the voice's range (no graph runs)."""
        out = c.c_void_p()
        request = self._sing_request(score, voice_id, transpose=transpose, register=register)
        self.library.check(self.library.lib.scyllasband_sing_fit_json(self.handle, c.byref(request), c.byref(out)), "sing_fit")
        return json.loads(self.library.take_string(out))

    def sing_preview(self, score: Mapping[str, Any] | str, voice_id: str | None = None, *, transpose: int = 0, register: bool = True,
                     sample_rate: int = 0) -> SynthesisResult:
        """The score as plain tones at the pitch the voice will sing it (no graph runs)."""
        request = self._sing_request(score, voice_id, transpose=transpose, register=register and voice_id is not None)
        audio = Audio()
        lib = self.library.lib
        try:
            self.library.check(lib.scyllasband_sing_preview(self.handle if voice_id is not None else None, c.byref(request), int(sample_rate),
                                                            c.byref(audio)), "sing_preview")
            samples = np.ctypeslib.as_array(audio.samples, shape=(audio.sample_count,)).copy() if audio.sample_count else np.zeros(0, np.float32)
            return SynthesisResult(audio=samples, sample_rate=int(audio.sample_rate), metadata=json.loads(audio.metadata_json.decode("utf-8")))
        finally:
            lib.scyllasband_audio_free(c.byref(audio))

    # --- status ------------------------------------------------------------------------------------------------------
    def warmup(self, *, voice_id: str | None = None) -> dict[str, Any]:
        voice = voice_id or self.available_voices()[0]
        self.library.check(self.library.lib.scyllasband_warmup(self.handle, _text(voice, "voice")), "warmup")
        return dict(voice=voice, engine="native")

    def runtime_status(self) -> dict[str, Any]:
        singing = self.info.get("singing") or {}
        return dict(backend=self.backend, bundle_dir=str(self.bundle_dir), voices=self.available_voices(), engine="native",
                    noise=NOISE["native"], library=str(self.library.path), abi=self.library.version, accelerator=self.info.get("accelerator"),
                    graph_contract=self.info.get("graph_contract"), singing=bool(singing.get("supported")),
                    decoders=sorted(singing.get("decoders") or []))


# --- choosing the engine -------------------------------------------------------------------------------------------------
def _wants_gpu(compute_units: Any) -> bool:
    return any(part.split("=")[-1].strip() in ("gpu", "cuda") for part in str(compute_units or "").split(","))


def open_runtime(bundle_dir: str | Path, *, engine: str | None = None, backend: str | None = None, threads: int | None = None,
                 compute_units: str | Mapping[str, str] | None = None, validate: bool = True, log: TextIO | None = sys.stderr):
    """A runtime for the bundle on the chosen engine: ``native``, ``python``, or by default (SCYLLASBAND_ENGINE, else
    ``auto``) native where a library for the bundle's backend is found, else the Python engine with one notice line."""
    choice = str(engine or os.environ.get(ENGINE_ENV) or "auto").strip().lower()
    if choice not in ("auto", "native", "python"):
        raise ValueError(f"Unknown engine {choice!r}; choose native, python or auto")
    bundle_dir = Path(bundle_dir)
    flavor = backend or _bundle_backend(bundle_dir)

    def python_runtime():
        return ScyllasBandRuntime.from_bundle(bundle_dir, backend=backend, threads=threads, compute_units=compute_units, validate=validate)

    if choice == "python":
        return python_runtime()
    if choice == "auto" and (library_backend(flavor) == "onnx" or (flavor == "onnx" and _wants_gpu(compute_units))):
        return python_runtime()   # ONNX Runtime with CUDA and the WebGPU bundle stay on the Python engine
    try:
        return NativeRuntime(bundle_dir, backend=flavor, threads=threads, compute_units=compute_units, validate=validate)
    except NativeUnavailable as exc:
        if choice == "native":
            raise NativeUnavailable(f"{exc}. Build it with: python -m scyllasband build-native --backend {library_backend(flavor)}, "
                                    f"or set {LIBRARY_ENV}") from None
        if log is not None:
            print(f"[scyllasband] Python engine: {exc} (python -m scyllasband build-native --backend {library_backend(flavor)} "
                  "builds the native runtime)", file=log, flush=True)
        return python_runtime()
