"""ctypes bindings for the native libscyllasband runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import ctypes
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys

from .contract import SUPPORTED_AFFECT_AXES

SCYLLASBAND_STATUS_OK = 0
SCYLLASBAND_STATUS_INVALID_ARGUMENT = 1
SCYLLASBAND_STATUS_NOT_IMPLEMENTED = 2
SCYLLASBAND_STATUS_RUNTIME_ERROR = 3

SCYLLASBAND_BACKEND_AUTO = 0
SCYLLASBAND_BACKEND_LITERT = 1
SCYLLASBAND_BACKEND_COREML = 2
SCYLLASBAND_BACKEND_ONNX = 3

SCYLLASBAND_SAMPLER_EULER = 0
SCYLLASBAND_SAMPLER_HEUN = 1

SCYLLASBAND_LITERT_ACCELERATOR_AUTO = 0
SCYLLASBAND_LITERT_ACCELERATOR_CPU = 1
SCYLLASBAND_LITERT_ACCELERATOR_GPU = 2
SCYLLASBAND_LITERT_ACCELERATOR_NPU = 3

SCYLLASBAND_STREAM_EVENT_PLAN_READY = 1
SCYLLASBAND_STREAM_EVENT_CHUNK_STARTED = 2
SCYLLASBAND_STREAM_EVENT_AUDIO_CHUNK = 3
SCYLLASBAND_STREAM_EVENT_CHUNK_FINISHED = 4
SCYLLASBAND_STREAM_EVENT_WARNING = 5
SCYLLASBAND_STREAM_EVENT_DONE = 6

_BACKEND_IDS = {
    "auto": SCYLLASBAND_BACKEND_AUTO,
    "onnx": SCYLLASBAND_BACKEND_ONNX,
    "litert": SCYLLASBAND_BACKEND_LITERT,
    "coreml": SCYLLASBAND_BACKEND_COREML,
}
_SAMPLER_IDS = {"euler": SCYLLASBAND_SAMPLER_EULER, "heun": SCYLLASBAND_SAMPLER_HEUN}
_LITERT_ACCELERATOR_IDS = {
    "auto": SCYLLASBAND_LITERT_ACCELERATOR_AUTO,
    "cpu": SCYLLASBAND_LITERT_ACCELERATOR_CPU,
    "gpu": SCYLLASBAND_LITERT_ACCELERATOR_GPU,
    "npu": SCYLLASBAND_LITERT_ACCELERATOR_NPU,
}
_STREAM_EVENT_NAMES = {
    SCYLLASBAND_STREAM_EVENT_PLAN_READY: "plan_ready",
    SCYLLASBAND_STREAM_EVENT_CHUNK_STARTED: "chunk_started",
    SCYLLASBAND_STREAM_EVENT_AUDIO_CHUNK: "audio_chunk",
    SCYLLASBAND_STREAM_EVENT_CHUNK_FINISHED: "chunk_finished",
    SCYLLASBAND_STREAM_EVENT_WARNING: "warning",
    SCYLLASBAND_STREAM_EVENT_DONE: "done",
}

_AUTO_BUILD_ENV = "SCYLLASBAND_NATIVE_AUTO_BUILD"
_AUTO_STAGE_LITERT_ENV = "SCYLLASBAND_NATIVE_AUTO_STAGE_LITERT"
_AUTO_DOWNLOAD_LITERT_ENV = "SCYLLASBAND_NATIVE_AUTO_DOWNLOAD_LITERT"
_BUILD_DIR_ENV = "SCYLLASBAND_NATIVE_BUILD_DIR"
_BUILD_TYPE_ENV = "SCYLLASBAND_NATIVE_BUILD_TYPE"
_BUILD_JOBS_ENV = "SCYLLASBAND_NATIVE_BUILD_JOBS"
_CMAKE_ENV = "SCYLLASBAND_NATIVE_CMAKE"
_QUIET_BUILD_ENV = "SCYLLASBAND_NATIVE_BUILD_QUIET"
_LITERT_RUNTIME_LIBS = {
    "linux-x86_64": ("libLiteRt.so",),
    "linux-arm64": ("libLiteRt.so",),
    "macos-arm64": ("libLiteRt.dylib",),
    "windows-x86_64": ("libLiteRt.dll", "LiteRt.dll"),
}
_DLL_DIRECTORY_HANDLES: list[Any] = []
_DLL_DIRECTORY_PATHS: set[str] = set()


class ScyllasBandNativeError(RuntimeError):
    """Raised when libscyllasband cannot be loaded or returns an error."""


class _ScyllasBandRuntimeOptions(ctypes.Structure):
    _fields_ = [
        ("bundle_dir", ctypes.c_char_p),
        ("backend", ctypes.c_int),
        ("validate_bundle", ctypes.c_int32),
        ("litert_accelerator", ctypes.c_int),
        ("litert_max_threads", ctypes.c_int32),
    ]


class _ScyllasBandSynthesisRequest(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("explicit_phones", ctypes.c_char_p),
        ("voice_id", ctypes.c_char_p),
        ("language", ctypes.c_char_p),
        ("emotion", ctypes.c_char_p),
        ("emotion_guidance", ctypes.c_char_p),
        ("guidance_null_reference", ctypes.c_int32),
        ("emotion_embed_scale", ctypes.c_float),
        ("prefix_latents", ctypes.POINTER(ctypes.c_float)),
        ("prefix_latent_dim", ctypes.c_int32),
        ("prefix_latent_frames", ctypes.c_int32),
        ("context_before", ctypes.c_char_p),
        ("context_after", ctypes.c_char_p),
        ("chunk_index", ctypes.c_int32),
        ("chunk_count", ctypes.c_int32),
        ("boundary_before", ctypes.c_char_p),
        ("boundary_after", ctypes.c_char_p),
        ("min_sentence_pause_ms", ctypes.c_float),
        ("min_clause_pause_ms", ctypes.c_float),
        ("steps", ctypes.c_int32),
        ("sampler", ctypes.c_int),
        ("seed", ctypes.c_uint64),
        ("has_seed", ctypes.c_int32),
        ("speed", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("affect", ctypes.c_char_p),
        ("affect_guidance_scale", ctypes.c_float),
        ("has_affect_guidance_scale", ctypes.c_int32),
    ]


class _ScyllasBandSynthesisResult(ctypes.Structure):
    _fields_ = [
        ("samples", ctypes.POINTER(ctypes.c_float)),
        ("sample_count", ctypes.c_int32),
        ("sample_rate", ctypes.c_int32),
        ("metadata_json", ctypes.c_char_p),
        ("latents", ctypes.POINTER(ctypes.c_float)),
        ("latent_dim", ctypes.c_int32),
        ("latent_frames", ctypes.c_int32),
    ]


class _ScyllasBandDurationEstimateResult(ctypes.Structure):
    _fields_ = [
        ("predicted_latent_frames", ctypes.c_int32),
        ("fixed_latent_frames", ctypes.c_int32),
        ("metadata_json", ctypes.c_char_p),
    ]


class _ScyllasBandLongFormSynthesisRequest(ctypes.Structure):
    _fields_ = [
        ("request", _ScyllasBandSynthesisRequest),
        ("max_chunk_chars", ctypes.c_int32),
        ("min_chunk_chars", ctypes.c_int32),
        ("pause_ms", ctypes.c_int32),
        ("continuation_pause_ms", ctypes.c_int32),
        ("use_prefix_latents", ctypes.c_int32),
        ("disable_auto_split_overlong", ctypes.c_int32),
        ("preflight_chunks", ctypes.c_int32),
    ]


class _ScyllasBandChunkPlanRequest(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("max_chunk_chars", ctypes.c_int32),
        ("min_chunk_chars", ctypes.c_int32),
    ]


class _ScyllasBandChunkPlanResult(ctypes.Structure):
    _fields_ = [("metadata_json", ctypes.c_char_p)]


class _ScyllasBandStreamingEvent(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("chunk_index", ctypes.c_int32),
        ("chunk_count", ctypes.c_int32),
        ("chunk_id", ctypes.c_char_p),
        ("metadata_json", ctypes.c_char_p),
        ("samples", ctypes.POINTER(ctypes.c_float)),
        ("sample_count", ctypes.c_int32),
        ("sample_rate", ctypes.c_int32),
        ("latents", ctypes.POINTER(ctypes.c_float)),
        ("latent_dim", ctypes.c_int32),
        ("latent_frames", ctypes.c_int32),
    ]


_ScyllasBandStreamingCallback = ctypes.CFUNCTYPE(
    ctypes.c_int32,
    ctypes.POINTER(_ScyllasBandStreamingEvent),
    ctypes.c_void_p,
)


@dataclass(frozen=True)
class NativeSynthesisResult:
    audio: list[float]
    sample_rate: int
    duration_seconds: float
    metadata: dict[str, Any]
    latents: list[float] | None = None
    latent_dim: int = 0
    latent_frames: int = 0


@dataclass(frozen=True)
class NativeDurationEstimate:
    predicted_latent_frames: int
    fixed_latent_frames: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class NativeStreamingEvent:
    event_type: str
    event_type_id: int
    chunk_index: int
    chunk_count: int
    chunk_id: str | None
    metadata: dict[str, Any]
    audio: list[float]
    sample_count: int
    sample_rate: int
    latents: list[float] | None = None
    latent_dim: int = 0
    latent_frames: int = 0


def load_library(path: str | Path | None = None, *, backend: str = "onnx") -> ctypes.CDLL:
    library_path = _resolve_library_path(path, backend=backend)
    _prepare_dynamic_loader(library_path)
    try:
        lib = ctypes.CDLL(str(library_path))
    except OSError as exc:
        rebuild_error: BaseException | None = None
        if _can_auto_build(path):
            root = Path(__file__).resolve().parents[1]
            try:
                _auto_build_native_library(
                    root,
                    reason=f"failed to load {library_path.name}",
                    backend=backend,
                )
                library_path = _resolve_library_path(None, auto_build=False, backend=backend)
                _prepare_dynamic_loader(library_path)
                lib = ctypes.CDLL(str(library_path))
            except (ScyllasBandNativeError, OSError) as build_exc:
                rebuild_error = build_exc
            else:
                _configure_library(lib)
                return lib
        message = f"Could not load libscyllasband from {library_path}: {exc}"
        if rebuild_error is not None:
            message += f". Automatic native build/reload failed: {rebuild_error}"
        raise ScyllasBandNativeError(message) from exc
    _configure_library(lib)
    return lib


def plan_long_form_chunks(text: str, *, max_chunk_chars: int, min_chunk_chars: int, library_path: str | Path | None = None) -> dict[str, Any]:
    lib = load_library(library_path)
    request = _ScyllasBandChunkPlanRequest(
        text=_bytes(text),
        max_chunk_chars=int(max_chunk_chars),
        min_chunk_chars=int(min_chunk_chars),
    )
    result = _ScyllasBandChunkPlanResult()
    status = lib.scyllasband_plan_long_form_chunks(ctypes.byref(request), ctypes.byref(result))
    try:
        _check_status(lib, status)
        payload = _decode_json(result.metadata_json)
        return payload
    finally:
        lib.scyllasband_chunk_plan_result_free(ctypes.byref(result))


class NativeScyllasBandRuntime:
    def __init__(
        self,
        bundle_dir: str | Path,
        *,
        backend: str = "onnx",
        validate_bundle: bool = True,
        library_path: str | Path | None = None,
        litert_accelerator: str = "cpu",
        max_cached_target_buckets: int = 0,
    ) -> None:
        self.bundle_dir = Path(bundle_dir)
        self.backend = backend
        self.litert_accelerator = litert_accelerator
        self.max_cached_target_buckets = int(max_cached_target_buckets)
        if self.max_cached_target_buckets < 0:
            raise ValueError("max_cached_target_buckets must be non-negative")
        self._lib = load_library(library_path, backend=backend)
        self._handle = ctypes.c_void_p()
        options = _ScyllasBandRuntimeOptions(
            bundle_dir=_bytes(str(self.bundle_dir)),
            backend=_backend_id(backend),
            validate_bundle=1 if validate_bundle else 0,
            litert_accelerator=_litert_accelerator_id(litert_accelerator),
            litert_max_threads=0,
        )
        status = self._lib.scyllasband_runtime_create(ctypes.byref(options), ctypes.byref(self._handle))
        _check_status(self._lib, status)
        try:
            status = self._lib.scyllasband_runtime_set_target_bucket_cache_capacity(
                self._handle,
                self.max_cached_target_buckets,
            )
            _check_status(self._lib, status)
        except BaseException:
            self._lib.scyllasband_runtime_destroy(self._handle)
            self._handle = ctypes.c_void_p()
            raise

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle and handle.value:
            self._lib.scyllasband_runtime_destroy(handle)
            self._handle = ctypes.c_void_p()

    def __enter__(self) -> "NativeScyllasBandRuntime":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def synthesize(self, request: Any) -> NativeSynthesisResult:
        native_request, _prefix_buffer = _build_synthesis_request(request)
        result = _ScyllasBandSynthesisResult()
        status = self._lib.scyllasband_runtime_synthesize(self._handle, ctypes.byref(native_request), ctypes.byref(result))
        try:
            _check_status(self._lib, status)
            return _convert_result(result)
        finally:
            self._lib.scyllasband_synthesis_result_free(ctypes.byref(result))

    def estimate_latent_frames(self, request: Any) -> NativeDurationEstimate:
        native_request, _prefix_buffer = _build_synthesis_request(request)
        result = _ScyllasBandDurationEstimateResult()
        status = self._lib.scyllasband_runtime_estimate_latent_frames(
            self._handle,
            ctypes.byref(native_request),
            ctypes.byref(result),
        )
        try:
            _check_status(self._lib, status)
            return _convert_duration_estimate(result)
        finally:
            self._lib.scyllasband_duration_estimate_result_free(ctypes.byref(result))

    def plan_long_form(
        self,
        request: Any,
        *,
        max_chunk_chars: int,
        min_chunk_chars: int,
        pause_ms: int,
        continuation_pause_ms: int,
        use_prefix_latents: bool = True,
        disable_auto_split_overlong: bool = False,
        preflight_chunks: bool = False,
    ) -> dict[str, Any]:
        native_request, _prefix_buffer = _build_synthesis_request(request)
        long_request = _ScyllasBandLongFormSynthesisRequest(
            request=native_request,
            max_chunk_chars=int(max_chunk_chars),
            min_chunk_chars=int(min_chunk_chars),
            pause_ms=int(pause_ms),
            continuation_pause_ms=int(continuation_pause_ms),
            use_prefix_latents=1 if use_prefix_latents else 0,
            disable_auto_split_overlong=1 if disable_auto_split_overlong else 0,
            preflight_chunks=1 if preflight_chunks else 0,
        )
        result = _ScyllasBandChunkPlanResult()
        status = self._lib.scyllasband_runtime_plan_long_form(self._handle, ctypes.byref(long_request), ctypes.byref(result))
        try:
            _check_status(self._lib, status)
            return _decode_json(result.metadata_json)
        finally:
            self._lib.scyllasband_chunk_plan_result_free(ctypes.byref(result))

    def synthesize_long_form(
        self,
        request: Any,
        *,
        max_chunk_chars: int,
        min_chunk_chars: int,
        pause_ms: int,
        continuation_pause_ms: int,
        use_prefix_latents: bool = True,
        disable_auto_split_overlong: bool = False,
        preflight_chunks: bool = False,
    ) -> NativeSynthesisResult:
        native_request, _prefix_buffer = _build_synthesis_request(request)
        long_request = _ScyllasBandLongFormSynthesisRequest(
            request=native_request,
            max_chunk_chars=int(max_chunk_chars),
            min_chunk_chars=int(min_chunk_chars),
            pause_ms=int(pause_ms),
            continuation_pause_ms=int(continuation_pause_ms),
            use_prefix_latents=1 if use_prefix_latents else 0,
            disable_auto_split_overlong=1 if disable_auto_split_overlong else 0,
            preflight_chunks=1 if preflight_chunks else 0,
        )
        result = _ScyllasBandSynthesisResult()
        status = self._lib.scyllasband_runtime_synthesize_long_form(self._handle, ctypes.byref(long_request), ctypes.byref(result))
        try:
            _check_status(self._lib, status)
            return _convert_result(result)
        finally:
            self._lib.scyllasband_synthesis_result_free(ctypes.byref(result))

    def synthesize_long_form_stream(
        self,
        request: Any,
        *,
        max_chunk_chars: int,
        min_chunk_chars: int,
        pause_ms: int,
        continuation_pause_ms: int,
        callback: Any,
        use_prefix_latents: bool = True,
        disable_auto_split_overlong: bool = False,
        preflight_chunks: bool = False,
    ) -> None:
        native_request, _prefix_buffer = _build_synthesis_request(request)
        long_request = _ScyllasBandLongFormSynthesisRequest(
            request=native_request,
            max_chunk_chars=int(max_chunk_chars),
            min_chunk_chars=int(min_chunk_chars),
            pause_ms=int(pause_ms),
            continuation_pause_ms=int(continuation_pause_ms),
            use_prefix_latents=1 if use_prefix_latents else 0,
            disable_auto_split_overlong=1 if disable_auto_split_overlong else 0,
            preflight_chunks=1 if preflight_chunks else 0,
        )
        callback_errors: list[BaseException] = []

        def _on_event(event_ptr: Any, _user_data: Any) -> int:
            try:
                decision = callback(_convert_streaming_event(event_ptr.contents))
            except BaseException as exc:  # pragma: no cover - exercised through native callback state
                callback_errors.append(exc)
                return 1
            return 0 if decision is None else int(decision)

        c_callback = _ScyllasBandStreamingCallback(_on_event)
        status = self._lib.scyllasband_runtime_synthesize_long_form_stream(
            self._handle,
            ctypes.byref(long_request),
            c_callback,
            None,
        )
        if callback_errors:
            raise callback_errors[0]
        _check_status(self._lib, status)


def _configure_library(lib: ctypes.CDLL) -> None:
    lib.scyllasband_last_error.restype = ctypes.c_char_p
    lib.scyllasband_runtime_create.argtypes = [ctypes.POINTER(_ScyllasBandRuntimeOptions), ctypes.POINTER(ctypes.c_void_p)]
    lib.scyllasband_runtime_create.restype = ctypes.c_int
    lib.scyllasband_runtime_destroy.argtypes = [ctypes.c_void_p]
    lib.scyllasband_runtime_destroy.restype = None
    lib.scyllasband_runtime_set_target_bucket_cache_capacity.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.scyllasband_runtime_set_target_bucket_cache_capacity.restype = ctypes.c_int
    lib.scyllasband_runtime_synthesize.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ScyllasBandSynthesisRequest), ctypes.POINTER(_ScyllasBandSynthesisResult)]
    lib.scyllasband_runtime_synthesize.restype = ctypes.c_int
    lib.scyllasband_runtime_synthesize_long_form.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ScyllasBandLongFormSynthesisRequest), ctypes.POINTER(_ScyllasBandSynthesisResult)]
    lib.scyllasband_runtime_synthesize_long_form.restype = ctypes.c_int
    lib.scyllasband_runtime_plan_long_form.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ScyllasBandLongFormSynthesisRequest), ctypes.POINTER(_ScyllasBandChunkPlanResult)]
    lib.scyllasband_runtime_plan_long_form.restype = ctypes.c_int
    lib.scyllasband_runtime_synthesize_long_form_stream.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_ScyllasBandLongFormSynthesisRequest),
        _ScyllasBandStreamingCallback,
        ctypes.c_void_p,
    ]
    lib.scyllasband_runtime_synthesize_long_form_stream.restype = ctypes.c_int
    lib.scyllasband_runtime_estimate_latent_frames.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ScyllasBandSynthesisRequest), ctypes.POINTER(_ScyllasBandDurationEstimateResult)]
    lib.scyllasband_runtime_estimate_latent_frames.restype = ctypes.c_int
    lib.scyllasband_duration_estimate_result_free.argtypes = [ctypes.POINTER(_ScyllasBandDurationEstimateResult)]
    lib.scyllasband_duration_estimate_result_free.restype = None
    lib.scyllasband_synthesis_result_free.argtypes = [ctypes.POINTER(_ScyllasBandSynthesisResult)]
    lib.scyllasband_synthesis_result_free.restype = None
    lib.scyllasband_plan_long_form_chunks.argtypes = [ctypes.POINTER(_ScyllasBandChunkPlanRequest), ctypes.POINTER(_ScyllasBandChunkPlanResult)]
    lib.scyllasband_plan_long_form_chunks.restype = ctypes.c_int
    lib.scyllasband_chunk_plan_result_free.argtypes = [ctypes.POINTER(_ScyllasBandChunkPlanResult)]
    lib.scyllasband_chunk_plan_result_free.restype = None


def _build_synthesis_request(request: Any) -> tuple[_ScyllasBandSynthesisRequest, Any]:
    explicit_phones = getattr(request, "explicit_phones", None)
    if isinstance(explicit_phones, (tuple, list)):
        explicit_phones = " ".join(str(item) for item in explicit_phones)
    prefix_buffer, prefix_dim, prefix_frames = _prefix_buffer(getattr(request, "prefix_latents", None))
    seed = getattr(request, "seed", None)
    native = _ScyllasBandSynthesisRequest(
        text=_bytes(getattr(request, "text", None)),
        explicit_phones=_bytes(explicit_phones),
        voice_id=_bytes(getattr(request, "voice_id", None)),
        language=_bytes(getattr(request, "language", None)),
        emotion=_bytes(getattr(request, "emotion", None)),
        emotion_guidance=_bytes(getattr(request, "emotion_guidance", None)),
        guidance_null_reference=1 if getattr(request, "guidance_null_reference", True) else 0,
        emotion_embed_scale=float(getattr(request, "emotion_embed_scale", 1.0)),
        prefix_latents=prefix_buffer,
        prefix_latent_dim=prefix_dim,
        prefix_latent_frames=prefix_frames,
        context_before=_bytes(getattr(request, "context_before", None)),
        context_after=_bytes(getattr(request, "context_after", None)),
        chunk_index=_int_or_default(getattr(request, "chunk_index", None), -1),
        chunk_count=_int_or_default(getattr(request, "chunk_count", None), -1),
        boundary_before=_bytes(getattr(request, "boundary_before", None)),
        boundary_after=_bytes(getattr(request, "boundary_after", None)),
        min_sentence_pause_ms=float(getattr(request, "min_sentence_pause_ms", 0.0)),
        min_clause_pause_ms=float(getattr(request, "min_clause_pause_ms", 0.0)),
        steps=int(getattr(request, "steps", 8)),
        sampler=_sampler_id(getattr(request, "sampler", "heun")),
        seed=0 if seed is None else int(seed),
        has_seed=0 if seed is None else 1,
        speed=float(getattr(request, "speed", 1.0)),
        temperature=float(getattr(request, "temperature", 1.0)),
        affect=_bytes(_affect_spec(getattr(request, "affect", None))),
        affect_guidance_scale=float(getattr(request, "affect_guidance_scale", 1.0)),
        has_affect_guidance_scale=1,
    )
    return native, prefix_buffer


def _convert_duration_estimate(result: _ScyllasBandDurationEstimateResult) -> NativeDurationEstimate:
    metadata = _decode_json(result.metadata_json)
    return NativeDurationEstimate(
        predicted_latent_frames=max(0, int(result.predicted_latent_frames)),
        fixed_latent_frames=max(0, int(result.fixed_latent_frames)),
        metadata=metadata,
    )


def _convert_streaming_event(event: _ScyllasBandStreamingEvent) -> NativeStreamingEvent:
    event_type_id = int(event.type)
    sample_count = max(0, int(event.sample_count))
    sample_rate = int(event.sample_rate or 0)
    audio = [float(event.samples[index]) for index in range(sample_count)] if event.samples else []
    latent_dim = max(0, int(event.latent_dim))
    latent_frames = max(0, int(event.latent_frames))
    latent_count = latent_dim * latent_frames
    latents = [float(event.latents[index]) for index in range(latent_count)] if event.latents and latent_count else None
    chunk_id = event.chunk_id.decode("utf-8", errors="replace") if event.chunk_id else None
    return NativeStreamingEvent(
        event_type=_STREAM_EVENT_NAMES.get(event_type_id, f"unknown_{event_type_id}"),
        event_type_id=event_type_id,
        chunk_index=int(event.chunk_index),
        chunk_count=int(event.chunk_count),
        chunk_id=chunk_id,
        metadata=_decode_json(event.metadata_json),
        audio=audio,
        sample_count=sample_count,
        sample_rate=sample_rate,
        latents=latents,
        latent_dim=latent_dim,
        latent_frames=latent_frames,
    )


def _convert_result(result: _ScyllasBandSynthesisResult) -> NativeSynthesisResult:
    sample_count = max(0, int(result.sample_count))
    sample_rate = int(result.sample_rate or 0)
    audio = [float(result.samples[index]) for index in range(sample_count)] if result.samples else []
    latent_dim = max(0, int(result.latent_dim))
    latent_frames = max(0, int(result.latent_frames))
    latent_count = latent_dim * latent_frames
    latents = [float(result.latents[index]) for index in range(latent_count)] if result.latents and latent_count else None
    metadata = _decode_json(result.metadata_json)
    duration = float(sample_count) / float(sample_rate) if sample_rate > 0 else 0.0
    return NativeSynthesisResult(
        audio=audio,
        sample_rate=sample_rate,
        duration_seconds=duration,
        metadata=metadata,
        latents=latents,
        latent_dim=latent_dim,
        latent_frames=latent_frames,
    )


def _prefix_buffer(value: Any) -> tuple[Any, int, int]:
    if value is None:
        return None, 0, 0
    try:
        shape = tuple(int(item) for item in value.shape)
        flat = [float(item) for item in value.reshape(-1)]
    except AttributeError:
        flat = [float(item) for item in value]
        shape = (1, 0, len(flat))
    if len(shape) == 3:
        latent_dim = int(shape[1])
        latent_frames = int(shape[2])
    elif len(shape) == 2:
        latent_dim = int(shape[0])
        latent_frames = int(shape[1])
    else:
        latent_dim = 0
        latent_frames = 0
    if not flat or latent_dim <= 0 or latent_frames <= 0:
        return None, 0, 0
    array_type = ctypes.c_float * len(flat)
    buffer = array_type(*flat)
    return buffer, latent_dim, latent_frames


def _resolve_library_path(
    path: str | Path | None = None,
    *,
    auto_build: bool = True,
    backend: str = "onnx",
) -> Path:
    if path:
        candidate = Path(path).expanduser().resolve()
        if candidate.exists():
            return candidate
        raise ScyllasBandNativeError(f"SCYLLASBAND_NATIVE_LIBRARY does not exist: {candidate}")
    env_path = os.environ.get("SCYLLASBAND_NATIVE_LIBRARY")
    if env_path:
        return _resolve_library_path(env_path, auto_build=False, backend=backend)

    root = Path(__file__).resolve().parents[1]
    names = _library_names()
    search_dirs = _library_search_dirs(root)
    candidate = _find_library_path(search_dirs, names)
    if candidate is not None:
        if (
            auto_build
            and _can_auto_build(path)
            and _managed_native_library_is_stale(root, candidate)
        ):
            try:
                _auto_build_native_library(
                    root,
                    reason=f"native sources are newer than {candidate.name}",
                    backend=backend,
                )
            except ScyllasBandNativeError as exc:
                raise ScyllasBandNativeError(
                    f"Found stale auto-built libscyllasband at {candidate}, but rebuilding it failed: {exc}"
                ) from exc
            candidate = _find_library_path(search_dirs, names)
            if candidate is None:
                raise ScyllasBandNativeError(
                    "Automatic native rebuild completed but libscyllasband was not found"
                )
        return candidate

    build_error: ScyllasBandNativeError | None = None
    if auto_build and _can_auto_build(path):
        try:
            _auto_build_native_library(
                root,
                reason="libscyllasband shared library was not found",
                backend=backend,
            )
        except ScyllasBandNativeError as exc:
            build_error = exc
        else:
            candidate = _find_library_path(search_dirs, names)
            if candidate is not None:
                return candidate

    searched = ", ".join(str(directory / names[0]) for directory in search_dirs)
    message = (
        "Could not find libscyllasband shared library. Build scyllasband/libscyllasband with "
        "-DSCYLLASBAND_ENABLE_ONNX=ON (the default backend), explicitly build with "
        "-DSCYLLASBAND_ENABLE_LITERT=ON for LiteRT, or set SCYLLASBAND_NATIVE_LIBRARY. "
        f"Searched: {searched}"
    )
    if build_error is not None:
        message += f" Automatic native build failed: {build_error}"
    elif not _env_flag(_AUTO_BUILD_ENV, default=True):
        message += f" Automatic native build is disabled by {_AUTO_BUILD_ENV}."
    raise ScyllasBandNativeError(message)


def _library_search_dirs(root: Path) -> list[Path]:
    source_dir = root / "libscyllasband"
    default_build_dir = source_dir / "build"
    search_dirs = [
        default_build_dir,
        default_build_dir / "Debug",
        default_build_dir / "Release",
    ]
    if os.environ.get(_BUILD_DIR_ENV):
        override_build_dir = _native_build_dir(source_dir)
        search_dirs = [
            override_build_dir,
            override_build_dir / "Debug",
            override_build_dir / "Release",
            *search_dirs,
        ]
    search_dirs.append(Path.cwd())
    return search_dirs


def _find_library_path(search_dirs: list[Path], names: list[str]) -> Path | None:
    for directory in search_dirs:
        for name in names:
            candidate = directory / name
            if candidate.exists():
                return candidate.resolve()
    return None


def _managed_native_library_is_stale(root: Path, library_path: Path) -> bool:
    source_dir = root / "libscyllasband"
    managed_dirs = {source_dir / "build", _native_build_dir(source_dir)}
    if not any(_path_is_within(library_path, directory) for directory in managed_dirs):
        return False
    try:
        library_mtime_ns = library_path.stat().st_mtime_ns
    except OSError:
        return True
    for source_path in _native_build_inputs(source_dir):
        try:
            if source_path.stat().st_mtime_ns > library_mtime_ns:
                return True
        except OSError:
            continue
    return False


def _native_build_inputs(source_dir: Path) -> list[Path]:
    inputs: list[Path] = []
    cmakelists = source_dir / "CMakeLists.txt"
    if cmakelists.is_file():
        inputs.append(cmakelists)
    source_suffixes = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".mm"}
    for directory_name in ("include", "src"):
        directory = source_dir / directory_name
        if not directory.is_dir():
            continue
        inputs.extend(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in source_suffixes
        )
    return inputs


def _path_is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except (OSError, ValueError):
        return False
    return True


def _can_auto_build(path: str | Path | None) -> bool:
    return (
        path is None
        and not os.environ.get("SCYLLASBAND_NATIVE_LIBRARY")
        and _env_flag(_AUTO_BUILD_ENV, default=True)
    )


def _auto_build_native_library(root: Path, *, reason: str, backend: str = "onnx") -> None:
    source_dir = root / "libscyllasband"
    cmakelists = source_dir / "CMakeLists.txt"
    if not cmakelists.is_file():
        raise ScyllasBandNativeError(f"Cannot auto-build libscyllasband; missing {cmakelists}")

    backend = str(backend).strip().lower()
    if backend == "auto":
        backend = "onnx"
    if backend not in {"onnx", "litert"}:
        raise ScyllasBandNativeError(
            f"Cannot auto-build the unsupported native backend {backend!r}; "
            "use ONNX or explicitly configure libscyllasband for that backend."
        )

    host_platform = _host_litert_platform()
    if not _env_flag(_QUIET_BUILD_ENV, default=False):
        print(
            f"Scylla's Band native runtime: {reason}; preparing the {backend} libscyllasband "
            f"backend for {host_platform}.",
            file=sys.stderr,
        )
    if backend == "litert":
        _ensure_litert_runtime_staged(source_dir, host_platform)

    cmake = os.environ.get(_CMAKE_ENV) or shutil.which("cmake")
    if not cmake:
        raise ScyllasBandNativeError(
            "Cannot auto-build libscyllasband because CMake was not found. "
            f"Install CMake or set SCYLLASBAND_NATIVE_LIBRARY; set {_AUTO_BUILD_ENV}=0 to skip auto-build."
        )

    build_dir = _native_build_dir(source_dir)
    build_type = os.environ.get(_BUILD_TYPE_ENV, "Release")
    configure_cmd = [
        cmake,
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        f"-DSCYLLASBAND_ENABLE_ONNX={'ON' if backend == 'onnx' else 'OFF'}",
        f"-DSCYLLASBAND_ENABLE_LITERT={'ON' if backend == 'litert' else 'OFF'}",
        "-DSCYLLASBAND_BUILD_TOOLS=ON",
        "-DSCYLLASBAND_BUILD_TESTS=OFF",
        f"-DCMAKE_BUILD_TYPE={build_type}",
    ]
    _run_native_build_command(configure_cmd, cwd=root, label="CMake configure")

    build_cmd = [
        cmake,
        "--build",
        str(build_dir),
        "--target",
        "scyllasband_native",
        "--config",
        build_type,
        "--parallel",
        _native_build_jobs(),
    ]
    _run_native_build_command(build_cmd, cwd=root, label="CMake build")


def _native_build_dir(source_dir: Path) -> Path:
    override = os.environ.get(_BUILD_DIR_ENV)
    if not override:
        return source_dir / "build"
    path = Path(override).expanduser()
    if not path.is_absolute():
        path = source_dir / path
    return path.resolve()


def _native_build_jobs() -> str:
    override = os.environ.get(_BUILD_JOBS_ENV)
    if override:
        return override
    return str(max(1, os.cpu_count() or 1))


def _ensure_litert_runtime_staged(source_dir: Path, host_platform: str) -> None:
    if _staged_litert_runtime(source_dir, host_platform) is not None:
        return
    if not _env_flag(_AUTO_STAGE_LITERT_ENV, default=True):
        raise ScyllasBandNativeError(
            "LiteRT runtime is not staged for the host platform and automatic staging is disabled by "
            f"{_AUTO_STAGE_LITERT_ENV}."
        )

    script = source_dir / "scripts" / "stage_litert_sdk.py"
    if not script.is_file():
        raise ScyllasBandNativeError(f"Cannot stage LiteRT runtime; missing {script}")

    attempts = [[sys.executable, str(script), "--platform", host_platform]]
    if _env_flag(_AUTO_DOWNLOAD_LITERT_ENV, default=True):
        attempts.append([sys.executable, str(script), "--platform", host_platform, "--download-runtime"])

    errors: list[str] = []
    for cmd in attempts:
        try:
            _run_native_build_command(cmd, cwd=source_dir, label="LiteRT staging")
        except ScyllasBandNativeError as exc:
            errors.append(str(exc))
            continue
        if _staged_litert_runtime(source_dir, host_platform) is not None:
            return
        errors.append("LiteRT staging completed but the runtime library was still missing")

    detail = "; ".join(errors[-2:])
    raise ScyllasBandNativeError(
        f"Could not stage LiteRT runtime for {host_platform}. {detail}"
    )


def _staged_litert_runtime(source_dir: Path, host_platform: str) -> Path | None:
    lib_dir = source_dir / "third_party" / "litert" / "lib" / host_platform
    for name in _LITERT_RUNTIME_LIBS.get(host_platform, ()):  # pragma: no branch - platform checked earlier
        candidate = lib_dir / name
        if candidate.is_file():
            return candidate
    return None


def _host_litert_platform() -> str:
    machine = platform.machine().lower()
    if sys.platform == "darwin":
        if machine in {"arm64", "aarch64"}:
            return "macos-arm64"
        raise ScyllasBandNativeError(
            "Automatic Scylla's Band LiteRT native builds currently require Apple Silicon on macOS; "
            f"detected machine={machine!r}. Set SCYLLASBAND_NATIVE_LIBRARY to a manually built library."
        )
    if sys.platform.startswith("linux"):
        if machine in {"x86_64", "amd64"}:
            return "linux-x86_64"
        if machine in {"aarch64", "arm64"}:
            return "linux-arm64"
        raise ScyllasBandNativeError(f"Unsupported Linux machine for automatic Scylla's Band native build: {machine}")
    if sys.platform.startswith("win"):
        if machine in {"amd64", "x86_64"}:
            return "windows-x86_64"
        raise ScyllasBandNativeError(f"Unsupported Windows machine for automatic Scylla's Band native build: {machine}")
    raise ScyllasBandNativeError(f"Unsupported host platform for automatic Scylla's Band native build: {sys.platform}")


def _prepare_dynamic_loader(library_path: Path) -> None:
    if os.name != "nt":
        return
    root = Path(__file__).resolve().parents[1]
    directories = [library_path.parent]
    try:
        host_platform = _host_litert_platform()
    except ScyllasBandNativeError:
        host_platform = None
    if host_platform:
        directories.append(root / "libscyllasband" / "third_party" / "litert" / "lib" / host_platform)
    for directory in directories:
        if directory.is_dir():
            _add_windows_dll_directory(directory)


def _add_windows_dll_directory(directory: Path) -> None:
    path = str(directory.resolve())
    if path in _DLL_DIRECTORY_PATHS:
        return
    _DLL_DIRECTORY_PATHS.add(path)
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is not None:
        _DLL_DIRECTORY_HANDLES.append(add_dll_directory(path))
        return
    current = os.environ.get("PATH", "")
    os.environ["PATH"] = path if not current else path + os.pathsep + current


def _run_native_build_command(cmd: list[str], *, cwd: Path, label: str) -> None:
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise ScyllasBandNativeError(f"{label} could not run {shlex.join(cmd)}: {exc}") from exc
    if completed.returncode != 0:
        output = _command_output_tail(completed.stdout, completed.stderr)
        raise ScyllasBandNativeError(
            f"{label} failed with exit code {completed.returncode}: {shlex.join(cmd)}{output}"
        )


def _command_output_tail(stdout: str, stderr: str, *, max_lines: int = 40) -> str:
    combined = "\n".join(part.strip() for part in (stdout, stderr) if part and part.strip())
    if not combined:
        return ""
    lines = combined.splitlines()
    if len(lines) > max_lines:
        lines = ["...", *lines[-max_lines:]]
    return "\n" + "\n".join(lines)


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _library_names() -> list[str]:
    if sys.platform == "darwin":
        return ["libscyllasband_native.dylib", "scyllasband_native.dylib"]
    if os.name == "nt":
        return ["scyllasband_native.dll", "libscyllasband_native.dll"]
    return ["libscyllasband_native.so", "scyllasband_native.so"]


def _check_status(lib: ctypes.CDLL, status: int) -> None:
    if int(status) == SCYLLASBAND_STATUS_OK:
        return
    raw = lib.scyllasband_last_error()
    message = raw.decode("utf-8", errors="replace") if raw else f"Scylla's Band native status {status}"
    raise ScyllasBandNativeError(message)


def _decode_json(value: bytes | None) -> dict[str, Any]:
    if not value:
        return {}
    return json.loads(value.decode("utf-8"))


def _bytes(value: Any) -> bytes | None:
    if value is None:
        return None
    text = str(value)
    if not text:
        return None
    return text.encode("utf-8")


def _affect_spec(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if not isinstance(value, Mapping):
        raise ValueError("Emotion must be a preset, axis=value string, or axis mapping")
    axes = SUPPORTED_AFFECT_AXES
    normalized = {str(key).strip().lower(): item for key, item in value.items()}
    unknown = sorted(set(normalized) - set(axes))
    if unknown:
        raise ValueError(f"Unknown emotion axis: {', '.join(unknown)}")
    parts: list[str] = []
    for axis in axes:
        if axis in normalized:
            parts.append(f"{axis}={float(normalized[axis]):.9g}")
    return ",".join(parts) or None


def _backend_id(value: str) -> int:
    try:
        return _BACKEND_IDS[str(value or "onnx").lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported Scylla's Band backend: {value!r}") from exc


def _sampler_id(value: str) -> int:
    try:
        return _SAMPLER_IDS[str(value or "heun").lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported Scylla's Band sampler: {value!r}") from exc


def _litert_accelerator_id(value: str) -> int:
    try:
        return _LITERT_ACCELERATOR_IDS[str(value or "cpu").lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported Scylla's Band LiteRT accelerator: {value!r}") from exc


def _int_or_default(value: Any, default: int) -> int:
    return default if value is None else int(value)
