"""Public runtime facade for Scylla's Band duration-flow bundles."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import threading
import time
from typing import Any, Mapping

from .contract import (
    ScyllasBandBundleManifest,
    VoiceSpec,
    coreai_host_supported,
    load_bundle_manifest,
    validate_bundle_layout,
)
from .litert import LiteRTRunner
from .native import NativeScyllasBandRuntime
from .onnx import ONNXRunner
from .text_normalizer import normalize_spoken_text


SUPPORTED_BACKENDS = ("auto", "onnx", "litert", "coreml", "coreai")
IMPLEMENTED_BACKENDS = ("onnx", "litert", "coreai")
SUPPORTED_SAMPLERS = ("euler", "heun")
DEFAULT_MIN_SENTENCE_PUNCTUATION_PAUSE_MS = 107.0
DEFAULT_MIN_CLAUSE_PUNCTUATION_PAUSE_MS = 160.0


@dataclass(frozen=True)
class SynthesisRequest:
    text: str | None
    voice_id: str
    language: str | None = None
    explicit_phones: tuple[str, ...] | str | None = None
    normalize_text: bool = True
    style_id: str = "default"
    emotion: str | None = None
    affect: Mapping[str, float] | str | None = None
    affect_guidance_scale: float = 1.0
    emotion_guidance: str | None = None
    guidance_null_reference: bool = True
    emotion_embed_scale: float = 1.0
    speed: float = 1.0
    pitch: float = 0.0
    energy: float = 1.0
    seed: int | None = None
    steps: int = 8
    sampler: str = "euler"
    temperature: float = 1.0
    context_before: str | None = None
    context_after: str | None = None
    chunk_index: int | None = None
    chunk_count: int | None = None
    boundary_before: str | None = None
    boundary_after: str | None = None
    min_sentence_pause_ms: float = DEFAULT_MIN_SENTENCE_PUNCTUATION_PAUSE_MS
    min_clause_pause_ms: float = DEFAULT_MIN_CLAUSE_PUNCTUATION_PAUSE_MS
    prefix_latents: Any | None = None


@dataclass(frozen=True)
class SynthesisResult:
    audio: Any
    sample_rate: int
    duration_seconds: float
    metadata: dict[str, Any]
    latents: Any | None = None


class ScyllasBandRuntime:
    """Load and run a Scylla's Band bundle.

    This class deliberately exposes only the public inference surface. Training
    artifacts and monitor-only helpers belong in `scyllasband-trainer`.
    """

    def __init__(
        self,
        bundle_dir: str | Path,
        manifest: ScyllasBandBundleManifest,
        backends: list[str] | None = None,
        litert_accelerator: str = "auto",
        onnx_providers: list[str] | None = None,
        onnx_intra_op_num_threads: int = 0,
        onnx_inter_op_num_threads: int = 0,
        onnx_autotune_threads: bool = False,
        onnx_autotune_candidates: list[int] | None = None,
        onnx_thread_cache_path: str | Path | None = None,
    ) -> None:
        self.bundle_dir = Path(bundle_dir)
        self.manifest = manifest
        self.backends = _validate_backends(backends or ["auto"])
        self.litert_accelerator = litert_accelerator
        self.onnx_providers = list(onnx_providers) if onnx_providers else None
        self.onnx_intra_op_num_threads = int(onnx_intra_op_num_threads)
        self.onnx_inter_op_num_threads = int(onnx_inter_op_num_threads)
        self.onnx_autotune_threads = bool(onnx_autotune_threads)
        self.onnx_autotune_candidates = (
            list(onnx_autotune_candidates) if onnx_autotune_candidates is not None else None
        )
        self.onnx_thread_cache_path = (
            Path(onnx_thread_cache_path).expanduser()
            if onnx_thread_cache_path is not None
            else None
        )
        self._sessions: dict[str, Any] = {}
        self._litert_runner: LiteRTRunner | None = None
        self._onnx_runner: ONNXRunner | None = None
        self._native_runner: NativeScyllasBandRuntime | None = None
        self._warmup_lock = threading.RLock()
        self._warmup_metadata: dict[str, Any] | None = None

    @classmethod
    def from_bundle(
        cls,
        bundle_dir: str | Path,
        *,
        backends: list[str] | None = None,
        validate_files: bool = True,
        litert_accelerator: str = "auto",
        onnx_providers: list[str] | None = None,
        onnx_intra_op_num_threads: int = 0,
        onnx_inter_op_num_threads: int = 0,
        onnx_autotune_threads: bool = False,
        onnx_autotune_candidates: list[int] | None = None,
        onnx_thread_cache_path: str | Path | None = None,
    ) -> "ScyllasBandRuntime":
        manifest = (
            validate_bundle_layout(bundle_dir)
            if validate_files
            else load_bundle_manifest(bundle_dir)
        )
        return cls(
            bundle_dir,
            manifest,
            backends=backends,
            litert_accelerator=litert_accelerator,
            onnx_providers=onnx_providers,
            onnx_intra_op_num_threads=onnx_intra_op_num_threads,
            onnx_inter_op_num_threads=onnx_inter_op_num_threads,
            onnx_autotune_threads=onnx_autotune_threads,
            onnx_autotune_candidates=onnx_autotune_candidates,
            onnx_thread_cache_path=onnx_thread_cache_path,
        )

    def available_voices(self) -> list[str]:
        return [voice.id for voice in self.manifest.voices]

    def voice_for_id(self, voice_id: str) -> VoiceSpec:
        for voice in self.manifest.voices:
            if voice.id == voice_id:
                return voice
        raise ValueError(f"Unknown voice_id: {voice_id}")

    def resolve_language_for_voice(self, voice_id: str, language: str | None = None) -> str:
        voice = self.voice_for_id(voice_id)
        requested = _normalize_request_language(language)
        if requested in {None, "en"}:
            resolved = voice.default_language
        else:
            resolved = requested
        if resolved not in self.manifest.languages:
            options = ", ".join(self.manifest.languages)
            raise ValueError(
                f"Unsupported language {resolved!r}; expected one of: {options}"
            )
        if resolved not in voice.languages:
            options = ", ".join(voice.languages)
            raise ValueError(
                f"Unsupported language {resolved!r} for voice {voice_id!r}; expected one of: {options}"
            )
        return resolved

    def normalize_text(
        self,
        text: str,
        *,
        language: str | None = None,
        voice_id: str | None = None,
    ) -> str:
        resolved_language = (
            self.resolve_language_for_voice(voice_id, language)
            if voice_id is not None
            else (_normalize_request_language(language) or self.manifest.default_language)
        )
        return normalize_spoken_text(
            text,
            language=resolved_language,
        )

    def plan_text(
        self,
        text: str,
        *,
        voice_id: str,
        language: str | None = None,
        emotion: str | None = None,
        affect: Mapping[str, float] | str | None = None,
        affect_guidance_scale: float | None = None,
        emotion_guidance: str | None = None,
        options: object | None = None,
        **overrides: Any,
    ) -> Any:
        from .planner import plan_text

        return plan_text(
            self,
            text,
            voice=voice_id,
            language=language,
            emotion=emotion,
            affect=affect,
            affect_guidance_scale=affect_guidance_scale,
            emotion_guidance=emotion_guidance,
            options=options,
            **overrides,
        )

    def plan_records(
        self,
        records: list[dict[str, Any]],
        *,
        options: object | None = None,
        source_kind: str = "records",
        **overrides: Any,
    ) -> Any:
        from .planner import plan_records

        return plan_records(
            self,
            records,
            options=options,
            source_kind=source_kind,
            **overrides,
        )

    def synthesize_stream(
        self,
        records: list[dict[str, Any]] | None = None,
        *,
        text: str | None = None,
        voice_id: str | None = None,
        language: str | None = None,
        emotion: str | None = None,
        affect: Mapping[str, float] | str | None = None,
        affect_guidance_scale: float | None = None,
        emotion_guidance: str | None = None,
        options: object | None = None,
        source_kind: str | None = None,
        **overrides: Any,
    ) -> Any:
        if records is not None:
            from .streaming import synthesize_records_stream

            return synthesize_records_stream(
                self,
                records,
                options=options,
                source_kind=source_kind or "records",
                **overrides,
            )
        if text is None or voice_id is None:
            raise ValueError("synthesize_stream requires records or text with voice_id")
        from .streaming import synthesize_text_stream

        return synthesize_text_stream(
            self,
            text,
            voice=voice_id,
            language=language,
            emotion=emotion,
            affect=affect,
            affect_guidance_scale=affect_guidance_scale,
            emotion_guidance=emotion_guidance,
            options=options,
            **overrides,
        )

    def warmup(
        self,
        *,
        voice_id: str | None = None,
        language: str | None = None,
        emotion: str | None = None,
        text: str = "Warmup.",
        autotune_onnx_threads: bool | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Initialize frontend, vector, and vocoder sessions for a persistent runtime.

        The discarded one-step render intentionally exercises the same smallest-fit
        path used by a real first chunk. Call this during application startup; doing
        it immediately before a one-shot request only moves the same work earlier.
        """
        with self._warmup_lock:
            if self._warmup_metadata is not None and not force:
                return {
                    **self._warmup_metadata,
                    "reused": True,
                    "runtime_status": self.runtime_status(),
                }
            selected_voice = voice_id or self.available_voices()[0]
            selected_language = self.resolve_language_for_voice(selected_voice, language)
            backend = self._select_backend()
            if backend == "onnx":
                runner = self._onnx()
                if autotune_onnx_threads is True:
                    runner.enable_thread_autotune(
                        candidates=self.onnx_autotune_candidates,
                        cache_path=self.onnx_thread_cache_path,
                    )
            started_at = time.perf_counter()
            result = self.synthesize(
                SynthesisRequest(
                    text=str(text or "Warmup."),
                    voice_id=selected_voice,
                    language=selected_language,
                    emotion=emotion,
                    normalize_text=True,
                    seed=0,
                    steps=1,
                    sampler="euler",
                    boundary_before="paragraph_start",
                    boundary_after="paragraph_end",
                )
            )
            metadata = {
                "backend": backend,
                "elapsed_ms": (time.perf_counter() - started_at) * 1000.0,
                "voice_id": selected_voice,
                "language": selected_language,
                "audio_samples_discarded": len(result.audio),
                "fixed_latent_frames": result.metadata.get("fixed_latent_frames"),
                "reused": False,
            }
            self._warmup_metadata = metadata
            return {**metadata, "runtime_status": self.runtime_status()}

    def runtime_status(self) -> dict[str, Any]:
        """Describe initialized process-local runners without loading new sessions."""
        backend = self._select_backend()
        status: dict[str, Any] = {
            "backend": backend,
            "warmup_completed": self._warmup_metadata is not None,
        }
        if backend == "onnx":
            status["runner_initialized"] = self._onnx_runner is not None
            status["session_pool"] = (
                self._onnx_runner.session_status()
                if self._onnx_runner is not None
                else {"backend": "onnx", "session_count": 0, "sessions": []}
            )
        elif backend == "litert" and self._litert_runner is not None:
            status["runner_initialized"] = True
            status["session_pool"] = self._litert_runner.session_status()
        else:
            status["runner_initialized"] = self._native_runner is not None
            status["session_pool"] = {
                "backend": backend,
                "native_runtime_initialized": self._native_runner is not None,
            }
        return status

    def estimate_latent_frames(self, request: SynthesisRequest) -> dict[str, Any]:
        self.voice_for_id(request.voice_id)
        if not request.text and not request.explicit_phones:
            raise ValueError("Either text or explicit_phones is required")
        language = self.resolve_language_for_voice(request.voice_id, request.language)
        text = request.text
        if text is not None and request.normalize_text:
            text = self.normalize_text(text, language=language, voice_id=request.voice_id)
        backend = self._select_backend()
        self._validate_affect_request(request, backend=backend)
        if backend in {"litert", "coreai"}:
            if backend == "litert" and self._litert_runner is not None:
                metadata = self._litert_runner.estimate_latent_frames(request, text=text, language=language)
            else:
                native_request = SynthesisRequest(
                    **{**request.__dict__, "text": text, "language": language}
                )
                estimate = self._native(backend).estimate_latent_frames(native_request)
                metadata = {
                    **estimate.metadata,
                    "predicted_latent_frames": estimate.predicted_latent_frames,
                    "fixed_latent_frames": estimate.fixed_latent_frames,
                    "python_shim": "scyllasband.native",
                }
            return {
                **metadata,
                "normalized_text": text if text is not None else None,
                "text_was_normalized": text != request.text if request.text is not None else False,
                "context_before": request.context_before,
                "context_after": request.context_after,
                "chunk_index": request.chunk_index,
                "chunk_count": request.chunk_count,
                "request_boundary_before": request.boundary_before,
                "request_boundary_after": request.boundary_after,
            }
        if backend == "onnx":
            metadata = self._onnx().estimate_latent_frames(request, text=text, language=language)
            return {
                **metadata,
                "normalized_text": text if text is not None else None,
                "text_was_normalized": text != request.text if request.text is not None else False,
                "context_before": request.context_before,
                "context_after": request.context_after,
                "chunk_index": request.chunk_index,
                "chunk_count": request.chunk_count,
                "request_boundary_before": request.boundary_before,
                "request_boundary_after": request.boundary_after,
                "python_shim": "scyllasband.reference_onnx",
            }
        raise NotImplementedError(
            f"{backend} duration preflight is not implemented yet. Use a LiteRT bundle "
            "or disable chunk preflight."
        )

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        self.voice_for_id(request.voice_id)
        if not request.text and not request.explicit_phones:
            raise ValueError("Either text or explicit_phones is required")
        if request.steps <= 0:
            raise ValueError("steps must be positive")
        if request.sampler not in SUPPORTED_SAMPLERS:
            options = ", ".join(SUPPORTED_SAMPLERS)
            raise ValueError(f"Unsupported sampler {request.sampler!r}; expected one of: {options}")
        language = self.resolve_language_for_voice(request.voice_id, request.language)
        text = request.text
        if text is not None and request.normalize_text:
            text = self.normalize_text(text, language=language, voice_id=request.voice_id)
        backend = self._select_backend()
        self._validate_affect_request(request, backend=backend)
        if backend in {"litert", "coreai"}:
            if backend == "litert" and self._litert_runner is not None:
                result = self._litert_runner.synthesize(request, text=text, language=language)
                return SynthesisResult(
                    audio=result.audio,
                    sample_rate=self.manifest.audio.sample_rate,
                    duration_seconds=float(len(result.audio)) / float(self.manifest.audio.sample_rate),
                    metadata={
                        **result.metadata,
                        "normalized_text": text if text is not None else None,
                        "text_was_normalized": text != request.text if request.text is not None else False,
                        "context_before": request.context_before,
                        "context_after": request.context_after,
                        "chunk_index": request.chunk_index,
                        "chunk_count": request.chunk_count,
                        "request_boundary_before": request.boundary_before,
                        "request_boundary_after": request.boundary_after,
                        "python_shim": "scyllasband.reference_litert",
                    },
                    latents=result.latents,
                )
            native_request = SynthesisRequest(
                **{**request.__dict__, "text": text, "language": language}
            )
            result = self._native(backend).synthesize(native_request)
            return SynthesisResult(
                audio=result.audio,
                sample_rate=result.sample_rate,
                duration_seconds=result.duration_seconds,
                metadata={
                    **result.metadata,
                    "normalized_text": text if text is not None else None,
                    "text_was_normalized": text != request.text if request.text is not None else False,
                    "context_before": request.context_before,
                    "context_after": request.context_after,
                    "chunk_index": request.chunk_index,
                    "chunk_count": request.chunk_count,
                    "request_boundary_before": request.boundary_before,
                    "request_boundary_after": request.boundary_after,
                    "python_shim": "scyllasband.native",
                },
                latents=result.latents,
            )
        if backend == "onnx":
            result = self._onnx().synthesize(request, text=text, language=language)
            return SynthesisResult(
                audio=result.audio,
                sample_rate=self.manifest.audio.sample_rate,
                duration_seconds=float(len(result.audio)) / float(self.manifest.audio.sample_rate),
                metadata={
                    **result.metadata,
                    "normalized_text": text if text is not None else None,
                    "text_was_normalized": text != request.text if request.text is not None else False,
                    "context_before": request.context_before,
                    "context_after": request.context_after,
                    "chunk_index": request.chunk_index,
                    "chunk_count": request.chunk_count,
                    "request_boundary_before": request.boundary_before,
                    "request_boundary_after": request.boundary_after,
                    "runtime_status": self.runtime_status(),
                    "python_shim": "scyllasband.reference_onnx",
                },
                latents=result.latents,
            )
        raise NotImplementedError(
            f"{backend} graph execution is not implemented yet. Use a Core AI, LiteRT, or ONNX bundle "
            "or export a runtime path with an implemented backend first."
        )

    def _select_backend(self) -> str:
        auto_order = ("coreai", "onnx") if coreai_host_supported() else ("onnx",)
        requested: list[str] = []
        for backend in self.backends:
            # `auto` prefers Core AI on macOS 27+ hosts when the bundle ships
            # Core AI artifacts, and resolves to ONNX everywhere else.
            resolved = auto_order if backend == "auto" else (backend,)
            for name in resolved:
                if name not in requested:
                    requested.append(name)
        for backend in requested:
            if backend not in IMPLEMENTED_BACKENDS:
                continue
            if self._has_backend_artifacts(backend):
                return backend
        unimplemented = [backend for backend in requested if backend in SUPPORTED_BACKENDS and backend not in IMPLEMENTED_BACKENDS]
        if unimplemented and not any(backend in IMPLEMENTED_BACKENDS for backend in requested):
            options = ", ".join(unimplemented)
            raise NotImplementedError(
                f"Backend(s) {options} are declared but not implemented yet. "
                "Use ONNX, LiteRT, or Core AI on a supported Apple host."
            )
        raise ValueError(
            f"No usable backend artifacts found for {requested!r}. "
            "ONNX is the default; LiteRT and Core AI require an explicit matching bundle."
        )

    def _has_backend_artifacts(self, backend: str) -> bool:
        for component in self.manifest.components.values():
            if not component.required:
                continue
            artifact = component.artifacts.get(backend)
            if artifact is not None:
                if not (self.bundle_dir / artifact.path).exists():
                    return False
                continue
            if backend == "litert" and not component.artifacts and component.path is not None:
                if not (self.bundle_dir / component.path).exists():
                    return False
                continue
            return False
        return True

    def _native(self, backend: str) -> NativeScyllasBandRuntime:
        if (
            self._native_runner is None
            or self._native_runner.backend != backend
            or self._native_runner.litert_accelerator != self.litert_accelerator
        ):
            self._native_runner = NativeScyllasBandRuntime(
                self.bundle_dir,
                backend=backend,
                validate_bundle=False,
                litert_accelerator=self.litert_accelerator,
            )
        return self._native_runner

    def _litert(self) -> LiteRTRunner:
        if self._litert_runner is None:
            self._litert_runner = LiteRTRunner(self.bundle_dir, self.manifest)
        return self._litert_runner

    def _affect_bundle_enabled(self) -> bool:
        controls = self.manifest.controls if isinstance(self.manifest.controls, dict) else {}
        affect = controls.get("affect")
        return isinstance(affect, dict) and bool(affect.get("enabled", False))

    def _validate_affect_request(self, request: SynthesisRequest, *, backend: str) -> None:
        scale = float(request.affect_guidance_scale)
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError("emotion_scale must be finite and non-negative")
        requested = request.affect is not None or scale != 1.0
        if requested and not self._affect_bundle_enabled():
            raise ValueError("This bundle does not support continuous affect conditioning")

    def _onnx(self) -> ONNXRunner:
        if self._onnx_runner is None:
            self._onnx_runner = ONNXRunner(
                self.bundle_dir,
                self.manifest,
                providers=self.onnx_providers,
                intra_op_num_threads=self.onnx_intra_op_num_threads,
                inter_op_num_threads=self.onnx_inter_op_num_threads,
                autotune_intra_op_threads=self.onnx_autotune_threads,
                autotune_candidates=self.onnx_autotune_candidates,
                autotune_cache_path=self.onnx_thread_cache_path,
            )
        return self._onnx_runner


ScyllasBandEngine = ScyllasBandRuntime
ScyllasBandSynthesisRequest = SynthesisRequest
ScyllasBandSynthesisResult = SynthesisResult

def _normalize_request_language(value: str | None) -> str | None:
    language = str(value or "").strip().lower().replace("-", "_")
    if language in {"", "auto"}:
        return None
    if language in {"en", "eng", "english"}:
        return "en"
    if language in {"en_us", "eng_us", "english_us", "us", "american", "american_english"}:
        return "en_us"
    if language in {
        "en_gb",
        "en_uk",
        "eng_gb",
        "english_gb",
        "english_uk",
        "gb",
        "uk",
        "british",
        "british_english",
    }:
        return "en_gb"
    if language in {"es", "es_mx", "es_es", "spa", "spanish"}:
        return "es"
    if language in {"it", "it_it", "ita", "italian"}:
        return "it"
    if language in {"fr", "fr_fr", "fra", "fre", "french"}:
        return "fr"
    if language in {"de", "de_de", "deu", "ger", "german"}:
        return "de"
    if language in {"vi", "vi_vn", "vi_hn", "vie", "vietnamese"}:
        return "vi"
    return language


def _validate_backends(backends: list[str]) -> list[str]:
    normalized = [str(item).strip().lower() for item in backends if str(item).strip()]
    if not normalized:
        normalized = ["onnx"]
    unsupported = [item for item in normalized if item not in SUPPORTED_BACKENDS]
    if unsupported:
        options = ", ".join(SUPPORTED_BACKENDS)
        raise ValueError(f"Unsupported backend(s) {unsupported!r}; expected one of: {options}")
    return normalized
