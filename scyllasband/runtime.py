"""Public runtime facade for Scylla's Band bundles."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, TextIO

import numpy as np

from .contract import validate_bundle_layout
from .engine import Engine
from .planner import SynthesisPlan, plan_records, records_from_text
from .streaming import StreamingEvent, StreamOptions, render_plan, synthesize_plan_stream
from .text_normalizer import normalize_spoken_text

SUPPORTED_BACKENDS = ("onnx", "litert", "coreml", "coreai")
SUPPORTED_SAMPLERS = ("heun", "euler")


@dataclass(frozen=True)
class SynthesisRequest:
    text: str
    voice_id: str
    language: str | None = None
    delivery: Mapping[str, Any] | str | None = None
    speed: float = 1.0
    steps: int | None = None          # bundle default (4)
    sampler: str | None = None        # bundle default (euler)
    seed: int | None = None
    temperature: float = 1.0
    normalize_text: bool = True


@dataclass(frozen=True)
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VoiceSpec:
    id: str
    languages: tuple[str, ...]
    default_language: str


class ScyllasBandRuntime:
    def __init__(self, engine: Engine, *, bundle_dir: Path, backend: str) -> None:
        self.engine = engine
        self.bundle_dir = Path(bundle_dir)
        self.backend = backend
        self.manifest = engine.manifest
        self.sample_rate = int(self.manifest["audio"]["sample_rate"])
        self.voices = {v["id"]: VoiceSpec(v["id"], tuple(v["languages"]), v["default_language"]) for v in self.manifest["voices"]}
        controls = self.manifest["controls"]
        self.default_steps = int(controls.get("steps", {}).get("default", 8))
        self.default_sampler = str(controls.get("sampler", {}).get("default", "heun"))

    @classmethod
    def from_bundle(cls, bundle_dir: str | Path, *, backend: str | None = None, backends: Sequence[str] | None = None,
                    threads: int | None = None, compute_units: str | Mapping[str, str] | None = None, validate: bool = True,
                    **_: Any) -> "ScyllasBandRuntime":
        """``compute_units`` (Core ML and Core AI): auto (the bundle's recommendation), cpu, gpu, ane (the graphs the bundle
        lists as accurate on the Neural Engine, the rest on the CPU), or per asset ("gpu,g2p=cpu")."""
        bundle_dir = Path(bundle_dir)
        if validate:
            validate_bundle_layout(bundle_dir)
        chosen = backend or next((b for b in (backends or ()) if b in SUPPORTED_BACKENDS), None) or _bundle_backend(bundle_dir)
        if chosen not in SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported backend {chosen!r}; choose one of {', '.join(SUPPORTED_BACKENDS)}")
        factory = None
        if chosen == "litert":
            from .litert import litert_session_factory
            factory = litert_session_factory(bundle_dir, threads=threads)
        elif chosen in ("coreml", "coreai"):
            units = _compute_units(bundle_dir, chosen, compute_units)
            if chosen == "coreml":
                from .coreml import coreml_session_factory
                factory = coreml_session_factory(bundle_dir, compute_units=units)
            else:
                from .coreai import coreai_session_factory
                factory = coreai_session_factory(bundle_dir, compute_units=units)
        return cls(Engine(bundle_dir, backend=chosen, threads=threads, session_factory=factory), bundle_dir=bundle_dir, backend=chosen)

    # --- voices and text ----------------------------------------------------------------------------------------------
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

    def normalize_text(self, text: str, *, language: str = "en_us") -> str:
        return normalize_spoken_text(text, language=language)

    # --- planning -----------------------------------------------------------------------------------------------------
    def plan_records(self, records: Sequence[Mapping[str, Any]], *, normalize: bool = True, source_kind: str = "text") -> SynthesisPlan:
        return plan_records(records, resolve_language=self.resolve_language_for_voice, normalize=normalize, source_kind=source_kind)

    def plan_text(self, text: str, *, voice_id: str, language: str | None = None, delivery: Any = None,
                  normalize: bool = True) -> SynthesisPlan:
        return self.plan_records(records_from_text(text, voice=voice_id, language=language, delivery=delivery), normalize=normalize)

    # --- synthesis ----------------------------------------------------------------------------------------------------
    def stream_options(self, *, steps: int | None = None, sampler: str | None = None, speed: float = 1.0, seed: int | None = None,
                       temperature: float = 1.0, pause_ms: float = 0.0) -> StreamOptions:
        sampler = (sampler or self.default_sampler).lower()
        if sampler not in SUPPORTED_SAMPLERS:
            raise ValueError(f"Unsupported sampler {sampler!r}; choose one of {', '.join(SUPPORTED_SAMPLERS)}")
        if float(speed) <= 0:
            raise ValueError("speed must be positive")
        return StreamOptions(steps=int(steps or self.default_steps), sampler=sampler, speed=float(speed), seed=seed,
                             temperature=float(temperature), pause_ms=float(pause_ms))

    def synthesize(self, request: SynthesisRequest, *, progress: TextIO | None = None) -> SynthesisResult:
        plan = self.plan_text(request.text, voice_id=request.voice_id, language=request.language, delivery=request.delivery,
                              normalize=request.normalize_text)
        if not plan.chunks:
            raise ValueError("Nothing to speak: the text has no words")
        options = self.stream_options(steps=request.steps, sampler=request.sampler, speed=request.speed, seed=request.seed,
                                      temperature=request.temperature)
        audio, sample_rate, metadata = render_plan(self.engine, plan, options, progress=progress)
        metadata.update(backend=self.backend, bundle_dir=str(self.bundle_dir), voice_id=request.voice_id,
                        language=plan.chunks[0].language, delivery=plan.chunks[0].delivery, steps=options.steps,
                        sampler=options.sampler, speed=options.speed, seed=request.seed)
        return SynthesisResult(audio=audio, sample_rate=sample_rate, metadata=metadata)

    def synthesize_stream(self, records: Sequence[Mapping[str, Any]], *, normalize: bool = True, progress: TextIO | None = None,
                          **options: Any) -> Iterator[StreamingEvent]:
        plan = self.plan_records(records, normalize=normalize)
        return synthesize_plan_stream(self.engine, plan, self.stream_options(**options), progress=progress)

    def render_records(self, records: Sequence[Mapping[str, Any]], *, normalize: bool = True, progress: TextIO | None = None,
                       **options: Any) -> SynthesisResult:
        plan = self.plan_records(records, normalize=normalize)
        audio, sample_rate, metadata = render_plan(self.engine, plan, self.stream_options(**options), progress=progress)
        return SynthesisResult(audio=audio, sample_rate=sample_rate, metadata=dict(metadata, backend=self.backend))

    def warmup(self, *, voice_id: str | None = None) -> dict[str, Any]:
        """Create every session once with a short sentence so the first real request starts fast."""
        voice = voice_id or self.available_voices()[0]
        result = self.synthesize(SynthesisRequest(text="Hello.", voice_id=voice, steps=1, sampler="euler", seed=0))
        return dict(voice=voice, samples=int(result.audio.size))

    def runtime_status(self) -> dict[str, Any]:
        return dict(backend=self.backend, bundle_dir=str(self.bundle_dir), voices=self.available_voices(),
                    buckets=[b[0] for b in self.engine.buckets], sessions=sorted(self.engine._sessions))


def _compute_units(bundle_dir: Path, backend: str, requested: str | Mapping[str, str] | None) -> str | Mapping[str, str]:
    """``auto`` (or None): SCYLLASBAND_COMPUTE_UNITS, else the bundle's recommendation, else the GPU — as the native runtime does."""
    import json
    import os
    if requested == "ane":
        # Only the graphs the bundle lists as accurate on the Neural Engine run there; the rest use the CPU.
        manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
        assets = (manifest.get("controls", {}).get(backend) or {}).get("neural_engine_assets") or []
        return ",".join(["cpu", *(f"{asset}=ane" for asset in assets)])
    if requested not in (None, "", "auto"):
        return requested
    override = os.environ.get("SCYLLASBAND_COMPUTE_UNITS")
    if override:
        return override
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    return (manifest.get("controls", {}).get(backend) or {}).get("compute_units") or "gpu"


def _bundle_backend(bundle_dir: Path) -> str:
    import json
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    preferred = [b for b in manifest.get("preferred_backends", []) if b in SUPPORTED_BACKENDS]
    return preferred[0] if preferred else "litert"
