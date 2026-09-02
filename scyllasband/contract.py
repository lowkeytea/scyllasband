"""Stable bundle contract for Scylla's Band.

The inference package is the source of truth for this contract. Training code
may emit this manifest format, but public runtime code should never depend on
training modules.
"""

from __future__ import annotations

import ctypes.util
from dataclasses import dataclass
import hashlib
import json
import math
import os
import platform
from pathlib import Path
from typing import Any


CONTRACT_VERSION = "1.0.0"
REQUIRED_COMPONENTS = (
    "g2p",
    "duration_predictor",
    "vector_estimator",
    "vocoder",
)
SPLIT_VECTOR_COMPONENTS = ("vector_estimator_prefix", "vector_estimator_tail")
SHIPPING_BACKENDS = ("onnx", "litert", "coreml", "coreai")
DEFAULT_PREFERRED_BACKENDS = ("onnx",)
COREAI_MINIMUM_MACOS_MAJOR = 27
DURATION_PAUSE_PRESENCE_SCHEMA = "scyllasband_duration_pause_presence_v1"
DURATION_HIERARCHY_SAMPLING_SCHEMA = "scyllasband_duration_hierarchy_sampling_v1"
VECTOR_TIMING_CONDITIONING_SCHEMA = "scyllasband_vector_timing_conditioning_v1"
VECTOR_TIMING_INPUTS = (
    "expanded_boundary_event_ids",
    "expanded_modifier_event_ids",
    "expanded_phone_phase",
    "expanded_phone_log_duration",
)
_VECTOR_PUNCTUATION_PHONES = frozenset({
    "<pause_comma>",
    "<pause_semicolon>",
    "<pause_colon>",
    "<pause_dash>",
    "<ellipsis>",
    "<end_stmt>",
    "<end_question>",
    "<end_exclaim>",
})
_VECTOR_MODIFIER_PHONES = frozenset({"ˈ", "ˌ", "ː", "ˑ", "̃", "̩", "̪"})


def coreai_host_supported() -> bool:
    """True when this host can execute Core AI bundles (macOS 27 or later)."""

    if platform.system() != "Darwin":
        return False
    release = platform.mac_ver()[0]
    try:
        return int(release.split(".")[0]) >= COREAI_MINIMUM_MACOS_MAJOR
    except (ValueError, IndexError):
        return False
AFFECT_AXES_V1 = (
    "calm",
    "joy",
    "anger",
    "sadness",
    "sarcasm",
    "questioning",
)
AFFECT_AXES_V2 = (
    "calm",
    "joy",
    "anger",
    "sadness",
    "sarcasm",
    "whisper",
)
AFFECT_AXES_V3 = ("calm", "joy", "anger", "sadness", "whisper")
AFFECT_AXIS_ORDERS = {
    1: AFFECT_AXES_V1,
    2: AFFECT_AXES_V2,
    3: AFFECT_AXES_V3,
}
AFFECT_VALUE_SEMANTICS = "independent_continuous_intensities_no_simplex_v1"
CANONICAL_AFFECT_AXES = AFFECT_AXES_V1
SUPPORTED_AFFECT_AXES = (*AFFECT_AXES_V1, "whisper")
_AFFECT_DURATION_INPUTS = (
    "phone_ids",
    "voice_id",
    "language_id",
    "affect_values",
    "boundary_before_id",
    "boundary_after_id",
    "phone_mask",
    "affect_condition_mask",
    "reference_style",
    "reference_prosody",
    "reference_mask",
    "reference_condition_mask",
)
_AFFECT_VECTOR_INPUTS = (
    "noise",
    "time",
    "expanded_phone_ids",
    "voice_id",
    "language_id",
    "affect_values",
    "boundary_before_id",
    "boundary_after_id",
    "latent_mask",
    "affect_condition_mask",
    "reference_style",
    "reference_prosody",
    "reference_mask",
    "reference_condition_mask",
    "prefix_latents",
    "prefix_mask",
)
_V4_AFFECT_DURATION_INPUTS = (
    "phone_ids",
    "voice_id",
    "language_id",
    "affect_values",
    "boundary_before_id",
    "boundary_after_id",
    "phone_mask",
    "affect_condition_mask",
    "identity_reference",
    "identity_reference_mask",
    "prosody_baseline",
    "prosody_delta",
    "prosody_feature_mask",
    "prosody_confidence",
)
_V4_AFFECT_VECTOR_INPUTS = (
    "noise",
    "time",
    "expanded_phone_ids",
    "voice_id",
    "language_id",
    "affect_values",
    "boundary_before_id",
    "boundary_after_id",
    "latent_mask",
    "affect_condition_mask",
    "identity_reference",
    "identity_reference_mask",
    "prosody_baseline",
    "prosody_delta",
    "prosody_feature_mask",
    "prosody_confidence",
    "prefix_latents",
    "prefix_mask",
)


class BundleValidationError(ValueError):
    """Raised when a model bundle does not match the public contract."""


@dataclass(frozen=True)
class AudioSpec:
    sample_rate: int
    hop_length: int
    n_mels: int
    latent_dim: int
    latent_hop_length: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AudioSpec":
        return cls(
            sample_rate=int(data["sample_rate"]),
            hop_length=int(data["hop_length"]),
            n_mels=int(data["n_mels"]),
            latent_dim=int(data["latent_dim"]),
            latent_hop_length=int(data["latent_hop_length"]),
        )


@dataclass(frozen=True)
class BackendArtifactSpec:
    backend: str
    path: str
    format: str

    @classmethod
    def from_dict(cls, backend: str, data: dict[str, Any]) -> "BackendArtifactSpec":
        return cls(
            backend=backend,
            path=str(data["path"]),
            format=str(data.get("format", backend)),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "format": self.format,
        }


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    path: str | None
    format: str | None
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    artifacts: dict[str, BackendArtifactSpec]
    required: bool = True

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "ComponentSpec":
        artifacts = {
            str(backend): BackendArtifactSpec.from_dict(str(backend), spec)
            for backend, spec in data.get("artifacts", {}).items()
        }
        path = str(data["path"]) if data.get("path") else None
        if path is None and not artifacts:
            raise BundleValidationError(
                f"Component {name!r} must declare either path or backend artifacts"
            )
        return cls(
            name=name,
            path=path,
            format=str(data["format"]) if data.get("format") else None,
            inputs=tuple(str(item) for item in data.get("inputs", ())),
            outputs=tuple(str(item) for item in data.get("outputs", ())),
            artifacts=artifacts,
            required=bool(data.get("required", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "required": self.required,
        }
        if self.path is not None:
            data["path"] = self.path
        if self.format is not None:
            data["format"] = self.format
        if self.artifacts:
            data["artifacts"] = {
                backend: artifact.to_dict()
                for backend, artifact in self.artifacts.items()
            }
        return data


@dataclass(frozen=True)
class VoiceSpec:
    id: str
    name: str
    languages: tuple[str, ...]
    default_language: str
    embedding_path: str | None = None
    styles: tuple[str, ...] = ("default",)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VoiceSpec":
        languages = tuple(str(item) for item in data.get("languages", ()))
        if not languages and data.get("language"):
            languages = (str(data["language"]),)
        if not languages:
            languages = ("en",)
        default_language = str(data.get("default_language", languages[0]))
        if default_language not in languages:
            raise BundleValidationError(
                f"Voice {data['id']!r} default_language is not listed in languages"
            )
        return cls(
            id=str(data["id"]),
            name=str(data.get("name", data["id"])),
            languages=languages,
            default_language=default_language,
            embedding_path=(
                str(data["embedding_path"]) if data.get("embedding_path") else None
            ),
            styles=tuple(str(item) for item in data.get("styles", ("default",))),
        )


@dataclass(frozen=True)
class ScyllasBandBundleManifest:
    contract_version: str
    model_name: str
    architecture: str
    audio: AudioSpec
    languages: tuple[str, ...]
    default_language: str
    voices: tuple[VoiceSpec, ...]
    components: dict[str, ComponentSpec]
    controls: dict[str, Any]
    assets: dict[str, str]
    preferred_backends: tuple[str, ...] = DEFAULT_PREFERRED_BACKENDS
    model_version: str = "3"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScyllasBandBundleManifest":
        version = str(data.get("contract_version", ""))
        if version != CONTRACT_VERSION:
            raise BundleValidationError(
                f"Unsupported contract_version {version!r}; expected {CONTRACT_VERSION!r}"
            )

        components = {
            name: ComponentSpec.from_dict(name, spec)
            for name, spec in data.get("components", {}).items()
        }
        missing = [name for name in REQUIRED_COMPONENTS if name not in components]
        if missing:
            raise BundleValidationError(
                "Bundle manifest is missing components: " + ", ".join(missing)
            )

        languages = tuple(str(item) for item in data.get("languages", ()))
        if not languages:
            raise BundleValidationError("Bundle manifest must declare at least one language")

        default_language = str(data.get("default_language", languages[0]))
        if default_language not in languages:
            raise BundleValidationError(
                f"default_language {default_language!r} is not listed in languages"
            )

        preferred_backends = tuple(
            str(item) for item in data.get("preferred_backends", DEFAULT_PREFERRED_BACKENDS)
        )
        if not preferred_backends:
            raise BundleValidationError("Bundle manifest must declare at least one backend")
        unsupported_backends = [
            backend for backend in preferred_backends if backend not in SHIPPING_BACKENDS
        ]
        if unsupported_backends:
            raise BundleValidationError(
                "Unsupported preferred_backends: " + ", ".join(unsupported_backends)
            )

        return cls(
            contract_version=version,
            model_name=str(data["model_name"]),
            model_version=str(data.get("model_version", "3")),
            architecture=str(data.get("architecture", "scyllasband-duration-flow")),
            audio=AudioSpec.from_dict(data["audio"]),
            languages=languages,
            default_language=default_language,
            voices=tuple(VoiceSpec.from_dict(item) for item in data.get("voices", ())),
            components=components,
            controls=dict(data.get("controls", {})),
            assets={str(key): str(value) for key, value in data.get("assets", {}).items()},
            preferred_backends=preferred_backends,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "architecture": self.architecture,
            "audio": {
                "sample_rate": self.audio.sample_rate,
                "hop_length": self.audio.hop_length,
                "n_mels": self.audio.n_mels,
                "latent_dim": self.audio.latent_dim,
                "latent_hop_length": self.audio.latent_hop_length,
            },
            "languages": list(self.languages),
            "default_language": self.default_language,
            "preferred_backends": list(self.preferred_backends),
            "voices": [
                {
                    "id": voice.id,
                    "name": voice.name,
                    "languages": list(voice.languages),
                    "default_language": voice.default_language,
                    "embedding_path": voice.embedding_path,
                    "styles": list(voice.styles),
                }
                for voice in self.voices
            ],
            "components": {
                name: component.to_dict()
                for name, component in self.components.items()
            },
            "controls": self.controls,
            "assets": self.assets,
        }


def load_bundle_manifest(bundle_dir: str | Path) -> ScyllasBandBundleManifest:
    path = Path(bundle_dir) / "manifest.json"
    if not path.is_file():
        raise BundleValidationError(f"Bundle manifest is missing: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise BundleValidationError(f"Bundle manifest is invalid JSON: {path}: {exc}") from exc
    return ScyllasBandBundleManifest.from_dict(data)


def bundle_runtime_acceleration_report(
    bundle_dir: str | Path,
    manifest: ScyllasBandBundleManifest | None = None,
) -> dict[str, Any]:
    """Summarize whether a bundle can use its declared accelerated path."""

    bundle_path = Path(bundle_dir)
    manifest = manifest or load_bundle_manifest(bundle_path)
    metadata, metadata_source = _runtime_acceleration_metadata(bundle_path, manifest)

    if metadata.get("backend") == "coreai":
        required = [
            component
            for component in manifest.components.values()
            if component.required and "coreai" in component.artifacts
        ]
        model_ready = bool(required) and all(
            (bundle_path / component.artifacts["coreai"].path).exists()
            for component in required
        )
        runtime_ready = coreai_host_supported()
        warnings = []
        if not model_ready:
            warnings.append("one or more required Core AI assets are missing")
        if not runtime_ready:
            warnings.append("Core AI execution requires iOS 27 or macOS 27")
        vector_metadata = _dict_value(metadata, "vector_estimator")
        return {
            "status": "warning" if warnings else "ok",
            "metadata_present": True,
            "metadata_source": metadata_source,
            "backend": "coreai",
            "runtime_version": metadata.get("runtime_version"),
            "gpu_ready": bool(model_ready and runtime_ready),
            "gpu_model_ready": model_ready,
            "native_gpu_acceleration": {
                "cuda_required": False,
                "platform_targets": {"ios": "27.0", "macos": "27.0"},
                "current_platform_target": "macos" if runtime_ready else None,
                "accelerator_plugin_declared": False,
                "accelerator_plugin_library": None,
                "accelerator_plugin_available": None,
                "accelerator_plugin_path": None,
            },
            "vector_estimator": {
                "execution": vector_metadata.get("execution", "fixed_shape_multifunction"),
                "policy": vector_metadata.get("policy"),
                "metadata_split_artifacts_available": False,
                "min_gpu_split_vector_latent_frames": 0,
                "split_artifacts_declared": False,
                "split_artifacts_available": False,
                "split_artifacts": [],
            },
            "warnings": warnings,
        }

    native_metadata = _dict_value(metadata, "native_gpu_acceleration")
    vector_metadata = _dict_value(metadata, "vector_estimator")
    split_details = [
        _split_vector_component_status(bundle_path, manifest, name)
        for name in SPLIT_VECTOR_COMPONENTS
    ]
    split_artifacts_available = all(item["available"] for item in split_details)
    split_artifacts_declared = all(item["declared"] for item in split_details)
    metadata_expects_split = _metadata_expects_split_vector(vector_metadata)
    gpu_plugin = _native_gpu_accelerator_plugin_status(bundle_path, native_metadata)

    warnings: list[str] = []
    if not metadata:
        warnings.append(
            "runtime_acceleration metadata is missing; split-vector readiness "
            "was inferred from bundle components"
        )
    elif metadata_expects_split and not split_artifacts_available:
        warnings.append(
            "runtime_acceleration expects split vector artifacts, but one or "
            "more bundle files are missing"
        )

    cuda_required = native_metadata.get("cuda_required")
    if cuda_required is True:
        warnings.append("native GPU acceleration metadata unexpectedly requires CUDA")
    if gpu_plugin["declared"] and not gpu_plugin["available"]:
        warnings.append(
            f"native GPU accelerator plugin {gpu_plugin['library']!r} is not available "
            f"for {gpu_plugin['platform_target']}; GPU execution will fail or require fallback"
        )

    execution = vector_metadata.get("execution")
    if not execution:
        execution = (
            "split_prefix_tail"
            if split_artifacts_available
            else "single_graph_or_unknown"
        )

    gpu_model_ready = bool(split_artifacts_available and cuda_required is not True)
    gpu_runtime_ready = not gpu_plugin["declared"] or bool(gpu_plugin["available"])
    return {
        "status": "warning" if warnings else "ok",
        "metadata_present": bool(metadata),
        "metadata_source": metadata_source,
        "backend": metadata.get("backend"),
        "runtime_version": native_metadata.get("runtime_version")
        or metadata.get("runtime_version"),
        "gpu_ready": bool(gpu_model_ready and gpu_runtime_ready),
        "gpu_model_ready": gpu_model_ready,
        "native_gpu_acceleration": {
            "cuda_required": cuda_required,
            "platform_targets": native_metadata.get("platform_targets", {}),
            "current_platform_target": gpu_plugin["platform_target"],
            "accelerator_plugin_declared": gpu_plugin["declared"],
            "accelerator_plugin_library": gpu_plugin["library"],
            "accelerator_plugin_available": gpu_plugin["available"],
            "accelerator_plugin_path": gpu_plugin["path"],
        },
        "vector_estimator": {
            "execution": execution,
            "policy": vector_metadata.get("policy"),
            "metadata_split_artifacts_available": vector_metadata.get(
                "split_artifacts_available"
            ),
            "min_gpu_split_vector_latent_frames": vector_metadata.get(
                "min_gpu_split_vector_latent_frames"
            ),
            "split_artifacts_declared": split_artifacts_declared,
            "split_artifacts_available": split_artifacts_available,
            "split_artifacts": split_details,
        },
        "warnings": warnings,
    }


def _native_gpu_accelerator_plugin_status(
    bundle_path: Path,
    native_metadata: dict[str, Any],
) -> dict[str, Any]:
    machine = platform.machine().strip().lower()
    targets = _dict_value(native_metadata, "platform_targets")
    if platform.system() == "Darwin":
        candidates = (
            ["macos-arm64", "macos-x86_64"]
            if machine in {"arm64", "aarch64"}
            else ["macos-x86_64", "macos-arm64"]
        )
    elif platform.system() == "Linux":
        candidates = (
            ["linux-arm64", "linux-x86_64"]
            if machine in {"arm64", "aarch64"}
            else ["linux-x86_64", "linux-arm64"]
        )
    elif platform.system() == "Windows":
        candidates = ["windows-x86_64"]
    else:
        candidates = []

    platform_target = next((item for item in candidates if item in targets), None)
    if platform_target is None and candidates:
        platform_target = candidates[0]
    target = _dict_value(targets, platform_target) if platform_target else {}
    library = str(target.get("gpu_accelerator_library") or "").strip()
    if not library:
        return {
            "platform_target": platform_target,
            "declared": False,
            "library": None,
            "available": None,
            "path": None,
        }

    repo_root = Path(__file__).resolve().parents[1]
    search_dirs = [
        bundle_path,
        bundle_path / "runtime" / str(platform_target),
        bundle_path / "lib" / str(platform_target),
        repo_root / "runtime" / str(platform_target),
        repo_root / "libscyllasband" / "third_party" / "litert" / "lib" / str(platform_target),
    ]
    configured_dir = os.environ.get("SCYLLASBAND_LITERT_LIBRARY_DIR")
    if configured_dir:
        search_dirs.insert(0, Path(configured_dir).expanduser())
    for directory in search_dirs:
        candidate = directory / library
        if candidate.is_file():
            return {
                "platform_target": platform_target,
                "declared": True,
                "library": library,
                "available": True,
                "path": str(candidate),
            }

    lookup_name = library
    if lookup_name.startswith("lib"):
        lookup_name = lookup_name[3:]
    lookup_name = lookup_name.split(".", 1)[0]
    discovered = ctypes.util.find_library(lookup_name)
    return {
        "platform_target": platform_target,
        "declared": True,
        "library": library,
        "available": bool(discovered),
        "path": discovered,
    }


def validate_bundle_layout(bundle_dir: str | Path) -> ScyllasBandBundleManifest:
    bundle_path = Path(bundle_dir)
    manifest = load_bundle_manifest(bundle_path)
    validate_vector_timing_conditioning_contract(manifest, bundle_path)
    _validate_affect_contract(manifest)
    validate_duration_pause_presence_contract(manifest)
    validate_duration_hierarchy_sampling_contract(manifest)

    for component in manifest.components.values():
        if not component.required:
            continue
        if component.artifacts:
            for artifact in component.artifacts.values():
                artifact_path = bundle_path / artifact.path
                if not artifact_path.exists():
                    raise BundleValidationError(
                        f"Required component {component.name!r} artifact "
                        f"{artifact.backend!r} is missing: {artifact_path}"
                    )
        elif component.path is not None:
            component_path = bundle_path / component.path
            if not component_path.is_file():
                raise BundleValidationError(
                    f"Required component {component.name!r} is missing: {component_path}"
                )

    for asset_name, asset_path in manifest.assets.items():
        full_path = bundle_path / asset_path
        if not full_path.exists():
            raise BundleValidationError(
                f"Declared asset {asset_name!r} is missing: {full_path}"
            )

    for voice in manifest.voices:
        if voice.embedding_path and not (bundle_path / voice.embedding_path).exists():
            raise BundleValidationError(
                f"Voice {voice.id!r} embedding is missing: {voice.embedding_path}"
            )

    return manifest


def validate_duration_pause_presence_contract(
    manifest: ScyllasBandBundleManifest,
) -> dict[str, Any]:
    """Validate the optional two-output duration presence contract."""

    controls = manifest.controls if isinstance(manifest.controls, dict) else {}
    raw = controls.get("duration_pause_presence")
    duration = manifest.components.get("duration_predictor")
    if duration is None:
        raise BundleValidationError("Bundle lacks duration_predictor")
    if not isinstance(raw, dict) or not bool(raw.get("enabled", False)):
        if "pause_presence_logits" in duration.outputs:
            raise BundleValidationError(
                "duration_predictor declares pause_presence_logits without "
                "duration_pause_presence controls"
            )
        return {"enabled": False}
    if raw.get("schema") != DURATION_PAUSE_PRESENCE_SCHEMA:
        raise BundleValidationError(
            "Unsupported duration pause-presence schema"
        )
    if raw.get("activation") != "sigmoid":
        raise BundleValidationError(
            "Duration pause-presence activation must be sigmoid"
        )
    try:
        threshold = float(raw.get("threshold_probability"))
    except (TypeError, ValueError) as exc:
        raise BundleValidationError(
            "Duration pause-presence threshold must be numeric"
        ) from exc
    if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise BundleValidationError(
            "Duration pause-presence threshold must be in (0, 1]"
        )
    hierarchy = controls.get("duration_hierarchy_sampling")
    hierarchy_enabled = isinstance(hierarchy, dict) and bool(
        hierarchy.get("enabled", False)
    )
    expected_outputs = (
        "durations",
        "pause_presence_logits",
        *(("duration_quantiles",) if hierarchy_enabled else ()),
    )
    if duration.outputs != expected_outputs:
        raise BundleValidationError(
            "duration_predictor outputs must be durations,pause_presence_logits "
            "when duration_pause_presence is enabled"
        )
    declared_outputs = tuple(str(item) for item in raw.get("outputs", ()))
    if declared_outputs != ("durations", "pause_presence_logits"):
        raise BundleValidationError(
            "duration_pause_presence controls must declare both duration outputs"
        )
    owners = raw.get("eligible_pause_owners")
    if not isinstance(owners, dict) or owners != {
        "punctuation": "following_silence_only",
        "word_boundary": "explicit_candidate_mask_only",
    }:
        raise BundleValidationError(
            "Duration pause-presence ownership contract is invalid"
        )
    return dict(raw)


def validate_duration_hierarchy_sampling_contract(
    manifest: ScyllasBandBundleManifest,
) -> dict[str, Any]:
    """Validate optional portable p10/p50/p90 hierarchy sampling."""

    controls = manifest.controls if isinstance(manifest.controls, dict) else {}
    raw = controls.get("duration_hierarchy_sampling")
    duration = manifest.components.get("duration_predictor")
    if duration is None:
        raise BundleValidationError("Bundle lacks duration_predictor")
    if not isinstance(raw, dict) or not bool(raw.get("enabled", False)):
        if "duration_quantiles" in duration.outputs:
            raise BundleValidationError(
                "duration_predictor declares duration_quantiles without "
                "duration_hierarchy_sampling controls"
            )
        return {"enabled": False}
    if raw.get("schema") != DURATION_HIERARCHY_SAMPLING_SCHEMA:
        raise BundleValidationError("Unsupported duration hierarchy sampling schema")
    if raw.get("policy") != (
        "coherent_utterance_phrase_quantile_with_presence_hurdle_v1"
    ):
        raise BundleValidationError("Unsupported duration hierarchy sampling policy")
    presence = controls.get("duration_pause_presence")
    presence_enabled = isinstance(presence, dict) and bool(
        presence.get("enabled", False)
    )
    expected_outputs = (
        "durations",
        *(("pause_presence_logits",) if presence_enabled else ()),
        "duration_quantiles",
    )
    if duration.outputs != expected_outputs:
        raise BundleValidationError(
            "duration_predictor outputs do not match the duration hierarchy contract"
        )
    if tuple(str(item) for item in raw.get("outputs", ())) != expected_outputs:
        raise BundleValidationError(
            "duration hierarchy controls must declare the exact duration outputs"
        )
    quantiles = raw.get("quantiles")
    if not isinstance(quantiles, dict) or quantiles != {
        "output": "duration_quantiles",
        "levels": [0.10, 0.50, 0.90],
        "domain": "positive_duration_frames",
        "ordering": "monotonic_p10_p50_p90",
    }:
        raise BundleValidationError("Duration quantile output contract is invalid")
    if tuple(str(item) for item in raw.get("modes", ())) != ("sampled", "p50"):
        raise BundleValidationError("Duration hierarchy modes must be sampled,p50")
    if raw.get("default_mode") not in {"sampled", "p50"}:
        raise BundleValidationError("Duration hierarchy default_mode is invalid")
    if raw.get("seed") != {
        "source": "request_seed",
        "missing_request_seed": 0,
        "algorithm": "sha256_box_muller_v1",
    }:
        raise BundleValidationError("Duration hierarchy seed contract is invalid")
    if raw.get("sampling_key") != (
        "utf8_byte_length_prefixed_language_voice_phones_v1"
    ):
        raise BundleValidationError("Duration hierarchy sampling key is invalid")
    defaults = raw.get("defaults")
    if not isinstance(defaults, dict) or set(defaults) != {
        "pause_strength",
        "speech_strength",
        "sample_presence",
        "max_abs_z",
    }:
        raise BundleValidationError("Duration hierarchy defaults are incomplete")
    try:
        pause_strength = float(defaults["pause_strength"])
        speech_strength = float(defaults["speech_strength"])
        max_abs_z = float(defaults["max_abs_z"])
    except (TypeError, ValueError) as exc:
        raise BundleValidationError("Duration hierarchy defaults must be numeric") from exc
    if (
        pause_strength != 1.0
        or speech_strength != 0.0
        or max_abs_z != 2.0
        or type(defaults["sample_presence"]) is not bool
        or bool(defaults["sample_presence"]) != presence_enabled
    ):
        raise BundleValidationError(
            "Duration hierarchy defaults do not match the supported portable policy"
        )
    if raw.get("phrase_boundary") != (
        "punctuation_owned_silence_after_boundary_v1"
    ):
        raise BundleValidationError("Duration hierarchy phrase boundary is invalid")
    if raw.get("pause_owners") != {
        "punctuation": "following_or_remapped_silence_only",
        "word_boundary": "explicit_candidate_mask_only",
    }:
        raise BundleValidationError("Duration hierarchy pause ownership is invalid")
    return dict(raw)


def validate_vector_timing_conditioning_contract(
    manifest: ScyllasBandBundleManifest,
    bundle_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate optional event/local timing inputs and their vocabulary binding."""

    controls = manifest.controls if isinstance(manifest.controls, dict) else {}
    raw = controls.get("vector_timing_conditioning")
    vector_components = _vector_timing_components(manifest)
    enabled = isinstance(raw, dict) and bool(raw.get("enabled", False))
    if not enabled:
        for component in vector_components:
            leaked = set(VECTOR_TIMING_INPUTS).intersection(component.inputs)
            if leaked:
                raise BundleValidationError(
                    f"Component {component.name!r} exposes vector timing inputs "
                    "without vector_timing_conditioning controls"
                )
        return {"enabled": False}
    assert isinstance(raw, dict)
    if raw.get("schema") != VECTOR_TIMING_CONDITIONING_SCHEMA:
        raise BundleValidationError("Unsupported vector timing conditioning schema")
    if tuple(str(item) for item in raw.get("inputs", ())) != VECTOR_TIMING_INPUTS:
        raise BundleValidationError("Vector timing controls must declare the exact input quartet")
    features = raw.get("features")
    if not isinstance(features, dict):
        raise BundleValidationError("Vector timing controls lack feature flags")
    expected_feature_keys = {"boundary_events", "modifier_events", "local_timing"}
    if set(features) != expected_feature_keys or any(
        not isinstance(features[key], bool) for key in expected_feature_keys
    ):
        raise BundleValidationError("Vector timing feature flags are invalid")
    if not any(bool(features[key]) for key in expected_feature_keys):
        raise BundleValidationError("Enabled vector timing controls must enable a feature")
    for component in vector_components:
        if component.inputs[-len(VECTOR_TIMING_INPUTS) :] != VECTOR_TIMING_INPUTS:
            raise BundleValidationError(
                f"Component {component.name!r} must end with the vector timing input quartet"
            )
        if any(component.inputs.count(name) != 1 for name in VECTOR_TIMING_INPUTS):
            raise BundleValidationError(
                f"Component {component.name!r} has duplicate vector timing inputs"
            )

    phone = raw.get("phone_vocab")
    if not isinstance(phone, dict):
        raise BundleValidationError("Vector timing controls lack phone vocabulary binding")
    asset = str(phone.get("asset") or "")
    if asset != str(manifest.assets.get("phone_vocab") or ""):
        raise BundleValidationError("Vector timing phone vocabulary asset does not match bundle")
    digest = str(phone.get("sha256") or "")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise BundleValidationError("Vector timing phone vocabulary SHA-256 is invalid")
    try:
        vocab_size = int(phone.get("size"))
    except (TypeError, ValueError) as exc:
        raise BundleValidationError("Vector timing phone vocabulary size is invalid") from exc
    if vocab_size <= 0:
        raise BundleValidationError("Vector timing phone vocabulary must be non-empty")

    token_to_id: dict[str, int] | None = None
    if bundle_dir is not None:
        vocab_path = Path(bundle_dir) / asset
        if not vocab_path.is_file():
            raise BundleValidationError(
                f"Vector timing phone vocabulary is missing: {vocab_path}"
            )
        actual = hashlib.sha256(vocab_path.read_bytes()).hexdigest()
        if actual != digest:
            raise BundleValidationError("Vector timing phone vocabulary SHA-256 mismatch")
        try:
            payload = json.loads(vocab_path.read_text(encoding="utf-8"))
            token_to_id = {
                str(token): int(index)
                for token, index in dict(payload["token_to_id"]).items()
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise BundleValidationError("Vector timing phone vocabulary is invalid") from exc
        if len(token_to_id) != vocab_size or sorted(token_to_id.values()) != list(
            range(vocab_size)
        ):
            raise BundleValidationError(
                "Vector timing phone vocabulary IDs must be contiguous and match size"
            )

    boundary = raw.get("boundary_events")
    if not isinstance(boundary, dict):
        raise BundleValidationError("Vector timing boundary-event controls are missing")
    if bool(boundary.get("enabled")) != bool(features["boundary_events"]):
        raise BundleValidationError("Vector timing boundary-event flags disagree")
    if boundary.get("encoding") != "phone_vocab_id_plus_synthetic_word_boundary_v1":
        raise BundleValidationError("Vector timing boundary-event encoding is invalid")
    if int(boundary.get("synthetic_word_boundary_id", -1)) != vocab_size:
        raise BundleValidationError("Vector timing synthetic word-boundary ID is invalid")
    owners = boundary.get("owners")
    if owners != {
        "punctuation": "following_silence_then_first_surviving_frame",
        "word_boundary": "candidate_silence_then_first_surviving_frame",
    }:
        raise BundleValidationError("Vector timing boundary ownership is invalid")
    punctuation_symbols = [str(item) for item in boundary.get("punctuation_symbols", ())]
    punctuation_ids = [int(item) for item in boundary.get("punctuation_phone_ids", ())]
    if len(punctuation_symbols) != len(punctuation_ids) or len(set(punctuation_symbols)) != len(
        punctuation_symbols
    ):
        raise BundleValidationError("Vector timing punctuation mapping is invalid")
    if token_to_id is not None:
        expected = [
            token
            for token, _index in sorted(token_to_id.items(), key=lambda item: item[1])
            if token in _VECTOR_PUNCTUATION_PHONES
        ]
        if expected != punctuation_symbols or any(
            token_to_id.get(symbol) != phone_id
            for symbol, phone_id in zip(punctuation_symbols, punctuation_ids, strict=True)
        ):
            raise BundleValidationError("Vector timing punctuation mapping mismatches phone vocabulary")

    modifiers = raw.get("modifier_events")
    if not isinstance(modifiers, dict):
        raise BundleValidationError("Vector timing modifier-event controls are missing")
    if bool(modifiers.get("enabled")) != bool(features["modifier_events"]):
        raise BundleValidationError("Vector timing modifier-event flags disagree")
    if modifiers.get("encoding") != "bitmask_v1":
        raise BundleValidationError("Vector timing modifier-event encoding is invalid")
    bits = int(modifiers.get("bits") or 0)
    symbols = [str(item) for item in modifiers.get("symbols", ())]
    phone_ids = [int(item) for item in modifiers.get("phone_ids", ())]
    bit_masks = [int(item) for item in modifiers.get("bit_masks", ())]
    expected_count = bits if bool(features["modifier_events"]) else 0
    if bits < 0 or bits > 30 or not (len(symbols) == len(phone_ids) == len(bit_masks) == expected_count):
        raise BundleValidationError("Vector timing modifier vocabulary is invalid")
    if bit_masks != [1 << bit for bit in range(bits)]:
        raise BundleValidationError("Vector timing modifier bit masks are invalid")
    if modifiers.get("ownership") != (
        "stress_bidirectional_postfix_exact_base_segment_bounded_v2"
    ):
        raise BundleValidationError("Vector timing modifier ownership is invalid")
    if modifiers.get("zero_quantized_fallback") != (
        "audit_unrepresented_zero_frame_modifier_v1"
    ):
        raise BundleValidationError("Vector timing modifier fallback is invalid")
    if token_to_id is not None and bool(features["modifier_events"]):
        expected = [
            token
            for token, _index in sorted(token_to_id.items(), key=lambda item: item[1])
            if token in _VECTOR_MODIFIER_PHONES
        ]
        if expected != symbols or any(
            token_to_id.get(symbol) != phone_id
            for symbol, phone_id in zip(symbols, phone_ids, strict=True)
        ):
            raise BundleValidationError("Vector timing modifier mapping mismatches phone vocabulary")

    local = raw.get("local_timing")
    if not isinstance(local, dict) or bool(local.get("enabled")) != bool(features["local_timing"]):
        raise BundleValidationError("Vector local-timing flags disagree")
    if local.get("phase") != "(frame_index_plus_0_5)/phone_duration_frames":
        raise BundleValidationError("Vector phone-phase formula is invalid")
    if local.get("log_duration") != "log1p(phone_duration_frames)":
        raise BundleValidationError("Vector phone-duration formula is invalid")
    if local.get("duration_source") != "final_post_pause_presence_and_floor_frames":
        raise BundleValidationError("Vector local-timing duration source is invalid")
    return dict(raw)


def _vector_timing_components(
    manifest: ScyllasBandBundleManifest,
) -> list[ComponentSpec]:
    return [
        component
        for name, component in manifest.components.items()
        if name == "vector_estimator"
        or name in SPLIT_VECTOR_COMPONENTS
        or name.removeprefix("vector_estimator_").isdigit()
        or name.removeprefix("vector_estimator_prefix_").isdigit()
        or name.removeprefix("vector_estimator_tail_").isdigit()
    ]


def _validate_affect_contract(manifest: ScyllasBandBundleManifest) -> None:
    controls = manifest.controls if isinstance(manifest.controls, dict) else {}
    affect = controls.get("affect")
    if not isinstance(affect, dict) or not bool(affect.get("enabled", False)):
        return
    graph_input_contract = str(controls.get("graph_input_contract") or "")
    axis_order_version = int(affect.get("axis_order_version") or 0)
    expected_axes = AFFECT_AXIS_ORDERS.get(axis_order_version)
    if expected_axes is None:
        raise BundleValidationError(
            f"Unsupported affect axis_order_version {axis_order_version}"
        )
    expected_graph_contract = f"scyllasband_affect_v{axis_order_version}"
    if graph_input_contract != expected_graph_contract:
        raise BundleValidationError(
            "Affect bundles must declare graph_input_contract "
            f"{expected_graph_contract!r}"
        )
    axes = tuple(str(item) for item in affect.get("axes", ()))
    if axes != expected_axes:
        raise BundleValidationError(
            f"Affect axes must be {list(expected_axes)!r}, got {list(axes)!r}"
        )
    if int(affect.get("dimension") or 0) != len(expected_axes):
        raise BundleValidationError(
            f"Affect dimension must be {len(expected_axes)}"
        )
    # The affect contract is self-describing: axis_order_version alone fixes both
    # the axis names and the graph input contract, and both are checked above, so
    # v1 `questioning` can never be read as v2 `whisper`. model_version is release
    # metadata and is deliberately NOT cross-checked against the axis order --
    # doing so hardcodes a release-numbering scheme into the runtime.
    if affect.get("range") != [0.0, 1.0]:
        raise BundleValidationError("Affect range must be [0.0, 1.0]")
    if not bool(affect.get("condition_mask_input", False)):
        raise BundleValidationError("Affect contract must enable affect_condition_mask")
    if not bool(affect.get("all_zero_is_explicit", False)):
        raise BundleValidationError("Affect contract must distinguish explicit all-zero from null")
    value_semantics = affect.get("value_semantics")
    if value_semantics is not None and value_semantics != AFFECT_VALUE_SEMANTICS:
        raise BundleValidationError(
            "Affect values must be independent continuous intensities, not a simplex"
        )
    emotions = controls.get("emotions")
    if isinstance(emotions, dict) and bool(emotions.get("enabled", False)):
        raise BundleValidationError("Categorical emotion inputs cannot be enabled with affect conditioning")

    presets = affect.get("presets")
    if not isinstance(presets, dict) or not presets:
        raise BundleValidationError("Affect bundles must declare at least one preset")
    default_preset = str(affect.get("default_preset") or "")
    if default_preset not in presets:
        raise BundleValidationError("Affect default_preset must name a declared preset")
    for name, vector in presets.items():
        if not isinstance(vector, list) or len(vector) != len(expected_axes):
            raise BundleValidationError(
                f"Affect preset {name!r} must contain {len(expected_axes)} values"
            )
        try:
            values = [float(item) for item in vector]
        except (TypeError, ValueError) as exc:
            raise BundleValidationError(f"Affect preset {name!r} contains a non-numeric value") from exc
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
            raise BundleValidationError(f"Affect preset {name!r} values must be within [0, 1]")
    partial_defaults = affect.get("partial_defaults")
    axis_minimums = affect.get("axis_minimums")
    axis_maximums = affect.get("axis_maximums")
    for field, raw in (
        ("partial_defaults", partial_defaults),
        ("axis_minimums", axis_minimums),
        ("axis_maximums", axis_maximums),
    ):
        if raw is None:
            continue
        if not isinstance(raw, dict) or not set(raw).issubset(expected_axes):
            raise BundleValidationError(
                f"Affect {field} must map declared axes to numeric values"
            )
        try:
            values = [float(value) for value in raw.values()]
        except (TypeError, ValueError) as exc:
            raise BundleValidationError(f"Affect {field} contains a non-numeric value") from exc
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
            raise BundleValidationError(f"Affect {field} values must be within [0, 1]")
    minimums = {str(key): float(value) for key, value in (axis_minimums or {}).items()}
    maximums = {str(key): float(value) for key, value in (axis_maximums or {}).items()}
    for axis in set(minimums).intersection(maximums):
        if minimums[axis] > maximums[axis]:
            raise BundleValidationError(
                f"Affect minimum exceeds maximum for axis {axis!r}"
            )
    for axis, value in (partial_defaults or {}).items():
        numeric = float(value)
        if numeric < minimums.get(axis, 0.0) or numeric > maximums.get(axis, 1.0):
            raise BundleValidationError(
                f"Affect partial default violates bounds for axis {axis!r}"
            )
    for name, vector in presets.items():
        for axis, value in zip(expected_axes, vector, strict=True):
            numeric = float(value)
            if numeric < minimums.get(axis, 0.0) or numeric > maximums.get(axis, 1.0):
                raise BundleValidationError(
                    f"Affect preset {name!r} violates bounds for axis {axis!r}"
                )

    reference_packs = controls.get("reference_packs")
    reference_schema = (
        int(reference_packs.get("schema_version") or 0)
        if isinstance(reference_packs, dict)
        else 0
    )
    duration_inputs = (
        _V4_AFFECT_DURATION_INPUTS
        if reference_schema == 4
        else _AFFECT_DURATION_INPUTS
    )
    vector_inputs = (
        _V4_AFFECT_VECTOR_INPUTS
        if reference_schema == 4
        else _AFFECT_VECTOR_INPUTS
    )
    timing = controls.get("vector_timing_conditioning")
    timing_enabled = isinstance(timing, dict) and bool(timing.get("enabled", False))
    duration = manifest.components.get("duration_predictor")
    if duration is None or duration.inputs != duration_inputs:
        raise BundleValidationError(
            f"Affect duration_predictor inputs do not match {expected_graph_contract}"
        )
    vector_components = [
        component
        for name, component in manifest.components.items()
        if name == "vector_estimator" or (
            name.startswith("vector_estimator_")
            and name.removeprefix("vector_estimator_").isdigit()
        )
    ]
    for component in vector_components:
        expected = (
            vector_inputs
            + (("span_context_hidden",) if "span_context_hidden" in component.inputs else ())
            + (VECTOR_TIMING_INPUTS if timing_enabled else ())
        )
        if component.inputs != expected:
            raise BundleValidationError(
                f"Affect component {component.name!r} inputs do not match "
                f"{expected_graph_contract}"
            )
    forbidden = {"emotion_id", "emotion_condition_mask"}
    for component in (duration, *vector_components):
        present = forbidden.intersection(component.inputs)
        if present:
            raise BundleValidationError(
                f"Affect component {component.name!r} exposes legacy inputs: {sorted(present)}"
            )


def _runtime_acceleration_metadata(
    bundle_path: Path,
    manifest: ScyllasBandBundleManifest,
) -> tuple[dict[str, Any], str | None]:
    # The manifest is the only public source of runtime-acceleration metadata.
    # export_status.json is build provenance -- it records local checkpoint paths
    # and is not part of the published bundle contract, so it is not consulted.
    del bundle_path
    controls = manifest.controls if isinstance(manifest.controls, dict) else {}
    manifest_metadata = controls.get("runtime_acceleration")
    if isinstance(manifest_metadata, dict):
        return dict(manifest_metadata), "manifest.controls.runtime_acceleration"

    return {}, None


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _dict_value(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    return dict(value) if isinstance(value, dict) else {}


def _metadata_expects_split_vector(vector_metadata: dict[str, Any]) -> bool:
    if vector_metadata.get("execution") == "split_prefix_tail":
        return True
    if vector_metadata.get("split_artifacts_available") is True:
        return True
    split_artifacts = vector_metadata.get("split_artifacts")
    return bool(split_artifacts)


def _split_vector_component_status(
    bundle_path: Path,
    manifest: ScyllasBandBundleManifest,
    name: str,
) -> dict[str, Any]:
    component = manifest.components.get(name)
    artifact_path = _component_litert_path(component) if component is not None else None
    exists = bool(artifact_path and (bundle_path / artifact_path).is_file())
    return {
        "component": name,
        "declared": component is not None,
        "path": artifact_path,
        "available": exists,
    }


def _component_litert_path(component: ComponentSpec | None) -> str | None:
    if component is None:
        return None
    artifact = component.artifacts.get("litert")
    if artifact is not None:
        return artifact.path
    return component.path
