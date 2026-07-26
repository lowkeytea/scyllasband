"""Stable bundle contract for Scylla's Band.

The inference package is the source of truth for this contract. Training code
may emit this manifest format, but public runtime code should never depend on
training modules.
"""

from __future__ import annotations

import ctypes.util
from dataclasses import dataclass
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
SHIPPING_BACKENDS = ("onnx", "litert", "coreml")
DEFAULT_PREFERRED_BACKENDS = ("onnx",)
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
AFFECT_AXIS_ORDERS = {
    1: AFFECT_AXES_V1,
    2: AFFECT_AXES_V2,
}
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
    """Summarize whether a bundle can use the native LiteRT GPU split path."""

    bundle_path = Path(bundle_dir)
    manifest = manifest or load_bundle_manifest(bundle_path)
    metadata, metadata_source = _runtime_acceleration_metadata(bundle_path, manifest)

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
    _validate_affect_contract(manifest)

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
            "Six-axis affect bundles must declare graph_input_contract "
            f"{expected_graph_contract!r}"
        )
    axes = tuple(str(item) for item in affect.get("axes", ()))
    if axes != expected_axes:
        raise BundleValidationError(
            f"Affect axes must be {list(expected_axes)!r}, got {list(axes)!r}"
        )
    if int(affect.get("dimension") or 0) != len(expected_axes):
        raise BundleValidationError("Affect dimension must be 6")
    try:
        model_major = int(manifest.model_version.split(".", 1)[0])
    except ValueError as exc:
        raise BundleValidationError(
            f"model_version must start with an integer: {manifest.model_version!r}"
        ) from exc
    expected_model_axis_version = 2 if model_major >= 4 else 1
    if axis_order_version != expected_model_axis_version:
        raise BundleValidationError(
            f"model_version {manifest.model_version} requires affect axis order "
            f"version {expected_model_axis_version}, got {axis_order_version}"
        )
    if affect.get("range") != [0.0, 1.0]:
        raise BundleValidationError("Affect range must be [0.0, 1.0]")
    if not bool(affect.get("condition_mask_input", False)):
        raise BundleValidationError("Affect contract must enable affect_condition_mask")
    if not bool(affect.get("all_zero_is_explicit", False)):
        raise BundleValidationError("Affect contract must distinguish explicit all-zero from null")
    emotions = controls.get("emotions")
    if isinstance(emotions, dict) and bool(emotions.get("enabled", False)):
        raise BundleValidationError("Categorical emotion inputs cannot be enabled with six-axis affect")

    presets = affect.get("presets")
    if not isinstance(presets, dict) or not presets:
        raise BundleValidationError("Affect bundles must declare at least one preset")
    default_preset = str(affect.get("default_preset") or "")
    if default_preset not in presets:
        raise BundleValidationError("Affect default_preset must name a declared preset")
    for name, vector in presets.items():
        if not isinstance(vector, list) or len(vector) != len(expected_axes):
            raise BundleValidationError(f"Affect preset {name!r} must contain six values")
        try:
            values = [float(item) for item in vector]
        except (TypeError, ValueError) as exc:
            raise BundleValidationError(f"Affect preset {name!r} contains a non-numeric value") from exc
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
            raise BundleValidationError(f"Affect preset {name!r} values must be within [0, 1]")

    duration = manifest.components.get("duration_predictor")
    if duration is None or duration.inputs != _AFFECT_DURATION_INPUTS:
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
        expected = _AFFECT_VECTOR_INPUTS + (
            ("span_context_hidden",) if "span_context_hidden" in component.inputs else ()
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
    controls = manifest.controls if isinstance(manifest.controls, dict) else {}
    manifest_metadata = controls.get("runtime_acceleration")
    if isinstance(manifest_metadata, dict):
        return dict(manifest_metadata), "manifest.controls.runtime_acceleration"

    export_status = _load_optional_json(bundle_path / "export_status.json")
    status_metadata = export_status.get("runtime_acceleration")
    if isinstance(status_metadata, dict):
        return dict(status_metadata), "export_status.runtime_acceleration"

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
