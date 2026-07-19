#!/usr/bin/env python3
"""Run one Scylla's Band LiteRT component through the native session ABI.

The probe is intentionally lower level than scyllasband_native_speak: it bypasses
G2P, duration expansion, sampling, and vocoding so accelerator failures can be
isolated to one .tflite component.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np


SCYLLASBAND_TENSOR_FLOAT32 = 1
SCYLLASBAND_TENSOR_INT64 = 4
SCYLLASBAND_TENSOR_INT32 = 5
SCYLLASBAND_TENSOR_BOOL = 9

ACCELERATORS = {
    "auto": 0,
    "cpu": 1,
    "gpu": 2,
    "npu": 3,
}

COMPONENTS = (
    "vector_estimator",
    "vector_estimator_prefix",
    "vector_estimator_tail",
    "vocoder",
)

LITERT_STAGE_PLATFORMS = {
    "android-arm64",
    "android-x86_64",
    "linux-arm64",
    "linux-x86_64",
    "macos-arm64",
    "ios-arm64",
    "ios-sim-arm64",
    "windows-x86_64",
}

LITERT_PLATFORM_ALIASES = {
    "android_aarch64": "android-arm64",
    "android-aarch64": "android-arm64",
    "android_arm64": "android-arm64",
    "android-arm64-v8a": "android-arm64",
    "android_arm64_v8a": "android-arm64",
    "arm64-v8a": "android-arm64",
    "android_x86_64": "android-x86_64",
    "linux_aarch64": "linux-arm64",
    "linux-aarch64": "linux-arm64",
    "linux_arm64": "linux-arm64",
    "linux_x86_64": "linux-x86_64",
    "linux_amd64": "linux-x86_64",
    "linux-amd64": "linux-x86_64",
    "macos_arm64": "macos-arm64",
    "macos-aarch64": "macos-arm64",
    "darwin-arm64": "macos-arm64",
    "darwin-aarch64": "macos-arm64",
    "ios_arm64": "ios-arm64",
    "ios_sim_arm64": "ios-sim-arm64",
    "ios-simulator-arm64": "ios-sim-arm64",
    "iphonesimulator-arm64": "ios-sim-arm64",
    "windows_x86_64": "windows-x86_64",
    "windows-amd64": "windows-x86_64",
    "windows_amd64": "windows-x86_64",
    "win32-x86_64": "windows-x86_64",
    "win32-amd64": "windows-x86_64",
}

DEFAULT_LITERT_STAGE_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "litert"


def _native_library_filename(platform_name: str | None = None, os_name: str | None = None) -> str:
    platform_name = sys.platform if platform_name is None else platform_name
    os_name = os.name if os_name is None else os_name
    if platform_name == "darwin":
        return "libscyllasband_native.dylib"
    if os_name == "nt":
        return "scyllasband_native.dll"
    return "libscyllasband_native.so"


def _default_native_library_path() -> Path:
    return Path(__file__).resolve().parents[1] / "build" / _native_library_filename()


class ScyllasBandTensorView(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("data_type", ctypes.c_int32),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("rank", ctypes.c_int32),
        ("data", ctypes.c_void_p),
        ("byte_length", ctypes.c_uint64),
    ]


class ScyllasBandOwnedTensor(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("data_type", ctypes.c_int32),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("rank", ctypes.c_int32),
        ("data", ctypes.c_void_p),
        ("byte_length", ctypes.c_uint64),
    ]


class TensorInput:
    def __init__(self, name: str, data_type: int, array: np.ndarray) -> None:
        self.name_bytes = name.encode("utf-8")
        self.array = np.ascontiguousarray(array)
        self.shape_values = [int(dim) for dim in self.array.shape]
        self.shape = (ctypes.c_int64 * len(self.shape_values))(*self.shape_values)
        self.view = ScyllasBandTensorView(
            self.name_bytes,
            int(data_type),
            self.shape,
            len(self.shape_values),
            self.array.ctypes.data_as(ctypes.c_void_p),
            int(self.array.nbytes),
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=_default_native_library_path())
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--component", default="vector_estimator", choices=COMPONENTS)
    parser.add_argument("--accelerator", action="append", choices=sorted(ACCELERATORS), default=None)
    stage_group = parser.add_mutually_exclusive_group()
    stage_group.add_argument("--litert-lib-dir", type=Path, help="Prepended to the platform shared-library path before loading libscyllasband")
    stage_group.add_argument("--litert-stage-root", type=Path, help="LiteRT staging root containing lib/<platform>")
    parser.add_argument("--litert-platform", help="LiteRT staged platform name or alias for --litert-stage-root")
    parser.add_argument("--metadata", type=Path, help="Native synthesis metadata JSON used to seed realistic vector inputs")
    parser.add_argument("--voice-pack", type=Path, help="Voice pack .npz for reference_style/reference_prosody")
    parser.add_argument("--language", default="en_us")
    parser.add_argument("--emotion", default="neutral")
    parser.add_argument("--input-profile", choices=["zeros", "random"], default="random")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--time", type=float, default=0.0625)
    parser.add_argument("--active-frames", type=int, default=None)
    parser.add_argument("--input-frames", type=int, default=None, help="Override component time-axis frame count for fixed-shape signature probes")
    parser.add_argument("--signature", default="serving_default", help="LiteRT signature name to run")
    parser.add_argument("--resize-inputs", action="store_true", help="Shrink vector time-axis inputs to active frames and call the non-strict resize ABI")
    parser.add_argument("--id-dtype", choices=["int64", "int32"], default="int64")
    parser.add_argument("--emotion-condition-scale", type=float, default=1.0)
    parser.add_argument("--reference-condition-scale", type=float, default=1.0)
    parser.add_argument("--reference-mask-scale", type=float, default=None)
    parser.add_argument("--raw-reference", action="store_true", help="Use raw .npz reference arrays instead of runtime-normalized features")
    parser.add_argument("--dump-output-dir", type=Path, help="Write raw float outputs as .npy files and compare matching accelerator outputs")
    parser.add_argument("--max-relative-diff-rms", type=float, help="Fail if any compared accelerator output exceeds this relative RMS error")
    parser.add_argument("--min-cosine-similarity", type=float, help="Fail if any compared accelerator output is below this cosine similarity")
    parser.add_argument("--json", type=Path, help="Write probe summary JSON")
    return parser.parse_args()


def _library_path_env_name() -> str:
    if sys.platform == "darwin":
        return "DYLD_LIBRARY_PATH"
    if os.name == "nt":
        return "PATH"
    return "LD_LIBRARY_PATH"


def _maybe_reexec_for_litert_lib_dir(path: Path | None) -> None:
    if path is None:
        return
    lib_dir = str(path.resolve())
    if os.environ.get("SCYLLASBAND_LITERT_PROBE_LIB_DIR") == lib_dir:
        return
    env = os.environ.copy()
    env_name = _library_path_env_name()
    current = env.get(env_name, "")
    env[env_name] = f"{lib_dir}{os.pathsep}{current}" if current else lib_dir
    env["SCYLLASBAND_LITERT_PROBE_LIB_DIR"] = lib_dir
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def _normalize_litert_platform(value: str) -> str:
    normalized = value.strip().lower().replace("/", "-")
    normalized = LITERT_PLATFORM_ALIASES.get(normalized, normalized)
    if normalized not in LITERT_STAGE_PLATFORMS:
        supported = ", ".join(sorted(LITERT_STAGE_PLATFORMS))
        raise SystemExit(f"Unsupported LiteRT platform {value!r}. Supported: {supported}")
    return normalized


def _host_litert_platform_name() -> str:
    machine = platform.machine().lower()
    if sys.platform == "darwin":
        if machine not in {"arm64", "aarch64"}:
            raise SystemExit("LiteRT prebuilts currently publish macos-arm64 for macOS probe hosts.")
        return "macos-arm64"
    if sys.platform.startswith("linux"):
        if machine in {"aarch64", "arm64"}:
            return "linux-arm64"
        return "linux-x86_64"
    if sys.platform.startswith("win"):
        return "windows-x86_64"
    raise SystemExit(f"Unsupported host platform for LiteRT staging: {sys.platform}")


def _resolve_litert_lib_dir(args: argparse.Namespace) -> tuple[Path | None, str | None, Path | None]:
    if args.litert_lib_dir is not None:
        if args.litert_platform is not None:
            raise SystemExit("Use --litert-platform only with --litert-stage-root.")
        return args.litert_lib_dir.expanduser().resolve(), _host_litert_platform_name(), None
    if args.litert_stage_root is None and args.litert_platform is None:
        return None, None, None
    stage_root = (args.litert_stage_root or DEFAULT_LITERT_STAGE_ROOT).expanduser().resolve()
    platform_name = _normalize_litert_platform(args.litert_platform) if args.litert_platform else _host_litert_platform_name()
    lib_dir = stage_root / "lib" / platform_name
    if not lib_dir.is_dir():
        raise SystemExit(f"Resolved LiteRT lib directory does not exist: {lib_dir}")
    return lib_dir, platform_name, stage_root


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_bundle_config(bundle: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _load_json(bundle / "manifest.json")
    status = _load_json(bundle / "export_status.json")
    return manifest, status


def _load_library(path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path.resolve()))
    lib.scyllasband_litert_session_create.argtypes = [ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32]
    lib.scyllasband_litert_session_create.restype = ctypes.c_void_p
    lib.scyllasband_litert_session_destroy.argtypes = [ctypes.c_void_p]
    lib.scyllasband_litert_session_destroy.restype = None
    lib.scyllasband_litert_session_run.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ScyllasBandTensorView),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.POINTER(ScyllasBandOwnedTensor)),
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.scyllasband_litert_session_run.restype = ctypes.c_int
    try:
        lib.scyllasband_litert_session_run_resized.argtypes = lib.scyllasband_litert_session_run.argtypes
        lib.scyllasband_litert_session_run_resized.restype = ctypes.c_int
        lib._scyllasband_has_run_resized = True
    except AttributeError:
        lib._scyllasband_has_run_resized = False
    lib.scyllasband_tensors_destroy.argtypes = [ctypes.POINTER(ScyllasBandOwnedTensor), ctypes.c_int32]
    lib.scyllasband_tensors_destroy.restype = None
    lib.scyllasband_last_error.argtypes = []
    lib.scyllasband_last_error.restype = ctypes.c_char_p
    return lib


def _last_error(lib: ctypes.CDLL) -> str:
    value = lib.scyllasband_last_error()
    return "" if value is None else value.decode("utf-8", errors="replace")


def _component_artifact_path(bundle: Path, manifest: dict[str, Any], component: str) -> Path:
    component_meta = manifest.get("components", {}).get(component, {})
    rel_path = component_meta.get("artifacts", {}).get("litert", {}).get("path")
    if not rel_path:
        raise SystemExit(f"manifest does not declare a LiteRT artifact for {component}")
    return bundle / rel_path


def _id_array(values: list[int], dtype_name: str) -> tuple[int, np.ndarray]:
    if dtype_name == "int32":
        return SCYLLASBAND_TENSOR_INT32, np.asarray(values, dtype=np.int32)
    return SCYLLASBAND_TENSOR_INT64, np.asarray(values, dtype=np.int64)


def _reference_arrays(
    args: argparse.Namespace,
    bundle: Path,
    manifest: dict[str, Any],
    metadata: dict[str, Any],
    style_dim: int,
    prosody_dim: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    voice_pack = args.voice_pack
    prepared = metadata.get("prepared_inputs", {})
    if voice_pack is None and prepared.get("reference_pack_path"):
        voice_pack = Path(prepared["reference_pack_path"])
    if voice_pack is None:
        return (
            np.zeros((1, style_dim), dtype=np.float32),
            np.zeros((1, prosody_dim), dtype=np.float32),
            0.0,
        )
    key = prepared.get("reference_key") or f"{args.language}|{args.emotion}"
    language, emotion = str(key).split("|", 1)
    with np.load(voice_pack) as data:
        style = _array_or_zeros(data.get(f"style_embedding__{language}__{emotion}"), style_dim)
        prosody = _array_or_zeros(data.get(f"prosody_stats__{language}__{emotion}"), prosody_dim)
        mask_key = f"reference_mask__{language}__{emotion}"
        mask = float(np.asarray(data[mask_key]).reshape(-1)[0]) if mask_key in data else 1.0
    if not args.raw_reference:
        style_mean, style_std, prosody_mean, prosody_std = _reference_normalization_stats(
            bundle,
            manifest,
            style_dim,
            prosody_dim,
        )
        style = _normalize_style(style, style_mean, style_std)
        prosody = _normalize_prosody(prosody, prosody_mean, prosody_std)
    return style.reshape(1, style_dim), prosody.reshape(1, prosody_dim), mask


def _array_or_zeros(value: np.ndarray | None, dim: int) -> np.ndarray:
    if value is None:
        return np.zeros((int(dim),), dtype=np.float32)
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    output = np.zeros((int(dim),), dtype=np.float32)
    count = min(int(dim), int(array.size))
    if count > 0:
        output[:count] = array[:count]
    return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _mask_value(value: np.ndarray | None) -> float:
    if value is None:
        return 0.0
    array = np.asarray(value).reshape(-1)
    if array.size == 0:
        return 0.0
    return 1.0 if float(array[0]) > 0.0 else 0.0


def _reference_normalization_stats(
    bundle: Path,
    manifest: dict[str, Any],
    style_dim: int,
    prosody_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pack_rel = manifest.get("assets", {}).get("voice_packs")
    pack_dir = bundle / pack_rel if pack_rel else None
    style_rows: list[np.ndarray] = []
    prosody_rows: list[np.ndarray] = []
    if pack_dir is not None and pack_dir.is_dir():
        for path in sorted(pack_dir.glob("*.npz")):
            with np.load(path) as data:
                for key in data.files:
                    if not key.startswith("reference_mask__") or _mask_value(data.get(key)) <= 0.0:
                        continue
                    suffix = key[len("reference_mask__") :]
                    style_rows.append(_array_or_zeros(data.get(f"style_embedding__{suffix}"), style_dim))
                    prosody_rows.append(
                        _transform_prosody(_array_or_zeros(data.get(f"prosody_stats__{suffix}"), prosody_dim))
                    )
    style_mean, style_std = _standardize_rows(style_rows, style_dim)
    prosody_mean, prosody_std = _standardize_rows(prosody_rows, prosody_dim)
    return style_mean, style_std, prosody_mean, prosody_std


def _standardize_rows(rows: list[np.ndarray], dim: int) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        return np.zeros((int(dim),), dtype=np.float32), np.ones((int(dim),), dtype=np.float32)
    matrix = np.stack([_array_or_zeros(row, dim) for row in rows]).astype(np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    mean = matrix.mean(axis=0).astype(np.float32)
    std = matrix.std(axis=0).astype(np.float32)
    std = np.where(std < 1.0e-4, 1.0, std).astype(np.float32)
    return mean, std


def _normalize_style(style: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    output = (np.asarray(style, dtype=np.float32) - mean.astype(np.float32)) / np.maximum(std.astype(np.float32), 1.0e-4)
    output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    norm = max(float(np.linalg.norm(output)), 1.0e-6)
    return (output / norm).astype(np.float32)


def _normalize_prosody(prosody: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    transformed = _transform_prosody(prosody)
    output = (transformed - mean.astype(np.float32)) / np.maximum(std.astype(np.float32), 1.0e-4)
    output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if output.size > 0:
        output[0] = 0.0
    if output.size > 14:
        output[14] = 0.0
    return output


def _transform_prosody(prosody: np.ndarray) -> np.ndarray:
    output = np.asarray(prosody, dtype=np.float32).reshape(-1).copy()
    for index in (1, 2, 5, 6, 7, 8, 9, 10, 11, 13):
        if index < output.size:
            output[index] = np.log1p(max(float(output[index]), 0.0))
    if output.size > 0:
        output[0] = 0.0
    if output.size > 14:
        output[14] = 0.0
    return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _expanded_phone_ids_from_metadata(metadata: dict[str, Any], latent_frames: int) -> tuple[list[int], int]:
    prepared = metadata.get("prepared_inputs", {})
    expansion = metadata.get("duration_expansion", {})
    phone_ids = [int(value) for value in prepared.get("active_phone_ids", [])]
    durations = [int(value) for value in expansion.get("predicted_durations", [])]
    expanded: list[int] = []
    for phone_id, duration in zip(phone_ids, durations):
        expanded.extend([phone_id] * max(0, duration))
    active = int(expansion.get("predicted_latent_frames") or len(expanded) or min(135, latent_frames))
    if not expanded:
        expanded = [0] * active
    expanded = expanded[:latent_frames]
    expanded.extend([0] * max(0, latent_frames - len(expanded)))
    return expanded, min(active, latent_frames)


def _build_vector_inputs(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    status: dict[str, Any],
    metadata: dict[str, Any],
) -> list[TensorInput]:
    component_name = args.component if args.component.startswith("vector_estimator") else "vector_estimator"
    exported_components = status.get("exported_components", {})
    component = exported_components.get(component_name) or exported_components.get("vector_estimator", {})
    input_meta = component.get("inputs", {})
    latent_dim = int(input_meta.get("latent_dim", 24))
    fixed_latent_frames = int(input_meta.get("latent_frames") or status.get("fixed_shapes", {}).get("latent_frames", 640))
    latent_frames = int(args.input_frames or fixed_latent_frames)
    if latent_frames <= 0:
        raise SystemExit("--input-frames must be positive")
    style_dim = int(input_meta.get("reference_style_dim", 128))
    prosody_dim = int(input_meta.get("reference_prosody_dim", 32))
    prefix_frames = int(input_meta.get("prefix_frames", 0))
    prepared = metadata.get("prepared_inputs", {})
    request = metadata.get("request", {})

    expanded_phone_ids, metadata_active = _expanded_phone_ids_from_metadata(metadata, latent_frames)
    active_frames = int(args.active_frames if args.active_frames is not None else metadata_active)
    active_frames = max(1, min(active_frames, latent_frames))
    run_frames = active_frames if args.resize_inputs and args.input_frames is None else latent_frames
    effective_active_frames = min(active_frames, run_frames)
    expanded_phone_ids = expanded_phone_ids[:run_frames]
    expanded_phone_ids.extend([0] * max(0, run_frames - len(expanded_phone_ids)))
    prefix_enabled = bool(input_meta.get("prefix_conditioning", False))
    hidden_frames = int(input_meta.get("hidden_frames") or (latent_frames + (prefix_frames if prefix_enabled else 0)))
    if args.resize_inputs and args.input_frames is None:
        hidden_frames = run_frames + (prefix_frames if prefix_enabled else 0)
    hidden_size = int(input_meta.get("hidden_size", 512))

    rng = np.random.default_rng(args.seed)
    if args.input_profile == "zeros":
        noise = np.zeros((1, latent_dim, run_frames), dtype=np.float32)
    else:
        noise = rng.standard_normal((1, latent_dim, run_frames)).astype(np.float32)
    if effective_active_frames < run_frames:
        noise[:, :, effective_active_frames:] = 0.0

    latent_mask = np.zeros((1, run_frames), dtype=np.bool_)
    latent_mask[:, :effective_active_frames] = True
    style, prosody, reference_mask = _reference_arrays(args, args.bundle, manifest, metadata, style_dim, prosody_dim)
    if args.reference_mask_scale is not None:
        reference_mask = float(args.reference_mask_scale)

    scalar_ids = {
        "voice_id": int(prepared.get("voice_id", request.get("voice_index", 0))),
        "language_id": int(prepared.get("language_id", request.get("language_index", 0))),
        "emotion_id": int(prepared.get("emotion_id", request.get("emotion_index", 0))),
        "boundary_before_id": int(prepared.get("boundary_before_id", 0)),
        "boundary_after_id": int(prepared.get("boundary_after_id", 0)),
    }
    arrays: dict[str, tuple[int, np.ndarray]] = {
        "hidden": (
            SCYLLASBAND_TENSOR_FLOAT32,
            np.zeros((1, hidden_frames, hidden_size), dtype=np.float32)
            if args.input_profile == "zeros"
            else rng.standard_normal((1, hidden_frames, hidden_size)).astype(np.float32),
        ),
        "noise": (SCYLLASBAND_TENSOR_FLOAT32, noise),
        "time": (SCYLLASBAND_TENSOR_FLOAT32, np.asarray([args.time], dtype=np.float32)),
        "latent_mask": (SCYLLASBAND_TENSOR_BOOL, latent_mask),
        "emotion_condition_mask": (SCYLLASBAND_TENSOR_FLOAT32, np.asarray([args.emotion_condition_scale], dtype=np.float32)),
        "reference_style": (SCYLLASBAND_TENSOR_FLOAT32, style),
        "reference_prosody": (SCYLLASBAND_TENSOR_FLOAT32, prosody),
        "reference_mask": (SCYLLASBAND_TENSOR_FLOAT32, np.asarray([reference_mask], dtype=np.float32)),
        "reference_condition_mask": (SCYLLASBAND_TENSOR_FLOAT32, np.asarray([args.reference_condition_scale], dtype=np.float32)),
        "prefix_latents": (SCYLLASBAND_TENSOR_FLOAT32, np.zeros((1, latent_dim, prefix_frames), dtype=np.float32)),
        "prefix_mask": (SCYLLASBAND_TENSOR_BOOL, np.zeros((1, prefix_frames), dtype=np.bool_)),
    }
    dtype, ids = _id_array(expanded_phone_ids, args.id_dtype)
    arrays["expanded_phone_ids"] = (dtype, ids.reshape(1, run_frames))
    for semantic, value in scalar_ids.items():
        dtype, arr = _id_array([value], args.id_dtype)
        arrays[semantic] = (dtype, arr)

    inputs = manifest.get("components", {}).get(component_name, {}).get("inputs", [])
    if not inputs and component_name != "vector_estimator":
        inputs = manifest.get("components", {}).get("vector_estimator", {}).get("inputs", [])
    if not inputs:
        raise SystemExit(f"manifest does not declare {component_name} inputs")
    result = []
    for index, semantic in enumerate(inputs):
        if semantic not in arrays:
            raise SystemExit(f"unsupported vector_estimator input semantic: {semantic}")
        data_type, array = arrays[semantic]
        result.append(TensorInput(f"args_{index}", data_type, array))
    return result


def _build_vocoder_inputs(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    status: dict[str, Any],
    metadata: dict[str, Any],
) -> list[TensorInput]:
    component = status.get("exported_components", {}).get("vocoder", {})
    input_meta = component.get("inputs", {})
    latent_dim = int(input_meta.get("latent_dim", 24))
    fixed_latent_frames = int(input_meta.get("latent_frames") or status.get("fixed_shapes", {}).get("latent_frames", 640))
    latent_frames = int(args.input_frames or fixed_latent_frames)
    if latent_frames <= 0:
        raise SystemExit("--input-frames must be positive")
    style_dim = int(input_meta.get("reference_style_dim", 128))
    prosody_dim = int(input_meta.get("reference_prosody_dim", 32))
    prepared = metadata.get("prepared_inputs", {})
    request = metadata.get("request", {})

    active_frames = int(args.active_frames if args.active_frames is not None else min(135, latent_frames))
    active_frames = max(1, min(active_frames, latent_frames))
    rng = np.random.default_rng(args.seed)
    if args.input_profile == "zeros":
        latents = np.zeros((1, latent_dim, latent_frames), dtype=np.float32)
    else:
        latents = rng.standard_normal((1, latent_dim, latent_frames)).astype(np.float32)
    if active_frames < latent_frames:
        latents[:, :, active_frames:] = 0.0

    style, prosody, reference_mask = _reference_arrays(args, args.bundle, manifest, metadata, style_dim, prosody_dim)
    if args.reference_mask_scale is not None:
        reference_mask = float(args.reference_mask_scale)

    scalar_ids = {
        "voice_id": int(prepared.get("voice_id", request.get("voice_index", 0))),
        "language_id": int(prepared.get("language_id", request.get("language_index", 0))),
        "emotion_id": int(prepared.get("emotion_id", request.get("emotion_index", 0))),
    }
    arrays: dict[str, tuple[int, np.ndarray]] = {
        "latents": (SCYLLASBAND_TENSOR_FLOAT32, latents),
        "reference_style": (SCYLLASBAND_TENSOR_FLOAT32, style),
        "reference_prosody": (SCYLLASBAND_TENSOR_FLOAT32, prosody),
        "reference_mask": (SCYLLASBAND_TENSOR_FLOAT32, np.asarray([reference_mask], dtype=np.float32)),
        "reference_condition_mask": (SCYLLASBAND_TENSOR_FLOAT32, np.asarray([args.reference_condition_scale], dtype=np.float32)),
    }
    for semantic, value in scalar_ids.items():
        dtype, arr = _id_array([value], args.id_dtype)
        arrays[semantic] = (dtype, arr)

    inputs = manifest.get("components", {}).get("vocoder", {}).get("inputs", [])
    if not inputs:
        raise SystemExit("manifest does not declare vocoder inputs")
    result = []
    for index, semantic in enumerate(inputs):
        if semantic not in arrays:
            raise SystemExit(f"unsupported vocoder input semantic: {semantic}")
        data_type, array = arrays[semantic]
        result.append(TensorInput(f"args_{index}", data_type, array))
    return result


def _build_component_inputs(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    status: dict[str, Any],
    metadata: dict[str, Any],
) -> list[TensorInput]:
    if args.component.startswith("vector_estimator"):
        return _build_vector_inputs(args, manifest, status, metadata)
    if args.component == "vocoder":
        return _build_vocoder_inputs(args, manifest, status, metadata)
    raise SystemExit(f"unsupported component: {args.component}")


def _summarize_array(name: str, data_type: int, shape: list[int], data: np.ndarray) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "name": name,
        "dtype": int(data_type),
        "shape": shape,
        "values": int(data.size),
    }
    if data.dtype.kind == "f":
        finite = np.isfinite(data)
        summary.update(
            {
                "non_finite_count": int(data.size - int(finite.sum())),
                "finite_count": int(finite.sum()),
            }
        )
        if finite.any():
            finite_values = data[finite]
            summary.update(
                {
                    "min": float(finite_values.min()),
                    "max": float(finite_values.max()),
                    "mean": float(finite_values.mean()),
                    "rms": float(math.sqrt(float(np.mean(finite_values * finite_values)))),
                }
            )
    return summary


def _safe_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    return safe or "output"


def _copy_output_tensor(tensor: ScyllasBandOwnedTensor, dump_path: Path | None = None) -> dict[str, Any]:
    name = "" if tensor.name is None else tensor.name.decode("utf-8", errors="replace")
    shape = [int(tensor.shape[index]) for index in range(int(tensor.rank))]
    values = int(tensor.byte_length // ctypes.sizeof(ctypes.c_float))
    if tensor.data_type == SCYLLASBAND_TENSOR_FLOAT32 and tensor.data:
        raw = np.ctypeslib.as_array(
            ctypes.cast(tensor.data, ctypes.POINTER(ctypes.c_float)),
            shape=(values,),
        ).copy().reshape(shape)
        summary = _summarize_array(name, int(tensor.data_type), shape, raw)
        if dump_path is not None:
            dump_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(dump_path, raw)
            summary["dump_path"] = str(dump_path)
        return summary
    return {"name": name, "dtype": int(tensor.data_type), "shape": shape, "bytes": int(tensor.byte_length)}


def _compare_dumped_outputs(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    ok_results = [item for item in results if item.get("status") == "ok"]
    if len(ok_results) < 2:
        return None
    baseline = next((item for item in ok_results if item.get("accelerator") == "cpu"), ok_results[0])
    baseline_outputs = baseline.get("outputs") or []
    if not baseline_outputs:
        return None
    comparisons = []
    for other in ok_results:
        if other is baseline:
            continue
        other_outputs = other.get("outputs") or []
        for index, base_meta in enumerate(baseline_outputs):
            if index >= len(other_outputs):
                continue
            base_path = base_meta.get("dump_path")
            other_path = other_outputs[index].get("dump_path")
            if not base_path or not other_path:
                continue
            left = np.load(base_path).astype(np.float64).reshape(-1)
            right = np.load(other_path).astype(np.float64).reshape(-1)
            count = min(int(left.size), int(right.size))
            if count <= 0:
                continue
            left = left[:count]
            right = right[:count]
            diff = left - right
            left_rms = math.sqrt(float(np.mean(left * left)))
            right_rms = math.sqrt(float(np.mean(right * right)))
            diff_rms = math.sqrt(float(np.mean(diff * diff)))
            denom = max(math.sqrt(float(np.sum(left * left))) * math.sqrt(float(np.sum(right * right))), 1.0e-12)
            comparisons.append(
                {
                    "baseline_accelerator": baseline.get("accelerator"),
                    "accelerator": other.get("accelerator"),
                    "output_index": index,
                    "values": count,
                    "baseline_rms": left_rms,
                    "rms": right_rms,
                    "rms_ratio": right_rms / max(left_rms, 1.0e-12),
                    "diff_rms": diff_rms,
                    "relative_diff_rms": diff_rms / max(left_rms, 1.0e-12),
                    "mean_abs_diff": float(np.mean(np.abs(diff))),
                    "max_abs_diff": float(np.max(np.abs(diff))),
                    "cosine_similarity": float(np.dot(left, right) / denom),
                    "baseline_dump_path": base_path,
                    "dump_path": other_path,
                }
            )
    return {"comparisons": comparisons} if comparisons else None


def _check_output_comparison(
    comparison: dict[str, Any] | None,
    *,
    max_relative_diff_rms: float | None = None,
    min_cosine_similarity: float | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add_check(name: str, passed: bool, **extra: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), **extra})

    if max_relative_diff_rms is None and min_cosine_similarity is None:
        return {"status": "ok", "checks": checks, "failure_count": 0}
    comparisons = comparison.get("comparisons") if isinstance(comparison, dict) else None
    if not comparisons:
        add_check("output_comparison_available", False, reason="thresholds require --dump-output-dir and at least two successful accelerator runs")
        return {"status": "failed", "checks": checks, "failure_count": 1}

    for index, item in enumerate(comparisons):
        label = f"{item.get('baseline_accelerator', 'baseline')}->{item.get('accelerator', 'candidate')}#{item.get('output_index', index)}"
        if max_relative_diff_rms is not None:
            value = item.get("relative_diff_rms")
            add_check(
                "relative_diff_rms",
                value is not None and float(value) <= float(max_relative_diff_rms),
                comparison=label,
                value=value,
                max_value=float(max_relative_diff_rms),
            )
        if min_cosine_similarity is not None:
            value = item.get("cosine_similarity")
            add_check(
                "cosine_similarity",
                value is not None and float(value) >= float(min_cosine_similarity),
                comparison=label,
                value=value,
                min_value=float(min_cosine_similarity),
            )
    failures = [check for check in checks if not check.get("passed")]
    return {"status": "ok" if not failures else "failed", "checks": checks, "failure_count": len(failures)}


def _run_probe(
    lib: ctypes.CDLL,
    model_path: Path,
    inputs: list[TensorInput],
    accelerator: str,
    resize_inputs: bool,
    signature: str,
    dump_output_dir: Path | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    session = lib.scyllasband_litert_session_create(
        str(model_path).encode("utf-8"),
        ACCELERATORS[accelerator],
        0,
    )
    create_seconds = time.perf_counter() - started
    if not session:
        return {
            "accelerator": accelerator,
            "status": "create_failed",
            "create_seconds": create_seconds,
            "error": _last_error(lib),
        }
    output_ptr = ctypes.POINTER(ScyllasBandOwnedTensor)()
    output_count = ctypes.c_int32()
    input_array = (ScyllasBandTensorView * len(inputs))(*(item.view for item in inputs))
    try:
        run_started = time.perf_counter()
        if resize_inputs and not bool(getattr(lib, "_scyllasband_has_run_resized", False)):
            return {
                "accelerator": accelerator,
                "status": "run_failed",
                "create_seconds": create_seconds,
                "run_seconds": 0.0,
                "error": "libscyllasband does not export scyllasband_litert_session_run_resized",
            }
        run_fn = lib.scyllasband_litert_session_run_resized if resize_inputs else lib.scyllasband_litert_session_run
        rc = run_fn(
            session,
            signature.encode("utf-8"),
            input_array,
            len(inputs),
            ctypes.byref(output_ptr),
            ctypes.byref(output_count),
        )
        run_seconds = time.perf_counter() - run_started
        if rc != 0:
            return {
                "accelerator": accelerator,
                "status": "run_failed",
                "create_seconds": create_seconds,
                "run_seconds": run_seconds,
                "error": _last_error(lib),
            }
        outputs = []
        for index in range(output_count.value):
            tensor = output_ptr[index]
            name = "" if tensor.name is None else tensor.name.decode("utf-8", errors="replace")
            dump_path = None
            if dump_output_dir is not None:
                dump_path = dump_output_dir / f"{_safe_filename(accelerator)}_{_safe_filename(signature)}_{index}_{_safe_filename(name)}.npy"
            outputs.append(_copy_output_tensor(tensor, dump_path=dump_path))
        return {
            "accelerator": accelerator,
            "signature": signature,
            "status": "ok",
            "resize_inputs": bool(resize_inputs),
            "create_seconds": create_seconds,
            "run_seconds": run_seconds,
            "outputs": outputs,
        }
    finally:
        if output_ptr:
            lib.scyllasband_tensors_destroy(output_ptr, output_count)
        lib.scyllasband_litert_session_destroy(session)


def main() -> int:
    args = _parse_args()
    resolved_litert_lib_dir, resolved_litert_platform, resolved_litert_stage_root = _resolve_litert_lib_dir(args)
    _maybe_reexec_for_litert_lib_dir(resolved_litert_lib_dir)
    manifest, status = _load_bundle_config(args.bundle)
    metadata = _load_json(args.metadata)
    model_path = _component_artifact_path(args.bundle, manifest, args.component)
    lib = _load_library(args.library)
    inputs = _build_component_inputs(args, manifest, status, metadata)
    results = [
        _run_probe(lib, model_path, inputs, accelerator, args.resize_inputs, args.signature, args.dump_output_dir)
        for accelerator in (args.accelerator or ["cpu", "gpu"])
    ]
    summary = {
        "bundle": str(args.bundle),
        "component": args.component,
        "model": str(model_path),
        "litert_lib_dir": str(resolved_litert_lib_dir) if resolved_litert_lib_dir is not None else None,
        "litert_stage_root": str(resolved_litert_stage_root) if resolved_litert_stage_root is not None else None,
        "litert_platform": resolved_litert_platform,
        "input_profile": args.input_profile,
        "seed": args.seed,
        "time": args.time,
        "id_dtype": args.id_dtype,
        "signature": args.signature,
        "input_frames": args.input_frames,
        "emotion_condition_scale": args.emotion_condition_scale,
        "reference_condition_scale": args.reference_condition_scale,
        "reference_mask_scale": args.reference_mask_scale,
        "raw_reference": bool(args.raw_reference),
        "resize_inputs": bool(args.resize_inputs),
        "dump_output_dir": str(args.dump_output_dir) if args.dump_output_dir is not None else None,
        "input_shapes": {item.name_bytes.decode("utf-8"): item.shape_values for item in inputs},
        "results": results,
    }
    comparison = _compare_dumped_outputs(results) if args.dump_output_dir is not None else None
    if comparison is not None:
        summary["output_comparison"] = comparison
    threshold_check = _check_output_comparison(
        comparison,
        max_relative_diff_rms=args.max_relative_diff_rms,
        min_cosine_similarity=args.min_cosine_similarity,
    )
    if threshold_check["checks"]:
        summary["threshold_check"] = threshold_check
    for result in results:
        print(json.dumps(result, ensure_ascii=False), flush=True)
    if comparison is not None:
        print(json.dumps({"output_comparison": comparison}, ensure_ascii=False), flush=True)
    if threshold_check["checks"]:
        print(json.dumps({"threshold_check": threshold_check}, ensure_ascii=False), flush=True)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if all(item["status"] == "ok" for item in results) and threshold_check["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
