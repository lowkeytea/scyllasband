from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
from typing import Iterable

from .contract import coreai_host_supported, validate_bundle_layout


V1_INFERENCE_REPO_ID = "spybyscript/scyllasband"
V2_INFERENCE_REPO_ID = "spybyscript/scyllasbandv2"
DEFAULT_MODEL_VERSION = "v2"
SUPPORTED_MODEL_VERSIONS = ("v1", "v2")
MODEL_VERSION_REPO_IDS = {
    "v1": V1_INFERENCE_REPO_ID,
    "v2": V2_INFERENCE_REPO_ID,
}
DEFAULT_INFERENCE_REPO_ID = V2_INFERENCE_REPO_ID
DEFAULT_MODELS_DIR = Path("scyllasband/models")
DEFAULT_BUNDLE_SUBDIR = "onnx"
DEFAULT_ONNX_INT8_BUNDLE_SUBDIR = "onnx-int8"
DEFAULT_VOICES_SUBDIR = "voices"
SUPPORTED_BUNDLE_SUBDIRS = (
    DEFAULT_BUNDLE_SUBDIR,
    DEFAULT_ONNX_INT8_BUNDLE_SUBDIR,
    "litert",
    "coreai",
    "coreai-fp32",
)

# Shown by the interactive download prompt. Sizes are approximate.
BUNDLE_SUBDIR_DESCRIPTIONS = {
    "coreai": ("Core AI int8 bundle for iOS 27 / macOS 27", "~220 MB"),
    "coreai-fp32": ("Core AI fp32 variant of the default", "~420 MB"),
    DEFAULT_ONNX_INT8_BUNDLE_SUBDIR: ("int8 ONNX bundle, CPU-friendly, all platforms", "~280 MB"),
    DEFAULT_BUNDLE_SUBDIR: ("fp32 ONNX variant of the default", "~410 MB"),
    "litert": ("LiteRT bundle for Android and embedded runtimes", ""),
}
BUNDLE_SUBDIR_GROUPS = {
    "both": (DEFAULT_BUNDLE_SUBDIR, "litert"),
    "all": SUPPORTED_BUNDLE_SUBDIRS,
}


def default_bundle_subdirs() -> tuple[str, ...]:
    """Platform-aware download set: Core AI plus int8 ONNX on macOS 27+, int8 ONNX elsewhere.

    onnx-int8 is the default ONNX artifact everywhere — desktop synthesis and
    the sample apps both prefer it. The larger fp32 `onnx` bundle, `litert`,
    and anything else must be requested explicitly (or via `all`).
    """

    if coreai_host_supported():
        return ("coreai", DEFAULT_ONNX_INT8_BUNDLE_SUBDIR)
    return (DEFAULT_ONNX_INT8_BUNDLE_SUBDIR,)


def download_litert_bundle(
    *,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    repo_id: str | None = None,
    model_version: str = DEFAULT_MODEL_VERSION,
    bundle_subdir: str = DEFAULT_BUNDLE_SUBDIR,
    token: str | None = None,
    revision: str | None = None,
    force: bool = False,
    validate: bool = True,
) -> Path:
    """Download the default Scylla's Band runtime bundle and return the local bundle root."""

    snapshot_download = _require_snapshot_download()
    model_version = normalize_model_version(model_version)
    repo_id = str(repo_id or MODEL_VERSION_REPO_IDS[model_version])
    target = (Path(models_dir) / model_version / bundle_subdir).resolve()
    if force and target.exists():
        shutil.rmtree(target)
    with tempfile.TemporaryDirectory(prefix="scyllasband_bundle_") as tmp_dir:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=[f"{bundle_subdir}/**"],
            local_dir=tmp_dir,
            token=token,
            revision=revision,
        )
        source = Path(tmp_dir) / bundle_subdir
        _copy_tree_contents(source, target)
    if validate:
        validate_bundle_layout(target)
    return target


def download_voice_packs(
    *,
    voices_dir: str | Path | None = None,
    repo_id: str | None = None,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    model_version: str = DEFAULT_MODEL_VERSION,
    bundle_subdir: str = DEFAULT_BUNDLE_SUBDIR,
    token: str | None = None,
    revision: str | None = None,
    force: bool = False,
) -> Path:
    """Download the separately addressable Scylla's Band voice-pack directory."""

    snapshot_download = _require_snapshot_download()
    model_version = normalize_model_version(model_version)
    repo_id = str(repo_id or MODEL_VERSION_REPO_IDS[model_version])
    target = Path(voices_dir or (Path(models_dir) / model_version / DEFAULT_VOICES_SUBDIR)).resolve()
    if force and target.exists():
        shutil.rmtree(target)
    with tempfile.TemporaryDirectory(prefix="scyllasband_voices_") as tmp_dir:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=[
                f"{DEFAULT_VOICES_SUBDIR}/**",
                f"{bundle_subdir}/assets/voice_packs/**",
            ],
            local_dir=tmp_dir,
            token=token,
            revision=revision,
        )
        source = _downloaded_voice_source(Path(tmp_dir), bundle_subdir)
        if source is not None:
            _copy_tree_contents(source, target)
        else:
            target.mkdir(parents=True, exist_ok=True)
    return target


def download_base_resources(
    *,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    repo_id: str | None = None,
    model_version: str = DEFAULT_MODEL_VERSION,
    bundle_subdir: str = "default",
    bundle_subdirs: Iterable[str] | None = None,
    token: str | None = None,
    revision: str | None = None,
    force: bool = False,
    validate: bool = True,
    include_voices: bool = True,
) -> tuple[Path, Path | None]:
    model_version = normalize_model_version(model_version)
    repo_id = str(repo_id or MODEL_VERSION_REPO_IDS[model_version])
    requested_subdirs = _normalize_bundle_subdirs(bundle_subdirs or (bundle_subdir,))
    bundle_dirs = []
    for subdir in requested_subdirs:
        bundle_dirs.append(
            download_litert_bundle(
                models_dir=models_dir,
                repo_id=repo_id,
                model_version=model_version,
                bundle_subdir=subdir,
                token=token,
                revision=revision,
                force=force,
                validate=validate,
            )
        )
    voices_dir = None
    if include_voices:
        voices_dir = download_voice_packs(
            models_dir=models_dir,
            repo_id=repo_id,
            model_version=model_version,
            bundle_subdir=requested_subdirs[0],
            token=token,
            revision=revision,
            force=force,
        )
    return bundle_dirs[0], voices_dir


def bundle_dirs_for_subdirs(
    models_dir: str | Path,
    subdirs: Iterable[str],
    *,
    model_version: str = DEFAULT_MODEL_VERSION,
) -> dict[str, Path]:
    version = normalize_model_version(model_version)
    return {
        subdir: (Path(models_dir) / version / subdir).resolve()
        for subdir in _normalize_bundle_subdirs(subdirs)
    }


def normalize_model_version(value: str | int | None) -> str:
    version = str(value or DEFAULT_MODEL_VERSION).strip().lower()
    if version in {"1", "v1"}:
        return "v1"
    if version in {"2", "v2"}:
        return "v2"
    raise ValueError(
        f"Unsupported model release {value!r}; expected one of: "
        f"{', '.join(SUPPORTED_MODEL_VERSIONS)}"
    )


def installed_model_release_paths(
    models_dir: str | Path,
    model_version: str,
) -> tuple[Path, ...]:
    """Return exact local paths owned by one release, including legacy flat v1 installs."""

    root = Path(models_dir).expanduser().resolve()
    version = normalize_model_version(model_version)
    paths: list[Path] = []
    version_root = root / version
    if version_root.exists() or version_root.is_symlink():
        paths.append(version_root)
    if version == "v1":
        legacy_bundles: list[Path] = []
        for subdir in SUPPORTED_BUNDLE_SUBDIRS:
            candidate = root / subdir
            if _bundle_model_major(candidate) == 1:
                legacy_bundles.append(candidate)
        paths.extend(legacy_bundles)
        legacy_voices = root / DEFAULT_VOICES_SUBDIR
        if legacy_bundles and (legacy_voices.exists() or legacy_voices.is_symlink()):
            paths.append(legacy_voices)
    return tuple(paths)


def model_release_installed(models_dir: str | Path, model_version: str) -> bool:
    return bool(installed_model_release_paths(models_dir, model_version))


def delete_model_release(models_dir: str | Path, model_version: str) -> tuple[Path, ...]:
    """Delete only paths classified as the requested installed model release."""

    root = Path(models_dir).expanduser().resolve()
    paths = installed_model_release_paths(root, model_version)
    for path in paths:
        if path.parent.resolve() != root:
            raise ValueError(f"Refusing to delete release path outside models_dir: {path}")
    for path in paths:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    return paths


def _bundle_model_major(bundle_dir: Path) -> int | None:
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return int(str(manifest.get("model_version") or "0").split(".", 1)[0])
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _normalize_bundle_subdirs(values: Iterable[str]) -> tuple[str, ...]:
    output: list[str] = []
    for value in values:
        item = str(value).strip().lower()
        if not item:
            continue
        if item == "default":
            candidates = default_bundle_subdirs()
        else:
            candidates = BUNDLE_SUBDIR_GROUPS.get(item, (item,))
        for candidate in candidates:
            if candidate not in SUPPORTED_BUNDLE_SUBDIRS:
                options = ", ".join(
                    (*SUPPORTED_BUNDLE_SUBDIRS, *BUNDLE_SUBDIR_GROUPS)
                )
                raise ValueError(f"Unsupported runtime bundle {candidate!r}; expected one of: {options}")
            if candidate not in output:
                output.append(candidate)
    return tuple(output or default_bundle_subdirs())


def _downloaded_voice_source(root: Path, bundle_subdir: str) -> Path | None:
    legacy_source = root / DEFAULT_VOICES_SUBDIR
    if legacy_source.is_dir():
        return legacy_source
    bundle_source = root / bundle_subdir / "assets" / "voice_packs"
    if bundle_source.is_dir():
        return bundle_source
    return None


def _require_snapshot_download():
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - exercised only without optional dep
        raise RuntimeError("Install huggingface_hub to download Scylla's Band model artifacts") from exc
    return snapshot_download


def _relative_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path.relative_to(root) for path in root.rglob("*") if path.is_file())


def _copy_tree_contents(source: Path, target: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Downloaded payload is missing expected directory: {source}")
    for rel_path in _relative_files(source):
        destination = target / rel_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / rel_path, destination)
