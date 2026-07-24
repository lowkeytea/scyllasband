from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
from typing import Iterable

from .contract import validate_bundle_layout


DEFAULT_INFERENCE_REPO_ID = "spybyscript/scyllasband"
DEFAULT_MODELS_DIR = Path("scyllasband/models")
DEFAULT_BUNDLE_SUBDIR = "onnx"
DEFAULT_ONNX_INT8_BUNDLE_SUBDIR = "onnx-int8"
DEFAULT_VOICES_SUBDIR = "voices"
SUPPORTED_BUNDLE_SUBDIRS = (
    DEFAULT_BUNDLE_SUBDIR,
    DEFAULT_ONNX_INT8_BUNDLE_SUBDIR,
    "litert",
)
BUNDLE_SUBDIR_GROUPS = {
    "both": (DEFAULT_BUNDLE_SUBDIR, "litert"),
    "all": SUPPORTED_BUNDLE_SUBDIRS,
}


def download_litert_bundle(
    *,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    repo_id: str = DEFAULT_INFERENCE_REPO_ID,
    bundle_subdir: str = DEFAULT_BUNDLE_SUBDIR,
    token: str | None = None,
    revision: str | None = None,
    force: bool = False,
    validate: bool = True,
) -> Path:
    """Download the default Scylla's Band runtime bundle and return the local bundle root."""

    snapshot_download = _require_snapshot_download()
    target = (Path(models_dir) / bundle_subdir).resolve()
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
    voices_dir: str | Path = DEFAULT_MODELS_DIR / DEFAULT_VOICES_SUBDIR,
    repo_id: str = DEFAULT_INFERENCE_REPO_ID,
    bundle_subdir: str = DEFAULT_BUNDLE_SUBDIR,
    token: str | None = None,
    revision: str | None = None,
    force: bool = False,
) -> Path:
    """Download the separately addressable Scylla's Band voice-pack directory."""

    snapshot_download = _require_snapshot_download()
    target = Path(voices_dir).resolve()
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
    repo_id: str = DEFAULT_INFERENCE_REPO_ID,
    bundle_subdir: str = DEFAULT_BUNDLE_SUBDIR,
    bundle_subdirs: Iterable[str] | None = None,
    token: str | None = None,
    revision: str | None = None,
    force: bool = False,
    validate: bool = True,
    include_voices: bool = True,
) -> tuple[Path, Path | None]:
    requested_subdirs = _normalize_bundle_subdirs(bundle_subdirs or (bundle_subdir,))
    bundle_dirs = []
    for subdir in requested_subdirs:
        bundle_dirs.append(
            download_litert_bundle(
                models_dir=models_dir,
                repo_id=repo_id,
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
            voices_dir=Path(models_dir) / DEFAULT_VOICES_SUBDIR,
            repo_id=repo_id,
            bundle_subdir=requested_subdirs[0],
            token=token,
            revision=revision,
            force=force,
        )
    return bundle_dirs[0], voices_dir


def bundle_dirs_for_subdirs(models_dir: str | Path, subdirs: Iterable[str]) -> dict[str, Path]:
    return {subdir: (Path(models_dir) / subdir).resolve() for subdir in _normalize_bundle_subdirs(subdirs)}


def _normalize_bundle_subdirs(values: Iterable[str]) -> tuple[str, ...]:
    output: list[str] = []
    for value in values:
        item = str(value).strip().lower()
        if not item:
            continue
        candidates = BUNDLE_SUBDIR_GROUPS.get(item, (item,))
        for candidate in candidates:
            if candidate not in SUPPORTED_BUNDLE_SUBDIRS:
                options = ", ".join(
                    (*SUPPORTED_BUNDLE_SUBDIRS, *BUNDLE_SUBDIR_GROUPS)
                )
                raise ValueError(f"Unsupported runtime bundle {candidate!r}; expected one of: {options}")
            if candidate not in output:
                output.append(candidate)
    return tuple(output or (DEFAULT_BUNDLE_SUBDIR,))


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
