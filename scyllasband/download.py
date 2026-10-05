"""Download model bundles from Hugging Face."""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import uuid

from .contract import validate_bundle_layout

REPO_ID = "spybyscript/scyllasband"
RELEASE = "v2-20261005"
DEFAULT_MODELS_DIR = Path("scyllasband/models")
DEFAULT_FLAVOR = "litert"
FLAVORS = {
    "litert": "LiteRT bundle: the default on desktop, Android and embedded devices",
    "onnx": "ONNX Runtime bundle, for hosts that already use ONNX Runtime",
}


def bundle_dir(models_dir: str | Path = DEFAULT_MODELS_DIR, flavor: str = DEFAULT_FLAVOR) -> Path:
    return (Path(models_dir) / _flavor(flavor)).resolve()


def installed_bundles(models_dir: str | Path = DEFAULT_MODELS_DIR) -> dict[str, Path]:
    return {flavor: bundle_dir(models_dir, flavor) for flavor in FLAVORS if (bundle_dir(models_dir, flavor) / "manifest.json").is_file()}


def download_bundle(*, flavor: str = DEFAULT_FLAVOR, models_dir: str | Path = DEFAULT_MODELS_DIR, repo_id: str = REPO_ID,
                    revision: str | None = RELEASE, token: str | None = None, force: bool = False, validate: bool = True) -> Path:
    """Download one bundle flavour, validate it, then swap it into place (an installed bundle survives a failed download)."""
    flavor = _flavor(flavor)
    snapshot_download = _require_snapshot_download()
    target = bundle_dir(models_dir, flavor)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".scyllasband_bundle_", dir=target.parent) as tmp:
        snapshot_download(repo_id=repo_id, repo_type="model", allow_patterns=[f"{flavor}/**"], local_dir=tmp,
                          force_download=force, token=token, revision=revision)
        source = Path(tmp) / flavor
        if not source.is_dir():
            raise FileNotFoundError(f"{repo_id}@{revision} has no {flavor!r} bundle")
        if validate:
            validate_bundle_layout(source)
        backup = target.with_name(f"{target.name}.previous-{uuid.uuid4().hex}")
        had_target = target.exists()
        if had_target:
            target.rename(backup)
        try:
            source.rename(target)
        except BaseException:
            if had_target:
                backup.rename(target)
            raise
        if had_target:
            shutil.rmtree(backup)
    return target


def _flavor(value: str) -> str:
    flavor = str(value or DEFAULT_FLAVOR).strip().lower()
    if flavor not in FLAVORS:
        raise ValueError(f"Unknown bundle {value!r}; choose one of {', '.join(FLAVORS)}")
    return flavor


def _require_snapshot_download():
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:   # pragma: no cover - depends on the environment
        raise RuntimeError("Downloading requires huggingface_hub: pip install huggingface_hub") from exc
    return snapshot_download
