"""Download model bundles from Hugging Face, and pick the bundle that suits this machine.

This module imports only the standard library at import time, so first-run setup can ask it which bundle (and so
which runtime package) a command will use.
"""

from __future__ import annotations

from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import uuid

from .contract import validate_bundle_layout

REPO_ID = "spybyscript/scyllasband"
RELEASE = "v2-20261005"
DEFAULT_MODELS_DIR = Path("scyllasband/models")
DEFAULT_FLAVOR = "litert"
FLAVORS = {
    "litert": "LiteRT bundle: the default on Linux, Windows, Android and Intel Macs",
    "coreml": "Core ML bundle: the default on Apple silicon Macs, and for iOS 18 apps",
    "coreai": "Core AI bundle for apps on iOS, iPadOS and visionOS 27 (also runs on macOS 27)",
    "onnx": "ONNX Runtime bundle, for hosts that already use ONNX Runtime",
}
MIN_MACOS = {"coreai": 27, "coreml": 15}
OS27_SDKS = ("iphoneos", "xros")   # iOS/iPadOS and visionOS SDKs whose apps can use the Core AI bundle


def macos_major() -> int | None:
    """The macOS major version on an Apple silicon Mac; None elsewhere (Core ML and Core AI bundles need Apple silicon)."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return None
    try:
        return int(platform.mac_ver()[0].split(".")[0])
    except (ValueError, IndexError):
        return None


def xcode_sdk_major(sdk: str) -> int | None:
    """Major version of the selected Xcode's SDK for ``sdk`` (e.g. iphoneos); None without Xcode or that SDK."""
    try:
        result = subprocess.run(["xcrun", "--sdk", sdk, "--show-sdk-version"], capture_output=True, text=True, timeout=20)
        return int(result.stdout.strip().split(".")[0]) if result.returncode == 0 else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def targets_os27() -> bool:
    """True on macOS 27, or when Xcode can build iOS/iPadOS or visionOS 27 apps: the setups that use the Core AI bundle."""
    major = macos_major()
    if major is None:
        return False
    return major >= MIN_MACOS["coreai"] or any((xcode_sdk_major(sdk) or 0) >= MIN_MACOS["coreai"] for sdk in OS27_SDKS)


def supported_flavors() -> list[str]:
    """Bundles this machine can run, the default first. Apple silicon Macs run Core ML; Core AI also runs on macOS 27 but is
    for iOS/iPadOS/visionOS 27 apps, so it is never the Mac's default."""
    major = macos_major()
    apple = [flavor for flavor in ("coreml", "coreai") if major is not None and major >= MIN_MACOS[flavor]]
    return [*apple, "litert", "onnx"]


def recommended_flavor() -> str:
    """Core ML on Apple silicon Macs, LiteRT everywhere else."""
    return supported_flavors()[0]


def choose_flavors(*, interactive: bool, ask=input, out=sys.stderr) -> list[str]:
    """The bundles to download when none was named. A Mac gets the Core ML bundle it runs; when it can also build for
    iOS/iPadOS/visionOS 27 devices (macOS 27, or Xcode with those SDKs) it is offered the Core AI bundle as well."""
    recommended = recommended_flavor()
    if recommended != "coreml" or not targets_os27():
        return [recommended]
    if not interactive:
        print("Downloading the Core ML bundle this Mac runs. For iOS, iPadOS or visionOS 27 apps, also run: "
              "python -m scyllasband download --flavor coreai", file=out)
        return ["coreml"]
    print("This Mac runs the Core ML bundle. It can also build apps for iOS, iPadOS or visionOS 27, which use the Core AI "
          "bundle (about 155 MB).", file=out)
    while True:
        try:
            answer = ask("Download the Core AI bundle too? [y/N] ").strip().lower()
        except EOFError:
            answer = ""
        if answer in ("", "n", "no"):
            return ["coreml"]
        if answer in ("y", "yes"):
            return ["coreml", "coreai"]
        print("Answer y or n.", file=out)


def bundle_dir(models_dir: str | Path = DEFAULT_MODELS_DIR, flavor: str = DEFAULT_FLAVOR) -> Path:
    return (Path(models_dir) / _flavor(flavor)).resolve()


def installed_bundles(models_dir: str | Path = DEFAULT_MODELS_DIR) -> dict[str, Path]:
    return {flavor: bundle_dir(models_dir, flavor) for flavor in FLAVORS if (bundle_dir(models_dir, flavor) / "manifest.json").is_file()}


def default_installed_flavor(models_dir: str | Path = DEFAULT_MODELS_DIR) -> str | None:
    """The installed bundle a command uses when none is named: Core ML on an Apple silicon Mac, else LiteRT, then ONNX."""
    installed = installed_bundles(models_dir)
    return next((flavor for flavor in supported_flavors() if flavor in installed), None)


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
