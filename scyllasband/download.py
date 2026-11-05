"""Download model bundles from Hugging Face, and pick the bundle that suits this machine.

Two repositories are known: the singing release (``SING_REPO_ID``, a v4 bundle that speaks and sings) and the public
release (``REPO_ID``). ``download_bundle`` tries the singing repository first when the user's Hugging Face login can
reach it (the token from the environment or ``huggingface-cli login``) and the bundle flavour is published there, and
otherwise falls back to the public repository at its own release tag. Where a bundle came from is recorded in
``download_provenance.json`` beside its manifest.

This module imports only the standard library at import time, so first-run setup can ask it which bundle (and so
which runtime package) a command will use.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import uuid

from .contract import validate_bundle_layout

REPO_ID = "spybyscript/scyllasband"
RELEASE = "v3-20261104-apple"   # v3-20261104 with its Core ML and Core AI bundles
SING_REPO_ID = "spybyscript/scyllasband_sing"
SING_RELEASE = "v4-20261105"    # the singing release candidate (LiteRT, ONNX, PyTorch); the official release is RELEASE in REPO_ID
PROVENANCE_FILE = "download_provenance.json"
DEFAULT_MODELS_DIR = Path("scyllasband/models")
DEFAULT_FLAVOR = "litert"
FLAVORS = {
    "litert": "LiteRT bundle: the default on Linux, Windows, Android and Intel Macs",
    "coreml": "Core ML bundle: the default on Apple silicon Macs, and for iOS 18 apps",
    "coreai": "Core AI bundle for apps on iOS, iPadOS and visionOS 27 (also runs on macOS 27)",
    "onnx": "ONNX Runtime bundle, for hosts that already use ONNX Runtime",
    "webgpu": "ONNX bundle with 8-bit weights: for WebGPU in browsers, and for NVIDIA GPUs (--backend onnx --compute-units gpu)",
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
    return [*apple, "litert", "onnx", "webgpu"]


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


def default_revision(repo_id: str) -> str | None:
    """The release tag a known repository is pinned to; None (the default branch) for any other repository."""
    return {SING_REPO_ID: SING_RELEASE, REPO_ID: RELEASE}.get(repo_id)


def download_candidates(*, repo_id: str | None = None, revision: str | None = None, public: bool = False) -> list[tuple[str, str | None]]:
    """The (repository, revision) pairs ``download_bundle`` tries in order: an explicit repository alone (at ``revision``,
    else its known release tag); with ``public`` the public repository alone; otherwise the singing repository (at
    ``revision``, else ``SING_RELEASE``) and then the public repository at ``RELEASE``."""
    if repo_id:
        return [(repo_id, revision or default_revision(repo_id))]
    if public:
        return [(REPO_ID, revision or RELEASE)]
    return [(SING_REPO_ID, revision or SING_RELEASE), (REPO_ID, RELEASE)]


def download_bundle(*, flavor: str = DEFAULT_FLAVOR, models_dir: str | Path = DEFAULT_MODELS_DIR, repo_id: str | None = None,
                    revision: str | None = None, token: str | None = None, force: bool = False, validate: bool = True,
                    public: bool = False, log=sys.stderr) -> Path:
    """Download one bundle flavour, validate it, then swap it into place (an installed bundle survives a failed download).

    Without ``repo_id`` the singing repository is tried first; when it is not accessible to this user (no or wrong token:
    401/403/404, a gated, missing or private repository, a missing revision) or does not publish ``flavor``, one line is
    logged and the public repository is used at ``RELEASE``. ``public`` skips the singing repository; ``repo_id`` names a
    repository explicitly (``revision`` its tag, else the repository's known release tag or its default branch).
    """
    flavor = _flavor(flavor)
    snapshot_download = _require_snapshot_download()
    target = bundle_dir(models_dir, flavor)
    target.parent.mkdir(parents=True, exist_ok=True)
    candidates = download_candidates(repo_id=repo_id, revision=revision, public=public)
    with tempfile.TemporaryDirectory(prefix=".scyllasband_bundle_", dir=target.parent) as tmp:
        for index, (repo, rev) in enumerate(candidates):
            last = index == len(candidates) - 1
            local = Path(tmp) / f"try{index}"
            try:
                snapshot_download(repo_id=repo, repo_type="model", allow_patterns=[f"{flavor}/**"], local_dir=str(local),
                                  force_download=force, token=token, revision=rev)
            except Exception as exc:   # noqa: BLE001 - only the Hub's "not accessible" errors fall through to the next repository
                if last or not inaccessible(exc):
                    raise
                _log(log, f"{repo}@{rev} is not accessible with the current Hugging Face login ({type(exc).__name__}); "
                          f"using {candidates[index + 1][0]}@{candidates[index + 1][1]}")
                continue
            source = local / flavor
            if source.is_dir():
                break
            if last:
                raise FileNotFoundError(f"{repo}@{rev} has no {flavor!r} bundle")
            _log(log, f"{repo}@{rev} has no {flavor!r} bundle; using {candidates[index + 1][0]}@{candidates[index + 1][1]}")
        if validate:
            validate_bundle_layout(source)
        _write_provenance(source, repo_id=repo, revision=rev, flavor=flavor)
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


def inaccessible(exc: BaseException) -> bool:
    """True for the Hub errors that mean this user cannot read the repository or revision (401, 403, 404; a repository
    that is private, gated or missing; a missing revision), as opposed to a network or disk failure."""
    try:
        from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError, RevisionNotFoundError
    except ImportError:   # pragma: no cover - depends on the environment
        return False
    if isinstance(exc, (RepositoryNotFoundError, GatedRepoError, RevisionNotFoundError)):
        return True
    if isinstance(exc, HfHubHTTPError):
        response = getattr(exc, "response", None)
        return getattr(response, "status_code", None) in (401, 403, 404)
    return False


def read_provenance(bundle: str | Path) -> dict | None:
    """Where an installed bundle came from (``download_provenance.json``), or None for a bundle installed by other means."""
    path = Path(bundle) / PROVENANCE_FILE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def _write_provenance(source: Path, *, repo_id: str, revision: str | None, flavor: str) -> None:
    contract = None
    try:
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        contract = (manifest.get("controls") or {}).get("graph_input_contract")
    except (OSError, ValueError):
        pass
    record = dict(repo_id=repo_id, revision=revision, flavor=flavor, graph_contract=contract,
                  downloaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    (source / PROVENANCE_FILE).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def _log(stream, message: str) -> None:
    if stream is not None:
        print(message, file=stream, flush=True)


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
