#!/usr/bin/env python3
"""Stage LiteRT shared libraries into
`scyllasband/libscyllasband/third_party/litert/lib/<platform>/`.

The headers under `third_party/litert/include/` are committed and pinned to
the matching upstream tag (currently v2.1.5). Headers are not staged by this
script -- only the runtime library and optional GPU accelerator prebuilts. If
you bump the LiteRT version, refresh the headers manually from upstream
`google-ai-edge/LiteRT` at the matching tag, then run this script to refresh
the runtime.
"""
from __future__ import annotations

import argparse
import importlib
import platform
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DEFAULT_DOWNLOAD_VERSION = "2.1.5"
DOWNLOAD_BASE_URL = "https://storage.googleapis.com/litert/binaries"


@dataclass(frozen=True)
class LiteRTPlatform:
    stage_dir: str
    download_platform: str
    runtime_lib: str
    gpu_accelerator_lib: Optional[str]
    gpu_backend: Optional[str]


SUPPORTED_PLATFORMS = {
    "android-arm64": LiteRTPlatform(
        stage_dir="android-arm64",
        download_platform="android_arm64",
        runtime_lib="libLiteRt.so",
        gpu_accelerator_lib="libLiteRtClGlAccelerator.so",
        gpu_backend="OpenCL + OpenGL",
    ),
    "android-x86_64": LiteRTPlatform(
        stage_dir="android-x86_64",
        download_platform="android_x86_64",
        runtime_lib="libLiteRt.so",
        gpu_accelerator_lib="libLiteRtClGlAccelerator.so",
        gpu_backend="OpenCL + OpenGL",
    ),
    "linux-x86_64": LiteRTPlatform(
        stage_dir="linux-x86_64",
        download_platform="linux_x86_64",
        runtime_lib="libLiteRt.so",
        gpu_accelerator_lib="libLiteRtWebGpuAccelerator.so",
        gpu_backend="WebGPU (Vulkan)",
    ),
    "linux-arm64": LiteRTPlatform(
        stage_dir="linux-arm64",
        download_platform="linux_arm64",
        runtime_lib="libLiteRt.so",
        gpu_accelerator_lib="libLiteRtWebGpuAccelerator.so",
        gpu_backend="WebGPU (Vulkan)",
    ),
    "macos-arm64": LiteRTPlatform(
        stage_dir="macos-arm64",
        download_platform="macos_arm64",
        runtime_lib="libLiteRt.dylib",
        gpu_accelerator_lib="libLiteRtMetalAccelerator.dylib",
        gpu_backend="Metal",
    ),
    "ios-arm64": LiteRTPlatform(
        stage_dir="ios-arm64",
        download_platform="ios_arm64",
        runtime_lib="libLiteRt.dylib",
        gpu_accelerator_lib="libLiteRtMetalAccelerator.dylib",
        gpu_backend="Metal",
    ),
    "ios-sim-arm64": LiteRTPlatform(
        stage_dir="ios-sim-arm64",
        download_platform="ios_sim_arm64",
        runtime_lib="libLiteRt.dylib",
        gpu_accelerator_lib="libLiteRtMetalAccelerator.dylib",
        gpu_backend="Metal",
    ),
    "windows-x86_64": LiteRTPlatform(
        stage_dir="windows-x86_64",
        download_platform="windows_x86_64",
        runtime_lib="libLiteRt.dll",
        gpu_accelerator_lib="libLiteRtWebGpuAccelerator.dll",
        gpu_backend="WebGPU (Direct3D)",
    ),
}


PLATFORM_ALIASES = {
    "android_aarch64": "android-arm64",
    "android-aarch64": "android-arm64",
    "android_arm64": "android-arm64",
    "android-arm64-v8a": "android-arm64",
    "android_arm64_v8a": "android-arm64",
    "arm64-v8a": "android-arm64",
    "android_x86_64": "android-x86_64",
    "android-amd64": "android-x86_64",
    "android_amd64": "android-x86_64",
    "android-x64": "android-x86_64",
    "android_x64": "android-x86_64",
    "linux_x86_64": "linux-x86_64",
    "linux_amd64": "linux-x86_64",
    "linux-amd64": "linux-x86_64",
    "linux_aarch64": "linux-arm64",
    "linux-aarch64": "linux-arm64",
    "linux_arm64": "linux-arm64",
    "macos_arm64": "macos-arm64",
    "macos-aarch64": "macos-arm64",
    "darwin-arm64": "macos-arm64",
    "darwin-aarch64": "macos-arm64",
    "ios_arm64": "ios-arm64",
    "ios-aarch64": "ios-arm64",
    "iphoneos-arm64": "ios-arm64",
    "iphoneos-aarch64": "ios-arm64",
    "ios_sim_arm64": "ios-sim-arm64",
    "ios-sim-aarch64": "ios-sim-arm64",
    "ios-simulator-arm64": "ios-sim-arm64",
    "ios-simulator-aarch64": "ios-sim-arm64",
    "iphonesimulator-arm64": "ios-sim-arm64",
    "iphonesimulator-aarch64": "ios-sim-arm64",
    "windows_x86_64": "windows-x86_64",
    "windows-amd64": "windows-x86_64",
    "windows_amd64": "windows-x86_64",
    "win32-x86_64": "windows-x86_64",
    "win32-amd64": "windows-x86_64",
}


def _normalize_platform(value: str) -> str:
    normalized = value.strip().lower().replace("/", "-")
    normalized = PLATFORM_ALIASES.get(normalized, normalized)
    if normalized not in SUPPORTED_PLATFORMS:
        supported = ", ".join(sorted(SUPPORTED_PLATFORMS))
        raise RuntimeError(f"Unsupported LiteRT platform '{value}'. Supported: {supported}")
    return normalized


def _host_platform_name() -> str:
    machine = platform.machine().lower()
    if sys.platform == "darwin":
        if machine not in {"arm64", "aarch64"}:
            raise RuntimeError("LiteRT prebuilt downloads currently publish macos-arm64 for macOS hosts.")
        return "macos-arm64"
    if sys.platform.startswith("linux"):
        if machine in {"aarch64", "arm64"}:
            return "linux-arm64"
        return "linux-x86_64"
    if sys.platform.startswith("win"):
        return "windows-x86_64"
    raise RuntimeError(f"Unsupported host platform for staging: {sys.platform}")


def _download_url(target: LiteRTPlatform, version: str, filename: str) -> str:
    return f"{DOWNLOAD_BASE_URL}/{version}/{target.download_platform}/{filename}"


def _discover_from_python_package(package_name: str, runtime_lib: str) -> Path:
    mod = importlib.import_module(package_name)
    pkg_dir = Path(mod.__file__).resolve().parent
    candidate = pkg_dir / runtime_lib
    if not candidate.exists():
        raise RuntimeError(f"{package_name} package at {pkg_dir} does not contain {runtime_lib}")
    return candidate


def _copy_file(src: Path, dst: Path, *, overwrite: bool, label: str) -> None:
    if dst.exists() and not overwrite:
        print(f"Already staged {label}: {dst} (pass --overwrite to replace)")
        return
    shutil.copy2(src, dst)
    print(f"Staged {label}: {src} -> {dst}")


def _download_file(url: str, dst: Path, *, overwrite: bool, label: str) -> None:
    if dst.exists() and not overwrite:
        print(f"Already staged {label}: {dst} (pass --overwrite to replace)")
        return
    tmp = dst.with_suffix(dst.suffix + ".download")
    request = urllib.request.Request(url, headers={"User-Agent": "scyllasband-litert-stage/1.0"})
    with urllib.request.urlopen(request) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out)
    tmp.replace(dst)
    print(f"Downloaded {label}: {url} -> {dst}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        help=(
            "Target platform to stage for (default: current host). Supported: "
            + ", ".join(sorted(SUPPORTED_PLATFORMS))
        ),
    )
    parser.add_argument(
        "--python-package",
        default="ai_edge_litert",
        help="Python package to inspect for the runtime library (default: ai_edge_litert).",
    )
    parser.add_argument(
        "--source-lib",
        type=Path,
        help="Explicit path to libLiteRt shared library. Overrides --python-package.",
    )
    parser.add_argument(
        "--source-gpu-accelerator-lib",
        type=Path,
        help="Explicit path to a LiteRT GPU accelerator shared library for --platform.",
    )
    parser.add_argument(
        "--download-version",
        default=DEFAULT_DOWNLOAD_VERSION,
        help=f"LiteRT prebuilt version to download (default: {DEFAULT_DOWNLOAD_VERSION}).",
    )
    parser.add_argument(
        "--download-runtime",
        action="store_true",
        help="Download the matching prebuilt LiteRT runtime shared library.",
    )
    parser.add_argument(
        "--download-gpu-accelerator",
        action="store_true",
        help="Download the matching prebuilt LiteRT GPU accelerator shared library.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "third_party" / "litert",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing staged libraries.",
    )
    args = parser.parse_args()

    if args.source_lib is not None and args.download_runtime:
        parser.error("--source-lib and --download-runtime are mutually exclusive")
    if args.source_gpu_accelerator_lib is not None and args.download_gpu_accelerator:
        parser.error("--source-gpu-accelerator-lib and --download-gpu-accelerator are mutually exclusive")

    platform_name = _normalize_platform(args.platform) if args.platform else _host_platform_name()
    target = SUPPORTED_PLATFORMS[platform_name]
    output_root = args.output_root.expanduser().resolve()
    lib_output_dir = output_root / "lib" / target.stage_dir
    lib_output_dir.mkdir(parents=True, exist_ok=True)

    runtime_dst = lib_output_dir / target.runtime_lib
    if args.download_runtime:
        _download_file(
            _download_url(target, str(args.download_version), target.runtime_lib),
            runtime_dst,
            overwrite=bool(args.overwrite),
            label="LiteRT runtime",
        )
    elif args.source_lib is not None:
        src = args.source_lib.expanduser().resolve()
        if not src.is_file():
            raise RuntimeError(f"Source LiteRT library does not exist: {src}")
        _copy_file(src, runtime_dst, overwrite=bool(args.overwrite), label="LiteRT runtime")
    else:
        host_platform = _host_platform_name()
        if platform_name != host_platform:
            raise RuntimeError(
                "Cross-platform staging requires --download-runtime or --source-lib "
                f"(host={host_platform}, target={platform_name})."
            )
        src = _discover_from_python_package(str(args.python_package), target.runtime_lib)
        _copy_file(src, runtime_dst, overwrite=bool(args.overwrite), label="LiteRT runtime")

    if args.download_gpu_accelerator:
        if target.gpu_accelerator_lib is None:
            raise RuntimeError(f"No LiteRT GPU accelerator prebuilt is known for {platform_name}")
        _download_file(
            _download_url(target, str(args.download_version), target.gpu_accelerator_lib),
            lib_output_dir / target.gpu_accelerator_lib,
            overwrite=bool(args.overwrite),
            label=f"LiteRT GPU accelerator ({target.gpu_backend})",
        )
    elif args.source_gpu_accelerator_lib is not None:
        if target.gpu_accelerator_lib is None:
            raise RuntimeError(f"No LiteRT GPU accelerator prebuilt is known for {platform_name}")
        src = args.source_gpu_accelerator_lib.expanduser().resolve()
        if not src.is_file():
            raise RuntimeError(f"Source LiteRT GPU accelerator library does not exist: {src}")
        _copy_file(
            src,
            lib_output_dir / target.gpu_accelerator_lib,
            overwrite=bool(args.overwrite),
            label=f"LiteRT GPU accelerator ({target.gpu_backend})",
        )

    if args.download_gpu_accelerator or args.source_gpu_accelerator_lib is not None:
        print(
            "LiteRT GPU accelerator staged next to the runtime library. "
            "Reconfigure/rebuild libscyllasband before testing --litert-accelerator gpu; "
            "libscyllasband preloads staged accelerator plugins before LiteRT environment creation."
        )
        if platform_name.startswith("linux-"):
            print(
                "If a standalone diagnostic still cannot find the accelerator, use fallback: "
                f"LD_LIBRARY_PATH={lib_output_dir}:${{LD_LIBRARY_PATH:-}}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
