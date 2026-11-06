"""python -m scyllasband build-native: build libscyllasband from this checkout for the Python package.

Runs CMake on libscyllasband/ with one graph backend (litert, apple or onnx), then copies the shared library and the SDK
libraries it loads (staged under libscyllasband/third_party/<sdk>/lib/<platform>/) into scyllasband/_native/<backend>/,
where scyllasband.native finds it. The copied library looks for the SDK libraries beside itself. The first LiteRT build
stages LiteRT's prebuilt runtime for this machine (libscyllasband/scripts/stage_litert_sdk.py downloads it).
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TextIO

from .native import CHECKOUT_DIR, PACKAGE_DIR, library_name

BACKENDS = ("litert", "apple", "onnx")
SDKS = {"litert": "litert", "onnx": "onnxruntime"}


def default_backend() -> str:
    return "litert"


def sdk_platform() -> str:
    """The staged SDK's platform directory, as libscyllasband's CMakeLists names it."""
    machine = platform.machine().lower()
    if sys.platform == "darwin":
        return "macos-arm64"
    if sys.platform == "win32":
        return "windows-x86_64"
    return "linux-arm64" if machine in ("aarch64", "arm64") else "linux-x86_64"


def stage_sdk(backend: str, *, log: TextIO = sys.stderr) -> None:
    """Stage LiteRT's prebuilt runtime for this machine when it is not staged yet; other SDKs are staged by hand."""
    if backend != "litert":
        return
    staged = CHECKOUT_DIR / "third_party" / "litert" / "lib" / sdk_platform()
    if any(staged.glob("libLiteRt.*")):
        return
    print(f"[scyllasband] staging LiteRT for {sdk_platform()} (scripts/stage_litert_sdk.py)", file=log, flush=True)
    subprocess.run([sys.executable, str(CHECKOUT_DIR / "scripts" / "stage_litert_sdk.py"), "--platform", sdk_platform(),
                    "--download-runtime"], check=True)


def build_native(backend: str = "litert", *, jobs: int | None = None, log: TextIO = sys.stderr) -> Path:
    if backend not in BACKENDS:
        raise ValueError(f"Unknown native backend {backend!r}; choose one of {', '.join(BACKENDS)}")
    if backend == "apple" and sys.platform != "darwin":
        raise ValueError("The apple backend (Core ML and Core AI) builds only on macOS")
    if not (CHECKOUT_DIR / "CMakeLists.txt").is_file():
        raise FileNotFoundError("build-native needs a checkout of the repository: libscyllasband/ beside the scyllasband package")
    cmake = shutil.which("cmake")
    if cmake is None:
        raise FileNotFoundError("build-native needs CMake 3.20 or later on PATH")
    stage_sdk(backend, log=log)
    build = CHECKOUT_DIR / "build" / f"python-{backend}"
    configure = [cmake, "-S", str(CHECKOUT_DIR), "-B", str(build), f"-DSCYLLASBAND_BACKEND={backend}", "-DSCYLLASBAND_BUILD_TOOLS=OFF",
                 "-DSCYLLASBAND_BUILD_TESTS=OFF", "-DCMAKE_BUILD_TYPE=Release"]
    if sys.platform == "darwin":
        configure.append("-DCMAKE_SHARED_LINKER_FLAGS=-Wl,-rpath,@loader_path")
    elif sys.platform != "win32":
        configure.append("-DCMAKE_SHARED_LINKER_FLAGS=-Wl,-rpath,$ORIGIN")
    if shutil.which("ninja") and not (build / "CMakeCache.txt").is_file():
        configure += ["-G", "Ninja"]   # the apple backend's Swift sources need Ninja or Xcode
    elif backend == "apple" and not shutil.which("ninja") and not (build / "CMakeCache.txt").is_file():
        raise FileNotFoundError("The apple backend builds with Ninja (brew install ninja)")
    print(f"[scyllasband] configuring {build}", file=log, flush=True)
    subprocess.run(configure, check=True)
    command = [cmake, "--build", str(build), "--target", "scyllasband_native"]
    if jobs:
        command += ["-j", str(int(jobs))]
    print(f"[scyllasband] building libscyllasband ({backend})", file=log, flush=True)
    subprocess.run(command, check=True)
    built = build / library_name()
    if not built.exists():
        raise FileNotFoundError(f"The build produced no {built}")
    target = PACKAGE_DIR / backend
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    shutil.copy2(built.resolve(), target / library_name())
    sdk = SDKS.get(backend)
    if sdk:
        for path in sorted((CHECKOUT_DIR / "third_party" / sdk / "lib" / sdk_platform()).glob("*")):
            if path.is_file() and not path.name.startswith("."):
                shutil.copy2(path.resolve(), target / path.name)
    print(f"[scyllasband] installed {target / library_name()}", file=log, flush=True)
    return target / library_name()
