"""First-run environment setup for a checkout of this repository.

``python -m scyllasband``, ``speak.py`` and ``groupSpeak.py`` call :func:`ensure_environment` before importing anything
that needs third-party packages. When packages are missing it offers to create ``.venv`` in the checkout (or to install
into the virtual environment already active), installs the runtime there and runs the same command again with it.
Later runs from the checkout switch to ``.venv`` on their own.

This module uses only the standard library and syntax that older Python versions can parse, so it can explain the
problem on any interpreter. Set ``SCYLLASBAND_NO_BOOTSTRAP=1`` to skip it, or ``SCYLLASBAND_ASSUME_YES=1`` to accept the
setup without a prompt.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

MIN_PYTHON = (3, 10)
REQUIRED = {"numpy": "numpy", "huggingface_hub": "huggingface_hub", "ai_edge_litert": "ai-edge-litert"}
ONNX = {"onnxruntime": "onnxruntime"}
REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = REPO_ROOT / ".venv"
_RELAUNCHED = "SCYLLASBAND_BOOTSTRAPPED"


class SetupError(RuntimeError):
    pass


def ensure_environment(argv, relaunch=None):
    """Return when the packages ``argv`` needs are importable; otherwise set them up and re-run, or exit with advice.

    ``relaunch`` is the argument list after the interpreter that re-runs this command (default ``-m scyllasband argv``).
    """
    if os.environ.get("SCYLLASBAND_NO_BOOTSTRAP"):
        return
    argv = list(argv)
    relaunch = list(relaunch) if relaunch is not None else ["-m", "scyllasband", *argv]
    missing = missing_packages(argv)
    if not missing:
        return
    onnx = wants_onnx(argv)
    action = plan_setup(missing, checkout=is_checkout(), venv_ready=venv_python().is_file(), in_venv=in_virtualenv(),
                        in_repo_venv=in_repo_venv(), python_ok=sys.version_info[:2] >= MIN_PYTHON,
                        previous=os.environ.get(_RELAUNCHED))
    try:
        if action == "use-venv":
            _run_again(venv_python(), relaunch, "use-venv")
        if action.startswith("fail-"):
            raise SetupError(_failure(action, missing, onnx))
        target = sys.executable if action == "install-here" else str(venv_python())
        where = f"the active environment ({sys.prefix})" if action == "install-here" else f"a new virtual environment at {VENV_DIR}"
        print(f"Scylla's Band needs {_names(missing)}, which {sys.executable} does not have.", file=sys.stderr)
        if not _confirm(f"Install {'them' if len(missing) > 1 else 'it'} into {where}? [Y/n] "):
            reason = ("setup needs confirmation; run the command in a terminal or set SCYLLASBAND_ASSUME_YES=1"
                      if not _interactive() else "setup skipped")
            raise SetupError(f"{reason}.\n{_manual_steps(onnx)}")
        if action == "create-venv":
            print(f"Creating {VENV_DIR} ...", file=sys.stderr)
            subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
        _install(target, onnx)
        if action == "create-venv":
            print(f"Ready. Commands run from {REPO_ROOT} use {VENV_DIR} from now on.", file=sys.stderr)
        _run_again(Path(target), relaunch, "installed")
    except SetupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except subprocess.CalledProcessError as exc:
        print(f"error: setup failed ({' '.join(map(str, exc.cmd))} exited with {exc.returncode}).\n{_platform_note()}\n"
              f"{_manual_steps(onnx)}", file=sys.stderr)
        raise SystemExit(2)


def plan_setup(missing, *, checkout, venv_ready, in_venv, in_repo_venv, python_ok, previous=None):
    """Pick one action: use-venv, install-here, create-venv, or a fail-* reason.

    ``previous`` is what the process that launched this one already did ("use-venv" or "installed"), so setup never loops.
    """
    if not missing:
        return "ok"
    if previous == "installed" or (previous == "use-venv" and not in_repo_venv):
        return "fail-loop"
    if checkout and venv_ready and not in_repo_venv:
        return "use-venv"           # the checkout's environment exists: switch to it (it installs anything still missing)
    if not checkout:
        return "fail-installed"     # an installed copy is missing its own dependencies
    if not python_ok:
        return "fail-python"
    return "install-here" if in_venv else "create-venv"


def missing_packages(argv):
    needed = dict(REQUIRED)
    if wants_onnx(argv):
        needed.update(ONNX)
    return {module: package for module, package in needed.items() if importlib.util.find_spec(module) is None}


def wants_onnx(argv):
    """True when the command will run the ONNX backend: ``--backend onnx`` or a ``--bundle`` whose manifest prefers it."""
    args = list(argv)
    for index, arg in enumerate(args):
        value = None
        if arg in ("--backend", "--bundle") and index + 1 < len(args):
            value = args[index + 1]
        elif arg.startswith(("--backend=", "--bundle=")):
            value = arg.split("=", 1)[1]
        if value is None:
            continue
        if arg.startswith("--backend"):
            if value == "onnx":
                return True
        else:
            try:
                manifest = json.loads((Path(value) / "manifest.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if "--backend" not in " ".join(args) and (manifest.get("preferred_backends") or [None])[0] == "onnx":
                return True
    return False


def is_checkout():
    return (REPO_ROOT / "pyproject.toml").is_file() and (REPO_ROOT / "scyllasband" / "__init__.py").is_file()


def venv_python():
    return VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def in_virtualenv():
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def in_repo_venv():
    try:
        return Path(sys.prefix).resolve() == VENV_DIR.resolve()
    except OSError:
        return False


def _install(python, onnx):
    target = f"{REPO_ROOT}[onnx]" if onnx else str(REPO_ROOT)
    print(f"Installing the Scylla's Band runtime{' with ONNX Runtime' if onnx else ''} ...", file=sys.stderr)
    subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check", "-e", target], check=True)


def _run_again(python, relaunch, done):
    env = dict(os.environ, **{_RELAUNCHED: done})
    command = [str(python), *relaunch]
    sys.stdout.flush()
    sys.stderr.flush()
    if os.name == "nt":
        raise SystemExit(subprocess.call(command, env=env))
    os.execve(command[0], command, env)


def _confirm(question):
    if os.environ.get("SCYLLASBAND_ASSUME_YES"):
        return True
    if not _interactive():
        return False
    try:
        answer = input(question).strip().lower()
    except EOFError:
        return False
    return answer in ("", "y", "yes")


def _interactive():
    return bool(sys.stdin) and sys.stdin.isatty()


def _names(missing):
    names = sorted(set(missing.values()))
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _manual_steps(onnx):
    python = "python3" if os.name != "nt" else "py -3"
    activate = ".venv\\Scripts\\activate" if os.name == "nt" else "source .venv/bin/activate"
    extra = '".[onnx]"' if onnx else "."
    return (f"To set up by hand, from {REPO_ROOT}:\n"
            f"    {python} -m venv .venv\n    {activate}\n    pip install -e {extra}")


def _platform_note():
    return ("The runtime packages have wheels for Python 3.10–3.14 on macOS with Apple silicon, Linux (x86_64, aarch64) "
            "and Windows (x86_64).")


def _failure(action, missing, onnx):
    version = ".".join(map(str, sys.version_info[:3]))
    if action == "fail-python":
        return (f"Scylla's Band needs Python 3.10 or newer; {sys.executable} is {version}. Install a newer Python "
                "(python.org, or `brew install python@3.12` on macOS) and run the command with it, e.g. "
                "`python3.12 -m scyllasband ...`.")
    if action == "fail-installed":
        return f"Missing {_names(missing)}. Install with: {sys.executable} -m pip install {' '.join(sorted(set(missing.values())))}"
    return f"{_names(missing)} still cannot be imported after setup.\n{_platform_note()}\n{_manual_steps(onnx)}"
