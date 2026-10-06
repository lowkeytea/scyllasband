"""Command-line interface: python -m scyllasband <command>."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import wave

import numpy as np

from .contract import validate_bundle_layout
from .download import (DEFAULT_MODELS_DIR, FLAVORS, RELEASE, REPO_ID, bundle_dir, choose_flavors, default_installed_flavor, download_bundle,
                       supported_flavors)
from .planner import parse_group_lines, records_from_text
from .runtime import SUPPORTED_BACKENDS, SUPPORTED_SAMPLERS, ScyllasBandRuntime
from .text_normalizer import normalize_spoken_text


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog or "scyllasband", allow_abbrev=False,
                                     description="Scylla's Band text-to-speech")
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download", help="download the model from Hugging Face", allow_abbrev=False)
    download.add_argument("--flavor", choices=[*FLAVORS, "all"],
                          help="bundle to download (default: Core ML on Apple silicon Macs, which are offered Core AI too when they "
                               "can build iOS/iPadOS/visionOS 27 apps; LiteRT elsewhere)")
    download.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    download.add_argument("--repo-id", default=REPO_ID)
    download.add_argument("--revision", default=RELEASE)
    download.add_argument("--token")
    download.add_argument("--force", action="store_true")
    download.add_argument("-y", "--yes", action="store_true", help="download the default bundle without asking")
    download.add_argument("--no-validate-bundle", action="store_true")
    download.set_defaults(func=_download)

    validate = sub.add_parser("validate-bundle", allow_abbrev=False)
    validate.add_argument("bundle", nargs="?", type=Path)
    validate.set_defaults(func=_validate)

    voices = sub.add_parser("list-voices", allow_abbrev=False)
    voices.add_argument("bundle", nargs="?", type=Path)
    voices.set_defaults(func=_list_voices)

    normalize = sub.add_parser("normalize-text", allow_abbrev=False)
    _add_text_args(normalize)
    normalize.add_argument("--language", default="en_us")
    normalize.set_defaults(func=_normalize)

    for name, func, help_text in (("speak", _speak, "speak text into a WAV file"),
                                  ("group-speak", _group_speak, "speak [voice:language:delivery] tagged dialogue"),
                                  ("stream", _stream, "speak passage by passage, printing JSON events"),
                                  ("plan", _plan, "print the sentence plan as JSON")):
        command = sub.add_parser(name, help=help_text, allow_abbrev=False)
        _add_synthesis_args(command, group=name == "group-speak")
        command.set_defaults(func=func)
    return parser


def _add_text_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("text_arg", nargs="?", metavar="TEXT")
    parser.add_argument("-t", "--text")
    parser.add_argument("-f", "--file", type=Path)


def _add_synthesis_args(parser: argparse.ArgumentParser, *, group: bool) -> None:
    _add_text_args(parser)
    parser.add_argument("-o", "--output", type=Path, help="WAV path (speak/group-speak), JSON path (plan), chunk WAV dir (stream)")
    parser.add_argument("--metadata", type=Path, help="write synthesis metadata JSON")
    parser.add_argument("--events", type=Path, help="stream: write JSON events here instead of stdout")
    parser.add_argument("--bundle", type=Path, help="model bundle directory (default: the installed bundle for --backend)")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--backend", choices=SUPPORTED_BACKENDS)
    parser.add_argument("--threads", type=int, help="CPU threads per graph")
    parser.add_argument("--compute-units", help="Core ML / Core AI: auto (default: the bundle's recommendation), gpu, cpu, ane "
                                                "(the graphs the bundle marks for the Neural Engine, the rest on the CPU), or "
                                                "per graph, e.g. gpu,g2p=cpu")
    parser.add_argument("--voice", default=None if group else "scylla")
    parser.add_argument("--language")
    parser.add_argument("--delivery", help="e.g. energy=2.3,tension=2,valence=2.2,assertiveness=2; neutral; auto")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--sampler", choices=SUPPORTED_SAMPLERS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--pause-ms", type=float, default=0.0, help="extra silence when the voice, language or delivery changes")
    parser.add_argument("--no-normalize-text", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-validate-bundle", action="store_true")


# --- commands --------------------------------------------------------------------------------------------------------
def _download(args: argparse.Namespace) -> int:
    if args.flavor == "all":
        flavors = [flavor for flavor in FLAVORS if flavor in supported_flavors()]
    else:
        flavors = [args.flavor] if args.flavor else choose_flavors(interactive=not args.yes and sys.stdin.isatty())
    for flavor in flavors:
        path = download_bundle(flavor=flavor, models_dir=args.models_dir, repo_id=args.repo_id, revision=args.revision,
                               token=args.token, force=args.force, validate=not args.no_validate_bundle)
        print(f"Installed {flavor} bundle: {path}")
    return 0


def _validate(args: argparse.Namespace) -> int:
    print(json.dumps(validate_bundle_layout(args.bundle or _default_bundle(args)), indent=2))
    return 0


def _list_voices(args: argparse.Namespace) -> int:
    runtime = ScyllasBandRuntime.from_bundle(args.bundle or _default_bundle(args))
    for voice in runtime.available_voices():
        spec = runtime.voice_for_id(voice)
        print(f"{voice}\t{spec.default_language}\t{','.join(spec.languages)}")
    return 0


def _normalize(args: argparse.Namespace) -> int:
    print(normalize_spoken_text(_read_text(args), language=args.language))
    return 0


def _speak(args: argparse.Namespace) -> int:
    runtime = _runtime(args)
    records = records_from_text(_read_text(args), voice=args.voice, language=args.language, delivery=args.delivery)
    result = runtime.render_records(records, normalize=not args.no_normalize_text, progress=_progress(args), **_options(args))
    _write_outputs(args, result.audio, result.sample_rate, result.metadata)
    return 0


def _group_speak(args: argparse.Namespace) -> int:
    runtime = _runtime(args)
    records = parse_group_lines(_read_text(args), default_voice=args.voice, default_language=args.language,
                                default_delivery=args.delivery)
    if not records:
        raise ValueError("No speakable group-speak lines found")
    result = runtime.render_records(records, normalize=not args.no_normalize_text, progress=_progress(args), **_options(args))
    _write_outputs(args, result.audio, result.sample_rate, result.metadata)
    return 0


def _stream(args: argparse.Namespace) -> int:
    runtime = _runtime(args)
    records = records_from_text(_read_text(args), voice=args.voice, language=args.language, delivery=args.delivery)
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
    sink = args.events.open("w", encoding="utf-8") if args.events else sys.stdout
    try:
        for event in runtime.synthesize_stream(records, normalize=not args.no_normalize_text, progress=_progress(args), **_options(args)):
            payload = event.to_dict()
            if event.type == "audio_chunk" and args.output is not None and event.chunk is not None:
                path = args.output / f"{event.chunk.chunk_id}.wav"
                write_wav(path, event.audio, int(event.sample_rate))
                payload["metadata"] = dict(payload["metadata"], chunk_wav=str(path))
            print(json.dumps(payload, ensure_ascii=False, default=_json_default), file=sink, flush=True)
    finally:
        if sink is not sys.stdout:
            sink.close()
    return 0


def _plan(args: argparse.Namespace) -> int:
    runtime = _runtime(args)
    plan = runtime.plan_records(records_from_text(_read_text(args), voice=args.voice, language=args.language, delivery=args.delivery),
                                normalize=not args.no_normalize_text)
    text = json.dumps(plan.to_dict(), indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


# --- helpers ---------------------------------------------------------------------------------------------------------
def _default_bundle(args: argparse.Namespace) -> Path:
    """The --backend's bundle, else the best installed bundle this machine can run."""
    models_dir = getattr(args, "models_dir", DEFAULT_MODELS_DIR)
    flavor = getattr(args, "backend", None) or default_installed_flavor(models_dir)
    if flavor is None:
        raise FileNotFoundError(f"No model installed under {Path(models_dir).resolve()}. Run: python -m scyllasband download")
    path = bundle_dir(models_dir, flavor)
    if not (path / "manifest.json").is_file():
        raise FileNotFoundError(f"No {flavor} model at {path}. Run: python -m scyllasband download --flavor {flavor}")
    return path


def _runtime(args: argparse.Namespace) -> ScyllasBandRuntime:
    return ScyllasBandRuntime.from_bundle(args.bundle or _default_bundle(args), backend=args.backend, threads=args.threads,
                                          compute_units=args.compute_units, validate=not args.no_validate_bundle)


def _options(args: argparse.Namespace) -> dict:
    return dict(steps=args.steps, sampler=args.sampler, speed=args.speed, seed=args.seed, temperature=args.temperature,
                pause_ms=args.pause_ms)


def _progress(args: argparse.Namespace):
    return None if args.no_progress else sys.stderr


def _read_text(args: argparse.Namespace) -> str:
    if args.file is not None:
        return args.file.read_text(encoding="utf-8")
    text = args.text if args.text is not None else args.text_arg
    if not text or not text.strip():
        raise ValueError("Pass text as an argument, with --text, or with --file")
    return text


def _write_outputs(args: argparse.Namespace, audio: np.ndarray, sample_rate: int, metadata: dict) -> None:
    if args.output is None:
        raise ValueError("Pass -o/--output for the WAV file")
    write_wav(args.output, audio, sample_rate)
    if args.metadata:
        args.metadata.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(np.asarray(audio, np.float32), -1.0, 1.0) * 32767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _run(args: argparse.Namespace) -> int:
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    return _run(build_parser(prog).parse_args(argv))


def speak_main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    return main(["speak", *(sys.argv[1:] if argv is None else argv)], prog=prog)


def group_main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    return main(["group-speak", *(sys.argv[1:] if argv is None else argv)], prog=prog)
