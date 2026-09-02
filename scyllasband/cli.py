from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import time
import struct
import sys
import wave

from .contract import (
    BundleValidationError,
    SUPPORTED_AFFECT_AXES,
    bundle_runtime_acceleration_report,
    validate_bundle_layout,
)
from .download import (
    BUNDLE_SUBDIR_GROUPS,
    DEFAULT_BUNDLE_SUBDIR,
    DEFAULT_INFERENCE_REPO_ID,
    DEFAULT_MODEL_VERSION,
    DEFAULT_MODELS_DIR,
    DEFAULT_ONNX_INT8_BUNDLE_SUBDIR,
    MODEL_VERSION_REPO_IDS,
    SUPPORTED_BUNDLE_SUBDIRS,
    SUPPORTED_MODEL_VERSIONS,
    bundle_dirs_for_subdirs,
    default_bundle_subdirs,
    delete_model_release,
    download_base_resources,
    model_release_installed,
    normalize_model_version,
)
from .g2p_phrases import DEFAULT_G2P_PHRASE_MAX_CHARS
from .metadata_compare import DEFAULT_CHUNK_FIELDS, compare_metadata_files
from .runtime import ScyllasBandRuntime, SUPPORTED_BACKENDS, SUPPORTED_SAMPLERS, SynthesisRequest
from .text_normalizer import normalize_spoken_text


DEFAULT_LONG_FORM_CHUNK_MAX_CHARS = 220
DEFAULT_LONG_FORM_CHUNK_MIN_CHARS = 48
DEFAULT_LONG_FORM_BOUNDARY_FADE_MS = 8.0
DEFAULT_MIN_SENTENCE_PUNCTUATION_PAUSE_MS = 107.0
DEFAULT_MIN_CLAUSE_PUNCTUATION_PAUSE_MS = 160.0
DEFAULT_ADAPTIVE_CHUNK_SCHEDULE = "120,160,220"
DEFAULT_ADAPTIVE_CHUNK_MIN_CHARS = 32
DEFAULT_ADAPTIVE_BUFFER_MS = 6000
DEFAULT_ADAPTIVE_REALTIME_FACTOR = 1.15
DEFAULT_ADAPTIVE_MIN_CHUNKS_PER_STAGE = 2
DEFAULT_LOOKAHEAD_CHUNKS = 1
_GROUP_TAG_RE = re.compile(r"\[([-A-Za-z0-9_.,:=]+)\]")
_GROUP_LANGUAGE_TAGS = frozenset({"en", "en_us", "en_gb", "es", "it", "de", "fr", "vi"})
_GROUP_EMOTION_PRESET_TAGS = frozenset(SUPPORTED_AFFECT_AXES) | frozenset(
    {
        "neutral",
        "friendly",
        "happy",
        "excited",
        "angry",
        "frustrated",
        "sad",
        "sarcastic",
        "unsure",
    }
)


def _default_bundle_path(backend: str | None = None) -> Path:
    backend = str(backend or "auto").strip().lower()
    if backend == "auto":
        # Match the runtime's auto policy: Core AI first on macOS 27+ hosts,
        # then int8 ONNX, then the fp32 ONNX bundle if that is all that is
        # available locally.
        subdir_preference = (*default_bundle_subdirs(), DEFAULT_BUNDLE_SUBDIR)
    elif backend == "onnx":
        # The int8 bundle is the default ONNX artifact; fp32 is opt-in via an
        # explicit bundle path.
        subdir_preference = (DEFAULT_ONNX_INT8_BUNDLE_SUBDIR, DEFAULT_BUNDLE_SUBDIR)
    else:
        subdir_preference = (backend,)
    roots = (
        DEFAULT_MODELS_DIR,
        Path(__file__).resolve().parent / "models",
        Path(__file__).resolve().parents[1] / "models",
    )
    for bundle_subdir in subdir_preference:
        for root in roots:
            for release in ("v2", "v1", None):
                bundle = root / release / bundle_subdir if release else root / bundle_subdir
                if (bundle / "manifest.json").is_file():
                    return bundle
    return (
        Path(__file__).resolve().parents[1] / "models" / DEFAULT_MODEL_VERSION / subdir_preference[-1]
    )


def _looks_like_bundle_path(value: object) -> bool:
    if value is None:
        return False
    path = Path(str(value))
    return (path / "manifest.json").is_file() or path.is_dir()


def _parse_onnx_providers(value: str | None) -> list[str] | None:
    if value is None or str(value).strip() == "" or str(value).strip().lower() == "auto":
        return None
    providers = [item.strip() for item in str(value).split(",") if item.strip()]
    return providers or None


def _parse_onnx_thread_candidates(value: str | None) -> list[int] | None:
    if value is None or not str(value).strip():
        return None
    candidates: list[int] = []
    for raw in str(value).split(","):
        raw = raw.strip()
        if not raw:
            continue
        candidate = int(raw)
        if candidate < 0:
            raise ValueError("ONNX thread candidates must be >= 0")
        if candidate not in candidates:
            candidates.append(candidate)
    if not candidates:
        raise ValueError("ONNX thread autotuning requires at least one candidate")
    return candidates


def _runtime_from_args(args: argparse.Namespace) -> ScyllasBandRuntime:
    return ScyllasBandRuntime.from_bundle(
        args.bundle,
        backends=[args.backend],
        validate_files=not args.no_validate_bundle,
        litert_accelerator=args.litert_accelerator,
        onnx_providers=_parse_onnx_providers(getattr(args, "onnx_providers", None)),
        onnx_intra_op_num_threads=int(getattr(args, "onnx_intra_op_threads", 0) or 0),
        onnx_inter_op_num_threads=int(getattr(args, "onnx_inter_op_threads", 0) or 0),
        onnx_autotune_threads=bool(getattr(args, "onnx_autotune_threads", False)),
        onnx_autotune_candidates=_parse_onnx_thread_candidates(
            getattr(args, "onnx_autotune_candidates", None)
        ),
        onnx_thread_cache_path=getattr(args, "onnx_thread_cache", None),
    )


def _validate_bundle(args: argparse.Namespace) -> int:
    try:
        manifest = validate_bundle_layout(args.bundle)
    except BundleValidationError as exc:
        print(f"invalid: {exc}")
        return 1
    print(
        json.dumps(
            {
                "status": "ok",
                "model_name": manifest.model_name,
                "model_version": manifest.model_version,
                "contract_version": manifest.contract_version,
                "voices": len(manifest.voices),
                "languages": list(manifest.languages),
                "preferred_backends": list(manifest.preferred_backends),
                "runtime_acceleration": bundle_runtime_acceleration_report(
                    args.bundle,
                    manifest,
                ),
            },
            indent=2,
        )
    )
    return 0


def _list_voices(args: argparse.Namespace) -> int:
    manifest = validate_bundle_layout(args.bundle)
    for voice in manifest.voices:
        print(f"{voice.id}\t{','.join(voice.languages)}\t{','.join(voice.styles)}")
    return 0


def _normalize_text(args: argparse.Namespace) -> int:
    text = _read_text(args)
    if not text:
        raise ValueError("normalize-text requires text or --file")
    print(normalize_spoken_text(text, language=args.language))
    return 0


def _prompt_model_version(default_version: str = DEFAULT_MODEL_VERSION) -> str | None:
    """Choose one mutually exclusive public model release before bundle types."""

    default_version = normalize_model_version(default_version)
    entries = list(SUPPORTED_MODEL_VERSIONS)
    while True:
        print("\nScylla's Band model download — select release:\n")
        for index, version in enumerate(entries, start=1):
            mark = "x" if version == default_version else " "
            repo_id = MODEL_VERSION_REPO_IDS[version]
            print(f"  [{mark}] {index}. {version:2s}  {repo_id}")
        try:
            answer = input("Choose 1 or 2, Enter for v2, q to quit: ").strip().lower()
        except EOFError:
            return default_version
        if answer in {"", "d", "default"}:
            return default_version
        if answer in {"q", "quit", "exit"}:
            return None
        for index, version in enumerate(entries, start=1):
            if answer in {str(index), version, version.removeprefix("v")}:
                return version
        print("Choose v1 or v2, press Enter for v2, or q to quit.")


def _prompt_bundle_selection(default_subdirs: tuple[str, ...]) -> tuple[str, ...] | None:
    """Checkbox-style bundle picker; returns the selection or None if cancelled."""

    from .download import BUNDLE_SUBDIR_DESCRIPTIONS

    entries = list(BUNDLE_SUBDIR_DESCRIPTIONS)
    selected = {name for name in default_subdirs if name in entries}
    while True:
        print("\nScylla's Band model download — select bundles:\n")
        for index, name in enumerate(entries, start=1):
            description, size = BUNDLE_SUBDIR_DESCRIPTIONS[name]
            mark = "x" if name in selected else " "
            default_tag = "  (default)" if name in default_subdirs else ""
            size_tag = f"  {size}" if size else ""
            print(f"  [{mark}] {index}. {name:12s} {description}{size_tag}{default_tag}")
        print("\nVoice packs are always included.")
        try:
            answer = input(
                "Toggle with numbers (e.g. \"2\" or \"2 4\"), Enter to download, q to quit: "
            ).strip().lower()
        except EOFError:
            return tuple(sorted(selected)) if selected else None
        if answer in {"q", "quit", "exit"}:
            return None
        if answer in {"", "d", "download"}:
            if not selected:
                print("Nothing selected; toggle at least one bundle or press q to quit.")
                continue
            return tuple(name for name in entries if name in selected)
        toggled_any = False
        for token in answer.replace(",", " ").split():
            if token.isdigit() and 1 <= int(token) <= len(entries):
                name = entries[int(token) - 1]
                selected.symmetric_difference_update({name})
                toggled_any = True
            elif token in entries:
                selected.symmetric_difference_update({token})
                toggled_any = True
        if not toggled_any:
            print("Enter bundle numbers to toggle, Enter to start, or q to quit.")


def _prompt_delete_v1_checkbox() -> bool:
    """Optional destructive checkbox shown only after v2 and bundle selection."""

    selected = False
    while True:
        mark = "x" if selected else " "
        print("\nOptional cleanup after the validated v2 download:\n")
        print(f"  [{mark}] 1. Delete locally installed v1 models and v1 voice packs")
        try:
            answer = input("Toggle with 1, Enter to continue, q to keep v1: ").strip().lower()
        except EOFError:
            return False
        if answer in {"", "d", "continue"}:
            return selected
        if answer in {"q", "quit", "keep"}:
            return False
        if answer in {"1", "v1", "delete", "remove"}:
            selected = not selected
        else:
            print("Enter 1 to toggle v1 deletion, Enter to continue, or q to keep v1.")


def _download(args: argparse.Namespace) -> int:
    interactive = (
        not bool(getattr(args, "yes", False))
        and sys.stdin.isatty()
        and sys.stdout.isatty()
    )
    requested_version = getattr(args, "model_version", None)
    if interactive and requested_version is None:
        chosen_version = _prompt_model_version()
        if chosen_version is None:
            print("Download cancelled.")
            return 1
        requested_version = chosen_version
    model_version = normalize_model_version(requested_version or DEFAULT_MODEL_VERSION)
    delete_v1 = bool(getattr(args, "delete_v1", False))
    if delete_v1 and model_version != "v2":
        raise ValueError("--delete-v1 is valid only when downloading --model-version v2")

    bundle_subdirs = _download_bundle_subdirs(args)
    explicit_request = bool(
        str(getattr(args, "bundle_subdir", "") or "").strip()
        or (getattr(args, "runtime_bundles", None) or "default") != "default"
    )
    if not explicit_request and interactive:
        chosen = _prompt_bundle_selection(bundle_subdirs)
        if chosen is None:
            print("Download cancelled.")
            return 1
        bundle_subdirs = _normalize_requested_bundle_subdirs(chosen)
    if (
        model_version == "v2"
        and not delete_v1
        and interactive
        and model_release_installed(args.models_dir, "v1")
    ):
        delete_v1 = _prompt_delete_v1_checkbox()

    repo_id = str(args.repo_id or MODEL_VERSION_REPO_IDS[model_version])
    bundle_dir, voices_dir = download_base_resources(
        models_dir=args.models_dir,
        repo_id=repo_id,
        model_version=model_version,
        bundle_subdirs=bundle_subdirs,
        token=args.token,
        revision=args.revision,
        force=bool(args.force),
        validate=not bool(args.no_validate_bundle),
        include_voices=not bool(args.no_voices),
    )
    bundle_dirs = bundle_dirs_for_subdirs(
        args.models_dir,
        bundle_subdirs,
        model_version=model_version,
    )
    deleted_v1_paths: tuple[Path, ...] = ()
    if delete_v1:
        deleted_v1_paths = delete_model_release(args.models_dir, "v1")
    print(
        json.dumps(
            {
                "model_version": model_version,
                "repo_id": repo_id,
                "runtime_bundles": list(bundle_subdirs),
                "bundle_dir": str(bundle_dir),
                "bundle_dirs": {key: str(value) for key, value in bundle_dirs.items()},
                "voices_dir": str(voices_dir) if voices_dir is not None else None,
                "deleted_v1_paths": [str(path) for path in deleted_v1_paths],
            },
            indent=2,
        )
    )
    return 0


def _download_bundle_subdirs(args: argparse.Namespace) -> tuple[str, ...]:
    override = str(getattr(args, "bundle_subdir", "") or "").strip()
    if override:
        return _normalize_requested_bundle_subdirs((override,))
    return _normalize_requested_bundle_subdirs((getattr(args, "runtime_bundles", None) or "default",))


def _normalize_requested_bundle_subdirs(values: tuple[str, ...]) -> tuple[str, ...]:
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
            if candidate not in (*SUPPORTED_BUNDLE_SUBDIRS,):
                options = ", ".join(
                    (*SUPPORTED_BUNDLE_SUBDIRS, *BUNDLE_SUBDIR_GROUPS)
                )
                raise ValueError(f"Unsupported runtime bundle {candidate!r}; expected one of: {options}")
            if candidate not in output:
                output.append(candidate)
    return tuple(output or default_bundle_subdirs())


def _compare_metadata(args: argparse.Namespace) -> int:
    ignored_chunk_fields = {str(item) for item in (args.ignore_chunk_field or [])}
    chunk_fields = tuple(field for field in DEFAULT_CHUNK_FIELDS if field not in ignored_chunk_fields)
    report = compare_metadata_files(
        args.left,
        args.right,
        left_label=args.left_label,
        right_label=args.right_label,
        chunk_fields=chunk_fields,
        max_mismatches=args.max_mismatches,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "ok" else 1


def _plan(args: argparse.Namespace) -> int:
    _normalize_optional_bundle_and_text(args)
    text = _read_text(args)
    if not text:
        raise ValueError("plan requires text or --file")
    records, source_kind = _records_from_text_args(args, text)
    runtime = _runtime_from_args(args)
    plan = runtime.plan_records(records, options=args, source_kind=source_kind)
    _write_json(args.output, plan.to_dict())
    return 0


def _stream(args: argparse.Namespace) -> int:
    _normalize_optional_bundle_and_text(args)
    text = _read_text(args)
    if not text:
        raise ValueError("stream requires text or --file")
    records, source_kind = _records_from_text_args(args, text)
    runtime = _runtime_from_args(args)
    output_dir = getattr(args, "output", None)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    event_handle = args.events.open("w", encoding="utf-8") if args.events else None
    try:
        sink = event_handle or sys.stdout
        for event in runtime.synthesize_stream(
            records,
            options=args,
            source_kind=source_kind,
            progress_stream=sys.stderr,
        ):
            payload = event.to_dict(include_audio=False)
            if event.type == "audio_chunk" and output_dir is not None and event.chunk is not None:
                chunk_wav = output_dir / f"{event.chunk.chunk_id}.wav"
                _write_wav(chunk_wav, event.audio or [], int(event.sample_rate or 0))
                metadata = dict(payload.get("metadata") or {})
                metadata["chunk_wav"] = str(chunk_wav)
                payload["metadata"] = metadata
            print(json.dumps(payload, ensure_ascii=False), file=sink, flush=True)
    finally:
        if event_handle is not None:
            event_handle.close()
    return 0


def _speak(args: argparse.Namespace) -> int:
    _normalize_optional_bundle_and_text(args)
    text = _read_text(args)
    explicit_phones = _parse_phones(args.phones)
    if not text and not explicit_phones:
        raise ValueError("Pass text as an argument, --text, --file, or --phones")
    runtime = _runtime_from_args(args)
    if args.stream:
        if explicit_phones:
            raise ValueError("--stream is only supported for text input, not --phones")
        rendered, sample_rate, metadata = _render_streaming_speak(
            runtime,
            [
                {
                    "voice": args.voice,
                    "language": args.language,
                    "emotion": args.emotion,
                    "affect": args.affect,
                    "affect_guidance_scale": args.affect_guidance_scale,
                    "emotion_guidance": args.emotion_guidance,
                    "text": text,
                }
            ],
            args,
        )
        _write_wav(args.output, rendered, sample_rate)
        if args.metadata:
            args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return 0
    if args.chunk_text and text and not explicit_phones:
        rendered, sample_rate, metadata = _render_text_chunks(
            runtime,
            [
                {
                    "voice": args.voice,
                    "language": args.language,
                    "emotion": args.emotion,
                    "affect": args.affect,
                    "affect_guidance_scale": args.affect_guidance_scale,
                    "emotion_guidance": args.emotion_guidance,
                    "text": text,
                }
            ],
            args,
        )
        _write_wav(args.output, rendered, sample_rate)
        if args.metadata:
            args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return 0

    result = runtime.synthesize(
        SynthesisRequest(
            text=text,
            explicit_phones=explicit_phones,
            voice_id=args.voice,
            language=args.language,
            emotion=args.emotion,
            affect=args.affect,
            affect_guidance_scale=float(args.affect_guidance_scale),
            emotion_guidance=args.emotion_guidance,
            guidance_null_reference=not bool(args.guidance_keep_reference),
            emotion_embed_scale=float(args.emotion_embed_scale),
            normalize_text=not args.no_normalize_text,
            seed=args.seed,
            steps=args.steps,
            sampler=args.sampler,
            speed=args.speed,
            duration_hierarchy_mode=args.duration_hierarchy_mode,
            min_sentence_pause_ms=float(getattr(args, "min_sentence_pause_ms", 0.0)),
            min_clause_pause_ms=float(getattr(args, "min_clause_pause_ms", 0.0)),
        )
    )
    _write_wav(args.output, result.audio, result.sample_rate)
    if args.metadata:
        args.metadata.write_text(json.dumps(result.metadata, indent=2) + "\n", encoding="utf-8")
    return 0


def _group_speak(args: argparse.Namespace) -> int:
    _normalize_optional_bundle_and_text(args)
    text = _read_text(args)
    if not text:
        raise ValueError("group-speak requires text or --file")
    chunks = _parse_group_lines(
        text,
        default_voice=args.voice,
        default_language=args.language,
        default_emotion=args.emotion,
        default_affect=args.affect,
        default_emotion_guidance=args.emotion_guidance,
    )
    if not chunks:
        raise ValueError("No speakable group-speak lines found")
    runtime = _runtime_from_args(args)

    rendered, sample_rate, metadata = _render_text_chunks(runtime, chunks, args)
    _write_wav(args.output, rendered, sample_rate)
    if args.metadata:
        args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return 0


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog or "scyllasband", allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-bundle", allow_abbrev=False)
    validate_parser.add_argument("bundle", nargs="?", type=Path, default=_default_bundle_path())
    validate_parser.set_defaults(func=_validate_bundle)

    voices_parser = subparsers.add_parser("list-voices", allow_abbrev=False)
    voices_parser.add_argument("bundle", nargs="?", type=Path, default=_default_bundle_path())
    voices_parser.set_defaults(func=_list_voices)

    normalize_parser = subparsers.add_parser("normalize-text", allow_abbrev=False)
    normalize_parser.add_argument("text_arg", nargs="?", help="Input text")
    normalize_parser.add_argument("-t", "--text", dest="text_flag", help="Input text")
    normalize_parser.add_argument("-f", "--file", type=Path, help="Input text file")
    normalize_parser.add_argument("--language", default="en", help="Language code")
    normalize_parser.set_defaults(func=_normalize_text)

    download_parser = subparsers.add_parser("download", allow_abbrev=False)
    download_parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    download_parser.add_argument(
        "--model-version",
        choices=SUPPORTED_MODEL_VERSIONS,
        default=None,
        help="Public model release; defaults to v2 (interactive downloads ask first)",
    )
    download_parser.add_argument(
        "--repo-id",
        default=None,
        help=f"Expert repository override; defaults by release (v2: {DEFAULT_INFERENCE_REPO_ID})",
    )
    download_parser.add_argument(
        "--runtime-bundles",
        choices=(*SUPPORTED_BUNDLE_SUBDIRS, *BUNDLE_SUBDIR_GROUPS, "default"),
        default="default",
        help=(
            "Runtime bundle(s) to download; default fetches onnx-int8 plus "
            "Core AI on macOS 27+ hosts and onnx-int8 alone elsewhere. The "
            "fp32 onnx bundle, litert, and other options must be requested "
            "explicitly; all downloads every option"
        ),
    )
    download_parser.add_argument("--bundle-subdir", default=None, help="Legacy/expert override for the exact HF bundle subdir to download")
    download_parser.add_argument("--token", help="Hugging Face token; defaults to the hub client configuration")
    download_parser.add_argument("--revision", help="Optional Hugging Face revision")
    download_parser.add_argument("--force", action="store_true", help="Replace existing local bundle/voice directories")
    download_parser.add_argument(
        "--delete-v1",
        action="store_true",
        help="After a validated v2 download, delete locally installed v1 bundles and voice packs",
    )
    download_parser.add_argument("-y", "--yes", action="store_true", help="Skip interactive release/bundle selection and download v2 platform defaults")
    download_parser.add_argument("--no-voices", action="store_true", help="Only download the runtime bundle")
    download_parser.add_argument("--no-validate-bundle", action="store_true", help="Skip bundle layout validation after download")
    download_parser.set_defaults(func=_download)

    compare_parser = subparsers.add_parser("compare-metadata", allow_abbrev=False)
    compare_parser.add_argument("left", type=Path, help="Left long-form metadata JSON")
    compare_parser.add_argument("right", type=Path, help="Right long-form metadata JSON")
    compare_parser.add_argument("--left-label", default="left", help="Label for the left metadata source")
    compare_parser.add_argument("--right-label", default="right", help="Label for the right metadata source")
    compare_parser.add_argument(
        "--ignore-chunk-field",
        action="append",
        default=[],
        help="Chunk metadata field to ignore during comparison; may be repeated",
    )
    compare_parser.add_argument("--max-mismatches", type=int, default=100, help="Maximum mismatch details to print; total count is still reported")
    compare_parser.set_defaults(func=_compare_metadata)

    plan_parser = subparsers.add_parser("plan", allow_abbrev=False)
    _add_synthesis_args(
        plan_parser,
        output_default=None,
        output_help="Output plan JSON path; defaults to stdout",
        include_metadata=False,
        require_voice=False,
    )
    plan_parser.add_argument("text_arg", nargs="?", help="Input text")
    plan_parser.add_argument("--group", action="store_true", help="Parse input as group-speak records")
    plan_parser.set_defaults(func=_plan)

    stream_parser = subparsers.add_parser("stream", allow_abbrev=False)
    _add_synthesis_args(
        stream_parser,
        output_default=None,
        output_help="Directory for per-chunk WAV files; omit to emit JSONL events only",
        include_metadata=False,
        require_voice=False,
    )
    stream_parser.add_argument("text_arg", nargs="?", help="Input text")
    stream_parser.add_argument("--group", action="store_true", help="Parse input as group-speak records")
    stream_parser.add_argument("--events", type=Path, help="Optional newline JSON event output path; defaults to stdout")
    stream_parser.set_defaults(func=_stream)

    speak_parser = subparsers.add_parser("speak", allow_abbrev=False)
    _add_synthesis_args(speak_parser, include_stream_mode=True)
    speak_parser.add_argument("text_arg", nargs="?", help="Input text")
    speak_parser.add_argument("--phones", help="Explicit whitespace-separated Scylla's Band phone sequence")
    speak_parser.set_defaults(func=_speak)

    group_parser = subparsers.add_parser("group-speak", allow_abbrev=False)
    _add_synthesis_args(group_parser, require_voice=False)
    group_parser.add_argument("text_arg", nargs="?", help="Input text with optional [voice] labels")
    group_parser.set_defaults(func=_group_speak)

    return parser


def speak_main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=prog or "scyllasband-speak",
        description="Synthesize speech with Scylla's Band.",
        allow_abbrev=False,
    )
    _add_synthesis_args(parser, include_stream_mode=True)
    parser.add_argument("text_arg", nargs="?", help="Input text")
    parser.add_argument("--phones", help="Explicit whitespace-separated Scylla's Band phone sequence")
    return _run_with_errors(_speak, _parse_cli_args(parser, argv))


def group_main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=prog or "scyllasband-group-speak",
        description="Synthesize multi-voice speech with Scylla's Band.",
        allow_abbrev=False,
    )
    _add_synthesis_args(parser, require_voice=False)
    parser.add_argument("text_arg", nargs="?", help="Input text with optional [voice] labels")
    return _run_with_errors(_group_speak, _parse_cli_args(parser, argv))


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    parser = build_parser(prog=prog)
    args = _parse_cli_args(parser, argv)
    return _run_with_errors(args.func, args)


def _parse_cli_args(parser: argparse.ArgumentParser, argv: list[str] | None) -> argparse.Namespace:
    args, extra = parser.parse_known_args(argv)
    if not extra:
        _apply_synthesis_shortcuts(args)
        return args
    if any(item.startswith("-") for item in extra):
        parser.error("unrecognized arguments: " + " ".join(extra))
    if hasattr(args, "text_arg"):
        existing = getattr(args, "text_arg", None)
        args.text_arg = " ".join([str(existing), *extra] if existing else extra)
        _apply_synthesis_shortcuts(args)
        return args
    parser.error("unrecognized arguments: " + " ".join(extra))


def _apply_synthesis_shortcuts(args: argparse.Namespace) -> None:
    # The public CLI uses emotion terminology. Runtime/bundle structures keep
    # their existing affect field names so saved metadata and ABI contracts do
    # not change as part of this command-line cleanup.
    if hasattr(args, "emotion_spec"):
        args.affect = args.emotion_spec
    if hasattr(args, "emotion_scale"):
        args.affect_guidance_scale = float(args.emotion_scale)
    if bool(getattr(args, "faster", False)):
        args.adaptive_chunking = True
        args.steps = 2
        args.max_chunk_chars = 180
        return
    if bool(getattr(args, "fast", False)):
        args.adaptive_chunking = True
        args.steps = 4


def _run_with_errors(func: object, args: argparse.Namespace) -> int:
    try:
        return int(func(args))  # type: ignore[misc]
    except (BundleValidationError, ValueError, NotImplementedError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _add_synthesis_args(
    parser: argparse.ArgumentParser,
    *,
    output_default: Path | None = Path("output.wav"),
    output_help: str = "Output WAV path",
    include_metadata: bool = True,
    include_stream_mode: bool = False,
    require_voice: bool = True,
) -> None:
    parser.add_argument(
        "bundle",
        nargs="?",
        type=Path,
        default=None,
        help="Scylla's Band bundle directory; defaults to the requested backend bundle under scyllasband/models",
    )
    parser.add_argument("-t", "--text", dest="text_flag", help="Input text")
    parser.add_argument("-f", "--file", type=Path, help="Input text file")
    parser.add_argument("-o", "--output", type=Path, default=output_default, help=output_help)
    if include_metadata:
        parser.add_argument("--metadata", type=Path, help="Optional JSON metadata output path")
    parser.add_argument("--voice", required=require_voice, help="Voice ID")
    parser.add_argument("--language", default=None, help="Language code")
    emotion_group = parser.add_mutually_exclusive_group()
    emotion_group.add_argument(
        "--emotion",
        dest="emotion_spec",
        metavar="EMOTION",
        help=(
            "Emotion preset or comma-separated axis values, for example "
            "joy or calm=0.5,joy=0.8,whisper=0.3"
        ),
    )
    emotion_group.add_argument(
        "--affect",
        dest="emotion_spec",
        help=argparse.SUPPRESS,
    )
    emotion_scale_group = parser.add_mutually_exclusive_group()
    emotion_scale_group.add_argument(
        "--emotion-scale",
        dest="emotion_scale",
        type=float,
        default=1.0,
        metavar="SCALE",
        help="Emotion CFG strength (1=direct conditioning, 0=learned null branch)",
    )
    emotion_scale_group.add_argument(
        "--affect-guidance-scale",
        dest="emotion_scale",
        type=float,
        help=argparse.SUPPRESS,
    )
    emotion_scale_group.add_argument(
        "--emotion-guidance-scale",
        dest="emotion_scale",
        type=float,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--emotion-guidance",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--guidance-keep-reference",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--emotion-embed-scale", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.set_defaults(
        emotion=None,
        affect=None,
        affect_guidance_scale=1.0,
        emotion_guidance=None,
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=SUPPORTED_BACKENDS,
        help="Runtime backend; auto prefers Core AI on macOS 27+ hosts with a Core AI bundle and resolves to onnx otherwise",
    )
    parser.add_argument("--litert-accelerator", default="auto", choices=("auto", "cpu", "gpu", "npu"), help="Native LiteRT accelerator or preferred Core AI compute unit; auto keeps LiteRT on CPU and Core AI on GPU")
    parser.add_argument("--onnx-providers", default=None, help="ONNX Runtime providers, comma-separated, e.g. CUDAExecutionProvider,CPUExecutionProvider; defaults to CPUExecutionProvider")
    parser.add_argument("--onnx-intra-op-threads", type=int, default=0, help="ONNX intra-op thread count; 0 uses ONNX Runtime or a cached Scylla's Band tuning result")
    parser.add_argument("--onnx-inter-op-threads", type=int, default=0, help="ONNX inter-op thread count; only relevant to parallel graph execution")
    parser.add_argument("--onnx-autotune-threads", action="store_true", help="Benchmark vector-estimator thread candidates on the first invocation and cache the fastest count; adds one-time startup cost")
    parser.add_argument("--onnx-autotune-candidates", default=None, help="Comma-separated intra-op thread candidates; 0 represents the ONNX Runtime default")
    parser.add_argument("--onnx-thread-cache", type=Path, default=None, help="Override the persistent ONNX thread-tuning cache path")
    parser.add_argument("--steps", type=int, default=8, help="Flow sampling steps")
    parser.add_argument("--sampler", default="heun", choices=SUPPORTED_SAMPLERS, help="Flow sampler")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Shortcut for speed mode: --adaptive-chunking --steps 4",
    )
    parser.add_argument(
        "--faster",
        action="store_true",
        help="Shortcut for faster speed mode: --adaptive-chunking --steps 2 --max-chunk-chars 180",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Speaking speed multiplier")
    parser.add_argument(
        "--duration-hierarchy-mode",
        choices=("default", "sampled", "p50"),
        default="default",
        help="Duration hierarchy policy: bundle default, coherent sampled pauses, or deterministic p50",
    )
    parser.add_argument("--seed", type=int, default=None, help="Deterministic seed")
    parser.add_argument(
        "--chunk-text",
        dest="chunk_text",
        action="store_true",
        default=True,
        help="Split text into synthesis-safe chunks; enabled by default",
    )
    parser.add_argument(
        "--no-chunk-text",
        dest="chunk_text",
        action="store_false",
        help="Synthesize text as one request for debugging",
    )
    parser.add_argument("--max-chunk-chars", type=int, default=DEFAULT_LONG_FORM_CHUNK_MAX_CHARS, help="Maximum characters per long-form synthesis chunk")
    parser.add_argument("--min-chunk-chars", type=int, default=DEFAULT_LONG_FORM_CHUNK_MIN_CHARS, help="Merge or rebalance adjacent chunks when possible to avoid very short chunks")
    parser.add_argument("--no-auto-split-overlong", action="store_true", help="Do not retry by splitting chunks that exceed the fixed LiteRT latent-frame budget")
    parser.add_argument("--no-preflight-chunks", action="store_true", help="Skip G2P+duration preflight splitting before synthesis")
    parser.add_argument("--pause-ms", type=int, default=0, help="Silence inserted after sentence or paragraph chunk boundaries")
    parser.add_argument("--continuation-pause-ms", type=int, default=0, help="Silence inserted after clause or artificial continuation chunks")
    parser.add_argument("--min-sentence-pause-ms", type=float, default=DEFAULT_MIN_SENTENCE_PUNCTUATION_PAUSE_MS, help="Minimum duration for nonterminal sentence-ending pauses inside a chunk")
    parser.add_argument("--min-clause-pause-ms", type=float, default=DEFAULT_MIN_CLAUSE_PUNCTUATION_PAUSE_MS, help="Minimum duration for nonterminal comma, dash, colon, and semicolon pauses inside a chunk")
    parser.add_argument("--boundary-fade-ms", type=float, default=DEFAULT_LONG_FORM_BOUNDARY_FADE_MS, help="Short fade/crossfade at chunk joins to suppress clicks")
    parser.add_argument("--adaptive-chunking", action="store_true", help="Start streaming with small chunks, then merge future chunks when measured buffer allows it")
    parser.add_argument("--adaptive-chunk-schedule", default=DEFAULT_ADAPTIVE_CHUNK_SCHEDULE, help="Comma-separated max-char stages for --adaptive-chunking, e.g. 120,160,220")
    parser.add_argument("--adaptive-min-chunk-chars", type=int, default=DEFAULT_ADAPTIVE_CHUNK_MIN_CHARS, help="Minimum chunk chars used by --adaptive-chunking stages")
    parser.add_argument("--adaptive-buffer-ms", type=int, default=DEFAULT_ADAPTIVE_BUFFER_MS, help="Buffered audio needed before advancing an adaptive chunk stage")
    parser.add_argument("--adaptive-realtime-factor", type=float, default=DEFAULT_ADAPTIVE_REALTIME_FACTOR, help="Minimum generated-audio-ms / generation-ms ratio before advancing an adaptive chunk stage")
    parser.add_argument("--adaptive-min-chunks-per-stage", type=int, default=DEFAULT_ADAPTIVE_MIN_CHUNKS_PER_STAGE, help="Minimum completed chunks before advancing an adaptive chunk stage")
    parser.add_argument("--adaptive-preflight-chunks", action="store_true", help="Keep full duration preflight enabled when using --adaptive-chunking")
    parser.add_argument(
        "--eager-preflight",
        dest="lazy_preflight",
        action="store_false",
        default=True,
        help="Run full-document G2P+duration preflight before synthesis instead of lazy streaming preflight",
    )
    parser.add_argument(
        "--no-pipeline-preflight",
        dest="pipeline_preflight",
        action="store_false",
        default=True,
        help="Disable background preflight of future chunks while streaming",
    )
    parser.add_argument("--lookahead-chunks", type=int, default=DEFAULT_LOOKAHEAD_CHUNKS, help="Future text chunks to expose to span-context conditioning; 0 disables future context")
    if include_stream_mode:
        parser.add_argument("--stream", action="store_true", help="Emit long-form chunk events while writing the final WAV")
        parser.add_argument("--events", type=Path, help="Optional newline JSON event output path for --stream")
        parser.add_argument("--chunk-output-dir", type=Path, help="Optional directory for per-chunk WAV files during --stream")
        parser.add_argument("--no-progress", action="store_true", help="Suppress streaming progress lines on stderr")
    parser.add_argument("--no-normalize-text", action="store_true", help="Pass raw text directly to G2P")
    parser.add_argument("--no-validate-bundle", action="store_true", help="Skip bundle file validation")


def _normalize_optional_bundle_and_text(args: argparse.Namespace) -> None:
    bundle = getattr(args, "bundle", None)
    backend = getattr(args, "backend", None)
    if bundle is None:
        args.bundle = _default_bundle_path(backend)
        return
    has_explicit_input = bool(getattr(args, "text_flag", None) or getattr(args, "file", None) or getattr(args, "phones", None))
    if has_explicit_input or getattr(args, "text_arg", None):
        return
    if bundle is not None and not _looks_like_bundle_path(bundle):
        args.text_arg = str(bundle)
        args.bundle = _default_bundle_path(backend)


def _read_text(args: argparse.Namespace) -> str | None:
    values = [value for value in (getattr(args, "text_flag", None), getattr(args, "text_arg", None)) if value]
    if len(values) > 1:
        raise ValueError("Pass text either positionally or with --text, not both")
    if values and getattr(args, "file", None):
        raise ValueError("Pass text or --file, not both")
    if values:
        return str(values[0])
    if getattr(args, "file", None):
        return args.file.read_text(encoding="utf-8")
    return None


def _records_from_text_args(
    args: argparse.Namespace,
    text: str,
) -> tuple[list[dict[str, str | None]], str]:
    if getattr(args, "group", False):
        records = _parse_group_lines(
            text,
            default_voice=args.voice,
            default_language=args.language,
            default_emotion=args.emotion,
            default_affect=args.affect,
            default_emotion_guidance=args.emotion_guidance,
        )
        if not records:
            raise ValueError("No speakable group records found")
        return records, "dialogue"
    if not args.voice:
        raise ValueError("--voice is required for plain text input; omit it only for --group input with [voice] labels")
    return [
        {
            "voice": args.voice,
            "language": args.language,
            "emotion": args.emotion,
            "affect": args.affect,
            "affect_guidance_scale": args.affect_guidance_scale,
            "emotion_guidance": args.emotion_guidance,
            "text": text,
        }
    ], "text"


def _write_json(path: Path | None, payload: object) -> None:
    data = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path is None:
        print(data, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")


def _parse_phones(value: str | None) -> tuple[str, ...] | None:
    if not value:
        return None
    phones = tuple(item.strip() for item in value.split() if item.strip())
    return phones or None


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_group_affect_spec(value: str) -> str:
    terms: list[str] = []
    seen: set[str] = set()
    for raw_term in str(value).split(","):
        term = raw_term.strip()
        axis, separator, raw_score = term.partition("=")
        axis = axis.strip().lower()
        if not separator or axis not in SUPPORTED_AFFECT_AXES:
            options = ", ".join(f"{name}=VALUE" for name in SUPPORTED_AFFECT_AXES)
            raise ValueError(
                f"Invalid group-speak emotion term {term!r}; expected one of {options}"
            )
        if axis in seen:
            raise ValueError(f"Duplicate group-speak emotion axis {axis!r}")
        try:
            score = float(raw_score)
        except ValueError as exc:
            raise ValueError(
                f"Invalid group-speak emotion value for {axis!r}: {raw_score!r}"
            ) from exc
        if not math.isfinite(score) or score < 0.0 or score > 1.0:
            raise ValueError(
                f"Group-speak emotion value for {axis!r} must remain within [0, 1]"
            )
        seen.add(axis)
        terms.append(f"{axis}={score:g}")
    if not terms:
        raise ValueError("Group-speak emotion vector must contain at least one axis")
    return ",".join(terms)


def _parse_group_tag(
    label: str,
    *,
    default_voice: str | None,
    default_language: str | None,
    default_emotion: str | None,
    default_affect: str | None,
    default_emotion_guidance: str | None,
) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    voice = default_voice
    language = default_language
    emotion = default_emotion
    affect = default_affect
    emotion_guidance = default_emotion_guidance
    clean = label.strip()
    if ":" in clean:
        voice_part, language_part, *rest = clean.split(":", 2)
        voice = voice_part.strip() or default_voice
        language = language_part.strip() or default_language
        if rest:
            emotion_part = rest[0].strip()
            if "=" in emotion_part:
                affect = _parse_group_affect_spec(emotion_part)
                emotion = None
                emotion_guidance = None
            elif "," in emotion_part or ":" in emotion_part:
                emotion_guidance = emotion_part or default_emotion_guidance
                emotion = default_emotion
                affect = None
            else:
                affect = emotion_part or default_affect
                emotion = None
                emotion_guidance = None
    elif clean in _GROUP_LANGUAGE_TAGS:
        language = clean
    elif clean in _GROUP_EMOTION_PRESET_TAGS:
        emotion = None
        affect = clean
        emotion_guidance = None
    elif clean:
        voice = clean
    return voice, language, emotion, affect, emotion_guidance


def _append_group_row(
    rows: list[dict[str, str | None]],
    *,
    line_number: int,
    text: str,
    voice: str | None,
    language: str | None,
    emotion: str | None,
    affect: str | None,
    emotion_guidance: str | None,
) -> None:
    line = text.strip()
    if not line:
        return
    if not voice:
        raise ValueError(
            f"Group-speak line {line_number} needs a [voice] label or --voice default"
        )
    rows.append(
        {
            "voice": voice,
            "language": language,
            "emotion": emotion,
            "affect": affect,
            "emotion_guidance": emotion_guidance,
            "text": line,
        }
    )


def _parse_group_lines(
    text: str,
    *,
    default_voice: str | None,
    default_language: str | None,
    default_emotion: str | None = None,
    default_affect: str | None = None,
    default_emotion_guidance: str | None = None,
) -> list[dict[str, str | None]]:
    rows = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        voice = default_voice
        language = default_language
        emotion = default_emotion
        affect = default_affect
        emotion_guidance = default_emotion_guidance
        position = 0
        found_tag = False
        for match in _GROUP_TAG_RE.finditer(line):
            found_tag = True
            _append_group_row(
                rows,
                line_number=line_number,
                text=line[position : match.start()],
                voice=voice,
                language=language,
                emotion=emotion,
                affect=affect,
                emotion_guidance=emotion_guidance,
            )
            voice, language, emotion, affect, emotion_guidance = _parse_group_tag(
                match.group(1),
                default_voice=voice,
                default_language=language,
                default_emotion=emotion,
                default_affect=affect,
                default_emotion_guidance=emotion_guidance,
            )
            position = match.end()
        _append_group_row(
            rows,
            line_number=line_number,
            text=line[position:] if found_tag else line,
            voice=voice,
            language=language,
            emotion=emotion,
            affect=affect,
            emotion_guidance=emotion_guidance,
        )
    return rows


def _split_text_chunks(text: str, *, max_chars: int, min_chars: int = 0) -> list[str]:
    from .planner import split_text_chunks

    return split_text_chunks(text, max_chars=max_chars, min_chars=min_chars)


def _split_text_chunk_records(
    text: str,
    *,
    max_chars: int,
    min_chars: int = 0,
) -> list[dict[str, object]]:
    from .planner import split_text_chunk_records

    return split_text_chunk_records(text, max_chars=max_chars, min_chars=min_chars)


def _render_text_chunks(
    runtime: ScyllasBandRuntime,
    chunks: list[dict[str, str | None]],
    args: argparse.Namespace,
) -> tuple[list[float], int, dict[str, object]]:
    from .streaming import render_text_records

    return render_text_records(runtime, chunks, args, lazy_preflight=False)


def _render_streaming_speak(
    runtime: ScyllasBandRuntime,
    chunks: list[dict[str, str | None]],
    args: argparse.Namespace,
) -> tuple[list[float], int, dict[str, object]]:
    from .planner import planner_options_from
    from .streaming import (
        _append_rendered_audio,
        _fade_sample_count,
        _gap_after_event,
        _render_metadata,
        synthesize_records_stream,
    )

    opts = planner_options_from(args)
    rendered: list[float] = []
    metadata_chunks: list[dict[str, object]] = []
    sample_rate = int(getattr(runtime.manifest.audio, "sample_rate", 0) or 0)
    previous_gap_samples = 0
    final_plan = None
    started_at = time.perf_counter()
    previous_event_at = started_at
    first_audio_ms: int | None = None
    events_handle = args.events.open("w", encoding="utf-8") if args.events else None
    try:
        for event in synthesize_records_stream(
            runtime,
            chunks,
            options=opts,
            source_kind="text",
            progress_stream=None,
        ):
            now = time.perf_counter()
            elapsed_ms = int(round((now - started_at) * 1000.0))
            delta_ms = int(round((now - previous_event_at) * 1000.0))
            previous_event_at = now
            if event.plan is not None:
                final_plan = event.plan
            if event.type == "audio_chunk" and first_audio_ms is None:
                first_audio_ms = elapsed_ms
            payload = event.to_dict(include_audio=False)
            payload["elapsed_ms"] = elapsed_ms
            payload["delta_ms"] = delta_ms
            payload["first_audio_ms"] = -1 if first_audio_ms is None else first_audio_ms
            if event.type == "audio_chunk":
                sample_rate = int(event.sample_rate or sample_rate)
                payload["sample_count"] = len(event.audio or [])
                chunk_wav_path: Path | None = None
                if args.chunk_output_dir is not None and event.chunk is not None:
                    args.chunk_output_dir.mkdir(parents=True, exist_ok=True)
                    chunk_wav_path = args.chunk_output_dir / f"{event.chunk.chunk_id}.wav"
                    _write_wav(chunk_wav_path, event.audio or [], sample_rate)
                    metadata = dict(payload.get("metadata") or {})
                    metadata["chunk_wav"] = str(chunk_wav_path)
                    payload["metadata"] = metadata
                fade_samples = _fade_sample_count(float(opts.boundary_fade_ms), sample_rate)
                next_gap_samples = _gap_after_event(event, sample_rate)
                start_sample, end_sample = _append_rendered_audio(
                    rendered,
                    event.audio,
                    fade_samples=fade_samples,
                    previous_gap_samples=previous_gap_samples,
                    next_gap_samples=next_gap_samples,
                )
                chunk_metadata = dict(event.metadata or {})
                chunk_metadata["start_sample"] = start_sample
                chunk_metadata["end_sample"] = end_sample
                if chunk_wav_path is not None:
                    chunk_metadata["chunk_wav"] = str(chunk_wav_path)
                metadata_chunks.append(chunk_metadata)
                if next_gap_samples:
                    rendered.extend([0.0] * next_gap_samples)
                    previous_gap_samples = next_gap_samples
                else:
                    previous_gap_samples = 0
            if events_handle is not None:
                print(json.dumps(payload, ensure_ascii=False), file=events_handle, flush=True)
            if not bool(getattr(args, "no_progress", False)):
                _print_stream_progress(payload)
    finally:
        if events_handle is not None:
            events_handle.close()
    if final_plan is None:
        raise ValueError("No synthesis plan was produced")
    metadata = _render_metadata(
        opts,
        final_plan=final_plan,
        sample_rate=sample_rate,
        chunks=metadata_chunks,
    )
    metadata["stream"] = {
        "events": str(args.events) if args.events else None,
        "chunk_output_dir": str(args.chunk_output_dir) if args.chunk_output_dir else None,
        "first_audio_ms": -1 if first_audio_ms is None else first_audio_ms,
    }
    return rendered, sample_rate, metadata


def _print_stream_progress(payload: dict[str, object]) -> None:
    event_type = str(payload.get("type") or "event")
    chunk = payload.get("chunk") if isinstance(payload.get("chunk"), dict) else {}
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    parts = [
        f"[stream] t={int(payload.get('elapsed_ms') or 0)}ms",
        f"+{int(payload.get('delta_ms') or 0)}ms",
        f"type={event_type}",
    ]
    chunk_id = chunk.get("chunk_id") or metadata.get("chunk_id")
    if chunk_id:
        parts.append(f"chunk={chunk_id}")
    chunk_count = metadata.get("chunk_count")
    if chunk_count:
        parts.append(f"count={chunk_count}")
    if event_type == "audio_chunk":
        sample_count = payload.get("sample_count")
        if sample_count:
            parts.append(f"samples={sample_count}")
        parts.append(f"first_audio_ms={int(payload.get('first_audio_ms') or -1)}")
        if "chunk_wav" in metadata:
            parts.append(f"wav={metadata['chunk_wav']}")
    print(" ".join(parts), file=sys.stderr, flush=True)


def _prepare_render_chunks(
    runtime: ScyllasBandRuntime,
    chunks: list[dict[str, str | None]],
    args: argparse.Namespace,
) -> list[dict[str, str | bool | int | None]]:
    from .planner import prepare_render_chunks

    return prepare_render_chunks(runtime, chunks, args)


def _preflight_split_overlong_chunks(
    runtime: ScyllasBandRuntime,
    chunks: list[dict[str, str | bool | int | None]],
    args: argparse.Namespace,
) -> list[dict[str, str | bool | int | None]]:
    from .planner import preflight_split_overlong_chunks

    return preflight_split_overlong_chunks(runtime, chunks, args)


def _estimate_chunk_duration_metadata(
    runtime: ScyllasBandRuntime,
    chunk: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    from .planner import estimate_chunk_duration_metadata

    return estimate_chunk_duration_metadata(runtime, chunk, args)


def _pause_after_ms(chunk: dict[str, object], *, args: argparse.Namespace, is_last: bool) -> int:
    from .planner import pause_after_ms

    return pause_after_ms(chunk, options=args, is_last=is_last)


def _chunk_context_before(chunks: list[dict[str, object]], index: int) -> str | None:
    from .planner import chunk_context_before

    return chunk_context_before(chunks, index)


def _chunk_context_after(
    chunks: list[dict[str, object]],
    index: int,
    *,
    lookahead_chunks: int = DEFAULT_LOOKAHEAD_CHUNKS,
) -> str | None:
    from .planner import chunk_context_after

    return chunk_context_after(chunks, index, lookahead_chunks=lookahead_chunks)


def _is_overlong_latent_error(exc: Exception) -> bool:
    from .planner import is_overlong_latent_error

    return is_overlong_latent_error(exc)


def _overlong_latent_counts(exc: Exception) -> tuple[int | None, int | None]:
    from .planner import overlong_latent_counts

    return overlong_latent_counts(exc)


def _is_overlong_phone_error(exc: Exception) -> bool:
    from .planner import is_overlong_phone_error

    return is_overlong_phone_error(exc)


def _overlong_phone_counts(exc: Exception) -> tuple[int | None, int | None]:
    from .planner import overlong_phone_counts

    return overlong_phone_counts(exc)


def _is_overlong_g2p_error(exc: Exception) -> bool:
    from .planner import is_overlong_g2p_error

    return is_overlong_g2p_error(exc)


def _overlong_g2p_counts(exc: Exception) -> tuple[int | None, int | None]:
    from .planner import overlong_g2p_counts

    return overlong_g2p_counts(exc)


def _split_chunk_for_overlong_retry(
    exc: Exception,
    chunk: dict[str, object],
    *,
    max_chars: int,
    min_chars: int = 0,
) -> tuple[str | None, list[dict[str, str | bool | int | None]]]:
    from .planner import split_chunk_for_overlong_retry

    return split_chunk_for_overlong_retry(exc, chunk, max_chars=max_chars, min_chars=min_chars)


def _split_chunk_for_latent_retry(
    chunk: dict[str, object],
    *,
    max_chars: int,
    min_chars: int = 0,
    predicted_latent_frames: int | None = None,
    fixed_latent_frames: int | None = None,
) -> list[dict[str, str | bool | int | None]]:
    from .planner import split_chunk_for_latent_retry

    return split_chunk_for_latent_retry(
        chunk,
        max_chars=max_chars,
        min_chars=min_chars,
        predicted_latent_frames=predicted_latent_frames,
        fixed_latent_frames=fixed_latent_frames,
    )


def _split_chunk_for_phone_retry(
    chunk: dict[str, object],
    *,
    max_chars: int,
    min_chars: int = 0,
    phone_count: int | None = None,
    fixed_frames: int | None = None,
) -> list[dict[str, str | bool | int | None]]:
    from .planner import split_chunk_for_phone_retry

    return split_chunk_for_phone_retry(
        chunk,
        max_chars=max_chars,
        min_chars=min_chars,
        phone_count=phone_count,
        fixed_frames=fixed_frames,
    )


def _split_chunk_for_g2p_retry(
    chunk: dict[str, object],
    *,
    max_chars: int,
    min_chars: int = 0,
    encoded_tokens: int | None = None,
    fixed_tokens: int | None = None,
) -> list[dict[str, str | bool | int | None]]:
    from .planner import split_chunk_for_g2p_retry

    return split_chunk_for_g2p_retry(
        chunk,
        max_chars=max_chars,
        min_chars=min_chars,
        encoded_tokens=encoded_tokens,
        fixed_tokens=fixed_tokens,
    )


def _write_wav(path: Path, audio: object, sample_rate: int) -> None:
    values = [float(item) for item in audio]
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        frames = bytearray()
        for value in values:
            clipped = max(-1.0, min(1.0, value))
            frames.extend(struct.pack("<h", int(round(clipped * 32767.0))))
        handle.writeframes(bytes(frames))
