#!/usr/bin/env python3
"""Generate the public Scylla's Band affect sample gallery from a curated script."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import wave
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from scyllasband import ScyllasBandRuntime, SynthesisRequest  # noqa: E402


AFFECT_AXES = ("calm", "joy", "anger", "sadness", "sarcasm", "questioning", "whisper")
DELIVERIES = ("calm", "joy", "anger", "sadness", "sarcasm", "whisper")
GALLERY_URL = "https://lowkeytea.github.io/scyllasband/"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--script",
        type=Path,
        default=SAMPLES_DIR / "affect_script.json",
        help="Curated voice/line/affect JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SAMPLES_DIR / "affect",
        help="Directory for generated affect WAV files",
    )
    parser.add_argument(
        "--multilingual-output-dir",
        type=Path,
        default=SAMPLES_DIR,
        help="Directory for the voice comparison WAV files",
    )
    parser.add_argument(
        "--gallery",
        type=Path,
        default=SAMPLES_DIR / "README.md",
        help="Markdown gallery to write after generation",
    )
    parser.add_argument(
        "--pages-dir",
        type=Path,
        default=REPO_ROOT / "docs",
        help="Self-contained GitHub Pages directory to refresh",
    )
    parser.add_argument("--bundle", type=Path, help="Bundle directory; defaults to the selected packaged backend")
    parser.add_argument("--backend", choices=("onnx", "litert"), default="onnx")
    parser.add_argument("--litert-accelerator", choices=("auto", "cpu", "gpu", "npu"), default="cpu")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--sampler", choices=("euler", "heun"), default="heun")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--min-sentence-pause-ms", type=float, default=320.0)
    parser.add_argument("--min-clause-pause-ms", type=float, default=160.0)
    parser.add_argument("--voices", nargs="+", help="Generate only these voice IDs")
    parser.add_argument("--deliveries", nargs="+", choices=DELIVERIES, help="Generate only these deliveries")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing WAV files")
    parser.add_argument(
        "--include-multilingual",
        action="store_true",
        help="Also rebuild each voice's English, Spanish, and Italian comparison clips",
    )
    parser.add_argument("--gallery-only", action="store_true", help="Only rebuild the Markdown gallery")
    return parser.parse_args()


def _load_script(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    voices = payload.get("voices")
    if payload.get("version") != 1 or not isinstance(voices, dict) or not voices:
        raise ValueError(f"Unsupported or empty affect script: {path}")
    comparison_lines = payload.get("comparison_lines")
    if not isinstance(comparison_lines, dict):
        raise ValueError(f"Affect script has no multilingual comparison lines: {path}")
    for language in ("en_us", "en_gb", "es", "it"):
        if not str(comparison_lines.get(language, "")).strip():
            raise ValueError(f"Affect script has no comparison line for {language}")
    for voice, spec in voices.items():
        if not isinstance(spec, dict) or not str(spec.get("language", "")).strip():
            raise ValueError(f"Voice {voice!r} is missing a language")
        demos = spec.get("demos")
        if not isinstance(demos, list) or not demos:
            raise ValueError(f"Voice {voice!r} has no demos")
        seen: set[str] = set()
        for demo in demos:
            delivery = str(demo.get("delivery", "")).strip()
            if delivery not in DELIVERIES or delivery in seen:
                raise ValueError(f"Voice {voice!r} has invalid or duplicate delivery {delivery!r}")
            seen.add(delivery)
            if not str(demo.get("text", "")).strip():
                raise ValueError(f"Voice {voice!r} delivery {delivery!r} has no text")
            guidance_scale = float(demo.get("guidance_scale", 1.0))
            if guidance_scale < 0.0:
                raise ValueError(f"Voice {voice!r} delivery {delivery!r} has negative CFG")
            affect = demo.get("affect")
            if not isinstance(affect, dict) or not affect:
                raise ValueError(f"Voice {voice!r} delivery {delivery!r} has no affect values")
            for axis, value in affect.items():
                if axis not in AFFECT_AXES or not 0.0 <= float(value) <= 1.0:
                    raise ValueError(f"Voice {voice!r} delivery {delivery!r} has invalid {axis}={value}")
    return payload


def _write_wav(path: Path, audio: object, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        frames = bytearray()
        for item in audio:  # type: ignore[union-attr]
            value = max(-1.0, min(1.0, float(item)))
            frames.extend(struct.pack("<h", int(round(value * 32767.0))))
        handle.writeframes(bytes(frames))


def _relative_link(target: Path, gallery: Path) -> str:
    return Path(os.path.relpath(target, gallery.parent)).as_posix()


def _gallery_link(anchor: str) -> str:
    return f"[Listen]({GALLERY_URL}#{anchor})"


def _affect_label(affect: Mapping[str, object]) -> str:
    return ", ".join(f"{axis}={float(value):g}" for axis, value in affect.items())


def _escape_table(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _render_gallery(
    payload: Mapping[str, Any],
    *,
    gallery_path: Path,
    output_dir: Path,
    backend: str,
    steps: int,
    sampler: str,
    seed: int,
) -> str:
    lines = [
        "# Scylla's Band Voice Samples",
        "",
        "These clips preview the ten managed voices and the six-axis affect controls in the current Scylla's Band model.",
        "The affect demonstrations use lines written for the requested delivery rather than neutral carrier text.",
        "The multilingual previews use the same comparison line in each language so voice differences are easier to hear.",
        f"[Open the interactive audio gallery]({GALLERY_URL}) to play every clip directly in the browser.",
        "",
        "Affect strengths are normalized model inputs: `0.25`, `0.5`, `0.75`, and `1.0` correspond to human ratings `1`, `2`, `3`, and `4`.",
        f"The generated clips below use the `{backend}` backend, {steps}-step `{sampler}`, deterministic seed `{seed}`, and the per-line CFG shown in each table.",
        "",
        "## Multilingual Voice Previews",
        "",
        "| Voice | English | Spanish | Italian |",
        "| --- | --- | --- | --- |",
    ]
    voices = payload["voices"]
    for voice in voices:
        en = _gallery_link(f"{voice}-en")
        es = _gallery_link(f"{voice}-es")
        it = _gallery_link(f"{voice}-it")
        lines.append(f"| {voice.title()} | {en} | {es} | {it} |")

    lines.extend(
        [
            "",
            "## Affect Demonstrations",
            "",
            "`calm`, `joy`, `anger`, and `sadness` are core delivery axes. `sarcasm` is demonstrated as an overlay mixed with a compatible core delivery.",
        ]
    )
    for voice, spec in voices.items():
        language = spec["language"]
        lines.extend(
            [
                "",
                f"### {voice.title()} (`{language}`)",
                "",
                "| Delivery | Affect values | CFG | Line | Audio |",
                "| --- | --- | ---: | --- | --- |",
            ]
        )
        for demo in spec["demos"]:
            delivery = demo["delivery"]
            player = _gallery_link(f"{voice}-{delivery}")
            lines.append(
                "| "
                + " | ".join(
                    (
                        delivery.title(),
                        f"`{_affect_label(demo['affect'])}`",
                        f"{float(demo['guidance_scale']):g}",
                        _escape_table(demo["text"]),
                        player,
                    )
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Rebuild",
            "",
            "From the Scylla's Band repository root:",
            "",
            "```bash",
            "python samples/generate_affect_samples.py --include-multilingual --overwrite",
            "```",
            "",
            "The same command refreshes the self-contained `docs/` directory used for GitHub Pages branch publishing.",
            "",
            "Use `--backend litert` to render the same script through the native LiteRT bundle, or `--voices ink rex` / `--deliveries anger sarcasm` for a focused subset.",
            "",
        ]
    )
    return "\n".join(lines)


def _sync_pages_site(pages_dir: Path) -> None:
    pages_dir.mkdir(parents=True, exist_ok=True)
    affect_dir = pages_dir / "affect"
    affect_dir.mkdir(parents=True, exist_ok=True)

    for stale in pages_dir.glob("*.wav"):
        stale.unlink()
    for stale in affect_dir.glob("*.wav"):
        stale.unlink()

    shutil.copy2(SAMPLES_DIR / "index.html", pages_dir / "index.html")
    shutil.copy2(SAMPLES_DIR / "affect_script.json", pages_dir / "affect_script.json")
    for source in SAMPLES_DIR.glob("*.wav"):
        shutil.copy2(source, pages_dir / source.name)
    for source in (SAMPLES_DIR / "affect").glob("*.wav"):
        shutil.copy2(source, affect_dir / source.name)
    (pages_dir / ".nojekyll").write_text("", encoding="utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _result_record(
    *,
    voice: str,
    language: str,
    demo: Mapping[str, Any],
    output_path: Path,
    result: object,
    seed: int,
    steps: int,
    sampler: str,
) -> dict[str, Any]:
    metadata = getattr(result, "metadata", {})
    return {
        "kind": "affect",
        "voice": voice,
        "language": language,
        "delivery": demo["delivery"],
        "affect": demo["affect"],
        "affect_guidance_scale": float(demo["guidance_scale"]),
        "text": demo["text"],
        "output": _relative_link(output_path, SAMPLES_DIR / "README.md"),
        "sample_rate": int(getattr(result, "sample_rate")),
        "duration_seconds": round(float(getattr(result, "duration_seconds")), 6),
        "seed": seed,
        "steps": steps,
        "sampler": sampler,
        "predicted_latent_frames": metadata.get("predicted_latent_frames"),
        "fixed_latent_frames": metadata.get("fixed_latent_frames"),
        "trimmed_audio_samples": metadata.get("trimmed_audio_samples"),
    }


def main() -> int:
    args = _parse_args()
    payload = _load_script(args.script)
    selected_voices = set(args.voices or payload["voices"].keys())
    unknown_voices = selected_voices.difference(payload["voices"])
    if unknown_voices:
        raise ValueError(f"Unknown voices: {', '.join(sorted(unknown_voices))}")
    selected_deliveries = set(args.deliveries or DELIVERIES)
    args.gallery.parent.mkdir(parents=True, exist_ok=True)

    if not args.gallery_only:
        bundle = args.bundle or (REPO_ROOT / "scyllasband" / "models" / args.backend)
        runtime = ScyllasBandRuntime.from_bundle(
            bundle,
            backends=[args.backend],
            litert_accelerator=args.litert_accelerator,
        )
        records: list[dict[str, Any]] = []
        generation_manifest = args.output_dir / "generation.json"
        if args.include_multilingual:
            for voice, spec in payload["voices"].items():
                if voice not in selected_voices:
                    continue
                languages = (
                    ("en", str(spec["language"])),
                    ("es", "es"),
                    ("it", "it"),
                )
                for suffix, language in languages:
                    text = str(payload["comparison_lines"][language])
                    output_path = args.multilingual_output_dir / f"{voice}_{suffix}.wav"
                    if output_path.exists() and not args.overwrite:
                        print(f"skip {voice}/{language}: {output_path}", flush=True)
                        continue
                    print(f"generate {voice}/{language}: {text}", flush=True)
                    result = runtime.synthesize(
                        SynthesisRequest(
                            text=text,
                            voice_id=voice,
                            language=language,
                            steps=args.steps,
                            sampler=args.sampler,
                            seed=args.seed,
                            min_sentence_pause_ms=args.min_sentence_pause_ms,
                            min_clause_pause_ms=args.min_clause_pause_ms,
                        )
                    )
                    _write_wav(output_path, result.audio, result.sample_rate)
                    metadata = result.metadata
                    records.append(
                        {
                            "kind": "multilingual",
                            "voice": voice,
                            "language": language,
                            "text": text,
                            "output": _relative_link(output_path, SAMPLES_DIR / "README.md"),
                            "sample_rate": int(result.sample_rate),
                            "duration_seconds": round(float(result.duration_seconds), 6),
                            "seed": args.seed,
                            "steps": args.steps,
                            "sampler": args.sampler,
                            "predicted_latent_frames": metadata.get("predicted_latent_frames"),
                            "fixed_latent_frames": metadata.get("fixed_latent_frames"),
                            "trimmed_audio_samples": metadata.get("trimmed_audio_samples"),
                        }
                    )
                    _write_json(
                        generation_manifest,
                        {
                            "generated_at": datetime.now(timezone.utc).isoformat(),
                            "backend": args.backend,
                            "bundle": _display_path(bundle),
                            "samples": records,
                        },
                    )
        for voice, spec in payload["voices"].items():
            if voice not in selected_voices:
                continue
            language = str(spec["language"])
            for demo in spec["demos"]:
                delivery = str(demo["delivery"])
                if delivery not in selected_deliveries:
                    continue
                output_path = args.output_dir / f"{voice}_{delivery}.wav"
                if output_path.exists() and not args.overwrite:
                    print(f"skip {voice}/{delivery}: {output_path}", flush=True)
                    continue
                print(f"generate {voice}/{delivery}: {demo['text']}", flush=True)
                result = runtime.synthesize(
                    SynthesisRequest(
                        text=str(demo["text"]),
                        voice_id=voice,
                        language=language,
                        affect=demo["affect"],
                        affect_guidance_scale=float(demo["guidance_scale"]),
                        steps=args.steps,
                        sampler=args.sampler,
                        seed=args.seed,
                        min_sentence_pause_ms=args.min_sentence_pause_ms,
                        min_clause_pause_ms=args.min_clause_pause_ms,
                    )
                )
                _write_wav(output_path, result.audio, result.sample_rate)
                records.append(
                    _result_record(
                        voice=voice,
                        language=language,
                        demo=demo,
                        output_path=output_path,
                        result=result,
                        seed=args.seed,
                        steps=args.steps,
                        sampler=args.sampler,
                    )
                )
                _write_json(
                    generation_manifest,
                    {
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                        "backend": args.backend,
                        "bundle": _display_path(bundle),
                        "samples": records,
                    },
                )

    args.gallery.write_text(
        _render_gallery(
            payload,
            gallery_path=args.gallery,
            output_dir=args.output_dir,
            backend=args.backend,
            steps=args.steps,
            sampler=args.sampler,
            seed=args.seed,
        ),
        encoding="utf-8",
    )
    _sync_pages_site(args.pages_dir)
    print(f"gallery: {args.gallery}")
    print(f"pages site: {args.pages_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
