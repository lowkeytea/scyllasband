"""Render a spoken-and-sung script as one track with Scylla's Band.

The script (``script.json``) is a list of cues. A ``speak`` cue is one voice saying a line with optional delivery controls;
a ``sing`` cue is one or more voices singing a melody score (``melodies/*.json``, the ``sing`` command's format). Cues are
rendered one by one with the runtime, placed on a timeline with a gap between them (a ``sing`` cue with several voices renders
the same score once per voice and sums them, so the voices sing in unison, each in its own octave), normalized, and written
as one WAV with a cue sheet (start and end of every cue, for a video or a mix). Rendered cues are cached, so editing one
line re-renders only that line.

    python examples/ghost_in_the_machine/make_track.py                      # the installed bundle (download first)
    python examples/ghost_in_the_machine/make_track.py --bundle <dir> --backend litert --only chorus,tag
    python examples/ghost_in_the_machine/make_track.py --out track.wav --cues track.cues.json

Singing needs a bundle with delivery modes (the singing release); speech works with any bundle.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from scyllasband.cli import write_wav
from scyllasband.download import DEFAULT_MODELS_DIR, bundle_dir, default_installed_flavor
from scyllasband.runtime import ScyllasBandRuntime, SynthesisRequest

HERE = Path(__file__).resolve().parent


def load_runtime(args: argparse.Namespace) -> ScyllasBandRuntime:
    bundle = args.bundle
    if bundle is None:
        flavor = args.backend or default_installed_flavor(args.models_dir)
        if flavor is None:
            sys.exit("No bundle installed: run `python -m scyllasband download` first, or pass --bundle")
        bundle = bundle_dir(args.models_dir, flavor)
    return ScyllasBandRuntime.from_bundle(bundle, backend=args.backend, threads=args.threads)


def render_cue(runtime: ScyllasBandRuntime, cue: dict, cache: Path, *, force: bool) -> tuple[np.ndarray, int, dict]:
    """One cue's audio (mono float32), its sample rate and metadata; cached as <cache>/<name>.wav + .json."""
    wav, meta_path = cache / f"{cue['name']}.wav", cache / f"{cue['name']}.json"
    if wav.is_file() and meta_path.is_file() and not force:
        audio, sample_rate = read_wav(wav)
        return audio, sample_rate, json.loads(meta_path.read_text())
    if cue["type"] == "speak":
        delivery = cue.get("delivery") or {}
        result = runtime.synthesize(SynthesisRequest(text=cue["text"], voice_id=cue["voice"], language=cue.get("language"),
                                                     intensity=delivery.get("intensity"), pace=delivery.get("pace"),
                                                     pitch=delivery.get("pitch"), seed=cue.get("seed")), progress=sys.stderr)
        audio, sample_rate, meta = result.audio, result.sample_rate, dict(type="speak", voice=cue["voice"], text=cue["text"])
    elif cue["type"] == "sing":
        score = json.loads((HERE / cue["score"]).read_text(encoding="utf-8"))
        voices, gains = list(cue["voices"]), list(cue.get("gains") or [1.0] * len(cue["voices"]))
        takes, plans = [], []
        for voice, gain in zip(voices, gains):
            result = runtime.sing(score, voice, cue.get("language"), seed=cue.get("seed"), transpose=int(cue.get("transpose", 0)),
                                  steps=cue.get("steps"), decoder=cue.get("decoder"), progress=sys.stderr)
            takes.append(result.audio.astype(np.float32) * float(gain)); plans.append(result.metadata); sample_rate = result.sample_rate
        length = max(len(t) for t in takes)     # the takes share the score's timeline; lengths differ by a few samples
        audio = np.zeros(length, dtype=np.float32)
        for take in takes:
            audio[: len(take)] += take
        audio /= max(1.0, float(np.abs(audio).max()) / 0.95)
        meta = dict(type="sing", voices=voices, score=cue["score"], text=score["text"], transpose=cue.get("transpose", 0),
                    phrases=[dict(start=p["start"], end=p["end"], text=p["text"]) for p in plans[0]["phrases"]])
    else:
        raise ValueError(f"cue {cue['name']!r}: type must be speak or sing")
    cache.mkdir(parents=True, exist_ok=True)
    write_wav(wav, audio, sample_rate); meta_path.write_text(json.dumps(meta, indent=2))
    return audio.astype(np.float32), sample_rate, meta


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    import wave
    with wave.open(str(path)) as handle:
        frames = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        return frames, handle.getframerate()


def trim_silence(audio: np.ndarray, sample_rate: int, threshold: float = 1e-3, keep: float = 0.05) -> np.ndarray:
    """Cut leading and trailing silence, keeping ``keep`` seconds of it, so the gap between cues is the script's gap."""
    loud = np.flatnonzero(np.abs(audio) > threshold)
    if not len(loud):
        return audio
    pad = int(keep * sample_rate)
    return audio[max(0, loud[0] - pad): min(len(audio), loud[-1] + pad)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--script", type=Path, default=HERE / "script.json")
    parser.add_argument("--out", type=Path, default=HERE / "out" / "ghost_in_the_machine.wav")
    parser.add_argument("--cues", type=Path, help="cue sheet JSON (default: next to --out, .cues.json)")
    parser.add_argument("--cache", type=Path, help="rendered cues (default: next to --out, cues/)")
    parser.add_argument("--only", help="comma-separated cue names to (re)render; the others come from the cache")
    parser.add_argument("--force", action="store_true", help="re-render every cue")
    parser.add_argument("--bundle", type=Path, help="model bundle directory (default: the installed bundle)")
    parser.add_argument("--backend", help="litert, onnx, coreml or coreai (default: the bundle's)")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()
    script = json.loads(args.script.read_text(encoding="utf-8"))
    out = args.out; out.parent.mkdir(parents=True, exist_ok=True)
    cache = args.cache or out.parent / "cues"
    only = set(args.only.split(",")) if args.only else None
    runtime = load_runtime(args)
    pieces, sheet, cursor, sample_rate = [], [], 0.0, int(script.get("sample_rate", 24000))
    for cue in script["cues"]:
        audio, rate, meta = render_cue(runtime, cue, cache, force=args.force or (only is not None and cue["name"] in only))
        if rate != sample_rate:
            raise ValueError(f"cue {cue['name']}: sample rate {rate} differs from the script's {sample_rate}")
        audio = trim_silence(audio, rate)
        start, end = cursor, cursor + len(audio) / rate
        pieces.append((start, audio)); sheet.append(dict(name=cue["name"], start=round(start, 3), end=round(end, 3), **meta))
        cursor = end + float(cue.get("gap", script.get("gap", 0.5)))
        print(f"{cue['name']:12s} {start:7.2f} - {end:7.2f} s  {meta['text'][:60]}", flush=True)
    track = np.zeros(int(cursor * sample_rate) + sample_rate // 2, dtype=np.float32)
    for start, audio in pieces:
        i = int(start * sample_rate); track[i: i + len(audio)] += audio
    track *= 0.9 / max(1e-6, float(np.abs(track).max()))
    write_wav(out, track, sample_rate)
    cues = args.cues or out.with_suffix(".cues.json")
    cues.write_text(json.dumps(dict(title=script.get("title"), sample_rate=sample_rate, seconds=round(len(track) / sample_rate, 3),
                                    cues=sheet), indent=2))
    print(f"wrote {out} ({len(track) / sample_rate:.1f} s) and {cues}")


if __name__ == "__main__":
    main()
