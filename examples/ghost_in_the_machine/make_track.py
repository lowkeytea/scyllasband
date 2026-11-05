"""Render a spoken-and-sung script as one track with Scylla's Band.

The script (``script.json``) is a list of cues. A ``speak`` cue is one voice saying a line with optional delivery controls;
a ``sing`` cue is one or more voices singing a melody score (``melodies/*.json``, the ``sing`` command's format). Cues are
rendered one by one with the runtime, placed on a timeline with a gap between them (a ``sing`` cue with several voices renders
the same score once per voice, aligns the other voices' notes onto the first voice's onsets and sums them, so the voices
sing in unison, each in its own octave), normalized, and written
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

ALIGN_SEARCH = 0.10      # seconds: how far a double's note may be shifted to meet the lead's
ALIGN_ONSET = 0.08       # seconds before a note's start included in the match (the consonant lands there)
ALIGN_MAX_STEP = 0.06    # seconds: the most the shift may change from one note to the next
ALIGN_FADE = 0.012       # seconds of crossfade where shifted segments join


def envelope(audio: np.ndarray, sample_rate: int, hop: float = 0.002, window: float = 0.01) -> np.ndarray:
    """RMS envelope every ``hop`` seconds over ``window`` seconds."""
    h, w = max(1, int(hop * sample_rate)), max(1, int(window * sample_rate))
    squares = np.concatenate([[0.0], np.cumsum(audio.astype(np.float64) ** 2)])
    starts = np.arange(0, max(1, audio.size - w), h)
    return np.sqrt((squares[starts + w] - squares[starts]) / w)


def align_to_lead(lead: np.ndarray, double: np.ndarray, notes: list[tuple[float, float]], sample_rate: int) -> tuple[np.ndarray, list[float]]:
    """Shift each note of ``double`` so its onset lands where the lead's does, and return the shifted take plus the shift
    (seconds, positive = the double was late) found per note.

    The score gives both takes the same note boundaries, but the model decides for itself where inside a note the consonant
    falls, and two voices differ by up to about 150 ms there; summed as they are, every consonant doubles ("ma-ma-chine").
    For each note the lead's and the double's RMS envelopes are compared over the onset region (``ALIGN_ONSET`` before the
    note to 400 ms into it) at shifts up to ``ALIGN_SEARCH``; the best-correlated shift is kept, limited to a step of
    ``ALIGN_MAX_STEP`` from the previous note's so a weak match cannot throw a note out. The double is then rebuilt note by
    note from its shifted segments, joined with short crossfades at the note boundaries (where the consonants are), so vowels
    keep their pitch and the voice its timbre."""
    hop = 0.002
    env_lead, env_double = envelope(lead, sample_rate, hop), envelope(double, sample_rate, hop)
    search = int(ALIGN_SEARCH / hop)
    shifts, previous = [], 0.0
    for start, end in notes:
        a = max(0, int((start - ALIGN_ONSET) / hop)); b = min(len(env_lead), int(min(end, start + 0.4) / hop))
        best, best_score = previous, -1.0
        if b - a >= 10:
            ref = env_lead[a:b] - env_lead[a:b].mean()
            for k in range(-search, search + 1):
                seg = env_double[a + k: b + k] if a + k >= 0 else None
                if seg is None or len(seg) != len(ref):
                    continue
                seg = seg - seg.mean()
                score = float(np.dot(ref, seg) / (np.linalg.norm(ref) * np.linalg.norm(seg) + 1e-9))
                if score > best_score:
                    best, best_score = k * hop, score
            if best_score < 0.3:
                best = previous
        best = float(np.clip(best, previous - ALIGN_MAX_STEP, previous + ALIGN_MAX_STEP))
        shifts.append(best); previous = best
    out = np.zeros_like(double)
    fade = max(1, int(ALIGN_FADE * sample_rate)); ramp = np.linspace(0.0, 1.0, 2 * fade, dtype=np.float32)
    edges = [0] + [int((notes[i][1] + notes[i + 1][0]) / 2 * sample_rate) for i in range(len(notes) - 1)] + [double.size]
    for (lo, hi), shift in zip(zip(edges, edges[1:]), shifts):   # segment k, shifted; linear crossfades centred on the edges
        lo2, hi2 = max(0, lo - fade), min(double.size, hi + fade)
        piece = double[np.clip(np.arange(lo2, hi2) + int(round(shift * sample_rate)), 0, double.size - 1)].copy()
        if lo > 0:
            piece[: 2 * fade] *= ramp[: min(2 * fade, piece.size)]
        if hi < double.size:
            piece[-2 * fade:] *= ramp[::-1][-min(2 * fade, piece.size):]
        out[lo2:hi2] += piece
    return out, shifts



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
        for voice in voices:
            result = runtime.sing(score, voice, cue.get("language"), seed=cue.get("seed"), transpose=int(cue.get("transpose", 0)),
                                  steps=cue.get("steps"), decoder=cue.get("decoder"), progress=sys.stderr)
            takes.append(result.audio.astype(np.float32)); plans.append(result.metadata); sample_rate = result.sample_rate
        length = max(len(t) for t in takes)     # the takes share the score's timeline; lengths differ by a few samples
        takes = [np.pad(t, (0, length - len(t))) for t in takes]
        notes = [(p["start"] + n["start"], p["start"] + n["end"]) for p in plans[0]["phrases"] for n in p["notes"] if n["note"] != "rest"]
        alignment = {}
        if len(takes) > 1 and cue.get("align", "lead") == "lead":   # the first voice leads; the others are shifted onto it, note by note
            for voice, k in zip(voices[1:], range(1, len(takes))):
                takes[k], shifts = align_to_lead(takes[0], takes[k], notes, sample_rate)
                alignment[voice] = [round(x, 3) for x in shifts]
        audio = np.zeros(length, dtype=np.float32)
        for take, gain in zip(takes, gains):
            audio += take * float(gain)
        audio /= max(1.0, float(np.abs(audio).max()) / 0.95)
        meta = dict(type="sing", voices=voices, gains=gains, score=cue["score"], text=score["text"], transpose=cue.get("transpose", 0),
                    phrases=[dict(start=p["start"], end=p["end"], text=p["text"]) for p in plans[0]["phrases"]],
                    notes=[dict(label=n["label"], start=round(p["start"] + n["start"], 3), end=round(p["start"] + n["end"], 3))
                           for p in plans[0]["phrases"] for n in p["notes"] if n["note"] != "rest"], alignment=alignment)
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
