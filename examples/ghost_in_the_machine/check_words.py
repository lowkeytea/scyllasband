"""Transcribe every rendered cue and compare it with the words it was given: a quick, objective check that a line or a sung
section came out intelligible before you listen through the whole track. Needs ``faster-whisper`` (``pip install
faster-whisper``); it is not a dependency of the runtime.

    python examples/ghost_in_the_machine/check_words.py                    # out/ghost_in_the_machine.cues.json
    python examples/ghost_in_the_machine/check_words.py --cues track.cues.json --model medium --device cpu

Precision = share of the transcript's words that are in the text, coverage = share of the text's words that were heard
(order-free, lower-cased, punctuation dropped, numbers compared by value). Whisper writes "14" for "fourteen" and "syla"
for "Scylla": read the transcript before trusting a low score on a name."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
NUMBERS = {w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
                                           "fifteen sixteen seventeen eighteen nineteen twenty".split())}


def words(text: str) -> list[str]:
    out = []
    for w in re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split():
        w = w.strip("'")
        if w:
            out.append(NUMBERS.get(w, w))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cues", type=Path, default=HERE / "out" / "ghost_in_the_machine.cues.json")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    from faster_whisper import WhisperModel
    asr = WhisperModel(args.model, device=args.device, compute_type="float16" if args.device != "cpu" else "int8")
    sheet = json.loads(args.cues.read_text(encoding="utf-8"))
    cache = args.cues.parent / "cues"
    for cue in sheet["cues"]:
        segments, _ = asr.transcribe(str(cache / f"{cue['name']}.wav"), language="en", beam_size=5, condition_on_previous_text=False)
        heard, given = words(" ".join(s.text for s in segments)), words(cue["text"])
        precision = sum(w in given for w in heard) / max(1, len(heard))
        coverage = sum(w in heard for w in given) / max(1, len(given))
        print(f"{cue['name']:14s} {cue['start']:6.1f}-{cue['end']:6.1f} s  precision {precision:.2f} coverage {coverage:.2f}  | {' '.join(heard)}")


if __name__ == "__main__":
    main()
