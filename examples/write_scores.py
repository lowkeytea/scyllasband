#!/usr/bin/env python3
"""Compile a piece's score source (score.txt) into the melody JSON files the sing command reads.

    python examples/write_scores.py examples/face_down          # writes examples/face_down/melodies/<section>.json

The source is one text file per piece, readable as a lead sheet:

    [verse_1] bpm=120 style=pop
    Phone:C4:.5 face:D4:.5 down:E4:1 on:D4:.5 the:C4:.5 ta-ble:D4,C4:.5,.5 to-night:A3,C4:.5,2 | 1.5

One section per [name] header with its tempo and style; one line of the lyric per line of text. A token is
word:pitches:beats. A word sung over several notes lists one pitch and one beat per note, in order, with the syllables
marked by hyphens (to-night); a syllable held across pitches simply has more notes than syllables (night:C4,D4:1,1).
A bar | r is a rest of r beats; a line's final rest separates it from the next line (the runtime starts a new phrase at
rests of a quarter second or more). Lines starting with # are comments. The labels spell the lyric, so the compiled
score's text is the words in order with the hyphens removed.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HEADER = re.compile(r"^\[(?P<name>[A-Za-z0-9_]+)\]\s*(?P<attrs>.*)$")


def compile_source(text: str) -> dict[str, dict]:
    scores: dict[str, dict] = {}
    current: dict | None = None
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        header = HEADER.match(line)
        if header:
            attrs = dict(a.split("=", 1) for a in header["attrs"].split())
            current = scores[header["name"]] = dict(text="", bpm=float(attrs.get("bpm", 100)), style=attrs.get("style", "pop"),
                                                     notes=[], words=[])
            if current["bpm"].is_integer():
                current["bpm"] = int(current["bpm"])
            continue
        if current is None:
            raise ValueError(f"line {number}: a [section] header must come first")
        for token in line.split():
            if token == "|":
                raise ValueError(f"line {number}: a rest is written '| beats'")
            if token.startswith("|"):
                beats = float(token[1:]) if len(token) > 1 else None
                if beats is None:
                    raise ValueError(f"line {number}: a rest is written '| beats' (no space)")
                current["notes"].append(["-", "rest", beats])
                continue
            try:
                word, pitches, beats = token.split(":")
            except ValueError as error:
                raise ValueError(f"line {number}: token {token!r} is not word:pitches:beats") from error
            pitch_list, beat_list = pitches.split(","), [float(b) for b in beats.split(",")]
            if len(pitch_list) != len(beat_list):
                raise ValueError(f"line {number}: {token!r} lists {len(pitch_list)} pitches and {len(beat_list)} beats")
            plain = word.replace("-", "")
            current["words"].append(plain)
            if len(pitch_list) == 1:
                current["notes"].append([plain, pitch_list[0], beat_list[0]])
            else:
                for index, (pitch, beat) in enumerate(zip(pitch_list, beat_list), start=1):
                    current["notes"].append([f"{plain}~{index}", pitch, beat])
    for score in scores.values():
        score["text"] = " ".join(score.pop("words"))
        for note in score["notes"]:
            if float(note[2]).is_integer():
                note[2] = int(note[2])
    return scores


def main() -> int:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    source = folder / "score.txt"
    scores = compile_source(source.read_text(encoding="utf-8"))
    out = folder / "melodies"
    out.mkdir(exist_ok=True)
    for name, score in scores.items():
        beats = sum(float(n[2]) for n in score["notes"])
        seconds = beats * 60 / score["bpm"]
        syllables = sum(1 for n in score["notes"] if n[1] != "rest")
        (out / f"{name}.json").write_text(json.dumps(score, indent=1), encoding="utf-8")
        print(f"{name:12s} {score['bpm']:>4} bpm {score['style']:6s} {beats:6.2f} beats {seconds:5.1f} s {syllables:3d} notes  {score['text'][:50]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
