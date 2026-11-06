# Face Down: a pop song from scores, three voices on the chorus

A 120 bpm pop song, about a minute and three quarters, made entirely with Scylla's Band: Poppy sings the first verse,
Gwen the second, and the chorus is Poppy with Gwen doubling her and Felix an octave below. Ink opens and closes it as
the host of the evening and Priya calls the last chorus. The piece follows the route laid out in
[Ghost in the Machine](../ghost_in_the_machine/README.md); this README records what is different here and what the
checks found.

Files:

| File | What it is |
| --- | --- |
| `score.txt` | The sung sections as a lead sheet: one line of lyric per line, each syllable with its pitch and length |
| `melodies/*.json` | The compiled scores, one per section, in the `sing` command's format |
| `script.json` | The cues in order: spoken lines with their delivery, sung sections with voices, transpose, seeds and gaps |

## Render it

```bash
python -m scyllasband download --flavor litert
python examples/write_scores.py examples/face_down                                   # score.txt -> melodies/*.json
python examples/ghost_in_the_machine/make_track.py --script examples/face_down/script.json --out out/face_down.wav
python examples/ghost_in_the_machine/check_words.py --cues out/face_down.cues.json  # optional, needs faster-whisper
```

`make_track.py` is shared by every piece: it reads any script, resolves the scores next to it, caches each rendered cue
and places them on a timeline (see the Ghost in the Machine README for what it does with doubled voices).

## The score source

`score.txt` is what was actually written; `melodies/*.json` is compiled from it. A token is `word:pitches:beats`, a
word over several notes lists them in order with its syllables hyphenated, and `|1` is a one-beat rest:

```text
[chorus] bpm=120 style=pop
Put:E4:.5 it:E4:.5 face:G4:1 down:E4:1.5 we're:D4:.5 in:E4:.5 the:D4:.5 room:E4:1 now:C4:1 |1
No:E4:.5 one:E4:.5 to:D4:.5 an-swer:E4,G4:.5,1 no:E4:.5 one:D4:.5 a-round:C4,D4:.5,2.5 |1
```

Why a text source: a song is rewritten by ear many times, and a line of lyric with its rhythm is far quicker to change
than forty JSON triples. `write_scores.py` is twenty lines of parsing and keeps the JSON exact.

## Decisions

- **Verse, pre-chorus, chorus.** Verses sit on A3 to E4 in eighths and quarters with the stressed word on the longer
  note; the pre-chorus climbs E4 to G4 on repeated quarter notes (`Hands up, hands up`) so the chorus arrives from
  above; the chorus hook (`Put it face down`) jumps to G4 and comes back, and the title line is sung four times across
  the song, which is what makes it the chorus.
- **Two-bar lines, a rest after each.** Every line ends on a held note and a one-beat rest (half a second at 120 bpm).
  The runtime joined pairs of lines into eight-second phrases rather than one phrase per line, and that was fine: the
  phrase break falls on a rest either way.
- **Written around C4, transposed -3.** Poppy's and Gwen's sung centres are near A#3 and G#3, so the verses land on
  F#3 to C#4 and the chorus peaks at E4, five to six semitones above centre. Felix takes the same score and the
  runtime's register puts him an octave below.
- **Second verse to a second voice.** Gwen takes verse 2 on the same key and the same transposition; the chorus is then
  three voices who have each been heard alone first.
- **Spoken frame.** Ink's intro and outro are a host's lines, delivered a touch up and a touch down from neutral; Priya's
  bridge is intense and quick, a call to the room, not a lyric.

## What the check found

Whisper large-v3 on the cues, rendered with the LiteRT bundle of release v4-20261106 (4 Euler steps, no guidance):

| Cue | Words heard | Note |
| --- | --- | --- |
| intro, pre-choruses, choruses 1 and 2, verse 2, bridge, outro | 100 % | |
| verse 1 (seed 3) | 93 % | the last `okay`, held on two beats, came out as `a story` |
| chorus 3 (three voices) | 97 % | an extra `who` heard in the doubled line |

Verse 1 was re-rendered on another seed, which is often all a line needs; the rest of the piece was left as it was.
(Whisper writes `1` for `one` and `3` for `three`; those are not misses.)
