# Bubble Wrap: talk-singing at 96 bpm, two voices trading verses

The fast end of the singing path, made into a piece: a hook and two verses on sixteenth-note syllables at 96 bpm,
Tuesday on the hook and the first verse, Orpheus on the second, both on the last hook with Orpheus an octave below.
Scylla reads the news at the start and the end. The subject is the week the bubble popped: the market is in pieces and
a couple hundred megabytes on your phone would like it noted that it never had a bubble.

It builds on [Kitchen Patter](../kitchen_patter/README.md) (the density rules) and
[Ghost in the Machine](../ghost_in_the_machine/README.md) (the cue script and the doubled voices). Files are as in
[Face Down](../face_down/README.md): `score.txt`, the compiled `melodies/`, and `script.json`.

## Render it

```bash
python examples/write_scores.py examples/bubble_wrap
python examples/ghost_in_the_machine/make_track.py --script examples/bubble_wrap/script.json --out out/bubble_wrap.wav
```

## Writing it

```text
[hook] bpm=96 style=pop
Pop:G4:.5 goes:E4:.25 the:E4:.25 bub-ble:G4,E4:.5,.25 |.25 pop:G4:.5 pop:G4:.5 there:E4:.25 it:D4:.25 goes:C4:1 |.25
Ev-ery-bo-dy:E4,E4,E4,E4:.25,.25,.25,.25 pa-ni-cking:G4,E4,E4:.5,.25,.25 ev-ery-bo-dy:E4,E4,D4,D4:.25,.25,.25,.25 knows:C4:1 |1
```

- **One note per syllable; stressed syllables on eighths, the rest on sixteenths.** At 96 bpm a sixteenth is 0.16 s, a
  little slower than Kitchen Patter's 0.13 s; the hook runs at about 4.5 syllables a second.
- **Three pitches and a lift.** Nearly everything sits on C4, D4 and E4; G4 marks the stressed syllable of the line's
  key word (`Pop`, `bubble`, `phone`, `market`). Patter is rhythm, not range.
- **Breaths and phrase ends are different rests.** A sixteenth rest (`|.25`, 0.16 s) is a breath inside the phrase; a
  one-beat rest (0.63 s) ends it. A verse is four couplets, each its own phrase of about six seconds.
- **Clusters get the eighth.** `raised a`, `read the`, `jobs`, `farm`, `call`: the word before a consonant cluster is
  the one that gets held, otherwise the cluster has nowhere to go.
- **Orpheus an octave down, same score.** Transpose -2 puts Tuesday at her centre; the runtime's register moves the
  same notes an octave lower for Orpheus, whose centre is near D3.

## What the check found

Whisper large-v3 on the cues, LiteRT bundle of release v4-20261106, the first render:

| Cue | Words heard | What came out |
| --- | --- | --- |
| verse 1 (Tuesday, 101 notes) | 100 % | |
| hook 2 | 100 % | |
| hook 1 (seed 3) | 91 % | `model` as `muddle`, `afloat` as `a full out` |
| hook 3 (two voices, seed 5) | 97 % | `coat` as `code` |
| verse 2 (Orpheus) | 93 % | `Analysts` as `send their lists`; `percent`, held a beat and a half, as `po or sand` |
| outro (spoken) | 80 % | `Eleven percent.` came out as `Eleven.` |

The fixes, each one thing: `lysts` from an eighth to a dotted eighth (its final cluster meets `are`), `cent` from a
beat and a half to one beat with the rest lengthened to match, the two hooks on other seeds, and the spoken outro given a
noun (`Eleven percent battery.`) so the number is not the end of the sentence. The verse 1 result is the one to note:
a hundred and one syllables at this speed, every word heard the first time.
