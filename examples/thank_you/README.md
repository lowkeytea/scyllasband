# Thank You for Participating: a ballad, sung sweetly, spoken flat

A 72 bpm ballad from a facility voice that is being very gracious about your departure. Scylla sings the verses and the
chorus and delivers the spoken lines with the intensity turned well down, which is where the deadpan comes from; Thistle
joins the last chorus. About two minutes. Files are as in [Face Down](../face_down/README.md).

## Render it

```bash
python examples/write_scores.py examples/thank_you
python examples/ghost_in_the_machine/make_track.py --script examples/thank_you/script.json --out out/thank_you.wav
```

## Writing it

```text
[chorus] bpm=72 style=ballad
Thank:E4:1 you:D4:.5 for:E4:.5 par-ti-ci-pa-ting:G4,G4,E4,D4,E4:1,.5,.5,1,2 |1
Your:E4:.5 re-sults:D4,E4:.5,1 are:D4:.5 fa-sci-na-ting:G4,E4,D4,C4:1,.5,1,2 |1
```

- **`ballad` style, long notes, one line per phrase.** Quarter notes with the line's last word held for two beats and a
  one-beat rest (0.83 s) after it, so every line is its own phrase of six to seven seconds. The ballad delivery holds
  vowels longer and moves between notes more smoothly than `pop`; the same score in `pop` style sounds clipped.
- **A narrow verse, a lifted chorus.** Verses on A3 to E4, mostly by step, with one reach to G4 on the word that turns
  the line (`failed`, `locked`); the chorus starts on E4 and puts the five syllables of `participating` on a rising,
  falling shape ending on a two-beat note. Written around C4 and transposed -3 for Scylla and Thistle, both centred near
  B3.
- **The spoken lines carry the character.** Intensity -1.2 and a slightly slow pace for the intro, -1.0 for the
  bridge, -0.6 for the outro: the words are polite, the delivery is flat, and the gap between them is the joke. The sung
  lines are left at the voice's own warmth on purpose.
- **Thistle doubles the last chorus** at half gain, aligned onto Scylla's notes (see Ghost in the Machine, step 5): a
  second sweet voice makes the last chorus sound like a choir of two that has done this before.

## What the check found

Whisper large-v3 on the cues, LiteRT bundle of release v4-20261106: every cue at 100 % of its words, the first time.
The transcript shows a `see` before `the tests you failed` and an `and` before two verse lines, which is the model
voicing the lead-in of a phrase (the same thing noted in Ghost in the Machine); the runtime gates the lead-in before the
first note, and what is left is a breath Whisper spells.
