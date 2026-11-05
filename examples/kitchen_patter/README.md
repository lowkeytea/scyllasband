# Kitchen Patter: fast talk-singing from one score

A companion to [Ghost in the Machine](../ghost_in_the_machine/README.md). That piece is a mid-tempo song; this one is the
other end of what the singing path does: a run of sixteenth-note syllables at 112 bpm, one note per syllable, closer to
rap or patter than to a ballad. It is a single score and one command, kept here because it shows what the score rules in
the other tutorial look like at speed, and because it was the piece used to check that the fast setting of the
singing flow keeps every word.

```bash
python -m scyllasband download --flavor litert       # a bundle that sings (the singing release)
python -m scyllasband sing --score examples/kitchen_patter/melody.json --voice tuesday -o kitchen_patter.wav
```

Tuesday is the voice it was written for; Orpheus is the other that carries it well. Add `--metadata kitchen_patter.json`
to see how the runtime split it into phrases.

## The score

`melody.json` holds the lyric, the tempo, the style and 71 notes. The format is the one described in the Ghost in the
Machine tutorial (`[label, pitch, beats]`, syllables numbered `word~1 word~2`, rests as `["-", "rest", beats]`). What is
different here is the density:

| | This score | Ghost in the Machine verse | Reference pop tracks (vocal + guitar) |
| --- | --- | --- | --- |
| Tempo | 112 | 104 | 112–129 |
| Syllables per second | 5.2 | 2.3 | 3.0–4.7 |
| Syllables per beat | 2.8 | 1.3 | 1.5–2.5 |
| Typical note | sixteenth (0.25 beat) | eighth to half | sixteenth to eighth |

The reference column comes from measuring three vocal-and-guitar pop tracks (tempo, syllable rate, note lengths); the
densest of them sits at 4.7 syllables a second on sixteenths, and this score was written a little past it on purpose, to
find out where the model stops keeping up.

Rules that made it work, in addition to the ones in the other tutorial:

- **One note per syllable, stressed syllables a little longer.** Every syllable gets its own sixteenth; the stressed one
  of a two-syllable word (`kitchen~1`, `talking~1`, `critic~1`) gets an eighth. That is the rhythm of spoken English laid
  on a grid, and it is what keeps the words intelligible at this speed. Holding an unstressed syllable is what produced
  "ma-mah-chine" in the other piece; at sixteenth speed the same mistake swallows the word entirely.
- **Breaths as rests, every four bars or so.** A sixteenth rest after "listen" and a whole beat after "missing": the runtime
  cuts phrases at rests of a quarter second or more, so the quarter-beat rest at 112 bpm (0.13 s) stays inside the phrase
  as a breath while the one-beat rest (0.54 s) starts a new phrase. A phrase of this density should stay under about
  twelve seconds; the runtime's limit is ten seconds of notes per phrase.
- **A narrow melody.** C4 to G4, mostly steps and repeated notes, with the lift to G4 on the stressed syllable of the line's
  key word (`kitchen`, `toaster`, `critic`). Patter lives on rhythm, not on range; wide leaps at this speed cost clarity
  and the reference tracks move by step four times out of five.
- **Clusters still need room.** "fridge is in the" is the hardest run in the score and the one place a voice may slur;
  `fridge` gets an eighth for that reason. Where a word ends in a consonant cluster and the next begins with one, lengthen
  the first.
- **`pop` style, no `rap` style.** The flow was trained with a rap delivery row, but the release exposes `pop` and
  `ballad` only; `pop` is the right style here. The sung delivery at this density already reads as talk-singing.

## What it measures

Rendered with the singing release at its default (4 Euler steps, no guidance) through the full-precision reference bundle,
Whisper large-v3 recovers 96 % of the words (precision; the misses are consonant clusters on sixteenths, where a human
singer would slur too) and the sung pitch sits a median of 10 cents from the score with 99 % of notes within 100 cents.
The first singing candidate at its heavier setting (8 Heun steps with guidance) scored the same words and pitch on this
score at eight times the flow cost, which is the comparison the distillation was checked against.
