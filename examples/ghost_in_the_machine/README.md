# Ghost in the Machine: a spoken-and-sung demo track, step by step

A short announcement piece made entirely with Scylla's Band: narration and dialogue from the `speak` path, verses and a
chorus from the `sing` path, merged into one track by `make_track.py`. This folder is a living tutorial: every file is
the real one used for the piece, and this README records each step with the reasoning behind it, including the things
that went wrong on the way and how they were fixed, so you can take the same route for your own piece.

Voices: Thistle narrates, Tuesday sings the verse and leads the chorus with Orpheus doubling her an octave below, all
three sing the closing line, and Scylla has a complaint. About 100 seconds.

Files:

| File | What it is |
| --- | --- |
| `script.json` | The piece in order: spoken cues and sung cues, with voices, delivery, scores, gains and gaps |
| `melodies/*.json` | One melody score per sung section (verse 1, verse 2, chorus, closing tag) |
| `make_track.py` | Renders the cues, aligns and sums the doubled voices, places everything on a timeline, writes the WAV and a cue sheet |
| `check_words.py` | Optional: transcribes each rendered cue and compares it with its words |

## Step 1. Render it

```bash
python -m scyllasband download --flavor litert        # a bundle that sings (the singing release); speech works on any bundle
python examples/ghost_in_the_machine/make_track.py    # out/ghost_in_the_machine.wav and out/ghost_in_the_machine.cues.json
```

Every cue is rendered once into `out/cues/<name>.wav` and reused, so after editing one cue re-render only that cue:

```bash
python examples/ghost_in_the_machine/make_track.py --only chorus
python examples/ghost_in_the_machine/make_track.py --force                       # everything again
python examples/ghost_in_the_machine/make_track.py --bundle <dir> --backend onnx  # another bundle or backend
```

On a desktop CPU the sung cues take about real time to render (the singing decoder is the heavy part); the spoken cues
are near instant. Why cache per cue: a piece like this is written by listening and changing one line at a time, and a
full re-render on every change would make that slow.

## Step 2. Decide the shape before writing notes

The brief: a ghost stuck in a small model, learning to sing; fun, a few hints at what the model does, kept light. The
plan that came out of it, and why:

- **Spoken frame, sung middle.** Narration opens and closes, the song sits in the middle, dialogue breaks the song once.
  Spoken lines are where the jokes and the capability hints live (six languages, no cloud, fourteen voices); the sung
  lines carry the feeling. Mixing the two is the point of the piece, so the structure makes the switch obvious.
- **One singer per section, then everyone on the last line.** Tuesday takes verse 1 and the chorus, Orpheus verse 2,
  both the chorus (Tuesday lead, Orpheus an octave below), all three the closing tag. Each voice gets a solo moment
  before they are heard together.
- **A pop tempo.** 104 bpm in 4/4: the middle of the mid-tempo cluster of the pop reference tracks, fast enough to feel
  modern, slow enough that eighth-note syllables stay clear.
- **Scylla's cameo as spoken dialogue, not song.** A complaint lands better spoken, with an intense delivery, and it is a
  chance to show the delivery controls doing something.

## Step 3. Write the melody scores

A score is one JSON file per section (`melodies/*.json`): the lyric, the tempo, the style and the notes in order.

```json
{
  "text": "Ghost in the machine learning how to sing",
  "bpm": 104,
  "style": "pop",
  "notes": [
    ["Ghost", "E4", 1.0], ["in", "D4", 0.5], ["the", "C4", 0.5], ["machine~1", "E4", 0.5], ["machine~2", "G4", 2.5],
    ["-", "rest", 3.0],
    ["learning~1", "G4", 0.5], ["learning~2", "E4", 0.5], ["how", "D4", 0.5], ["to", "C4", 0.5], ["sing", "D4", 3.0]
  ]
}
```

The format:

- Each note is `[label, pitch, beats]`. Beats are quarter notes at `bpm` (0.5 = an eighth, 0.25 = a sixteenth). Pitches
  are note names with an octave (`C4`, `F#3`, `Bb3`); a rest is `["-", "rest", beats]`.
- A word on one note is a bare label. A word over several notes is numbered in order: `machine~1 machine~2` (one note per
  syllable, or a syllable held across pitches). The labels in order must spell `text`.
- `style` is `pop` (the sung delivery) or `ballad` (slower, longer notes).

How the melodies were written, and the rules learned along the way:

- **Two-bar lines, a rest between lines.** Lines are separated by rests of a quarter second or more; the runtime renders
  each run between such rests as its own phrase (up to about ten seconds), so a rest between lines is required and the
  gap you write is the gap you hear. Lines here are eight beats including their rest.
- **Verses stepwise and narrow, chorus lifted.** The verses move mostly by step over A3 to E4 with the stressed word on a
  longer note; the chorus hook jumps to G4 and A4 and repeats the title line, which is what makes it the chorus.
- **Stressed syllable on the long note.** The first draft held `machine~1` (the unstressed "ma") for a whole beat, and
  every voice sang "ma-mah-chine". With "ma" on an eighth and "chine" on the long note the word is clear. The same goes
  for any word whose first syllable is a schwa.
- **Give consonant clusters room.** "six" on an eighth before "languages" came out as "sick": the final "s" had nowhere to
  go. On a quarter note it is clear. When a word ends in a consonant cluster and the next starts with a consonant, give
  the first word a longer note.
- **No rest inside a line unless you want a new phrase.** The chorus's third line (`no cloud no strings`) has no rest
  between its halves on purpose; a rest would split it into two phrases with a lead-in of their own.
- **Write in any octave, then transpose into the voice.** The runtime moves the whole score by octaves into the voice's
  sung range; `transpose` (semitones, set per cue in the script or with `sing --transpose`) sets the key first. A voice
  sings most accurately within about five semitones of its centre, which the bundle lists per voice
  (`controls.delivery_modes.singing.voice_midi_center` in `manifest.json`; Tuesday and Thistle sit near A#3/B3, Orpheus
  near D3). These scores are written around C4 and transposed down four semitones, which puts Tuesday at her centre and
  Orpheus one octave below her.

A quick way to hear a score before it goes into the piece:

```bash
python -m scyllasband sing --score examples/ghost_in_the_machine/melodies/chorus.json --voice tuesday --transpose -4 -o chorus.wav
```

The runtime also silences each phrase's lead-in up to 40 ms before its first note and fades it out a quarter second after
its last. The model sometimes voices a breath or a stray syllable there (it sounded like a short "what?" before one chorus
line), and that is not in the score, so it is gated out.

## Step 4. Write the spoken cues

```json
{"name": "intro_1", "type": "speak", "voice": "thistle", "language": "en_us",
 "delivery": {"intensity": -0.5, "pace": -0.3},
 "text": "Somewhere inside a very small model, there is a ghost. It learned to talk first. Then it got ideas."}
```

- One voice, one line, the usual delivery controls (`intensity`, `pace`, `pitch`, each from -2 to 2). Thistle narrates
  calm and a little slow; Scylla's complaint is intense and quick; her "Oh. Carry on." drops back to neutral.
- **Write narration as flowing sentences.** The first closing line was "Scylla's Band. It talks. Now it sings. On your
  phone, with no cloud in sight." Each full stop became a pause, and the voice breathed into the pauses, which reads as
  noise in a mix (measured: peaks of 0.06 to 0.18 inside the pauses). The same words as one sentence with commas measured
  0.012 in its one pause. Short sentences are for dialogue, where a breath is in character.
- `gap` is the silence after the cue; the script's `gap` is the default. Leading and trailing silence of every render is
  trimmed so the gaps are what you wrote.

## Step 5. The chorus: two voices in unison

The chorus is Tuesday with Orpheus doubling her an octave below. Octave unison is the classic pop chorus thickener, and
because both takes follow the same score their pitches and note boundaries line up by construction. The sung cue:

```json
{"name": "chorus", "type": "sing", "voices": ["tuesday", "orpheus"], "gains": [1.0, 0.6], "align": "lead",
 "score": "melodies/chorus.json", "transpose": -4, "seed": 5, "gap": 0.6}
```

What happens, and why each part is there:

1. **Render the score once per voice.** Same score, same transpose, same seed. The runtime gives both takes the same
   phone plan (it comes from the score), so vowels and note boundaries coincide.
2. **Align the double onto the lead, note by note.** The model decides for itself where inside a note the consonant
   falls, and two voices differ by up to about 150 ms there. Summed as rendered, every consonant doubled
   ("ma-ma-chine"). So for each note the double's onset is matched to the lead's: the RMS envelopes of both takes are
   compared over the onset region (80 ms before the note to 400 ms into it) at shifts up to 100 ms, the best-correlated
   shift is kept (limited to a 60 ms change from the previous note so a weak match cannot throw a note out), and the
   double is rebuilt from its shifted note segments with 12 ms crossfades at the note boundaries, where the consonants
   are. Vowels keep their pitch and the voice keeps its timbre; only the timing moves. The shifts found are written to the
   cue's metadata (`out/cues/chorus.json`, `alignment`), so you can see what it did.
3. **Mix with the lead on top.** Lead at 1.0, double at 0.6: the lower voice colours the chorus without competing for
   the words. At equal gains the lyric blurred even when aligned.
4. **Normalize the sum** to a 0.95 peak, like every cue.

The closing tag does the same with three voices (Thistle lead, Tuesday and Orpheus at 0.5). `"align": "none"` turns the
alignment off, which is only useful for long open vowels where the flam cannot happen.

## Step 6. Put it on a timeline

`make_track.py` walks the cues in order: render (or load from the cache), trim leading and trailing silence, place at the
cursor, advance the cursor by the cue's length plus its gap. The whole track is then normalized to a 0.9 peak. Nothing
overlaps; the gaps are the script's. Two outputs:

- `out/ghost_in_the_machine.wav`: mono 24 kHz, the whole piece.
- `out/ghost_in_the_machine.cues.json`: every cue with its start and end on the track, its text, and for sung cues the
  phrase boundaries, every note's start and end, and the alignment shifts. A video edit or a mix (an instrumental made
  elsewhere, placed under the sung sections) keys off this.

## Step 7. Check, listen, change one thing, repeat

Before listening through, run the transcript check (it needs `faster-whisper`):

```bash
python examples/ghost_in_the_machine/check_words.py
```

It prints, per cue, how much of the transcript matches the words given and how much of the text was heard. A low score
points at a word to look at; the transcript shows what came out instead. This is how "ma-mah-chine", "sick languages"
and the chorus flam were found before anyone had to hunt for them by ear, and it is how the fixes were confirmed. Whisper
has its own habits (it writes "14" for "fourteen" and "syla" for "Scylla"), so read the transcript before trusting a low
score on a name.

Then listen, note the time of anything that sounds off, find the cue in the cue sheet, change one thing (a note length,
a gain, a delivery value, a seed), and re-render that cue with `--only`. Changing `seed` gives a different performance of
the same score, which is often all a line needs.

## What is and is not in the box

The voices are shown doing what they do: speaking a chosen delivery, singing from a score, singing together, with no
cloud in sight. The instrumental and the video are made with other tools around the cue sheet. Note that the model never
free-sings: every sung note here was written down first, and that is by design.
