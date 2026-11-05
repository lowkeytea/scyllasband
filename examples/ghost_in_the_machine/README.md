# Ghost in the Machine: a spoken-and-sung demo track

A short announcement piece made entirely with Scylla's Band: narration and dialogue from the `speak` path, verses and a
chorus from the `sing` path, merged into one track by `make_track.py`. It walks through writing a melody score, writing a
cue script that mixes spoken lines with sung sections, and rendering the result with a cue sheet (for a video edit or an
instrumental mix made elsewhere).

Voices: Thistle narrates, Tuesday and Orpheus sing (an octave apart on the chorus, which is what unison in two registers
sounds like), and Scylla has a complaint. About 95 seconds.

## 1. Render it

```bash
python -m scyllasband download --flavor litert        # a bundle that sings (the singing release); speech works on any bundle
python examples/ghost_in_the_machine/make_track.py    # writes out/ghost_in_the_machine.wav and out/ghost_in_the_machine.cues.json
```

Every cue is rendered once into `out/cues/<name>.wav` and reused; after editing one cue, re-render only that cue:

```bash
python examples/ghost_in_the_machine/make_track.py --only chorus
python examples/ghost_in_the_machine/make_track.py --force                       # everything again
python examples/ghost_in_the_machine/make_track.py --bundle <dir> --backend onnx  # another bundle or backend
```

On a desktop CPU the sung cues take about real time to render (the singing decoder is the heavy part); the spoken cues are
near instant.

## 2. Write a melody score

A score is one JSON file per section (`melodies/*.json`): the lyric, the tempo, the style and the notes in order.

```json
{
  "text": "Ghost in the machine learning how to sing",
  "bpm": 104,
  "style": "pop",
  "notes": [
    ["Ghost", "E4", 1.0], ["in", "D4", 0.5], ["the", "C4", 0.5], ["machine~1", "E4", 1.0], ["machine~2", "G4", 2.0],
    ["-", "rest", 3.0],
    ["learning~1", "G4", 0.5], ["learning~2", "E4", 0.5], ["how", "D4", 0.5], ["to", "C4", 0.5], ["sing", "D4", 3.0]
  ]
}
```

- Each note is `[label, pitch, beats]`. Beats are quarter notes at `bpm` (0.5 = an eighth, 0.25 = a sixteenth). Pitches
  are note names with an octave (`C4`, `F#3`, `Bb3`); a rest is `["-", "rest", beats]`.
- A word on one note is a bare label. A word over several notes is numbered in order: `machine~1 machine~2` (one note per
  syllable, or a syllable held across pitches). The labels in order must spell `text`.
- `style` is `pop` (the sung delivery) or `ballad` (slower, longer notes).
- Lines are separated by rests of a quarter second or more: the runtime renders each line as its own phrase (up to about
  ten seconds), so a rest between lines is required, and the gap you write is the gap you hear.
- Write in any octave. The runtime moves the whole score by octaves into the voice's sung range; `transpose` (semitones,
  in the cue script or `sing --transpose`) sets the key first. A voice sings most accurately within about five semitones of
  its centre, which the bundle lists per voice (`controls.delivery_modes.singing.voice_midi_center` in `manifest.json`;
  Tuesday and Thistle sit near A#3/B3, Orpheus near D3). The scores here are written around C4 and transposed down four
  semitones, which puts Tuesday at her centre and Orpheus an octave below.

How these were written: a modern pop feel at 104 bpm in 4/4 (the mid-tempo cluster of the reference tracks), two-bar
lines, syllables mostly on eighths with the stressed word on a longer note, verses stepwise in a narrow range (A3 to E4)
and the chorus hook lifting to G4/A4 with the title line repeated. The chorus's third line (`no cloud no strings`) has no
rest between its two halves on purpose: a rest would split it into two phrases.

Put the stressed syllable on the long note. The first draft held `machine~1` (the unstressed "ma") for a whole beat and
every voice sang "ma-mah-chine"; with `["machine~1", "E4", 0.5], ["machine~2", "G4", 2.5]` the word is clear. The same
goes for any word whose first syllable is a schwa: give it an eighth and let the stressed syllable carry the pitch.

The runtime silences each phrase's lead-in up to 40 ms before its first note and fades it out a quarter second after its
last, because the model sometimes voices a breath or a stray syllable there that is not in the score.

A quick way to check a score before rendering it: `python -m scyllasband sing --score melodies/chorus.json --voice tuesday
--transpose -4 -o chorus.wav`.

## 3. Write the cue script

`script.json` is the piece in order. Two cue types:

```json
{"name": "intro_1", "type": "speak", "voice": "thistle", "language": "en_us",
 "delivery": {"intensity": -0.5, "pace": -0.3}, "text": "Somewhere inside a very small model, there is a ghost."}

{"name": "chorus", "type": "sing", "voices": ["tuesday", "orpheus"], "gains": [1.0, 0.8],
 "score": "melodies/chorus.json", "transpose": -4, "seed": 5, "gap": 0.6}
```

- `speak`: one voice, one line, the usual delivery controls (`intensity`, `pace`, `pitch`, each from -2 to 2). Write
  narration as flowing sentences: a run of very short sentences ("It talks. Now it sings.") puts a pause after each, and
  the voice breathes into those pauses, which reads as noise in a mix. One sentence with commas is clean.
- `sing`: one or more voices on one score. With several voices the score is rendered once per voice and the takes are
  summed (they share the score's timeline, so they line up), each scaled by its gain: unison, each voice in its own
  register. Each voice times its consonants a little differently (up to about 150 ms on a syllable), so a unison sum
  flams on consonant-heavy words; this piece keeps every sung section to one voice and passes the title line between
  voices instead (Tuesday's chorus, Orpheus's answer, Thistle's tag). Use several voices on a cue for long open vowels, or
  as a quiet double under a lead. `seed` fixes the take; change it for a different performance of the same score.
- `gap` is the silence after the cue (the script's `gap` is the default). Leading and trailing silence of each render is
  trimmed so the gaps are what you wrote.

## 4. What comes out

- `out/ghost_in_the_machine.wav`: the whole track, mono 24 kHz, peak-normalized.
- `out/ghost_in_the_machine.cues.json`: every cue with its start and end on the track, its text, and for sung cues the
  phrase boundaries. This is what a video edit or a mix (an instrumental made elsewhere, placed under the sung sections)
  keys off.
- `out/cues/`: the individual renders, for auditioning or for a multitrack mix.

The voices are shown doing what they do (talk in a chosen delivery, sing from a score, together) without a cloud in sight;
anything beyond that, such as the instrumental or the video, is made with other tools around the cue sheet.
