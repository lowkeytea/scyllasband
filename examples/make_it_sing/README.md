# Make It Sing: a score from scratch

The scores from [Make It Sing](https://youtu.be/e6yHVbkVcTs), a three-minute video for anyone who has never written
music (also as a [Short](https://youtube.com/shorts/ZHL3FEunhlI)). It covers:
- what each part of a score means;
- what goes wrong when the values are bad;
- how to stretch one tune over a longer lyric.

Every voice and every sung note in the video is Scylla's Band, and every score it shows is in `melodies/`. The other
singing tutorials go further:
- [Ghost in the Machine](../ghost_in_the_machine/README.md): a whole piece, step by step.
- [Kitchen Patter](../kitchen_patter/README.md): fast talk-singing.
- Pieces built from score sources with `write_scores.py`: [Face Down](../face_down/README.md),
  [Bubble Wrap](../bubble_wrap/README.md) and [Thank You for Participating](../thank_you/README.md).

```bash
python -m scyllasband download
python -m scyllasband sing --score examples/make_it_sing/melodies/readme.json --voice gwen -o ring.wav
```

## A score

`readme.json` is the example from the main README:

```json
{
  "text": "give me a ring baby make me feel so",
  "bpm": 108,
  "style": "pop",
  "notes": [
    ["give", "C4", 0.5], ["me", "D4", 0.5], ["a", "E4", 0.5], ["ring", "G4", 1.0],
    ["-", "rest", 0.5],
    ["baby~1", "E4", 0.5], ["baby~2", "D4", 0.5],
    ["make", "C4", 0.5], ["me", "D4", 0.5], ["feel", "E4", 1.0], ["so", "C4", 1.5]
  ]
}
```

- **`text`**: the words to be sung.
- **`bpm`**: the beat, in beats per minute: the tap your foot finds. Bigger is faster; 108 is a comfortable pop speed.
- **`style`**: `pop`, or `ballad` for slow songs.
- **`notes`**: one row per word (or per syllable), each `[word, pitch, beats]`.
  - **Pitch** is how high. It is a letter and a number. The letters climb C D E F G A B and then start again one
    round up, and the number counts the rounds. C4 is the middle of a piano.
  - **Beats** are how long: 1 is one tap, 0.5 half a tap, 2 is held for two.
- **Rests:** `["-", "rest", 0.5]` is a breath. Put one between lines. A rest of a quarter second or more starts a new
  phrase.
- **Long words:** a word over several notes is split with a tilde and a number, `baby~1 baby~2`. The parts are numbered
  in order, with no rest between them.

`stairs.json` sings the note names up a scale. It is spelled as it sounds, `see dee ee ef gee eh bee see`, because the
G2P reads "eff" as "eff eff" and "ay" as "eye". It is written from C4 to C5. Gwen sings it a round lower, because the
runtime moves every song by whole rounds (octaves) to where each voice sings best. So write in any octave.

## What breaks it

Some bad scores are refused before anything is sung. The runtime's message names the note at fault:

| Score | The runtime says |
| --- | --- |
| `err_octave.json` (`"G"`, no number) | Unknown note 'G': use a pitch name with octave (C4, F#3, Bb3) or rest |
| `err_zero.json` (0 beats) | Note 4 ('ring') must last a positive number of beats |
| `err_word.json` (one word too many in `text`) | Score text does not match the note labels: … |

Other scores are valid but sing badly. In the video, Whisper checked each take, and these are its transcripts:

| Score | What goes wrong | Whisper heard | The fix |
| --- | --- | --- | --- |
| `crammed.json` | "unbelievable", five syllables, on one sixteenth | "That's it." | `unbelievable.json`: one note per syllable, `un~1 be~2 lie~3 va~4 ble~5` |
| `fast.json` | the README score at bpm 400 | "Come here and beat me with your soul." | the same score at 108 |
| `leaps.json` | C3 and C5 in turn, every note | the right words, but the pitch misses by more than half an octave | small steps, near the middle |

Two more rules from making the video:
- **Consonant pile-ups:** a short word that ends in a pile of consonants ("six", "words") needs a longer note.
- **Long notes:** hold a single note for a bar or two. Much longer, and the voice starts making things up.

## A longer song

`verse.json` and `chorus.json` are one song, "Sing It Once": 120 bpm, every line two bars with a rest at the end.
`count.json` is Max's count-in. The lyric is the lesson:
- **Reuse a line's tune with new words.** Lines 1 and 3 of the verse share one tune, and lines 2 and 4 share another.
- **End where you started.** The song starts on C4 and its last line comes home to C4, which is what makes it sound
  finished.
- **Safe notes:** the whole song uses only C, D, E, G and A, five notes that always get along.
- **Choruses:** the chorus sits higher than the verse and repeats its first line.

The song is written in C. `--transpose -6` moves it to where Gwen, Thistle and Max all sing comfortably, and the
runtime puts Max a round below the other two by itself:

```bash
python -m scyllasband sing --score examples/make_it_sing/melodies/verse.json --voice gwen --transpose -6 -o verse.wav
for v in thistle gwen max; do
  python -m scyllasband sing --score examples/make_it_sing/melodies/chorus.json --voice $v --transpose -6 -o chorus_$v.wav
done
```

In the video, the three chorus takes are lined up note by note and mixed, as `make_track.py` does in Ghost in the
Machine. The backing was made separately: drums generated in ACE Studio around the finished vocals, and string parts
written for the song and played by ACE Studio's AI instruments.
