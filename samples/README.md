# Scylla's Band Voice Samples

[Open the voice gallery](https://lowkeytea.github.io/scyllasband/) to play every clip in the browser.

The clips use the default LiteRT bundle (release `v2-20261005`), heun sampling with 8 steps and
seed `2027`. To regenerate them after downloading the model:

```bash
python samples/generate_gallery.py --bundle scyllasband/models/litert
```

The script reads `samples/gallery_script.json` and writes `docs/index.html`, `docs/audio/` and this file.

## Long form

- [Single-voice narration](https://lowkeytea.github.io/scyllasband/#narration) (5:08), from `data/test_document.txt`
- [Multi-voice multilingual dialogue](https://lowkeytea.github.io/scyllasband/#dialogue) (1:44), from `data/walkthrough_demo.txt`

## Languages

Each voice reads the same line in its own English and in five more languages.

| Voice | English | Spanish | Italian | French | German | Vietnamese |
| --- | --- | --- | --- | --- | --- | --- |
| Ariadne | [Listen](https://lowkeytea.github.io/scyllasband/#ariadne-en) | [Listen](https://lowkeytea.github.io/scyllasband/#ariadne-es) | [Listen](https://lowkeytea.github.io/scyllasband/#ariadne-it) | [Listen](https://lowkeytea.github.io/scyllasband/#ariadne-fr) | [Listen](https://lowkeytea.github.io/scyllasband/#ariadne-de) | [Listen](https://lowkeytea.github.io/scyllasband/#ariadne-vi) |
| Felix | [Listen](https://lowkeytea.github.io/scyllasband/#felix-en) | [Listen](https://lowkeytea.github.io/scyllasband/#felix-es) | [Listen](https://lowkeytea.github.io/scyllasband/#felix-it) | [Listen](https://lowkeytea.github.io/scyllasband/#felix-fr) | [Listen](https://lowkeytea.github.io/scyllasband/#felix-de) | [Listen](https://lowkeytea.github.io/scyllasband/#felix-vi) |
| Gwen | [Listen](https://lowkeytea.github.io/scyllasband/#gwen-en) | [Listen](https://lowkeytea.github.io/scyllasband/#gwen-es) | [Listen](https://lowkeytea.github.io/scyllasband/#gwen-it) | [Listen](https://lowkeytea.github.io/scyllasband/#gwen-fr) | [Listen](https://lowkeytea.github.io/scyllasband/#gwen-de) | [Listen](https://lowkeytea.github.io/scyllasband/#gwen-vi) |
| Ink | [Listen](https://lowkeytea.github.io/scyllasband/#ink-en) | [Listen](https://lowkeytea.github.io/scyllasband/#ink-es) | [Listen](https://lowkeytea.github.io/scyllasband/#ink-it) | [Listen](https://lowkeytea.github.io/scyllasband/#ink-fr) | [Listen](https://lowkeytea.github.io/scyllasband/#ink-de) | [Listen](https://lowkeytea.github.io/scyllasband/#ink-vi) |
| Max | [Listen](https://lowkeytea.github.io/scyllasband/#max-en) | [Listen](https://lowkeytea.github.io/scyllasband/#max-es) | [Listen](https://lowkeytea.github.io/scyllasband/#max-it) | [Listen](https://lowkeytea.github.io/scyllasband/#max-fr) | [Listen](https://lowkeytea.github.io/scyllasband/#max-de) | [Listen](https://lowkeytea.github.io/scyllasband/#max-vi) |
| Orpheus | [Listen](https://lowkeytea.github.io/scyllasband/#orpheus-en) | [Listen](https://lowkeytea.github.io/scyllasband/#orpheus-es) | [Listen](https://lowkeytea.github.io/scyllasband/#orpheus-it) | [Listen](https://lowkeytea.github.io/scyllasband/#orpheus-fr) | [Listen](https://lowkeytea.github.io/scyllasband/#orpheus-de) | [Listen](https://lowkeytea.github.io/scyllasband/#orpheus-vi) |
| Rex | [Listen](https://lowkeytea.github.io/scyllasband/#rex-en) | [Listen](https://lowkeytea.github.io/scyllasband/#rex-es) | [Listen](https://lowkeytea.github.io/scyllasband/#rex-it) | [Listen](https://lowkeytea.github.io/scyllasband/#rex-fr) | [Listen](https://lowkeytea.github.io/scyllasband/#rex-de) | [Listen](https://lowkeytea.github.io/scyllasband/#rex-vi) |
| Scylla | [Listen](https://lowkeytea.github.io/scyllasband/#scylla-en) | [Listen](https://lowkeytea.github.io/scyllasband/#scylla-es) | [Listen](https://lowkeytea.github.io/scyllasband/#scylla-it) | [Listen](https://lowkeytea.github.io/scyllasband/#scylla-fr) | [Listen](https://lowkeytea.github.io/scyllasband/#scylla-de) | [Listen](https://lowkeytea.github.io/scyllasband/#scylla-vi) |
| Stone | [Listen](https://lowkeytea.github.io/scyllasband/#stone-en) | [Listen](https://lowkeytea.github.io/scyllasband/#stone-es) | [Listen](https://lowkeytea.github.io/scyllasband/#stone-it) | [Listen](https://lowkeytea.github.io/scyllasband/#stone-fr) | [Listen](https://lowkeytea.github.io/scyllasband/#stone-de) | [Listen](https://lowkeytea.github.io/scyllasband/#stone-vi) |
| Tuesday | [Listen](https://lowkeytea.github.io/scyllasband/#tuesday-en) | [Listen](https://lowkeytea.github.io/scyllasband/#tuesday-es) | [Listen](https://lowkeytea.github.io/scyllasband/#tuesday-it) | [Listen](https://lowkeytea.github.io/scyllasband/#tuesday-fr) | [Listen](https://lowkeytea.github.io/scyllasband/#tuesday-de) | [Listen](https://lowkeytea.github.io/scyllasband/#tuesday-vi) |

## Delivery

Every voice speaks each line below in its own English, with the settings shown.

| Delivery | Settings | Line |
| --- | --- | --- |
| Neutral (default) | `energy=2,tension=2,valence=2,assertiveness=2,whisper=off` | Please put the small blue box beside the kitchen window, then close the door behind you. |
| Calm and soft | `energy=1.8,tension=1,valence=2.5,assertiveness=2.1,whisper=off` | Take a slow breath. There's no rush tonight; the rain will keep the city quiet for a while. |
| Assertive and quick | `energy=2.6,tension=2.4,valence=2.1,assertiveness=2.8,whisper=off` | Listen carefully. We leave at six, we stay together, and nobody goes back for the bags. |
| Joyful | `energy=2.4,tension=2,valence=2.8,assertiveness=2.2,whisper=off` | We actually did it! The whole street came out to watch, and nobody wanted the night to end! |
| Angry | `energy=2.7,tension=3.1,valence=1.3,assertiveness=2.5,whisper=off` | No. You don't get to walk in here and change the rules again. Not after everything. |
| Sad | `energy=1.7,tension=1.7,valence=1.8,assertiveness=2,whisper=off` | I kept the letter for years... I never opened it. Maybe I was afraid of what it would say. |
| Whisper | `energy=2,tension=2,valence=2,assertiveness=2,whisper=on` | Keep your voice down. The guard changes at four, and the side door is still unlocked. |
