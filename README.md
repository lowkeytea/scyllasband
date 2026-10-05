# Scylla's Band

Scylla's Band is a local text-to-speech runtime for ten managed voices. It predicts phone durations, generates continuous acoustic latents with rectified flow, and decodes speech at 24 kHz through an acoustic adapter and Vocos.

Delivery is controlled with **energy, tension, valence and assertiveness**, each from **0 to 4** with **2 as neutral**, plus **whisper on/off**.

## Samples and community

- **[Listen to the voice gallery](https://lowkeytea.github.io/scyllasband/)**: every voice in all six languages, seven delivery styles per voice, a long-form narration and a ten-voice dialogue.
- The [sample index](samples/README.md) lists every clip with the settings used, and `samples/generate_gallery.py` regenerates them.
- Join the [Scylla's Band Discord](https://discord.gg/cNdBuM3tS) for release updates, help, and community discussion.

## Highlights

- Ten voices in seven locales: American and British English, Spanish, Italian, French, German and Vietnamese.
- Delivery controls for energy, tension, valence and assertiveness on 0–4 scales, plus whisper, combinable into calm, assertive, joyful, angry or sad delivery.
- Runs locally on CPU. The default LiteRT bundle synthesizes about 14× faster than real time on an 8-thread desktop CPU.
- LiteRT and ONNX Runtime bundles with INT8 weights; every latent-length bucket shares one set of weights.
- Long text is spoken sentence by sentence, with neighbouring sentences as context and acoustic continuity from one sentence to the next, and can be streamed as it is generated.
- Multi-voice, multilingual dialogue with inline `[voice:language:delivery]` tags.
- A C++ runtime with a C API (`libscyllasband`) for the same bundles.
- 24 kHz mono output.

## Quick start

```bash
git clone https://github.com/lowkeytea/scyllasband
cd scyllasband

python3 -m scyllasband download
python3 -m scyllasband list-voices
python3 -m scyllasband speak --voice scylla -o hello.wav "Hello from Scylla's Band."
```

The first command offers to create a virtual environment in `.venv` and install the runtime into it, then carries on. Later commands run from the checkout use that environment automatically, with or without activating it; `python3 speak.py` and `python3 groupSpeak.py` behave the same way. `--backend onnx` adds ONNX Runtime to the environment the first time it is used.

Requirements: Python 3.10–3.14 on macOS with Apple silicon, Linux (x86_64 or aarch64) or Windows (x86_64). To set the environment up yourself instead:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .                   # pip install -e ".[onnx]" for ONNX Runtime
```

The model is downloaded from [`spybyscript/scyllasband`](https://huggingface.co/spybyscript/scyllasband), pinned to release `v2-20261005`, into `scyllasband/models/litert`. Voice conditioning is embedded in the graphs; no separate voice files are needed.

## Delivery controls

| Control | Request | Default |
| --- | --- | --- |
| Energy | 0–4 or `auto` | 2 |
| Tension | 0–4 or `auto` | 2 |
| Valence | 0–4 or `auto` | 2 |
| Assertiveness | 0–4 or `auto` | 2 |
| Whisper | `on`, `off`, or `auto` | `off` |

With every control at 2, the default, voices speak in a calm, conversational delivery with a neutral tone. Moving a control away from 2 changes the delivery:

| Control | Toward 0 | Toward 4 |
| --- | --- | --- |
| Energy | Lower, darker voice with more pauses between phrases | Higher, brighter, more projected voice |
| Tension | Slightly looser and steadier | Tighter, with wider pitch swings |
| Valence | Flatter pitch, a little slower, longer pauses | Livelier pitch, a little quicker, shorter pauses |
| Assertiveness | Hesitant, with longer pauses between phrases | Faster and clipped with few pauses; the strongest control over pace |

`whisper=on` whispers: quieter and breathy.

Combining the controls gives familiar deliveries:

| Delivery | Settings |
| --- | --- |
| Calm and soft | `energy=1.2,tension=0.8,valence=2.6` |
| Assertive and rapid-fire | `energy=2.6,tension=2.2,assertiveness=3.6` |
| Joyful | `energy=3.2,valence=3.6` |
| Angry | `energy=3.4,tension=3.4,valence=0.8,assertiveness=3.4` |
| Sad | `energy=1,tension=1.6,valence=0.8,assertiveness=1.6` |
| Whispered | `whisper=on` |

The [voice gallery](https://lowkeytea.github.io/scyllasband/#delivery) plays each of these for all ten voices with the exact settings used. How far a control moves the delivery varies with voice, language and text, so start near 2 and listen as you move toward the extremes.

```bash
python -m scyllasband speak --voice ariadne \
    --delivery energy=3.2,valence=3.6 \
    -o joyful.wav "We actually did it! The whole street came out to watch!"

python -m scyllasband speak --voice ink --language en_gb \
    --delivery whisper=on -o whisper.wav "Keep your voice down."
```

Unspecified controls use the neutral default of 2 (whisper off). `--delivery auto` leaves every control unspecified, and `energy=auto` leaves only that one unspecified; an unspecified control is passed to the model as absent, which is different from requesting 2.

## CLI

```bash
python -m scyllasband download                  # LiteRT bundle; --flavor onnx or --flavor all for others
python -m scyllasband validate-bundle
python -m scyllasband list-voices
python -m scyllasband normalize-text --language es "Cuesta 12,50 € el 22/05/2026."

python -m scyllasband speak --voice gwen --language en_us \
    --delivery energy=1.2,tension=0.8,valence=2.6 -o calm.wav "Take a slow breath."
python -m scyllasband speak --voice rex --file data/test_document.txt --metadata story.json -o story.wav
python -m scyllasband group-speak --file data/walkthrough_demo.txt --pause-ms 250 -o dialogue.wav
python -m scyllasband stream --voice scylla --file data/test_document.txt -o chunks/
python -m scyllasband plan --voice scylla "Print the sentence plan as JSON."
```

`python3 speak.py ...` and `python3 groupSpeak.py ...` are shortcuts for `speak` and `group-speak`. An installed package also provides `scyllasband`, `scyllasband-speak` and `scyllasband-group-speak` commands.

| Option | Default | Effect |
| --- | --- | --- |
| `--voice`, `--language` | `scylla`, the voice's English | Voice and locale; `en` picks the voice's own English |
| `--delivery` | all 2, whisper off | `energy=..,tension=..,valence=..,assertiveness=..,whisper=on\|off`, `neutral` or `auto` |
| `--speed` | 1.0 | Speaking rate; 1.2 is 20% faster |
| `--steps`, `--sampler` | 8, `heun` | Flow sampling. Heun evaluates the model twice per step and Euler once, so `--sampler euler` is about twice as fast at the same step count; fewer steps are faster still. Listen when trading quality for speed |
| `--seed`, `--temperature` | random, 1.0 | Repeatable output; scale of the sampling noise |
| `--threads` | backend default | CPU threads per graph |
| `--backend`, `--bundle` | installed LiteRT bundle | Choose another installed bundle |
| `--pause-ms` | 0 | Extra silence where the voice, language or delivery changes |
| `--metadata` | none | Write phones, durations and timings as JSON |
| `--no-normalize-text` | off | Skip spoken-text normalization for text that is already normalized |

## Backends

| Bundle | Download | Runtime package |
| --- | --- | --- |
| LiteRT (default) | `python -m scyllasband download` | `ai-edge-litert` (installed with this package) |
| ONNX Runtime | `python -m scyllasband download --flavor onnx` | `onnxruntime`, installed the first time you use `--backend onnx` (or `pip install -e ".[onnx]"`) |

Both bundles hold INT8 transformer weights and the G2P model in a matching precision; the LiteRT bundle also quantizes the vocoder, which makes it the smaller and faster download. The backend follows the bundle; `--backend` and `--bundle` select another installed bundle. `--threads` sets the CPU threads per graph. Sampling uses Heun with 8 steps by default (`--sampler`, `--steps`).

## Python API

```python
from scyllasband import ScyllasBandRuntime, SynthesisRequest

runtime = ScyllasBandRuntime.from_bundle("scyllasband/models/litert")
result = runtime.synthesize(SynthesisRequest(
    text="It is a real pleasure to meet you.",
    voice_id="ariadne",
    language="en_us",
    delivery={"energy": 2.3, "tension": 1.7, "valence": 2.6,
              "assertiveness": 2.0, "whisper": "off"},
    seed=2027,
))
# result.audio (float32 numpy array), result.sample_rate, result.metadata
```

Reuse the runtime across requests; `runtime.warmup()` creates the sessions before interactive use. `runtime.synthesize_stream(records)` yields one audio chunk per sentence as it is ready, and `runtime.plan_records(records)` returns the sentence plan.

## Long form and dialogue

Text is spoken one sentence at a time. Each sentence sees its neighbours as context and continues from the acoustics of the sentence before it, and consecutive sentences are decoded as one continuous waveform, so long text needs no chunking settings.

```bash
python -m scyllasband speak --voice ariadne --file story.txt \
    --delivery energy=2.2,valence=2.3 --metadata story.json -o story.wav

python -m scyllasband stream --voice scylla --file story.txt -o chunks/

python -m scyllasband group-speak --file dialogue.txt -o dialogue.wav
```

A dialogue file sets the voice, language and controls per tagged span:

```text
[ariadne:en_us:energy=2.3,valence=2.5] Good to see you. [es] Me alegra verte.
[rex:en_us:tension=2.6,assertiveness=2.5] We should get going.
[ink:en_gb:whisper=on] Keep this between us.
```

Language tags keep the active delivery request. A new delivery tag replaces it, with unspecified coordinates reset to neutral. `--pause-ms` adds silence where the voice, language or delivery changes.

## Voices and languages

| Voices | English dialect |
| --- | --- |
| Ariadne, Felix, Gwen, Max, Rex, Scylla, Stone | `en_us` |
| Ink, Orpheus, Tuesday | `en_gb` |

All ten also support `es`, `it`, `fr`, `de`, and `vi`: 60 trained voice/locale pairs. `en` selects the voice's own English dialect. The runtime rejects unsupported voice/dialect combinations.

## Text handling

- Spoken-text normalization expands numbers, decimals, dates, times, currency, percentages, ordinals, fractions, initialisms and common abbreviations for each language.
- The Scylla's Band G2P model in the bundle converts each phrase to phones; punctuation is kept as pause and intonation cues, and word boundaries become optional silences.
- Text is split into sentences after normalization; a sentence too long for one pass is split again at clause punctuation.

`--no-normalize-text` skips normalization when the text is already in spoken form.

## How it works

```text
text
  -> spoken-text normalization
  -> Scylla's Band G2P: phones, word boundaries and punctuation cues
  -> sentence plan; each sentence also sees the phones of its neighbours
  -> duration predictor (voice, language, delivery, neighbouring text) -> frames per phone
  -> timing events per frame: word starts, punctuation, sentence type
  -> rectified-flow latent generator, continuing from the previous sentence's latents
  -> 24-channel acoustic latents
  -> acoustic adapter + Vocos decoder at 24 kHz, decoded with the previous sentence as left context
  -> one continuous waveform, or one chunk per sentence when streaming
```

## Bundle layout

```text
scyllasband/models/litert/                scyllasband/models/onnx/
  manifest.json                             manifest.json
  litert/                                   onnx/
    g2p.tflite                                g2p/model.onnx, tokenizer.json, phoneme_dict.json
    duration_predictor.tflite                 components/
    vector_context_encoder.tflite               duration_predictor.onnx
    vector_estimator.tflite  (one signature     vector_context_encoder.onnx
    vocoder.tflite            per bucket)       vector_estimator[_128..._768].onnx
  assets/                                       vocoder[_128..._768].onnx
    voices.json, languages.json                 shared_weights.bin, shared_weights.json
    phone_vocab.json, fixed_references.json   assets/  (same as LiteRT)
    g2p/  tokenizer, normalization, overrides
```

Latent-length buckets are 128, 256, 384, 512, 768 and 1024 frames (about 2.7 to 21.8 seconds); each sentence uses the smallest bucket that fits. The manifest declares the graph contract, inputs, buckets and controls, and the runtime validates it before loading.

## Native and mobile integration

`libscyllasband/` is the C++ runtime for the same bundles, built against LiteRT or ONNX Runtime; see [its README](libscyllasband/README.md). [Android](examples/android/README.md) and [iOS](examples/ios/README.md) examples use it.

## Validation and limitations

Release checks include export parity against PyTorch, end-to-end comparison of full passages with the PyTorch reference path, comparison of the quantized bundles with full precision on real sentences, and listening across all ten voices. Passing tests does not establish naturalness; listening decides quality.

Known limitations include occasional pronunciation errors, occasional roughness on fast, wide pitch movements in some voices, and limited strong anger or sadness. No arbitrary-speaker cloning is provided.

See [the model card](MODEL_CARD.md) for the model contract.

## Repository structure

```text
scyllasband/          Python API, CLI, first-run setup, download, bundle contract and execution
libscyllasband/       C++ runtime with a C API, for LiteRT and ONNX Runtime
examples/             Android and iOS applications
samples/              gallery script and generator, sample index
docs/                 voice gallery (GitHub Pages)
data/                 example texts and a dialogue script
tests/                runtime tests (python -m pytest tests)
MODEL_CARD.md         model card for the current release
```

Training data, checkpoints and export tooling are not part of this repository.

## License

Apache 2.0. See [LICENSE](LICENSE).

## Acknowledgments

Scylla's Band builds on:

- [LiteRT](https://ai.google.dev/edge/litert) for the default runtime backend.
- [ONNX Runtime](https://onnxruntime.ai/) for the ONNX bundle.
- [Vocos](https://arxiv.org/abs/2306.00814) and `charactr/vocos-mel-24khz` for the 24 kHz decoder backbone.
- [DeepPhonemizer](https://github.com/as-ideas/DeepPhonemizer) lineage for the Scylla's Band G2P training workflow.
- [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/) for alignment-derived duration supervision in the training pipeline.
