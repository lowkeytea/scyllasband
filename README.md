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
- LiteRT, Core ML, Core AI and ONNX Runtime bundles with INT8 weights; every latent-length bucket shares one set of weights. On Apple devices, Core ML and Core AI run on the GPU or the Neural Engine.
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

On an Apple silicon Mac, `download` fetches the Core ML bundle, which runs on the Mac's GPU, and offers the Core AI bundle as well when the Mac can build iOS, iPadOS or visionOS 27 apps. The first command that runs a Core ML or Core AI bundle adds `coremltools` or `coreai-core` to the environment.

Requirements: Python 3.10–3.14 on macOS with Apple silicon, Linux (x86_64 or aarch64) or Windows (x86_64). To set the environment up yourself instead:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .                   # ".[onnx]", ".[coreml]" or ".[coreai]" for the other backends
```

The model is downloaded from [`spybyscript/scyllasband`](https://huggingface.co/spybyscript/scyllasband), pinned to the tag `v2-20261005-apple` (release v2-20261005 with its Core ML and Core AI bundles), into `scyllasband/models/<bundle>` (`litert`, `coreai`, `coreml` or `onnx`). Voice conditioning is embedded in the graphs; no separate voice files are needed.

## Delivery controls

| Control | Request | Default |
| --- | --- | --- |
| Energy | 0–4 or `auto` | 2 |
| Tension | 0–4 or `auto` | 2 |
| Valence | 0–4 or `auto` | 2 |
| Assertiveness | 0–4 or `auto` | 2 |
| Whisper | `on`, `off`, or `auto` | `off` |

With every control at 2, the default, voices speak in a calm, conversational delivery with a neutral tone. Raising or lowering a control changes the delivery:

| Control | Lower | Higher |
| --- | --- | --- |
| Energy | Lower, darker voice with a few more pauses | Higher, brighter, more projected voice |
| Tension | Slightly looser and steadier | Tighter, with wider pitch swings |
| Valence | Flatter, more downcast pitch | Livelier pitch, a little quicker, shorter pauses |
| Assertiveness | More pauses between phrases | Faster and clipped with fewer pauses; the strongest control over pace |

`whisper=on` whispers: quieter and breathy.

The model learned delivery mostly from settings between about 1.7 and 2.8 for energy, 1.9 and 3.0 for valence, 1.8 and 2.7 for assertiveness, and 1.0 and 3.1 for tension. The controls are most reliable inside those ranges; further out, voices can sound strained or a style can become ambiguous.

Combining the controls gives familiar deliveries:

| Delivery | Settings |
| --- | --- |
| Calm and soft | `energy=1.8,tension=1,valence=2.5,assertiveness=2.1` |
| Assertive and quick | `energy=2.6,tension=2.4,valence=2.1,assertiveness=2.8` |
| Joyful | `energy=2.4,tension=2,valence=2.8,assertiveness=2.2` |
| Angry | `energy=2.7,tension=3.1,valence=1.3,assertiveness=2.5` |
| Sad | `energy=1.7,tension=1.7,valence=1.8,assertiveness=2` |
| Whispered | `whisper=on` |

Angry takes valence a little below the trained range so that it separates clearly from joyful. These are moderate styles; how strongly each comes through varies with voice, language and text. The [voice gallery](https://lowkeytea.github.io/scyllasband/#delivery) plays each of them for all ten voices.

```bash
python -m scyllasband speak --voice ariadne \
    --delivery energy=2.4,tension=2,valence=2.8,assertiveness=2.2 \
    -o joyful.wav "We actually did it! The whole street came out to watch!"

python -m scyllasband speak --voice ink --language en_gb \
    --delivery whisper=on -o whisper.wav "Keep your voice down."
```

Unspecified controls use the neutral default of 2 (whisper off). `--delivery auto` leaves every control unspecified, and `energy=auto` leaves only that one unspecified; an unspecified control is passed to the model as absent, which is different from requesting 2.

## CLI

```bash
python -m scyllasband download                  # Core ML on Apple silicon Macs, LiteRT elsewhere; --flavor <bundle> or all
python -m scyllasband validate-bundle
python -m scyllasband list-voices
python -m scyllasband normalize-text --language es "Cuesta 12,50 € el 22/05/2026."

python -m scyllasband speak --voice gwen --language en_us \
    --delivery energy=1.8,tension=1,valence=2.5 -o calm.wav "Take a slow breath."
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
| `--backend`, `--bundle` | the installed bundle for this machine (Core ML on a Mac, LiteRT elsewhere) | Choose another installed bundle |
| `--compute-units` | `auto` | Core ML / Core AI: `auto` (the bundle's recommendation), `gpu`, `cpu` or `ane`; see [Backends](#backends) |
| `--pause-ms` | 0 | Extra silence where the voice, language or delivery changes |
| `--metadata` | none | Write phones, durations and timings as JSON |
| `--no-normalize-text` | off | Skip spoken-text normalization for text that is already normalized |

## Backends

| Bundle | Used on | Download | Runtime package |
| --- | --- | --- | --- |
| Core ML | Apple silicon Macs (macOS 15 and later); iOS and iPadOS 18 apps | 142 MB | `coremltools`, installed the first time a Core ML bundle runs (or `pip install -e ".[coreml]"`) |
| Core AI | iOS, iPadOS and visionOS 27 apps; also runs on macOS 27 | 155 MB | `coreai-core`, installed the first time a Core AI bundle runs (or `pip install -e ".[coreai]"`) |
| LiteRT | Linux, Windows, Android and Intel Macs | 164 MB | `ai-edge-litert` (installed with this package) |
| ONNX Runtime | anywhere ONNX Runtime runs | 250 MB | `onnxruntime`, installed the first time you use `--backend onnx` (or `pip install -e ".[onnx]"`) |

`python -m scyllasband download` fetches the Core ML bundle on an Apple silicon Mac and the LiteRT bundle elsewhere. A Mac that can build apps for iOS, iPadOS or visionOS 27 (macOS 27, or an Xcode with those SDKs) is also offered the Core AI bundle those apps use; the Mac itself keeps running Core ML. `--flavor` downloads a particular bundle. Commands use the installed bundle for the machine; `--backend` and `--bundle` select another, for example `--backend coreai` on macOS 27. Every bundle holds INT8 weights for the transformers and the G2P; LiteRT, Core ML and Core AI also quantize the vocoder. `--threads` sets the CPU threads per graph. Sampling uses Heun with 8 steps by default (`--sampler`, `--steps`).

On Core ML and Core AI, `--compute-units` chooses where the graphs run: `auto` (the bundle's recommendation: the GPU), `gpu`, `cpu`, or `ane`, which runs the graphs the bundle marks as accurate on the Neural Engine there (Core AI's flow) and the rest on the CPU. iOS does not allow GPU work from background apps, so an app that keeps speaking in the background uses `ane` (Core AI) or `cpu`. The first load of a Core AI bundle specializes its graphs for the device (a few seconds on the GPU, about half a minute for the Neural Engine); the system caches that until the next OS update.

Measured with the native runtime on a three-sentence passage (11.6 s of speech); first audio is the first sentence:

| Device | Bundle (compute units) | Launch to first audio | Warm first audio | Real-time factor | Peak memory |
| --- | --- | --- | --- | --- | --- |
| iPhone 17 Pro Max | Core AI (GPU) | 0.96 s | 244 ms | 0.063 | 320 MB |
| iPhone 17 Pro Max | Core AI (Neural Engine + CPU) | 0.86 s | 390 ms | 0.10 | 215 MB |
| iPhone 17 Pro Max | Core ML (GPU) | 1.9 s | 210 ms | 0.06 | 220 MB |
| iPhone 17 Pro Max | Core ML (CPU) | 1.05 s | 760 ms | 0.20 | 330 MB |
| Galaxy Z Fold 8 | LiteRT (CPU) | 1.6 s | 450 ms | 0.12 | |
| M5 Max Mac | Core AI (GPU) | 0.43 s | 73 ms | 0.019 | 444 MB |
| M5 Max Mac | Core ML (GPU) | 1.3 s | 59 ms | 0.015 | 350 MB |
| M5 Max Mac | LiteRT (CPU) | 1.0 s | 371 ms | 0.098 | 768 MB |

Launch to first audio is a new process with the Core AI specialization already cached; warm is a later request on a loaded runtime.

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
- Text is split into sentences after normalization. Each pass speaks 64 to 420 latent frames (about 1.4 to 9 seconds), the lengths the model was trained on: a very short sentence is spoken together with its neighbours in the same paragraph, and a sentence too long for one pass is split at clause punctuation.

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

`libscyllasband/` is the C++ runtime for the same bundles, built against LiteRT, ONNX Runtime, or Core ML and Core AI on Apple platforms; see [its README](libscyllasband/README.md). The [Android](examples/android/README.md) example runs the LiteRT bundle; the [iOS](examples/ios/README.md) example runs Core AI on iOS 27 and Core ML on iOS 18–26 through the `ScyllasBandKit` pod.

## Validation and limitations

Release checks include export parity against PyTorch, end-to-end comparison of full passages with the PyTorch reference path, comparison of the quantized bundles with full precision on real sentences (for Core ML and Core AI on each compute unit they use), and listening across all ten voices. Passing tests does not establish naturalness; listening decides quality.

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
