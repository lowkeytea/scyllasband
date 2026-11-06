# Scylla's Band

Scylla's Band is a local text-to-speech runtime for fourteen managed voices. It predicts phone durations and a pitch contour, generates continuous acoustic latents with rectified flow that follows that contour, and decodes speech at 24 kHz through an acoustic adapter and Vocos.

Delivery is controlled with **intensity, pace and pitch**. The emotion comes from the text; intensity sets how much the voice puts into it.

## Samples and community

- **[Listen to the voices](https://lowkeytea.dev/scyllasband/samples/)**: every voice saying hello, playing a scene and singing a chorus, four songs built from scores, long-form narration and a nine-voice dialogue in six languages.
- **[Try it in your browser](https://lowkeytea.dev/scyllasband/demo/)**: the runtime compiled to WebAssembly, with the model downloaded once.
- Join the [Scylla's Band Discord](https://discord.gg/mBhpn7WB5N) for release updates, help, and community discussion.

## Highlights

- Fourteen voices in seven locales: American and British English, Spanish, Italian, French, German and Vietnamese.
- Delivery controls for intensity (calm to intense), pace (slower to faster, words and pauses together) and pitch, each measured in the voice's own range.
- Runs locally on CPU. The default LiteRT bundle synthesizes more than 30× faster than real time on an 8-thread desktop CPU.
- LiteRT, Core ML, Core AI and ONNX Runtime bundles with INT8 weights; every latent-length bucket shares one set of weights. On Apple devices, Core ML and Core AI run on the GPU or the Neural Engine.
- Long text is spoken in passages of up to about nine seconds, each with the surrounding text as context and continuing from the sound of the one before, and can be streamed as it is generated.
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
pip install -e .                   # ".[onnx]", ".[cuda]", ".[coreml]" or ".[coreai]" for the other backends
```

The model is downloaded from [`spybyscript/scyllasband`](https://huggingface.co/spybyscript/scyllasband), pinned to the tag `v4-20261106-apple` (release `v4-20261106`, which speaks and [sings](#singing), with its Core ML and Core AI bundles), into `scyllasband/models/<bundle>` (`litert`, `coreai`, `coreml`, `onnx` or `webgpu`), with the release's example scores in `scyllasband/models/melodies`. `--repo` and `--revision` name another repository or tag (the previous release, `v3-20261104-apple`, speaks only and needs a runtime from before this one's singing support), and `download_provenance.json` in the installed bundle records where it came from. Voice conditioning is embedded in the graphs; no separate voice files are needed.

## Delivery controls

| Control | Range | Default | Effect |
| --- | --- | --- | --- |
| Intensity | −2 to 2 | 0 | Calm (negative) to intense (positive): how much the voice puts into the line |
| Pace | −2 to 2 | 0 | Slower (negative) to faster (positive): the words and the pauses between them |
| Pitch | −2 to 2 | 0 | Shifts the voice's pitch, in semitones |

0 is each voice's own delivery. Intensity and pace are measured in each voice's own range, in standard deviations around its average, so the same value is a comparable change for every voice and language. The emotion comes from the text: an angry line sounds angrier at higher intensity and more restrained at lower intensity. Requests outside the range are clamped into it.

```bash
python -m scyllasband speak --voice ariadne --intensity 1.5 \
    -o excited.wav "We actually did it! The whole street came out to watch!"

python -m scyllasband speak --voice ink --language en_gb --intensity -1 --pace -1 \
    -o quiet.wav "I kept the letter for years."

python -m scyllasband speak --voice gwen --delivery intensity=1,pace=0.5,pitch=1 -o bright.wav "Good morning, everyone!"
```

Unspecified controls take the voice's default. A passage of only a word or two, such as a lone "Yes.", is spoken with the voice's own intensity and pace, as the model was trained; its pitch shift still applies.

## CLI

```bash
python -m scyllasband download                  # Core ML on Apple silicon Macs, LiteRT elsewhere; --flavor <bundle> or all
python -m scyllasband validate-bundle
python -m scyllasband list-voices
python -m scyllasband normalize-text --language es "Cuesta 12,50 € el 22/05/2026."

python -m scyllasband speak --voice gwen --language en_us \
    --intensity -1.5 --pace -0.5 -o calm.wav "Take a slow breath."
python -m scyllasband speak --voice rex --file data/test_document.txt --metadata story.json -o story.wav
python -m scyllasband group-speak --file data/walkthrough_demo.txt --pause-ms 250 -o dialogue.wav
python -m scyllasband stream --voice scylla --file data/test_document.txt -o chunks/
python -m scyllasband plan --voice scylla "Print the sentence plan as JSON."
```

`python3 speak.py ...` and `python3 groupSpeak.py ...` are shortcuts for `speak` and `group-speak`. An installed package also provides `scyllasband`, `scyllasband-speak` and `scyllasband-group-speak` commands.

| Option | Default | Effect |
| --- | --- | --- |
| `--voice`, `--language` | `scylla`, the voice's English | Voice and locale; `en` picks the voice's own English |
| `--intensity`, `--pace`, `--pitch` | the voice's own | Delivery controls, −2 to 2 (see [Delivery controls](#delivery-controls)) |
| `--delivery` | the voice's own | The same as one string, e.g. `intensity=1,pace=-0.5,pitch=0` |
| `--speed` | 1.0 | A plain time stretch on top of pace; 1.2 is 20% faster |
| `--steps`, `--sampler` | 4, `euler` | Flow sampling. The flow is trained for few-step sampling, so Euler with 4 steps (four model evaluations) is the default. Heun evaluates the model twice per step: `--sampler heun --steps 8` costs four times as much and sounds slightly sharper. Fewer steps are faster still; listen when trading quality for speed |
| `--seed`, `--temperature` | random, 1.0 | Repeatable output; scale of the sampling noise |
| `--threads` | LiteRT: the CPUs available, up to 4 | CPU threads per graph |
| `--backend`, `--bundle` | the installed bundle for this machine (Core ML on a Mac, LiteRT elsewhere) | Choose another installed bundle |
| `--compute-units` | `auto` | Core ML / Core AI: `auto` (the bundle's recommendation), `gpu`, `cpu` or `ane`. ONNX Runtime: `cpu`, or `gpu` for NVIDIA GPUs. See [Backends](#backends) |
| `--engine` | `native` where built, else `python` | The runtime that runs the bundle; see [Engines](#engines) |
| `--pause-ms` | 0 | Extra silence where the voice, language or delivery changes |
| `--metadata` | none | Write phones, durations and timings as JSON |
| `--no-normalize-text` | off | Skip spoken-text normalization for text that is already normalized |

## Backends

| Bundle | Used on | Download | Runtime package |
| --- | --- | --- | --- |
| Core ML | Apple silicon Macs (macOS 15 and later); iOS and iPadOS 18 and visionOS 2 apps | 406 MB | `coremltools`, installed the first time a Core ML bundle runs (or `pip install -e ".[coreml]"`) |
| Core AI | iOS, iPadOS and visionOS 27 apps; also runs on macOS 27 | 418 MB | `coreai-core`, installed the first time a Core AI bundle runs (or `pip install -e ".[coreai]"`) |
| LiteRT | Linux, Windows, Android and Intel Macs | 470 MB | `ai-edge-litert` 2.2 (installed with this package; the native runtime links LiteRT 2.2.0) |
| ONNX Runtime | anywhere ONNX Runtime runs | 1.1 GB | `onnxruntime`, installed the first time you use `--backend onnx` (or `pip install -e ".[onnx]"`) |
| WebGPU (ONNX) | browsers through ONNX Runtime Web; NVIDIA GPUs through ONNX Runtime with CUDA | 782 MB | `onnxruntime-gpu` for NVIDIA GPUs: `pip install -e ".[cuda]"` |

`python -m scyllasband download` fetches the Core ML bundle on an Apple silicon Mac and the LiteRT bundle elsewhere. A Mac that can build apps for iOS, iPadOS or visionOS 27 (macOS 27, or an Xcode with those SDKs) is also offered the Core AI bundle those apps use; the Mac itself keeps running Core ML. `--flavor` downloads a particular bundle. Commands use the installed bundle for the machine; `--backend` and `--bundle` select another, for example `--backend coreai` on macOS 27. The LiteRT, Core ML, Core AI and WebGPU bundles hold 8-bit weights for the transformers and the G2P (the ONNX bundle is full precision); LiteRT, Core ML and Core AI also quantize the speech vocoder, and keep the singing decoder (BigVGAN) in FP16. `--threads` sets the CPU threads per graph; LiteRT uses the CPUs available, up to four, as the native runtime does. Sampling uses Euler with 4 steps by default (`--sampler`, `--steps`).

On an NVIDIA GPU, ONNX Runtime runs the WebGPU bundle, whose 8-bit weights stay on the GPU (`MatMulNBits`; dynamic INT8 would fall back to the CPU there). Install ONNX Runtime for CUDA in an environment without the CPU-only `onnxruntime` package; the `cuda` extra brings the CUDA and cuDNN libraries it needs:

```bash
pip install -e ".[cuda]"
python -m scyllasband download --flavor webgpu
python -m scyllasband speak --backend onnx --compute-units gpu --voice tuesday -o hello.wav "Hello from the GPU."
```

`--backend onnx --compute-units gpu` picks the installed WebGPU bundle; per-graph choices such as `gpu,g2p=cpu` also work. With the Python runtime on an NVIDIA RTX PRO 6000 (Blackwell), the three-sentence passage in the table below runs at about 90× real time with 58 ms to first audio when warm, against about 7× and 680 ms for the ONNX bundle on the same machine's CPU (Ryzen 7 7800X3D).

On Core ML and Core AI, `--compute-units` chooses where the graphs run: `auto` (the bundle's recommendation: the GPU), `gpu`, `cpu`, or `ane`, which runs the graphs the bundle marks as accurate on the Neural Engine there (Core AI's flow and pitch predictor) and the rest on the CPU. iOS does not allow GPU work from background apps, so an app that keeps speaking in the background uses `ane` (Core AI) or `cpu`. The first load of a Core AI bundle specializes its graphs for the device (seconds to tens of seconds; see the table below); the system caches that until the next OS update.

Measured with the native runtime on a three-sentence passage (10.3 s of speech) with the default sampler. First run is the first launch after installing the bundle (when Core AI specializes its graphs and LiteRT writes its weight cache), cold start a later launch, warm first audio a later request on a loaded runtime, and real-time speed the seconds of audio generated per second when warm. The [model card](https://huggingface.co/spybyscript/scyllasband) describes the measurements.

| Device | Bundle | Compute | First run | Cold start | Warm first audio | Real-time speed |
| --- | --- | --- | --- | --- | --- | --- |
| MacBook Pro, M5 Max (macOS 27) | Core ML | GPU (default) | 2.73 s | 1.34 s | 30 ms | ~170× |
|  | Core ML | CPU | 2.17 s | 0.33 s | 221 ms | ~20× |
|  | Core AI | GPU (default) | 5.80 s | 0.46 s | 32 ms | ~150× |
|  | Core AI | Neural Engine | 34.2 s | 0.60 s | 316 ms | ~20× |
|  | Core AI | CPU | 3.54 s | 0.93 s | 844 ms | ~5.0× |
|  | LiteRT | CPU | 0.41 s | 0.30 s | 105 ms | ~34× |
|  | ONNX Runtime | CPU | 0.88 s | 0.85 s | 216 ms | ~21× |
| iPhone 17 Pro Max (iOS 27.2) | Core ML | GPU (default) | 15.5 s | 1.79 s | 77 ms | ~65× |
|  | Core ML | CPU | 2.87 s | 0.49 s | 294 ms | ~15× |
|  | Core AI | GPU (default) | 18.5 s | 0.59 s | 70 ms | ~68× |
|  | Core AI | Neural Engine | 8.00 s | 0.74 s | 367 ms | ~17× |
|  | Core AI | CPU | 7.58 s | 1.16 s | 1010 ms | ~4.1× |
| Galaxy Z Fold 8, Snapdragon SM8850 (Android 17) | LiteRT | CPU | 0.81 s | 0.52 s | 120 ms | ~27× |
|  | ONNX Runtime | CPU | 1.85 s | 1.89 s | 461 ms | ~9.2× |

## Engines

Commands run one of two engines over the same bundles:

- `native`: `libscyllasband`, the C++ runtime the iOS and Android apps use, through `scyllasband.native` (ctypes over its C API). Build it once from this checkout; CMake 3.20 and a compiler are needed, and the Core ML / Core AI build also needs Ninja:

  ```bash
  python -m scyllasband build-native                    # LiteRT bundles
  python -m scyllasband build-native --backend apple    # Core ML and Core AI bundles (macOS)
  ```

  The first LiteRT build downloads LiteRT's prebuilt runtime for this machine (`libscyllasband/scripts/stage_litert_sdk.py`). The library and the SDK libraries it loads go to `scyllasband/_native/<backend>/` (not tracked by git). `SCYLLASBAND_NATIVE_LIB` names another build of ABI 3.1 or later.
- `python`: this package's engine, over ONNX Runtime, LiteRT, Core ML or Core AI sessions from Python.

By default a command runs the native engine when a library for the bundle's backend is built and the Python engine otherwise, printing one line that says so. ONNX Runtime bundles, including CUDA through the WebGPU bundle, stay on the Python engine. `--engine native` or `--engine python` (or `SCYLLASBAND_ENGINE`) chooses explicitly. The engines plan and speak the same way: `libscyllasband/tests/parity_harness.py` checks plans, passages, dialogue and songs with the same noise fed to both. Each engine draws its own noise, so a `--seed` gives a different take on each; `--metadata` records the engine and its noise.

## Python API

```python
from scyllasband import ScyllasBandRuntime, SynthesisRequest

runtime = ScyllasBandRuntime.from_bundle("scyllasband/models/litert")
result = runtime.synthesize(SynthesisRequest(
    text="It is a real pleasure to meet you.",
    voice_id="ariadne",
    language="en_us",
    intensity=1.0,       # -2 (calm) to 2 (intense); None is the voice's own
    pace=-0.5,           # -2 (slower) to 2 (faster)
    pitch=0.0,           # semitones
    seed=2027,
))
# result.audio (float32 numpy array), result.sample_rate, result.metadata
```

`ScyllasBandRuntime.from_bundle` always runs the Python engine. `scyllasband.native.open_runtime(bundle, engine=None)` chooses the engine as the CLI does and returns a runtime with the same methods for speech, dialogue and singing; a `NativeRuntime` also has `score_parse`, `sing_fit` and `sing_preview`, the calls the apps' melody editors use.

Reuse the runtime across requests; `runtime.warmup()` creates the sessions before interactive use. `runtime.delivery_ranges()` and `runtime.voice_delivery_defaults(voice)` give the supported ranges and a voice's defaults. `runtime.synthesize_stream(records)` yields each passage's audio as it is ready, and `runtime.plan_records(records)` returns the sentence plan the passages are cut from.

## Long form and dialogue

Long text is spoken in passages. The first is short, ending at the first sentence break, so audio starts quickly; each later one runs as close to the model's trained length (about nine seconds) as the text allows, ending at a sentence break, else a clause, else between words, so text without punctuation is handled too. Lengths are measured with the requested delivery, so slower paces get shorter passages. Each passage sees the surrounding text as context and continues from the sound of the one before, and consecutive passages are decoded as one continuous waveform, so long text needs no chunking settings.

```bash
python -m scyllasband speak --voice ariadne --file story.txt \
    --intensity 0.5 --metadata story.json -o story.wav

python -m scyllasband stream --voice scylla --file story.txt -o chunks/

python -m scyllasband group-speak --file dialogue.txt -o dialogue.wav
```

A dialogue file sets the voice, language and controls per tagged span:

```text
[ariadne:en_us:intensity=1] Good to see you. [es] Me alegra verte.
[rex:en_us:intensity=1.5,pace=1] We should get going.
[ink:en_gb:intensity=-1,pace=-0.5] Keep this between us.
```

Language tags keep the active delivery request. A new delivery tag replaces it, with unspecified controls back at the voice's own. `--pause-ms` adds silence where the voice, language or delivery changes.

## Singing

A v4 bundle can sing a melody score. The `sing` command takes a score in JSON and writes the sung line, phrase by phrase, on the song's timeline with silence in the rests:

```bash
python -m scyllasband sing --score song.json --voice gwen --seed 0 -o song.wav
python -m scyllasband sing --score song.json --voice felix --turboSing --metadata song_plan.json -o song.wav
python -m scyllasband sing --score scyllasband/models/melodies/phrase_pop_00.json --voice poppy -o ring.wav
```

`download` also installs the release's twenty example scores (four songs, sixteen phrases) in `scyllasband/models/melodies`.

An installed package also provides `scyllasband-sing`. A score is the lyric, the tempo, a style and the notes:

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

- `notes` is an ordered list of `[label, note, beats]`. Time runs from the first entry; beats are fractions of a quarter note at `bpm` (0.25 a sixteenth, 0.5 an eighth, 1 a quarter).
- `note` is a pitch name with octave (`C4`, `F#3`, `Bb3`) or `rest`; rests use the label `-`.
- A word on one note is a bare label (`give`). A word over several notes is numbered: `baby~1 baby~2` (the whole word on each part, or its syllables `ba~1 by~2`) or `home~1 home~2` (a melisma, one syllable held over two pitches). The numbered parts must be consecutive, with no rest between them.
- `text` is the lyric, the labels' words in order; it may be left out.
- Two styles: `pop` (the sung delivery) and `ballad` (slower, sustained). The style picks the delivery mode the model was trained with; the notes set the pitch and the timing.
- Phrases are split at rests of 0.25 s or more and rendered one at a time, up to about ten seconds each (less when the bundle's largest bucket is shorter), so put a rest between lines. The whole song is moved by octaves into the voice's sung range (`--no-register` keeps the written octave), so write in any octave; `--transpose N` adds N semitones first, to put a song in the voice's key (a voice sings most accurately within about five semitones of its centre, which the bundle lists per voice).

Singing uses the bundle's singing settings (`v4-20261106`: 4 Euler steps without guidance, the flow distilled from guided 8-step Heun sampling, so a phrase costs the flow what a spoken passage does; the BigVGAN decoder where the bundle carries it; `--steps`, `--sampler`, `--guidance` and `--decoder` override them) and the score's timing in place of the duration and pitch predictors. `--turboSing` (same as `--decoder vocos`) decodes with Vocos, the speech decoder: decoding then costs about as much as for speech instead of most of the render, at the price of some harmonic artefacts on sustained notes in some voices. In Python, `runtime.sing(score, voice_id, language=None, seed=None, steps=None, sampler=None, decoder=None)` returns the same `SynthesisResult` as `synthesize`, with the phrase plan in `metadata["phrases"]`; `runtime.supports_singing` tells whether the loaded bundle can sing. Singing requires a v4 bundle (graph contract `scyllasband_measured_controls_v4`); v3 bundles speak exactly as before and report that they cannot sing.

## Voices and languages

| Voices | English dialect |
| --- | --- |
| Ariadne, Felix, Gwen, Kincade, Max, Poppy, Priya, Rex, Scylla, Stone, Thistle | `en_us` |
| Ink, Orpheus, Tuesday | `en_gb` |

All fourteen also support `es`, `it`, `fr`, `de`, and `vi`: 84 trained voice/locale pairs. `en` selects the voice's own English dialect. The runtime rejects unsupported voice/dialect combinations.

## Text handling

- Spoken-text normalization expands numbers, decimals, dates, times, currency, percentages, ordinals, fractions, initialisms and common abbreviations for each language, and splits letters joined to digits ("V3", "MP3", "4K") into letter names and numbers. English also reads common units, multipliers and plural numbers ("100ms", "68x", "4x4", "1990s", "100s").
- The Scylla's Band G2P model in the bundle converts each phrase to phones; punctuation is kept as pause and intonation cues, and word boundaries become optional silences.
- A small built-in lexicon reads common initialisms written in capitals letter by letter ("GPU", "USB", "API", also with a plural "s") and corrects a few words the G2P misreads. A bundle's `assets/g2p/phoneme_dict.json` (`{language: {word: phones}}`) adds or replaces word entries.
- Text is split into sentences after normalization, and the sentences of a paragraph are read as one stream of phones. Each pass speaks 64 to 420 latent frames (about 1.4 to 9 seconds), the lengths the model was trained on, cut from that stream at a sentence end, else a clause, else a word break; a pass is measured with its own delivery and speed before it is synthesized.

`--no-normalize-text` skips normalization when the text is already in spoken form.

## How it works

```text
text
  -> spoken-text normalization
  -> Scylla's Band G2P: phones, word boundaries and punctuation cues
  -> sentence plan, read as one stream of phones per paragraph
  -> passages of 64-420 frames cut from the stream (short first one, later ones near the maximum),
     each with the phones around it as context
  -> duration predictor (voice, language, intensity, pace, neighbouring text) -> frames per phone,
     pauses stretched or shortened with the pace
  -> timing events per frame: word starts, punctuation, sentence type
  -> pitch predictor -> a pitch and voicing contour per frame, shifted by the pitch control and kept
     inside the voice's range
  -> rectified-flow latent generator following that contour, continuing from the previous passage's latents
  -> 24-channel acoustic latents
  -> acoustic adapter + Vocos decoder at 24 kHz, decoded with the previous passage as left context
  -> one continuous waveform, or one chunk per passage when streaming
```

## Bundle layout

```text
scyllasband/models/litert/                scyllasband/models/onnx/
  manifest.json                             manifest.json
  litert/                                   onnx/
    g2p.tflite                                g2p/model.onnx, tokenizer.json, phoneme_dict.json
    duration_predictor.tflite                 components/
    vector_context_encoder.tflite               duration_predictor.onnx
    pitch_context_encoder.tflite                vector_context_encoder.onnx, pitch_context_encoder.onnx
    pitch_estimator.tflite   (one signature     pitch_estimator[_128..._768].onnx
    vector_estimator.tflite   per bucket)       vector_estimator[_128..._768].onnx
    vocoder.tflite                              vocoder[_128..._768].onnx
  assets/
    voices.json, languages.json                 shared_weights.bin, shared_weights.json
    phone_vocab.json, fixed_references.json   assets/  (same as LiteRT)
    g2p/  tokenizer, normalization, overrides
```

Latent-length buckets are 128, 256, 384, 512, 768 and 1024 frames (about 2.7 to 21.8 seconds); each sentence uses the smallest bucket that fits. The manifest declares the graph contract, inputs, buckets and controls, and the runtime validates it before loading.

## Native and mobile integration

`libscyllasband/` is the C++ runtime for the same bundles, built against LiteRT, ONNX Runtime, or Core ML and Core AI on Apple platforms; see [its README](libscyllasband/README.md). The [Android](examples/android/README.md) example runs the LiteRT bundle; the [iOS](examples/ios/README.md) example runs Core AI on iOS 27 and Core ML on iOS 18–26 through the `ScyllasBandKit` pod. Two singing tutorials sit beside them: [Ghost in the Machine](examples/ghost_in_the_machine/README.md) (a spoken-and-sung piece built step by step, with the melody format and how speech and song are merged) and [Kitchen Patter](examples/kitchen_patter/README.md) (fast talk-singing from one score).

## Validation and limitations

Release checks include export parity against PyTorch, end-to-end comparison of full passages with the PyTorch reference path, comparison of the quantized bundles with full precision on real sentences (for Core ML and Core AI on each compute unit they use), and listening across all fourteen voices. Passing tests does not establish naturalness; listening decides quality.

Known limitations include occasional pronunciation errors and occasional roughness on fast, wide pitch movements in some voices. The emotion follows the text; intensity changes how much of it comes through, not which emotion it is. No arbitrary-speaker cloning is provided.

See [the model card](MODEL_CARD.md) for the model contract.

## Repository structure

```text
scyllasband/          Python API, CLI, first-run setup, download, bundle contract and execution
libscyllasband/       C++ runtime with a C API, for LiteRT and ONNX Runtime
examples/             Android and iOS applications; the singing pieces, each a tutorial (ghost_in_the_machine, kitchen_patter,
                      face_down, bubble_wrap, thank_you) and write_scores.py, which compiles a score source into melody JSON
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
