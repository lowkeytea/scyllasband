# Scylla's Band

Scylla's Band is the public inference/runtime package for the `scyllasband` continuous-latent TTS model. The 1.0 bundle predicts phone durations and rectified acoustic latent flow, then decodes those latents through a Scylla's Band acoustic adapter and Vocos vocoder.

The default public runtime path is ONNX across the Python CLI, Python API, Android sample, and native `libscyllasband` API. LiteRT remains available as an experimental, explicitly selected backend for platform acceleration work.

## Highlights

- ONNX Runtime is the default desktop/server inference path.
- Experimental LiteRT bundle is available for explicit native and mobile validation.
- 24 kHz output with 100-mel acoustic features and 24-D acoustic latents.
- Four public text-input languages: `en_us`, `en_gb`, `es`, and `it`.
- Ten managed voices: `ariadne`, `felix`, `gwen`, `ink`, `max`, `orpheus`, `rex`, `scylla`, `stone`, and `tuesday`.
- Six independently scored affect controls: `calm`, `joy`, `anger`, `sadness`, `sarcasm`, and `questioning`.
- Affect CFG acts on both duration and acoustic-flow prediction while retaining voice/reference conditioning. The current model was trained with affect dropout, so guidance above `1` can strengthen delivery without changing the six-axis input range.
- Long-form chunking is enabled by default, including boundary metadata, punctuation pause floors, prefix-latent carryover, and span context.
- Group-speak input can label lines or inline spans with `[voice]`, `[voice:language]`, or `[voice:language:axis=value,...]`.
- Runtime code is self-contained in `scyllasband/`; training and export tooling are intentionally outside this inference package.

## Audio Samples

Open the [interactive voice and affect gallery](https://lowkeytea.github.io/scyllasband/) to play all ten voices in English, Spanish, and Italian, plus English demonstrations of calm, joy, anger, sadness, and sarcasm-overlay delivery. The [sample index and generation details](samples/README.md) remain available in the repository.

## Quick Start

Commands below assume you run from the Scylla's Band repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -e .
pip install numpy huggingface_hub onnxruntime

python -m scyllasband download
python -m scyllasband validate-bundle
python -m scyllasband list-voices
python -m scyllasband speak --voice scylla --language en_us --emotion calm=0.5 -o hello.wav "Hello from Scylla's Band."
```

`python -m scyllasband download` fetches the default ONNX bundle and voice assets from `spybyscript/scyllasband`. By default it writes:

```text
scyllasband/models/onnx/    # default ONNX inference bundle
scyllasband/models/voices/  # voice packs copied from the selected bundle
```

To fetch the experimental LiteRT bundle as well:

```bash
python -m scyllasband download --runtime-bundles both
```

The CLI defaults to `scyllasband/models/onnx` and does not fall back to LiteRT. To use LiteRT, pass its bundle path and `--backend litert` explicitly. You can pass an explicit bundle path as the first positional argument to `speak`, `plan`, `stream`, `group-speak`, `validate-bundle`, or `list-voices`.

## CLI

Common commands:

```bash
python -m scyllasband download
python -m scyllasband validate-bundle
python -m scyllasband list-voices
python -m scyllasband normalize-text --language es "Cuesta EUR 12,50 el 22/05/2026."

python -m scyllasband speak \
    --voice ariadne \
    --language en_us \
    --emotion calm=0.5,joy=0.5 \
    -o hello.wav \
    "This is the ONNX Scylla's Band runtime."

python -m scyllasband speak \
    --file data/emotional_text.txt \
    --voice gwen \
    --language en_us \
    --emotion joy=0.7,sarcasm=0.4 \
    --emotion-scale 1.25 \
    --metadata emotional.json \
    -o emotional.wav

python -m scyllasband group-speak \
    --file data/groupSpeak.txt \
    --emotion-scale 1.5 \
    -o dialogue.wav
```

Installed entry points:

```bash
scyllasband speak --voice scylla --language en_us -o hello.wav "Hello."
scyllasband-speak --voice scylla --language en_us -o hello.wav "Hello."
scyllasband-group-speak --file data/groupSpeak.txt -o dialogue.wav
```

Six-axis group files use the same normalized `[0, 1]` strengths as `--emotion`:

```text
[ink:en_gb:calm=0.5,joy=0.75] Lovely, [es] muy elegante, [en_gb] and very subtle.
[rex:en_us:anger=0.5,questioning=0.75] Are you really going to do that?
```

Language-only inline tags retain the active affect vector, while a new `axis=value` tag replaces it.

### Emotion Controls and CFG

The six values are continuous, composable controls rather than mutually exclusive emotion classes. `calm`, `joy`, `anger`, and `sadness` describe the core delivery; `sarcasm` and `questioning` are overlays that can be mixed with any core delivery. Each axis stays in `[0, 1]`, omitted axes are zero, and multiple axes may be nonzero at the same time.

The training ratings use a `0` to `4` scale. Runtime values are those ratings divided by four: `1 -> 0.25`, `2 -> 0.5`, `3 -> 0.75`, and `4 -> 1.0`. Start with the axis values themselves when changing the kind or mixture of delivery:

```bash
# Strong anger on the learned axis, with direct conditioning.
python -m scyllasband speak --voice scylla --emotion anger=0.75 -o angry.wav "That was not the agreement."

# A questioning overlay on a quieter core delivery.
python -m scyllasband speak --voice rex --emotion calm=0.25,questioning=0.75 -o question.wav "You meant to do that?"
```

`--emotion-scale` is a separate control. It does not change or extend the `[0, 1]` axis values; it controls how far duration and acoustic-flow predictions move from the learned null-emotion branch:

```text
guided = null + scale * (conditioned - null)
```

- `0` selects the learned null-affect prediction.
- `1` uses the requested emotion vector directly and is the default.
- Values above `1` amplify the difference. `1.25` to `1.5` is a useful first range when the direct effect is too subtle.

The runtime deliberately has no fixed upper cap, so values such as `2.5` are valid. Higher CFG is extrapolation, however, and can eventually produce exaggerated timing, voice instability, or distortion depending on the voice and sentence. The current model's vector estimator used `0.15` affect-condition dropout during training, which supplies the null branch used by CFG. Voice/reference conditioning remains enabled in that branch so guidance targets delivery instead of intentionally discarding identity. A scale other than `1` evaluates both null and conditioned branches, so it costs more than direct conditioning.

### Backend Selection

The default backend is `onnx`. The legacy `auto` spelling remains accepted as an alias for ONNX; it never opts into LiteRT. LiteRT requires both an explicit LiteRT bundle and `--backend litert`.

```bash
# Default ONNX path
python -m scyllasband speak --voice scylla --language en_us -o hello.wav "Hello."

# Explicit ONNX path and provider list
python -m scyllasband speak scyllasband/models/onnx \
    --backend onnx \
    --onnx-providers CPUExecutionProvider \
    --voice scylla \
    --language en_us \
    -o hello_onnx.wav \
    "Hello from ONNX."

# Experimental LiteRT path
python -m scyllasband speak scyllasband/models/litert \
    --backend litert \
    --voice scylla \
    --language en_us \
    -o hello_litert.wav \
    "Hello from LiteRT."
```

The Python ONNX backend requires `onnxruntime`. The Python LiteRT reference path requires one of `ai-edge-litert`, `tflite-runtime`, or TensorFlow. Native LiteRT uses staged LiteRT shared libraries through `libscyllasband`.

ONNX, Python LiteRT, and native LiteRT expose the same six-axis affect and CFG request contract. The selected bundle manifest remains authoritative, and bundles that do not declare `controls.affect.enabled` reject six-axis requests.

### Long-Form Controls

`--chunk-text` is enabled by default. Use `--no-chunk-text` only when debugging a single raw synthesis request.

Relevant defaults:

```text
backend: onnx
steps: 8
sampler: heun
chunk text: enabled
max chunk chars: 220
min chunk chars: 48
boundary fade/crossfade: 8 ms
minimum nonterminal in-chunk clause pause: 160 ms
minimum nonterminal in-chunk sentence pause: 320 ms
lookahead chunks: 1
```

Useful flags:

```bash
python -m scyllasband speak --file long.txt --voice scylla --language en_us -o long.wav
python -m scyllasband speak --file long.txt --voice scylla --language en_us --stream --events events.jsonl -o long.wav
python -m scyllasband plan --file long.txt --voice scylla --language en_us -o plan.json
python -m scyllasband stream --file long.txt --voice scylla --language en_us --events events.jsonl -o stream_chunks
```

`stream` emits JSONL events and can optionally write each completed chunk as a WAV file. Production hosts should call the Python runtime API or native C ABI directly.

### Speed Shortcuts

Scylla's Band uses a flow sampler. More steps generally cost more runtime and can improve stability.

```bash
python -m scyllasband speak --voice scylla --steps 8 --sampler heun -o hq.wav "Default quality path."
python -m scyllasband speak --voice scylla --fast -o fast.wav "Four-step adaptive chunking."
python -m scyllasband speak --voice scylla --faster -o faster.wav "Two-step adaptive chunking."
```

`--fast` expands to `--adaptive-chunking --steps 4`. `--faster` expands to `--adaptive-chunking --steps 2 --max-chunk-chars 180`.

## Python API

```python
from scyllasband import ScyllasBandRuntime, SynthesisRequest

runtime = ScyllasBandRuntime.from_bundle(
    "scyllasband/models/onnx",
    backends=["onnx"],
)

result = runtime.synthesize(
    SynthesisRequest(
        text="Hello from Scylla's Band.",
        voice_id="scylla",
        language="en_us",
        affect={"joy": 0.7, "questioning": 0.25},
        affect_guidance_scale=1.0,
        sampler="heun",
        steps=8,
    )
)
```

The public bundle uses the ordered axes `calm, joy, anger, sadness, sarcasm, questioning`. Values stay independently bounded in `[0, 1]`; overlays do not replace the core delivery. `affect_guidance_scale` is independent of those values and follows the CFG behavior described above.

Long-lived applications can pay graph initialization before the first user request:

```python
runtime = ScyllasBandRuntime.from_bundle(
    "scyllasband/models/onnx",
    backends=["onnx"],
    onnx_autotune_threads=True,
)

# Run once during application startup. The discarded one-step render initializes
# G2P, duration, context, the smallest vector bucket, and its vocoder. ONNX vector
# thread tuning is cached by bundle/runtime/machine fingerprint for later processes.
warmup = runtime.warmup(voice_id="scylla", language="en_us")
print(warmup["runtime_status"])
```

Every chunk synthesized by the same `ScyllasBandRuntime` already reuses its component
sessions. `warmup()` is idempotent unless `force=True`; it is intended for a
persistent application or service, not immediately before a one-shot CLI request.
An explicit `onnx_intra_op_num_threads` value disables autotuning. The CLI exposes
the same controls through `--onnx-intra-op-threads`, `--onnx-inter-op-threads`,
`--onnx-autotune-threads`, and `--onnx-autotune-candidates`. Autotuning applies
only when `CPUExecutionProvider` is primary; accelerator providers should use
their own execution and device-buffer controls. The CLI autotune flag adds its
one-time benchmark cost to that process's first synthesis; run it once to seed
the cache, or hide the work behind application startup with `warmup()`.

For long-form text, prefer `ScyllasBandRuntime.synthesize_stream()` or the CLI so chunk planning, punctuation pause floors, prefix latents, span context, and retry splitting stay in the shared planning path.

## Voices

The current release exposes ten managed voice IDs. Each voice is available for the four public language IDs in the bundle manifest. Some voices default to `en_gb`; passing `--language` is the most explicit way to select dialect.

| Voice | Default Language | Notes |
| --- | --- | --- |
| `ariadne` | `en_us` | Managed public voice |
| `felix` | `en_us` | Managed public voice |
| `gwen` | `en_us` | Managed public voice |
| `ink` | `en_gb` | Managed public voice |
| `max` | `en_us` | Managed public voice |
| `orpheus` | `en_gb` | Managed public voice |
| `rex` | `en_us` | Managed public voice |
| `scylla` | `en_us` | Managed public voice |
| `stone` | `en_us` | Managed public voice |
| `tuesday` | `en_gb` | Managed public voice |

Voice/reference assets live under `assets/voice_packs/` in both ONNX and LiteRT bundles. Runtime reference packs include 128-D style features and 32-D prosody features for the managed voice/language entries. These identity and reference features remain separate from the six-dimensional affect vector.

## How It Works

Scylla's Band is a continuous-latent duration-flow TTS system:

```text
raw text or explicit phones
  -> spoken-text normalization
  -> Scylla's Band G2P / phone frontend
  -> long-form chunk planning, boundary labels, and span context
  -> duration predictor conditioned on voice, language, reference, and affect
  -> phone-duration expansion to latent frames
  -> rectified-flow vector estimator with prefix/context conditioning
     and optional affect CFG
  -> acoustic latents
  -> Scylla's Band acoustic adapter plus frozen Vocos 24 kHz vocoder
  -> waveform and chunk metadata
```

## Current Model Snapshot

This snapshot describes the public `scyllasband` 1.0 inference release.

| Component | Current release |
| --- | --- |
| Architecture | Continuous-latent duration prediction plus rectified acoustic flow |
| Default public backend | ONNX Runtime bundle under `onnx/` |
| Experimental backend | LiteRT bundle under `litert/` |
| Sample rate | 24,000 Hz |
| Acoustic features | 100 mel bins, hop length 256 |
| Latent representation | 24-D latents, latent hop length 512 |
| Languages | `en_us`, `en_gb`, `es`, `it` |
| Voices | 10 managed voices |
| Affect | 6 independent `[0, 1]` axes: `calm`, `joy`, `anger`, `sadness`, `sarcasm`, `questioning` |
| Affect CFG | Null/conditioned linear guidance on duration and vector velocity; default `1`, minimum `0`, no fixed upper cap |
| G2P | Scylla's Band G2P transformer, phrase input, CTC decode, fixed 512 text tokens |
| Duration predictor | 74-phone vocabulary, 192 hidden size, 4 layers, 4 heads, 512 positions |
| Vector estimator | 512 hidden size, 12 layers, 8 heads, AdaLN conditioning, QK norm |
| Context conditioning | Prefix latents up to 24 frames plus 3-segment span context over up to 768 phones |
| Vocoder path | Scylla's Band acoustic adapter plus frozen `charactr/vocos-mel-24khz` backend |
| Fixed export budgets | 512 G2P text tokens, 512 phone frames, 640 latent frames |
| Target buckets | 256, 384, 512, and 640 latent frames, selected by smallest fit |

Training data is not distributed with this runtime package. Scylla's Band was trained primarily on synthetic multilingual speech: seed vocabulary came from Wiktionary-derived word lists, scripted prompt generation covered many target words in unique sentences, and the resulting text was rendered as synthetic speech before filtering with ASR/alignment checks. The current release uses independent per-sample affect scores instead of a single desired emotion category, so one clip can supervise several core/overlay dimensions. The selected vector estimator was trained with `0.15` affect-condition dropout for CFG, alongside chunk-context views, prefix conditioning, and span context. Punctuation boundary phones are followed by explicit `<sil>` slots, matching the aligned frontend targets. Runtime pause floors extend nonterminal silence slots inside an active request, while the final learned silence keeps its model-predicted duration; clean inter-chunk silence is added by the host assembler.

## Bundle Layout

### ONNX Bundle

```text
bundle/
  manifest.json
  onnx/
    g2p/
      model.onnx
      tokenizer.json
      phoneme_dict.json
    components/
      duration_predictor.onnx
      vector_context_encoder.onnx
      vector_estimator_b256.onnx
      vocoder_b256.onnx
      vocoder_adapter_b256.onnx
      ...
      vector_estimator_b640.onnx
      vocoder_b640.onnx
      vocoder_adapter_b640.onnx
      shared_weights.bin
      shared_weights.json
  assets/
    phone_vocab.json
    languages.json
    emotions.json
    voices.json
    g2p/
      tokenizer.json
      normalization.json
      language_map.json
      export_config.json
      phoneme_dict.json
      pronunciation_overrides.json
    voice_packs/
      reference_packs.json
      <voice_id>.npz
```

The ONNX export uses opset 18 and shared external data. The four-bucket release stores shared component weights in a single `shared_weights.bin` blob instead of duplicating every large initializer inside each bucket graph. The current affect model uses one full vector graph per bucket; it does not require separate prefix and tail estimator artifacts.

### LiteRT Bundle

```text
bundle/
  manifest.json
  litert/
    g2p.tflite
    duration_predictor.tflite
    vector_context_encoder.tflite
    vector_estimator_256.tflite
    vocoder_256.tflite
    vocoder_adapter_256.tflite
    ...
    vector_estimator.tflite
    vocoder.tflite
    vocoder_adapter.tflite
  assets/
    ...
```

LiteRT currently carries more duplicated bucket-specific graph data than ONNX, but it remains useful for mobile/native runtime validation and accelerator experiments. Like ONNX, the current public bundle ships full vector graphs for the 256, 384, 512, and 640 frame buckets.

## Native runtimes and Android

`libscyllasband/` owns the native runtime, long-form planning, streaming callbacks, host-language C ABI, and ONNX/LiteRT execution paths. Native ONNX uses the same G2P, duration, reference-pack, emotion CFG, flow-sampling, target-bucket, and vocoder orchestration as native LiteRT; only the graph-session adapter changes.

The [Android sample](examples/android/README.md) packages the current ONNX bundle and exposes all managed voices, manifest-declared languages, the six emotion axes with strength, and non-negative emotion CFG. Its editor uses inline speaker points and streams each completed native audio chunk while later chunks render. The Kotlin wrapper creates and warms one persistent runtime; `libscyllasband` owns a configurable target-bucket LRU, with a capacity-one memory profile used by default on Android.

For normal CLI use, the first LiteRT-backed `speak`, `stream`, `plan`, or `group-speak` run automatically prepares the native runtime when `libscyllasband` is missing. The loader detects the host platform (`linux-x86_64`, `linux-arm64`, `macos-arm64`, or `windows-x86_64`), stages the matching LiteRT runtime from an installed `ai_edge_litert` package or downloads the prebuilt runtime, configures CMake, and builds the shared `scyllasband_native` library. This requires CMake plus a working C++ toolchain for the host.

Set `SCYLLASBAND_NATIVE_AUTO_BUILD=0` to disable this behavior, or set `SCYLLASBAND_NATIVE_LIBRARY=/path/to/libscyllasband_native.*` to use a specific library. Set `SCYLLASBAND_NATIVE_AUTO_DOWNLOAD_LITERT=0` if runtime startup should never download the LiteRT prebuilt.

The manual equivalent is still useful for packaging, debugging, or building the C++ smoke tool:

```bash
python -m scyllasband download --runtime-bundles litert

cd libscyllasband
python scripts/stage_litert_sdk.py --download-runtime --overwrite
cmake -S . -B build -DSCYLLASBAND_ENABLE_LITERT=ON -DSCYLLASBAND_BUILD_TOOLS=ON
cmake --build build --target scyllasband_native -j
cmake --build build --target scyllasband_native_speak -j
```

For native GPU accelerator experiments, also stage the platform GPU prebuilt:

```bash
python scripts/stage_litert_sdk.py \
    --download-runtime \
    --download-gpu-accelerator \
    --overwrite
```

The native LiteRT path supports `cpu`, `gpu`, and `auto` accelerator requests. The public bundle uses full vector graphs rather than the optional split prefix/tail layout. The selected accelerator policy and any fallback remain visible in synthesis metadata.

When `speak` or `group-speak` uses the default managed native library, Scylla's Band compares the built library with `libscyllasband`'s CMake, header, and source files. It automatically rebuilds before loading when those inputs are newer. An explicitly supplied `SCYLLASBAND_NATIVE_LIBRARY` remains caller-managed and is never replaced.

## Text Handling

Text preparation is intentionally runtime-owned:

- Spoken-text normalization expands numbers, dates, times, currency, percentages, ordinals, punctuation, and common language aliases.
- Scylla's Band G2P supports phrase-level frontend conversion for `en_us`, `en_gb`, `es`, and `it`.
- Long-form chunking runs after normalization so chunk budgets reflect what the model will speak.
- Boundary labels distinguish sentence starts/ends, paragraph starts/ends, clause continuations, and artificial chunk continuations.
- Punctuation boundary phones are followed by explicit `<sil>` slots; runtime pause floors keep internal clause and sentence boundaries controllable without stretching punctuation phones or the final learned silence out of distribution.

Use `--no-normalize-text` only for pre-normalized regression tests. Use `--phones` only when intentionally bypassing text and G2P with model-ready phone symbols.

## Repository Structure

```text
scyllasband/
  scyllasband/                 Python package, CLI, download, ONNX/LiteRT runners
  libscyllasband/              native runtime, C ABI, ONNX/LiteRT execution
  examples/android/            ONNX Android sample and Kotlin wrapper
  data/                        small local CLI examples
  models/                      downloaded runtime bundles and voice assets
  README.md                    runtime usage and architecture overview
  MODEL_CARD.md                model-card details for the current release
```

Training data, trainer checkpoints, and export tooling are not part of this public inference package.

## Limitations

- Public text-input languages are limited to `en_us`, `en_gb`, `es`, and `it`.
- The native ONNX path is available when `libscyllasband` is built with `SCYLLASBAND_ENABLE_ONNX=ON` and ONNX Runtime headers/library paths.
- LiteRT is experimental for this release and should be validated on the target device before being treated as production.
- Very long inputs are chunked. Host applications should preserve chunk-relative loudness and normalize only after stitching a full utterance if they apply extra loudness processing.
- Affect values and CFG are conditioning controls, not guarantees of a particular perceived emotion in every voice, language, or sentence. Strong CFG should be listening-tested for the intended text.

## License

Apache 2.0. See [LICENSE](LICENSE).

## Acknowledgments

Scylla's Band builds on:

- [ONNX Runtime](https://onnxruntime.ai/) for the default runtime backend.
- [LiteRT](https://ai.google.dev/edge/litert) for experimental native/mobile execution.
- [Vocos](https://arxiv.org/abs/2306.00814) and `charactr/vocos-mel-24khz` for the frozen 24 kHz vocoder backbone.
- [DeepPhonemizer](https://github.com/as-ideas/DeepPhonemizer) lineage for the Scylla's Band G2P training workflow.
- [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/) for alignment-derived duration supervision in the training pipeline.
