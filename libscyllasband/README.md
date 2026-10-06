# libscyllasband

`libscyllasband` is the native C++ runtime for Scylla's Band model bundles (graph contract
`scyllasband_measured_delivery_v2`). It implements the same pipeline as the Python runtime and produces
the same durations, latents and audio for the same inputs. The public C ABI is `include/scyllasband.h`.

Each build links one graph runtime:

| Backend | Bundle | Download |
| --- | --- | --- |
| LiteRT 2.2 (default) | `scyllasband/models/litert` | `python -m scyllasband download --flavor litert` |
| ONNX Runtime 1.20+ | `scyllasband/models/onnx` | `python -m scyllasband download --flavor onnx` |
| Apple: Core AI (iOS/macOS 27) and Core ML (iOS 18 / macOS 15) | `scyllasband/models/coreai`, `scyllasband/models/coreml` | `python -m scyllasband download --flavor coreai` (or `coreml`) |

`scyllasband_backend()` reports the linked runtime (`litert`, `onnx` or `apple`), and a runtime only opens bundles with
artifacts for it. The Apple build runs whichever of Core AI or Core ML the bundle is for; Core AI is weak-linked, so the
library still loads on systems without it.

On Apple bundles `ScyllasBandRuntimeOptions.accelerator` chooses where graphs run: `CPU`, `GPU`, `AUTO` (the bundle's
recommendation per graph, `controls.<backend>.compute_units`; the GPU for both release bundles) or `NEURAL_ENGINE` (the
graphs the bundle lists in `controls.<backend>.neural_engine_assets` run on the Neural Engine, the rest on the CPU). iOS
allows no GPU work from background apps, so apps that speak in the background use `NEURAL_ENGINE` (Core AI) or `CPU`.
`SCYLLASBAND_COMPUTE_UNITS` (e.g. `gpu,vector_estimator=ane`) overrides the recommendation used by `AUTO`.

## Pipeline

Text is normalized to its spoken form and split into paragraphs and sentences. The bundle G2P converts the
sentences to phones with word starts, and the sentences of a chain (the same voice, language and delivery) are
read as one stream of phones, as connected text. The stream is spoken as targets, each a span from one silence
to another. For each target:

1. the duration graph predicts frames for a 512-phone span of the target plus up to 180 context phones on
   each side from the neighbouring text, and the target's frames are rounded to integers;
2. per-frame phone ids and text events (punctuation, word starts, stress and length marks, phone phase,
   sentence type) are built;
3. the flow integrates noise to latents with Heun (default) or Euler steps in the smallest fitting
   latent-frame bucket, conditioned on the span and on the last 96 latent frames already spoken;
4. the vocoder decodes the target with up to 48 frames of the preceding latents as left context, so
   consecutive targets form one continuous waveform with no inserted pauses or crossfades.

Every target is sized by its predicted durations to stay within the 64 to 420 latent frames (about 1.4 to 9
seconds) the model was trained on. The first target of a request ends at the first sentence end past the minimum,
so audio starts quickly (at a clause or between words only when the opening sentence is too long for one pass). Every later target reaches as far toward the maximum as the text
allows, ending at a sentence boundary, else at a clause, else between words. A paragraph ends a target once it
is long enough, a change of voice, language or delivery always does, and a short remainder is never left on its
own.

## C API

```c
#include "scyllasband.h"

ScyllasBandRuntimeOptions options = {0};
options.bundle_dir = "scyllasband/models/litert";
options.threads = 0;                                /* default */
options.accelerator = SCYLLASBAND_ACCELERATOR_CPU;  /* LiteRT: CPU, GPU or AUTO */

ScyllasBandRuntime* runtime = NULL;
if (scyllasband_runtime_create(&options, &runtime) != SCYLLASBAND_OK) {
    fprintf(stderr, "%s\n", scyllasband_last_error());
}

ScyllasBandRequest request;
scyllasband_request_init(&request);
request.text = "Did you really leave the gate open all night?";
request.voice_id = "scylla";
request.delivery = "energy=2.5,tension=2,valence=2,assertiveness=2,whisper=off";
request.seed = 2027;
request.has_seed = 1;

ScyllasBandAudio audio = {0};
if (scyllasband_synthesize(runtime, &request, &audio) == SCYLLASBAND_OK) {
    /* audio.samples: audio.sample_count mono floats at audio.sample_rate (24000) */
}
scyllasband_audio_free(&audio);
scyllasband_runtime_destroy(runtime);
```

| Function | Purpose |
| --- | --- |
| `scyllasband_runtime_create` / `_destroy` | Load a bundle; reuse one runtime for every request. |
| `scyllasband_voices_json` | Backend, sample rate, release and voices with their languages. |
| `scyllasband_warmup` | Load the graphs a short sentence needs before interactive use. |
| `scyllasband_plan_json` | The normalized sentence plan for a request, without synthesizing. |
| `scyllasband_synthesize` | One waveform plus JSON metadata (phones, durations, buckets, timings). |
| `scyllasband_synthesize_stream` | Plan, sentence-started, audio and done events, one sentence at a time. |
| `scyllasband_cancel` | Stop requests already issued on the runtime, from any thread. |

Request fields:

| Field | Meaning |
| --- | --- |
| `text`, `voice_id` | Required. |
| `language` | `NULL` or `en`: the voice's own English dialect; otherwise one of the voice's languages. |
| `delivery` | `NULL`/`neutral`, `auto`, or `energy=…,tension=…,valence=…,assertiveness=…,whisper=on/off/auto`; axes 0–4, 2 neutral, `auto` per axis. Omitted axes stay neutral. |
| `speed` | Duration scale; 1 is the model's own pace. |
| `steps`, `sampler` | Flow steps (0: bundle default 8) and Heun or Euler. |
| `seed`, `has_seed` | Sentence *i* of the plan draws its noise from `seed + i`; without a seed every request differs. |
| `temperature` | Noise scale (1 = default). |
| `normalize_text` | Expand numbers, dates, times, currency, symbols and abbreviations before G2P. |

A runtime serializes its requests; use separate runtimes for concurrent synthesis. Stream callbacks run on
the calling thread and audio pointers are valid only during the callback; returning nonzero stops the
request. Errors return a status and set the thread-local `scyllasband_last_error()`.

## Build

The LiteRT C headers (2.2.0) and ONNX Runtime headers (1.30.0) are staged under `third_party/`. Shared
libraries go under `third_party/<sdk>/lib/<platform>/` (`linux-x86_64`, `android-arm64`, `ios-arm64`, …) and
are not committed:

```bash
# LiteRT runtime (and GPU accelerator) prebuilts for the host or another platform
python scripts/stage_litert_sdk.py --download-runtime --download-gpu-accelerator --overwrite
python scripts/stage_litert_sdk.py --platform android-arm64 --download-runtime --download-gpu-accelerator --overwrite

# ONNX Runtime for Linux x86-64
curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.30.0/onnxruntime-linux-x64-1.30.0.tgz | tar xz
mkdir -p third_party/onnxruntime/lib/linux-x86_64
cp -a onnxruntime-linux-x64-1.30.0/lib/libonnxruntime.so* third_party/onnxruntime/lib/linux-x86_64/
```

```bash
cmake -S . -B build/litert -DCMAKE_BUILD_TYPE=Release                          # LiteRT (default)
cmake -S . -B build/onnx -DCMAKE_BUILD_TYPE=Release -DSCYLLASBAND_BACKEND=onnx  # ONNX Runtime
cmake --build build/litert -j
ctest --test-dir build/litert --output-on-failure
```

Apple (macOS; needs the Ninja or Xcode generator for the Swift Core AI bridge):

```bash
cmake -S . -B build/apple -G Ninja -DCMAKE_BUILD_TYPE=Release -DSCYLLASBAND_BACKEND=apple -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0
cmake --build build/apple
build/apple/scyllasband_speak --bundle ../scyllasband/models/coreai --voice scylla --text "Hello." --accelerator auto \
    --output hello.wav
build/apple/scyllasband_speak --bundle ../scyllasband/models/coreai --voice scylla --text "Hello. Again." --timings 4
```

`--timings N` speaks the text N times in one process and prints load, first-audio and total times as JSON. For iOS apps,
use the `ScyllasBandKit` pod (`ScyllasBandKit.podspec`, [apple/README.md](apple/README.md)).

| CMake option | Default | |
| --- | --- | --- |
| `SCYLLASBAND_BACKEND` | `litert` | `litert`, `onnx` or `apple` |
| `SCYLLASBAND_LITERT_INCLUDE_DIR`, `SCYLLASBAND_LITERT_LIBRARY` | staged SDK | LiteRT headers and `libLiteRt` |
| `SCYLLASBAND_ONNXRUNTIME_INCLUDE_DIR`, `SCYLLASBAND_ONNXRUNTIME_LIBRARY` | staged SDK | ONNX Runtime headers and library |
| `SCYLLASBAND_BUILD_TOOLS` | `ON` | `scyllasband_speak` |
| `SCYLLASBAND_BUILD_TESTS` | `ON` | unit tests (and the parity test, see below) |

The library target is `scyllasband_native` (alias `scyllasband::native`). Only `scyllasband_*` symbols are
exported. On Android, pass the `libLiteRt.so` from the `com.google.ai.edge.litert:litert` AAR or the
`libonnxruntime.so` from the `com.microsoft.onnxruntime:onnxruntime-android` AAR for `${ANDROID_ABI}`.

LiteRT runs on CPU through XNNPACK. `SCYLLASBAND_ACCELERATOR_GPU` and `_AUTO` load the GPU accelerator
library from the directory of `libLiteRt` and run the ops it supports on the GPU, the rest on the CPU; with
`_AUTO`, a graph that the GPU accelerator cannot compile runs entirely on the CPU instead of failing.

## Command-line tool

```bash
build/litert/scyllasband_speak --bundle ../scyllasband/models/litert --voice ink \
    --delivery energy=2.4,whisper=off --seed 2027 \
    --text "Keep this between us." --output keep.wav --metadata keep.json
```

`--plan-only` prints the sentence plan; `--stream` prints the stream events.

## Tests

`ctest` runs the unit suites (`events`, `durations`, `context`, `delivery`, `sentences`, `segments`,
`normalizer`, `unicode`, `targets`) against fixtures produced by the Python runtime; `targets` also runs the
streaming loop on fake graphs. Regenerate them after a reference
change:

```bash
PYTHONPATH=.. python tests/generate_fixtures.py
```

`tests/parity_harness.py` compares the built library with the Python runtime on a bundle of the build's
flavor: text normalization, sentence splitting, punctuation segments and Unicode handling; G2P phones and word
starts; per-target durations, latents (given the same noise) and decoded audio with passage context and
prefix; the long-form plan, targets and whole-passage audio. It needs numpy and `onnxruntime` or `ai-edge-litert`:

```bash
PYTHONPATH=.. python tests/parity_harness.py --library build/litert/libscyllasband_native.so \
    --bundle ../scyllasband/models/litert --report parity.json
```

Configure with `-DSCYLLASBAND_PARITY_BUNDLE=<bundle> -DSCYLLASBAND_PARITY_PYTHON=<python>` to run it under
`ctest` as `scyllasband_parity`.

The Unicode tables in `src/scyllasband_unicode_tables.inc` come from Python's `unicodedata`; regenerate them
with `python tools/generate_unicode_tables.py`.
