# libscyllasband

`libscyllasband` is the native C++ runtime for Scylla's Band model bundles (graph contract
`scyllasband_measured_controls_v3`). It implements the same pipeline as the Python runtime and produces
the same durations, latents and audio for the same inputs. The public C ABI is `include/scyllasband.h`.

Each build links one graph runtime:

| Backend | Bundle | Download |
| --- | --- | --- |
| LiteRT 2.2 (default) | `scyllasband/models/litert` | `python -m scyllasband download --flavor litert` |
| ONNX Runtime 1.20+ | `scyllasband/models/onnx` | `python -m scyllasband download --flavor onnx` |
| Apple: Core AI (iOS/visionOS/macOS 27) and Core ML (iOS 18 / visionOS 2 / macOS 15) | `scyllasband/models/coreai`, `scyllasband/models/coreml` | `python -m scyllasband download --flavor coreai` (or `coreml`) |

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
   each side from the neighbouring text, and the target's frames are rounded to integers, with pauses scaled
   by the requested pace (`controls.pace_pauses`);
2. per-frame phone ids and text events (punctuation, word starts, stress and length marks, phone phase,
   sentence type) are built;
3. the pitch predictor integrates noise to a pitch and voicing contour for those frames (its sampler and steps
   come from `controls.pitch`); the pitch shift is added, the voice's soft limit applied, and the contour becomes
   the flow's melody input;
4. the flow integrates noise to latents with Euler (default, 4 steps) or Heun steps in the smallest fitting
   latent-frame bucket, conditioned on the span, the melody and the last 96 latent frames already spoken;
5. the vocoder decodes the target with up to 48 frames of the preceding latents as left context, so
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
request.delivery = "intensity=1,pace=-0.5";   /* NULL: the voice's own delivery */
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
| `scyllasband_voices_json` | Backend, sample rate, release, voices with their languages, and the delivery ranges and per-voice defaults. |
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
| `delivery` | `NULL`/`default`, or `intensity=…,pace=…,pitch=…`: intensity (calm, negative, to intense, positive) and pace (slower to faster, words and pauses) in the voice's own standard deviations, pitch as a shift in semitones; each clamped into the bundle's ranges (−2 to 2 in the released model). Omitted controls take the voice's default. A target of fewer than four spoken phones (a lone "Yes.") gets the voice's own intensity and pace, as in training. |
| `speed` | A plain time stretch on top of pace; 1 is none. |
| `steps`, `sampler` | Flow steps (0: the bundle default, 4) and sampler (default: the bundle's, Euler). |
| `seed`, `has_seed` | Sentence *i* of the plan draws its noise from `seed + i`; without a seed every request differs. |
| `temperature` | Noise scale (1 = default). |
| `normalize_text` | Expand numbers, dates, times, currency, symbols and abbreviations before G2P. |

A runtime serializes its requests; use separate runtimes for concurrent synthesis. Stream callbacks run on
the calling thread and audio pointers are valid only during the callback; returning nonzero stops the
request. Errors return a status and set the thread-local `scyllasband_last_error()`.

### Records (ABI 3.1)

A records request speaks several voices, languages and deliveries in one go, as `group-speak` does:

```c
ScyllasBandRecordsRequest request;
scyllasband_records_request_init(&request);   /* sets struct_size, which every call checks */
request.records_json = "[{\"text\": \"Hello there.\", \"voice\": \"scylla\"},"
                       " {\"text\": \"It has been a while.\", \"voice\": \"ink\", \"language\": \"en_gb\", \"pace\": -0.5}]";
request.pause_ms = 250;
request.seed = 2027;
request.has_seed = 1;
scyllasband_synthesize_records_stream(runtime, &request, on_event, user_data);
```

| Function | Purpose |
| --- | --- |
| `scyllasband_records_request_init` | Defaults and `struct_size` for a `ScyllasBandRecordsRequest`. |
| `scyllasband_plan_records_json` | The sentence plan for the records, without synthesizing. |
| `scyllasband_synthesize_records_stream` | The records spoken target by target, with the same events as `scyllasband_synthesize_stream`. |

Each record is `{"text", "voice", "language"?, "delivery"?, "intensity"?, "pace"?, "pitch"?}`: the delivery as a
`control=value` string or a `{"intensity", "pace", "pitch"}` object, or the controls as fields, not both. Consecutive
records with the same voice, language and delivery form one chain, spoken continuously with the previous target's
acoustics as context; a record that changes any of the three starts a new chain, even when it has no text, and
`pause_ms` of silence opens every chain after the first (at the start of that target's audio). Chunk *i* of the
whole plan, counted across records, draws its noise from `seed + i`. The rules and messages are the Python runtime's (`planner.plan_records`).

### Singing (ABI 3.1)

A v4 bundle (`voices_json` → `"singing": {"supported": true, ...}`) sings a melody score:

```c
ScyllasBandSingRequest request;
scyllasband_sing_request_init(&request);   /* sets struct_size, which every call checks */
request.score_json = "{\"text\": \"hold on tight\", \"bpm\": 104, \"style\": \"pop\","
                     " \"notes\": [[\"hold\", \"E3\", 1], [\"on\", \"G3\", 1], [\"tight\", \"A3\", 2]]}";
request.voice_id = "gwen";
request.decoder = "vocos";                 /* NULL: the bundle's singing decoder (BigVGAN); "vocos": turbo */
request.seed = 7;
request.has_seed = 1;
scyllasband_sing_stream(runtime, &request, on_event, user_data);
```

| Function | Purpose |
| --- | --- |
| `scyllasband_sing_request_init` | Defaults and `struct_size` for a `ScyllasBandSingRequest`. |
| `scyllasband_score_parse` | Checks a score without a runtime and returns it normalized (notes with MIDI and times, words, range, lines); an error names the note at fault. |
| `scyllasband_sing_fit_json` | How the song sits in the voice's range after transpose and octave register, with a suggested transpose. No graph runs. |
| `scyllasband_sing_plan_json` | The phrase plan (phones, durations, notes per phrase) and the settings a sing would use. |
| `scyllasband_sing` | The whole song as one waveform, silence in the rests. |
| `scyllasband_sing_stream` | Plan, phrase-started, audio and done events; the audio events are contiguous, so a player appends them. |
| `scyllasband_sing_preview` | The score as plain tones at the pitch it will be sung, for an instant preview. No graph runs. |
| `scyllasband_warmup_singing` | Loads the graphs a short phrase needs with a decoder. |

A score is `{"text", "bpm", "style": "pop" | "ballad", "notes": [[label, note, beats], ...]}`: a note is a pitch name with
octave (`C4`, `F#3`, `Bb3`) or `rest` (label `-`), and a word over several notes is numbered (`ba~1`, `by~2`). The song is
split into phrases at rests of 0.25 s or more, moved by octaves into the voice's sung range (`use_register`) after
`transpose`, and each phrase is sung with the bundle's singing settings (`v4-20261106`: 4 Euler steps without guidance;
`v4-20261105`: 8 Heun steps, guidance 2) through BigVGAN, or
Vocos with `decoder = "vocos"`. Phrase *i* draws its noise from `seed + i`. Score parse, fit and preview don't wait for a
request in progress, so an editor can call them while a song renders.

On LiteRT each bucket graph a process opens keeps its own working memory, about 2 MB per latent frame for BigVGAN, so a
song whose phrases fall into several buckets holds several. A decoder window decodes every phrase through one bucket graph
instead, with `context` frames of overlap either side of each cut: `controls.decoders.bigvgan.window: {"frames": 256,
"context": 16}` in the manifest, or `SCYLLASBAND_DECODER_WINDOW=bigvgan=256:16` (`bigvgan=0` turns it off). The 71 s
example song then peaks at 1.5 GB instead of 3.7 GB on a Galaxy Z Fold 8 (3 % slower) and at 1.0 GB instead of 3.5 GB on a
Mac (7 % slower), with audio within 68 dB of whole-phrase decoding (16 frames of overlap is enough; 8 leaves audible
joins). A phrase shorter than the window is padded to it: a 3.6 s phrase (176 frames) takes 36 % longer in a 256-frame
window than in its own 192-frame bucket. Only a decoder whose output is whole frames (BigVGAN) can window. A windowed
decoder decodes any length, so a bundle may carry it at the window's bucket only (its phrase limit is then the flow's):
the LiteRT bundle's BigVGAN as the one 256-frame signature peaks at 1.4 GB on the Fold for the same song, at the same
speed. FP16 BigVGAN weights halve the download but cost memory on LiteRT's CPU path: XNNPACK expands them to FP32 when
it loads them, and its FP16 compute mode cannot use the weight cache in LiteRT 2.2.0. They suit the GPU: a singing LiteRT
bundle can name BigVGAN as its one GPU graph (`controls.litert.gpu: {"graphs": ["vocoder_bigvgan"], "precision":
"fp16_fp32_accumulation"}`), so `SCYLLASBAND_ACCELERATOR_AUTO` sings through the GPU and speaks on the CPU. With
datasetGen's channels-last BigVGAN graph (`full_retrain/bigvgan_nwc.py`; the v4-20261105 graph does not run on the GPU
delegate) the Fold renders the 71 s song in 33 s at 764 MB, against 90 s on its CPU.

`ScyllasBandRecordsRequest` and `ScyllasBandSingRequest` start with `struct_size`; initialize requests with their
init function so a caller built against another layout gets an error instead of misread fields. Every 3.0.0 type and
function is unchanged in 3.1 (`scyllasband_abi` checks it).

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

LiteRT runs on CPU through XNNPACK. `SCYLLASBAND_ACCELERATOR_GPU` loads the GPU accelerator library from the directory of
`libLiteRt` and runs every graph's supported ops on the GPU, the rest on the CPU. `_AUTO` puts only the graphs the bundle
lists in `controls.litert.gpu.graphs` on the GPU (a singing bundle lists BigVGAN) and runs everything else on the CPU, and
every graph of a bundle that lists none: with LiteRT 2.2.0 the duration predictor returns wrong durations on the GPU. A
graph the GPU accelerator cannot compile runs entirely on the CPU instead of failing.

## Command-line tool

```bash
build/litert/scyllasband_speak --bundle ../scyllasband/models/litert --voice ink \
    --delivery intensity=1.5,pace=-0.5 --seed 2027 \
    --text "Keep this between us." --output keep.wav --metadata keep.json
```

`--plan-only` prints the sentence plan; `--stream` prints the stream events.

```bash
build/litert/scyllasband_speak --bundle ../scyllasband/models/litert --voice gwen \
    --score ../scyllasband/models/melodies/phrase_pop_02.json --seed 7 --output phrase.wav   # BigVGAN
build/litert/scyllasband_speak --bundle ../scyllasband/models/litert --voice gwen \
    --score ../scyllasband/models/melodies/phrase_pop_02.json --turbo-sing --timings 3       # Vocos, timed
```

`--score` sings a score on a v4 bundle, with `--decoder bigvgan|vocos` (`--turbo-sing` = Vocos), `--sing-steps`,
`--guidance`, `--transpose` and `--no-register`.

## Tests

`ctest` runs the unit suites (`events`, `durations`, `context`, `delivery`, `sentences`, `segments`,
`normalizer`, `unicode`, `targets`, `singing`, `records`) against fixtures produced by the Python runtime; `targets` also runs the
streaming loop on fake graphs, and `records` plans records (delivery forms, chains, errors) and streams them with
`pause_ms`. `scyllasband_abi` checks that the 3.0.0 types keep their layout and every 3.0.0 symbol is still exported
(`fixtures/abi_3_0_layout.txt`, `fixtures/abi_3_0_symbols.txt`). Regenerate the fixtures after a reference
change:

```bash
PYTHONPATH=.. python tests/generate_fixtures.py
```

The pronunciation lexicon is embedded from `scyllasband/lexicon.py` as `src/scyllasband_lexicon.inc`; regenerate it
after a lexicon change (the Python tests fail while it is out of date):

```bash
python tools/generate_lexicon.py
```

`tests/parity_harness.py` compares the built library with the Python runtime on a bundle of the build's
flavor: text normalization, sentence splitting, punctuation segments and Unicode handling; G2P phones and word
starts; per-target durations, latents (given the same noise) and decoded audio with passage context and
prefix; the long-form plan, targets and whole-passage audio; records plans, error messages and the records stream with
`pause_ms`. It loads the library through `scyllasband.native`, the Python binding, and needs numpy and `onnxruntime` or
`ai-edge-litert`:

```bash
PYTHONPATH=.. python tests/parity_harness.py --library build/litert/libscyllasband_native.so \
    --bundle ../scyllasband/models/litert --report parity.json
```

Configure with `-DSCYLLASBAND_PARITY_BUNDLE=<bundle> -DSCYLLASBAND_PARITY_PYTHON=<python>` to run it under
`ctest` as `scyllasband_parity`.

The Unicode tables in `src/scyllasband_unicode_tables.inc` come from Python's `unicodedata`; regenerate them
with `python tools/generate_unicode_tables.py`.
