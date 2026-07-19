---
license: apache-2.0
language:
  - en
  - es
  - it
library_name: onnxruntime
tags:
  - text-to-speech
  - tts
  - onnx
  - litert
  - speech-synthesis
  - duration-flow
  - multilingual
  - expressive-tts
  - emotion
  - cpu-inference
pipeline_tag: text-to-speech
---

# Scylla's Band

Scylla's Band is a multilingual, multi-voice, expressive text-to-speech model for local and self-hosted inference. It predicts phone durations, generates continuous acoustic latents with rectified flow, and decodes those latents to a 24 kHz waveform through a Scylla's Band acoustic adapter and Vocos vocoder.

Scylla's Band is developed and published by [Spybyscript](https://huggingface.co/spybyscript).

Public resources:

- Model bundles and voice assets: [`spybyscript/scyllasband`](https://huggingface.co/spybyscript/scyllasband)
- Runtime source: [`lowkeytea/scyllasband`](https://github.com/lowkeytea/scyllasband)
- [Interactive voice, language, and affect samples](https://lowkeytea.github.io/scyllasband/)

The public resources are inference-only. Training data, trainer checkpoints, and export tooling are not distributed with the model.

## Intended Use

Scylla's Band is intended for:

- Single-voice speech synthesis with ten managed voices.
- English, Spanish, and Italian synthesis.
- Long-form narration with automatic planning and chunking.
- Multi-voice dialogue from tagged text.
- Continuous control over calm, joy, anger, sadness, sarcasm, and questioning delivery.
- Desktop and server inference through ONNX Runtime.
- Native and mobile execution through ONNX Runtime and `libscyllasband`, with LiteRT available as an explicit experimental backend.

Scylla's Band is not an arbitrary-speaker cloning system. It is not intended for impersonation, fraud, deception, or generating speech that falsely represents a real person as speaking.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -e .
pip install numpy huggingface_hub onnxruntime

python -m scyllasband download
python -m scyllasband validate-bundle
python -m scyllasband speak \
    --voice scylla \
    --language en_us \
    --emotion calm=0.5 \
    -o hello.wav \
    "Hello from Scylla's Band."
```

Download both public backend bundles with:

```bash
python -m scyllasband download --runtime-bundles both
```

An expressive example:

```bash
python -m scyllasband speak \
    --backend onnx \
    --voice gwen \
    --language en_us \
    --emotion joy=0.75,sarcasm=0.25 \
    --emotion-scale 1.5 \
    --steps 8 \
    --sampler heun \
    -o expressive.wav \
    "Oh, brilliant! The dragon ate the map, and somehow this is going perfectly."
```

## Model Details

| Field | Value |
| --- | --- |
| Model family | Continuous-latent duration/flow TTS |
| Public bundle contract | `1.0.0` / `scyllasband-duration-flow` |
| Main public backend | ONNX Runtime |
| Additional backend | LiteRT with native `libscyllasband` support |
| License | Apache 2.0 |
| Output sample rate | 24 kHz |
| Acoustic representation | 100 mel bins, 24-dimensional acoustic latents |
| Hop lengths | 256 waveform hop, 512 latent hop |
| Public languages | `en_us`, `en_gb`, `es`, `it` |
| Managed voices | 10 |
| Affect controls | 6 independently scored axes |
| CLI quality default | 8-step Heun sampling |
| Fixed graph budgets | 512 G2P text tokens, 512 phone frames, 640 latent frames |
| Latent target buckets | 256, 384, 512, 640, selected by smallest fit |

Managed voices:

```text
ariadne, felix, gwen, ink, max, orpheus, rex, scylla, stone, tuesday
```

English defaults to `en_us` for most voices. Ink, Orpheus, and Tuesday default to `en_gb`. Every managed voice supports `en_us`, `en_gb`, `es`, and `it` in the current public bundles.

## Architecture

```text
text
  -> spoken-text normalization and phrase planning
  -> Scylla's Band G2P
  -> phone IDs, punctuation, and boundary/context features
  -> duration prediction
  -> duration-expanded phone conditioning
  -> rectified-flow acoustic latent estimation
  -> Scylla's Band acoustic adapter and Vocos vocoder
  -> 24 kHz waveform
  -> long-form assembly when needed
```

| Component | Details |
| --- | --- |
| Text frontend | Phrase-level multilingual G2P with a 74-phone vocabulary and fixed 512-token input budget |
| Duration predictor | 192 hidden size, 4 layers, 4 heads, up to 512 phone positions |
| Acoustic generator | 24-dimensional rectified-flow latents, 512 hidden size, 12 layers, 8 heads, AdaLN conditioning, and QK normalization |
| Span context | Three context segments over up to 768 phones with a 512-dimensional context state |
| Prefix context | Up to 24 acoustic latent frames from the preceding chunk |
| Voice references | 128-dimensional style and 32-dimensional prosody features |
| Vocoder | Scylla's Band acoustic adapter into a frozen `charactr/vocos-mel-24khz` backbone |

The duration predictor and acoustic generator are conditioned on voice, language, reference features, boundary labels, and the six-axis affect vector. The acoustic generator additionally receives prefix latents and span context for long-form continuity. Voice/reference conditioning remains active in the affect CFG null branch so guidance changes delivery without intentionally removing speaker identity.

Punctuation boundaries use explicit silence phones. Runtime pause floors apply to internal sentence and clause boundaries, while the final learned silence keeps its predicted duration and host assembly supplies clean inter-chunk silence.

## Affect Controls

Scylla's Band exposes four core delivery axes and two overlays:

```text
core:     calm, joy, anger, sadness
overlays: sarcasm, questioning
```

Each value is independently bounded in `[0, 1]`. Multiple values may be nonzero, so a request can combine a core delivery with sarcasm or questioning instead of selecting one mutually exclusive category.

The training ratings used a `0` to `4` scale and are normalized for inference:

| Human rating | Runtime value |
| ---: | ---: |
| 0 | 0.0 |
| 1 | 0.25 |
| 2 | 0.5 |
| 3 | 0.75 |
| 4 | 1.0 |

Values around `0.5` or `0.75` often provide more natural range than setting every requested axis to `1.0`. The shipped presets use `calm=0.5`, `joy=0.75`, `anger=0.75`, and `sadness=0.75`; sarcasm and questioning presets combine a quieter calm core with a `0.75` overlay.

Affect classifier-free guidance is controlled separately:

```text
guided = null + scale * (conditioned - null)
```

- `0` selects the learned null-affect prediction.
- `1` uses the requested affect vector directly.
- Values above `1` strengthen the conditional difference.
- The runtime accepts finite non-negative values without a fixed upper cap.

The selected flow estimator was trained with `0.15` affect-condition dropout to provide the null branch used by CFG. A practical first range for stronger delivery is `1.25` to `1.5`. Higher values are extrapolation and can produce exaggerated timing, voice instability, or distortion depending on the voice and sentence. Guidance scales other than `1` also require both null and conditioned model evaluation and therefore cost more than direct conditioning.

## Runtime Bundles

The selected bundle's `manifest.json` is authoritative for its shapes, controls, components, and preferred backend.

### ONNX

ONNX is the default path across Python, desktop/server, native `libscyllasband`, and the Android sample. The export uses opset 18, four full acoustic-generator/vocoder target buckets, and shared external weights. CPU execution through `CPUExecutionProvider` is the validated baseline.

```text
scyllasband/models/onnx/
  manifest.json
  onnx/g2p/model.onnx
  onnx/components/duration_predictor.onnx
  onnx/components/vector_context_encoder.onnx
  onnx/components/vector_estimator_b{256,384,512,640}.onnx
  onnx/components/vocoder_b{256,384,512,640}.onnx
  onnx/components/vocoder_adapter_b{256,384,512,640}.onnx
  onnx/components/shared_weights.bin
  onnx/components/shared_weights.json
  assets/
```

### LiteRT

LiteRT provides an explicit experimental native/mobile bundle for `libscyllasband`; it is never selected as an implicit fallback. The current public bundle uses one full acoustic-generator graph per target bucket rather than separate prefix/tail graphs. Accelerator support and fallback policy remain platform- and runtime-dependent.

```text
scyllasband/models/litert/
  manifest.json
  litert/g2p.tflite
  litert/duration_predictor.tflite
  litert/vector_context_encoder.tflite
  litert/vector_estimator_{256,384,512}.tflite
  litert/vector_estimator.tflite
  litert/vocoder_{256,384,512}.tflite
  litert/vocoder.tflite
  litert/vocoder_adapter_{256,384,512}.tflite
  litert/vocoder_adapter.tflite
  assets/
```

Both bundles include phone and language indexes, G2P sidecars, affect metadata, and managed voice/reference packs under `assets/`.

## Long-Form Synthesis

Long-form planning is enabled by default in the CLI. It provides:

- Normalization before chunk planning.
- Sentence, paragraph, clause, and artificial-continuation boundary labels.
- Duration preflight and retry splitting for fixed graph budgets.
- Prefix-latent carryover and span/lookahead context.
- Internal punctuation pause floors.
- Short fades and clean host-side silence at assembled boundaries.

Use `--no-chunk-text` only when intentionally debugging one raw synthesis request.

## Training Data and Labels

Training data is not distributed with the public inference resources. Scylla's Band was trained primarily on synthetic multilingual speech. Seed vocabulary came from Wiktionary-derived word lists, scripted prompt generation placed target words in varied sentences, and the text was rendered as synthetic speech across the managed voices and languages.

Prepared audio was filtered with text/audio validation, ASR checks, alignment targets, and audio-quality review. Alignment-derived durations supervise the duration predictor and explicit punctuation silences.

The original desired-delivery categories were not treated as ground truth. A human-validated pilot was scored across calm, joy, anger, sadness, sarcasm, and questioning, allowing multiple nonzero values for one clip. Those labels were used to fit a programmatic rescoring process for the remaining emotional dataset. This gives the model continuous per-axis supervision instead of one mutually exclusive emotion label.

## Validation

The public ONNX and LiteRT bundles pass the 1.0 bundle contract. Machine-local `export_status.json` files are exporter diagnostics and are intentionally omitted from the public inference payload. Runtime validation uses matched text, voice, language, affect, guidance, sampler, steps, and seed, and checks each backend against its validated deterministic reference output.

Useful checks:

```bash
python -m scyllasband validate-bundle scyllasband/models/onnx
python -m scyllasband validate-bundle scyllasband/models/litert

python -m scyllasband speak scyllasband/models/onnx \
    --backend onnx \
    --voice scylla \
    --language en_us \
    --emotion anger=0.75 \
    --emotion-scale 1.5 \
    --metadata /tmp/scyllasband_onnx_smoke.json \
    -o /tmp/scyllasband_onnx_smoke.wav \
    "That was not the agreement."
```

Compare normalized synthesis metadata with:

```bash
python -m scyllasband compare-metadata LEFT.json RIGHT.json
```

The public sample gallery contains 30 matched multilingual voice previews and 50 affect demonstrations covering all ten voices.

## Limitations

- Public text input is limited to `en_us`, `en_gb`, `es`, and `it`.
- The model uses managed voice/reference packs and does not accept arbitrary speaker cloning references.
- Affect response varies with the voice, language, wording, requested mixture, sampler, steps, and CFG scale. Some voices and sentences are naturally subtler than others.
- Strong CFG can exaggerate timing or destabilize the voice even when the six-axis values themselves remain within range.
- ONNX, LiteRT, and the training runtime can have small numerical or acoustic differences. Validate the chosen backend on representative text and target hardware.
- Long-form output is planned in chunks. Very unusual boundaries or extremely short fragments may still sound less natural than complete sentences.
- Names, rare words, code, malformed text, and language-mismatched input can be mispronounced.
- Synthetic training data can carry the pronunciation, prosody, and language biases of its source systems and filtering process.

## Safety and Misuse

Do not use Scylla's Band to impersonate people, deceive listeners, bypass consent, or create speech that represents a real person as saying something they did not say. Generated audio should be disclosed as synthetic whenever the context could otherwise confuse a listener.

## License

Scylla's Band is released under Apache 2.0. See [`LICENSE`](LICENSE).

## Citation

```bibtex
@software{spybyscript_scyllasband2026,
  author = {Spybyscript},
  title = {Scylla's Band: Multilingual Expressive Duration-Flow Text-to-Speech},
  year = {2026},
  url = {https://huggingface.co/spybyscript/scyllasband},
  license = {Apache-2.0}
}
```

## Acknowledgments

Scylla's Band uses ONNX Runtime as the default Python, desktop/server, native, and mobile backend; LiteRT is an explicit experimental alternative. Vocos provides the frozen 24 kHz vocoder backbone. Wiktionary-derived vocabulary and alignment tooling contributed to data preparation and duration supervision.
