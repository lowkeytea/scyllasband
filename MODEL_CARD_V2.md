---
license: apache-2.0
language:
  - en
  - es
  - it
  - fr
  - de
  - vi
library_name: onnxruntime
tags:
  - text-to-speech
  - tts
  - onnx
  - litert
  - pytorch
  - speech-synthesis
  - duration-flow
  - multilingual
  - expressive-tts
  - emotion
  - whisper
pipeline_tag: text-to-speech
---

# Scylla's Band v2

Scylla's Band v2 is a multilingual, multi-voice, expressive text-to-speech
model for local and self-hosted inference. It predicts phone durations,
generates continuous acoustic latents with rectified flow, and decodes those
latents to a 24 kHz waveform through a learned acoustic adapter and a frozen
Vocos waveform decoder.

The v2 release expands language and delivery coverage while keeping the same
managed ten-voice identity set. Its primary runtime targets are ONNX Runtime
for desktop, server, Android, and iOS use, plus LiteRT for native and embedded
deployments.

Public resources:

- V2 model bundles and PyTorch checkpoints:
  [`spybyscript/scyllasbandv2`](https://huggingface.co/spybyscript/scyllasbandv2)
- Runtime source: [`lowkeytea/scyllasband`](https://github.com/lowkeytea/scyllasband)
- [Interactive voice, language, and affect samples](https://lowkeytea.github.io/scyllasband/)
- [Scylla's Band Discord](https://discord.gg/cNdBuM3tS)

The Hugging Face repository contains both deployable inference bundles and the
corresponding `.pt` checkpoints. Training data is not distributed.

## What Changed in V2

- French, German, and Vietnamese join English, Spanish, and Italian.
- The public language IDs are `en_us`, `en_gb`, `es`, `it`, `fr`, `de`, and
  `vi`.
- Affect axis-order version 3 exposes five trained dimensions: `calm`, `joy`,
  `anger`, `sadness`, and `whisper`. It does not invent an untrained sarcasm
  coordinate to preserve the shape of v1.
- The affect contract is explicitly versioned in every bundle, preventing v1
  axis positions from being misinterpreted by the v2 runtime.
- Runtime text normalization is expanded for French, German, and Vietnamese,
  including language-specific number and punctuation handling.
- English dialect ownership is explicit. Ink, Orpheus, and Tuesday use
  `en_gb`; the other managed voices use `en_us`. Audio trained as British
  English remains labeled `en_gb` rather than being folded into `en_us`.
- Duration targets use alignment-derived timing tied to the actual audio
  files. The v2 acoustic model is trained from the aligned/eSpeak phone
  representation; the shipped phrase-level G2P is the runtime frontend, not a
  source of acoustic-training labels.
- Duration prediction preserves candidate ordinary word boundaries derived
  from frozen MFA word intervals and jointly predicts whether a pause is
  present and how long it lasts. Spaces are not converted into a uniform
  silence.
- Training includes long and chunked views, explicit punctuation silences,
  three-segment span context, stronger condition dropout, and a vocoder adapter
  trained on both oracle and generated acoustic latents.
- The model repository now ships PyTorch checkpoints with the inference
  bundles instead of placing them in a separate training-model repository.
- LiteRT is shipped alongside ONNX for native and embedded use. The iOS sample
  uses the ONNX bundle in this release.

## Intended Use

Scylla's Band v2 is intended for:

- Speech synthesis with ten managed voices.
- American or British English, Spanish, Italian, French, German, and
  Vietnamese synthesis, subject to each voice's manifest-declared English
  dialect.
- Long-form narration with automatic planning and chunking.
- Multi-voice dialogue from tagged text.
- Continuous and mixed control over calm, joy, anger, sadness, and whisper.
- Cross-platform inference through ONNX Runtime.
- Native and embedded inference through LiteRT.
- Research, inspection, and re-export from the included PyTorch checkpoints.

Scylla's Band is not an arbitrary-speaker cloning system. It is not intended
for impersonation, fraud, deception, or generating speech that falsely
represents a real person as speaking.

## Quick Start

Install the public runtime. Downloads select v2 and its repository by default:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -e .
pip install numpy huggingface_hub onnxruntime

python -m scyllasband download \
    --model-version v2 \
    --runtime-bundles onnx \
    --yes
python -m scyllasband validate-bundle scyllasband/models/v2/onnx
python -m scyllasband speak scyllasband/models/v2/onnx \
    --backend onnx \
    --voice scylla \
    --language en_us \
    --emotion calm=0.5 \
    -o hello.wav \
    "Hello from Scylla's Band v2."
```

An expressive whisper example:

```bash
python -m scyllasband speak scyllasband/models/v2/onnx \
    --backend onnx \
    --voice ink \
    --language en_gb \
    --emotion calm=0.25,whisper=0.8 \
    --emotion-scale 1.25 \
    --steps 8 \
    --sampler heun \
    -o whispered.wav \
    "Keep your voice down. Someone is still in the corridor."
```

The selected bundle's `manifest.json` is authoritative for available bundle
variants, voices, languages, affect axes, shapes, and backend defaults.

## Model Details

| Field | Value |
| --- | --- |
| Model family | Continuous-latent duration/flow TTS |
| Public model version | `2` |
| Primary portable backend | ONNX Runtime |
| Mobile backends | ONNX Runtime and LiteRT |
| License | Apache 2.0 |
| Output sample rate | 24 kHz |
| Acoustic representation | 100 mel bins, 24-dimensional acoustic latents |
| Waveform / latent hop | 256 / 512 samples |
| Public language IDs | `en_us`, `en_gb`, `es`, `it`, `fr`, `de`, `vi` |
| Managed voices | 10 |
| Affect controls | 5 independently scored axes, axis-order version 3 |
| Default quality profile | 8-step Heun sampling |
| Fixed graph budgets | 512 G2P text tokens, 512 phone frames, 640 latent frames |
| Latent target buckets | 256, 384, 512, 640, selected by smallest fit |

Managed voices:

```text
ariadne, felix, gwen, ink, max, orpheus, rex, scylla, stone, tuesday
```

English dialect assignment:

| English language ID | Voices |
| --- | --- |
| `en_gb` | Ink, Orpheus, Tuesday |
| `en_us` | Ariadne, Felix, Gwen, Max, Rex, Scylla, Stone |

All managed voices are trained for Spanish, Italian, French, German, and
Vietnamese. The bundle manifest lists the exact supported languages and default
language for each voice.

## Architecture

```text
text
  -> spoken-text normalization and phrase planning
  -> Scylla's Band phrase-level G2P
  -> phone IDs, punctuation, and boundary/context features
  -> duration prediction
  -> duration-expanded phone conditioning
  -> rectified-flow acoustic latent estimation
  -> Scylla's Band acoustic adapter and frozen Vocos decoder
  -> 24 kHz waveform
  -> long-form assembly when needed
```

| Component | Details |
| --- | --- |
| Text frontend | Seven-language phrase-level G2P with fixed 512-token input budget |
| Duration predictor | 192 hidden size, 4 layers, 4 heads, up to 512 phone positions |
| Acoustic generator | 24-D rectified-flow latents, 512 hidden size, 12 layers, 8 heads, AdaLN conditioning, QK normalization |
| Span context | Three context segments over up to 768 phones |
| Conditioning | Voice, language, boundary/span context, five-axis affect, and schema-v4 identity/prosody reference features |
| Vocoder | Six-layer, 384-channel acoustic adapter into frozen `charactr/vocos-mel-24khz` |

V2 uses explicit punctuation silence targets plus learned, MFA-derived pause
candidates at ordinary word boundaries. The runtime protects sentence-ending
punctuation with a 107 ms minimum while leaving ordinary word-boundary timing
under model control. Long-form planning prefers real clause punctuation when a
chunk must be divided so learned timing survives the join.

## Affect and Whisper Controls

V2 exposes four core delivery axes and one overlay:

```text
core:     calm, joy, anger, sadness
overlay:  whisper
```

Each axis is independently bounded in `[0, 1]` and multiple axes may be
nonzero. For example, whisper can be combined with sadness or anger instead of
being selected as a mutually exclusive speaking style.

With no affect supplied, the neutral preset is `[0.5, 0.25, 0, 0, 0]`
(`calm=0.5,joy=0.25`). A partial affect request starts from `calm=0.5` and
zeroes the other axes before applying the supplied values. Thus
`anger=0.25` means `[0.5, 0, 0.25, 0, 0]`, not an unsupported calm-zero
condition. Explicit calm is supported over `0.25` to `0.75`; runtimes reject
or floor zero according to the selected bundle contract.

Training ratings use a `0` to `4` scale and are normalized for inference:

| Human rating | Runtime value |
| ---: | ---: |
| 0 | 0.0 |
| 1 | 0.25 |
| 2 | 0.5 |
| 3 | 0.75 |
| 4 | 1.0 |

`--emotion-scale` controls classifier-free guidance separately from the axis
values:

```text
guided = null + scale * (conditioned - null)
```

`1` uses the requested affect vector directly. Values around `1.25` to `1.5`
are a practical first range for stronger delivery. Higher values extrapolate
beyond direct conditioning and can produce exaggerated timing, voice drift, or
distortion. Guidance scales other than `1` require both null and conditioned
model evaluation.

Whisper response varies by voice, language, wording, sampler, and strength. It
is a learned delivery overlay, not a post-processing whisper effect.

## Runtime Bundles

The Hugging Face repository uses one directory per runtime variant:

```text
onnx/       # FP32 ONNX Runtime bundle
onnx-int8/  # dynamically quantized ONNX Runtime bundle
litert/     # LiteRT native and embedded bundle
pytorch/    # matching training and export checkpoints
```

### ONNX

ONNX Runtime is the primary cross-platform inference path. Full-precision and
INT8 variants use the same manifest-driven runtime contract.
The export uses opset 18, target-bucket graphs, and shared external weights.

```text
onnx/
  manifest.json
  onnx/g2p/model.onnx
  onnx/g2p/tokenizer.json
  onnx/components/duration_predictor.onnx
  onnx/components/vector_context_encoder.onnx
  onnx/components/vector_estimator_b{256,384,512,640}.onnx
  onnx/components/vocoder_adapter_b{256,384,512,640}.onnx
  onnx/components/vocoder_b{256,384,512,640}.onnx
  onnx/components/shared_weights.bin
  assets/
```

### LiteRT

The LiteRT bundle contains the same duration, flow, reference-routing, G2Pv2,
and acoustic-adapter/Vocos path for native and embedded runtimes. The native
runtime consumes stored (uncompressed) schema-v4 NPZ voice packs directly.

```text
litert/
  manifest.json
  litert/g2p.tflite
  litert/duration_predictor.tflite
  litert/vector_context_encoder.tflite
  litert/vector_estimator_{256,384,512}.tflite
  litert/vector_estimator.tflite
  litert/vocoder_adapter_{256,384,512}.tflite
  litert/vocoder_adapter.tflite
  litert/vocoder_{256,384,512}.tflite
  litert/vocoder.tflite
  assets/
```

Download and run it with:

```bash
python -m scyllasband download \
    --model-version v2 \
    --runtime-bundles litert \
    --yes
python -m scyllasband speak scyllasband/models/v2/litert \
    --backend litert \
    --voice gwen \
    --language en_us \
    -o hello_litert.wav \
    "Hello from LiteRT."
```

ONNX and LiteRT are the primary v2 release targets. The selected manifest
declares the exact graphs, buckets, controls, and runtime requirements.

## PyTorch Checkpoints

The v2 Hugging Face repository keeps export and research checkpoints beside
the inference bundles:

```text
pytorch/
  duration.pt
  vector_estimator.pt
  vocoder.pt
  autoencoder.pt
  g2p.pt
  scyllasband_v2_clean.json
```

These files are not required for normal ONNX or LiteRT inference. Components
must be used with their matching configuration, condition indexes, phone
vocabulary, G2P tokenizer, and language/affect metadata. Mixing checkpoints or
sidecars from another release is unsupported.

## Training Data and Labels

Training data is not distributed. V2 was trained primarily on synthetic speech
covering the managed voices and seven public language IDs. Text generation was
designed for broad vocabulary, difficult pronunciations, heteronyms, numbers,
acronyms, varied sentence lengths, and emotional delivery.

Prepared audio was checked with ASR, alignment, acoustic-quality detection,
and human review. Selected clips were regenerated at higher synthesis quality
when the original contained clipping, micro-stutters, chirps, unstable pitch,
or other mechanical artifacts. Duration supervision comes from forced
alignment against the actual audio rather than a duration value stored in the
source manifest.

The affect dataset is rated independently across all five axes. Multiple
nonzero values can describe one clip, allowing mixtures such as quiet anger,
sad joy, or emotional whispering. Ratings are used as
continuous supervision rather than treating the original requested emotion as
ground truth.

## Validation

Release validation should keep the same text, phones, voice, language, affect,
guidance, sampler, steps, seed, and duration scale when comparing PyTorch,
ONNX, and LiteRT. The public validation tooling produces per-voice,
per-language, and per-affect ASR/WER reports with review audio and transcript
diffs.

Useful checks:

```bash
python -m scyllasband validate-bundle scyllasband/models/v2/onnx
python -m scyllasband validate-bundle scyllasband/models/v2/litert
python -m scyllasband compare-metadata LEFT.json RIGHT.json
```

The selected bundle's manifest, not this prose, is authoritative for finalized
artifacts and controls.

## Limitations

- The model supports managed voices rather than arbitrary speaker cloning.
- Each voice uses one manifest-declared English dialect; requesting the other
  English dialect for that voice is unsupported.
- Affect and whisper response varies by voice, language, wording, sampler,
  steps, and CFG scale.
- Strong CFG can exaggerate timing or destabilize identity.
- Very long text is synthesized in planned chunks; unusual fragments and poor
  punctuation can still produce awkward pacing.
- Names, rare words, code, malformed input, and language-mismatched text can be
  mispronounced.
- The FP32 and INT8 G2P graphs can select different phones on low-confidence
  or rare phrases even when the acoustic graphs otherwise use matching inputs.
- Synthetic training data can preserve pronunciation, prosody, dialect, and
  language biases from its source systems.
- LiteRT acceleration depends on the host runtime and available accelerator;
  CPU fallback remains important for unsupported graphs.

## Safety and Misuse

Do not use Scylla's Band to impersonate people, deceive listeners, bypass
consent, or create speech that represents a real person as saying something
they did not say. Generated audio should be disclosed as synthetic whenever
the context could otherwise confuse a listener.

## License

Scylla's Band v2 is released under Apache 2.0. See [`LICENSE`](LICENSE).

## Citation

```bibtex
@software{spybyscript_scyllasband_v2_2026,
  author = {Spybyscript},
  title = {Scylla's Band v2: Multilingual Expressive Duration-Flow Text-to-Speech},
  year = {2026},
  url = {https://huggingface.co/spybyscript/scyllasbandv2},
  license = {Apache-2.0}
}
```

## Acknowledgments

Scylla's Band uses ONNX Runtime and LiteRT for portable and native inference.
Vocos provides the frozen 24 kHz waveform-decoder
backbone. Wiktionary-derived vocabulary, eSpeak phonemization, and Montreal
Forced Aligner tooling contributed to text coverage and duration supervision.
