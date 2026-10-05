---
license: apache-2.0
language: [en, es, it, fr, de, vi]
library_name: litert
pipeline_tag: text-to-speech
tags: [text-to-speech, tts, litert, onnx, pytorch, multilingual, expressive-tts, duration-flow, whisper]
---

# Scylla's Band v2

Release **`v2-20261005`**. The [Scylla's Band runtime](https://github.com/lowkeytea/scyllasband) supports this release.

The model exposes **energy, tension, valence and assertiveness** on **0–4 scales with neutral at 2**, plus **binary whisper**. Combining the controls changes the generated delivery; a request is not a guarantee of a particular perceived emotion or exact external scorer value.

## Quick start

Install the [runtime](https://github.com/lowkeytea/scyllasband) (`pip install -e .` from a checkout), then:

```bash
python -m scyllasband download --yes
python -m scyllasband speak --voice ariadne --language en_us \
    --delivery energy=2.3,tension=1.7,valence=2.6,assertiveness=2,whisper=off \
    --sampler heun --steps 8 -o hello.wav "It is a pleasure to meet you."
```

The default download is the LiteRT bundle, installed under `scyllasband/models/litert`. `--flavor onnx` downloads the ONNX Runtime bundle instead (`pip install onnxruntime`). Downloads pin this release tag. The manifest declares graph contract `scyllasband_measured_delivery_v2`; runtimes written for earlier releases cannot run these graphs.

## Controls

| Input | Values | Default |
| --- | --- | --- |
| Energy, tension, valence, assertiveness | 0–4 or unspecified | 2 each |
| Whisper | on/off or unspecified | off |

With every control at 2, the default, voices speak in a calm, conversational delivery with a neutral tone. Moving a control away from 2 changes the delivery:

| Control | Toward 0 | Toward 4 |
| --- | --- | --- |
| Energy | Lower, darker voice with more pauses between phrases | Higher, brighter, more projected voice |
| Tension | Slightly looser and steadier | Tighter, with wider pitch swings |
| Valence | Flatter pitch, a little slower, longer pauses | Livelier pitch, a little quicker, shorter pauses |
| Assertiveness | Hesitant, with longer pauses between phrases | Faster and clipped with few pauses; the strongest control over pace |

Whisper on gives quieter, breathy speech. Combining the controls gives familiar deliveries:

| Delivery | Settings |
| --- | --- |
| Calm and soft | `energy=1.2,tension=0.8,valence=2.6` |
| Assertive and rapid-fire | `energy=2.6,tension=2.2,assertiveness=3.6` |
| Joyful | `energy=3.2,valence=3.6` |
| Angry | `energy=3.4,tension=3.4,valence=0.8,assertiveness=3.4` |
| Sad | `energy=1,tension=1.6,valence=0.8,assertiveness=1.6` |
| Whispered | `whisper=on` |

The [voice gallery](https://lowkeytea.github.io/scyllasband/#delivery) plays each of these for all ten voices with the exact settings used. How far a control moves the delivery varies with voice, language and text, so start near 2 and listen as you move toward the extremes.

`--delivery auto` omits all coordinates. A per-axis `auto` omits just that coordinate; other unspecified fields in a partial request use neutral defaults. Omission is represented by a separate mask, not by the value 2. It is not an automatic text-to-performance planner.

The graph receives `(rating - 2) / 4` for the four continuous axes and 0/1 for whisper, plus five presence bits. The public interface accepts the original 0–4 values. Expressiveness is an auxiliary training target, not an inference axis.

Controls are **global within each request**. This release does not support control curves, singing, or trained automatic variation over a multi-sentence passage.

## Model and inference

| Property | Value |
| --- | --- |
| Public model version | 2 |
| Architecture | Phone-duration predictor → continuous rectified-flow latent generator → acoustic adapter/Vocos |
| Audio | Mono 24 kHz; 100 mel features; 24 latent channels |
| Mel / latent hop | 256 / 512 samples |
| Duration model | Hidden size 320; 8 layers; reads the neighbouring text as context |
| Vector model | Hidden size 512; 12 layers; adaptive layer normalization and QK normalization |
| Maximum graph phones per sentence | 512 |
| Latent buckets | 128, 256, 384, 512, 768, 1024 frames |
| Reference sampler | Heun, 8 steps |
| Voice conditioning | Fixed per-voice/per-locale identity and prosody references embedded in the graphs |
| Text frontend | Trained Scylla's Band G2P with its matched tokenizer and vocabulary |

Long text is spoken one sentence at a time. Each sentence sees the neighbouring sentences as context and continues from the acoustics of the sentence before it, and consecutive sentences are decoded as one continuous waveform. Durations are predicted deterministically; different controls can alter them.

## Voices and language coverage

Ten voices: Ariadne, Felix, Gwen, Ink, Max, Orpheus, Rex, Scylla, Stone, Tuesday.

All ten have Spanish (`es`), Italian (`it`), French (`fr`), German (`de`) and Vietnamese (`vi`). Ink, Orpheus and Tuesday use British English (`en_gb`); the other seven use American English (`en_us`). This is **60 trained voice/locale pairs**, not every combination of ten voices and seven locale IDs.

## Artifacts

- `litert/`: LiteRT bundle; default download. Transformer and vocoder weights are dynamic-range INT8; each size bucket is a signature of one model file, sharing weights.
- `onnx/`: ONNX Runtime bundle with dynamic INT8 transformer weights; the vocoder stays in full precision.
- `pytorch/`: duration, vector, adapter/vocoder, autoencoder and G2P checkpoints, with the release configuration assets.
- `SHA256SUMS.json`: published-file hashes.

Both runtime bundles include the G2P model in a matching precision.

## Validation and limitations

Release validation separates graph accuracy, frontend/timing behaviour and perceived speech quality:

- Component comparisons against PyTorch during export, and end-to-end comparison of full passages against the PyTorch reference path.
- Quantized bundles compared with full precision on real sentences: identical phonemes, phone durations and generated latents within small tolerances, followed by listening across all ten voices.
- G2P conversions compared with the full-precision G2P on multilingual text.

Listening is decisive; external emotion-score agreement is approximate. Scorer measurements are imperfect supervision and do not establish emotional ground truth. Training audio is not distributed.

Known problems include occasional pronunciation errors, occasional roughness on fast, wide pitch movements in some voices, and limited strong anger or sadness.

The system supports managed voices, not arbitrary speaker cloning. Intended uses include local speech, narration, character dialogue and delivery-control research. Do not use it for impersonation or deceptive attribution of speech.

## License and acknowledgments

Apache 2.0. LiteRT and ONNX Runtime provide inference; Vocos (`charactr/vocos-mel-24khz`) is the base of the 24 kHz waveform decoder. DeepPhonemizer-derived tooling, Wiktionary-derived vocabulary, eSpeak phonemization and Montreal Forced Aligner contributed to the frontend and alignment workflow.
