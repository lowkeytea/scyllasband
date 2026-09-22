---
license: apache-2.0
language: [en, es, it, fr, de, vi]
library_name: onnxruntime
pipeline_tag: text-to-speech
tags: [text-to-speech, tts, onnx, pytorch, multilingual, expressive-tts, duration-flow, whisper]
---

# Scylla's Band v2 — measured delivery

Release **`v2-measured-20260922`** replaces the earlier emotion-based v2 development release. It promotes the completed measured-delivery training continuation, called v3 during development, under the public v2 name. The public runtime's `dev` branch supports this release and preserves the separate v1 contract.

The model exposes **energy, tension, valence and assertiveness** on **0–4 scales with neutral at 2**, plus **binary whisper**. Emotion names and coarse L0–L4 labels are no longer the conditioning inputs. Combining the controls changes the generated delivery; a request is not a guarantee of a particular perceived emotion or exact external scorer value.

## Quick start

Install the current [runtime dev branch](https://github.com/lowkeytea/scyllasband/tree/dev), plus `numpy`, `huggingface_hub` and `onnxruntime`:

```bash
python -m scyllasband download --model-version v2 --yes
python -m scyllasband speak --voice ariadne --language en_us \
    --delivery energy=2.3,tension=1.7,valence=2.6,assertiveness=2,whisper=off \
    --sampler heun --steps 8 -o hello.wav "It is a pleasure to meet you."
```

The default is FP32 ONNX. Downloads pin this release tag and install under `scyllasband/models/v2/onnx`. The manifest declares graph contract `scyllasband_measured_delivery_v1`; a runtime that only understands old v2 affect tensors cannot run these graphs.

An installed old v2 triggers a replacement notice. V1 remains separately available from [`spybyscript/scyllasband`](https://huggingface.co/spybyscript/scyllasband) and remains compatible with its emotion controls. Upgrading v2 does not remove v1 unless the user explicitly selects deletion.

## Controls

| Input | Values | Default |
| --- | --- | --- |
| Energy, tension, valence, assertiveness | 0–4 or unspecified | 2 each |
| Whisper | on/off or unspecified | off |

`--delivery auto` omits all coordinates. A per-axis `auto` omits just that coordinate; other unspecified fields in a partial request use neutral defaults. Omission is represented by a separate mask, not by the value 2. It is not an automatic text-to-performance planner.

The graph receives `(rating - 2) / 4` for the four continuous axes and 0/1 for whisper, plus five presence bits. The public interface accepts the original 0–4 values. Expressiveness is an auxiliary training target, not an inference axis. Legacy emotion presets and emotion CFG are unsupported by this model.

Controls are **global within each generated chunk**. Frame-level physical targets and qualified window-level rating targets supervised training, but local window ratings were not supplied as time-varying control inputs. This release does not support control curves, singing, or trained automatic variation over a multi-sentence passage.

## Model and inference

| Property | Value |
| --- | --- |
| Public model version | 2 |
| Architecture | Phone-duration predictor → continuous rectified-flow latent generator → acoustic adapter/Vocos |
| Audio | Mono 24 kHz; 100 mel features; 24 latent channels |
| Mel / latent hop | 256 / 512 samples |
| Duration model | Hidden size 192; 4 layers |
| Vector model | Hidden size 512; 12 layers; 8 attention heads; adaptive layer normalization and QK normalization |
| Maximum graph phone frames | 512 |
| Latent buckets | 256, 512, 768, 1024 frames |
| Reference sampler | Heun, 8 steps |
| Voice conditioning | Fixed per-voice/per-locale identity and prosody references embedded in the graphs |
| Text frontend | Trained Scylla's Band G2P with its matched tokenizer and vocabulary |

Inference matches the accepted measured training/evaluation path: utterance-only span context, fixed references, no prefix requests, no emotion-guided reference routing, and no old v2 hierarchical pause-sampling heads. The duration continuation includes relative word timing and adjacent-word/pause objectives. Durations remain predicted deterministically; different controls can alter them.

## Voices and language coverage

Ten voices: Ariadne, Felix, Gwen, Ink, Max, Orpheus, Rex, Scylla, Stone, Tuesday.

All ten have Spanish (`es`), Italian (`it`), French (`fr`), German (`de`) and Vietnamese (`vi`). Ink, Orpheus and Tuesday use British English (`en_gb`); the other seven use American English (`en_us`). This is **60 trained voice/locale pairs**, not every combination of ten voices and seven locale IDs.

## Training data and objectives

The continuation used the prepared multilingual v3 corpus merged with the refreshed English recordings and selected original English alternate takes. Original and refreshed audio were measured to improve coverage of delivery ranges and distinct takes. Coarse generation labels were replaced as request conditioning by acoustic measurements and an audio emotion model's assessments.

The completed full continuation contains **1,786,967 compiled training records and 46,605 validation records**. These are prepared-record counts, not a claim of that many unique source recordings. Ten voices and all 60 locale pairs were included. Duration and vector stages each completed 20 epochs; the promoted duration checkpoint is best epoch 16 and the vector checkpoint is best epoch 20.

The vector training combines the flow objective with local physical, qualified window-rating and whisper objectives. Window-rating targets include energy, tension, valence, assertiveness and expressiveness. Request-axis dropout and all-request dropout teach absent-coordinate cases. The duration continuation adds relative rhythm objectives. The acoustic adapter and Vocos decoder were retained from the matched frontend assets rather than retrained for this continuation.

Source audio is synthetic TTS. Scorer measurements are imperfect supervision and do not establish emotional ground truth. Coverage is uneven, especially for extreme delivery and expressive connected multi-sentence English. Training audio is not distributed.

## Artifacts

- `onnx/`: FP32 release reference; default download.
- `onnx-int8/`: optional dynamic INT8 vector-core/G2P variant. Duration, context and vocoder remain FP32. Listen before choosing it for a quality-sensitive application.
- `pytorch/`: matched duration, vector, adapter/vocoder, autoencoder and G2P checkpoints, with associated release configuration assets.
- `SHA256SUMS.json`: published-file hashes and cross-bundle contract metadata.

Measured LiteRT and Core AI artifacts are **not** included in this release. Earlier platform bundles must not be mixed with the new duration/vector graphs. Python and native C++ execution use ONNX Runtime; target-device performance remains platform dependent.

Duration checkpoint SHA-256: `5e62af2e78d5578f27c9a259cf3c9739c59adfe0c432277e8c45fbd33ac2d9db`.

Vector checkpoint SHA-256: `36f630770e4023d5efbb4f4b3c463122ccca5af8abb76c2600140a4beae73bfd`.

## Validation and limitations

Release validation separates graph accuracy, frontend/timing behavior and perceived speech quality:

- FP32 component comparisons against PyTorch during export.
- Real public-runtime synthesis across ten voices, all seven locale IDs, mixed controls, whisper and absent coordinates.
- Python/native comparisons of phones and rounded durations.
- Real v1 synthesis with exact waveform comparison against the prior dev runtime.
- FP32/INT8 comparisons: phone sequences and durations agree for the ten-voice English comparison set; waveforms differ. This does not establish perceptual equivalence.

The developer's listening evaluations support more gradual and useful control than earlier models. No blinded perceptual benchmark or universal quality improvement is claimed. Listening is decisive; external emotion-score agreement is approximate.

Known problems include occasional pronunciation errors and cracking, metallic or thin quality on difficult pitches, limited strong anger/sadness, and stilted or uniform rhythm across multiple sentences. Global control requests do not solve temporal performance planning. Long-form host chunking helps manage model length but does not supply learned discourse prosody.

The system supports managed voices, not arbitrary speaker cloning. Intended uses include local speech, narration, character dialogue and delivery-control research. Do not use it for impersonation or deceptive attribution of speech.

## License and acknowledgments

Apache 2.0. ONNX Runtime provides inference; Vocos provides the frozen 24 kHz waveform decoder. DeepPhonemizer-derived tooling, Wiktionary-derived vocabulary, eSpeak phonemization and Montreal Forced Aligner contributed to the frontend and alignment workflow.
