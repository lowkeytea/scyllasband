---
license: apache-2.0
language: [en, es, it, fr, de, vi]
library_name: litert
pipeline_tag: text-to-speech
tags: [text-to-speech, tts, litert, onnx, pytorch, multilingual, expressive-tts, duration-flow]
---

# Scylla's Band v3

Release **`v3-20261103`**. The [Scylla's Band runtime](https://github.com/lowkeytea/scyllasband) supports this release from version 3.0.0. The manifest declares graph contract `scyllasband_measured_controls_v3`; runtimes written for v2 releases cannot run these graphs and keep downloading their v2 tags.

Delivery is controlled with **intensity, pace and pitch**, each from −2 to 2 with 0 as the voice's own delivery. The emotion comes from the text; intensity sets how much the voice puts into it.

## Quick start

Install the [runtime](https://github.com/lowkeytea/scyllasband) (`pip install -e .` from a checkout), then:

```bash
python -m scyllasband download --flavor litert
python -m scyllasband speak --voice ariadne --language en_us --intensity 1 --pace -0.5 \
    -o hello.wav "It is a pleasure to meet you."
```

`--flavor litert` or `--flavor onnx` chooses the bundle, installed under `scyllasband/models/<bundle>`. Downloads pin a release tag.

## Controls

| Control | Range | Default | Effect |
| --- | --- | --- | --- |
| Intensity | −2 to 2 | 0 | Calm (negative) to intense (positive): how much the voice puts into the line |
| Pace | −2 to 2 | 0 | Slower (negative) to faster (positive): the words and the pauses between them |
| Pitch | −2 to 2 | 0 | Shifts the voice's pitch, in semitones |

0 is each voice's own delivery. Intensity and pace are measured in each voice's own range, in standard deviations around its average, so the same value is a comparable change for every voice and language. An angry line sounds angrier at higher intensity and more restrained at lower intensity; intensity changes how much of the emotion comes through, not which emotion it is. Requests outside the range are clamped into it, and unspecified controls take the voice's default (`controls.delivery.defaults`; 0 for every voice in this release).

How the runtime applies them:

- The duration and flow graphs take intensity and pace as value / 5 in delivery slots 0 and 1, with presence bits; the other slots are always absent.
- Pace also scales the pauses between words by exp(−0.13 × pace) (`controls.pace_pauses`).
- A passage with fewer than four spoken phones, such as a lone "Yes.", keeps the voice's own intensity and pace, as the model was trained (`controls.delivery.min_spoken_phones`). Its pitch shift still applies.
- Pitch shifts the predicted contour in semitones; a soft limit keeps it inside the voice's measured range (`controls.pitch.edges`).

Controls are **global within each request**. This release does not support control curves, singing, or emotion labels.

## Model and inference

| Property | Value |
| --- | --- |
| Public model version | 3 |
| Architecture | Phone-duration predictor → pitch predictor (rectified flow) → continuous rectified-flow latent generator that follows the pitch contour → acoustic adapter/Vocos |
| Audio | Mono 24 kHz; 100 mel features; 24 latent channels |
| Mel / latent hop | 256 / 512 samples |
| Duration model | Hidden size 320; 8 layers; reads the neighbouring text as context |
| Pitch predictor | Hidden size 384; 8 layers; a pitch and voicing contour per latent frame |
| Vector model | Hidden size 512; 12 layers; adaptive layer normalization and QK normalization; reads the pitch contour |
| Maximum graph phones per sentence | 512 |
| Trained target length | 64–420 latent frames (about 1.4–9 s) per pass |
| Sampler | Flow: Euler, 4 steps. Pitch predictor: Heun, 8 steps |
| Voice conditioning | Fixed per-voice/per-locale identity and prosody references embedded in the graphs |
| Text frontend | Trained Scylla's Band G2P with its matched tokenizer and vocabulary |

For each passage the pitch predictor integrates a pitch and voicing contour from noise, the pitch control shifts it, and the flow generates the latents along it (graph input `expanded_melody`, [1, frames, 3]). Each passage draws its pitch noise ([2, frames]) and then its latent noise ([24, frames]) from the request's seed.

The flow is distilled for few-step sampling: it was trained to reproduce the base flow's 32-step Heun output, so Euler with 4 steps, four model evaluations per passage, is the default.

Long text is spoken as passages within the trained target lengths (64 to 420 latent frames, about 1.4 to 9 seconds): a short first passage so audio starts quickly, then passages as long as the text allows, cut at a sentence end, else a clause, else between words, and measured with the requested delivery. Each passage sees the surrounding text as context and continues from the acoustics of the one before it, and consecutive passages are decoded as one continuous waveform. Durations are predicted deterministically; different controls can alter them.

## Voices and language coverage

Fourteen voices: Ariadne, Felix, Gwen, Ink, Kincade, Max, Orpheus, Poppy, Priya, Rex, Scylla, Stone, Thistle, Tuesday.

All fourteen have Spanish (`es`), Italian (`it`), French (`fr`), German (`de`) and Vietnamese (`vi`). Ink, Orpheus and Tuesday use British English (`en_gb`); the other eleven use American English (`en_us`). This is **84 trained voice/locale pairs**, not every combination of fourteen voices and seven locale IDs.

## Artifacts

- `litert/` (229 MB): LiteRT bundle for Linux, Windows, Android and Intel Macs. Dynamic-range INT8 weights on the fully connected, convolution and embedding layers; the vocoder's depthwise convolutions stay in full precision, which keeps them on XNNPACK. Each size is a signature of one model file, sharing weights: latent buckets every 64 frames up to 512 (then 768 and 1024), flow graphs without the 96 prefix frames for a request's first passage (up to 448 frames), and narrower duration, span-context and pitch-context (128 and 256 phones) and G2P (32–256 tokens) widths beside the full ones.
- `onnx/` (291 MB): ONNX Runtime bundle with dynamic INT8 weights in the duration, context, pitch and flow transformers; the vocoder stays in full precision. Latent buckets of 128, 256, 384, 512, 768 and 1024 frames share one weight file.
- `webgpu/` (291 MB): ONNX Runtime bundle for ONNX Runtime Web's WebGPU backend. Transformer weights are INT8 with float compute (`MatMulNBits`), which runs entirely on the GPU; the vocoder stays in full precision.
- `pytorch/` (616 MB): duration, pitch predictor, vector, adapter/vocoder and G2P weights (no training state), with the release configuration assets.
- `SHA256SUMS.json`: published-file hashes.

Every runtime bundle includes the G2P model in a matching precision.

## Validation and limitations

Release validation separates graph accuracy, frontend/timing behaviour and perceived speech quality:

- The full-precision ONNX bundle run through the runtime against the PyTorch model, end to end with all three controls: identical durations, and latents within 7e-5 (values of magnitude 20–28).
- The C++ runtime against the Python runtime on the same bundle: identical G2P output, phone durations and passage plan; latents within 4e-6 relative error; decoded audio at 50 dB SNR (worst sentence) and 59 dB over a long passage.
- The distilled flow at 4 Euler steps compared with the undistilled flow by listening on paired lines across the voices, and by ASR word error rate and a MOS predictor on a 168-clip sweep of all fourteen voices in English, French and Spanish.
- Quantized bundles compared with full precision through the runtime on lines covering every voice, each control at its limits and each language. The WebGPU bundle reproduces the full-precision take (identical frame counts on 15 of 17 lines, 0.8 dB median mel-cepstral distance). Dynamic INT8 moves some phone durations by a frame, which redraws a passage's noise and gives a different take, so the ONNX and LiteRT bundles are judged on the same sweep: ASR word error rate 0.066 (ONNX) and 0.062 (LiteRT) against 0.062 for full precision, and a MOS-predictor score within 0.05 of full precision for both.
- Every LiteRT graph gives identical output with and without XNNPACK's weight cache.

Listening is decisive; automatic scores support it. Training audio is not distributed.

Known problems include occasional pronunciation errors and occasional roughness on fast, wide pitch movements in some voices.

The system supports managed voices, not arbitrary speaker cloning. Intended uses include local speech, narration, character dialogue and delivery-control research. Do not use it for impersonation or deceptive attribution of speech.

## License and acknowledgments

Apache 2.0. LiteRT and ONNX Runtime provide inference; Vocos (`charactr/vocos-mel-24khz`) is the base of the 24 kHz waveform decoder. DeepPhonemizer-derived tooling, Wiktionary-derived vocabulary, eSpeak phonemization and Montreal Forced Aligner contributed to the frontend and alignment workflow. Vocos and DeepPhonemizer are MIT-licensed; their notices are in `THIRD_PARTY_NOTICES.md`.
