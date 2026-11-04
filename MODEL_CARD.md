---
license: apache-2.0
language: [en, es, it, fr, de, vi]
library_name: litert
pipeline_tag: text-to-speech
tags: [text-to-speech, tts, litert, onnx, coreml, coreai, pytorch, multilingual, expressive-tts, duration-flow]
---

# Scylla's Band v3

Release **`v3-20261103`**. The [Scylla's Band runtime](https://github.com/lowkeytea/scyllasband) supports this release from version 3.0.0. Tag `v3-20261103-apple` adds Core ML and Core AI bundles of the same model; the runtime downloads from `v3-20261103-apple`. The manifest declares graph contract `scyllasband_measured_controls_v3`; runtimes written for v2 releases cannot run these graphs and keep downloading their v2 tags.

Delivery is controlled with **intensity, pace and pitch**, each from −2 to 2 with 0 as the voice's own delivery. The emotion comes from the text; intensity sets how much the voice puts into it.

## Quick start

Install the [runtime](https://github.com/lowkeytea/scyllasband) (`pip install -e .` from a checkout), then:

```bash
python -m scyllasband download --flavor litert
python -m scyllasband speak --voice ariadne --language en_us --intensity 1 --pace -0.5 \
    -o hello.wav "It is a pleasure to meet you."
```

The default download is the Core ML bundle on an Apple silicon Mac and the LiteRT bundle elsewhere, installed under `scyllasband/models/<bundle>`. A Mac that can build apps for iOS, iPadOS or visionOS 27 is also offered the Core AI bundle those apps use. `--flavor coreml`, `--flavor coreai`, `--flavor litert` or `--flavor onnx` choose explicitly. Downloads pin a release tag.

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

## Performance

Measured with the default samplers (flow: Euler, 4 steps; pitch predictor: Heun, 8 steps) through the native runtime: `scyllasband_speak` from [libscyllasband](https://github.com/lowkeytea/scyllasband/tree/main/libscyllasband) on the Mac and Android, and the [iOS sample](https://github.com/lowkeytea/scyllasband/tree/main/examples/ios)'s benchmark mode on the iPhone. The text is three sentences, 10.3 s of audio: "The last train leaves at midnight, so pack light. Bring a warm coat, and don't forget the map. If we miss it, we walk, and nobody wants to walk that far in the rain." (voice Scylla, seed 7, the voice's own delivery).

- **First run**: from launch to first audio on the first launch after installing the bundle. Core ML compiles and Core AI specializes the graphs for the device then (Core AI caches them until the next OS update); LiteRT writes its packed weights to a cache file beside each graph.
- **Cold start**: from launch to first audio on a later launch, including loading the bundle.
- **Warm first audio**: from a request to its first audio in a process that has already spoken (median of three requests).
- **Real-time speed**: seconds of audio generated per second when warm, rounded.

| Device | Bundle | Compute | First run | Cold start | Warm first audio | Real-time speed |
| --- | --- | --- | --- | --- | --- | --- |
| MacBook Pro, M5 Max (macOS 27) | Core ML | GPU (default) | 2.67 s | 1.39 s | 54 ms | ~97× |
|  | Core ML | CPU | 2.31 s | 0.43 s | 330 ms | ~12× |
|  | Core AI | GPU (default) | 6.01 s | 0.51 s | 57 ms | ~88× |
|  | Core AI | Neural Engine | 35.8 s | 0.62 s | 363 ms | ~16× |
|  | Core AI | CPU | 3.98 s | 1.32 s | 1239 ms | ~3.2× |
|  | LiteRT | CPU | 0.83 s | 0.39 s | 181 ms | ~20× |
|  | ONNX Runtime | CPU | 1.36 s | 1.00 s | 343 ms | ~13× |
| iPhone 17 Pro Max (iOS 27.0.1) | Core ML | GPU (default) | 16.3 s | 1.76 s | 113 ms | ~42× |
|  | Core ML | CPU | 3.15 s | 0.67 s | 449 ms | ~9× |
|  | Core AI | GPU (default) | 19.0 s | 0.65 s | 109 ms | ~42× |
|  | Core AI | Neural Engine | 7.65 s | 0.75 s | 418 ms | ~14× |
|  | Core AI | CPU | 8.40 s | 1.76 s | 1555 ms | ~2.5× |
| Galaxy Z Fold 8, Snapdragon SM8850 (Android 17) | LiteRT | CPU | 1.02 s | 0.56 s | 203 ms | ~16× |
|  | ONNX Runtime | CPU | 1.99 s | 1.99 s | 701 ms | ~5.9× |

The GPU is the default on Apple platforms. The Neural Engine placement runs the flow and the pitch predictor on the Neural Engine and the other graphs on the CPU, for apps that speak in the background, where iOS does not allow GPU work. Core ML reloads its graphs on every launch, which accounts for most of its cold start on the GPU. Each passage runs the pitch predictor 16 times before the flow's 4 steps.

## Voices and language coverage

Fourteen voices: Ariadne, Felix, Gwen, Ink, Kincade, Max, Orpheus, Poppy, Priya, Rex, Scylla, Stone, Thistle, Tuesday.

All fourteen have Spanish (`es`), Italian (`it`), French (`fr`), German (`de`) and Vietnamese (`vi`). Ink, Orpheus and Tuesday use British English (`en_gb`); the other eleven use American English (`en_us`). This is **84 trained voice/locale pairs**, not every combination of fourteen voices and seven locale IDs.

## Artifacts

- `coreai/` (178 MB): Core AI bundle for apps on iOS, iPadOS and visionOS 27 (`.aimodel` assets); it also runs on macOS 27, where the runtime uses the Core ML bundle by default. INT8 weights per output channel with FP16 compute; latent buckets of 128, 256, 384, 512, 768 and 1024 frames are functions of one asset sharing weights. The flow and the pitch predictor also run accurately on the Neural Engine (`controls.coreai.neural_engine_assets`), which is the placement for apps that speak in the background; the GPU is the default.
- `coreml/` (163 MB): Core ML bundle for Apple silicon with iOS 18 / visionOS 2 / macOS 15 and later (compiled `.mlmodelc` assets), with INT8 weights and FP16 compute; GPU by default, CPU in background apps.
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
- Core ML and Core AI bundles compared with the PyTorch model on every graph and through the runtime on 14 passages (every voice, each control at both limits, six languages, narration), on each compute unit they are used with (GPU, CPU, and for Core AI the Neural Engine): identical phonemes, fewer changed phone durations than the LiteRT bundle (0.2–0.7 % against 1.9 %), and smaller pitch-contour (0.02–0.04 semitones RMS against 0.045), latent and spectral differences. The Apple exports rewrite three computations into equivalent forms that FP16 and the Neural Engine evaluate exactly (span pooling, the modifier bit decoding, the duration model's combined phone/segment index), and scale the residual stream of the flow and the pitch predictor; in FP32 the rewritten graphs match the trained ones (identical outputs; the span encoders within 1e-6).

Listening is decisive; automatic scores support it. Training audio is not distributed.

Known problems include occasional pronunciation errors and occasional roughness on fast, wide pitch movements in some voices.

The system supports managed voices, not arbitrary speaker cloning. Intended uses include local speech, narration, character dialogue and delivery-control research. Do not use it for impersonation or deceptive attribution of speech.

## License and acknowledgments

Apache 2.0. LiteRT and ONNX Runtime provide inference; Vocos (`charactr/vocos-mel-24khz`) is the base of the 24 kHz waveform decoder. DeepPhonemizer-derived tooling, Wiktionary-derived vocabulary, eSpeak phonemization and Montreal Forced Aligner contributed to the frontend and alignment workflow. Vocos and DeepPhonemizer are MIT-licensed; their notices are in `THIRD_PARTY_NOTICES.md`.
