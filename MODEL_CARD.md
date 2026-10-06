---
license: apache-2.0
language: [en, es, it, fr, de, vi]
library_name: litert
pipeline_tag: text-to-speech
tags: [text-to-speech, tts, litert, onnx, coreml, coreai, pytorch, multilingual, expressive-tts, duration-flow]
---

# Scylla's Band v2

Release **`v2-20261006`**. The [Scylla's Band runtime](https://github.com/lowkeytea/scyllasband) supports this release. Tag `v2-20261006-apple` adds Core ML and Core AI bundles of the same model; the runtime downloads from that tag.

The model exposes **energy, tension, valence and assertiveness** on **0–4 scales with neutral at 2**. Combining the controls changes the generated delivery; a request is not a guarantee of a particular perceived emotion or exact external scorer value.

## Quick start

Install the [runtime](https://github.com/lowkeytea/scyllasband) (`pip install -e .` from a checkout), then:

```bash
python -m scyllasband download --yes
python -m scyllasband speak --voice ariadne --language en_us \
    --delivery energy=2.3,tension=1.7,valence=2.2,assertiveness=2 \
    -o hello.wav "It is a pleasure to meet you."
```

The default download is the Core ML bundle on an Apple silicon Mac and the LiteRT bundle elsewhere, installed under `scyllasband/models/<bundle>`. A Mac that can build apps for iOS, iPadOS or visionOS 27 is also offered the Core AI bundle those apps use. `--flavor coreml`, `--flavor coreai`, `--flavor litert` or `--flavor onnx` choose explicitly. Downloads pin a release tag. The manifest declares graph contract `scyllasband_measured_delivery_v2`; runtimes written for earlier releases cannot run these graphs.

## Controls

| Input | Values | Default |
| --- | --- | --- |
| Energy | 1.4–2.3 or unspecified | 2 |
| Tension | 1.0–2.4 or unspecified | 2 |
| Valence | 1.4–2.2 or unspecified | 2 |
| Assertiveness | 1.8–2.4 or unspecified | 2 |

With every control at 2, the default, voices speak in a calm, conversational delivery with a neutral tone. Raising or lowering a control changes the delivery:

| Control | Lower | Higher |
| --- | --- | --- |
| Energy | Lower, darker voice with a few more pauses | Higher, brighter, more projected voice |
| Tension | Slightly looser and steadier | Tighter, with wider pitch swings |
| Valence | Flatter, more downcast pitch | Livelier pitch, a little quicker, shorter pauses |
| Assertiveness | More pauses between phrases | Faster and clipped with fewer pauses; the strongest control over pace |

Requests use the 0–4 scale, and the runtime clamps each control into its supported range, where the voices render most reliably: energy 1.4–2.3, tension 1.0–2.4, valence 1.4–2.2 and assertiveness 1.8–2.4. Small changes inside these ranges are clearly audible, and over longer text the voices vary their delivery with the content on their own.

Combining the controls gives familiar deliveries:

| Delivery | Settings |
| --- | --- |
| Calm and soft | `energy=1.8,tension=1,valence=2.2,assertiveness=2.1` |
| Assertive and quick | `energy=2.3,tension=2.4,valence=2.1,assertiveness=2.4` |
| Joyful | `energy=2.3,tension=2,valence=2.2,assertiveness=2.2` |
| Angry | `energy=2.3,tension=2.4,valence=1.4,assertiveness=2.4` |
| Sad | `energy=1.7,tension=1.5,valence=1.4,assertiveness=2` |

These are moderate styles; how strongly each comes through varies with voice, language and text. The [voice gallery](https://lowkeytea.github.io/scyllasband/#delivery) plays each of them for all ten voices.

`--delivery auto` omits all coordinates. A per-axis `auto` omits just that coordinate; other unspecified fields in a partial request use neutral defaults. Omission is represented by a separate mask, not by the value 2. It is not an automatic text-to-performance planner.

The graph receives `(rating - 2) / 4` for the four continuous axes plus presence bits. Its fifth delivery input, whisper, was a training axis; it is not supported, and the runtimes always send it off. The public interface accepts the original 0–4 values. Expressiveness is an auxiliary training target, not an inference axis.

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
| Trained target length | 64–420 latent frames (about 1.4–9 s) per pass |
| Latent buckets | 128, 256, 384, 512, 768, 1024 frames |
| Sampler | Euler, 4 steps (default) |
| Voice conditioning | Fixed per-voice/per-locale identity and prosody references embedded in the graphs |
| Text frontend | Trained Scylla's Band G2P with its matched tokenizer and vocabulary |

The flow is distilled for few-step sampling: it was trained to reproduce the base flow's 32-step Heun output, so Euler with 4 steps, four model evaluations per passage, is the default. Heun with 8 steps costs four times as much and sounds slightly sharper.

Long text is spoken as passages within the trained target lengths (64 to 420 latent frames, about 1.4 to 9 seconds): a short first passage so audio starts quickly, then passages as long as the text allows, cut at a sentence end, else a clause, else between words, and measured with the requested delivery. Each passage sees the surrounding text as context and continues from the acoustics of the one before it, and consecutive passages are decoded as one continuous waveform. Durations are predicted deterministically; different controls can alter them.

## Performance

Measured with the default sampler (Euler, 4 steps) through the native runtime: `scyllasband_speak` from [libscyllasband](https://github.com/lowkeytea/scyllasband/tree/main/libscyllasband) on the Mac and Android, and the [iOS sample](https://github.com/lowkeytea/scyllasband/tree/main/examples/ios)'s benchmark mode on the iPhone. The text is three sentences, 11.6 s of audio: "The last train leaves at midnight, so pack light. Bring a warm coat, and don't forget the map. If we miss it, we walk, and nobody wants to walk that far in the rain." (voice Scylla, seed 7).

- **First run**: from launch to first audio on the first launch after installing the bundle. Core AI specializes its graphs for the device then and caches them until the next OS update.
- **Cold start**: from launch to first audio on a later launch, including loading the bundle.
- **Warm first audio**: from a request to its first audio in a process that has already spoken (median of three requests).
- **Real-time speed**: seconds of audio generated per second when warm, rounded.

| Device | Bundle | Compute | First run | Cold start | Warm first audio | Real-time speed |
| --- | --- | --- | --- | --- | --- | --- |
| MacBook Pro, M5 Max (macOS 27) | Core ML | GPU (default) | 2.44 s | 1.29 s | 24 ms | ~220× |
|  | Core ML | CPU | 1.82 s | 0.28 s | 192 ms | ~25× |
|  | Core AI | GPU (default) | 5.33 s | 0.42 s | 28 ms | ~200× |
|  | Core AI | Neural Engine | 25.3 s | 0.50 s | 310 ms | ~18× |
|  | Core AI | CPU | 2.49 s | 0.82 s | 746 ms | ~6.3× |
|  | LiteRT | CPU | 0.76 s | 0.74 s | 142 ms | ~34× |
|  | ONNX Runtime | CPU | 0.84 s | 0.89 s | 200 ms | ~24× |
| iPhone 17 Pro Max (iOS 27.0.1) | Core ML | GPU (default) | 2.79 s | 1.43 s | 63 ms | ~85× |
|  | Core ML | CPU | 2.04 s | 0.35 s | 240 ms | ~20× |
|  | Core AI | GPU (default) | 6.49 s | 0.42 s | 58 ms | ~92× |
|  | Core AI | Neural Engine | 27.8 s | 0.58 s | 370 ms | ~15× |
|  | Core AI | CPU | 3.30 s | 0.96 s | 882 ms | ~5.2× |
| Galaxy Z Fold 8, Snapdragon SM8850 (Android 17) | LiteRT | CPU | 1.60 s | 1.32 s | 178 ms | ~27× |
|  | ONNX Runtime | CPU | 1.90 s | 1.48 s | 375 ms | ~12× |

The GPU is the default on Apple platforms. The Neural Engine placement runs the flow on the Neural Engine and the other graphs on the CPU, for apps that speak in the background, where iOS does not allow GPU work. Core ML reloads its graphs on every launch, which accounts for most of its cold start on the GPU.

## Voices and language coverage

Ten voices: Ariadne, Felix, Gwen, Ink, Max, Orpheus, Rex, Scylla, Stone, Tuesday.

All ten have Spanish (`es`), Italian (`it`), French (`fr`), German (`de`) and Vietnamese (`vi`). Ink, Orpheus and Tuesday use British English (`en_gb`); the other seven use American English (`en_us`). This is **60 trained voice/locale pairs**, not every combination of ten voices and seven locale IDs.

## Artifacts

- `litert/` (164 MB): LiteRT bundle; default download. Transformer and vocoder weights are dynamic-range INT8; each size bucket is a signature of one model file, sharing weights.
- `coreai/` (155 MB): Core AI bundle for apps on iOS, iPadOS and visionOS 27 (`.aimodel` assets); it also runs on macOS 27, where the runtime uses the Core ML bundle by default. INT8 weights per output channel with FP16 compute; size buckets are functions of one asset sharing weights. The flow also runs accurately on the Neural Engine (`controls.coreai.neural_engine_assets`), which is the placement for apps that speak in the background; the GPU is the default.
- `coreml/` (142 MB): Core ML bundle for Apple silicon with iOS 18 / visionOS 2 / macOS 15 and later (compiled `.mlmodelc` assets), with INT8 weights and FP16 compute; GPU by default, CPU in background apps.
- `onnx/` (262 MB): ONNX Runtime bundle with dynamic INT8 transformer weights; the vocoder stays in full precision.
- `pytorch/`: duration, vector, adapter/vocoder and G2P weights (no training state), with the release configuration assets.
- `SHA256SUMS.json`: published-file hashes.

Every runtime bundle includes the G2P model in a matching precision. The Apple bundles also give the G2P narrower input widths (64–512 tokens), so short phrases are converted faster.

## Validation and limitations

Release validation separates graph accuracy, frontend/timing behaviour and perceived speech quality:

- Component comparisons against PyTorch during export, and end-to-end comparison of full passages against the PyTorch reference path.
- The distilled flow at 4 Euler steps compared with the base flow at 8 Heun steps on the same passages, seeds and noise: equal distance to the 32-step output, confirmed by listening across all ten voices.
- Quantized bundles compared with full precision on real sentences: identical phonemes, phone durations and generated latents within small tolerances, followed by listening across all ten voices.
- G2P conversions compared with the full-precision G2P on multilingual text.
- Core ML and Core AI bundles compared with the PyTorch model on every graph and on real passages, on each compute unit they are used with (GPU, CPU, and for Core AI the Neural Engine): identical phonemes apart from two near-tie stress marks also seen in FP16, fewer changed phone durations than the LiteRT bundle, and smaller latent and spectral differences. The Apple exports rewrite three computations into equivalent forms that FP16 and the Neural Engine evaluate exactly (span pooling, the modifier bit decoding, the duration model's combined phone/segment index), and scale the flow's residual stream; in FP32 the rewritten graphs are bit-identical to the trained ones.

Listening is decisive; external emotion-score agreement is approximate. Scorer measurements are imperfect supervision and do not establish emotional ground truth. Training audio is not distributed.

Known problems include occasional pronunciation errors, occasional roughness on fast, wide pitch movements in some voices, and limited strong anger or sadness.

The system supports managed voices, not arbitrary speaker cloning. Intended uses include local speech, narration, character dialogue and delivery-control research. Do not use it for impersonation or deceptive attribution of speech.

## License and acknowledgments

Apache 2.0. LiteRT and ONNX Runtime provide inference; Vocos (`charactr/vocos-mel-24khz`) is the base of the 24 kHz waveform decoder. DeepPhonemizer-derived tooling, Wiktionary-derived vocabulary, eSpeak phonemization and Montreal Forced Aligner contributed to the frontend and alignment workflow. Vocos and DeepPhonemizer are MIT-licensed; their notices are in `THIRD_PARTY_NOTICES.md`.
