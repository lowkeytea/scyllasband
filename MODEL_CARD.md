---
license: apache-2.0
language: [en, es, it, fr, de, vi]
library_name: litert
pipeline_tag: text-to-speech
tags: [text-to-speech, tts, singing, singing-voice-synthesis, litert, onnx, coreml, pytorch, multilingual, expressive-tts, duration-flow]
---

# Scylla's Band: speech and singing

Release **`v4-20261106`**: Scylla's Band speaks and now sings. Speech and singing share one model: one flow renders both,
selected by a delivery mode, with two waveform decoders. The [runtime](https://github.com/lowkeytea/scyllasband) (C ABI
3.1.0, with the `sing` command and the v4 graph contract) runs it on Linux, Windows, macOS, iOS, visionOS and Android.
The previous release, `v3-20261104`, stays available at its tags (`v3-20261104`, `v3-20261104-apple`) for runtimes built
for `v3`.

What changed from `v3-20261104`:

- **Singing.** Pop and ballad delivery from a melody score (lyrics, notes and beats as JSON) in all
  fourteen voices. The flow is `v3`'s distilled flow continued by reflow on singing material: spoken output starts identical
  to `v3` and the sung and ballad paths are distilled from a singing-trained teacher.
- **Singing at the speech cost (`v4-20261106`).** The sung and ballad paths are distilled a second time from the teacher's
  classifier-free-guided field (melody and mode against both switched off, guidance 2), so the flow reproduces the guided
  answer unguided: 4 Euler steps and one evaluation per step, as for speech, against 8 Heun steps with guidance (32
  evaluations) for the first singing candidate (`v4-20261105`, not published here). Lyric clarity and pitch accuracy are
  unchanged (below).
- **Second decoder.** Sung targets render through a BigVGAN v2 decoder fine-tuned on the voices; speech keeps the Vocos
  decoder. The runtime picks the decoder by delivery mode.
- **Graph contract `scyllasband_measured_controls_v4`.** The flow graphs take one more input, `delivery_mode_id`. Runtimes
  built for `v3` refuse this bundle and keep downloading `v3-20261104-apple`, which stays available here.

Delivery for speech is unchanged: **intensity, pace and pitch**, each from −2 to 2.

## Quick start

Install the [runtime](https://github.com/lowkeytea/scyllasband) (`pip install -e .` from a checkout), then:

```bash
python -m scyllasband download --flavor litert
python -m scyllasband speak --voice ariadne --language en_us --intensity 1 --pace -0.5 \
    -o hello.wav "It is a pleasure to meet you."
python -m scyllasband sing --voice gwen --score scyllasband/models/melodies/phrase_pop_00.json -o ring.wav
```

`download` installs the bundle and the example scores (`melodies/`) from this repository at the release tag. On macOS
`--flavor coreml` or `--flavor coreai` fetch the Apple bundles. `python -m scyllasband build-native` builds the native
runtime that the commands then use (much faster and lighter than the Python engine, the same plans).

## Singing

A song is a JSON score: the lyric text, a tempo, a style and the notes in order, each `[label, pitch, beats]`.

```json
{
  "text": "give me a ring baby make me feel so",
  "bpm": 108,
  "style": "pop",
  "notes": [
    ["give", "C4", 0.5], ["me", "D4", 0.5], ["a", "E4", 0.5], ["ring", "G4", 1.0],
    ["-", "rest", 0.5],
    ["baby~1", "E4", 0.5], ["baby~2", "D4", 0.5],
    ["make", "C4", 0.5], ["me", "D4", 0.5], ["feel", "E4", 1.0], ["so", "C4", 1.5]
  ]
}
```

- `style` is `pop` (the sung delivery token; the pop material was ingested under it) or `ballad`.
- A word over several notes is numbered (`baby~1 baby~2`, also for a syllable held over two pitches); rests use the label `-`.
- Phrases split at rests of a quarter second or more and render one at a time; the runtime shifts the whole song by octaves
  into the voice's sung range (`--no-register` keeps the written octave).
- Sung targets take the melody from the score, skip the pitch predictor and the spoken pitch limits, and render with 4 Euler
  steps and no guidance through the BigVGAN decoder (`--decoder vocos` selects the speech decoder; `--steps`, `--sampler`, `--guidance`
  override the manifest defaults). The model never free-sings: a score, written by the user or by a planner, is required.

The `melodies/` folder holds twenty example scores: four whole songs (two pop, two ballads) and sixteen phrases.

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
- A dash typed with no spaces around it (`a—dream`, `a--dream`) is read as an interruption: the pause after it lasts at least 0.3 seconds. Dashes with a space on either side keep the pause the model gives them.
- A passage with fewer than four spoken phones, such as a lone "Yes.", keeps the voice's own intensity and pace, as the model was trained (`controls.delivery.min_spoken_phones`). Its pitch shift still applies.
- Pitch shifts the predicted contour in semitones; a soft limit keeps it inside the voice's measured range (`controls.pitch.edges`).

Controls are **global within each request**. This release does not support control curves or emotion labels; singing takes its melody from a score (see Singing).

## Model and inference

| Property | Value |
| --- | --- |
| Public model version | 4 (the v3 speech models, with a delivery mode added to the flow for singing) |
| Architecture | Phone-duration predictor → pitch predictor (rectified flow) → continuous rectified-flow latent generator that follows the pitch contour → acoustic adapter/Vocos |
| Audio | Mono 24 kHz; 100 mel features; 24 latent channels |
| Mel / latent hop | 256 / 512 samples |
| Duration model | Hidden size 320; 8 layers; reads the neighbouring text as context |
| Pitch predictor | Hidden size 384; 8 layers; a pitch and voicing contour per latent frame |
| Vector model | Hidden size 512; 12 layers; adaptive layer normalization and QK normalization; reads the pitch contour |
| Maximum graph phones per sentence | 512 |
| Trained target length | 64–420 latent frames (about 1.4–9 s) per pass |
| Sampler | Flow: Euler, 4 steps. Pitch predictor: Euler, 4 steps |
| Voice conditioning | Fixed per-voice/per-locale identity and prosody references embedded in the graphs |
| Text frontend | Trained Scylla's Band G2P with its matched tokenizer and vocabulary |

For each passage the pitch predictor integrates a pitch and voicing contour from noise, the pitch control shifts it, and the flow generates the latents along it (graph input `expanded_melody`, [1, frames, 3]). Each passage draws its pitch noise ([2, frames]) and then its latent noise ([24, frames]) from the request's seed.

The flow and the pitch predictor are distilled for few-step sampling: each was trained to reproduce its base model's 32-step Heun output, so Euler with 4 steps, four evaluations of each model per passage, is the default for both.

Long text is spoken as passages within the trained target lengths (64 to 420 latent frames, about 1.4 to 9 seconds): a short first passage so audio starts quickly, then passages as long as the text allows, cut at a sentence end, else a clause, else between words, and measured with the requested delivery. Each passage sees the surrounding text as context and continues from the acoustics of the one before it, and consecutive passages are decoded as one continuous waveform. Durations are predicted deterministically; different controls can alter them.

## Performance

Measured with the default samplers (flow and pitch predictor: Euler, 4 steps each) through the native runtime: `scyllasband_speak` from [libscyllasband](https://github.com/lowkeytea/scyllasband/tree/main/libscyllasband) on the Mac and Android, and the [iOS sample](https://github.com/lowkeytea/scyllasband/tree/main/examples/ios)'s benchmark mode on the iPhone. The text is three sentences, 10.3 s of audio: "The last train leaves at midnight, so pack light. Bring a warm coat, and don't forget the map. If we miss it, we walk, and nobody wants to walk that far in the rain." (voice Scylla, seed 7, the voice's own delivery).

- **First run**: from launch to first audio on the first launch after installing the bundle. Core ML compiles and Core AI specializes the graphs for the device then (Core AI caches them until the next OS update); LiteRT writes its packed weights to a cache file beside each graph.
- **Cold start**: from launch to first audio on a later launch, including loading the bundle.
- **Warm first audio**: from a request to its first audio in a process that has already spoken (median of three requests).
- **Real-time speed**: seconds of audio generated per second when warm, rounded.

| Device | Bundle | Compute | First run | Cold start | Warm first audio | Real-time speed |
| --- | --- | --- | --- | --- | --- | --- |
| MacBook Pro, M5 Max (macOS 27) | Core ML | GPU (default) | 2.73 s | 1.34 s | 30 ms | ~170× |
|  | Core ML | CPU | 2.17 s | 0.33 s | 221 ms | ~20× |
|  | Core AI | GPU (default) | 5.80 s | 0.46 s | 32 ms | ~150× |
|  | Core AI | Neural Engine | 34.2 s | 0.60 s | 316 ms | ~20× |
|  | Core AI | CPU | 3.54 s | 0.93 s | 844 ms | ~5.0× |
|  | LiteRT | CPU | 0.41 s | 0.30 s | 105 ms | ~34× |
|  | ONNX Runtime | CPU | 0.88 s | 0.85 s | 216 ms | ~21× |
| iPhone 17 Pro Max (iOS 27.2) | Core ML | GPU (default) | 15.5 s | 1.79 s | 77 ms | ~65× |
|  | Core ML | CPU | 2.87 s | 0.49 s | 294 ms | ~15× |
|  | Core AI | GPU (default) | 18.5 s | 0.59 s | 70 ms | ~68× |
|  | Core AI | Neural Engine | 8.00 s | 0.74 s | 367 ms | ~17× |
|  | Core AI | CPU | 7.58 s | 1.16 s | 1010 ms | ~4.1× |
| Galaxy Z Fold 8, Snapdragon SM8850 (Android 17) | LiteRT | CPU | 0.81 s | 0.52 s | 120 ms | ~27× |
|  | ONNX Runtime | CPU | 1.85 s | 1.89 s | 461 ms | ~9.2× |

The GPU is the default on Apple platforms. The Neural Engine placement runs the flow and the pitch predictor on the Neural Engine and the other graphs on the CPU, for apps that speak in the background, where iOS does not allow GPU work. Core ML reloads its graphs on every launch, which accounts for most of its cold start on the GPU. Each passage runs the pitch predictor 4 times, then the flow 4 times. Against `v3-20261103`, whose pitch predictor ran 16 times, warm first audio is 30–45 % shorter on every device and bundle (12–13 % with the Neural Engine placement).

### Singing cost

Singing is heavier than speech because the BigVGAN decoder is about 90 times the work of Vocos per second of audio; from
`v4-20261106` the flow itself costs the same as for speech (4 Euler evaluations per phrase, where `v4-20261105` ran 8 Heun
steps with guidance, 32 evaluations). `--decoder vocos` ("turbo") sings through the speech decoder: much faster, with some
artefacts on held notes in some voices. Phrases render one at a time and the runtime streams each as it finishes, so first
audio is the first phrase (up to 10 s of song).

Measured with `v4-20261106` (`v4-20261106-apple` on Apple platforms) on the 71 s pop example (`melodies/song_pop_gwen.json`,
voice Gwen, seed 7) through the native runtime, warm; first audio in brackets:

| Device | Bundle | Compute | BigVGAN: time (× real time) | Vocos: time (× real time) | Peak memory (BigVGAN) |
| --- | --- | --- | --- | --- | --- |
| MacBook Pro, M5 Max | Core ML | GPU | 0.98 s (73×) [0.14 s] | 0.14 s (510×) | 1.0 GB |
|  | Core AI | GPU | 1.3 s (56×) [0.18 s] | 0.16 s (440×) | 1.1 GB |
|  | Core AI | Neural Engine | 1.6 s (45×) [0.22 s] | 0.56 s (126×) | 1.2 GB |
|  | LiteRT | CPU | 20.2 s (3.5×) [2.9 s] | 1.2 s (59×) | 1.1 GB |
| iPhone 17 Pro Max | Core AI | GPU | 12.5 s (5.7×) [1.8 s] | 0.95 s (74×) | 920 MB |
|  | Core AI | Neural Engine | 14.0 s (5.1×) [2.0 s] | 0.9 s (80×) | 850 MB |
|  | Core ML | GPU, BigVGAN on the CPU | 31 s (2.3×) [4.5 s] | 0.9 s (78×) | 300 MB |
| Galaxy Z Fold 8 | LiteRT | GPU for BigVGAN (`AUTO`) | 19.2 s (3.7×) [1.9 s] | 1.7 s (42×) [0.26 s] | 764 MB |
|  | LiteRT | CPU | 71 s (1.0×) [9.5 s] | 1.7 s (42×) | 1.85 GB |

With `v4-20261105` the same song took 15.7 s (Core AI GPU) and 4.8 s (Turbo) on the iPhone, and 33 s and 15.6 s on the Fold.

- On iOS and visionOS the Core ML bundle runs BigVGAN on the CPU (`controls.coreml.compute_units_ios`): Core ML's GPU path
  for its largest bucket runs out of an iPhone's GPU memory, which aborts the app. Core AI runs it on the GPU, or on the
  Neural Engine with that placement, so songs can render in the background.
- The LiteRT bundle names BigVGAN as its one GPU graph (`controls.litert.gpu`), so the `AUTO` accelerator sings through
  the GPU and speaks on the CPU; on the Fold the first launch after an install compiles the GPU program (first audio
  about 5 s), later launches load it from a cache beside the bundle.
- On `v4-20261105`, Vocos with 4 Heun steps and no guidance (`--steps 4 --guidance 1`) sang the same song in 3.0 s on the
  Fold's CPU (24×); it kept more of the lyric (word coverage 98 % against 92 % over 14 voices and 3 phrases) and held pitch
  a little less tightly (96.7 % of notes within a semitone against 99.1 %). That trade-off is what the `v4-20261106`
  distillation removes: the unguided 4-step flow keeps both the clarity and the pitch, at 1.7 s for the song.

## Voices and language coverage

Fourteen voices: Ariadne, Felix, Gwen, Ink, Kincade, Max, Orpheus, Poppy, Priya, Rex, Scylla, Stone, Thistle, Tuesday.

All fourteen have Spanish (`es`), Italian (`it`), French (`fr`), German (`de`) and Vietnamese (`vi`). Ink, Orpheus and Tuesday use British English (`en_gb`); the other eleven use American English (`en_us`). This is **84 trained voice/locale pairs**, not every combination of fourteen voices and seven locale IDs.

## Artifacts

Every bundle carries the BigVGAN decoder (112 M parameters) beside the Vocos decoder. Tags: `v4-20261106` and
`v4-20261106-apple` both name this release with every bundle below; `download` uses `v4-20261106-apple`, following the
earlier releases' naming. Earlier releases keep their tags (`v3-20261104`, `v3-20261104-apple`, ...).

- `litert/` (471 MB, of which the BigVGAN decoder is 253 MB): LiteRT bundle for Linux, Windows, Android and Intel Macs. Dynamic-range INT8 weights on the fully connected, convolution and embedding layers of the speech graphs as in `v3`; the Vocos vocoder's depthwise convolutions stay in full precision. Latent buckets every 64 frames up to 512, then 768 and 1024; flow graphs without the 96 prefix frames for a request's first passage; narrower duration, span-context and pitch-context widths and G2P widths beside the full ones. BigVGAN (`vocoder_bigvgan.tflite`) is one 256-frame graph with FP16 weights, written channels-last so it has no layout shuffles and runs on LiteRT's GPU delegate; the runtime decodes longer phrases through it in windows with 16 frames of overlap either side of each cut (`controls.decoders.bigvgan.window`; audio within 68 dB of whole-phrase decoding). Against `v4-20261105`'s first LiteRT bundle (one full-precision BigVGAN graph per bucket, 524 MB), with the `v4-20261105` flow, it rendered the 71 s example song in 90 s instead of 165 s on the Fold's CPU at half the memory, and in 33 s on its GPU; with the `v4-20261106` flow, 71 s and 19 s. Its decoder window needs a runtime with ABI 3.1.0 (native or Python engine).
- `coreml/` (406 MB): Core ML bundle for iOS 18, visionOS 2 and macOS 15 and later. FP16 ML Programs; BigVGAN in buckets of 128 to 512 frames, decoded whole (no window); on iOS and visionOS it runs on the CPU (`controls.coreml.compute_units_ios`).
- `coreai/` (418 MB): Core AI bundle for iOS, visionOS and macOS 27 and later. BigVGAN is listed in `neural_engine_assets`: it runs on the Neural Engine with that placement.
- `onnx/` (1.1 GB): the full-precision ONNX bundle, for ONNX Runtime on the CPU and the reference for the release checks and the Apple exports: one shared weight file, latent buckets of 128, 256, 384, 512, 768 and 1024 frames for the flow, the pitch predictor and both decoders.
- `webgpu/` (782 MB): ONNX Runtime bundle for ONNX Runtime Web's WebGPU backend and for NVIDIA GPUs (CUDA). Transformer weights are INT8 with float compute (`MatMulNBits`), which runs entirely on the GPU; both decoders stay in full precision (BigVGAN is 450 MB of the bundle).
- `pytorch/` (1.1 GB): duration, pitch predictor, flow, Vocos adapter/vocoder, BigVGAN adapter and BigVGAN v2 (`bigvgan_adapter.pt`, `bigvgan_v2.pt`) and G2P weights with their model configurations (no training state or training settings), and the release configuration assets (`scyllasband_pytorch_release_v4`).
- `melodies/`: twenty example scores (four whole songs, sixteen phrases) with a README on the format.
- `SHA256SUMS.json`: published-file hashes.

Every runtime bundle includes the G2P model in a matching precision.

## Validation and limitations

Release validation separates graph accuracy, frontend/timing behaviour and perceived speech quality:

- The full-precision ONNX bundle run through the runtime against the PyTorch model, end to end with all three controls (Scylla, Ink and Gwen, two targets each): identical durations, and latents within 8.5e-5 (values of magnitude 16–28), the same as `v3`.
- The C++ runtime against the Python runtime on the same bundle: identical G2P output, phone durations and passage plan; latents within 2e-5 relative error; decoded audio at 46 dB SNR (worst sentence) and 55 dB over a long passage.
- The distilled flow at 4 Euler steps compared with the undistilled flow by listening on paired lines across the voices, and by ASR word error rate and a MOS predictor on a 168-clip sweep of all fourteen voices in English, French and Spanish.
- The distilled pitch predictor at 4 Euler steps compared with the undistilled predictor at 8 Heun steps (release `v3-20261103`) by blind listening on paired lines across the voices and languages, and on the same 168-clip sweep: ASR word error rate 0.062 for both, and a MOS-predictor score 0.12 higher.
- Quantized bundles compared with full precision through the runtime on lines covering every voice, each control at its limits and each language. The WebGPU bundle reproduces the full-precision take (identical frame counts on 15 of 17 lines, 0.68 dB median mel-cepstral distance, against 29 dB for the full-precision bundle at another seed; a sung pop phrase comes out the same length with the same pitch). Dynamic INT8 moves some phone durations by a frame, which redraws a passage's noise and gives a different take, so the LiteRT bundle is judged on the same sweep (measured on `v3`'s speech models): ASR word error rate 0.062 against 0.062 for full precision, and a MOS-predictor score within 0.02 of full precision.
- Every LiteRT graph gives identical output with and without XNNPACK's weight cache.
- Core ML and Core AI bundles compared with the PyTorch model on every graph and through the runtime on 14 passages (every voice, each control at both limits, six languages, narration), on each compute unit they are used with (GPU, CPU, and for Core AI the Neural Engine): identical phonemes, fewer changed phone durations than the LiteRT bundle (0.2–0.7 % against 1.9 %), and smaller pitch-contour (0.016–0.045 semitones RMS against 0.051), latent and spectral differences. The Apple exports rewrite three computations into equivalent forms that FP16 and the Neural Engine evaluate exactly (span pooling, the modifier bit decoding, the duration model's combined phone/segment index), and scale the residual stream of the flow and the pitch predictor; in FP32 the rewritten graphs match the trained ones (identical outputs; the span encoders within 1e-6).

Listening is decisive; automatic scores support it. Training audio is not distributed.

Known problems include occasional pronunciation errors and occasional roughness on fast, wide pitch movements in some voices.

The system supports managed voices, not arbitrary speaker cloning. Intended uses include local speech, narration, character dialogue and delivery-control research. Do not use it for impersonation or deceptive attribution of speech.

## Singing validation

The singing path was validated on the PyTorch model through the release renderer (adapter + BigVGAN v2) and on the built
bundles through the runtime's `sing` command:

- **Score tests, all fourteen voices** (`v4-20261105` flow, 8 Heun steps, guidance 2). Eight hand-written pop phrases and
  eight ballad phrases per voice (lyrics and notes, each phrase transposed into the voice's register), scored by Whisper lyric precision and coverage
  and by the sung pitch against the score (per-note median absolute error in cents, share of notes within 100 cents):

  | Mode | Lyric precision (mean / lowest voice) | Coverage (mean) | Pitch error (median, cents) | Notes within 100 cents |
  | --- | --- | --- | --- | --- |
  | Pop (`sung`) | 0.981 / 0.933 (Orpheus) | 0.976 | 15 | 1.000 (every voice) |
  | Ballad | 0.981 / 0.934 (Rex) | 0.972 | 11 | 1.000 (every voice) |

  Eleven voices score 0.96 or better in both modes; Orpheus, Poppy and Rex are the lowest at 0.93–0.97.
- **Shared flow against `v3`.** On the four check voices (Felix, Gwen, Max, Scylla) the reflow-distilled release flow
  matches the undistilled singing teacher's lyric precision (0.987 against 0.987). Spoken output was compared with the `v3`
  release flow on the 17 release check lines (every voice, each control at its limits, six languages): the same durations,
  a median mel-cepstral distance of 2.4 dB, and no audible difference in paired listening. Distilling the spoken rows from
  the `v3` base (a second teacher) rather than from the singing teacher brought that distance down from 3.4 dB.
- **Guidance distillation (`v4-20261106`).** The second distillation was scored on eight pieces (four phrases, the two
  example songs, the Ghost in the Machine chorus and a sixteenth-note patter at 112 bpm) rendered through the full-precision
  ONNX bundle with the same seed, against the `v4-20261105` flow at its shipped and its fast settings:

  | Flow and setting | Flow evaluations | Lyric precision | Pitch error (median, cents) | Notes within 100 cents |
  | --- | --- | --- | --- | --- |
  | `v4-20261105`, 8 Heun steps, guidance 2 (shipped) | 32 | 0.979 | 9.4 | 0.993 |
  | `v4-20261105`, 4 Heun steps, no guidance | 8 | 0.986 | 15.0 | 0.966 |
  | `v4-20261106`, 4 Euler steps, no guidance (default) | 4 | 0.969 | 9.4 | 0.991 |
  | `v4-20261106`, 4 Heun steps, no guidance | 8 | 0.960 | 9.4 | 0.992 |

  The `v4-20261106` word scores carry one Whisper insertion ("and" before a Scylla phrase, also at step 8000 of the
  distillation, where the set scored 0.984) rather than a dropped lyric; pitch matches the guided setting. On the sung
  validation set the distilled flow's latent output at 4 Euler steps sits 0.547 (RMS) from the teacher's guided 32-step
  Heun output, against 0.726 for the `v4-20261105` flow at 8 Heun steps with guidance. Spoken output moved by a median
  mel-cepstral distance of 2.3 dB from `v4-20261105` on the 17 release check lines (2.7 dB from `v3`, against 2.4 dB for
  `v4-20261105`; the same durations), and export parity against PyTorch is unchanged (worst 8.5e-5).
- **Renderer.** Blind paired listening on Scylla, Felix, Max, Gwen and Poppy preferred the fine-tuned adapter + BigVGAN v2
  renderer over the stock BigVGAN v2 and over Vocos for sung targets; Vocos adds harmonic artefacts on sustained notes in
  some voices (Scylla most), which the BigVGAN path does not.
- **Built bundles through the runtime.** The `sing` command on the `v4-20261106` LiteRT bundle at its defaults (4 Euler
  steps, no guidance, windowed BigVGAN), every voice on one pop and one ballad phrase and the four example songs, scored the
  same way: pop phrase lyric precision 1.00 for all fourteen voices (median pitch error 10 cents, coverage 1.00); ballad
  phrase 0.92 mean (0.875 lowest, one inserted leading syllable on the long first note of that phrase, coverage 1.00);
  songs 0.92–0.98 precision with 9–19 cents median pitch error and 95–100 % of notes within 100 cents. The `v4-20261105`
  bundle at 8 Heun steps with guidance scored 1.00 / 0.89 (0.67 lowest) / 0.87–0.94 on the same set. The same phrases through the full-precision ONNX
  bundle give identical scores, and the LiteRT take sits 1.4 dB (mel-cepstral) from the full-precision take against 27–32 dB
  for a different seed: the dynamic-INT8 flow reproduces the full-precision singing take.
- **Whole songs.** Four example songs (`melodies/song_*.json`, 2 pop and 2 ballads, 20–40 s) render phrase by phrase through
  the runtime; on the PyTorch path through the singing teacher they scored a lyric precision of 0.90–0.93.

Known limitations of the singing path: the model never free-sings (a score is required); a rap delivery row was trained but
is not exposed as a style in this release; the runtime transposes a score into the voice's register by whole octaves only;
a syllable held over many notes is rendered as one vowel across them; and Orpheus, Poppy and Rex drop the occasional word.

- **Core ML and Core AI bundles (`v4-20261106-apple`).** Against the FP32 PyTorch model on 20 sung cases (every voice on a pop
  phrase, four ballads, two more), with the same noise: identical phrase plans (20/20) on every compute unit; flow latents
  within 0.9–1.2 % (LiteRT 1.8 %); BigVGAN within 0.017–0.019 log-mel on the GPU and Neural Engine and 0.090 on Core ML's CPU
  path; end to end 0.034–0.10 (a reseeded take of the same score is 0.90 away). The release's sing check on both bundles
  (every voice on pop and ballad phrase 02, the four songs; faster-whisper and pYIN): lyric coverage 1.00 on all 28 phrases,
  median pitch error about 10 cents, songs 0.87–1.00 coverage with 95–100 % of notes within 100 cents. Speech stays closer to
  FP32 than the LiteRT bundle on every Apple compute unit, as for `v4-20261105`.

## License and acknowledgments

Apache 2.0. LiteRT and ONNX Runtime provide inference; Vocos (`charactr/vocos-mel-24khz`) is the base of the 24 kHz speech waveform decoder, and BigVGAN v2 (`nvidia/bigvgan_v2_24khz_100band_256x`, NVIDIA, MIT) is the base of the singing waveform decoder; both are fine-tuned on the voices, with notices in `THIRD_PARTY_NOTICES.md`. DeepPhonemizer-derived tooling, Wiktionary-derived vocabulary, eSpeak phonemization and Montreal Forced Aligner contributed to the frontend and alignment workflow. Vocos and DeepPhonemizer are MIT-licensed; their notices are in `THIRD_PARTY_NOTICES.md`.
