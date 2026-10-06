# Scylla's Band Android sample

An on-device text-to-speech studio for the Scylla's Band v2 model (`scyllasband_measured_delivery_v2` bundles). It runs the native `libscyllasband` runtime through LiteRT by default; nothing leaves the phone.

## What it demonstrates

- All ten voices and each voice's languages, read from the runtime (`scyllasband_voices_json`) rather than hard-coded.
- Measured delivery: four continuous sliders, energy, tension, valence and assertiveness, each 0.5-3.1 with 2 neutral (0.1 steps), plus a whisper switch. Quick presets (Neutral, Calm, Assertive, Joyful, Angry, Sad, Whisper) only fill the sliders. The app limits the four axes to 0.5-3.1 (see Delivery range):

  | Preset | Delivery |
  | --- | --- |
  | Calm | `energy=1.2,tension=0.8,valence=2.6` |
  | Assertive | `energy=2.6,tension=2.2,assertiveness=3` |
  | Joyful | `energy=3,valence=3` |
  | Angry | `energy=3,tension=3,valence=0.8,assertiveness=3` |
  | Sad | `energy=1,tension=1.6,valence=0.8,assertiveness=1.6` |
  | Whisper | `whisper=on` |

- A text editor with inline speaker points (voice, language, delivery). Double-tap or long-press to add a point, tap one to edit or remove it. A point applies until the next one; new points copy the last one's settings.
- An example selector loading the repository's `data/` files: `walkthrough_demo.txt` (tagged multi-voice, multilingual dialogue; also the first-launch document), `test_document.txt` (long narration) and `emotional_text.txt` (energetic prose). Tagged files become editable speaker points.
- Streaming playback: each sentence the runtime finishes is queued to a mono float `AudioTrack` while the next renders. The playback transcript follows the audible sentence. Stop calls `scyllasband_cancel`, which ends synthesis at the next graph call.

### Dialogue tags

Imported text uses the same format as `python -m scyllasband group-speak`:

```text
[ariadne:en_us:energy=2.3,valence=2.5] Good to see you. [es] Me alegra verte.
[ink:en_gb:whisper=on] Keep this between us.
```

`[voice:language:delivery]` sets all three (an empty voice or language keeps the active one). A language-only tag such as `[es]` keeps the active voice and delivery. A tag with a delivery part replaces the delivery, and axes it does not name return to neutral. `[voice]` alone switches voice. Each line starts from the default settings. `auto` delivery is not supported by this editor, and axes outside 0.5-3.1 are clamped.

## Build

1. Download the LiteRT bundle (about 164 MB, INT8) from the repository root:

   ```bash
   python -m scyllasband download --yes
   ```

2. Stage the LiteRT Android prebuilts (git-ignored, under `libscyllasband/third_party/litert/lib/<platform>/`):

   ```bash
   python libscyllasband/scripts/stage_litert_sdk.py --platform android-arm64 --download-runtime --download-gpu-accelerator --overwrite
   python libscyllasband/scripts/stage_litert_sdk.py --platform android-x86_64 --download-runtime --download-gpu-accelerator --overwrite
   ```

3. Build from this directory (needs the Android SDK, NDK 29.0.13113456 and CMake):

   ```bash
   ./gradlew :app:assembleDebug
   ./gradlew :app:testDebugUnitTest :scyllasband-android:testDebugUnitTest
   ```

The APK is `app/build/outputs/apk/debug/app-debug.apk`. It embeds the bundle as assets and the first launch copies it to `noBackupFilesDir`, because the runtime needs filesystem paths. Production apps should deliver the model as an install-time asset pack or a one-time download, and keep only the target ABI. arm64-v8a and x86_64 are built; arm64-v8a is the one that matters on devices.

`libLiteRt.so` and `libLiteRtClGlAccelerator.so` are packaged with `jniLibs.useLegacyPackaging = true` so they are extracted to disk: LiteRT loads its GPU accelerator from the directory of `libLiteRt.so`.

### ONNX Runtime instead of LiteRT

```bash
python -m scyllasband download --flavor onnx --yes
./gradlew -Pscyllasband.backend=onnx :app:assembleDebug
```

This links `libscyllasband` against ONNX Runtime 1.30.0 (`onnxruntime-android`) and embeds `scyllasband/models/onnx` (about 250 MB). ONNX Runtime builds run on the CPU. LiteRT is the default.

## Accelerator

`ScyllasBand.initialize(threadCount, accelerator)` takes `CPU`, `GPU` or `AUTO`. The sample uses `CPU` (XNNPACK, up to 4 threads), the default.

CPU is fast enough: on a Galaxy Z Fold 8 (Snapdragon SM8850) the native runtime speaks a three-sentence passage (11.6 s of audio) with first audio about 1.3 s after launch and 180 ms once warm, about 27 times faster than real time. The LiteRT GPU accelerator (OpenCL/OpenGL, `libLiteRtClGlAccelerator.so`) did not help there with this model: it cannot run the INT64 `CAST`/`ADD` and `GATHER_ND` ops, so graphs are split across the GPU and CPU, and
- `AUTO` fails at the first invoke (`LITERT_OPENGL failed to invoke`, status 3, in the warmup). `initialize` then recreates the runtime on the CPU, so `AUTO` always ends up usable but only ever gives CPU speed here;
- `GPU` failed to build a delegate kernel on the Adreno (`Unable to parse bc coord for BATCH axis`) and initialization never finished.

The Android emulator behaves like `AUTO` on the device: the GPU graphs compile but fail when invoked.

## Delivery range

The app keeps energy, tension, valence and assertiveness between 0.5 and 3.1 (2 is neutral). The model accepts 0-4, but training data thins out past about 3 (tension reaches 3.1) and pushing further distorts the voice. The sliders are limited to that range, the presets (the same as the main README's) lie within it, and values read from tagged documents are clamped when they become speaker points. `ScyllasBandDelivery` itself still validates the full 0-4 range; the limit is an app-level choice.

## Minimal integration

```kotlin
private val scyllasband by lazy { ScyllasBand.create(applicationContext) }

// Off the main thread: installs the bundle, creates the runtime, warms it up.
val info = scyllasband.initialize()   // info.voices, info.sampleRate, info.backend, info.accelerator

val settings = ScyllasBandSegmentSettings(
    voiceId = "ariadne",
    language = "en_us",
    delivery = ScyllasBandDelivery(energy = 2.5f, valence = 2.3f),
)
scyllasband.synthesizeStreaming(
    text = text,
    settings = settings,
    seed = 31_415L,                      // optional; null draws fresh noise
    onChunkStarted = { index, count, sentence -> true },
    onAudioChunk = { samples, index, count -> true },   // return false to stop
)

scyllasband.cancel()   // from any thread
scyllasband.close()
```

Synthesis uses the bundle defaults (8 Heun steps); `steps` is an optional parameter. `ScyllasBandDelivery.spec()` produces the ABI delivery string, for example `energy=2.5,tension=2,valence=2.3,assertiveness=2,whisper=off`.

## Modules

- `libscyllasband` is the shared C++ runtime (ABI 2.0.0).
- `scyllasband-android` builds it with `-DSCYLLASBAND_BACKEND=litert|onnx`, adds the JNI layer, installs the bundle and exposes the Kotlin API above.
- `app` owns the marker editor, point dialog, document segmentation, lifecycle and audio playback.

The app needs no network, microphone or storage permission.
