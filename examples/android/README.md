# Scylla's Band Android sample

An on-device text-to-speech studio for the Scylla's Band v3 model (`scyllasband_measured_controls_v3` bundles). It runs the native `libscyllasband` runtime through LiteRT by default; nothing leaves the phone.

## What it demonstrates

- All fourteen voices and each voice's languages, read from the runtime (`scyllasband_voices_json`) rather than hard-coded.
- Delivery: three sliders, intensity (calm to intense) and pace (slower to faster, words and pauses) in the voice's own standard deviations and pitch in semitones, each from -2 to 2 in 0.1 steps; 0 is the voice's own delivery. A reset button sets all three back to 0 (see Delivery range).
- A text editor with inline speaker points (voice, language, delivery). Double-tap or long-press to add a point, tap one to edit or remove it. A point applies until the next one; new points copy the last one's settings.
- An example selector loading the repository's `data/` files: `walkthrough_demo.txt` (tagged multi-voice, multilingual dialogue; also the first-launch document), `test_document.txt` (long narration) and `emotional_text.txt` (emotional prose). Tagged files become editable speaker points.
- Streaming playback: each sentence the runtime finishes is queued to a mono float `AudioTrack` while the next renders. The playback transcript follows the audible sentence. Stop calls `scyllasband_cancel`, which ends synthesis at the next graph call.

### Dialogue tags

Imported text uses the same format as `python -m scyllasband group-speak`:

```text
[ariadne:en_us:intensity=1] Good to see you. [es] Me alegra verte.
[ink:en_gb:intensity=-1,pace=-0.5] Keep this between us.
```

`[voice:language:delivery]` sets all three (an empty voice or language keeps the active one). A language-only tag such as `[es]` keeps the active voice and delivery. A tag with a delivery part replaces the delivery, and controls it does not name return to the voice's own. `[voice]` alone switches voice. Each line starts from the default settings. Values outside -2 to 2 are clamped into the range, as the runtime would; v2 delivery axes (energy, tension, valence, assertiveness) are rejected.

## Build

1. Download the LiteRT bundle (about 229 MB, INT8) from the repository root:

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

This links `libscyllasband` against ONNX Runtime 1.30.0 (`onnxruntime-android`) and embeds `scyllasband/models/onnx` (about 291 MB). ONNX Runtime builds run on the CPU. LiteRT is the default.

## Accelerator

`ScyllasBand.initialize(threadCount, accelerator)` takes `CPU`, `GPU` or `AUTO`. The sample uses `CPU` (XNNPACK, up to 4 threads), the default.

CPU is fast enough: on a Galaxy Z Fold 8 (Snapdragon SM8850) the native runtime speaks a three-sentence passage (11.9 s of audio) with first audio about 0.5 s after launch (1.5 s on the first launch, which writes the weight cache) and 130 ms once warm, about 23 times faster than real time. The LiteRT GPU accelerator (OpenCL/OpenGL, `libLiteRtClGlAccelerator.so`) did not help there with this model: it cannot run the INT64 `CAST`/`ADD` and `GATHER_ND` ops, so graphs are split across the GPU and CPU, and
- `AUTO` fails at the first invoke (`LITERT_OPENGL failed to invoke`, status 3, in the warmup). `initialize` then recreates the runtime on the CPU, so `AUTO` always ends up usable but only ever gives CPU speed here;
- `GPU` failed to build a delegate kernel on the Adreno (`Unable to parse bc coord for BATCH axis`) and initialization never finished.

The Android emulator behaves like `AUTO` on the device: the GPU graphs compile but fail when invoked.

## Delivery range

The runtime supports intensity, pace and pitch from -2 to 2 (`scyllasband_voices_json` reports the ranges and each voice's defaults, read into `ScyllasBandBundleInfo.deliveryRanges` and `ScyllasBandVoice.deliveryDefaults`). The sliders cover exactly that range, and values read from tagged documents are clamped into it when they become speaker points. `ScyllasBandDelivery` rejects values outside it; a control left `null` takes the voice's default.

## Minimal integration

```kotlin
private val scyllasband by lazy { ScyllasBand.create(applicationContext) }

// Off the main thread: installs the bundle, creates the runtime, warms it up.
val info = scyllasband.initialize()   // info.voices, info.sampleRate, info.backend, info.accelerator

val settings = ScyllasBandSegmentSettings(
    voiceId = "ariadne",
    language = "en_us",
    delivery = ScyllasBandDelivery(intensity = 1f, pace = -0.5f),
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

Synthesis uses the bundle defaults (Euler, 4 steps); `steps` is an optional parameter. `ScyllasBandDelivery.spec()` produces the ABI delivery string, for example `intensity=1,pace=-0.5` (empty when every control takes the voice's default).

## Modules

- `libscyllasband` is the shared C++ runtime (ABI 3.0.0).
- `scyllasband-android` builds it with `-DSCYLLASBAND_BACKEND=litert|onnx`, adds the JNI layer, installs the bundle and exposes the Kotlin API above.
- `app` owns the marker editor, point dialog, document segmentation, lifecycle and audio playback.

The app needs no network, microphone or storage permission.
