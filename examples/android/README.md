# Scylla's Band Android sample

An on-device speech and singing studio for Scylla's Band (release `v4-20261106`, which speaks and sings). It runs the native `libscyllasband` runtime through LiteRT by default; nothing leaves the phone. Two tabs: **Speak** (a script with speaker points) and **Sing** (a melody built line by line).

## What it demonstrates

- All fourteen voices and each voice's languages, read from the runtime (`scyllasband_voices_json`) rather than hard-coded.
- Delivery: three sliders, intensity (calm to intense) and pace (slower to faster, words and pauses) in the voice's own standard deviations and pitch in semitones, each from -2 to 2 in 0.1 steps; 0 is the voice's own delivery. A reset button sets all three back to 0 (see Delivery range).
- A text editor with inline speaker points (voice, language, delivery). Double-tap or long-press to add a point, tap one to edit or remove it. A point applies until the next one; new points copy the last one's settings.
- An example selector loading the repository's `data/` files: `walkthrough_demo.txt` (tagged multi-voice, multilingual dialogue; also the first-launch document), `test_document.txt` (long narration) and `emotional_text.txt` (emotional prose). Tagged files become editable speaker points.
- Streaming playback: each sentence the runtime finishes is queued to a mono float `AudioTrack` while the next renders. The playback transcript follows the audible sentence. Stop calls `scyllasband_cancel`, which ends synthesis at the next graph call.

### Sing

The Sing tab builds a song as lines of notes and sings it with the runtime (ABI 3.1), as the iOS sample's Sing tab does:

- Type a line's lyrics (`ba-by` splits a word into syllables, `_` holds a syllable over another note), then tap a note chip to set its pitch on a one-octave keyboard, its length, or to make it a rest, split it, join it with the next note, insert a rest or delete it. Each line shows a sketch of its melody; rests of a quarter second or more separate the lines the voice sings one at a time.
- The song card opens the settings: title, voice, style (pop or ballad), tempo, sound (**Quality**, the BigVGAN decoder, or **Turbo**, Vocos) and key (transpose, with how the melody sits in the voice's range and a suggested transpose).
- Every change is checked by the runtime (`ScyllasBand.parseScore`, `singFit`; the note it objects to is outlined) without waiting for a song in progress. **Preview** plays the melody as tones (`preview`, no model runs).
- **Sing** streams the song while it renders when rendering keeps ahead of playback, else renders it, then plays; the take is kept until the song changes, and **Another take** sings it with a new seed.
- **Examples** loads the release's twenty example scores (`melodies/`, embedded from `scyllasband/models/melodies`); **Songs** starts a new song or opens one saved with **Save** (JSON in the app's files); **Share** sends the score (the runtime's format, which `python -m scyllasband sing --score` reads) and the last take as a WAV.

The song model (`app/.../sing/SongDocument.kt`, `SongEditing.kt`) is plain Kotlin with JVM tests against the runtime's own parse of every example (`./gradlew :app:testDebugUnitTest`). Debug builds take launch options for testing: `adb shell am start -n org.scyllasband.demo/.MainActivity --es sing_example song_pop_gwen [--ez sing_turbo true] --ez autosing true` opens the Sing tab, loads the example, sings it and logs `SBSING <status>` (tag `ScyllasBandSing`).

### Dialogue tags

Imported text uses the same format as `python -m scyllasband group-speak`:

```text
[ariadne:en_us:intensity=1] Good to see you. [es] Me alegra verte.
[ink:en_gb:intensity=-1,pace=-0.5] Keep this between us.
```

`[voice:language:delivery]` sets all three (an empty voice or language keeps the active one). A language-only tag such as `[es]` keeps the active voice and delivery. A tag with a delivery part replaces the delivery, and controls it does not name return to the voice's own. `[voice]` alone switches voice. Each line starts from the default settings. Values outside -2 to 2 are clamped into the range, as the runtime would; v2 delivery axes (energy, tension, valence, assertiveness) are rejected.

## Build

1. Download the LiteRT bundle (about 471 MB) and the example scores from the repository root:

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

`ScyllasBand.initialize(threadCount, accelerator)` takes `CPU` (XNNPACK, up to 4 threads; the default), `GPU` or `AUTO`. The sample uses `AUTO`: the graphs the bundle lists in `controls.litert.gpu.graphs` run on the GPU, everything else on the CPU, and a bundle that lists none runs entirely on the CPU.

LiteRT's GPU accelerator uses the vendor's OpenCL driver. Since Android 12 an app loads a vendor library only after declaring it, so the manifest has `<uses-native-library android:name="libOpenCL.so" android:required="false" />` (and the `-car` and `-pixel` variants). Without it LiteRT falls back to its OpenGL path, which fails, and every graph runs on the CPU: the GPU worked from `adb shell` but never inside the app.

CPU is fast enough for speech: on a Galaxy Z Fold 8 (Snapdragon SM8850) the native runtime speaks a three-sentence passage (10.3 s of audio) with first audio about 0.52 s after launch (0.8 s on the first launch, which writes the weight cache) and 120 ms once warm, about 27 times faster than real time.

For singing with BigVGAN use `AUTO` with a bundle that names BigVGAN as its GPU graph (`controls.litert.gpu.graphs`, written by datasetGen's LiteRT export with its channels-last BigVGAN graph): BigVGAN then runs on the GPU and every other graph on the CPU. With `v4-20261106` (the flow distilled to 4 Euler steps without guidance) the Fold sings the 71 s example song in 19 s with Quality (3.7 times faster than real time, first audio after 1.9 s, 764 MB; 71 s on the CPU alone) and in 1.7 s with Turbo (42 times, first audio after 0.26 s; Vocos runs on the CPU either way). `v4-20261105` took 33 s and 15.6 s. The first launch after an install compiles BigVGAN's GPU program (first audio about 5 s), later launches load it from the cache next to the bundle; the Sing tab warms up the song's decoder when it first opens. In the background (another app in front, or the screen locked) Android moves the app to its slower cores, so a song that should keep rendering there needs a foreground service.

Don't put every graph on the GPU with `GPU` (`AUTO` with a bundle that lists no GPU graphs stays on the CPU): with LiteRT 2.2.0 the duration predictor returns wrong durations there (speech plays too fast), the flow is slower than on the CPU, and the v4-20261105 BigVGAN graph aborts the process in the GPU delegate.

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

- `libscyllasband` is the shared C++ runtime (ABI 3.1.0).
- `scyllasband-android` builds it with `-DSCYLLASBAND_BACKEND=litert|onnx`, adds the JNI layer, installs the bundle and exposes the Kotlin API above.
- `app` owns the marker editor, point dialog, document segmentation, lifecycle and audio playback (Speak), and the song editor (`sing/`: SongDocument, SongEditing, SingScreen, PianoKeysView, ContourView, SongStore).

The app needs no network, microphone or storage permission.
