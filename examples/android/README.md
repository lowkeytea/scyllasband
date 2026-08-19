# Scylla's Band Android sample

This sample demonstrates the current Scylla's Band duration-flow model through the native ONNX Runtime backend.

## What it demonstrates

- The optional `scyllasband` INT8 ONNX bundle, including external shared weights and 24 kHz audio playback.
- All voices and each voice's supported languages, loaded from `manifest.json` rather than hard-coded UI lists.
- Neutral delivery or any v2 affect axis (`calm`, `joy`, `anger`, `sadness`, and `whisper`), a normalized strength from 0 to 1, and a separate non-negative CFG value.
- A paste-friendly text editor with inline speaker points. Double-tap or long-press to add a point; tap a point to edit or remove it. A point controls all following text until another point appears. New points copy the settings of the last point created.
- A preset selector populated from the repository's `data/` examples. The multilingual `groupSpeak.txt` tags are converted into editable inline speaker points when loaded.
- Language-aware spoken-text normalization matching the Python runtime for numbers, dates, times, currency, percentages, fractions, and common symbols.
- `groupSpeak`-style playback: the document becomes ordered text/settings segments, long segments use the native Scylla's Band long-form planner, and each completed native audio chunk is streamed directly to a mono float `AudioTrack` while later chunks render.
- A playback-only transcript replaces the locked editor while speaking and follows the current native-planned chunk, voice, and language. Transcript changes are scheduled against `AudioTrack.playbackHeadPosition`, so queued audio does not move the display ahead of what is audible.
- Startup performs one discarded one-step render to warm the persistent ONNX sessions. Normal Android playback uses the core Python `--faster` quality settings of two Heun steps and 180-character maximum chunks; native long-form retry splitting remains enabled.

## Build

From this directory:

```bash
./gradlew :app:assembleDebug
```

The build prefers the CPU-optimized v2 bundle at `scyllasband/models/v2/onnx-int8`, then v2 FP32, v1, and the legacy flat layout. If v2 is missing, run this first from the repository root:

```bash
python -m scyllasband download --model-version v2 --runtime-bundles onnx-int8 --yes
```

The debug APK is written to `app/build/outputs/apk/debug/app-debug.apk`.

The sample deliberately packages the complete INT8 ONNX bundle for offline use. The model bundle is roughly 292 MB (278 MiB), and first launch copies it into `noBackupFilesDir` because ONNX Runtime needs filesystem paths for the model and its external weight file. Production apps should normally deliver the model as an install-time asset pack or download it once, and should keep only the target ABI.

## Minimal integration

The app creates one wrapper, initializes it once on its worker, reuses it for
every segment, and closes it with the activity. The default initialization is
the documented mobile profile: warmup enabled, platform-derived thread count,
and one cached target bucket.

```kotlin
private val scyllasband by lazy { ScyllasBand.create(applicationContext) }

// Run off the main thread. This installs assets, creates one runtime, and warms it.
val bundleInfo = scyllasband.initialize()

scyllasband.synthesizeSegmentStreaming(
    text = text,
    settings = settings,
    seed = 31_415L,
    onChunkStarted = { _, _, _ -> true },
    onAudioChunk = { samples, _, _ ->
        // Retain or consume samples before returning. Return false to cancel.
        true
    },
)

scyllasband.close()
```

High-memory clients that prioritize steady-state throughput can explicitly use
`initialize(targetBucketCacheCapacity = 0)` to retain every encountered
bucket. Values above one provide intermediate profiles. Keep this decision in
the integration layer; the UI does not manage ONNX sessions.

## Modules

- `libscyllasband` owns model planning, target-bucket selection, ONNX session reuse, and LRU eviction. It exposes the same cache control through the public C ABI used by other hosts.
- `scyllasband-android` builds `libscyllasband` with `SCYLLASBAND_ENABLE_ONNX=ON`, links it to the native library from `onnxruntime-android`, installs the bundle, selects the documented mobile defaults, and exposes a small Kotlin API.
- Streaming playback hands each native ONNX audio callback to a dedicated `AudioTrack` thread through a two-chunk bounded queue. This lets synthesis overlap playback across chunk and speaker-segment boundaries without retaining a long queue of waveforms or letting the displayed transcript run far ahead of audible speech.
- `app` owns the marker editor, settings dialogs, document segmentation, lifecycle, and audio playback.

The app needs no network, microphone, or storage permission at runtime.

On first launch, the editor opens with a tagged walkthrough converted into editable speaker points for Scylla, Max, Tuesday, Spanish-speaking Rex, and Italian-speaking Ink. It demonstrates language switching, emotion strength, CFG, and the same tag-import workflow used by the bundled group conversation preset.

The sample logs time to first audio, total playback time, and native-heap change under the `ScyllasBandPlayback` log tag. These measurements make repeated-play regressions visible with `adb logcat` without adding profiling UI to the app.

## Memory and performance profile

The INT8 ONNX bundle uses separate vector/vocoder graphs for four latent-frame
buckets, backed by a roughly 260 MB (248 MiB) external-weight file. The default
capacity-one LRU bounds the number of warm vector/vocoder bucket sessions.
Re-measure native memory and latency on each target device: the earlier FP32
figures do not describe this INT8 build, and debug-build results remain
device-specific.

Useful checks:

```bash
adb logcat -s ScyllasBandPlayback:I '*:S'
adb shell dumpsys meminfo org.scyllasband.demo
```

When comparing profiles, use the same script and stop point. A capacity of one
should stabilize after the first complete bucket replacement; continued growth
across identical repeat runs is a regression.
