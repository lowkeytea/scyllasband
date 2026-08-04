# Scylla's Band iOS sample

This sample mirrors the Android Studio app's important integration behavior on
iPhone and iPad. It uses one persistent native ONNX runtime, warms it once,
reads voices/languages/affect axes from `manifest.json`, renders each speaker
segment with native long-form planning, and streams mono Float32 audio into
`AVAudioEngine` while subsequent chunks render.

The UI represents Android's inline speaker points as explicit speaker-segment
cards. This is friendlier to SwiftUI's text system while preserving the same
document model: settings apply to text until the next speaker segment. Tagged
`groupSpeak` scripts become editable segment cards when imported.

## Build and run

Requirements:

- Xcode 16 or newer
- CocoaPods
- An ONNX model bundle at `scyllasband/models/onnx-int8` (preferred) or
  `scyllasband/models/onnx`

From this directory:

```bash
pod install
open ScyllasBandStudio.xcworkspace
```

Choose the `ScyllasBandStudio` scheme. Select a Development Team when running
on a device. The app has no network, microphone, or user-storage permission.

The build phase embeds the complete model for a self-contained sample. It uses
`onnx-int8` when available and falls back to the larger repository-local ONNX
bundle. To use another bundle without editing the project:

```bash
SCYLLASBAND_IOS_BUNDLE_DIR=/absolute/path/to/bundle \
  xcodebuild -workspace ScyllasBandStudio.xcworkspace \
  -scheme ScyllasBandStudio \
  -destination 'generic/platform=iOS Simulator' build
```

The Xcode project is checked in. If sources are added, regenerate it with:

```bash
Scripts/generate_xcode_project.rb
pod install
```

## Reusing the bridge

The app owns no C or C++ integration code. That code lives with the native
runtime in `libscyllasband` and is packaged by
`libscyllasband/ScyllasBandKit.podspec`.

To bring it into another app today:

1. Copy the complete `libscyllasband` directory into the destination
   repository. Keeping the native sources beside the bridge avoids fragile
   relative paths and makes the copied unit self-contained.
2. Add the local pod to the app target:

   ```ruby
   use_frameworks! :linkage => :static
   pod 'ScyllasBandKit', :path => '../path/to/libscyllasband'
   ```

3. Run `pod install` and import `ScyllasBandKit` from Swift or Objective-C.
4. Deliver a complete model directory as an app resource, install-time asset,
   or one-time download. Preserve its internal directory layout and pass the
   directory's file URL to `SBScyllasBand`.
5. Create and warm one `SBScyllasBand` on a serial worker queue. Reuse it for
   all synthesis and release it when the owning service shuts down.

Minimal Swift integration:

```swift
let runtime = try SBScyllasBand(
    bundleURL: bundleURL,
    threadCount: min(ProcessInfo.processInfo.activeProcessorCount, 6),
    targetBucketCacheCapacity: SBScyllasBand.defaultMobileTargetBucketCacheCapacity
)
try runtime.warmUp()

let settings = SBScyllasBandSegmentSettings(
    voiceIdentifier: "scylla",
    language: "en_us",
    emotion: "joy",
    emotionStrength: 0.6,
    emotionCFG: 1.2
)

try runtime.synthesizeText(
    "Hello from an iPhone.",
    settings: settings,
    seed: 31_415,
    chunkStarted: { index, count, text in
        print("rendering \(index + 1)/\(count): \(text ?? "")")
        return true
    },
    audioChunk: { chunk in
        // Copy or schedule chunk.pcmFloat32Data before returning.
        // The bridge has already copied it out of libscyllasband-owned memory.
        return true
    }
)
```

`targetBucketCacheCapacity = 1` is the memory-first mobile profile. Zero keeps
every target bucket warm and uses substantially more memory. Keep synthesis on
one serial queue; `requestCancellation()` is the exception and is safe to call
from UI code. Call `resetCancellation()` once before enqueueing a new playback
run; individual synthesis calls deliberately do not erase an in-flight stop.

## Bridge boundary

- `libscyllasband` owns text normalization, G2P, duration prediction, chunk
  planning, ONNX sessions, target-bucket LRU, and waveform generation.
- `ScyllasBandKit` owns Objective-C/Swift value types, manifest discovery,
  runtime lifetime, validation, warmup defaults, cancellation, and safe PCM
  copies across the native callback boundary.
- The sample app owns document editing, preset import, lifecycle, status,
  transcript presentation, and `AVAudioEngine` playback.

ONNX Runtime's official iOS C/C++ distribution is the `onnxruntime-c`
CocoaPod. The podspec pins the compatible 1.21 release line; update and test
that dependency deliberately alongside Android rather than using an unbounded
version.
