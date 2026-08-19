# Scylla's Band iOS sample

This SwiftUI sample mirrors the Android app's integration behavior on iPhone
and iPad while using Apple's native Core AI runtime. It keeps one warmed
`SBScyllasBand`, reads voices/languages/affect axes from `manifest.json`, plans
native long-form chunks, streams mono Float32 audio into `AVAudioEngine`, and
keeps the currently spoken text visible while playback continues.

The UI represents Android's inline speaker points as explicit speaker-segment
cards. Settings apply until the next segment, and tagged `groupSpeak` scripts
become editable segment cards when imported. All colors use semantic system
colors so light and dark appearance remain readable.

## Build and run

Requirements:

- macOS with Xcode; the Core AI path needs Xcode 27 (its SDK ships
  `CoreAI.framework` for devices only, so Core AI does not run in Simulator)
- An iOS 16+ device or simulator; Core AI synthesis activates on iOS 27+
- CocoaPods
- Downloaded model bundles: run `python -m scyllasband download` at the
  repository root and select v2 plus the desired ONNX bundle

```bash
python -m scyllasband download --model-version v2 --runtime-bundles onnx-int8 --yes

cd examples/ios
pod install

xcodebuild -workspace ScyllasBandStudio.xcworkspace \
  -scheme ScyllasBandStudio \
  -destination 'generic/platform=iOS' build
```

Then open `ScyllasBandStudio.xcworkspace`, choose the
`ScyllasBandStudio` scheme, select a Development Team, and run it. The app
needs no network, microphone, or user-storage permission.

The asset build phase searches `scyllasband/models/v2` before `models/v1` and
the legacy flat layout. It embeds Core AI, when present, plus one ONNX bundle
(`onnx-int8` preferred, `onnx` otherwise) from the same release, preventing
cross-generation model mixing. At launch the app picks
Core AI on iOS 27+ when it was embedded and falls back to the ONNX bundle on
earlier systems. Set `SCYLLASBAND_IOS_BUNDLE_DIR` to embed one specific
bundle instead.

The Xcode project is checked in. Regenerate it only after adding sources or
changing generated build settings:

```bash
Scripts/generate_xcode_project.rb
pod install
```

Always open the `.xcworkspace` after installing pods. Opening only the
`.xcodeproj` produces `No such module 'ScyllasBandKit'`.

## Reusing the bridge

The app owns no C or C++ integration code. To move synthesis into another app:

1. Copy the complete `libscyllasband` directory into the destination
   repository.
2. Add the local static pod:

   ```ruby
   platform :ios, '16.0'
   use_frameworks! :linkage => :static
   pod 'ScyllasBandKit', :path => '../path/to/libscyllasband'
   ```

3. Run `pod install`, open the workspace, and import `ScyllasBandKit`.
4. Deliver a complete bundle (Core AI for iOS 27+, ONNX for anything
   earlier) as an app resource, install-time asset, or one-time download.
   Preserve its layout and pass its directory URL to `SBScyllasBand`; the
   runtime picks the backend from the bundle's `preferred_backends`.
5. Create and warm one runtime on a serial queue and reuse it for synthesis.

Minimal Swift integration:

```swift
let runtime = try SBScyllasBand(
    bundleURL: bundleURL,
    threadCount: min(ProcessInfo.processInfo.activeProcessorCount, 6),
    targetBucketCacheCapacity: SBScyllasBand.defaultMobileTargetBucketCacheCapacity
)
try runtime.warmUp()

let settings = SBScyllasBandSegmentSettings(
    voiceIdentifier: "gwen",
    language: "en_us",
    emotion: "neutral",
    emotionStrength: 0.6,
    emotionCFG: 1.2
)

try runtime.synthesizeText(
    "Finally, Scylla's Band is singing on Apple hardware.",
    settings: settings,
    seed: 31_415,
    chunkStarted: { index, count, text in
        print("rendering \(index + 1)/\(count): \(text ?? "")")
        return true
    },
    audioChunk: { chunk in
        // Schedule chunk.pcmFloat32Data before returning.
        return true
    }
)
```

The public bridge owns manifest discovery, runtime lifetime, warmup,
cancellation, safe PCM copies, and language-boundary types. `libscyllasband`
owns normalization, Core AI G2P, duration prediction, long-form planning,
shared-weight function selection, acoustic sampling, and waveform generation.
The app owns document editing, playback, interruptions, lifecycle, and UI.

The G2P asset is versioned independently: a future seven-language G2P update
can replace `coreai/g2p.aimodel` plus `assets/g2p/` without reconverting the
four-language acoustic checkpoints.
