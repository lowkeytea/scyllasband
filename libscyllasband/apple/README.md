# ScyllasBandKit

`ScyllasBandKit` is the Apple bridge for `libscyllasband`. It runs Core ML
bundles on iOS 18 and visionOS 2 and later, and Core AI bundles on iOS and
visionOS 27 and later, through
the Apple graph backend (`src/scyllasband_graph_apple.mm`, with the Core AI
Swift bridge `Sources/ScyllasBandCoreAI.swift`). Apps use only the
Objective-C API in `Sources/include/ScyllasBandKit.h`, which Swift can call,
and never see C++.

## Add it to an iOS or visionOS app

Copy the whole `libscyllasband` directory into the destination repository and
add the local pod:

```ruby
platform :ios, '18.0'        # or platform :visionos, '2.0'
use_frameworks! :linkage => :static
pod 'ScyllasBandKit', :path => '../path/to/libscyllasband'
```

The pod links CoreML and libc++ and has no ONNX Runtime or LiteRT dependency.
Device builds weak-link CoreAI, so the app still launches on iOS 18–26 and
visionOS 2–26. The Simulator SDKs have no CoreAI.framework, so Simulator
builds run Core ML only (on the CPU).

CocoaPods builds pods for the Podfile's one platform. An app target that
builds for both iPhone/iPad and Apple Vision (like the
[sample](../../examples/ios/Podfile)) also needs the visionOS SDKs added to
the pod targets in `post_install`.

## API

| Type | Purpose |
| --- | --- |
| `SBScyllasBand` | One runtime over one bundle: `initWithBundleURL:computeUnit:error:`, `warmUpWithVoice:error:`, `synthesizeRequest:chunkStarted:audioChunk:error:`, `requestCancellation`. |
| `+[SBScyllasBand preferredBundleURLInDirectory:]` | Picks `coreai/` on iOS and visionOS 27 and later, otherwise `coreml/`, from a directory holding the bundle folders. |
| `SBScyllasBandComputeUnit` | `Automatic` (the bundle's recommendation for each graph), `CPU`, `GPU`, `NeuralEngine`. |
| `SBScyllasBandBundleInfo` | `backend` (`coreml` or `coreai`), `releaseIdentifier`, `sampleRate`, `voices` (identifier, languages, default language), `defaultVoice`. |
| `SBScyllasBandRequest` | Text, voice, optional language, delivery, seed, flow steps, speed, text normalization. |
| `SBScyllasBandDelivery` | Energy, tension, valence and assertiveness (0–4, 2 neutral) plus whisper; `+neutral`. |
| `SBScyllasBandAudioChunk` | One sentence of mono Float32 PCM. The chunk owns its copy of the data. |

```swift
let runtime = try SBScyllasBand(bundleURL: bundleURL, computeUnit: .automatic)
try runtime.warmUp(withVoice: nil)

let request = SBScyllasBandRequest(text: "Keep this between us.", voiceIdentifier: "ink")
request.delivery = SBScyllasBandDelivery(energy: 2, tension: 2, valence: 2, assertiveness: 2, whisper: true)
try runtime.synthesizeRequest(request, chunkStarted: nil) { chunk in
    // Schedule chunk.pcmFloat32Data (chunk.sampleRate Hz) before returning; return false to stop.
    return true
}
```

Rules:

- Pass the URL of a complete bundle directory: `manifest.json`, `assets/`, and
  `coreml/*.mlmodelc` or `coreai/*.aimodel`.
- Create, warm, and call one runtime from a single serial queue, never the
  main thread. Callbacks run synchronously on that queue.
- The first load of a Core AI bundle specializes its graphs for the device,
  which can take tens of seconds. The system caches the result until the next
  OS update.
- `requestCancellation` is safe from any thread. It stops requests already
  issued at their next graph call, which then fail with
  `SBScyllasBandErrorCancelled`, and leaves later requests unaffected.
- iOS doesn't allow GPU work from background apps. For playback that
  continues in the background use `NeuralEngine` with a Core AI bundle (its
  speech generator runs on the Neural Engine, the other graphs on the CPU, as
  listed in the manifest's `neural_engine_assets`) or `CPU`.

## macOS library and CLI

The same backend builds with CMake on macOS 15 and later. Core AI bundles need
macOS 27.

```bash
cmake -S libscyllasband -B build/apple -G Ninja \
  -DSCYLLASBAND_BACKEND=apple -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0
cmake --build build/apple

build/apple/scyllasband_speak --bundle scyllasband/models/coreml \
  --voice ariadne --delivery energy=2.4,tension=2,valence=2.8,assertiveness=2.2 --accelerator gpu \
  --text "Hello from Core ML." --output /tmp/scyllasband.wav
```

`--accelerator` accepts `cpu` (the default), `gpu`, `auto` (the bundle's
recommendation), or `ane` (the graphs the bundle marks as accurate on the
Neural Engine run there, the rest on the CPU).

See the [iOS sample](../../examples/ios/README.md) for a complete SwiftUI app
with streaming playback, cancellation, and runtime settings.
