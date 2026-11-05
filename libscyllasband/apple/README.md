# ScyllasBandKit

`ScyllasBandKit` is the Apple bridge for `libscyllasband` (ABI 3.0.0, the
Scylla's Band v3 model, `scyllasband_measured_controls_v3` bundles). It runs
Core ML bundles on iOS 18 and visionOS 2 and later, and Core AI bundles on iOS
and visionOS 27 and later, through
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
| `SBScyllasBandBundleInfo` | `backend` (`coreml` or `coreai`), `releaseIdentifier`, `sampleRate`, `voices`, `defaultVoice`, and `deliveryRanges`: the range the runtime clamps each delivery control into, keyed by `SBScyllasBandDeliveryControl`. |
| `SBScyllasBandVoice` | `identifier`, `languages`, `defaultLanguage`, and `deliveryDefaults`: what a request that leaves a control out uses for this voice (0 for each control in the released model). |
| `SBScyllasBandRequest` | Text, voice, optional language, delivery (initially `+voiceDefault`), seed, flow steps, speed (a plain time stretch on top of pace), text normalization. |
| `SBScyllasBandDelivery` | Intensity, pace and pitch as optional `NSNumber`s (nil: the voice's default); `specification` is the runtime's delivery string, e.g. `intensity=1,pace=-0.5` (empty when no control is set). `+voiceDefault` sets no control, `+neutral` sets each to 0. |
| `SBScyllasBandDeliveryControl` | `.intensity`, `.pace`, `.pitch`: the keys of `deliveryRanges`. |
| `SBScyllasBandDeliveryRange` | `minimum`, `maximum` and `unit` (`sd` for intensity and pace, `semitones` for pitch). |
| `SBScyllasBandAudioChunk` | One sentence of mono Float32 PCM. The chunk owns its copy of the data. |

```swift
let runtime = try SBScyllasBand(bundleURL: bundleURL, computeUnit: .automatic)
try runtime.warmUp(withVoice: nil)

let request = SBScyllasBandRequest(text: "Keep this between us.", voiceIdentifier: "ink")
request.delivery = SBScyllasBandDelivery(intensity: -1, pace: -0.5, pitch: nil)   // pitch: the voice's default
try runtime.synthesizeRequest(request, chunkStarted: nil) { chunk in
    // Schedule chunk.pcmFloat32Data (chunk.sampleRate Hz) before returning; return false to stop.
    return true
}
```

### Delivery

A delivery has three controls, each optional; 0 is the voice's own delivery.

| Control | Negative | Positive | Unit |
| --- | --- | --- | --- |
| `intensity` | calmer | more intense | the voice's own standard deviations |
| `pace` | slower | faster (words and pauses together) | the voice's own standard deviations |
| `pitch` | lower | higher | semitones |

The emotion itself comes from the text. The released model supports -2 to 2
for each control: `SBScyllasBandDelivery` clamps finite values into that
range, and the runtime clamps every request into the bundle's ranges
(`bundleInfo.deliveryRanges`, read from `scyllasband_voices_json`). A control
left nil takes the voice's default (`voice.deliveryDefaults`, from the
manifest's `controls.delivery.defaults`). The v2 axes (energy, tension,
valence, assertiveness, whisper) and their 0–4 scale are gone; the runtime
rejects them.

```swift
let range = runtime.bundleInfo.deliveryRanges[.pitch]   // minimum -2, maximum 2, unit "semitones"
let defaults = runtime.bundleInfo.voice(withIdentifier: "ink")?.deliveryDefaults   // intensity=0,pace=0,pitch=0
let pace = request.delivery.pace?.floatValue   // nil: the voice's default
```

Rules:

- Pass the URL of a complete v3 or v4 bundle directory: `manifest.json`, `assets/`,
  and `coreml/*.mlmodelc` or `coreai/*.aimodel` (`g2p`, `duration_predictor`,
  `pitch_context_encoder`, `pitch_estimator`, `vector_context_encoder`,
  `vector_estimator`, `vocoder`; v4 adds `vocoder_bigvgan` for singing).
  Bundles from the v2 model don't load.
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

### Singing

A v4 bundle sings melody scores (`bundleInfo.singing.isSupported`). The
[iOS sample's](../../examples/ios/README.md) Sing tab builds scores in an editor.

| Type or method | Purpose |
| --- | --- |
| `SBScyllasBandSingRequest` | Score JSON, voice, optional language, seed, steps, guidance, `decoder`, `transpose`, `usesRegister`. |
| `SBScyllasBandSingingDecoder` | `.default` (the bundle's singing decoder, BigVGAN), `.bigVGAN`, `.vocos` (turbo: much faster, with some artefacts on held notes in some voices). |
| `SBScyllasBandSingingInfo` | `bundleInfo.singing`: styles, decoders, the singing steps, sampler and guidance, each voice's MIDI centre, the register band, and the longest phrase. |
| `sing(_:phraseStarted:phraseAudio:)` | Sings phrase by phrase; each `SBScyllasBandSungPhrase` is the next run of the song (`startSample`, contiguous), so a player appends them. |
| `sing(_:)` | The whole song as one `SBScyllasBandAudioChunk`. |
| `singPlan(for:)` | The phrase plan, without singing. |
| `parseScore(json:transpose:)` (class) | Checks and normalizes a score without a runtime; errors carry the note at fault (`SBScyllasBandScoreNoteIndexKey`). |
| `singFit(for:)`, `preview(for:)` | How the song sits in the voice's range (with a suggested transpose), and the melody as plain tones. Neither runs a graph or waits for a song in progress. |
| `warmUpSinging(voice:decoder:)` | Loads the graphs a short phrase needs. |

```swift
let request = SBScyllasBandSingRequest(scoreJSON: score, voiceIdentifier: "gwen")
request.decoder = .vocos   // turbo; .default sings through BigVGAN
try runtime.sing(request, phraseStarted: nil) { phrase in
    // Append phrase.pcmFloat32Data (phrase.sampleRate Hz) to the player; return false to stop.
    return true
}
```

BigVGAN runs on the GPU with `Automatic` and on the Neural Engine with
`NeuralEngine` on Core AI bundles (it is listed in `neural_engine_assets`), so
songs can render in the background too. On iOS and visionOS a Core ML bundle
runs it on the CPU with `Automatic` (`controls.coreml.compute_units_ios`):
Core ML's GPU path for its largest bucket runs out of an iPhone's GPU memory,
which aborts the app. On an iPhone that sings a 71 s song in 29 s.

## macOS library and CLI

The same backend builds with CMake on macOS 15 and later. Core AI bundles need
macOS 27.

```bash
cmake -S libscyllasband -B build/apple -G Ninja \
  -DSCYLLASBAND_BACKEND=apple -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0
cmake --build build/apple

build/apple/scyllasband_speak --bundle scyllasband/models/coreml \
  --voice ariadne --delivery intensity=1,pace=-0.5 --accelerator gpu \
  --text "Hello from Core ML." --output /tmp/scyllasband.wav
```

`--accelerator` accepts `cpu` (the default), `gpu`, `auto` (the bundle's
recommendation), or `ane` (the graphs the bundle marks as accurate on the
Neural Engine run there, the rest on the CPU).

See the [iOS sample](../../examples/ios/README.md) for a complete SwiftUI app
with streaming playback, cancellation, and runtime settings.
