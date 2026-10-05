# Scylla's Band iOS sample

This SwiftUI sample speaks multi-voice, multilingual documents on iPhone and
iPad with Scylla's Band v2 (release `v2-20261005`). It uses Apple's Core ML
runtime on iOS 18–26 and Core AI on iOS 27 and later. It keeps one warmed
`SBScyllasBand` runtime, reads voices and languages from the bundle at run
time, streams each sentence's mono Float32 audio into `AVAudioEngine` as soon
as it's ready, and keeps the sentence being spoken on screen.

A document is a list of speaker-segment cards. Each card has a voice, a
language, and a delivery: energy, tension, valence, and assertiveness, with 2
as neutral, plus whisper on/off. The model accepts 0–4 for each control, but
this app keeps them between 0.5 and 3.1: the training data thins out past
about 3, and pushing further distorts the voice. The segment settings also
offer the Neutral, Calm, Assertive, Joyful, Angry, Sad, and Whisper presets,
the same as the main README's (for example, Joyful is
`energy=2.4,tension=2,valence=2.8,assertiveness=2.2`). **Load
example** imports the repository's `data/` documents: `walkthrough_demo.txt`
(the first-launch document, a tagged dialogue across voices and languages),
`test_document.txt` (long narration), and `emotional_text.txt` (energetic
prose). Tagged lines become cards using the same rules as
`python -m scyllasband group-speak`:

```text
[ariadne:en_us:energy=2.3,valence=2.5] Good to see you. [es] Me alegra verte.
[ink:en_gb:whisper=on] Keep this between us.
```

A language tag such as `[es]` keeps the voice and delivery. A tag with a
delivery replaces the delivery, and controls it leaves out return to 2. Tag
values may use the full 0–4 range; the app clamps them to 0.5–3.1, so
`energy=3.3` becomes 3. `[ink]` alone switches to that voice in its default
language. Each line starts again from the defaults.

The gear button opens the runtime settings. They show the active backend
(Core ML or Core AI) and compute unit and let you choose where the models run:
Automatic (the bundle's recommendation, the default), CPU, GPU, or Neural
Engine. Changing the compute unit reloads the voices.

## Requirements

- A Mac with Xcode 27 and CocoaPods
- An iPhone or iPad with iOS 18 or later, or the iOS Simulator. Core AI needs
  a device with iOS 27 or later. Earlier systems and the Simulator run the
  Core ML bundle.
- At least one model bundle in `scyllasband/models`, downloaded at the
  repository root:

  ```bash
  python -m scyllasband download --flavor coreml --yes   # iOS 18 and later, and the Simulator
  python -m scyllasband download --flavor coreai --yes   # iOS 27 and later devices
  ```

  On a Mac, `python -m scyllasband download` without flags asks which bundle
  to download.

## Build and run

```bash
cd examples/ios
pod install

# Simulator (Core ML on the CPU)
xcodebuild -workspace ScyllasBandStudio.xcworkspace -scheme ScyllasBandStudio \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' build

# Device
xcodebuild -workspace ScyllasBandStudio.xcworkspace -scheme ScyllasBandStudio \
  -destination 'generic/platform=iOS' build
```

To run the app, open `ScyllasBandStudio.xcworkspace` (not the `.xcodeproj`,
which fails with `No such module 'ScyllasBandKit'`), select a development team,
and run the `ScyllasBandStudio` scheme. The app needs no network, microphone,
or storage permission.

The **Prepare Scylla's Band assets** build phase
(`Scripts/prepare_assets.sh`) checks `scyllasband/models/coreai` and
`scyllasband/models/coreml`, following symlinks. It embeds each bundle found
in the app as `scyllasband/coreai` and `scyllasband/coreml`, together with
the three example documents, and fails if neither bundle exists. Simulator
builds embed only Core ML. Set `SCYLLASBAND_IOS_BUNDLE_DIR` to embed one
specific bundle directory instead. At launch, the app runs Core AI on iOS 27
devices when that bundle was embedded and Core ML everywhere else, falling
back to Core ML if Core AI can't load.

In Debug builds, the launch argument `-autoplay` plays the first-launch
document as soon as the voices are ready. `-showSegmentSettings` opens the
first segment's settings. The app logs runtime timings under the
`org.scyllasband.studio` subsystem:

```bash
xcrun simctl launch booted org.scyllasband.studio -autoplay
xcrun simctl spawn booted log stream --predicate 'subsystem == "org.scyllasband.studio"'
```

The Xcode project is checked in. Regenerate it only after adding sources or
changing generated build settings:

```bash
Scripts/generate_xcode_project.rb
pod install
```

## First launch

The app creates and warms the runtime on a background queue and shows
**Preparing voices…** until it's ready. The first time a Core AI bundle loads
after the app is installed or iOS is updated, Core AI specializes its models
for the device. That can take tens of seconds; the system caches the result,
so later launches are quick. Core ML also takes longest on its first load.

The Simulator runs Core ML on the CPU only, so the app fixes its compute unit
to CPU there, and synthesis is much slower than on a device.

## Background playback

This sample doesn't declare the `audio` background mode, so speech stops when
the app leaves the foreground. To keep speaking in the background, add `audio`
to `UIBackgroundModes` and switch away from the GPU: iOS doesn't allow GPU work
from background apps, and Automatic uses the GPU (the bundles recommend it in
`controls.<backend>.compute_units`).

- **Core AI:** choose **Neural Engine**. The speech generator (the model that
  does most of the work) runs on the Neural Engine and the other models on the
  CPU; the bundle lists which models are accurate on the Neural Engine in
  `controls.coreai.neural_engine_assets`.
- **Core ML:** choose **CPU**. The Core ML bundle lists no Neural Engine models.

On an iPhone 17 Pro Max, a three-sentence passage (11.6 s of speech) starts
playing after about 240 ms on the GPU (Core AI or Core ML), 400 ms with Core
AI's Neural Engine placement and 770 ms on Core ML's CPU, with a warm runtime.
Core AI on the CPU alone is much slower (about 2.5 s).

## On-device benchmark

The app has a timing mode used for the repository's measurements. It creates no
studio runtime; it speaks a fixed passage four times, prints one `SBBENCH {…}`
JSON line (load time, launch-to-first-audio, warm first audio, real-time factor,
memory footprint) and exits:

```bash
xcrun devicectl device process launch --device <udid> --console --terminate-existing \
  org.scyllasband.studio -benchmark coreai auto 4
```

`<bundle>` is `coreai`, `coreml` or a folder copied to the app's
`Documents/benchmark/`; the unit is `auto`, `cpu`, `gpu` or `ane`. Pass
`--environment-variables '{"SCYLLASBAND_COMPUTE_UNITS": "cpu,vector_estimator=ane"}'`
with `auto` to try another per-model placement.

## Reusing the bridge

The app contains no C or C++. To add synthesis to another app:

1. Copy the complete `libscyllasband` directory into the destination
   repository.
2. Add the local static pod:

   ```ruby
   platform :ios, '18.0'
   use_frameworks! :linkage => :static
   pod 'ScyllasBandKit', :path => '../path/to/libscyllasband'
   ```

3. Run `pod install`, open the workspace, and `import ScyllasBandKit`.
4. Ship the `coreml` and/or `coreai` bundle directories unchanged, for
   example as app resources.
5. Create, warm, and use one runtime from a single serial queue, never the
   main thread.

```swift
let directory = Bundle.main.resourceURL!.appendingPathComponent("scyllasband")
guard let bundleURL = SBScyllasBand.preferredBundleURL(inDirectory: directory) else { return }
let runtime = try SBScyllasBand(bundleURL: bundleURL, computeUnit: .automatic)
try runtime.warmUp(withVoice: nil)

let request = SBScyllasBandRequest(text: "Finally, Scylla's Band is singing on Apple hardware.",
                                   voiceIdentifier: "ariadne")
request.language = "en_us"
request.delivery = SBScyllasBandDelivery(energy: 3, tension: 2, valence: 3, assertiveness: 2, whisper: false)

try runtime.synthesizeRequest(request, chunkStarted: { index, count, text in
    print("sentence \(index + 1)/\(count): \(text ?? "")")
    return true
}, audioChunk: { chunk in
    // chunk.pcmFloat32Data: mono Float32 at chunk.sampleRate; schedule it before returning.
    return true
})
```

`requestCancellation()` is safe from any thread. It stops requests already
issued at their next model call, and the call then throws
`SBScyllasBandErrorCancelled`. See [ScyllasBandKit](../../libscyllasband/apple/README.md)
for the full API.
