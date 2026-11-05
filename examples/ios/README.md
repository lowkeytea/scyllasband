# Scylla's Band iOS sample

This SwiftUI sample speaks multi-voice, multilingual documents on iPhone,
iPad and Apple Vision Pro with Scylla's Band v3 (release `v3-20261104`). It
uses Apple's Core ML runtime on iOS 18–26 and visionOS 2–26, and Core AI on
iOS and visionOS 27 and later. It keeps one warmed
`SBScyllasBand` runtime, reads voices, languages and delivery ranges from the
bundle at run time, streams each sentence's mono Float32 audio into
`AVAudioEngine` as soon as it's ready, and keeps the sentence being spoken on
screen.

A document is a list of speaker-segment cards. Each card has a voice, a
language, and a delivery made of three controls:

- **Intensity**, from calm (left) to intense (right), in the voice's own
  standard deviations. The emotion itself comes from the text.
- **Pace**, from slower to faster, words and pauses together, also in the
  voice's own standard deviations.
- **Pitch**, a shift of the voice in semitones.

Each slider runs from −2 to 2 in 0.1 steps, the range the bundle reports
through `scyllasband_voices_json`, and 0 is the voice's own delivery. A
control you haven't set takes the voice's default (0 for every voice in this
release). As in the Android sample, the segment settings have one preset,
**Voice default**, which sets all three controls back to the voice's own.
**Load example** imports the repository's `data/` documents:
`walkthrough_demo.txt` (the first-launch document, a tagged dialogue across
voices and languages that demonstrates the three controls),
`test_document.txt` (long narration), and `emotional_text.txt` (emotional
prose). Tagged lines become cards using the same rules as
`python -m scyllasband group-speak`:

```text
[ariadne:en_us:intensity=1] Good to see you. [es] Me alegra verte.
[ink:en_gb:intensity=-1,pace=-0.5] Keep this between us.
[poppy:fr:pitch=1.5] Et la hauteur !
```

A language tag such as `[es]` keeps the voice and delivery. A tag with a
delivery replaces the delivery, and controls it leaves out return to the
voice's own; `default` (alone, or as a value such as `pace=default`) leaves
controls at the voice's default. Values outside −2 to 2 are clamped into the
range, as the runtime would, so `intensity=3` becomes 2. The v2 delivery axes
(energy, tension, valence, assertiveness, whisper) are rejected with an error
that names the axis. `[ink]` alone switches to that voice in its default
language. Each line starts again from the defaults.

The gear button opens the runtime settings. They show the active backend
(Core ML or Core AI) and compute unit and let you choose where the models run:
Automatic (the bundle's recommendation, the default), CPU, GPU, or Neural
Engine. Changing the compute unit reloads the voices.

## Requirements

- A Mac with Xcode 27 and CocoaPods
- An iPhone or iPad with iOS 18 or later, an Apple Vision Pro with visionOS 2
  or later, or the iOS or visionOS Simulator. Core AI needs a device with iOS
  or visionOS 27 or later. Earlier systems and the Simulators run the Core ML
  bundle.
- At least one v3 model bundle in `scyllasband/models`, downloaded at the
  repository root:

  ```bash
  python -m scyllasband download --flavor coreml --yes   # iOS 18 / visionOS 2 and later, and the Simulators
  python -m scyllasband download --flavor coreai --yes   # iOS and visionOS 27 and later devices
  ```

  The Core ML and Core AI bundles of `v3-20261104` are published under the
  tag `v3-20261104-apple`, which these commands fetch. On a Mac,
  `python -m scyllasband download` without flags asks which bundle to
  download. Bundles from the v2 model don't work with this version of the
  app.

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

# Apple Vision Pro (or 'platform=visionOS Simulator,name=Apple Vision Pro')
xcodebuild -workspace ScyllasBandStudio.xcworkspace -scheme ScyllasBandStudio \
  -destination 'generic/platform=visionOS' build
```

The one `ScyllasBandStudio` target builds for iPhone, iPad and Apple Vision.
CocoaPods generates pod targets for a single platform, so the `Podfile`'s
`post_install` adds the visionOS SDKs to them; keep it when you run
`pod install`.

To run the app, open `ScyllasBandStudio.xcworkspace` (not the `.xcodeproj`,
which fails with `No such module 'ScyllasBandKit'`), set your development team
in `Config/Local.xcconfig` (git-ignored; one line, `DEVELOPMENT_TEAM = <team id>`,
included by the project's `Config/Studio.xcconfig`, so regenerating the project
keeps it), and run the `ScyllasBandStudio` scheme. The app needs no network, microphone,
or storage permission.

The **Prepare Scylla's Band assets** build phase
(`Scripts/prepare_assets.sh`) checks `scyllasband/models/coreai` and
`scyllasband/models/coreml`, following symlinks. It embeds each bundle found
in the app as `scyllasband/coreai` and `scyllasband/coreml`, together with
the three example documents, and fails if neither bundle exists. Each bundle
must contain all seven v3 graphs: `g2p`, `duration_predictor`, the pitch
predictor (`pitch_context_encoder` and `pitch_estimator`),
`vector_context_encoder`, `vector_estimator` and `vocoder`; the phase fails
on a v2 bundle, which has no pitch predictor. A v4 bundle that sings through
BigVGAN must also contain `vocoder_bigvgan`. The example scores in
`scyllasband/models/melodies` (installed by `python -m scyllasband download`
from the singing release) are embedded as `scyllasband/melodies` for the Sing
tab. Simulator
builds embed only Core ML. Set `SCYLLASBAND_IOS_BUNDLE_DIR` to embed one
specific bundle directory instead. At launch, the app runs Core AI on iOS and
visionOS 27 devices when that bundle was embedded and Core ML everywhere else, falling
back to Core ML if Core AI can't load.

In Debug builds, the launch argument `-autoplay` plays the first-launch
document as soon as the voices are ready. `-showSegmentSettings` opens the
first segment's settings. For the Sing tab, `-singTab` opens it,
`-singExample <name>` loads an example score (`song_pop_gwen`,
`phrase_pop_00`, …), `-singTurbo` selects Turbo, `-selectFirstNote` opens the
note editor and `-autosing` sings once the voices are ready. The app logs runtime timings under the
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

## Sing

The Sing tab builds a melody line by line and sings it with a v4 bundle (the
singing release). Type a line's lyrics (`ba-by` splits a word into syllables,
`_` holds the previous syllable over another note); each syllable becomes a
note chip under a sketch of the line's melody. Tap a chip to set its pitch on a
one-octave keyboard, its length (a sixteenth to a whole note), or to turn it
into a rest, split it, join it with the next note or delete it. Lines are sung
one at a time, separated by their rests. The song card sets the title, voice,
style (pop or ballad), tempo, sound and transpose:

- **Quality** renders with BigVGAN, the singing decoder; **Turbo** with Vocos,
  the speech decoder, which is much faster and adds some harmonic artefacts on
  sustained notes in some voices.
- The key readout says how the melody sits in the voice's range after the
  automatic octave register, and suggests a transpose when it sits outside.

**Preview** plays the melody as tones at the pitch it will be sung, at once.
**Sing** renders phrase by phrase; it plays while it renders when rendering
keeps ahead of playback (always with Turbo), otherwise it renders the song and
then plays it. The take is kept until the song changes, and **Another take**
draws a new one. Songs save to the app's Documents, and the share button exports
the score (the runtime's JSON, which `python -m scyllasband sing --score` also
reads) and the last take as a WAV file. The **Examples** menu loads the
release's twenty example scores as editable songs.

Every check of a song goes through the runtime: the score is validated by
`SBScyllasBand.parseScore(json:transpose:)` (an error names the note at fault,
which the editor outlines in red), the range by `singFit(for:)` and the tones
by `preview(for:)`. None of these runs a model or waits for a song in progress.

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
  does most of the work) and the pitch predictor run on the Neural Engine and
  the other models on the CPU; the bundle lists which models are accurate on
  the Neural Engine in `controls.coreai.neural_engine_assets`.
- **Core ML:** choose **CPU**. The Core ML bundle lists no Neural Engine models.

On an iPhone 17 Pro Max, a three-sentence passage (10.3 s of speech) starts
playing after about 110 ms on the GPU (Core AI or Core ML), 420 ms with Core
AI's Neural Engine placement and 450 ms on Core ML's CPU, with a warm runtime.
Core AI on the CPU alone is much slower (about 1.6 s).

## On-device benchmark

The app has a timing mode used for the repository's measurements. It creates no
studio runtime; it speaks a fixed passage four times with the voice's default
delivery, prints one `SBBENCH {…}`
JSON line (load time, launch-to-first-audio, warm first audio, real-time factor,
memory footprint) and exits:

```bash
xcrun devicectl device process launch --device <udid> --console --terminate-existing \
  org.scyllasband.studio -benchmark coreai auto 4
```

`<bundle>` is `coreai`, `coreml` or a folder copied to the app's
`Documents/benchmark/`; the unit is `auto`, `cpu`, `gpu` or `ane`.

`-benchmarkSing <bundle> <unit> <score> <decoder> [requests]` sings an embedded
example score (`phrase_pop_02.json`, `song_pop_gwen.json`, …) with `bigvgan` or
`vocos` and reports the same figures for singing. Pass
`--environment-variables '{"SCYLLASBAND_COMPUTE_UNITS": "cpu,vector_estimator=ane"}'`
with `auto` to try another per-model placement.

## Reusing the bridge

The app contains no C or C++. To add synthesis to another app:

1. Copy the complete `libscyllasband` directory into the destination
   repository.
2. Add the local static pod:

   ```ruby
   platform :ios, '18.0'        # or platform :visionos, '2.0'
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
request.delivery = SBScyllasBandDelivery(intensity: 1, pace: -0.5, pitch: nil)   // nil: the voice's default

try runtime.synthesizeRequest(request, chunkStarted: { index, count, text in
    print("sentence \(index + 1)/\(count): \(text ?? "")")
    return true
}, audioChunk: { chunk in
    // chunk.pcmFloat32Data: mono Float32 at chunk.sampleRate; schedule it before returning.
    return true
})
```

`runtime.bundleInfo.deliveryRanges` gives the range of each control
(`.intensity`, `.pace`, `.pitch`) and each voice's `deliveryDefaults` what a
control left nil uses. `requestCancellation()` is safe from any thread. It
stops requests already issued at their next model call, and the call then
throws `SBScyllasBandErrorCancelled`. See [ScyllasBandKit](../../libscyllasband/apple/README.md)
for the full API.
