# ScyllasBandKit

`ScyllasBandKit` is the reusable Apple bridge for `libscyllasband`. Its public
header is `Sources/include/ScyllasBandKit.h`; an app imports only
`ScyllasBandKit` and does not expose C++, Core AI, or model-runtime details to
its own source code.

The bridge requires iOS 27 or macOS 27 because it executes native Core AI
`.aimodel` assets. The pod contains the C ABI, shared duration-flow runtime,
Swift Core AI session adapter, Objective-C facade, and podspec as one copyable
unit.

## Add it to an iOS app

Copy `libscyllasband` into the destination repository, then add this local pod:

```ruby
platform :ios, '27.0'
use_frameworks! :linkage => :static
pod 'ScyllasBandKit', :path => '../path/to/libscyllasband'
```

Run `pod install`, open the generated `.xcworkspace`, and use:

```swift
import ScyllasBandKit
```

The pod links `CoreAI`, `Foundation`, and libc++, defines the Core AI backend,
and deliberately has no ONNX Runtime dependency. `CoreAISession.swift`
implements the private graph-session C ABI expected by the shared C++ planner,
including Core AI specialization, fixed-shape multifunction selection, tensor
conversion, and owned output buffers.

## Runtime integration rules

- Pass the file URL of a complete Core AI bundle directory. Preserve every
  relative path below `manifest.json`, including the `.aimodel` directories,
  G2P tokenizer/dictionary, voice packs, and language assets.
- Create and warm one `SBScyllasBand` from a serial worker queue, then reuse
  it. Core AI frame buckets are functions in shared-weight assets, so the
  bridge keeps the single model session alive instead of evicting logical
  buckets.
- Call synthesis off the main thread. Stream callbacks execute synchronously
  on the same worker thread.
- Each `SBScyllasBandAudioChunk` owns a copy of its mono Float32 PCM bytes and
  remains safe after the native callback returns.
- `requestCancellation` and `resetCancellation` are atomic. Other runtime
  calls remain serialized.
- The bundle may replace `coreai/g2p.aimodel` and its matching files under
  `assets/g2p/` independently of the acoustic assets. This is intentional for
  the evolving seven-language G2P release.

## macOS library and CLI

On macOS 27 with Xcode 27 selected, build the same Core AI bridge plus a native
smoke-test CLI with:

```bash
DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer \
  libscyllasband/scripts/build_coreai_macos.sh

libscyllasband/build/coreai-macos/scyllasband_coreai_speak \
  --bundle /path/to/coreai-bundle \
  --backend coreai \
  --litert-accelerator gpu \
  --text "Hello from Core AI." \
  --output /tmp/scyllasband.wav
```

The legacy `--litert-accelerator` spelling is retained in the C ABI and CLI
for compatibility; on the Core AI backend it selects the preferred Core AI
compute unit rather than LiteRT.

See the [iOS sample](../../examples/ios/README.md) for complete SwiftUI,
subtitle, cancellation, and `AVAudioEngine` integration.
