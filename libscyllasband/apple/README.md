# ScyllasBandKit

`ScyllasBandKit` is the Apple bridge for `libscyllasband`. Its public header is
`Sources/include/ScyllasBandKit.h`; clients never need to expose C++ or ONNX
Runtime headers to application code.

The bridge is intentionally packaged at the `libscyllasband` root so the unit
you copy contains the C ABI, native implementation, Apple façade, and podspec.
Add it to an iOS 16+ app with:

```ruby
use_frameworks! :linkage => :static
pod 'ScyllasBandKit', :path => '../path/to/libscyllasband'
```

The pod pulls Microsoft's `onnxruntime-c` iOS binary, compiles the native
runtime as C++17, and exposes the resulting static framework as
`ScyllasBandKit`.

Integration rules:

- Pass the file URL of a complete ONNX bundle directory. Keep all manifest,
  model, external-weight, G2P, and voice-pack relative paths intact.
- Create and warm one `SBScyllasBand` from a serial worker queue, then reuse
  it. A target-bucket cache capacity of one is the default mobile profile.
- Call synthesis off the main thread. Stream callbacks execute synchronously
  on that same calling thread.
- Each `SBScyllasBandAudioChunk` owns a copy of its mono Float32 PCM bytes, so
  it is safe to enqueue the data after the native callback returns.
- `requestCancellation` and `resetCancellation` are atomic. Reset once before
  enqueueing a new playback run; request cancellation from UI or lifecycle
  code. Other runtime calls stay on the serial worker.
- The bridge owns inference policy and safe language-boundary types. The host
  app owns asset delivery, audio playback, interruption handling, and UI.

See the [iOS sample](../../examples/ios/README.md) for a complete SwiftUI and
`AVAudioEngine` integration.
