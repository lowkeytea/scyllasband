# libscyllasband

`libscyllasband` is the native Scylla's Band inference runtime. It exposes the
same duration-flow bundle contract used by the Python package and is the
foundation for C, C++, Swift, Kotlin, and other host integrations.

The public ABI is defined in `include/scyllasband.h`. Host code should use the
`scyllasband_*` functions, `SCYLLASBAND_*` constants, and `ScyllasBand*`
types. The current ABI and public bundle contract are version `1.0.0`.

## Runtime contract

A synthesis request can provide raw text or explicit phones, a managed voice,
language, speed, deterministic seed, Euler or Heun flow sampling, prefix
latents, neighboring text context, chunk boundaries, and the selected bundle's
manifest-declared affect vector. Current v2 bundles use:

`calm, joy, anger, sadness, whisper`

Pass affect as a comma-separated `axis=value` string such as
`anger=0.75,whisper=0.25`. Axis strengths are independently bounded to
`[0, 1]`. `affect_guidance_scale` is separate CFG guidance: `0` selects
the learned null-affect branch, `1` uses the requested vector directly, and
values above `1` amplify it. The runtime does not impose an upper cap, but
high values are extrapolation and can destabilize timing or audio.

An ONNX- or LiteRT-enabled runtime performs the complete native path:

```text
text -> bundle G2P -> duration prediction -> duration expansion
     -> Euler/Heun rectified-flow sampling -> Vocos -> waveform
```

It loads the bundle's phone/language indexes, G2P assets, pronunciation
overrides, and managed NPZ voice/reference packs. The returned
`ScyllasBandSynthesisResult` owns waveform samples, metadata, and generated
latent-tail storage; release them with
`scyllasband_synthesis_result_free()`.

Schema-v4 runtime packs must use stored ZIP members (`ZIP_STORED`). This keeps
the mobile/native loader dependency-free; compressed training packs are repacked during export.

## Runtime lifetime and target-bucket memory

Create one `ScyllasBandRuntime` per long-lived client and reuse it. The runtime
owns the G2P, duration, vector, and vocoder sessions; individual synthesis
results never own sessions. Destroy the runtime once during client shutdown.

The 256, 384, 512, and 640-frame target buckets have separate vector/vocoder
graphs. By default, native runtimes retain every bucket they encounter for
maximum desktop/server throughput. Mobile clients should bound that cache:

```c
ScyllasBandRuntimeOptions options = {0};
options.bundle_dir = bundle_path;
options.backend = SCYLLASBAND_BACKEND_ONNX;
options.validate_bundle = 1;

ScyllasBandRuntime* runtime = NULL;
if (scyllasband_runtime_create(&options, &runtime) != SCYLLASBAND_STATUS_OK) {
    /* Read scyllasband_last_error(). */
}

/* One warm target bucket: the recommended memory-first mobile profile. */
scyllasband_runtime_set_target_bucket_cache_capacity(runtime, 1);

/* Reuse runtime for every request, then destroy it once. */
scyllasband_runtime_destroy(runtime);
```

The capacity is the number of target-bucket vector/vocoder session groups,
not the number of synthesis requests or text chunks. A value of `0` is
unbounded; positive values use least-recently-used eviction. Lowering the
capacity trims existing sessions immediately. G2P, duration, and context
sessions remain warm because they are small and shared by every target bucket.
Synthesis metadata includes `session_cache.target_bucket_capacity`, count, and
the current least-to-most-recent bucket frame list.

Capacity `1` minimizes resident memory but pays session creation time when the
requested bucket changes. Higher capacities trade memory for fewer reloads.
Choose a profile once at runtime initialization and measure it on the target
device; application code should not manually unload component sessions.

## Build

A build without a graph backend validates the ABI, bundle parsing, planning, and native
utilities:

```bash
cmake -S scyllasband/libscyllasband \
      -B /tmp/scyllasband_native_build \
      -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/scyllasband_native_build -j
ctest --test-dir /tmp/scyllasband_native_build --output-on-failure
```

Native ONNX synthesis requires ONNX Runtime headers and a shared library:

```bash
cmake -S scyllasband/libscyllasband \
    -B /tmp/scyllasband_native_onnx \
    -DCMAKE_BUILD_TYPE=Release \
    -DSCYLLASBAND_ENABLE_ONNX=ON \
    -DSCYLLASBAND_ONNXRUNTIME_INCLUDE_DIR=/path/to/onnxruntime/include \
    -DSCYLLASBAND_ONNXRUNTIME_LIBRARY=/path/to/libonnxruntime.so
cmake --build /tmp/scyllasband_native_onnx -j
ctest --test-dir /tmp/scyllasband_native_onnx --output-on-failure
```

The [Android sample](../examples/android/README.md) extracts these files from Microsoft's `onnxruntime-android` AAR and packages the current ONNX bundle for offline use.

The [iOS sample](../examples/ios/README.md) uses the
[`ScyllasBandKit`](apple/README.md) local CocoaPod in this directory. The pod
compiles the native sources against Microsoft's `onnxruntime-c` iOS package
and exposes a Swift-friendly Objective-C API for manifest metadata, warmup,
streaming synthesis, PCM ownership, and cancellation.

Native LiteRT synthesis requires a staged LiteRT SDK:

```bash
python scyllasband/libscyllasband/scripts/stage_litert_sdk.py \
    --download-runtime \
    --download-gpu-accelerator \
    --overwrite

cmake -S scyllasband/libscyllasband \
    -B /tmp/scyllasband_native_litert \
    -DCMAKE_BUILD_TYPE=Release \
    -DSCYLLASBAND_ENABLE_LITERT=ON
cmake --build /tmp/scyllasband_native_litert -j
ctest --test-dir /tmp/scyllasband_native_litert --output-on-failure
```

The CMake library target is `scyllasband_native`, with the
`scyllasband::native` alias. Tools and tests are enabled by default and can
be controlled with `SCYLLASBAND_BUILD_TOOLS` and
`SCYLLASBAND_BUILD_TESTS`.

The Python runtime normally locates or builds this library through its shared
loader. Set `SCYLLASBAND_NATIVE_LIBRARY` only when a host needs to select an
explicit build.

## Native smoke synthesis

`scyllasband_native_speak` writes the returned float samples as PCM16 WAV and
can preserve metadata for comparison:

```bash
/tmp/scyllasband_native_litert/scyllasband_native_speak \
    --bundle output/scyllasband/release/litert \
    --text "Oh, wonderful. The alarm is singing again." \
    --voice scylla \
    --language en_us \
    --emotion anger=0.75,whisper=0.25 \
    --emotion-scale 1.5 \
    --steps 8 \
    --sampler heun \
    --seed 2027 \
    --output /tmp/scyllasband_native.wav \
    --metadata /tmp/scyllasband_native.json
```

Use a selected bundle's manifest as the authority for components, shapes,
controls, and supported voices/languages.

## Long-form and streaming

`scyllasband_runtime_synthesize_long_form()` plans paragraph, sentence, and
budget-aware chunks; assigns deterministic per-chunk seeds; carries compatible
prefix latents; retries overlong text, phone, or latent sequences with smaller
chunks; and joins the result with boundary-aware pauses.

`scyllasband_runtime_plan_long_form()` returns the same plan without
synthesizing. `scyllasband_runtime_synthesize_long_form_stream()` emits
plan, chunk-start, audio, chunk-finish, warning, and completion events through
`ScyllasBandStreamingCallback`.

The smoke tool exposes these paths:

```bash
# Inspect a plan.
scyllasband_native_speak --bundle BUNDLE --file story.txt \
    --long-form --plan-only --metadata /tmp/plan.json

# Stream chunks and timing events.
scyllasband_native_speak --bundle BUNDLE --file story.txt \
    --long-form --stream \
    --events /tmp/events.jsonl \
    --chunk-output-dir /tmp/scyllasband_chunks \
    --output /tmp/story.wav
```

## LiteRT accelerators

The LiteRT core library and the platform GPU accelerator library must both be
present under `third_party/litert/lib/<platform>/`. The staging helper supports
Android ARM64/x86-64, Linux x86-64/ARM64, Apple Silicon macOS, iOS
device/simulator, and Windows x86-64. Local SDK libraries are build artifacts
and must not be committed.

To stage previously downloaded libraries without network access:

```bash
python scyllasband/libscyllasband/scripts/stage_litert_sdk.py \
    --source-lib /path/to/libLiteRt.so \
    --source-gpu-accelerator-lib /path/to/libLiteRtWebGpuAccelerator.so \
    --overwrite
```

CPU is the correctness baseline. `--litert-accelerator auto` and `gpu`
request the corresponding LiteRT accelerator, but individual components may
fall back to CPU when a graph or platform plugin cannot run correctly. The
current public release uses one full vector-estimator graph, not separate
prefix and tail estimator artifacts. Treat returned metadata as authoritative
for the accelerator actually used.

On Linux, standalone tools without the repository RPATH may need the staged
library directory on `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH="$PWD/scyllasband/libscyllasband/third_party/litert/lib/linux-x86_64:${LD_LIBRARY_PATH:-}"
```

Preflight a staged runtime before performance or audio comparison:

```bash
python scyllasband/libscyllasband/scripts/benchmark_native_litert.py --preflight \
    --tool /tmp/scyllasband_native_litert/scyllasband_native_speak \
    --bundle output/scyllasband/release/litert \
    --litert-stage-root scyllasband/libscyllasband/third_party/litert
```

Performance baselines are host- and accelerator-specific. Validate audio
correctness first, compare the same bundle/settings/backend, and only then use
the benchmark helpers for regression thresholds.

## Current scope

ONNX and LiteRT synthesis, bundle validation, duration estimation, long-form planning,
aggregate synthesis, streaming callbacks, metadata, and low-level LiteRT tensor
sessions are implemented. CoreML remains a reserved backend selection; richer
platform bindings and packaging are separate host-integration work.
