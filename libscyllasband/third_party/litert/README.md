# Staged LiteRT SDK

LiteRT 2.2.0 C API headers from `google-ai-edge/LiteRT` (tag `v2.2.0`): `include/litert/c/`, the
`options/` headers and the internal header they include. `include/litert/build_common/build_config.h` is
hand-written for the prebuilt runtimes, which are built with GPU and NPU support.

Shared libraries are local build inputs under `lib/<platform>/` and are not committed:

```text
lib/
├── linux-x86_64/    libLiteRt.so, libLiteRtWebGpuAccelerator.so
├── android-arm64/   libLiteRt.so, libLiteRtClGlAccelerator.so
├── android-x86_64/  libLiteRt.so, libLiteRtClGlAccelerator.so
├── ios-arm64/       libLiteRt.dylib, libLiteRtMetalAccelerator.dylib
├── ios-sim-arm64/   libLiteRt.dylib, libLiteRtMetalAccelerator.dylib
└── macos-arm64/     libLiteRt.dylib, libLiteRtMetalAccelerator.dylib
```

Stage the published prebuilts (the default version is 2.2.0):

```bash
python scripts/stage_litert_sdk.py --download-runtime --download-gpu-accelerator --overwrite
python scripts/stage_litert_sdk.py --platform android-arm64 --download-runtime --download-gpu-accelerator --overwrite
```

The same files ship in the `ai-edge-litert` 2.2.0 wheel (`ai_edge_litert/libLiteRt.so`, Linux) and in the
`com.google.ai.edge.litert:litert:2.2.0` Android AAR (`jni/<abi>/`). `--source-lib` and
`--source-gpu-accelerator-lib` stage local copies. Keep the headers, runtime and accelerator at one version.
