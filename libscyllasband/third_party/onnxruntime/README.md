# Staged ONNX Runtime SDK

ONNX Runtime 1.30.0 C/C++ headers from the official `onnxruntime-linux-x64-1.30.0` release archive.

Shared libraries are local build inputs under `lib/<platform>/` and are not committed:

```bash
curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.30.0/onnxruntime-linux-x64-1.30.0.tgz | tar xz
mkdir -p lib/linux-x86_64
cp -a onnxruntime-linux-x64-1.30.0/lib/libonnxruntime.so* lib/linux-x86_64/
```

Android builds link `jni/<abi>/libonnxruntime.so` from the `com.microsoft.onnxruntime:onnxruntime-android`
AAR, and iOS builds use the `onnxruntime-c` pod, both at a matching version.
