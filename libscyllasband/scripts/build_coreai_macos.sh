#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runtime_dir="$(cd "${script_dir}/.." && pwd)"
build_dir="${SCYLLASBAND_COREAI_BUILD_DIR:-${runtime_dir}/build/coreai-macos}"
deployment_target="${MACOSX_DEPLOYMENT_TARGET:-27.0}"
target_arch="${SCYLLASBAND_COREAI_ARCH:-$(uname -m)}"
swift_target="${target_arch}-apple-macosx${deployment_target}"
module_cache="${build_dir}/module-cache"

# Core AI ships with the macOS 27 SDK. Older SDKs compile the C++ sources
# fine and then fail on `import CoreAI` with an unhelpful missing-module
# error, so check up front and name the fix.
sdk_path="$(xcrun --sdk macosx --show-sdk-path)"
if [[ ! -e "${sdk_path}/System/Library/Frameworks/CoreAI.framework" ]]; then
  echo "error: the selected macOS SDK does not include CoreAI.framework:" >&2
  echo "error:   ${sdk_path}" >&2
  echo "error: select an Xcode with the macOS 27 SDK, e.g.:" >&2
  echo "error:   DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer ${BASH_SOURCE[0]}" >&2
  exit 1
fi

mkdir -p "${build_dir}/objects" "${module_cache}"

sources=(
  src/scyllasband.cpp
  src/scyllasband_backend.cpp
  src/scyllasband_bundle.cpp
  src/scyllasband_error.cpp
  src/scyllasband_execution_plan.cpp
  src/scyllasband_graph_session_dispatch.cpp
  src/scyllasband_request.cpp
  src/scyllasband_tensor.cpp
  src/scyllasband_text_normalizer.cpp
)
objects=()
for source in "${sources[@]}"; do
  object="${build_dir}/objects/$(basename "${source}" .cpp).o"
  xcrun --sdk macosx clang++ \
    -std=c++17 \
    -arch "${target_arch}" \
    -mmacosx-version-min="${deployment_target}" \
    -DSCYLLASBAND_WITH_COREAI=1 \
    -I "${runtime_dir}/include" \
    -I "${runtime_dir}/src" \
    -fmodules-cache-path="${module_cache}" \
    -fPIC \
    -c "${runtime_dir}/${source}" \
    -o "${object}"
  objects+=("${object}")
done

xcrun --sdk macosx swiftc \
  -parse-as-library \
  -target "${swift_target}" \
  -module-cache-path "${module_cache}" \
  -emit-library \
  -module-name ScyllasBandCoreAI \
  "${runtime_dir}/apple/Sources/CoreAISession.swift" \
  "${objects[@]}" \
  -framework CoreAI \
  -framework Foundation \
  -lc++ \
  -Xlinker -install_name \
  -Xlinker @rpath/libscyllasband_native.dylib \
  -o "${build_dir}/libscyllasband_native.dylib"

xcrun --sdk macosx clang++ \
  -std=c++17 \
  -arch "${target_arch}" \
  -mmacosx-version-min="${deployment_target}" \
  -I "${runtime_dir}/include" \
  -fmodules-cache-path="${module_cache}" \
  -c "${runtime_dir}/tools/scyllasband_native_speak.cpp" \
  -o "${build_dir}/objects/scyllasband_native_speak.o"

xcrun --sdk macosx swiftc \
  -parse-as-library \
  -target "${swift_target}" \
  -module-cache-path "${module_cache}" \
  "${runtime_dir}/apple/Sources/CoreAISession.swift" \
  "${objects[@]}" \
  "${build_dir}/objects/scyllasband_native_speak.o" \
  -framework CoreAI \
  -framework Foundation \
  -lc++ \
  -o "${build_dir}/scyllasband_coreai_speak"

printf '%s\n' \
  "Built ${build_dir}/libscyllasband_native.dylib" \
  "Built ${build_dir}/scyllasband_coreai_speak"
