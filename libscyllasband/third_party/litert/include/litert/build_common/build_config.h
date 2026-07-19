// Hand-written build_config.h for the staged LiteRT 2.1.2 SDK.
// Upstream generates this from build_config.h.in via CMake. We're consuming
// the prebuilt libLiteRt.so from the ai_edge_litert wheel, which is built
// with GPU + NPU enabled (see GPU symbols in nm output), so neither feature
// macro is defined here.

#ifndef LITERT_BUILD_COMMON_BUILD_CONFIG_H_
#define LITERT_BUILD_COMMON_BUILD_CONFIG_H_

#endif  // LITERT_BUILD_COMMON_BUILD_CONFIG_H_
