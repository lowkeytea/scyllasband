/**
 * @file scyllasband_litert_session.h
 * @brief LiteRT session management for the modern LiteRt C API
 *        (`libLiteRt.so`).
 *
 * Owns the process-wide LiteRT environment + per-`.tflite` compiled model.
 * Higher layers interact with LiteRT exclusively through this module.
 *
 * Scope: open a multi-signature `.tflite`, list signatures, run any
 * signature by name with a flat list of (name -> host-memory) inputs, and
 * receive a flat list of host-memory outputs. Duration-flow orchestration lives
 * above this layer.
 */
#pragma once

#include "scyllasband.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef SCYLLASBAND_WITH_LITERT
#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#endif

namespace scyllasband_detail {

#ifdef SCYLLASBAND_WITH_LITERT

/// Lazy-initialized singleton for the LiteRT environment: one env per process,
/// created on first use.
struct NativeLiteRtRuntime {
    LiteRtEnvironment env{nullptr};
    std::once_flag init_once;
    std::string init_error;
};

/// Per-signature cached metadata (input/output names in declared order).
struct NativeLiteRtSignature {
    std::string key;
    LiteRtSignature handle{nullptr};
    LiteRtParamIndex index{0};
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<LiteRtRankedTensorType> input_types;
    std::vector<LiteRtRankedTensorType> output_types;
    std::vector<LiteRtTensorBuffer> input_buffers;
    std::vector<LiteRtTensorBuffer> output_buffers;
    // input_index[name] -> position in input_names / input_types
    std::unordered_map<std::string, size_t> input_index_by_name;
    std::unordered_map<std::string, size_t> output_index_by_name;
};

/// Wraps one `.tflite` file loaded as a LiteRtModel + LiteRtCompiledModel,
/// plus the cached signature metadata for fast name lookups.
struct NativeLiteRtSession {
    std::string model_path;
    LiteRtModel model{nullptr};
    LiteRtCompiledModel compiled_model{nullptr};
    LiteRtOptions compilation_options{nullptr};
    std::vector<NativeLiteRtSignature> signatures;
    std::unordered_map<std::string, size_t> signature_index_by_name;
    mutable std::mutex run_mutex;
};

/// Return a reference to the process-wide LiteRT runtime singleton.
NativeLiteRtRuntime& litert_runtime();

/// Ensure the LiteRT environment is initialised. Safe to call from any thread.
/// @return false if initialisation failed (error recorded via set_error).
bool ensure_litert_runtime();

/// Convert a LiteRtElementType to the matching ScyllasBandTensorType. Returns 0
/// (unsupported) if no mapping exists.
int32_t scyllasband_tensor_type_from_litert(LiteRtElementType type);

/// Convert a ScyllasBandTensorType to the matching LiteRtElementType. Returns
/// kLiteRtElementTypeNone if no mapping exists.
LiteRtElementType litert_element_type_from_scyllasband(int32_t type);

#endif  // SCYLLASBAND_WITH_LITERT

}  // namespace scyllasband_detail
