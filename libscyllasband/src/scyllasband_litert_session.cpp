/**
 * @file scyllasband_litert_session.cpp
 * @brief Implementation of LiteRT environment + session management.
 *
 * Thin C ABI bridge around the LiteRT C API. Duration-flow orchestration lives
 * in higher-level backend code.
 */
#include "scyllasband_litert_session.h"

#include "scyllasband_error.h"
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>

#if defined(__APPLE__) || defined(__linux__)
#include <dlfcn.h>
#endif

namespace scyllasband_detail {

#ifdef SCYLLASBAND_WITH_LITERT

namespace {

char* duplicate_c_string(const std::string& value) {
    auto* out = static_cast<char*>(std::malloc(value.size() + 1));
    if (out == nullptr) {
        return nullptr;
    }
    std::memcpy(out, value.c_str(), value.size() + 1);
    return out;
}

void clear_owned_tensor(ScyllasBandOwnedTensor* tensor) {
    if (tensor == nullptr) {
        return;
    }
    std::free(tensor->name);
    std::free(tensor->shape);
    std::free(tensor->data);
    tensor->name = nullptr;
    tensor->shape = nullptr;
    tensor->data = nullptr;
    tensor->rank = 0;
    tensor->byte_length = 0;
}

const char* litert_status_name(LiteRtStatus status) {
    switch (status) {
        case kLiteRtStatusOk: return "ok";
        case kLiteRtStatusErrorInvalidArgument: return "invalid_argument";
        case kLiteRtStatusErrorMemoryAllocationFailure: return "alloc_failure";
        case kLiteRtStatusErrorRuntimeFailure: return "runtime_failure";
        case kLiteRtStatusErrorMissingInputTensor: return "missing_input_tensor";
        case kLiteRtStatusErrorUnsupported: return "unsupported";
        case kLiteRtStatusErrorNotFound: return "not_found";
        case kLiteRtStatusErrorTimeoutExpired: return "timeout";
        case kLiteRtStatusErrorWrongVersion: return "wrong_version";
        case kLiteRtStatusErrorUnknown: return "unknown";
        default: return "<other>";
    }
}

void preload_litert_accelerator_plugins() {
#if defined(SCYLLASBAND_LITERT_LIBRARY_DIR) && (defined(__APPLE__) || defined(__linux__))
    static std::once_flag preload_once;
    std::call_once(preload_once, [] {
        const std::string lib_dir = SCYLLASBAND_LITERT_LIBRARY_DIR;
        if (lib_dir.empty()) {
            return;
        }
#if defined(__APPLE__)
        const char* libraries[] = {
            "libLiteRtMetalAccelerator.dylib",
        };
#else
        const char* libraries[] = {
            "libLiteRtGpuAccelerator.so",
            "libLiteRtWebGpuAccelerator.so",
            "libLiteRtOpenClAccelerator.so",
            "libLiteRtVulkanAccelerator.so",
        };
#endif
        for (const char* library : libraries) {
            const std::string path = lib_dir + "/" + library;
            (void)dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
        }
    });
#endif
}

bool litert_check(LiteRtStatus status, const std::string& context) {
    if (status == kLiteRtStatusOk) return true;
    std::ostringstream oss;
    oss << "LiteRT " << context << " failed: " << litert_status_name(status)
        << " (" << static_cast<int>(status) << ")";
    set_error(oss.str());
    return false;
}

const char* scyllasband_accelerator_name(int32_t accelerator) {
    switch (accelerator) {
        case SCYLLASBAND_LITERT_ACCELERATOR_AUTO: return "auto";
        case SCYLLASBAND_LITERT_ACCELERATOR_CPU: return "cpu";
        case SCYLLASBAND_LITERT_ACCELERATOR_GPU: return "gpu";
        case SCYLLASBAND_LITERT_ACCELERATOR_NPU: return "npu";
        default: return "<invalid>";
    }
}

bool litert_hardware_for_accelerator(
    int32_t accelerator,
    LiteRtHwAcceleratorSet* out_hardware
) {
    if (out_hardware == nullptr) return false;
    switch (accelerator) {
        case SCYLLASBAND_LITERT_ACCELERATOR_AUTO:
            // Prefer GPU for Android auto mode; session creation retries CPU
            // explicitly if GPU compilation is unavailable.
            *out_hardware = kLiteRtHwAcceleratorGpu;
            return true;
        case SCYLLASBAND_LITERT_ACCELERATOR_CPU:
            *out_hardware = kLiteRtHwAcceleratorCpu;
            return true;
        case SCYLLASBAND_LITERT_ACCELERATOR_GPU:
            *out_hardware = kLiteRtHwAcceleratorGpu;
            return true;
        case SCYLLASBAND_LITERT_ACCELERATOR_NPU:
            *out_hardware = kLiteRtHwAcceleratorNpu;
            return true;
        default:
            set_error("unsupported LiteRT accelerator mode: " + std::to_string(accelerator));
            return false;
    }
}

bool create_litert_compilation_options(
    LiteRtHwAcceleratorSet hardware,
    LiteRtOptions* out_options
) {
    if (out_options == nullptr) return false;
    *out_options = nullptr;

    LiteRtOptions options = nullptr;
    if (!litert_check(LiteRtCreateOptions(&options), "LiteRtCreateOptions")) {
        return false;
    }
    if (!litert_check(LiteRtSetOptionsHardwareAccelerators(options, hardware),
                      "LiteRtSetOptionsHardwareAccelerators")) {
        LiteRtDestroyOptions(options);
        return false;
    }
    *out_options = options;
    return true;
}

LiteRtStatus compile_litert_model(
    NativeLiteRtSession& session,
    LiteRtHwAcceleratorSet hardware
) {
    LiteRtOptions options = nullptr;
    if (!create_litert_compilation_options(hardware, &options)) {
        return kLiteRtStatusErrorInvalidArgument;
    }

    LiteRtCompiledModel compiled_model = nullptr;
    const LiteRtStatus status = LiteRtCreateCompiledModel(
        litert_runtime().env, session.model, options, &compiled_model);
    if (status != kLiteRtStatusOk) {
        LiteRtDestroyOptions(options);
        return status;
    }

    session.compilation_options = options;
    session.compiled_model = compiled_model;
    return kLiteRtStatusOk;
}

// Compute the packed byte size of a ranked tensor type from its layout.
size_t packed_byte_size(const LiteRtRankedTensorType& type) {
    size_t elements = 1;
    for (unsigned int d = 0; d < type.layout.rank; ++d) {
        const int32_t dim = type.layout.dimensions[d];
        if (dim <= 0) return 0;  // dynamic - caller must resize first
        elements *= static_cast<size_t>(dim);
    }
    int32_t scyllasband_type = scyllasband_tensor_type_from_litert(type.element_type);
    if (scyllasband_type == 0) return 0;
    return elements * scyllasband_tensor_element_size(scyllasband_type);
}

bool tensor_input_type_is_compatible(int32_t source_type, int32_t expected_type) {
    return source_type == expected_type ||
           (source_type == SCYLLASBAND_TENSOR_INT64 && expected_type == SCYLLASBAND_TENSOR_INT32) ||
           (source_type == SCYLLASBAND_TENSOR_INT32 && expected_type == SCYLLASBAND_TENSOR_INT64) ||
           (source_type == SCYLLASBAND_TENSOR_BOOL && expected_type == SCYLLASBAND_TENSOR_FLOAT32);
}

uint64_t converted_source_byte_length(
    size_t expected_bytes,
    int32_t expected_type,
    int32_t source_type
) {
    const uint64_t expected_element_size = scyllasband_tensor_element_size(expected_type);
    const uint64_t source_element_size = scyllasband_tensor_element_size(source_type);
    if (expected_element_size == 0 || source_element_size == 0) return 0;
    if (static_cast<uint64_t>(expected_bytes) % expected_element_size != 0) return 0;
    const uint64_t elements = static_cast<uint64_t>(expected_bytes) / expected_element_size;
    return elements * source_element_size;
}

bool validate_input_view_for_signature(
    const ScyllasBandTensorView& view,
    size_t expected_bytes,
    int32_t expected_type,
    const std::string& name
) {
    if (!tensor_input_type_is_compatible(view.data_type, expected_type)) {
        std::ostringstream oss;
        oss << "scyllasband_litert_session_run: input '" << name << "' dtype "
            << view.data_type << " does not match signature dtype " << expected_type;
        set_error(oss.str());
        return false;
    }
    if (view.data == nullptr && expected_bytes > 0) {
        set_error("scyllasband_litert_session_run: input '" + name + "' has null data");
        return false;
    }
    const uint64_t expected_source_bytes = converted_source_byte_length(
        expected_bytes, expected_type, view.data_type);
    if (expected_source_bytes == 0) {
        set_error("scyllasband_litert_session_run: input '" + name +
                  "' has unsupported dtype conversion");
        return false;
    }
    if (view.byte_length != expected_source_bytes) {
        std::ostringstream oss;
        oss << "scyllasband_litert_session_run: input '" << name << "' byte_length "
            << view.byte_length << " != expected " << expected_source_bytes;
        if (view.data_type != expected_type) {
            oss << " after dtype conversion to signature dtype " << expected_type;
        }
        set_error(oss.str());
        return false;
    }
    return true;
}

bool copy_input_view_to_host(
    void* host_ptr,
    size_t expected_bytes,
    const ScyllasBandTensorView& view,
    int32_t expected_type,
    const std::string& name
) {
    if (view.data_type == expected_type) {
        std::memcpy(host_ptr, view.data, expected_bytes);
        return true;
    }

    if (view.data_type == SCYLLASBAND_TENSOR_INT64 && expected_type == SCYLLASBAND_TENSOR_INT32) {
        const uint64_t elements = static_cast<uint64_t>(expected_bytes) /
                                  scyllasband_tensor_element_size(SCYLLASBAND_TENSOR_INT32);
        const auto* source = static_cast<const int64_t*>(view.data);
        auto* dest = static_cast<int32_t*>(host_ptr);
        for (uint64_t index = 0; index < elements; ++index) {
            const int64_t value = source[index];
            if (value < std::numeric_limits<int32_t>::min() ||
                value > std::numeric_limits<int32_t>::max()) {
                std::ostringstream oss;
                oss << "scyllasband_litert_session_run: input '" << name
                    << "' value at index " << index << " cannot fit int32";
                set_error(oss.str());
                return false;
            }
            dest[index] = static_cast<int32_t>(value);
        }
        return true;
    }

    if (view.data_type == SCYLLASBAND_TENSOR_INT32 && expected_type == SCYLLASBAND_TENSOR_INT64) {
        const uint64_t elements = static_cast<uint64_t>(expected_bytes) /
                                  scyllasband_tensor_element_size(SCYLLASBAND_TENSOR_INT64);
        const auto* source = static_cast<const int32_t*>(view.data);
        auto* dest = static_cast<int64_t*>(host_ptr);
        for (uint64_t index = 0; index < elements; ++index) {
            dest[index] = static_cast<int64_t>(source[index]);
        }
        return true;
    }

    if (view.data_type == SCYLLASBAND_TENSOR_BOOL && expected_type == SCYLLASBAND_TENSOR_FLOAT32) {
        const uint64_t elements = static_cast<uint64_t>(expected_bytes) /
                                  scyllasband_tensor_element_size(SCYLLASBAND_TENSOR_FLOAT32);
        const auto* source = static_cast<const uint8_t*>(view.data);
        auto* dest = static_cast<float*>(host_ptr);
        for (uint64_t index = 0; index < elements; ++index) {
            dest[index] = source[index] == 0 ? 0.0f : 1.0f;
        }
        return true;
    }

    set_error("scyllasband_litert_session_run: input '" + name +
              "' has unsupported dtype conversion");
    return false;
}

// Cache the per-signature input/output metadata. Called once per signature
// during session creation.
bool populate_signature_metadata(NativeLiteRtSignature& sig) {
    LiteRtParamIndex num_inputs = 0;
    if (!litert_check(LiteRtGetNumSignatureInputs(sig.handle, &num_inputs),
                      "LiteRtGetNumSignatureInputs")) return false;
    sig.input_names.reserve(num_inputs);
    sig.input_types.reserve(num_inputs);
    for (LiteRtParamIndex i = 0; i < num_inputs; ++i) {
        const char* name = nullptr;
        if (!litert_check(LiteRtGetSignatureInputName(sig.handle, i, &name),
                          "LiteRtGetSignatureInputName")) return false;
        LiteRtTensor tensor = nullptr;
        if (!litert_check(LiteRtGetSignatureInputTensorByIndex(sig.handle, i, &tensor),
                          "LiteRtGetSignatureInputTensorByIndex")) return false;
        LiteRtRankedTensorType type{};
        if (!litert_check(LiteRtGetRankedTensorType(tensor, &type),
                          "LiteRtGetRankedTensorType (input)")) return false;
        sig.input_names.emplace_back(name);
        sig.input_types.emplace_back(type);
        sig.input_index_by_name[name] = static_cast<size_t>(i);
    }

    LiteRtParamIndex num_outputs = 0;
    if (!litert_check(LiteRtGetNumSignatureOutputs(sig.handle, &num_outputs),
                      "LiteRtGetNumSignatureOutputs")) return false;
    sig.output_names.reserve(num_outputs);
    sig.output_types.reserve(num_outputs);
    for (LiteRtParamIndex i = 0; i < num_outputs; ++i) {
        const char* name = nullptr;
        if (!litert_check(LiteRtGetSignatureOutputName(sig.handle, i, &name),
                          "LiteRtGetSignatureOutputName")) return false;
        LiteRtTensor tensor = nullptr;
        if (!litert_check(LiteRtGetSignatureOutputTensorByIndex(sig.handle, i, &tensor),
                          "LiteRtGetSignatureOutputTensorByIndex")) return false;
        LiteRtRankedTensorType type{};
        if (!litert_check(LiteRtGetRankedTensorType(tensor, &type),
                          "LiteRtGetRankedTensorType (output)")) return false;
        sig.output_names.emplace_back(name);
        sig.output_types.emplace_back(type);
        sig.output_index_by_name[name] = static_cast<size_t>(i);
    }
    return true;
}

void destroy_signature_buffers(NativeLiteRtSignature& sig) {
    for (auto* buf : sig.input_buffers) {
        if (buf != nullptr) {
            LiteRtDestroyTensorBuffer(buf);
        }
    }
    for (auto* buf : sig.output_buffers) {
        if (buf != nullptr) {
            LiteRtDestroyTensorBuffer(buf);
        }
    }
    sig.input_buffers.clear();
    sig.output_buffers.clear();
}

bool ensure_signature_buffers(NativeLiteRtRuntime& runtime, NativeLiteRtSignature& sig) {
    if (!sig.input_buffers.empty() || !sig.output_buffers.empty()) {
        return true;
    }

    sig.input_buffers.assign(sig.input_names.size(), nullptr);
    sig.output_buffers.assign(sig.output_names.size(), nullptr);

    for (size_t i = 0; i < sig.input_names.size(); ++i) {
        const auto& type = sig.input_types[i];
        const size_t expected_bytes = packed_byte_size(type);
        if (expected_bytes == 0) {
            destroy_signature_buffers(sig);
            set_error("scyllasband_litert_session_run: dynamic shape on input '" +
                      sig.input_names[i] + "' is not supported");
            return false;
        }
        if (!litert_check(LiteRtCreateManagedTensorBuffer(
                runtime.env, kLiteRtTensorBufferTypeHostMemory, &type, expected_bytes,
                &sig.input_buffers[i]),
                "LiteRtCreateManagedTensorBuffer (input " + sig.input_names[i] + ")")) {
            destroy_signature_buffers(sig);
            return false;
        }
    }

    for (size_t i = 0; i < sig.output_names.size(); ++i) {
        const auto& type = sig.output_types[i];
        const size_t expected_bytes = packed_byte_size(type);
        if (expected_bytes == 0) {
            destroy_signature_buffers(sig);
            set_error("scyllasband_litert_session_run: dynamic shape on output '" +
                      sig.output_names[i] + "' is not supported");
            return false;
        }
        if (!litert_check(LiteRtCreateManagedTensorBuffer(
                runtime.env, kLiteRtTensorBufferTypeHostMemory, &type, expected_bytes,
                &sig.output_buffers[i]),
                "LiteRtCreateManagedTensorBuffer (output " + sig.output_names[i] + ")")) {
            destroy_signature_buffers(sig);
            return false;
        }
    }

    return true;
}


bool resize_signature_to_input_views(
    NativeLiteRtSession& session,
    NativeLiteRtSignature& sig,
    const std::unordered_map<std::string, const ScyllasBandTensorView*>& input_by_name
) {
    destroy_signature_buffers(sig);
    bool resized_any = false;
    for (size_t i = 0; i < sig.input_names.size(); ++i) {
        const std::string& name = sig.input_names[i];
        auto it = input_by_name.find(name);
        if (it == input_by_name.end()) {
            set_error("scyllasband_litert_session_run_resized: missing input tensor '" + name + "'");
            return false;
        }
        const ScyllasBandTensorView* view = it->second;
        const LiteRtLayout& current_layout = sig.input_types[i].layout;
        if (view->rank != static_cast<int32_t>(current_layout.rank)) {
            std::ostringstream oss;
            oss << "scyllasband_litert_session_run_resized: input '" << name << "' rank "
                << view->rank << " != signature rank " << current_layout.rank;
            set_error(oss.str());
            return false;
        }
        if (view->rank > 0 && view->shape == nullptr) {
            set_error("scyllasband_litert_session_run_resized: input '" + name + "' has null shape");
            return false;
        }
        std::vector<int> dims(static_cast<size_t>(view->rank));
        bool differs = false;
        for (int32_t dim_index = 0; dim_index < view->rank; ++dim_index) {
            const int64_t dim = view->shape[dim_index];
            if (dim <= 0 || dim > static_cast<int64_t>(std::numeric_limits<int>::max())) {
                std::ostringstream oss;
                oss << "scyllasband_litert_session_run_resized: input '" << name
                    << "' has invalid resize dim " << dim;
                set_error(oss.str());
                return false;
            }
            dims[static_cast<size_t>(dim_index)] = static_cast<int>(dim);
            if (dim != static_cast<int64_t>(current_layout.dimensions[dim_index])) {
                differs = true;
            }
        }
        if (!differs) {
            continue;
        }
        resized_any = true;
        if (!litert_check(LiteRtCompiledModelResizeInputTensorNonStrict(
                session.compiled_model,
                sig.index,
                static_cast<LiteRtParamIndex>(i),
                dims.data(),
                dims.size()),
                "LiteRtCompiledModelResizeInputTensorNonStrict (input " + name + ")")) {
            return false;
        }
    }

    if (resized_any) {
        for (size_t i = 0; i < sig.input_names.size(); ++i) {
            LiteRtLayout layout{};
            if (!litert_check(LiteRtGetCompiledModelInputTensorLayout(
                    session.compiled_model,
                    sig.index,
                    static_cast<LiteRtParamIndex>(i),
                    &layout),
                    "LiteRtGetCompiledModelInputTensorLayout (input " + sig.input_names[i] + ")")) {
                return false;
            }
            sig.input_types[i].layout = layout;
        }
        if (!sig.output_types.empty()) {
            std::vector<LiteRtLayout> layouts(sig.output_types.size());
            if (!litert_check(LiteRtGetCompiledModelOutputTensorLayouts(
                    session.compiled_model,
                    sig.index,
                    layouts.size(),
                    layouts.data(),
                    true),
                    "LiteRtGetCompiledModelOutputTensorLayouts")) {
                return false;
            }
            for (size_t i = 0; i < sig.output_types.size(); ++i) {
                sig.output_types[i].layout = layouts[i];
            }
        }
    }
    return ensure_signature_buffers(litert_runtime(), sig);
}

}  // namespace

NativeLiteRtRuntime& litert_runtime() {
    static NativeLiteRtRuntime instance;
    return instance;
}

bool ensure_litert_runtime() {
    auto& rt = litert_runtime();
    std::call_once(rt.init_once, [&] {
        // Environment options are defaulted here; accelerator selection is
        // applied when compiling each model.
        if (LiteRtCreateEnvironment(0, nullptr, &rt.env) != kLiteRtStatusOk) {
            rt.init_error = "LiteRtCreateEnvironment failed";
            rt.env = nullptr;
        }
    });
    if (rt.env == nullptr) {
        set_error(rt.init_error.empty() ? "LiteRT environment unavailable" : rt.init_error);
        return false;
    }
    return true;
}

int32_t scyllasband_tensor_type_from_litert(LiteRtElementType type) {
    switch (type) {
        case kLiteRtElementTypeFloat32: return SCYLLASBAND_TENSOR_FLOAT32;
        case kLiteRtElementTypeFloat16: return SCYLLASBAND_TENSOR_FLOAT16;
        case kLiteRtElementTypeFloat64: return SCYLLASBAND_TENSOR_FLOAT64;
        case kLiteRtElementTypeInt64:   return SCYLLASBAND_TENSOR_INT64;
        case kLiteRtElementTypeInt32:   return SCYLLASBAND_TENSOR_INT32;
        case kLiteRtElementTypeInt16:   return SCYLLASBAND_TENSOR_INT16;
        case kLiteRtElementTypeInt8:    return SCYLLASBAND_TENSOR_INT8;
        case kLiteRtElementTypeUInt8:   return SCYLLASBAND_TENSOR_UINT8;
        case kLiteRtElementTypeBool:    return SCYLLASBAND_TENSOR_BOOL;
        default: return 0;
    }
}

LiteRtElementType litert_element_type_from_scyllasband(int32_t type) {
    switch (type) {
        case SCYLLASBAND_TENSOR_FLOAT32: return kLiteRtElementTypeFloat32;
        case SCYLLASBAND_TENSOR_FLOAT16: return kLiteRtElementTypeFloat16;
        case SCYLLASBAND_TENSOR_FLOAT64: return kLiteRtElementTypeFloat64;
        case SCYLLASBAND_TENSOR_INT64:   return kLiteRtElementTypeInt64;
        case SCYLLASBAND_TENSOR_INT32:   return kLiteRtElementTypeInt32;
        case SCYLLASBAND_TENSOR_INT16:   return kLiteRtElementTypeInt16;
        case SCYLLASBAND_TENSOR_INT8:    return kLiteRtElementTypeInt8;
        case SCYLLASBAND_TENSOR_UINT8:   return kLiteRtElementTypeUInt8;
        case SCYLLASBAND_TENSOR_BOOL:    return kLiteRtElementTypeBool;
        default: return kLiteRtElementTypeNone;
    }
}

#endif  // SCYLLASBAND_WITH_LITERT

}  // namespace scyllasband_detail


// ===========================================================================
// C API
// ===========================================================================

#ifdef SCYLLASBAND_WITH_LITERT

extern "C" {

ScyllasBandLiteRtSession* scyllasband_litert_session_create(
    const char* model_path,
    int32_t accelerator,
    int32_t max_threads
) {
    using namespace scyllasband_detail;
    if (model_path == nullptr || model_path[0] == '\0') {
        set_error("scyllasband_litert_session_create: model_path is required");
        return nullptr;
    }
    preload_litert_accelerator_plugins();
    if (!ensure_litert_runtime()) return nullptr;

    auto session = std::make_unique<NativeLiteRtSession>();
    session->model_path = model_path;

    if (!litert_check(LiteRtCreateModelFromFile(model_path, &session->model),
                      "LiteRtCreateModelFromFile")) {
        return nullptr;
    }
    LiteRtHwAcceleratorSet hardware = kLiteRtHwAcceleratorNone;
    if (!litert_hardware_for_accelerator(accelerator, &hardware)) {
        LiteRtDestroyModel(session->model);
        return nullptr;
    }

    // LiteRT 2.1.4's Android AAR exposes CPU option headers but does not
    // export the CPU option symbols, so thread count cannot be applied here.
    (void)max_threads;
    LiteRtStatus compile_status = compile_litert_model(*session, hardware);
    if (compile_status != kLiteRtStatusOk &&
        accelerator == SCYLLASBAND_LITERT_ACCELERATOR_AUTO) {
        compile_status = compile_litert_model(*session, kLiteRtHwAcceleratorCpu);
        if (compile_status == kLiteRtStatusOk) {
            set_error("");
        }
    }
    if (compile_status != kLiteRtStatusOk) {
        std::ostringstream oss;
        oss << "LiteRT LiteRtCreateCompiledModel failed for accelerator "
            << scyllasband_accelerator_name(accelerator);
        if (accelerator == SCYLLASBAND_LITERT_ACCELERATOR_AUTO) {
            oss << " (GPU, then CPU fallback)";
        }
        oss << ": " << litert_status_name(compile_status)
            << " (" << static_cast<int>(compile_status) << ")";
        set_error(oss.str());
        LiteRtDestroyModel(session->model);
        return nullptr;
    }

    LiteRtParamIndex num_sigs = 0;
    if (!litert_check(LiteRtGetNumModelSignatures(session->model, &num_sigs),
                      "LiteRtGetNumModelSignatures")) {
        LiteRtDestroyCompiledModel(session->compiled_model);
        LiteRtDestroyOptions(session->compilation_options);
        LiteRtDestroyModel(session->model);
        return nullptr;
    }

    session->signatures.resize(num_sigs);
    for (LiteRtParamIndex i = 0; i < num_sigs; ++i) {
        auto& sig = session->signatures[i];
        sig.index = i;
        if (!litert_check(LiteRtGetModelSignature(session->model, i, &sig.handle),
                          "LiteRtGetModelSignature")) {
            return nullptr;  // unique_ptr cleans up
        }
        const char* key = nullptr;
        if (!litert_check(LiteRtGetSignatureKey(sig.handle, &key),
                          "LiteRtGetSignatureKey")) {
            return nullptr;
        }
        sig.key = key;
        if (!populate_signature_metadata(sig)) {
            return nullptr;
        }
        session->signature_index_by_name[sig.key] = static_cast<size_t>(i);
    }

    return reinterpret_cast<ScyllasBandLiteRtSession*>(session.release());
}

void scyllasband_litert_session_destroy(ScyllasBandLiteRtSession* session_handle) {
    using namespace scyllasband_detail;
    if (session_handle == nullptr) return;
    auto* session = reinterpret_cast<NativeLiteRtSession*>(session_handle);
    for (auto& sig : session->signatures) {
        destroy_signature_buffers(sig);
    }
    if (session->compiled_model != nullptr) {
        LiteRtDestroyCompiledModel(session->compiled_model);
    }
    if (session->compilation_options != nullptr) {
        LiteRtDestroyOptions(session->compilation_options);
    }
    if (session->model != nullptr) {
        LiteRtDestroyModel(session->model);
    }
    delete session;
}

int scyllasband_litert_session_has_signature(
    const ScyllasBandLiteRtSession* session_handle,
    const char* signature_name
) {
    using namespace scyllasband_detail;
    if (session_handle == nullptr || signature_name == nullptr) return 0;
    auto* session = reinterpret_cast<const NativeLiteRtSession*>(session_handle);
    return session->signature_index_by_name.count(signature_name) ? 1 : 0;
}

static int scyllasband_litert_session_run_impl(
    const ScyllasBandLiteRtSession* session_handle,
    const char* signature_name,
    const ScyllasBandTensorView* inputs,
    int32_t input_count,
    ScyllasBandOwnedTensor** out_tensors,
    int32_t* out_tensor_count,
    bool resize_to_inputs
) {
    using namespace scyllasband_detail;
    if (out_tensors == nullptr || out_tensor_count == nullptr) {
        set_error("scyllasband_litert_session_run: output pointers are required");
        return -1;
    }
    *out_tensors = nullptr;
    *out_tensor_count = 0;

    if (session_handle == nullptr || signature_name == nullptr) {
        set_error("scyllasband_litert_session_run: session and signature_name are required");
        return -1;
    }
    if (input_count > 0 && inputs == nullptr) {
        set_error("scyllasband_litert_session_run: input_count > 0 but inputs == nullptr");
        return -1;
    }
    auto* session = const_cast<NativeLiteRtSession*>(
        reinterpret_cast<const NativeLiteRtSession*>(session_handle));
    auto sig_it = session->signature_index_by_name.find(signature_name);
    if (sig_it == session->signature_index_by_name.end()) {
        set_error(std::string("scyllasband_litert_session_run: unknown signature '") +
                  signature_name + "'");
        return -1;
    }
    auto& sig = session->signatures[sig_it->second];

    if (static_cast<size_t>(input_count) != sig.input_names.size()) {
        std::ostringstream oss;
        oss << "scyllasband_litert_session_run: signature '" << signature_name
            << "' expects " << sig.input_names.size() << " inputs, got " << input_count;
        set_error(oss.str());
        return -1;
    }

    // Index inputs by name so we can fill the tensor-buffer array in the
    // signature's declared order regardless of caller order.
    std::unordered_map<std::string, const ScyllasBandTensorView*> input_by_name;
    input_by_name.reserve(input_count);
    for (int32_t i = 0; i < input_count; ++i) {
        if (inputs[i].name == nullptr) {
            set_error("scyllasband_litert_session_run: input has null name");
            return -1;
        }
        input_by_name[inputs[i].name] = &inputs[i];
    }

    std::lock_guard<std::mutex> run_lock(session->run_mutex);
    auto& runtime = litert_runtime();
    if (resize_to_inputs) {
        if (!resize_signature_to_input_views(*session, sig, input_by_name)) {
            return -1;
        }
    } else if (!ensure_signature_buffers(runtime, sig)) {
        return -1;
    }

    // ---- inputs ----
    for (size_t i = 0; i < sig.input_names.size(); ++i) {
        const std::string& name = sig.input_names[i];
        auto it = input_by_name.find(name);
        if (it == input_by_name.end()) {
            set_error("scyllasband_litert_session_run: missing input tensor '" + name + "'");
            return -1;
        }
        const ScyllasBandTensorView* view = it->second;
        const LiteRtRankedTensorType& type = sig.input_types[i];
        const int32_t expected_scyllasband_type = scyllasband_tensor_type_from_litert(type.element_type);
        if (expected_scyllasband_type == 0) {
            set_error("scyllasband_litert_session_run: unsupported dtype on input '" + name + "'");
            return -1;
        }
        const size_t expected_bytes = packed_byte_size(type);
        if (expected_bytes == 0) {
            set_error("scyllasband_litert_session_run: dynamic shape on input '" + name +
                      "' is not supported");
            return -1;
        }
        if (!validate_input_view_for_signature(*view, expected_bytes, expected_scyllasband_type, name)) {
            return -1;
        }
        void* host_ptr = nullptr;
        if (!litert_check(LiteRtLockTensorBuffer(sig.input_buffers[i], &host_ptr,
                                                 kLiteRtTensorBufferLockModeWrite),
                          "LiteRtLockTensorBuffer (input " + name + ")")) {
            return -1;
        }
        if (!copy_input_view_to_host(host_ptr, expected_bytes, *view, expected_scyllasband_type, name)) {
            LiteRtUnlockTensorBuffer(sig.input_buffers[i]);
            return -1;
        }
        if (!litert_check(LiteRtUnlockTensorBuffer(sig.input_buffers[i]),
                          "LiteRtUnlockTensorBuffer (input " + name + ")")) {
            return -1;
        }
    }

    // ---- run ----
    if (!litert_check(LiteRtRunCompiledModel(session->compiled_model, sig.index,
                                             sig.input_buffers.size(), sig.input_buffers.data(),
                                             sig.output_buffers.size(), sig.output_buffers.data()),
                      "LiteRtRunCompiledModel")) {
        return -1;
    }

    // ---- extract outputs into ScyllasBandOwnedTensor[] ----
    auto* owned = static_cast<ScyllasBandOwnedTensor*>(
        std::calloc(sig.output_names.size(), sizeof(ScyllasBandOwnedTensor)));
    if (owned == nullptr) {
        set_error("scyllasband_litert_session_run: malloc failed for output array");
        return -1;
    }
    for (size_t i = 0; i < sig.output_names.size(); ++i) {
        const std::string& name = sig.output_names[i];
        const LiteRtRankedTensorType& type = sig.output_types[i];
        const size_t bytes = packed_byte_size(type);

        owned[i].name = duplicate_c_string(name);
        owned[i].data_type = scyllasband_tensor_type_from_litert(type.element_type);
        owned[i].rank = static_cast<int32_t>(type.layout.rank);
        owned[i].byte_length = static_cast<uint64_t>(bytes);

        owned[i].shape = static_cast<int64_t*>(std::calloc(type.layout.rank, sizeof(int64_t)));
        for (unsigned int d = 0; d < type.layout.rank; ++d) {
            owned[i].shape[d] = static_cast<int64_t>(type.layout.dimensions[d]);
        }
        owned[i].data = std::malloc(bytes);
        if (owned[i].data == nullptr) {
            for (size_t j = 0; j <= i; ++j) clear_owned_tensor(&owned[j]);
            std::free(owned);
            set_error("scyllasband_litert_session_run: malloc failed for output data");
            return -1;
        }
        void* host_ptr = nullptr;
        if (!litert_check(LiteRtLockTensorBuffer(sig.output_buffers[i], &host_ptr,
                                                 kLiteRtTensorBufferLockModeRead),
                          "LiteRtLockTensorBuffer (output " + name + ")")) {
            for (size_t j = 0; j <= i; ++j) clear_owned_tensor(&owned[j]);
            std::free(owned);
            return -1;
        }
        std::memcpy(owned[i].data, host_ptr, bytes);
        LiteRtUnlockTensorBuffer(sig.output_buffers[i]);
    }

    *out_tensors = owned;
    *out_tensor_count = static_cast<int32_t>(sig.output_names.size());
    return 0;
}

int scyllasband_litert_session_run(
    const ScyllasBandLiteRtSession* session,
    const char* signature_name,
    const ScyllasBandTensorView* inputs,
    int32_t input_count,
    ScyllasBandOwnedTensor** out_tensors,
    int32_t* out_tensor_count
) {
    return scyllasband_litert_session_run_impl(
        session, signature_name, inputs, input_count, out_tensors, out_tensor_count, false);
}

int scyllasband_litert_session_run_resized(
    const ScyllasBandLiteRtSession* session,
    const char* signature_name,
    const ScyllasBandTensorView* inputs,
    int32_t input_count,
    ScyllasBandOwnedTensor** out_tensors,
    int32_t* out_tensor_count
) {
    return scyllasband_litert_session_run_impl(
        session, signature_name, inputs, input_count, out_tensors, out_tensor_count, true);
}

}  // extern "C"

#elif !defined(SCYLLASBAND_WITH_ONNXRUNTIME)  // no graph runtime enabled

extern "C" {

ScyllasBandLiteRtSession* scyllasband_litert_session_create(const char*, int32_t, int32_t) {
    scyllasband_detail::set_error("LiteRT support is not enabled in this build");
    return nullptr;
}
void scyllasband_litert_session_destroy(ScyllasBandLiteRtSession*) {}
int scyllasband_litert_session_has_signature(const ScyllasBandLiteRtSession*, const char*) { return 0; }
int scyllasband_litert_session_run(const ScyllasBandLiteRtSession*, const char*,
                                const ScyllasBandTensorView*, int32_t,
                                ScyllasBandOwnedTensor**, int32_t*) {
    scyllasband_detail::set_error("LiteRT support is not enabled in this build");
    return -1;
}
int scyllasband_litert_session_run_resized(const ScyllasBandLiteRtSession*, const char*,
                                        const ScyllasBandTensorView*, int32_t,
                                        ScyllasBandOwnedTensor**, int32_t*) {
    scyllasband_detail::set_error("LiteRT support is not enabled in this build");
    return -1;
}

}  // extern "C"

#endif  // graph runtime selection
