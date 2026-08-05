/**
 * @file scyllasband_onnx_session.cpp
 * @brief ONNX Runtime implementation of the graph-session ABI used by the
 *        duration-flow backend.
 *
 * The orchestration layer historically called this ABI the "LiteRT session"
 * ABI.  Keeping the small C surface stable lets both runtimes share the same
 * G2P, duration, flow-sampling, reference-pack, and vocoder implementation.
 */

#include "scyllasband.h"
#include "scyllasband_error.h"

#ifdef SCYLLASBAND_WITH_ONNXRUNTIME

#ifdef SCYLLASBAND_WITH_COREAI
/*
 * Dual-backend builds route the public graph-session ABI through
 * scyllasband_graph_session_dispatch.cpp, which picks ONNX or Core AI per
 * model artifact. Rename this translation unit's exports so both
 * implementations can be linked into one binary.
 */
#define scyllasband_litert_session_create scyllasband_onnx_session_create
#define scyllasband_litert_session_destroy scyllasband_onnx_session_destroy
#define scyllasband_litert_session_has_signature scyllasband_onnx_session_has_signature
#define scyllasband_litert_session_run scyllasband_onnx_session_run
#define scyllasband_litert_session_run_resized scyllasband_onnx_session_run_resized
#endif

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace scyllasband_detail {
namespace {

struct NativeOnnxSession {
    NativeOnnxSession(Ort::Env& environment, const char* model_path, Ort::SessionOptions& options)
        : session(environment, model_path, options) {}

    Ort::Session session;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<ONNXTensorElementDataType> input_types;
    std::vector<std::vector<int64_t>> input_shapes;
    mutable std::mutex run_mutex;
};

struct PreparedOnnxInput {
    const void* data = nullptr;
    std::size_t byte_length = 0;
    std::vector<uint8_t> converted;
};

Ort::Env& onnx_environment() {
    static Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "scyllasband");
    return environment;
}

char* duplicate_c_string(const std::string& value) {
    auto* output = static_cast<char*>(std::malloc(value.size() + 1));
    if (output != nullptr) {
        std::memcpy(output, value.c_str(), value.size() + 1);
    }
    return output;
}

void clear_owned_tensor(ScyllasBandOwnedTensor* tensor) {
    if (tensor == nullptr) {
        return;
    }
    std::free(tensor->name);
    std::free(tensor->shape);
    std::free(tensor->data);
    *tensor = {};
}

int32_t scyllasband_type_from_onnx(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return SCYLLASBAND_TENSOR_FLOAT32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return SCYLLASBAND_TENSOR_FLOAT16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return SCYLLASBAND_TENSOR_FLOAT64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return SCYLLASBAND_TENSOR_INT64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return SCYLLASBAND_TENSOR_INT32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: return SCYLLASBAND_TENSOR_INT16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return SCYLLASBAND_TENSOR_INT8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return SCYLLASBAND_TENSOR_UINT8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return SCYLLASBAND_TENSOR_BOOL;
        default: return 0;
    }
}

std::size_t checked_element_count(const int64_t* shape, int32_t rank) {
    if (rank < 0 || (rank > 0 && shape == nullptr)) {
        throw std::runtime_error("tensor has an invalid shape");
    }
    std::size_t count = 1;
    for (int32_t index = 0; index < rank; ++index) {
        if (shape[index] <= 0) {
            throw std::runtime_error("tensor shape contains a non-positive dimension");
        }
        const auto dimension = static_cast<std::size_t>(shape[index]);
        if (count > std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::runtime_error("tensor element count overflows size_t");
        }
        count *= dimension;
    }
    return count;
}

void validate_shape(
    const ScyllasBandTensorView& view,
    const std::vector<int64_t>& expected,
    const std::string& name
) {
    if (view.rank != static_cast<int32_t>(expected.size())) {
        throw std::runtime_error(
            "ONNX input '" + name + "' expected rank " +
            std::to_string(expected.size()) + ", got " + std::to_string(view.rank)
        );
    }
    for (int32_t index = 0; index < view.rank; ++index) {
        if (expected[static_cast<std::size_t>(index)] > 0 &&
            view.shape[index] != expected[static_cast<std::size_t>(index)]) {
            throw std::runtime_error(
                "ONNX input '" + name + "' has an incompatible static dimension"
            );
        }
    }
}

template <typename Source, typename Destination>
void convert_values(
    const ScyllasBandTensorView& view,
    PreparedOnnxInput& output,
    std::size_t count
) {
    output.converted.resize(count * sizeof(Destination));
    const auto* source = static_cast<const Source*>(view.data);
    auto* destination = reinterpret_cast<Destination*>(output.converted.data());
    for (std::size_t index = 0; index < count; ++index) {
        destination[index] = static_cast<Destination>(source[index]);
    }
    output.data = output.converted.data();
    output.byte_length = output.converted.size();
}

PreparedOnnxInput prepare_input(
    const ScyllasBandTensorView& view,
    ONNXTensorElementDataType expected_onnx_type,
    const std::string& name
) {
    if (view.data == nullptr) {
        throw std::runtime_error("ONNX input '" + name + "' has null data");
    }
    const int32_t expected_type = scyllasband_type_from_onnx(expected_onnx_type);
    if (expected_type == 0) {
        throw std::runtime_error("ONNX input '" + name + "' uses an unsupported dtype");
    }
    const std::size_t count = checked_element_count(view.shape, view.rank);
    const uint64_t source_element_size = scyllasband_tensor_element_size(view.data_type);
    const uint64_t expected_source_bytes = source_element_size * count;
    if (source_element_size == 0 || view.byte_length != expected_source_bytes) {
        throw std::runtime_error("ONNX input '" + name + "' has an invalid byte length");
    }

    PreparedOnnxInput output;
    if (view.data_type == expected_type) {
        output.data = view.data;
        output.byte_length = static_cast<std::size_t>(view.byte_length);
        return output;
    }
    if (view.data_type == SCYLLASBAND_TENSOR_INT64 && expected_type == SCYLLASBAND_TENSOR_INT32) {
        const auto* source = static_cast<const int64_t*>(view.data);
        for (std::size_t index = 0; index < count; ++index) {
            if (source[index] < std::numeric_limits<int32_t>::min() ||
                source[index] > std::numeric_limits<int32_t>::max()) {
                throw std::runtime_error("ONNX input '" + name + "' cannot fit int32");
            }
        }
        convert_values<int64_t, int32_t>(view, output, count);
        return output;
    }
    if (view.data_type == SCYLLASBAND_TENSOR_INT32 && expected_type == SCYLLASBAND_TENSOR_INT64) {
        convert_values<int32_t, int64_t>(view, output, count);
        return output;
    }
    if (view.data_type == SCYLLASBAND_TENSOR_BOOL && expected_type == SCYLLASBAND_TENSOR_FLOAT32) {
        convert_values<uint8_t, float>(view, output, count);
        return output;
    }
    if (view.data_type == SCYLLASBAND_TENSOR_FLOAT32 && expected_type == SCYLLASBAND_TENSOR_BOOL) {
        output.converted.resize(count);
        const auto* source = static_cast<const float*>(view.data);
        for (std::size_t index = 0; index < count; ++index) {
            output.converted[index] = source[index] != 0.0f ? 1 : 0;
        }
        output.data = output.converted.data();
        output.byte_length = output.converted.size();
        return output;
    }
    throw std::runtime_error(
        "ONNX input '" + name + "' dtype " + std::to_string(view.data_type) +
        " cannot be converted to " + std::to_string(expected_type)
    );
}

int run_onnx_session(
    const ScyllasBandLiteRtSession* session_handle,
    const ScyllasBandTensorView* inputs,
    int32_t input_count,
    ScyllasBandOwnedTensor** out_tensors,
    int32_t* out_tensor_count
) {
    if (out_tensors == nullptr || out_tensor_count == nullptr) {
        set_error("ONNX session run requires output pointers");
        return -1;
    }
    *out_tensors = nullptr;
    *out_tensor_count = 0;
    if (session_handle == nullptr || inputs == nullptr || input_count < 0) {
        set_error("ONNX session run received invalid arguments");
        return -1;
    }

    auto* native = const_cast<NativeOnnxSession*>(
        reinterpret_cast<const NativeOnnxSession*>(session_handle)
    );
    try {
        if (static_cast<std::size_t>(input_count) != native->input_names.size()) {
            throw std::runtime_error(
                "ONNX graph expects " + std::to_string(native->input_names.size()) +
                " inputs, got " + std::to_string(input_count)
            );
        }

        std::lock_guard<std::mutex> lock(native->run_mutex);
        std::vector<PreparedOnnxInput> prepared;
        prepared.reserve(native->input_names.size());
        for (std::size_t index = 0; index < native->input_names.size(); ++index) {
            validate_shape(inputs[index], native->input_shapes[index], native->input_names[index]);
            prepared.push_back(prepare_input(
                inputs[index], native->input_types[index], native->input_names[index]
            ));
        }

        Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault
        );
        std::vector<Ort::Value> values;
        values.reserve(prepared.size());
        for (std::size_t index = 0; index < prepared.size(); ++index) {
            values.push_back(Ort::Value::CreateTensor(
                memory,
                const_cast<void*>(prepared[index].data),
                prepared[index].byte_length,
                inputs[index].shape,
                static_cast<std::size_t>(inputs[index].rank),
                native->input_types[index]
            ));
        }

        std::vector<const char*> input_names;
        std::vector<const char*> output_names;
        input_names.reserve(native->input_names.size());
        output_names.reserve(native->output_names.size());
        for (const auto& name : native->input_names) input_names.push_back(name.c_str());
        for (const auto& name : native->output_names) output_names.push_back(name.c_str());

        auto outputs = native->session.Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            values.data(),
            values.size(),
            output_names.data(),
            output_names.size()
        );

        auto* owned = static_cast<ScyllasBandOwnedTensor*>(
            std::calloc(outputs.size(), sizeof(ScyllasBandOwnedTensor))
        );
        if (owned == nullptr && !outputs.empty()) {
            throw std::bad_alloc();
        }
        for (std::size_t index = 0; index < outputs.size(); ++index) {
            try {
                if (!outputs[index].IsTensor()) {
                    throw std::runtime_error("ONNX output is not a tensor");
                }
                const auto info = outputs[index].GetTensorTypeAndShapeInfo();
                const auto type = info.GetElementType();
                const int32_t scyllasband_type = scyllasband_type_from_onnx(type);
                if (scyllasband_type == 0) {
                    throw std::runtime_error("ONNX output uses an unsupported dtype");
                }
                const std::vector<int64_t> shape = info.GetShape();
                const std::size_t count = info.GetElementCount();
                const uint64_t bytes = count * scyllasband_tensor_element_size(scyllasband_type);

                owned[index].name = duplicate_c_string(native->output_names[index]);
                owned[index].data_type = scyllasband_type;
                owned[index].rank = static_cast<int32_t>(shape.size());
                owned[index].byte_length = bytes;
                if (!shape.empty()) {
                    owned[index].shape = static_cast<int64_t*>(
                        std::malloc(shape.size() * sizeof(int64_t))
                    );
                    if (owned[index].shape == nullptr) throw std::bad_alloc();
                    std::memcpy(
                        owned[index].shape, shape.data(), shape.size() * sizeof(int64_t)
                    );
                }
                if (bytes > 0) {
                    owned[index].data = std::malloc(static_cast<std::size_t>(bytes));
                    if (owned[index].data == nullptr) throw std::bad_alloc();
                    std::memcpy(
                        owned[index].data,
                        outputs[index].GetTensorRawData(),
                        static_cast<std::size_t>(bytes)
                    );
                }
            } catch (...) {
                for (std::size_t cleanup = 0; cleanup <= index; ++cleanup) {
                    clear_owned_tensor(&owned[cleanup]);
                }
                std::free(owned);
                throw;
            }
        }
        *out_tensors = owned;
        *out_tensor_count = static_cast<int32_t>(outputs.size());
        clear_error();
        return 0;
    } catch (const Ort::Exception& error) {
        set_error(std::string("ONNX Runtime failure: ") + error.what());
        return -1;
    } catch (const std::exception& error) {
        set_error(std::string("ONNX session failure: ") + error.what());
        return -1;
    }
}

}  // namespace
}  // namespace scyllasband_detail

extern "C" {

ScyllasBandLiteRtSession* scyllasband_litert_session_create(
    const char* model_path,
    int32_t accelerator,
    int32_t max_threads
) {
    using namespace scyllasband_detail;
    (void)accelerator;
    if (model_path == nullptr || model_path[0] == '\0') {
        set_error("ONNX session model_path is required");
        return nullptr;
    }
    try {
        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (max_threads > 0) {
            options.SetIntraOpNumThreads(max_threads);
        }
        auto native = std::make_unique<NativeOnnxSession>(
            onnx_environment(), model_path, options
        );
        Ort::AllocatorWithDefaultOptions allocator;
        const std::size_t input_count = native->session.GetInputCount();
        const std::size_t output_count = native->session.GetOutputCount();
        native->input_names.reserve(input_count);
        native->input_types.reserve(input_count);
        native->input_shapes.reserve(input_count);
        native->output_names.reserve(output_count);
        for (std::size_t index = 0; index < input_count; ++index) {
            auto name = native->session.GetInputNameAllocated(index, allocator);
            native->input_names.emplace_back(name.get());
            auto type_info = native->session.GetInputTypeInfo(index);
            auto info = type_info.GetTensorTypeAndShapeInfo();
            native->input_types.push_back(info.GetElementType());
            native->input_shapes.push_back(info.GetShape());
        }
        for (std::size_t index = 0; index < output_count; ++index) {
            auto name = native->session.GetOutputNameAllocated(index, allocator);
            native->output_names.emplace_back(name.get());
        }
        clear_error();
        return reinterpret_cast<ScyllasBandLiteRtSession*>(native.release());
    } catch (const Ort::Exception& error) {
        set_error(std::string("Failed to create ONNX Runtime session: ") + error.what());
        return nullptr;
    } catch (const std::exception& error) {
        set_error(std::string("Failed to create ONNX session: ") + error.what());
        return nullptr;
    }
}

void scyllasband_litert_session_destroy(ScyllasBandLiteRtSession* session) {
    delete reinterpret_cast<scyllasband_detail::NativeOnnxSession*>(session);
}

int scyllasband_litert_session_has_signature(
    const ScyllasBandLiteRtSession* session,
    const char* signature_name
) {
    return session != nullptr && signature_name != nullptr &&
        std::strcmp(signature_name, "serving_default") == 0;
}

int scyllasband_litert_session_run(
    const ScyllasBandLiteRtSession* session,
    const char*,
    const ScyllasBandTensorView* inputs,
    int32_t input_count,
    ScyllasBandOwnedTensor** out_tensors,
    int32_t* out_tensor_count
) {
    return scyllasband_detail::run_onnx_session(
        session, inputs, input_count, out_tensors, out_tensor_count
    );
}

int scyllasband_litert_session_run_resized(
    const ScyllasBandLiteRtSession* session,
    const char*,
    const ScyllasBandTensorView* inputs,
    int32_t input_count,
    ScyllasBandOwnedTensor** out_tensors,
    int32_t* out_tensor_count
) {
    return scyllasband_detail::run_onnx_session(
        session, inputs, input_count, out_tensors, out_tensor_count
    );
}

}  // extern "C"

#endif  // SCYLLASBAND_WITH_ONNXRUNTIME
