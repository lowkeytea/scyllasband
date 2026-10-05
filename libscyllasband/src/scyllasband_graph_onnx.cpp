// ONNX Runtime graph sessions.
#include "scyllasband_graph.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace scyllasband {
namespace {

Ort::Env& environment() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "scyllasband");
    return env;
}

ONNXTensorElementDataType onnx_type(DType type) {
    switch (type) {
        case DType::f32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        case DType::i64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        case DType::b8: return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    }
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
}

std::size_t element_count(const std::vector<int64_t>& shape) {
    std::size_t count = 1;
    for (int64_t dim : shape) count *= static_cast<std::size_t>(dim);
    return count;
}

class OnnxGraph final : public Graph {
public:
    OnnxGraph(const std::string& path, const Json& spec, int threads) {
        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (threads > 0) options.SetIntraOpNumThreads(threads);
        session_ = std::make_unique<Ort::Session>(environment(), path.c_str(), options);
        Ort::AllocatorWithDefaultOptions allocator;
        std::vector<std::string> session_inputs;
        for (std::size_t i = 0; i < session_->GetInputCount(); ++i) {
            session_inputs.emplace_back(session_->GetInputNameAllocated(i, allocator).get());
            types_.push_back(session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType());
        }
        // Feed by name: map each session input to its position in the contract order.
        std::vector<std::string> contract;
        for (const Json& name : spec.get("inputs").elements()) contract.push_back(name.str());
        for (const std::string& name : session_inputs) {
            auto it = std::find(contract.begin(), contract.end(), name);
            if (it == contract.end()) throw std::runtime_error("ONNX graph " + path + " has input '" + name + "' outside the contract");
            order_.push_back(static_cast<std::size_t>(it - contract.begin()));
        }
        input_names_ = session_inputs;
        output_name_ = session_->GetOutputNameAllocated(0, allocator).get();
    }

    GraphOutput run(const std::vector<TensorView>& inputs) override {
        Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> values;
        std::vector<std::vector<uint8_t>> converted(order_.size());
        for (std::size_t i = 0; i < order_.size(); ++i) {
            const TensorView& view = inputs.at(order_[i]);
            const std::size_t count = element_count(view.shape);
            const void* data = view.data;
            std::size_t bytes = count * (view.type == DType::b8 ? 1 : view.type == DType::f32 ? 4 : 8);
            ONNXTensorElementDataType expected = types_[i];
            if (expected != onnx_type(view.type)) {
                if (view.type == DType::i64 && expected == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
                    converted[i].resize(count * 4);
                    auto* out = reinterpret_cast<int32_t*>(converted[i].data());
                    for (std::size_t k = 0; k < count; ++k) out[k] = static_cast<int32_t>(static_cast<const int64_t*>(view.data)[k]);
                    data = converted[i].data();
                    bytes = count * 4;
                } else {
                    throw std::runtime_error("ONNX input '" + input_names_[i] + "' has an unsupported element type");
                }
            }
            values.push_back(Ort::Value::CreateTensor(memory, const_cast<void*>(data), bytes, view.shape.data(), view.shape.size(), expected));
        }
        std::vector<const char*> names;
        for (const std::string& name : input_names_) names.push_back(name.c_str());
        const char* output = output_name_.c_str();
        auto outputs = session_->Run(Ort::RunOptions{nullptr}, names.data(), values.data(), values.size(), &output, 1);
        auto info = outputs[0].GetTensorTypeAndShapeInfo();
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) throw std::runtime_error("ONNX output is not float32");
        GraphOutput result;
        result.shape = info.GetShape();
        const float* data = outputs[0].GetTensorData<float>();
        result.values.assign(data, data + info.GetElementCount());
        return result;
    }

private:
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> input_names_;
    std::vector<ONNXTensorElementDataType> types_;
    std::vector<std::size_t> order_;
    std::string output_name_;
};

class OnnxBackend final : public Backend {
public:
    explicit OnnxBackend(BackendOptions options) : options_(std::move(options)) {}
    const char* name() const override { return "onnx"; }
    std::unique_ptr<Graph> open(const std::string& component, const Json& spec) override {
        const std::string path = options_.bundle_dir + "/" + spec.get("artifacts").get("onnx").get("path").str();
        try {
            return std::make_unique<OnnxGraph>(path, spec, options_.threads);
        } catch (const Ort::Exception& error) {
            throw std::runtime_error("Cannot load " + component + " (" + path + "): " + error.what());
        }
    }

private:
    BackendOptions options_;
};

}  // namespace

std::unique_ptr<Backend> make_backend(const BackendOptions& options) { return std::make_unique<OnnxBackend>(options); }

}  // namespace scyllasband
