// Graph sessions: one backend (ONNX Runtime or LiteRT) is linked into each build.
#pragma once

#include "scyllasband_json.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace scyllasband {

enum class DType { f32, i64, b8 };

struct TensorView {
    DType type;
    std::vector<int64_t> shape;
    const void* data;
};

struct GraphOutput {
    std::vector<int64_t> shape;
    std::vector<float> values;
};

class Graph {
public:
    virtual ~Graph() = default;
    // `inputs` follow the component's contract input order; returns the first output.
    virtual GraphOutput run(const std::vector<TensorView>& inputs) = 0;
};

struct BackendOptions {
    std::string bundle_dir;
    int threads = 0;
    int accelerator = 0;  // ScyllasBandAccelerator
};

class Backend {
public:
    virtual ~Backend() = default;
    virtual const char* name() const = 0;  // the manifest artifact key: "onnx" or "litert"
    virtual std::string accelerator() const { return "cpu"; }
    virtual std::unique_ptr<Graph> open(const std::string& component, const Json& spec) = 0;
};

std::unique_ptr<Backend> make_backend(const BackendOptions& options);

}  // namespace scyllasband
