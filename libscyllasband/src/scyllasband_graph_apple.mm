// Apple graph sessions: Core ML (iOS 18 / macOS 15 and later) and Core AI (iOS 27 / macOS 27 and later).
//
// The bundle decides: its preferred backend is "coreml" or "coreai". Each asset (.mlmodelc or .aimodel) holds
// several functions (size buckets, the fused flow) that share weights; a component's artifact names its function.
// Core AI has only a Swift API, reached through the C entry points in apple/Sources/ScyllasBandCoreAI.swift.
#include "scyllasband_graph.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <TargetConditionals.h>

#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
// apple/Sources/ScyllasBandCoreAI.swift
int32_t scyllasband_coreai_available(void);
void* scyllasband_coreai_open(const char* path, int32_t unit, char* error, long error_length);
void* scyllasband_coreai_function(void* model, const char* name, char* error, long error_length);
int32_t scyllasband_coreai_run(void* function, int32_t count, const char* const* names, const int32_t* dtypes, const int32_t* ranks,
                               const int64_t* shapes, const void* const* data, float** values, int64_t* value_count, int64_t* out_shape,
                               int32_t* out_rank, char* error, long error_length);
void scyllasband_coreai_free(float* values);
void scyllasband_coreai_release(void* handle);
}

namespace scyllasband {
namespace {

constexpr int kAcceleratorCpu = 0, kAcceleratorGpu = 1, kAcceleratorAuto = 2, kAcceleratorNeuralEngine = 3;

std::vector<std::string> contract_inputs(const Json& spec) {
    std::vector<std::string> names;
    for (const Json& name : spec.get("inputs").elements()) names.push_back(name.str());
    return names;
}

std::size_t element_count(const std::vector<int64_t>& shape) {
    std::size_t count = 1;
    for (int64_t dim : shape) count *= static_cast<std::size_t>(dim);
    return count;
}

double element(const TensorView& view, std::size_t index) {
    switch (view.type) {
        case DType::f32: return static_cast<const float*>(view.data)[index];
        case DType::i64: return static_cast<double>(static_cast<const int64_t*>(view.data)[index]);
        case DType::b8: return static_cast<const uint8_t*>(view.data)[index] ? 1.0 : 0.0;
    }
    return 0.0;
}

// The compute unit for an asset: the caller's choice, or for AUTO the bundle's recommendation ("gpu,g2p=cpu": a default
// and per-asset choices) from controls.<backend>.compute_units. SCYLLASBAND_COMPUTE_UNITS, in the same form, replaces
// the recommendation (for measuring other placements without editing the bundle).
int unit_for(const Json& manifest, const std::string& backend, const std::string& asset, int accelerator) {
    if (accelerator == kAcceleratorNeuralEngine) {
        // Only graphs the bundle lists as accurate there (controls.<backend>.neural_engine_assets) go to the Neural
        // Engine; the rest run on the CPU, which also keeps the whole request usable from a background app.
        for (const Json& name : manifest.get("controls").get(backend).get("neural_engine_assets").elements()) {
            if (name.str() == asset) return kAcceleratorNeuralEngine;
        }
        return kAcceleratorCpu;
    }
    if (accelerator != kAcceleratorAuto) return accelerator;
    const char* override_units = std::getenv("SCYLLASBAND_COMPUTE_UNITS");
    auto parse = [](const std::string& unit) {
        return unit == "cpu" ? kAcceleratorCpu : unit == "gpu" ? kAcceleratorGpu : unit == "ane" ? kAcceleratorNeuralEngine : kAcceleratorAuto;
    };
    const std::string spec = override_units && override_units[0] ? std::string(override_units)
                                                                 : manifest.get("controls").get(backend).get("compute_units").str();
    int fallback = kAcceleratorAuto;
    std::size_t start = 0;
    while (start < spec.size()) {
        std::size_t end = spec.find(',', start);
        if (end == std::string::npos) end = spec.size();
        const std::string part = spec.substr(start, end - start);
        const std::size_t eq = part.find('=');
        if (eq == std::string::npos) fallback = parse(part);
        else if (part.substr(0, eq) == asset) return parse(part.substr(eq + 1));
        start = end + 1;
    }
    return fallback;
}

std::string asset_name(const std::string& path) {
    std::string name = path.substr(path.find_last_of('/') + 1);
    return name.substr(0, name.find('.'));
}

// --- Core ML ---------------------------------------------------------------------------------------------------------
class CoreMLGraph final : public Graph {
public:
    CoreMLGraph(MLModel* model, std::vector<std::string> inputs) : model_(model), inputs_(std::move(inputs)) {
        NSDictionary<NSString*, MLFeatureDescription*>* described = model_.modelDescription.inputDescriptionsByName;
        for (const std::string& name : inputs_) {
            MLFeatureDescription* feature = described[[NSString stringWithUTF8String:name.c_str()]];
            // Inputs the graph does not read (none in practice) are skipped.
            types_.push_back(feature ? feature.multiArrayConstraint.dataType : (MLMultiArrayDataType)0);
        }
        output_ = model_.modelDescription.outputDescriptionsByName.allKeys.firstObject;
    }

    GraphOutput run(const std::vector<TensorView>& inputs) override {
        @autoreleasepool {
            NSMutableDictionary<NSString*, MLFeatureValue*>* features = [NSMutableDictionary dictionary];
            for (std::size_t i = 0; i < inputs_.size(); ++i) {
                if (!types_[i]) continue;
                const TensorView& view = inputs.at(i);
                NSMutableArray<NSNumber*>* shape = [NSMutableArray array];
                for (int64_t dim : view.shape) [shape addObject:@(dim)];
                NSError* error = nil;
                MLMultiArray* array = [[MLMultiArray alloc] initWithShape:shape dataType:types_[i] error:&error];
                if (!array) throw std::runtime_error("Core ML input " + inputs_[i] + ": " + error.localizedDescription.UTF8String);
                const std::size_t count = element_count(view.shape);
                const MLMultiArrayDataType type = types_[i];
                [array getMutableBytesWithHandler:^(void* bytes, NSInteger, NSArray<NSNumber*>*) {   // fresh arrays are contiguous
                    for (std::size_t k = 0; k < count; ++k) {
                        double value = element(view, k);
                        if (type == MLMultiArrayDataTypeInt32) static_cast<int32_t*>(bytes)[k] = static_cast<int32_t>(value);
                        else if (type == MLMultiArrayDataTypeFloat16) static_cast<_Float16*>(bytes)[k] = static_cast<_Float16>(value);
                        else if (type == MLMultiArrayDataTypeDouble) static_cast<double*>(bytes)[k] = value;
                        else static_cast<float*>(bytes)[k] = static_cast<float>(value);
                    }
                }];
                features[[NSString stringWithUTF8String:inputs_[i].c_str()]] = [MLFeatureValue featureValueWithMultiArray:array];
            }
            NSError* error = nil;
            MLDictionaryFeatureProvider* provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:features error:&error];
            id<MLFeatureProvider> result = provider ? [model_ predictionFromFeatures:provider error:&error] : nil;
            if (!result) throw std::runtime_error(std::string("Core ML prediction failed: ") + error.localizedDescription.UTF8String);
            MLMultiArray* array = [result featureValueForName:output_].multiArrayValue;
            if (!array) throw std::runtime_error("Core ML output is not a tensor");
            GraphOutput out;
            for (NSNumber* dim in array.shape) out.shape.push_back(dim.longLongValue);
            out.values.resize(element_count(out.shape));
            std::vector<int64_t> strides;
            for (NSNumber* stride in array.strides) strides.push_back(stride.longLongValue);
            const MLMultiArrayDataType type = array.dataType;
            std::vector<int64_t> shape = out.shape;
            float* destination = out.values.data();
            const std::size_t count = out.values.size();
            [array getBytesWithHandler:^(const void* bytes, NSInteger) {   // honour strides: GPU and Neural Engine outputs may be padded
                std::vector<int64_t> index(shape.size(), 0);
                for (std::size_t flat = 0; flat < count; ++flat) {
                    int64_t offset = 0;
                    for (std::size_t axis = 0; axis < shape.size(); ++axis) offset += index[axis] * strides[axis];
                    if (type == MLMultiArrayDataTypeFloat16) destination[flat] = static_cast<float>(static_cast<const _Float16*>(bytes)[offset]);
                    else if (type == MLMultiArrayDataTypeDouble) destination[flat] = static_cast<float>(static_cast<const double*>(bytes)[offset]);
                    else destination[flat] = static_cast<const float*>(bytes)[offset];
                    for (std::size_t axis = shape.size(); axis-- > 0;) {
                        if (++index[axis] < shape[axis]) break;
                        index[axis] = 0;
                    }
                }
            }];
            return out;
        }
    }

private:
    MLModel* model_;
    std::vector<std::string> inputs_;
    std::vector<MLMultiArrayDataType> types_;
    NSString* output_;
};

// --- Core AI ---------------------------------------------------------------------------------------------------------
struct CoreAIHandle {
    void* handle = nullptr;
    explicit CoreAIHandle(void* value) : handle(value) {}
    ~CoreAIHandle() { scyllasband_coreai_release(handle); }
};

class CoreAIGraph final : public Graph {
public:
    CoreAIGraph(std::shared_ptr<CoreAIHandle> model, void* function, std::vector<std::string> inputs)
        : model_(std::move(model)), function_(function), inputs_(std::move(inputs)) {}

    GraphOutput run(const std::vector<TensorView>& inputs) override {
        std::vector<const char*> names;
        std::vector<int32_t> dtypes, ranks;
        std::vector<int64_t> shapes;
        std::vector<const void*> data;
        for (std::size_t i = 0; i < inputs_.size(); ++i) {
            const TensorView& view = inputs.at(i);
            names.push_back(inputs_[i].c_str());
            dtypes.push_back(view.type == DType::f32 ? 0 : view.type == DType::i64 ? 1 : 2);
            ranks.push_back(static_cast<int32_t>(view.shape.size()));
            shapes.insert(shapes.end(), view.shape.begin(), view.shape.end());
            data.push_back(view.data);
        }
        float* values = nullptr;
        int64_t count = 0, shape[8] = {0};
        int32_t rank = 0;
        char error[512] = {0};
        if (scyllasband_coreai_run(function_.handle, static_cast<int32_t>(names.size()), names.data(), dtypes.data(), ranks.data(),
                                   shapes.data(), data.data(), &values, &count, shape, &rank, error, sizeof(error)) != 0) {
            throw std::runtime_error(error);
        }
        GraphOutput out;
        out.shape.assign(shape, shape + rank);
        out.values.assign(values, values + count);
        scyllasband_coreai_free(values);
        return out;
    }

private:
    std::shared_ptr<CoreAIHandle> model_;   // keeps the asset alive while its functions are in use
    CoreAIHandle function_;
    std::vector<std::string> inputs_;
};

// --- backend ---------------------------------------------------------------------------------------------------------
class AppleBackend final : public Backend {
public:
    AppleBackend(BackendOptions options, std::string backend, Json manifest)
        : options_(std::move(options)), backend_(std::move(backend)), manifest_(std::move(manifest)) {}

    const char* name() const override { return backend_.c_str(); }

    std::string accelerator() const override {
        switch (options_.accelerator) {
            case kAcceleratorCpu: return "cpu";
            case kAcceleratorGpu: return "gpu";
            case kAcceleratorNeuralEngine: return "neural_engine";
            default: return "auto";
        }
    }

    std::unique_ptr<Graph> open(const std::string& component, const Json& spec) override {
        const Json& artifact = spec.get("artifacts").get(backend_);
        const std::string path = options_.bundle_dir + "/" + artifact.get("path").str();
        const std::string function = artifact.get("function").str();
        const int unit = unit_for(manifest_, backend_, asset_name(path), options_.accelerator);
        return backend_ == "coreai" ? open_coreai(component, path, function, unit, spec) : open_coreml(component, path, function, unit, spec);
    }

private:
    std::unique_ptr<Graph> open_coreml(const std::string& component, const std::string& path, const std::string& function, int unit,
                                       const Json& spec) {
        @autoreleasepool {
            MLModelConfiguration* configuration = [[MLModelConfiguration alloc] init];
#if TARGET_OS_SIMULATOR
            unit = kAcceleratorCpu;   // the Simulator's Core ML has no usable GPU or Neural Engine path for these graphs
#endif
            configuration.computeUnits = unit == kAcceleratorCpu ? MLComputeUnitsCPUOnly
                                         : unit == kAcceleratorGpu ? MLComputeUnitsCPUAndGPU
                                         : unit == kAcceleratorNeuralEngine ? MLComputeUnitsCPUAndNeuralEngine
                                                                            : MLComputeUnitsAll;
            if (!function.empty()) {
                if (@available(iOS 18.0, macOS 15.0, visionOS 2.0, *)) {
                    configuration.functionName = [NSString stringWithUTF8String:function.c_str()];
                } else {
                    throw std::runtime_error("Core ML bundles need iOS 18 or macOS 15");
                }
            }
            NSError* error = nil;
            NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
            MLModel* model = [MLModel modelWithContentsOfURL:url configuration:configuration error:&error];
            if (!model) {
                throw std::runtime_error("Cannot load " + component + " (" + path + (function.empty() ? "" : ":" + function) + "): " +
                                         error.localizedDescription.UTF8String);
            }
            return std::make_unique<CoreMLGraph>(model, contract_inputs(spec));
        }
    }

    std::unique_ptr<Graph> open_coreai(const std::string& component, const std::string& path, const std::string& function, int unit,
                                       const Json& spec) {
        char error[512] = {0};
        std::shared_ptr<CoreAIHandle> model;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            std::weak_ptr<CoreAIHandle>& cached = models_[path + "#" + std::to_string(unit)];
            model = cached.lock();
            if (!model) {
                void* handle = scyllasband_coreai_open(path.c_str(), unit, error, sizeof(error));
                if (!handle) throw std::runtime_error("Cannot load " + component + ": " + error);
                model = std::make_shared<CoreAIHandle>(handle);
                cached = model;
            }
        }
        void* handle = scyllasband_coreai_function(model->handle, function.empty() ? nullptr : function.c_str(), error, sizeof(error));
        if (!handle) throw std::runtime_error("Cannot load " + component + ": " + error);
        return std::make_unique<CoreAIGraph>(model, handle, contract_inputs(spec));
    }

    BackendOptions options_;
    std::string backend_;
    Json manifest_;
    std::mutex mutex_;
    std::map<std::string, std::weak_ptr<CoreAIHandle>> models_;
};

}  // namespace

std::unique_ptr<Backend> make_backend(const BackendOptions& options) {
    Json manifest = Json::parse_file(options.bundle_dir + "/manifest.json");
    std::string backend;
    for (const Json& preferred : manifest.get("preferred_backends").elements()) {
        if (preferred.str() == "coreml" || preferred.str() == "coreai") {
            backend = preferred.str();
            break;
        }
    }
    if (backend.empty()) throw std::invalid_argument("This runtime runs Core ML and Core AI bundles; " + options.bundle_dir + " is neither");
#if TARGET_OS_SIMULATOR
    if (backend == "coreai") throw std::invalid_argument("Core AI does not run in the Simulator; use the Core ML bundle");
#endif
    if (backend == "coreai" && !scyllasband_coreai_available()) {
        throw std::invalid_argument("Core AI bundles need iOS 27 or macOS 27; use the Core ML bundle on this system");
    }
    if (backend == "coreml") {
        if (@available(iOS 18.0, macOS 15.0, visionOS 2.0, *)) {
        } else {
            throw std::invalid_argument("Core ML bundles need iOS 18 or macOS 15");
        }
    }
    return std::make_unique<AppleBackend>(options, backend, std::move(manifest));
}

}  // namespace scyllasband
