// LiteRT graph sessions: one compiled model per .tflite file, one signature per graph. The flow and the
// vocoder hold every latent bucket as a signature of one file, so their weights are loaded once.
//
// CPU: XNNPACK packs each signature's weights separately unless they go through a weight cache, so every graph gets
// one: a file of packed weights, memory-mapped and shared by all of its signatures (SCYLLASBAND_CACHE_DIR, else the
// graph's own directory; SCYLLASBAND_XNNPACK_CACHE=0 turns it off). GPU: constant tensors are shared between
// signatures, the original weights are released after upload, and compiled programs are cached in the same
// directory. The bundle can set the GPU precision (controls.litert.gpu.precision: fp16, fp32 or
// fp16_fp32_accumulation), allow int8 weights (controls.litert.gpu.quantized_weights) and name the graph files that use
// the GPU (controls.litert.gpu.graphs, file names without .tflite; the others run on the CPU; default all);
// SCYLLASBAND_GPU_PRECISION, SCYLLASBAND_GPU_PRIORITY (low, normal, high) and SCYLLASBAND_GPU_GRAPHS (comma-separated)
// override for a device. Options travel as LiteRT opaque TOML
// payloads ("xnnpack", "gpu_options"): libLiteRt 2.2.0 does not export the typed option setters.
#include "scyllasband_graph.h"

#include "scyllasband.h"

#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <vector>

#if defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

namespace scyllasband {
namespace {

void check(LiteRtStatus status, const std::string& what) {
    if (status != kLiteRtStatusOk) throw std::runtime_error("LiteRT " + what + " failed (status " + std::to_string(static_cast<int>(status)) + ")");
}

std::size_t element_size(LiteRtElementType type) {
    switch (type) {
        case kLiteRtElementTypeFloat32: case kLiteRtElementTypeInt32: return 4;
        case kLiteRtElementTypeInt64: return 8;
        case kLiteRtElementTypeBool: case kLiteRtElementTypeInt8: case kLiteRtElementTypeUInt8: return 1;
        default: return 0;
    }
}

std::size_t byte_size(const LiteRtRankedTensorType& type) {
    std::size_t count = 1;
    for (unsigned d = 0; d < type.layout.rank; ++d) {
        if (type.layout.dimensions[d] <= 0) throw std::runtime_error("LiteRT signature has a dynamic dimension");
        count *= static_cast<std::size_t>(type.layout.dimensions[d]);
    }
    return count * element_size(type.element_type);
}

int arg_index(const std::string& name) {
    std::size_t underscore = name.rfind('_');
    if (underscore == std::string::npos) return 0;
    std::string tail = name.substr(underscore + 1);
    return !tail.empty() && std::all_of(tail.begin(), tail.end(), [](unsigned char c) { return std::isdigit(c) != 0; }) ? std::atoi(tail.c_str()) : 0;
}

// The directory holding libLiteRt, where LiteRT looks for its accelerator plugins.
std::string runtime_library_dir() {
#if defined(__linux__) || defined(__APPLE__)
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&LiteRtCreateEnvironment), &info) && info.dli_fname) {
        std::string path(info.dli_fname);
        std::size_t slash = path.rfind('/');
        if (slash != std::string::npos) return path.substr(0, slash);
    }
#endif
    return {};
}

std::string env_string(const char* name) {
    const char* value = std::getenv(name);
    return value ? std::string(value) : std::string();
}

// Compile settings shared by every graph of a bundle.
struct Settings {
    int threads = 0;
    std::string cache_dir;            // empty: the graph's own directory
    bool weight_cache = true;         // XNNPACK weight cache (CPU)
    int gpu_precision = 0;            // LiteRtDelegatePrecision; 0: the accelerator's default
    int gpu_priority = 0;             // LiteRtGpuPriority; 0: default
    bool gpu_quantized_weights = false;
    std::vector<std::string> gpu_graphs;  // graph files (without .tflite) on the GPU; empty: all
};

int precision_value(const std::string& name) {
    if (name == "fp16") return 1;
    if (name == "fp32") return 2;
    if (name == "fp16_fp32_accumulation") return 3;
    return 0;
}

int priority_value(const std::string& name) {
    if (name == "low") return 1;
    if (name == "normal") return 2;
    if (name == "high") return 3;
    return 0;
}

// Opaque options for one accelerator, from a TOML payload.
LiteRtStatus add_opaque(LiteRtOptions options, const char* identifier, const std::string& toml) {
    char* payload = new char[toml.size() + 1];
    std::memcpy(payload, toml.c_str(), toml.size() + 1);
    LiteRtOpaqueOptions opaque = nullptr;
    LiteRtStatus status = LiteRtCreateOpaqueOptions(identifier, payload, [](void* data) { delete[] static_cast<char*>(data); }, &opaque);
    if (status != kLiteRtStatusOk) {
        delete[] payload;
        return status;
    }
    return LiteRtAddOpaqueOptions(options, opaque);
}

class Model {
public:
    Model(LiteRtEnvironment env, const std::string& path, int accelerator, const Settings& settings) : env_(env), settings_(settings) {
        check(LiteRtCreateModelFromFile(env, path.c_str(), &model_), "loading " + path);
        // Cache files are keyed by the graph's name, size and modification time, so a replaced graph never meets an old cache.
        std::error_code error;
        const std::filesystem::path graph(path);
        const auto size = std::filesystem::file_size(graph, error);
        const long long stamp = error ? 0 : static_cast<long long>(std::filesystem::last_write_time(graph, error).time_since_epoch().count());
        const std::filesystem::path dir = settings.cache_dir.empty() ? graph.parent_path() : std::filesystem::path(settings.cache_dir);
        if (!error && std::filesystem::is_directory(dir, error)) {
            cache_dir_ = dir.string();
            cache_key_ = graph.filename().string() + "-" + std::to_string(static_cast<unsigned long long>(size)) + "-" + std::to_string(stamp);
        }
        LiteRtHwAcceleratorSet hardware = kLiteRtHwAcceleratorCpu;
        if (accelerator == SCYLLASBAND_ACCELERATOR_GPU || accelerator == SCYLLASBAND_ACCELERATOR_AUTO) hardware = kLiteRtHwAcceleratorGpu | kLiteRtHwAcceleratorCpu;
        LiteRtStatus status = compile(hardware, !cache_dir_.empty());
        if (status != kLiteRtStatusOk && !cache_dir_.empty()) status = compile(hardware, false);  // e.g. a read-only directory
        if (status != kLiteRtStatusOk && accelerator == SCYLLASBAND_ACCELERATOR_AUTO) status = compile(kLiteRtHwAcceleratorCpu, !cache_dir_.empty());
        if (status != kLiteRtStatusOk && accelerator == SCYLLASBAND_ACCELERATOR_AUTO && !cache_dir_.empty()) status = compile(kLiteRtHwAcceleratorCpu, false);
        if (status != kLiteRtStatusOk) {
            LiteRtDestroyModel(model_);
            check(status, "compiling " + path);
        }
    }
    ~Model() {
        for (auto& [name, signature] : signatures_) {
            for (LiteRtTensorBuffer buffer : signature.inputs) LiteRtDestroyTensorBuffer(buffer);
            for (LiteRtTensorBuffer buffer : signature.outputs) LiteRtDestroyTensorBuffer(buffer);
        }
        if (compiled_) LiteRtDestroyCompiledModel(compiled_);
        if (model_) LiteRtDestroyModel(model_);
    }

    struct Signature {
        LiteRtParamIndex index = 0;
        std::vector<std::string> input_names;
        std::vector<LiteRtRankedTensorType> input_types;
        LiteRtRankedTensorType output_type{};
        std::vector<LiteRtTensorBuffer> inputs, outputs;
    };

    Signature& signature(const std::string& key) {
        auto found = signatures_.find(key);
        if (found != signatures_.end()) return found->second;
        LiteRtParamIndex count = 0;
        check(LiteRtGetNumModelSignatures(model_, &count), "listing signatures");
        for (LiteRtParamIndex i = 0; i < count; ++i) {
            LiteRtSignature handle = nullptr;
            const char* name = nullptr;
            check(LiteRtGetModelSignature(model_, i, &handle), "reading a signature");
            check(LiteRtGetSignatureKey(handle, &name), "reading a signature key");
            if (key != name) continue;
            Signature sig;
            sig.index = i;
            LiteRtParamIndex inputs = 0, outputs = 0;
            check(LiteRtGetNumSignatureInputs(handle, &inputs), "reading inputs");
            for (LiteRtParamIndex k = 0; k < inputs; ++k) {
                const char* input_name = nullptr;
                LiteRtTensor tensor = nullptr;
                LiteRtRankedTensorType type{};
                check(LiteRtGetSignatureInputName(handle, k, &input_name), "reading an input name");
                check(LiteRtGetSignatureInputTensorByIndex(handle, k, &tensor), "reading an input tensor");
                check(LiteRtGetRankedTensorType(tensor, &type), "reading an input type");
                sig.input_names.emplace_back(input_name);
                sig.input_types.push_back(type);
                LiteRtTensorBuffer buffer = nullptr;
                check(LiteRtCreateManagedTensorBuffer(env_, kLiteRtTensorBufferTypeHostMemory, &type, byte_size(type), &buffer), "allocating an input");
                sig.inputs.push_back(buffer);
            }
            check(LiteRtGetNumSignatureOutputs(handle, &outputs), "reading outputs");
            for (LiteRtParamIndex k = 0; k < outputs; ++k) {
                LiteRtTensor tensor = nullptr;
                LiteRtRankedTensorType type{};
                check(LiteRtGetSignatureOutputTensorByIndex(handle, k, &tensor), "reading an output tensor");
                check(LiteRtGetRankedTensorType(tensor, &type), "reading an output type");
                if (k == 0) sig.output_type = type;
                LiteRtTensorBuffer buffer = nullptr;
                check(LiteRtCreateManagedTensorBuffer(env_, kLiteRtTensorBufferTypeHostMemory, &type, byte_size(type), &buffer), "allocating an output");
                sig.outputs.push_back(buffer);
            }
            return signatures_.emplace(key, std::move(sig)).first->second;
        }
        throw std::runtime_error("LiteRT model has no signature '" + key + "'");
    }

    void run(Signature& sig) {
        check(LiteRtRunCompiledModel(compiled_, sig.index, sig.inputs.size(), sig.inputs.data(), sig.outputs.size(), sig.outputs.data()), "running");
    }

    std::mutex mutex;

private:
    LiteRtStatus compile(LiteRtHwAcceleratorSet hardware, bool cache) {
        LiteRtOptions options = nullptr;
        check(LiteRtCreateOptions(&options), "creating options");
        LiteRtStatus status = LiteRtSetOptionsHardwareAccelerators(options, hardware);
        // TOML basic strings: LiteRT does not unescape them, so paths travel with forward slashes (fine on Windows too).
        auto toml_path = [](std::string value) {
            std::replace(value.begin(), value.end(), '\\', '/');
            return value;
        };
        const std::string base = cache ? toml_path((std::filesystem::path(cache_dir_) / cache_key_).string()) : std::string();
        // CPU accelerator options are a TOML payload under the "xnnpack" identifier.
        std::string cpu;
        if (settings_.threads > 0) cpu += "num_threads = " + std::to_string(settings_.threads) + "\n";
        if (cache && settings_.weight_cache) cpu += "weight_cache_file_path = \"" + base + ".xnnpack-cache\"\n";
        if (status == kLiteRtStatusOk && !cpu.empty()) status = add_opaque(options, "xnnpack", cpu);
        if (status == kLiteRtStatusOk && (hardware & kLiteRtHwAcceleratorGpu)) {
            std::string gpu = "enable_constant_tensors_sharing = true\nmadvise_original_shared_tensors = true\n";
            if (settings_.gpu_precision) gpu += "precision = " + std::to_string(settings_.gpu_precision) + "\n";
            if (settings_.gpu_priority) gpu += "priority = " + std::to_string(settings_.gpu_priority) + "\n";
            if (settings_.gpu_quantized_weights) gpu += "allow_src_quantized_fc_conv_ops = true\n";
            if (cache) gpu += "serialization_dir = \"" + toml_path(cache_dir_) + "\"\nmodel_cache_key = \"" + cache_key_ + "\"\nserialize_program_cache = true\n";
            status = add_opaque(options, "gpu_options", gpu);
        }
        if (status == kLiteRtStatusOk) status = LiteRtCreateCompiledModel(env_, model_, options, &compiled_);
        LiteRtDestroyOptions(options);
        return status;
    }

    LiteRtEnvironment env_;
    Settings settings_;
    std::string cache_dir_, cache_key_;
    LiteRtModel model_ = nullptr;
    LiteRtCompiledModel compiled_ = nullptr;
    std::map<std::string, Signature> signatures_;
};

class LiteRtGraph final : public Graph {
public:
    LiteRtGraph(std::shared_ptr<Model> model, const std::string& signature, const Json& spec) : model_(std::move(model)) {
        std::lock_guard<std::mutex> lock(model_->mutex);
        Model::Signature& sig = model_->signature(signature);
        signature_ = signature;
        std::vector<std::string> contract;
        for (const Json& name : spec.get("inputs").elements()) contract.push_back(name.str());
        if (contract.size() != sig.input_names.size()) throw std::runtime_error("LiteRT signature '" + signature + "' does not match the contract inputs");
        // LiteRT keeps traced argument names when it can, else args_0..N in contract order.
        bool by_name = std::all_of(contract.begin(), contract.end(), [&](const std::string& name) {
            return std::find(sig.input_names.begin(), sig.input_names.end(), name) != sig.input_names.end();
        });
        std::vector<std::size_t> slots(sig.input_names.size());
        for (std::size_t i = 0; i < slots.size(); ++i) slots[i] = i;
        if (!by_name) {
            std::stable_sort(slots.begin(), slots.end(), [&](std::size_t a, std::size_t b) { return arg_index(sig.input_names[a]) < arg_index(sig.input_names[b]); });
        }
        slot_for_contract_.resize(contract.size());
        for (std::size_t c = 0; c < contract.size(); ++c) {
            if (by_name) {
                slot_for_contract_[c] = static_cast<std::size_t>(std::find(sig.input_names.begin(), sig.input_names.end(), contract[c]) - sig.input_names.begin());
            } else {
                slot_for_contract_[c] = slots[c];
            }
        }
    }

    GraphOutput run(const std::vector<TensorView>& inputs) override {
        std::lock_guard<std::mutex> lock(model_->mutex);
        Model::Signature& sig = model_->signature(signature_);
        for (std::size_t c = 0; c < inputs.size(); ++c) {
            const std::size_t slot = slot_for_contract_.at(c);
            const TensorView& view = inputs[c];
            const LiteRtRankedTensorType& type = sig.input_types[slot];
            const std::size_t bytes = byte_size(type), count = bytes / element_size(type.element_type);
            std::size_t given = 1;
            for (int64_t dim : view.shape) given *= static_cast<std::size_t>(dim);
            if (given != count) throw std::runtime_error("LiteRT input " + std::to_string(c) + " of '" + signature_ + "' has the wrong size");
            void* host = nullptr;
            check(LiteRtLockTensorBuffer(sig.inputs[slot], &host, kLiteRtTensorBufferLockModeWrite), "locking an input");
            const bool i64_to_i32 = view.type == DType::i64 && type.element_type == kLiteRtElementTypeInt32;
            if (i64_to_i32) {
                for (std::size_t k = 0; k < count; ++k) static_cast<int32_t*>(host)[k] = static_cast<int32_t>(static_cast<const int64_t*>(view.data)[k]);
            } else {
                const bool same = (view.type == DType::f32 && type.element_type == kLiteRtElementTypeFloat32) ||
                                  (view.type == DType::i64 && type.element_type == kLiteRtElementTypeInt64) ||
                                  (view.type == DType::b8 && type.element_type == kLiteRtElementTypeBool);
                if (!same) {
                    LiteRtUnlockTensorBuffer(sig.inputs[slot]);
                    throw std::runtime_error("LiteRT input " + std::to_string(c) + " of '" + signature_ + "' has an unsupported element type");
                }
                std::memcpy(host, view.data, bytes);
            }
            check(LiteRtUnlockTensorBuffer(sig.inputs[slot]), "unlocking an input");
        }
        model_->run(sig);
        GraphOutput out;
        for (unsigned d = 0; d < sig.output_type.layout.rank; ++d) out.shape.push_back(sig.output_type.layout.dimensions[d]);
        if (sig.output_type.element_type != kLiteRtElementTypeFloat32) throw std::runtime_error("LiteRT output is not float32");
        out.values.resize(byte_size(sig.output_type) / 4);
        void* host = nullptr;
        check(LiteRtLockTensorBuffer(sig.outputs[0], &host, kLiteRtTensorBufferLockModeRead), "locking the output");
        std::memcpy(out.values.data(), host, out.values.size() * 4);
        LiteRtUnlockTensorBuffer(sig.outputs[0]);
        return out;
    }

private:
    std::shared_ptr<Model> model_;
    std::string signature_;
    std::vector<std::size_t> slot_for_contract_;
};

class LiteRtBackend final : public Backend {
public:
    explicit LiteRtBackend(BackendOptions options) : options_(std::move(options)) {
        std::vector<LiteRtEnvOption> env_options;
        auto int_option = [&](LiteRtEnvOptionTag tag, int64_t value) {
            LiteRtEnvOption option{};
            option.tag = tag;
            option.value.type = kLiteRtAnyTypeInt;
            option.value.int_value = value;
            env_options.push_back(option);
        };
        int_option(kLiteRtEnvOptionTagMinLoggerSeverity, 3);  // errors only
        if (options_.accelerator == SCYLLASBAND_ACCELERATOR_CPU) {
            int_option(kLiteRtEnvOptionTagAutoRegisterAccelerators, kLiteRtHwAcceleratorCpu);  // no GPU/NPU plugin loading
        }
        library_dir_ = runtime_library_dir();
        if (!library_dir_.empty()) {
            LiteRtEnvOption option{};
            option.tag = kLiteRtEnvOptionTagRuntimeLibraryDir;
            option.value.type = kLiteRtAnyTypeString;
            option.value.str_value = library_dir_.c_str();
            env_options.push_back(option);
        }
        check(LiteRtCreateEnvironment(static_cast<int>(env_options.size()), env_options.data(), &env_), "creating the environment");
    }
    ~LiteRtBackend() override {
        models_.clear();
        if (env_) LiteRtDestroyEnvironment(env_);
    }
    const char* name() const override { return "litert"; }
    std::string accelerator() const override {
        switch (options_.accelerator) {
            case SCYLLASBAND_ACCELERATOR_GPU: return "gpu";
            case SCYLLASBAND_ACCELERATOR_AUTO: return "auto";
            default: return "cpu";
        }
    }
    void configure(const Json& manifest) override {
        const Json& gpu = manifest.get("controls").get("litert").get("gpu");
        settings_.threads = threads();
        settings_.cache_dir = env_string("SCYLLASBAND_CACHE_DIR");
        settings_.weight_cache = env_string("SCYLLASBAND_XNNPACK_CACHE") != "0";
        const std::string precision = env_string("SCYLLASBAND_GPU_PRECISION");
        settings_.gpu_precision = precision_value(precision.empty() ? gpu.get("precision").str() : precision);
        settings_.gpu_priority = priority_value(env_string("SCYLLASBAND_GPU_PRIORITY"));
        settings_.gpu_quantized_weights = gpu.get("quantized_weights").boolean(false);
        settings_.gpu_graphs.clear();
        const std::string graphs = env_string("SCYLLASBAND_GPU_GRAPHS");
        if (!graphs.empty()) {
            std::size_t start = 0;
            while (start <= graphs.size()) {
                const std::size_t comma = std::min(graphs.find(',', start), graphs.size());
                if (comma > start) settings_.gpu_graphs.push_back(graphs.substr(start, comma - start));
                start = comma + 1;
            }
        } else {
            for (const Json& name : gpu.get("graphs").elements()) settings_.gpu_graphs.push_back(name.str());
        }
        configured_ = true;
    }
    std::unique_ptr<Graph> open(const std::string& component, const Json& spec) override {
        const Json& artifact = spec.get("artifacts").get("litert");
        const std::string path = options_.bundle_dir + "/" + artifact.get("path").str();
        if (!configured_) configure(Json(Json::Object{}));
        std::shared_ptr<Model>& model = models_[path];
        if (!model) {
            int accelerator = options_.accelerator;
            if (accelerator != SCYLLASBAND_ACCELERATOR_CPU && !settings_.gpu_graphs.empty()) {
                const std::string stem = std::filesystem::path(path).stem().string();
                if (std::find(settings_.gpu_graphs.begin(), settings_.gpu_graphs.end(), stem) == settings_.gpu_graphs.end()) {
                    accelerator = SCYLLASBAND_ACCELERATOR_CPU;
                }
            }
            model = std::make_shared<Model>(env_, path, accelerator, settings_);
        }
        std::string signature = artifact.get("signature").str("serving_default");
        try {
            return std::make_unique<LiteRtGraph>(model, signature, spec);
        } catch (const std::exception& error) {
            throw std::runtime_error("Cannot open " + component + ": " + error.what());
        }
    }

private:
    // LiteRT runs CPU graphs on one thread unless told otherwise; default to up to four.
    int threads() const {
        if (options_.threads > 0) return options_.threads;
        return static_cast<int>(std::max(1u, std::min(4u, std::thread::hardware_concurrency())));
    }

    BackendOptions options_;
    Settings settings_;
    bool configured_ = false;
    std::string library_dir_;
    LiteRtEnvironment env_ = nullptr;
    std::map<std::string, std::shared_ptr<Model>> models_;
};

}  // namespace

std::unique_ptr<Backend> make_backend(const BackendOptions& options) { return std::make_unique<LiteRtBackend>(options); }

}  // namespace scyllasband
