// C ABI over the engine (include/scyllasband.h) and the parity-harness entry points (src/scyllasband_test_api.h).
#include "scyllasband.h"

#include "scyllasband_engine.h"
#include "scyllasband_test_api.h"
#include "scyllasband_text.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

using namespace scyllasband;

struct ScyllasBandRuntime {
    std::unique_ptr<Engine> engine;
    std::string voices_json;
    std::mutex mutex;
    ScyllasBandTestNoise test_noise = nullptr;
    void* test_noise_user = nullptr;
};

namespace {

thread_local std::string g_last_error;

template <typename Fn>
ScyllasBandStatus guarded(Fn&& fn) {
    try {
        fn();
        g_last_error.clear();
        return SCYLLASBAND_OK;
    } catch (const CancelledError& error) {
        g_last_error = error.what();
        return SCYLLASBAND_ERROR_CANCELLED;
    } catch (const std::invalid_argument& error) {
        g_last_error = error.what();
        return SCYLLASBAND_ERROR_INVALID_ARGUMENT;
    } catch (const std::exception& error) {
        g_last_error = error.what();
        return SCYLLASBAND_ERROR_RUNTIME;
    } catch (...) {
        g_last_error = "Unknown native error";
        return SCYLLASBAND_ERROR_RUNTIME;
    }
}

char* copy_string(const std::string& value) {
    char* out = static_cast<char*>(std::malloc(value.size() + 1));
    if (out == nullptr) throw std::bad_alloc();
    std::memcpy(out, value.c_str(), value.size() + 1);
    return out;
}

float* copy_floats(const std::vector<float>& values) {
    float* out = static_cast<float*>(std::malloc(std::max<std::size_t>(1, values.size()) * sizeof(float)));
    if (out == nullptr) throw std::bad_alloc();
    std::copy(values.begin(), values.end(), out);
    return out;
}

void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

SynthesisOptions options_from(const ScyllasBandRequest* request) {
    require(request != nullptr, "request is required");
    require(request->voice_id != nullptr && request->voice_id[0] != '\0', "request.voice_id is required");
    SynthesisOptions options;
    options.text = request->text ? request->text : "";
    options.voice = request->voice_id;
    options.language = request->language ? request->language : "";
    options.has_delivery = request->delivery != nullptr;
    options.delivery_spec = request->delivery ? request->delivery : "";
    options.speed = request->speed;
    options.steps = request->steps;
    switch (request->sampler) {
        case SCYLLASBAND_SAMPLER_DEFAULT: break;
        case SCYLLASBAND_SAMPLER_HEUN: options.sampler = "heun"; break;
        case SCYLLASBAND_SAMPLER_EULER: options.sampler = "euler"; break;
        default: throw std::invalid_argument("Unsupported sampler");
    }
    if (request->has_seed) options.seed = request->seed;
    options.temperature = request->temperature;
    options.normalize = request->normalize_text != 0;
    return options;
}

ChunkNoiseFn test_noise(ScyllasBandRuntime* runtime) {
    if (runtime->test_noise == nullptr) return nullptr;
    const int dim = runtime->engine->latent_dim();
    return [runtime, dim](int index, uint64_t seed, bool has_seed, int64_t frames, float* out) {
        runtime->test_noise(index, seed, has_seed ? 1 : 0, dim, frames, out, runtime->test_noise_user);
    };
}

Json annotate(Engine& engine, Json metadata, const SynthesisOptions& options, const std::string& language, const Json& delivery, int steps,
              const std::string& sampler) {
    metadata.set("backend", Json(engine.backend_name()));
    metadata.set("accelerator", Json(engine.accelerator()));
    metadata.set("bundle_dir", Json(engine.bundle_dir()));
    metadata.set("voice_id", Json(options.voice));
    metadata.set("language", Json(language));
    metadata.set("delivery", delivery);
    metadata.set("steps", Json(steps));
    metadata.set("sampler", Json(sampler));
    metadata.set("speed", Json(options.speed));
    metadata.set("seed", options.seed ? Json(static_cast<int64_t>(*options.seed)) : Json());
    return metadata;
}

std::vector<int64_t> int_list(const Json& value) {
    std::vector<int64_t> out;
    for (const Json& item : value.elements()) out.push_back(item.integer());
    return out;
}

}  // namespace

extern "C" {

const char* scyllasband_version(void) { return SCYLLASBAND_ABI_VERSION; }

const char* scyllasband_backend(void) {
#if defined(SCYLLASBAND_WITH_LITERT)
    return "litert";
#else
    return "onnx";
#endif
}

const char* scyllasband_last_error(void) { return g_last_error.c_str(); }

void scyllasband_request_init(ScyllasBandRequest* request) {
    if (request == nullptr) return;
    std::memset(request, 0, sizeof(*request));
    request->speed = 1.0f;
    request->temperature = 1.0f;
    request->normalize_text = 1;
}

ScyllasBandStatus scyllasband_runtime_create(const ScyllasBandRuntimeOptions* options, ScyllasBandRuntime** out_runtime) {
    return guarded([&] {
        require(out_runtime != nullptr, "out_runtime is required");
        *out_runtime = nullptr;
        require(options != nullptr && options->bundle_dir != nullptr && options->bundle_dir[0] != '\0', "options.bundle_dir is required");
        require(options->threads >= 0, "options.threads must be zero or positive");
        auto runtime = std::make_unique<ScyllasBandRuntime>();
        runtime->engine = std::make_unique<Engine>(BackendOptions{options->bundle_dir, options->threads, static_cast<int>(options->accelerator)});
        Engine& engine = *runtime->engine;
        Json info{Json::Object{}};
        info.set("backend", Json(engine.backend_name()));
        info.set("accelerator", Json(engine.accelerator()));
        info.set("sample_rate", Json(engine.sample_rate()));
        info.set("release_id", Json(engine.release()));
        info.set("default_voice", Json(engine.voices().front().id));
        Json::Array voices;
        for (const Voice& voice : engine.voices()) {
            Json item{Json::Object{}};
            item.set("id", Json(voice.id));
            item.set("languages", Json::array_of(voice.languages));
            item.set("default_language", Json(voice.default_language));
            voices.push_back(std::move(item));
        }
        info.set("voices", Json(std::move(voices)));
        runtime->voices_json = info.dump();
        *out_runtime = runtime.release();
    });
}

void scyllasband_runtime_destroy(ScyllasBandRuntime* runtime) { delete runtime; }

const char* scyllasband_voices_json(ScyllasBandRuntime* runtime) { return runtime == nullptr ? "{}" : runtime->voices_json.c_str(); }

ScyllasBandStatus scyllasband_warmup(ScyllasBandRuntime* runtime, const char* voice_id) {
    return guarded([&] {
        require(runtime != nullptr, "runtime is required");
        const uint64_t generation = runtime->engine->cancel_generation();
        std::lock_guard<std::mutex> lock(runtime->mutex);
        runtime->engine->begin_request(generation);
        SynthesisOptions options;
        options.text = "Hello.";
        options.voice = voice_id && voice_id[0] ? voice_id : runtime->engine->voices().front().id;
        options.steps = 1;
        options.sampler = "euler";
        options.seed = 0;
        synthesize_stream(*runtime->engine, options, [](const StreamEvent&) { return true; });
    });
}

ScyllasBandStatus scyllasband_plan_json(ScyllasBandRuntime* runtime, const ScyllasBandRequest* request, char** out_json) {
    return guarded([&] {
        require(runtime != nullptr && out_json != nullptr, "runtime and out_json are required");
        *out_json = nullptr;
        SynthesisOptions options = options_from(request);
        std::lock_guard<std::mutex> lock(runtime->mutex);
        *out_json = copy_string(plan_text(*runtime->engine, options).to_json().dump());
    });
}

ScyllasBandStatus scyllasband_synthesize(ScyllasBandRuntime* runtime, const ScyllasBandRequest* request, ScyllasBandAudio* out_audio) {
    return guarded([&] {
        require(runtime != nullptr && out_audio != nullptr, "runtime and out_audio are required");
        *out_audio = ScyllasBandAudio{};
        SynthesisOptions options = options_from(request);
        const uint64_t generation = runtime->engine->cancel_generation();
        std::lock_guard<std::mutex> lock(runtime->mutex);
        Engine& engine = *runtime->engine;
        engine.begin_request(generation);
        std::vector<float> samples;
        std::string language;
        Json delivery;
        Json summary = synthesize_stream(engine, options, [&](const StreamEvent& event) {
            if (event.type == StreamEvent::kAudio) samples.insert(samples.end(), event.audio->begin(), event.audio->end());
            if (event.type == StreamEvent::kPlan) {
                const Json& first = event.metadata.get("chunks")[0];
                language = first.get("language").str();
                delivery = first.get("delivery");
            }
            return true;
        }, test_noise(runtime));
        const int steps = options.steps > 0 ? options.steps : engine.default_steps();
        std::string sampler = options.sampler.empty() ? engine.default_sampler() : options.sampler;
        Json metadata = annotate(engine, std::move(summary), options, language, delivery, steps, sampler);
        out_audio->samples = copy_floats(samples);
        out_audio->sample_count = static_cast<int64_t>(samples.size());
        out_audio->sample_rate = engine.sample_rate();
        out_audio->metadata_json = copy_string(metadata.dump());
    });
}

ScyllasBandStatus scyllasband_synthesize_stream(ScyllasBandRuntime* runtime, const ScyllasBandRequest* request, ScyllasBandEventCallback callback,
                                                void* user_data) {
    return guarded([&] {
        require(runtime != nullptr && callback != nullptr, "runtime and callback are required");
        SynthesisOptions options = options_from(request);
        const uint64_t generation = runtime->engine->cancel_generation();
        std::lock_guard<std::mutex> lock(runtime->mutex);
        Engine& engine = *runtime->engine;
        engine.begin_request(generation);
        synthesize_stream(engine, options, [&](const StreamEvent& event) {
            const std::string metadata = event.metadata.dump();
            ScyllasBandEvent out{};
            out.type = static_cast<ScyllasBandEventType>(event.type);
            out.chunk_index = event.chunk ? event.chunk->index : -1;
            out.chunk_count = event.chunk_count;
            out.chunk_id = event.chunk ? event.chunk->chunk_id.c_str() : nullptr;
            out.metadata_json = metadata.c_str();
            out.samples = event.audio ? event.audio->data() : nullptr;
            out.sample_count = event.audio ? static_cast<int64_t>(event.audio->size()) : 0;
            out.sample_rate = engine.sample_rate();
            return callback(&out, user_data) == 0;
        }, test_noise(runtime));
    });
}

void scyllasband_cancel(ScyllasBandRuntime* runtime) {
    if (runtime != nullptr) runtime->engine->request_cancel();
}

void scyllasband_audio_free(ScyllasBandAudio* audio) {
    if (audio == nullptr) return;
    std::free(audio->samples);
    std::free(audio->metadata_json);
    *audio = ScyllasBandAudio{};
}

void scyllasband_string_free(char* value) { std::free(value); }

// --- parity harness entry points ------------------------------------------------------------------------------------

void scyllasband_test_free(void* pointer) { std::free(pointer); }

int32_t scyllasband_test_normalize(const char* text, const char* language, char** out_text) {
    return guarded([&] { *out_text = copy_string(normalize_spoken_text(text ? text : "", language ? language : "en_us")); });
}

int32_t scyllasband_test_split_sentences(const char* text, char** out_json) {
    return guarded([&] {
        Json::Array out;
        for (const Text& sentence : split_sentences(unicode::decode(text ? text : ""))) out.emplace_back(unicode::encode(sentence));
        *out_json = copy_string(Json(std::move(out)).dump());
    });
}

int32_t scyllasband_test_punctuated_segments(const char* text, int32_t max_chars, char** out_json) {
    return guarded([&] {
        Json::Array out;
        for (const auto& [segment, tokens] : punctuated_segments(unicode::decode(text ? text : ""), static_cast<std::size_t>(max_chars))) {
            out.push_back(Json(Json::Array{Json(unicode::encode(segment)), Json::array_of(tokens)}));
        }
        *out_json = copy_string(Json(std::move(out)).dump());
    });
}

int32_t scyllasband_test_unicode(const char* text, const char* operation, char** out_text) {
    return guarded([&] {
        unicode::Text value = unicode::decode(text ? text : "");
        std::string op = operation ? operation : "";
        require(op == "lower" || op == "nfc", "operation must be lower or nfc");
        *out_text = copy_string(unicode::encode(op == "lower" ? unicode::lower(value) : unicode::nfc(value)));
    });
}

int32_t scyllasband_test_phonemize(ScyllasBandRuntime* runtime, const char* text, const char* language, char** out_json) {
    return guarded([&] {
        require(runtime != nullptr && out_json != nullptr, "runtime and out_json are required");
        std::lock_guard<std::mutex> lock(runtime->mutex);
        Phonemized result = runtime->engine->phonemize(text ? text : "", language ? language : "en_us");
        Json out{Json::Object{}};
        out.set("phones", Json::array_of(result.phones));
        out.set("word_starts", Json::array_of(result.word_starts));
        out.set("word_boundary_candidates", Json::array_of(result.word_boundary_candidates));
        out.set("punctuation", result.punctuation);
        out.set("g2p_segments", Json::array_of(result.segments));
        *out_json = copy_string(out.dump());
    });
}

int32_t scyllasband_test_synthesize_sentence(ScyllasBandRuntime* runtime, const char* request_json, const float* noise, int64_t noise_frames,
                                             const float* prefix, int64_t prefix_frames, char** out_json, float** out_latents,
                                             int64_t* out_latent_count, float** out_audio, int64_t* out_sample_count) {
    return guarded([&] {
        require(runtime != nullptr && request_json != nullptr && out_json != nullptr, "runtime, request_json and out_json are required");
        const uint64_t generation = runtime->engine->cancel_generation();
        std::lock_guard<std::mutex> lock(runtime->mutex);
        Engine& engine = *runtime->engine;
        engine.begin_request(generation);
        Json request = Json::parse(request_json);
        Sentence sentence;
        for (const Json& phone : request.get("phones").elements()) sentence.phones.push_back(phone.str());
        for (int64_t index : int_list(request.get("word_starts"))) sentence.word_starts.push_back(static_cast<int>(index));
        sentence.before_ids = int_list(request.get("before_ids"));
        sentence.after_ids = int_list(request.get("after_ids"));
        const std::string voice = request.get("voice").str(), language = request.get("language").str();
        const std::string delivery_spec = request.get("delivery").str();
        DeliveryTensors delivery{};
        Delivery::parse(request.get("delivery").is_null() ? nullptr : delivery_spec.c_str()).tensors(delivery.values, delivery.present);
        const double speed = request.get("speed").num(1.0);
        Json out{Json::Object{}};
        std::vector<int> forced;
        if (request.get("durations").is_array()) {
            for (int64_t frames : int_list(request.get("durations"))) forced.push_back(static_cast<int>(frames));
        }
        if (noise == nullptr) {
            std::vector<int> durations = forced.empty()
                ? engine.durations(sentence, engine.voice_index(voice), engine.language_index(language), delivery, speed)
                : forced;
            out.set("durations", Json::array_of(durations));
            *out_json = copy_string(out.dump());
            return;
        }
        const std::size_t dim = static_cast<std::size_t>(engine.latent_dim());
        std::vector<float> prefix_values(prefix ? prefix : nullptr, prefix ? prefix + dim * static_cast<std::size_t>(prefix_frames) : nullptr);
        SentenceResult result = engine.synthesize_sentence(
            sentence, voice, language, delivery, prefix_values, prefix ? prefix_frames : 0, static_cast<int>(request.get("steps").integer(8)),
            request.get("sampler").str("heun"), speed,
            [&](int64_t frames, float* values) {
                if (frames != noise_frames) throw std::invalid_argument("noise has " + std::to_string(noise_frames) + " frames; the sentence needs " + std::to_string(frames));
                std::copy_n(noise, dim * static_cast<std::size_t>(frames), values);
            },
            forced.empty() ? nullptr : &forced);
        std::vector<float> audio = engine.decode(result.latents, result.frames, voice, language, prefix_values, prefix ? prefix_frames : 0);
        out = result.metadata;
        *out_json = copy_string(out.dump());
        if (out_latents) *out_latents = copy_floats(result.latents);
        if (out_latent_count) *out_latent_count = static_cast<int64_t>(result.latents.size());
        if (out_audio) *out_audio = copy_floats(audio);
        if (out_sample_count) *out_sample_count = static_cast<int64_t>(audio.size());
    });
}

void scyllasband_test_set_noise(ScyllasBandRuntime* runtime, ScyllasBandTestNoise noise, void* user_data) {
    if (runtime == nullptr) return;
    std::lock_guard<std::mutex> lock(runtime->mutex);
    runtime->test_noise = noise;
    runtime->test_noise_user = user_data;
}

}  // extern "C"
