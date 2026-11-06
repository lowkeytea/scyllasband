// C ABI over the engine (include/scyllasband.h) and the parity-harness entry points (src/scyllasband_test_api.h).
#include "scyllasband_abi_internal.h"
#include "scyllasband_song_render.h"
#include "scyllasband_speech_stream.h"
#include "scyllasband_text.h"

#include <cmath>
#include <cstdio>

using namespace scyllasband;
using namespace scyllasband::abi;

std::string& scyllasband::abi::last_error() {
    thread_local std::string message;
    return message;
}

namespace {

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

// A records request's settings (its text fields stay empty) and its records.
SynthesisOptions records_options_from(const ScyllasBandRecordsRequest* request, Json* records) {
    require(request != nullptr, "request is required");
    require(request->struct_size == sizeof(ScyllasBandRecordsRequest),
            "request.struct_size does not match this library's ScyllasBandRecordsRequest: initialize requests with scyllasband_records_request_init");
    require(request->records_json != nullptr && request->records_json[0] != '\0', "request.records_json is required");
    try {
        *records = Json::parse(request->records_json);
    } catch (const std::exception& error) {
        throw std::invalid_argument(std::string("request.records_json is not valid JSON: ") + error.what());
    }
    SynthesisOptions options;
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
    options.pause_ms = request->pause_ms;
    return options;
}

// A speech stream event as the C API passes it (valid during the callback); false stops the stream.
bool forward_event(const StreamEvent& event, int sample_rate, ScyllasBandEventCallback callback, void* user_data) {
    const std::string metadata = event.metadata.dump();
    ScyllasBandEvent out{};
    out.type = static_cast<ScyllasBandEventType>(event.type);
    out.chunk_index = event.chunk ? event.chunk->index : -1;
    out.chunk_count = event.chunk_count;
    out.chunk_id = event.chunk ? event.chunk->chunk_id.c_str() : nullptr;
    out.metadata_json = metadata.c_str();
    out.samples = event.audio ? event.audio->data() : nullptr;
    out.sample_count = event.audio ? static_cast<int64_t>(event.audio->size()) : 0;
    out.sample_rate = sample_rate;
    return callback(&out, user_data) == 0;
}

ChunkNoiseFn test_noise(ScyllasBandRuntime* runtime) {
    if (runtime->test_noise == nullptr) return nullptr;
    const int dim = 2 + runtime->engine->latent_dim();   // the pitch predictor's two rows, then the flow's
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
#elif defined(SCYLLASBAND_WITH_APPLE)
    return "apple";
#else
    return "onnx";
#endif
}

const char* scyllasband_last_error(void) { return last_error().c_str(); }

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
        info.set("delivery", engine.delivery_info());
        info.set("graph_contract", Json(engine.graph_contract()));
        info.set("singing", singing_info(engine));
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
            if (event.type == StreamEvent::kPlan) language = event.metadata.get("chunks")[0].get("language").str();
            if (event.type == StreamEvent::kAudio && delivery.is_null()) delivery = event.metadata.get("delivery");   // as used
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
            return forward_event(event, engine.sample_rate(), callback, user_data);
        }, test_noise(runtime));
    });
}

void scyllasband_records_request_init(ScyllasBandRecordsRequest* request) {
    if (request == nullptr) return;
    std::memset(request, 0, sizeof(*request));
    request->struct_size = sizeof(ScyllasBandRecordsRequest);
    request->speed = 1.0f;
    request->temperature = 1.0f;
    request->normalize_text = 1;
}

ScyllasBandStatus scyllasband_plan_records_json(ScyllasBandRuntime* runtime, const ScyllasBandRecordsRequest* request, char** out_json) {
    return guarded([&] {
        require(runtime != nullptr && out_json != nullptr, "runtime and out_json are required");
        *out_json = nullptr;
        Json records;
        const SynthesisOptions options = records_options_from(request, &records);
        std::lock_guard<std::mutex> lock(runtime->mutex);
        *out_json = copy_string(plan_records(*runtime->engine, records, options.normalize).to_json().dump());
    });
}

ScyllasBandStatus scyllasband_synthesize_records_stream(ScyllasBandRuntime* runtime, const ScyllasBandRecordsRequest* request,
                                                        ScyllasBandEventCallback callback, void* user_data) {
    return guarded([&] {
        require(runtime != nullptr && callback != nullptr, "runtime and callback are required");
        Json records;
        const SynthesisOptions options = records_options_from(request, &records);
        const uint64_t generation = runtime->engine->cancel_generation();
        std::lock_guard<std::mutex> lock(runtime->mutex);
        Engine& engine = *runtime->engine;
        engine.begin_request(generation);
        const Plan plan = plan_records(engine, records, options.normalize);
        synthesize_plan_stream(engine, plan, options, [&](const StreamEvent& event) {
            return forward_event(event, engine.sample_rate(), callback, user_data);
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
        // A sung target (a melody given) has no delivery request: sung clips were trained without intensity or pace.
        const DeliveryTensors delivery = request.get("melody").is_array()
            ? DeliveryTensors{}
            : engine.delivery_tensors(voice, Delivery::parse(request.get("delivery").is_null() ? nullptr : delivery_spec.c_str()));
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
        std::vector<float> melody;
        for (const Json& value : request.get("melody").elements()) melody.push_back(static_cast<float>(value.num()));
        const SungConditioning sung{request.get("melody").is_array() ? &melody : nullptr, request.get("delivery_mode").integer(0),
                                    request.get("guidance").num(1.0)};
        SentenceRequest sentence_request;
        sentence_request.voice = voice;
        sentence_request.language = language;
        sentence_request.delivery = delivery;
        sentence_request.prefix = &prefix_values;
        sentence_request.prefix_frames = prefix ? prefix_frames : 0;
        sentence_request.steps = static_cast<int>(request.get("steps").integer(8));
        sentence_request.sampler = request.get("sampler").str("heun");
        sentence_request.speed = speed;
        sentence_request.noise = [&](int64_t frames, float* pitch, float* flow) {   // noise: [2 + latent_dim, frames], the pitch rows first
            if (frames != noise_frames) throw std::invalid_argument("noise has " + std::to_string(noise_frames) + " frames; the sentence needs " + std::to_string(frames));
            const std::size_t T = static_cast<std::size_t>(frames);
            std::copy_n(noise, 2 * T, pitch);
            std::copy_n(noise + 2 * T, dim * T, flow);
        };
        sentence_request.durations = forced.empty() ? nullptr : &forced;
        sentence_request.sung = &sung;
        SentenceResult result = engine.synthesize_sentence(sentence, sentence_request);
        DecodeRequest decode;
        decode.voice = voice;
        decode.language = language;
        decode.left = &prefix_values;
        decode.left_frames = prefix ? prefix_frames : 0;
        decode.decoder = request.get("decoder").str("vocos");
        std::vector<float> audio = engine.decode(result.latents, result.frames, decode);
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
