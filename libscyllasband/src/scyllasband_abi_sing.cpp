// C ABI for singing (include/scyllasband.h, ABI 3.1.0): argument checks, locking and the JSON and audio handed back. Every
// rule about what is sung lives in scyllasband_score, scyllasband_song_plan and scyllasband_song_render.
#include "scyllasband_abi_internal.h"

#include "scyllasband_song_preview.h"
#include "scyllasband_song_render.h"

#include <cmath>
#include <cstdio>

using namespace scyllasband;
using namespace scyllasband::abi;

namespace {

void check_size(const ScyllasBandSingRequest* request) {
    require(request != nullptr, "request is required");
    require(request->struct_size == sizeof(ScyllasBandSingRequest),
            "request.struct_size does not match this library's ScyllasBandSingRequest: initialize requests with scyllasband_sing_request_init");
}

SingRequest sing_request_from(const ScyllasBandSingRequest* request) {
    check_size(request);
    require(request->voice_id != nullptr && request->voice_id[0] != '\0', "request.voice_id is required");
    require(request->score_json != nullptr && request->score_json[0] != '\0', "request.score_json is required");
    SingRequest out;
    out.voice = request->voice_id;
    out.language = request->language ? request->language : "";
    if (request->has_seed) out.seed = request->seed;
    out.steps = request->steps;
    switch (request->sampler) {
        case SCYLLASBAND_SAMPLER_DEFAULT: break;
        case SCYLLASBAND_SAMPLER_HEUN: out.sampler = "heun"; break;
        case SCYLLASBAND_SAMPLER_EULER: out.sampler = "euler"; break;
        default: throw std::invalid_argument("Unsupported sampler");
    }
    if (request->has_guidance) {
        require(std::isfinite(request->guidance), "request.guidance must be finite");
        out.guidance = static_cast<double>(request->guidance);
    }
    out.decoder = request->decoder ? request->decoder : "";
    out.plan.transpose = request->transpose;
    out.plan.use_register = request->use_register != 0;
    return out;
}

PhraseNoiseFn sing_test_noise(ScyllasBandRuntime* runtime) {
    if (runtime->test_noise == nullptr) return nullptr;
    const int dim = 2 + runtime->engine->latent_dim();
    return [runtime, dim](int index, uint64_t seed, bool has_seed, int64_t frames, float* out) {
        runtime->test_noise(index, seed, has_seed ? 1 : 0, dim, frames, out, runtime->test_noise_user);
    };
}

Json annotate(Engine& engine, Json metadata, const std::string& voice) {
    metadata.set("backend", Json(engine.backend_name()));
    metadata.set("accelerator", Json(engine.accelerator()));
    metadata.set("bundle_dir", Json(engine.bundle_dir()));
    metadata.set("voice_id", Json(voice));
    return metadata;
}

Json plan_json(Engine& engine, const ScyllasBandSingRequest* request, bool with_melody) {
    const SingRequest options = sing_request_from(request);
    ResolvedSing resolved;
    const SongPlan plan = plan_request(engine, parse_score_text(request->score_json), options, &resolved);
    return describe_plan(plan, resolved, options, engine.singing(), engine.sample_rate(), with_melody);
}

// The voice (stripped) after checking it exists; reads only the engine's configuration.
std::string known_voice(const Engine& engine, const char* voice_id) {
    require(voice_id != nullptr && voice_id[0] != '\0', "request.voice_id is required");
    std::string voice = voice_id;
    voice.erase(0, voice.find_first_not_of(" \t\r\n"));
    voice.erase(voice.find_last_not_of(" \t\r\n") + 1);
    engine.voice(voice);  // throws for an unknown voice
    return voice;
}

ScyllasBandEventType event_type(SingEvent::Type type) {
    switch (type) {
        case SingEvent::kPlan: return SCYLLASBAND_EVENT_PLAN;
        case SingEvent::kPhraseStarted: return SCYLLASBAND_EVENT_CHUNK_STARTED;
        case SingEvent::kAudio: return SCYLLASBAND_EVENT_AUDIO;
        case SingEvent::kDone: return SCYLLASBAND_EVENT_DONE;
    }
    throw std::logic_error("Unknown sing event");
}

}  // namespace

extern "C" {

void scyllasband_sing_request_init(ScyllasBandSingRequest* request) {
    if (request == nullptr) return;
    std::memset(request, 0, sizeof(*request));
    request->struct_size = sizeof(ScyllasBandSingRequest);
    request->use_register = 1;
}

ScyllasBandStatus scyllasband_score_parse(const char* score_json, int32_t transpose, char** out_json) {
    if (out_json != nullptr) *out_json = nullptr;
    return guarded([&] {
        require(out_json != nullptr, "out_json is required");
        require(score_json != nullptr, "score_json is required");
        try {
            *out_json = copy_string(score_summary(parse_score_text(score_json), transpose).dump());
        } catch (const ScoreError& error) {
            Json detail{Json::Object{}};
            detail.set("error", Json(error.what()));
            detail.set("note_index", error.note_index() >= 0 ? Json(error.note_index()) : Json());
            *out_json = copy_string(detail.dump());
            throw;
        }
    });
}

ScyllasBandStatus scyllasband_sing_fit_json(ScyllasBandRuntime* runtime, const ScyllasBandSingRequest* request, char** out_json) {
    return guarded([&] {
        require(runtime != nullptr && out_json != nullptr, "runtime and out_json are required");
        *out_json = nullptr;
        check_size(request);
        require(request->score_json != nullptr && request->score_json[0] != '\0', "request.score_json is required");
        const Engine& engine = *runtime->engine;   // configuration only: no lock
        require(engine.supports_singing(), "Singing needs a v4 bundle with delivery modes (controls.delivery_modes); this bundle has none");
        const std::string voice = known_voice(engine, request->voice_id);
        const SongFit fit = song_fit(engine.singing(), parse_score_text(request->score_json), voice, request->transpose, request->use_register != 0);
        *out_json = copy_string(fit.to_json().dump());
    });
}

ScyllasBandStatus scyllasband_sing_plan_json(ScyllasBandRuntime* runtime, const ScyllasBandSingRequest* request, char** out_json) {
    return guarded([&] {
        require(runtime != nullptr && out_json != nullptr, "runtime and out_json are required");
        *out_json = nullptr;
        std::lock_guard<std::mutex> lock(runtime->mutex);   // the G2P runs
        *out_json = copy_string(plan_json(*runtime->engine, request, false).dump());
    });
}

ScyllasBandStatus scyllasband_sing(ScyllasBandRuntime* runtime, const ScyllasBandSingRequest* request, ScyllasBandAudio* out_audio) {
    return guarded([&] {
        require(runtime != nullptr && out_audio != nullptr, "runtime and out_audio are required");
        *out_audio = ScyllasBandAudio{};
        const SingRequest options = sing_request_from(request);
        const Score score = parse_score_text(request->score_json);
        const uint64_t generation = runtime->engine->cancel_generation();
        std::lock_guard<std::mutex> lock(runtime->mutex);
        Engine& engine = *runtime->engine;
        engine.begin_request(generation);
        Json metadata;
        std::vector<float> samples = sing(engine, score, options, &metadata, sing_test_noise(runtime));
        metadata = annotate(engine, std::move(metadata), options.voice);
        out_audio->samples = copy_floats(samples);
        out_audio->sample_count = static_cast<int64_t>(samples.size());
        out_audio->sample_rate = engine.sample_rate();
        out_audio->metadata_json = copy_string(metadata.dump());
    });
}

ScyllasBandStatus scyllasband_sing_stream(ScyllasBandRuntime* runtime, const ScyllasBandSingRequest* request, ScyllasBandEventCallback callback,
                                          void* user_data) {
    return guarded([&] {
        require(runtime != nullptr && callback != nullptr, "runtime and callback are required");
        const SingRequest options = sing_request_from(request);
        const Score score = parse_score_text(request->score_json);
        const uint64_t generation = runtime->engine->cancel_generation();
        std::lock_guard<std::mutex> lock(runtime->mutex);
        Engine& engine = *runtime->engine;
        engine.begin_request(generation);
        sing_stream(engine, score, options, [&](const SingEvent& event) {
            const std::string metadata = event.metadata.dump();
            char chunk_id[32] = {0};
            if (event.phrase) std::snprintf(chunk_id, sizeof(chunk_id), "phrase_%04d", event.phrase->index + 1);
            ScyllasBandEvent out{};
            out.type = event_type(event.type);
            out.chunk_index = event.phrase ? event.phrase->index : -1;
            out.chunk_count = event.phrase_count;
            out.chunk_id = event.phrase ? chunk_id : nullptr;
            out.metadata_json = metadata.c_str();
            out.samples = event.audio ? event.audio->data() : nullptr;
            out.sample_count = event.audio ? static_cast<int64_t>(event.audio->size()) : 0;
            out.sample_rate = engine.sample_rate();
            return callback(&out, user_data) == 0;
        }, sing_test_noise(runtime));
    });
}

ScyllasBandStatus scyllasband_sing_preview(ScyllasBandRuntime* runtime, const ScyllasBandSingRequest* request, int32_t sample_rate,
                                           ScyllasBandAudio* out_audio) {
    return guarded([&] {
        require(out_audio != nullptr, "out_audio is required");
        *out_audio = ScyllasBandAudio{};
        check_size(request);
        require(request->score_json != nullptr && request->score_json[0] != '\0', "request.score_json is required");
        require(sample_rate >= 0, "sample_rate must be zero or positive");
        const Score score = parse_score_text(request->score_json).transposed(request->transpose);
        int shift = 0;
        if (runtime != nullptr && request->use_register) {   // configuration only: no lock
            const Engine& engine = *runtime->engine;
            const std::string voice = known_voice(engine, request->voice_id);
            auto center = engine.singing().voice_center.find(voice);
            shift = register_shift(score, center == engine.singing().voice_center.end() ? std::nullopt : std::optional<double>(center->second));
        }
        const int rate = sample_rate > 0 ? sample_rate : runtime != nullptr ? runtime->engine->sample_rate() : 24000;
        const Score sung = score.transposed(shift);
        const std::vector<float> samples = preview_tones(sung, rate);
        Json metadata{Json::Object{}};
        metadata.set("transpose", Json(request->transpose));
        metadata.set("register_shift", Json(shift));
        metadata.set("seconds", Json(sung.seconds()));
        out_audio->samples = copy_floats(samples);
        out_audio->sample_count = static_cast<int64_t>(samples.size());
        out_audio->sample_rate = rate;
        out_audio->metadata_json = copy_string(metadata.dump());
    });
}

ScyllasBandStatus scyllasband_warmup_singing(ScyllasBandRuntime* runtime, const char* voice_id, const char* decoder) {
    return guarded([&] {
        require(runtime != nullptr, "runtime is required");
        const uint64_t generation = runtime->engine->cancel_generation();
        std::lock_guard<std::mutex> lock(runtime->mutex);
        Engine& engine = *runtime->engine;
        require(engine.supports_singing(), "Singing needs a v4 bundle with delivery modes (controls.delivery_modes); this bundle has none");
        engine.begin_request(generation);
        SingRequest options;
        options.voice = voice_id && voice_id[0] ? voice_id : engine.voices().front().id;
        options.decoder = decoder ? decoder : "";
        options.steps = 1;
        options.sampler = "euler";
        options.seed = 0;
        const Score score = parse_score_text(R"({"bpm": 120, "notes": [["la", "C4", 1]]})");
        sing(engine, score, options);
    });
}

int32_t scyllasband_test_sing_plan(ScyllasBandRuntime* runtime, const ScyllasBandSingRequest* request, char** out_json) {
    return guarded([&] {
        require(runtime != nullptr && out_json != nullptr, "runtime and out_json are required");
        std::lock_guard<std::mutex> lock(runtime->mutex);
        *out_json = copy_string(plan_json(*runtime->engine, request, true).dump());
    });
}

}  // extern "C"
