#include "scyllasband_song_render.h"

#include "scyllasband_noise.h"
#include "scyllasband_numeric.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <optional>
#include <random>

namespace scyllasband {

namespace {

double elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace

int64_t EngineSongFrontend::max_phrase_frames() const {
    int64_t frames = engine_.max_frames();
    for (const std::string& decoder : engine_.available_decoders()) {
        const int64_t limit = engine_.decoder_max_frames(decoder);
        if (limit > 0) frames = std::min(frames, limit);
    }
    return frames;
}

Json sing_stream(Engine& engine, const Score& score, const SingRequest& request, const std::function<bool(const SingEvent&)>& emit,
                 const PhraseNoiseFn& phrase_noise) {
    EngineSongFrontend frontend(engine);
    const ResolvedSing resolved = resolve_sing_request(frontend, score, request);
    const std::string& voice = resolved.voice;
    const std::string& language = resolved.language;
    const SingSettings& settings = resolved.settings;
    const SongPlan plan = plan_song(resolved.score, frontend, voice, language, request.plan);
    const auto started = std::chrono::steady_clock::now();
    const int sample_rate = engine.sample_rate();
    const int phrase_count = static_cast<int>(plan.phrases.size());
    auto send = [&](SingEvent event) {
        event.phrase_count = phrase_count;
        if (!emit(event)) throw CancelledError();
    };
    const int64_t song_samples = static_cast<int64_t>(std::ceil(plan.seconds() * sample_rate));
    Json plan_json = plan.to_json();
    plan_json.set("samples", Json(song_samples));
    send(SingEvent{SingEvent::kPlan, nullptr, 0, nullptr, plan_json});

    // No delivery request: sung clips were trained without intensity or pace, and the score sets the timing.
    DeliveryTensors delivery{};
    const std::size_t D = static_cast<std::size_t>(engine.latent_dim());
    std::random_device entropy;
    std::optional<double> first_audio_ms;
    Json::Array rendered;
    for (const PhrasePlan& item : plan.phrases) {
        Json started_json = item.to_json();
        send(SingEvent{SingEvent::kPhraseStarted, &item, 0, nullptr, started_json});
        const std::optional<uint64_t> seed = request.seed ? std::optional<uint64_t>(*request.seed + static_cast<uint64_t>(item.index)) : std::nullopt;
        const uint64_t draw_seed = seed ? *seed : (static_cast<uint64_t>(entropy()) << 32) ^ entropy();
        NoiseFn noise = [&](int64_t frames, float* pitch, float* flow) {
            const std::size_t T = static_cast<std::size_t>(frames);
            std::fill_n(pitch, 2 * T, 0.0f);   // unused: the score gives the melody
            if (phrase_noise) {
                std::vector<float> all((2 + D) * T);
                phrase_noise(item.index, seed.value_or(0), seed.has_value(), frames, all.data());
                std::copy_n(all.begin() + static_cast<std::ptrdiff_t>(2 * T), D * T, flow);
            } else {
                gaussian_noise(draw_seed, static_cast<int64_t>(D * T), 1.0f, flow);
            }
        };
        const SungConditioning sung{&item.melody, plan.mode_id, settings.guidance};
        SentenceRequest sentence;
        sentence.voice = voice;
        sentence.language = language;
        sentence.delivery = delivery;
        sentence.steps = settings.steps;
        sentence.sampler = settings.sampler;
        sentence.noise = noise;
        sentence.durations = &item.durations;
        sentence.sung = &sung;
        SentenceResult result = engine.synthesize_sentence(item.sentence, sentence);
        DecodeRequest decode;
        decode.voice = voice;
        decode.language = language;
        decode.decoder = settings.decoder;
        std::vector<float> audio = engine.decode(result.latents, result.frames, decode);
        int64_t offset = round_half_even(item.phrase.start * sample_rate);
        if (offset < 0) {
            audio.erase(audio.begin(), audio.begin() + static_cast<std::ptrdiff_t>(std::min<int64_t>(-offset, static_cast<int64_t>(audio.size()))));
            offset = 0;
        }
        if (!first_audio_ms) first_audio_ms = elapsed_ms(started);
        Json phrase_json = item.to_json();
        phrase_json.set("bucket", Json(result.bucket));
        phrase_json.set("seconds", Json(round_to(static_cast<double>(audio.size()) / sample_rate, 3)));
        phrase_json.set("seed", seed ? Json(static_cast<int64_t>(*seed)) : Json());
        rendered.push_back(phrase_json);
        Json metadata = phrase_json;
        metadata.set("offset_samples", Json(offset));
        metadata.set("song_samples", Json(song_samples));
        metadata.set("decoder", Json(settings.decoder));
        metadata.set("first_audio_ms", Json(round_to(*first_audio_ms, 1)));
        metadata.set("elapsed_ms", Json(round_to(elapsed_ms(started), 1)));
        send(SingEvent{SingEvent::kAudio, &item, 0, &audio, metadata});
    }
    Json summary{Json::Object{}};
    summary.set("phrases", Json(phrase_count));
    summary.set("first_audio_ms", first_audio_ms ? Json(round_to(*first_audio_ms, 1)) : Json());
    summary.set("elapsed_ms", Json(round_to(elapsed_ms(started), 1)));
    summary.set("samples", Json(song_samples));
    summary.set("seconds", Json(round_to(plan.seconds(), 3)));
    send(SingEvent{SingEvent::kDone, nullptr, 0, nullptr, summary});

    Json metadata = plan.to_json();
    metadata.set("phrases", Json(std::move(rendered)));
    metadata.set("steps", Json(settings.steps));
    metadata.set("sampler", Json(settings.sampler));
    metadata.set("decoder", Json(settings.decoder));
    metadata.set("guidance", Json(settings.guidance));
    metadata.set("seed", request.seed ? Json(static_cast<int64_t>(*request.seed)) : Json());
    metadata.set("graph_contract", Json(engine.graph_contract()));
    metadata.set("samples", Json(song_samples));
    metadata.set("first_audio_ms", summary.get("first_audio_ms"));
    metadata.set("elapsed_ms", summary.get("elapsed_ms"));
    return metadata;
}

std::vector<float> sing(Engine& engine, const Score& score, const SingRequest& request, Json* metadata, const PhraseNoiseFn& phrase_noise) {
    std::vector<float> song;
    const int sample_rate = engine.sample_rate();
    Json result = sing_stream(engine, score, request, [&](const SingEvent& event) {
        if (event.type == SingEvent::kPlan) song.assign(static_cast<std::size_t>(event.metadata.get("samples").integer() + sample_rate / 10), 0.0f);
        if (event.type == SingEvent::kAudio) {
            const std::size_t offset = static_cast<std::size_t>(event.metadata.get("offset_samples").integer());
            const std::size_t count = offset < song.size() ? std::min(event.audio->size(), song.size() - offset) : 0;
            for (std::size_t i = 0; i < count; ++i) song[offset + i] += (*event.audio)[i];
        }
        return true;
    }, phrase_noise);
    song.resize(static_cast<std::size_t>(result.get("samples").integer()));
    for (float& sample : song) sample = std::clamp(sample, -1.0f, 1.0f);
    if (metadata) *metadata = std::move(result);
    return song;
}

Json singing_info(Engine& engine) {
    Json out{Json::Object{}};
    out.set("supported", Json(engine.supports_singing()));
    out.set("graph_contract", Json(engine.graph_contract()));
    const SingingControls& singing = engine.singing();
    out.set("styles", Json::array_of(engine.supports_singing() ? singing_styles(singing) : std::vector<std::string>{}));
    out.set("decoders", Json::array_of(engine.available_decoders()));
    if (engine.supports_singing()) {
        EngineSongFrontend frontend(engine);
        out.set("default_decoder", Json(engine.has_decoder(singing.decoder) ? singing.decoder : std::string("vocos")));
        out.set("sampler", Json(singing.sampler));
        out.set("steps", Json(singing.steps));
        out.set("guidance", Json(singing.guidance));
        Json centers{Json::Object{}};
        for (const Voice& voice : engine.voices()) {
            auto found = singing.voice_center.find(voice.id);
            if (found != singing.voice_center.end()) centers.set(voice.id, Json(found->second));
        }
        out.set("voice_midi_center", centers);
        Json band{Json::Array{}};
        band.push(Json(-kRegisterBelow));
        band.push(Json(kRegisterAbove));
        out.set("register_band", band);
        const double longest = static_cast<double>((frontend.max_phrase_frames() - 2) * engine.latent_hop()) / engine.sample_rate() - 2 * kPhrasePad;
        out.set("max_phrase_seconds", Json(round_to(std::min(kMaxPhraseSeconds, longest), 3)));
    }
    return out;
}

}  // namespace scyllasband
