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

Json describe_plan(const SongPlan& plan, const ResolvedSing& resolved, const SingRequest& request, const SingingControls& controls,
                   int sample_rate, bool with_melody) {
    Json out = plan.to_json(with_melody);
    out.set("samples", Json(static_cast<int64_t>(std::ceil(plan.seconds() * sample_rate))));
    out.set("decoder", Json(resolved.settings.decoder));
    out.set("steps", Json(resolved.settings.steps));
    out.set("sampler", Json(resolved.settings.sampler));
    out.set("guidance", Json(resolved.settings.guidance));
    out.set("fit", song_fit(controls, resolved.score, resolved.voice, request.plan.transpose, request.plan.use_register).to_json());
    return out;
}

SongPlan plan_request(Engine& engine, const Score& score, const SingRequest& request, ResolvedSing* resolved_out) {
    EngineSongFrontend frontend(engine);
    ResolvedSing resolved = resolve_sing_request(frontend, score, request);
    SongPlan plan = plan_song(resolved.score, frontend, resolved.voice, resolved.language, request.plan);
    if (resolved_out) *resolved_out = std::move(resolved);
    return plan;
}

Json sing_stream(Engine& engine, const Score& score, const SingRequest& request, const std::function<bool(const SingEvent&)>& emit,
                 const PhraseNoiseFn& phrase_noise) {
    ResolvedSing resolved;
    const SongPlan plan = plan_request(engine, score, request, &resolved);
    const std::string& voice = resolved.voice;
    const std::string& language = resolved.language;
    const SingSettings& settings = resolved.settings;
    const auto started = std::chrono::steady_clock::now();
    const int sample_rate = engine.sample_rate();
    const int phrase_count = static_cast<int>(plan.phrases.size());
    auto send = [&](SingEvent event) {
        event.phrase_count = phrase_count;
        if (!emit(event)) throw CancelledError();
    };
    const int64_t song_samples = static_cast<int64_t>(std::ceil(plan.seconds() * sample_rate));
    std::vector<int64_t> starts;
    for (const PhrasePlan& item : plan.phrases) starts.push_back(round_half_even(item.phrase.start * sample_rate));
    SongTimeline timeline(song_samples, starts);
    send(SingEvent{SingEvent::kPlan, nullptr, 0, nullptr, 0, describe_plan(plan, resolved, request, engine.singing(), sample_rate)});

    // No delivery request: sung clips were trained without intensity or pace, and the score sets the timing.
    DeliveryTensors delivery{};
    const std::size_t D = static_cast<std::size_t>(engine.latent_dim());
    std::random_device entropy;
    std::optional<double> first_audio_ms;
    Json::Array rendered;
    for (const PhrasePlan& item : plan.phrases) {
        Json started_json = item.to_json();
        send(SingEvent{SingEvent::kPhraseStarted, &item, 0, nullptr, 0, started_json});
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
        // The lead-in and the closing rest are gated (singing.py gate_phrase): stray breaths and syllables there are not in the score.
        double last_note = 0.0;
        for (const ScoreNote& note : item.phrase.notes) {
            if (note.midi) last_note = std::max(last_note, note.end);
        }
        const std::vector<float> audio = gate_phrase(engine.decode(result.latents, result.frames, decode), sample_rate,
                                                     item.phrase.notes.front().start, last_note);
        const int64_t offset = starts[static_cast<std::size_t>(item.index)];
        // The part of the phrase on the song's timeline (a lead-in before the song's start is cut).
        const int64_t placed = static_cast<int64_t>(audio.size()) - std::min<int64_t>(std::max<int64_t>(-offset, 0), static_cast<int64_t>(audio.size()));
        const int64_t start_sample = timeline.released();
        const std::vector<float> settled = timeline.add(audio);
        if (!first_audio_ms) first_audio_ms = elapsed_ms(started);
        Json phrase_json = item.to_json();
        phrase_json.set("bucket", Json(result.bucket));
        phrase_json.set("seconds", Json(round_to(static_cast<double>(placed) / sample_rate, 3)));
        phrase_json.set("seed", seed ? Json(static_cast<int64_t>(*seed)) : Json());
        rendered.push_back(phrase_json);
        Json metadata = phrase_json;
        metadata.set("start_sample", Json(start_sample));
        metadata.set("sample_count", Json(static_cast<int64_t>(settled.size())));
        metadata.set("song_samples", Json(song_samples));
        metadata.set("decoder", Json(settings.decoder));
        metadata.set("first_audio_ms", Json(round_to(*first_audio_ms, 1)));
        metadata.set("elapsed_ms", Json(round_to(elapsed_ms(started), 1)));
        send(SingEvent{SingEvent::kAudio, &item, 0, &settled, start_sample, metadata});
    }
    Json summary{Json::Object{}};
    summary.set("phrases", Json(phrase_count));
    summary.set("first_audio_ms", first_audio_ms ? Json(round_to(*first_audio_ms, 1)) : Json());
    summary.set("elapsed_ms", Json(round_to(elapsed_ms(started), 1)));
    summary.set("samples", Json(song_samples));
    summary.set("seconds", Json(round_to(plan.seconds(), 3)));
    send(SingEvent{SingEvent::kDone, nullptr, 0, nullptr, 0, summary});

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
    Json result = sing_stream(engine, score, request, [&](const SingEvent& event) {
        if (event.type == SingEvent::kAudio) song.insert(song.end(), event.audio->begin(), event.audio->end());
        return true;
    }, phrase_noise);
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
