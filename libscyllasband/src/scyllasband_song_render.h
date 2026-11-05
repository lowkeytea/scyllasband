// Song rendering (see singing.py sing()): a planned song through the engine, phrase by phrase.
//
// Each phrase is rendered with the score's durations and melody, the delivery mode, the singing sampler, steps and
// guidance, no delivery request and no acoustic prefix, then decoded with the chosen decoder (BigVGAN, or Vocos for
// "turbo") and placed on the song's timeline, so the rests between phrases are silence.
#pragma once

#include "scyllasband_engine.h"
#include "scyllasband_song_plan.h"

#include <cstdint>
#include <functional>
#include <vector>

namespace scyllasband {

// The planner's view of an engine. The phrase limit is the smaller of the flow's largest bucket and every available
// decoder's, so a plan is the same whichever decoder renders it.
class EngineSongFrontend : public SongFrontend {
public:
    explicit EngineSongFrontend(Engine& engine) : engine_(engine) {}
    Phonemized phonemize(const std::string& text, const std::string& language) override { return engine_.phonemize(text, language); }
    std::string resolve_language(const std::string& voice, const std::string& language) const override {
        return engine_.resolve_language(voice, language);
    }
    const SingingControls& singing() const override { return engine_.singing(); }
    int sample_rate() const override { return engine_.sample_rate(); }
    int latent_hop() const override { return engine_.latent_hop(); }
    int64_t max_phrase_frames() const override;
    bool has_decoder(const std::string& decoder) const override { return engine_.has_decoder(decoder); }
    std::vector<std::string> available_decoders() const override { return engine_.available_decoders(); }

private:
    Engine& engine_;
};

struct SingEvent {
    enum Type { kPlan = 1, kPhraseStarted = 2, kAudio = 3, kDone = 4 };
    Type type;
    const PhrasePlan* phrase = nullptr;
    int phrase_count = 0;
    const std::vector<float>* audio = nullptr;  // the phrase's samples; metadata "offset_samples" places them on the song timeline
    Json metadata;
};

// Per-phrase flow noise for tests: (phrase index, seed, has seed, frames, out [2 + latent_dim, frames]; the first two rows unused).
using PhraseNoiseFn = std::function<void(int, uint64_t, bool, int64_t, float*)>;

// Plans and renders a score phrase by phrase; `emit` returns false to stop. Returns the song's metadata (the plan, the
// rendered phrases, the settings) with "samples": the song's length in samples.
Json sing_stream(Engine& engine, const Score& score, const SingRequest& request, const std::function<bool(const SingEvent&)>& emit,
                 const PhraseNoiseFn& phrase_noise = nullptr);

// The whole song as one waveform (phrases placed on the timeline, clipped to [-1, 1]).
std::vector<float> sing(Engine& engine, const Score& score, const SingRequest& request, Json* metadata = nullptr,
                        const PhraseNoiseFn& phrase_noise = nullptr);

// {"supported", "graph_contract", "styles", "decoders", "default_decoder", "sampler", "steps", "guidance",
// "voice_midi_center", "register_band", "max_phrase_seconds"} for scyllasband_voices_json.
Json singing_info(Engine& engine);

}  // namespace scyllasband
