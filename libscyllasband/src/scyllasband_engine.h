// Scylla's Band synthesis engine (see engine.py, planner.py and streaming.py).
//
// Within a chain, the plan's sentences are read as one stream of phones and spoken as targets: spans of that stream
// from one silence to another, sized by the predicted durations to stay within the lengths the model was trained on.
// For each target:
// 1. its phones form the target of a before/target/after span whose context is up to 180 phones (silences
//    excluded) of the neighbouring text of the same chain;
// 2. the duration graph predicts frames for every span position; the target slice is rounded to integer frames, with
//    pauses scaled by the requested pace (controls.pace_pauses);
// 3. per-frame phone ids and text events are built;
// 4. the pitch predictor integrates noise to a pitch and voicing contour; the pitch shift is added and the voice's soft
//    limit applied (controls.pitch), giving the flow's melody input;
// 5. the flow integrates noise to latents in the smallest fitting bucket, conditioned on the span, the melody and the
//    last 96 latent frames already spoken (right-aligned prefix);
// 6. the vocoder decodes the target with up to 48 frames of the preceding latents as left context, so
//    consecutive targets join as one continuous waveform.
// Delivery is intensity and pace (graph slots 0 and 1, standard deviations / z_scale) and a pitch shift in semitones; a
// target with fewer than min_spoken_phones spoken phones gets neither slot, as in training.
#pragma once

#include "scyllasband_events.h"
#include "scyllasband_g2p.h"
#include "scyllasband_graph.h"
#include "scyllasband_json.h"

#include <array>
#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace scyllasband {

extern const char* const kGraphContract;

// The sentence needs more latent frames (or phones) than the largest bucket holds.
class OverlongError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class CancelledError : public std::runtime_error {
public:
    CancelledError() : std::runtime_error("Synthesis was cancelled") {}
};

struct Voice {
    std::string id;
    std::vector<std::string> languages;
    std::string default_language;
};

struct Sentence {
    std::vector<std::string> phones;
    std::vector<int> word_starts;
    std::vector<int64_t> before_ids;  // preceding passage phones, silences excluded
    std::vector<int64_t> after_ids;   // following passage phones, silences excluded
};

struct DeliveryTensors {
    float values[5];
    uint8_t present[5];
    double pitch = 0.0;  // semitones added to the predicted pitch before the voice's soft limit
};

struct SentenceResult {
    std::vector<float> latents;  // [latent_dim, frames], row-major
    int64_t frames = 0;
    std::vector<int> durations;
    int64_t bucket = 0;
    Json metadata;
};

// Fills the initial noise of one target: `pitch` [2, frames] for the pitch predictor, `flow` [latent_dim, frames] for the flow
// (row-major).
using NoiseFn = std::function<void(int64_t frames, float* pitch, float* flow)>;

class Engine {
public:
    explicit Engine(const BackendOptions& options);
    Engine(const BackendOptions& options, std::unique_ptr<Backend> backend);  // graphs from `backend` (tests)
    ~Engine();

    const std::string& bundle_dir() const { return bundle_dir_; }
    const char* backend_name() const { return backend_->name(); }
    std::string accelerator() const { return backend_->accelerator(); }
    const Json& manifest() const { return manifest_; }
    const std::vector<Voice>& voices() const { return voices_; }
    int sample_rate() const { return sample_rate_; }
    int latent_dim() const { return latent_dim_; }
    int prefix_frames() const { return prefix_frames_; }
    int decode_context() const { return decode_context_; }
    std::size_t context_phones() const { return context_phones_; }
    // Target lengths the model was trained on, in latent frames; the streaming loop keeps every target it synthesizes within them.
    int64_t min_target_frames() const { return min_target_frames_; }
    int64_t max_target_frames() const { return max_target_frames_; }
    int default_steps() const { return default_steps_; }
    const std::string& default_sampler() const { return default_sampler_; }
    int pitch_steps() const { return pitch_steps_; }
    const std::string& pitch_sampler() const { return pitch_sampler_; }
    // The graph inputs and pitch shift for a request: what it leaves out takes the voice's default; `resolved` (optional)
    // receives {intensity, pace, pitch} as used.
    DeliveryTensors delivery_tensors(const std::string& voice, const Delivery& delivery, Json* resolved = nullptr) const;
    // {"ranges": {control: [low, high]}, "defaults": {voice: {control: value}}, "units": {...}}.
    Json delivery_info() const;
    // Pitch contour [2, frames] -> the flow's melody input [frames, 3] (pitch shifted and soft-limited, voicing, flag).
    std::vector<float> melody(const std::vector<float>& contour, int64_t frames, const std::string& voice, double shift) const;
    std::string release() const;
    std::vector<int64_t> bucket_frames() const;

    const Voice& voice(const std::string& id) const;
    std::string resolve_language(const std::string& voice, const std::string& language) const;  // empty: the voice default
    int64_t voice_index(const std::string& voice) const;
    int64_t language_index(const std::string& language) const;

    Phonemized phonemize(const std::string& text, const std::string& language);
    std::vector<int64_t> context_ids(const std::vector<std::string>& phones) const;

    std::vector<int> durations(const Sentence& sentence, int64_t voice, int64_t language, const DeliveryTensors& delivery, double speed);
    SentenceResult synthesize_sentence(const Sentence& sentence, const std::string& voice, const std::string& language,
                                       const DeliveryTensors& delivery, const std::vector<float>& prefix, int64_t prefix_frames, int steps,
                                       const std::string& sampler, double speed, const NoiseFn& noise,
                                       const std::vector<int>* forced_durations = nullptr);
    // Waveform for `latents` [latent_dim, frames], decoded after up to decode_context frames of `left` context.
    std::vector<float> decode(const std::vector<float>& latents, int64_t frames, const std::string& voice, const std::string& language,
                              const std::vector<float>& left, int64_t left_frames);

    // Cancellation: a request records cancel_generation() when it is issued and passes it to begin_request();
    // request_cancel() then stops every request issued before it, including one still waiting for the runtime.
    uint64_t cancel_generation() const { return cancel_generation_.load(); }
    void begin_request(uint64_t generation) { active_generation_ = generation; }
    void request_cancel() { cancel_generation_.fetch_add(1); }

private:
    struct Bucket {
        int64_t frames;
        std::string vector_estimator, vocoder, pitch_estimator;
        std::string vector_flow;  // optional: every flow step in one graph, for fused_sampler_/fused_steps_
        // Optional graphs without the prefix tokens, for targets that have no prefix (a chain's first target). Masked
        // prefix keys contribute nothing, so they return what the full graphs return there, for less work.
        std::string vector_estimator_first, vector_flow_first;
    };
    Graph& graph(const std::string& component);
    GraphOutput run(const std::string& component, const std::vector<TensorView>& inputs);
    const Bucket& bucket(int64_t frames) const;
    void validate() const;
    // The delivery graph inputs for a target: none set when it has fewer than min_spoken_phones spoken phones.
    void target_delivery(const std::vector<std::string>& phones, const DeliveryTensors& delivery, float values[5], uint8_t present[5]) const;
    double pause_scale(const DeliveryTensors& delivery) const;

    std::unique_ptr<Backend> backend_;  // declared first: graphs are destroyed before their backend
    std::string bundle_dir_;
    Json manifest_;
    std::map<std::string, int> phone_to_id_;
    std::vector<std::string> id_to_token_;
    std::map<std::string, int64_t> voice_to_id_, language_to_id_, bits_;
    std::vector<Voice> voices_;
    std::vector<Bucket> buckets_;
    std::vector<std::pair<int64_t, std::string>> g2p_buckets_;  // optional narrower G2P inputs (text tokens, component), smallest first
    // Optional narrower span graphs (phones, duration component, context encoder component), smallest first: padding is
    // masked and positions are absolute, so the narrowest graph that holds the span gives the same values.
    struct SpanBucket {
        int64_t phones;
        std::string duration, context, pitch_context;
    };
    std::vector<SpanBucket> span_buckets_;
    const SpanBucket* span_bucket(std::size_t phones) const;
    std::string fused_sampler_;
    int fused_steps_ = 0;
    std::size_t span_width_ = 512, context_phones_ = 180;
    int prefix_frames_ = 96, decode_context_ = 48, latent_dim_ = 24, latent_hop_ = 512, sample_rate_ = 24000;
    int default_steps_ = 8;
    int64_t min_target_frames_ = 64, max_target_frames_ = 420;
    std::string default_sampler_ = "heun";
    // Delivery (controls.delivery, controls.pace_pauses) and pitch (controls.pitch).
    double z_scale_ = 5.0, z_limit_ = 2.5, pace_pause_per_sd_ = 0.0;
    int min_spoken_phones_ = 4;
    double ranges_[3][2] = {{-2.0, 2.0}, {-2.0, 2.0}, {-2.0, 2.0}};
    std::map<std::string, std::array<double, 3>> voice_defaults_;
    double pitch_center_ = 10.0, pitch_scale_ = 6.0, melody_center_ = 16.65, melody_scale_ = 24.0, voiced_threshold_ = 0.5, knee_ = 2.0;
    std::map<std::string, std::pair<double, double>> pitch_edges_;
    std::string pitch_context_ = "pitch_context_encoder", pitch_sampler_ = "euler";
    int pitch_steps_ = 4;
    std::unique_ptr<G2P> g2p_;
    std::mutex graphs_mutex_;
    std::map<std::string, std::unique_ptr<Graph>> graphs_;
    std::atomic<uint64_t> cancel_generation_{0};
    uint64_t active_generation_ = 0;
};

// --- planning and streaming ---------------------------------------------------------------------------------------

struct PlanChunk {
    std::string chunk_id;
    int index = 0;
    int record_index = 0;
    std::string chain_id, voice, language;
    Delivery delivery;
    std::string text, source_text;
    bool paragraph_end = false;
    Json to_json() const;
};

struct Plan {
    std::vector<PlanChunk> chunks;
    Json to_json() const;
};

struct SynthesisOptions {
    std::string text, voice, language, delivery_spec;
    bool has_delivery = false;
    double speed = 1.0;
    int steps = 0;           // 0: bundle default
    std::string sampler;     // empty: bundle default
    std::optional<uint64_t> seed;
    double temperature = 1.0;
    bool normalize = true;
};

Plan plan_text(Engine& engine, const SynthesisOptions& options);

struct StreamEvent {
    enum Type { kPlan = 1, kChunkStarted = 2, kAudio = 3, kDone = 4 };
    Type type;
    const PlanChunk* chunk = nullptr;
    int chunk_count = 0;
    const std::vector<float>* audio = nullptr;
    Json metadata;
};

// Per-chunk noise source for tests: (chunk index, seed, has seed, frames, out [2 + latent_dim, frames]: the pitch predictor's
// two rows, then the flow's).
using ChunkNoiseFn = std::function<void(int, uint64_t, bool, int64_t, float*)>;

// Plans `options.text` and speaks it target by target; `emit` returns false to stop. Returns the final metadata.
Json synthesize_stream(Engine& engine, const SynthesisOptions& options, const std::function<bool(const StreamEvent&)>& emit,
                       const ChunkNoiseFn& chunk_noise = nullptr);

// Speaks `plan` target by target (the text fields of `options` are unused).
Json synthesize_plan_stream(Engine& engine, const Plan& plan, const SynthesisOptions& options, const std::function<bool(const StreamEvent&)>& emit,
                            const ChunkNoiseFn& chunk_noise = nullptr);

// Silences that can end a target: the end of the chain, a paragraph or record boundary, and the silences after a
// sentence end, after a clause pause and between words.
enum class TargetCut { kEnd, kParagraph, kSentence, kClause, kWord };

struct TargetCandidate {
    std::size_t index;  // the silence's position in the chain's phone stream
    TargetCut kind;
    int64_t frames;     // predicted frames from the target's start through this silence
};

// The end of a target among `candidates` (in stream order): the first target of a request ends at the earliest
// sentence end past the trained minimum (a clause, else a word break, only when the opening sentence does not fit); later targets end at the chain's end or a paragraph if it fits,
// else reach toward the trained maximum, preferring a sentence boundary, then a clause, then a word break; a remainder
// shorter than the minimum is never left before the end or a paragraph.
std::size_t choose_target(const std::vector<TargetCandidate>& candidates, bool first, int64_t shortest, int64_t longest);

// Normal noise from a seeded xoshiro256** generator (Box-Muller), scaled by `temperature`.
void gaussian_noise(uint64_t seed, int64_t count, float temperature, float* out);

}  // namespace scyllasband
