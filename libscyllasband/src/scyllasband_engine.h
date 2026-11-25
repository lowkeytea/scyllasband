// Scylla's Band synthesis engine (see engine.py): the bundle (manifest, contract, assets) and its graphs.
//
// For one target (a span of phones from one silence to another, with its context; built by the speech stream,
// scyllasband_speech_stream.h, or the song planner):
// 1. its phones form the target of a before/target/after span whose context is up to 180 phones (silences
//    excluded) of the neighbouring text;
// 2. the duration graph predicts frames for every span position; the target slice is rounded to integer frames, with
//    pauses scaled by the requested pace (controls.pace_pauses);
// 3. per-frame phone ids and text events are built;
// 4. the pitch predictor integrates noise to a pitch and voicing contour; the pitch shift is added and the voice's soft
//    limit applied (controls.pitch), giving the flow's melody input;
// 5. the flow integrates noise to latents in the smallest fitting bucket, conditioned on the span, the melody and the
//    last 96 latent frames already spoken (right-aligned prefix);
// 6. a decoder turns latents into a waveform, with up to 48 frames of the preceding latents as left context.
// Delivery is intensity and pace (graph slots 0 and 1, standard deviations / z_scale) and a pitch shift in semitones; a
// target with fewer than min_spoken_phones spoken phones gets neither slot, as in training.
//
// A v5 bundle (scyllasband_measured_controls_v5, also written ..._v5_1) adds each voice's identity vector as a graph input and
// a table of style rows (controls.delivery_modes.ids): the duration, pitch and flow graphs all take the target's row. Speech
// uses the rows the duration and pitch graphs were trained on (duration_styles_trained: spoken, and rap, which is spoken with
// intensity and pace unset); the other rows are sung from a score. The pitch predictor continues from the contour rendered so
// far, pauses after punctuation carry the sounded-pause mark, and an opening "Uh-uh" keeps minimum lengths.
//
// A v4 bundle (graph contract scyllasband_measured_controls_v4) adds singing: the flow also takes a delivery mode id (0
// spoken, 1 sung, 2 rap, 3 ballad; controls.delivery_modes), a sung target brings its own melody from a score, which
// replaces steps 2 and 4 (durations come from the score, the pitch predictor and its soft limit are skipped), and the
// bundle may carry a second decoder (controls.decoders, e.g. BigVGAN beside Vocos). Speech on a v4 bundle feeds mode 0
// and is otherwise unchanged; a v3 bundle never sees the mode input.
//
// The engine knows nothing about plans, chains, scores or phrases; it is safe to read its configuration (voices,
// buckets, decoders, singing settings) from any thread once constructed, while graph calls are serialized by the caller.
#pragma once

#include "scyllasband_errors.h"
#include "scyllasband_events.h"
#include "scyllasband_g2p.h"
#include "scyllasband_graph.h"
#include "scyllasband_json.h"
#include "scyllasband_singing_controls.h"

#include <array>
#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace scyllasband {

extern const char* const kGraphContract;    // v3: speech
extern const char* const kGraphContractV4;  // v4: speech and singing
extern const char* const kGraphContractV5;  // v5: identity vectors, style rows, a pitch-contour prefix and pause marks
extern const char* const kGraphContractV5_1;  // the same contract under its other name
bool is_v5_contract(const std::string& contract);

struct Voice {
    std::string id;
    std::vector<std::string> languages;
    std::string default_language;
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
    // [2, frames]: the pitch contour as rendered (soft-limited inside the voice's edges, before any pitch shift; the voicing
    // channel as sampled). Empty for a sung target. A v5 bundle's next target continues from it (SentenceRequest::pitch_prefix).
    std::vector<float> contour;
};

// A waveform decoder (controls.decoders): the bucket key naming its graphs and its output length rule
// ("latent_frames * 512 - 256" for Vocos, "latent_frames * 512" for BigVGAN).
struct DecoderSpec {
    std::string name, component;
    int samples_per_frame = 512;
    int short_samples = 0;  // samples the output falls short of frames * samples_per_frame
    bool optional = false;  // controls.decoders.<name>.required is false: a bundle may leave its graphs out
    // Windowed decoding (0: off): every decode runs through the one bucket graph of window_frames, decoding window_context
    // frames either side of the frames each call keeps. On LiteRT each bucket graph a process opens keeps its own working
    // memory (BigVGAN: about 2 MB per frame), so one bucket bounds it whatever the phrase lengths. From the manifest's
    // controls.decoders.<name>.window {frames, context}, or SCYLLASBAND_DECODER_WINDOW ("bigvgan=256:32"; "bigvgan=0": off).
    int64_t window_frames = 0, window_context = 0;
};

// A target sung from a score: its melody input [frames, 3] replaces the pitch predictor's contour, `mode` is fed to the
// flow, and `guidance` > 1 contrasts the melody and mode against both switched off (classifier-free guidance: off = no melody
// and the spoken row). `guidance_until`: the flow time (0 to 1) from which evaluations run unguided; none guides them all.
struct SungConditioning {
    const std::vector<float>* melody = nullptr;
    int64_t mode = 0;
    double guidance = 1.0;
    std::optional<double> guidance_until;
};

// Fills the initial noise of one target: `pitch` [2, frames] for the pitch predictor, `flow` [latent_dim, frames] for the flow
// (row-major).
using NoiseFn = std::function<void(int64_t frames, float* pitch, float* flow)>;

// How one target is synthesized (its phones and context are the Sentence).
struct SentenceRequest {
    std::string voice, language;
    DeliveryTensors delivery{};
    const std::vector<float>* prefix = nullptr;  // [latent_dim, prefix_frames]: the latents already spoken before the target
    int64_t prefix_frames = 0;
    int steps = 0;                               // flow steps (> 0)
    std::string sampler;                         // heun or euler
    double speed = 1.0;
    NoiseFn noise;
    const std::vector<int>* durations = nullptr;  // phone durations to use instead of the duration graph's
    const SungConditioning* sung = nullptr;       // a melody from a score in place of the pitch predictor (v4 and v5 bundles)
    // v5 speech: the style row fed to the duration, pitch and flow graphs (Engine::speech_style); the spoken row when unset.
    std::optional<int64_t> mode;
    const std::vector<float>* pitch_prefix = nullptr;  // v5: [2, pitch_prefix_frames] contours rendered so far in this chain
    int64_t pitch_prefix_frames = 0;
};

// How latents are decoded to a waveform.
struct DecodeRequest {
    std::string voice, language;
    const std::vector<float>* left = nullptr;  // [latent_dim, left_frames]: preceding latents, up to decode_context of them used
    int64_t left_frames = 0;
    std::string decoder = "vocos";             // vocos, which every bundle has, or bigvgan where the bundle carries it
};

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
    int latent_hop() const { return latent_hop_; }
    int64_t max_frames() const { return buckets_.back().frames; }
    const std::string& graph_contract() const { return graph_contract_; }
    bool feeds_delivery_mode() const { return feeds_delivery_mode_; }
    bool v5() const { return v5_; }
    // The spoken row of the style table (0 when the bundle has no table).
    int64_t spoken_mode() const { return spoken_mode_; }
    // The styles speech may ask for: the rows the duration and pitch graphs were trained on (v5); spoken alone before.
    const std::vector<std::string>& speech_styles() const { return speech_styles_; }
    // (the style's row, whether intensity and pace are set) for speech in `style` ("" = spoken). Only the styles of
    // speech_styles() can be spoken; a style trained with neither intensity nor pace (controls.delivery_modes.controls_absent,
    // the score-free rap rows) takes neither.
    std::pair<int64_t, bool> speech_style(const std::string& style) const;
    // v5: a target never holds fewer phones than this while the passage has a neighbour to join (0: no such rule).
    int64_t min_target_phones() const { return min_target_phones_; }
    // v5: frames of rendered contour the pitch predictor takes as its prefix (0: none), and the sounded-pause mark the
    // streaming loop puts on every punctuation pause inside a passage (empty: no marks).
    int pitch_prefix_width() const { return pitch_prefix_width_; }
    const std::string& pause_mark() const { return pause_mark_; }
    // controls.delivery_modes.ids when enabled (empty on v3 bundles and v4 bundles without singing).
    const std::map<std::string, int64_t>& delivery_modes() const { return singing_.mode_ids; }
    const SingingControls& singing() const { return singing_; }
    bool supports_singing() const { return singing_.enabled(); }
    const std::map<std::string, DecoderSpec>& decoders() const { return decoders_; }
    // Whether the bundle has graphs for `decoder` (vocos always; bigvgan where exported for this backend).
    bool has_decoder(const std::string& decoder) const;
    std::vector<std::string> available_decoders() const;  // sorted
    // The largest bucket with a graph for `decoder` (0 when this backend has none).
    int64_t decoder_max_frames(const std::string& decoder) const;
    const std::map<std::string, std::pair<double, double>>& pitch_edges() const { return pitch_edges_; }
    double melody_center() const { return melody_center_; }
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

    Phonemized phonemize(const std::string& text, const std::string& language, bool nonverbal = true);
    std::vector<int64_t> context_ids(const std::vector<std::string>& phones) const;

    // Frames per phone of the target. `style` (v5): the style row fed to the duration graph; the spoken row by default.
    std::vector<int> durations(const Sentence& sentence, int64_t voice, int64_t language, const DeliveryTensors& delivery, double speed,
                               std::optional<int64_t> style = std::nullopt);
    // Latents for one target. With `request.sung`, the score's melody (and mode and guidance) replaces the pitch predictor,
    // and the noise function's pitch rows are unused.
    SentenceResult synthesize_sentence(const Sentence& sentence, const SentenceRequest& request);
    // Waveform for `latents` [latent_dim, frames], decoded after up to decode_context frames of left context.
    std::vector<float> decode(const std::vector<float>& latents, int64_t frames, const DecodeRequest& request);

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
    void validate_v5() const;
    void load_decoders();
    void load_decoder_windows(const Json& declared);
    // The smallest bucket of at least `frames` with a graph for `decoder`, and that graph.
    std::pair<int64_t, std::string> decoder_bucket(const std::string& decoder, int64_t frames) const;
    // decode() for a decoder with a window: fixed windows through one bucket graph, hard cuts between the frames each keeps.
    std::vector<float> decode_windowed(const std::vector<float>& latents, int64_t frames, const DecodeRequest& request, const DecoderSpec& spec);
    // The delivery graph inputs for a target: none set when it has fewer than min_spoken_phones spoken phones.
    void target_delivery(const std::vector<std::string>& phones, const DeliveryTensors& delivery, float values[5], uint8_t present[5]) const;
    double pause_scale(const DeliveryTensors& delivery) const;
    const std::vector<float>& identity(int64_t voice_row) const;  // v5: the voice's fixed identity vector

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
    // v4: delivery modes, singing settings and decoders.
    std::string graph_contract_;
    bool feeds_delivery_mode_ = false;
    bool v5_ = false;
    int64_t spoken_mode_ = 0;
    std::vector<std::string> speech_styles_ = {"spoken"};
    std::set<std::string> uncontrolled_styles_;  // speech styles trained with the intensity and pace slots absent
    int64_t min_target_phones_ = 0;
    std::map<int64_t, std::vector<float>> identity_;  // voice row -> its fixed identity vector (assets.voice_identity)
    int64_t identity_dim_ = 0;
    int pitch_prefix_width_ = 0;
    std::string pause_mark_;
    SingingControls singing_;
    std::map<std::string, DecoderSpec> decoders_;
    std::map<std::string, std::vector<std::pair<int64_t, std::string>>> decoder_graphs_;  // decoder -> (frames, component), smallest first
    std::unique_ptr<G2P> g2p_;
    std::mutex graphs_mutex_;
    std::map<std::string, std::unique_ptr<Graph>> graphs_;
    std::atomic<uint64_t> cancel_generation_{0};
    uint64_t active_generation_ = 0;
};

}  // namespace scyllasband
