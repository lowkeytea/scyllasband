#include "scyllasband_engine.h"

#include "scyllasband_text.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <random>

namespace scyllasband {

const char* const kGraphContract = "scyllasband_measured_delivery_v2";

namespace {

const char* const kContractVersion = "1.0.0";
const std::vector<std::string> kDurationInputs = {"span_phone_ids", "span_segment_ids", "span_mask", "voice_id", "language_id",
                                                  "delivery_values", "delivery_present"};
const std::vector<std::string> kContextInputs = {"span_context_phone_ids", "span_context_segment_ids", "span_context_mask"};
const std::vector<std::string> kVectorInputs = {"noise", "time", "expanded_phone_ids", "voice_id", "language_id", "delivery_values",
                                                "delivery_present", "latent_mask", "span_context_hidden", "prefix_latents", "prefix_mask",
                                                "expanded_boundary_event_ids", "expanded_modifier_event_ids", "expanded_phone_phase",
                                                "expanded_phone_log_duration", "expanded_sentence_type_ids"};
const std::vector<std::string> kVocoderInputs = {"latents", "latent_mask", "voice_id", "language_id", "emotion_id"};
const std::vector<std::string> kRequiredAssets = {"phone_vocab", "voice_index", "language_index", "g2p_config", "g2p_tokenizer", "g2p_language_map"};

std::map<std::string, int64_t> load_index(const std::string& path) {
    std::map<std::string, int64_t> out;
    const Json rows = Json::parse_file(path);
    for (const Json& row : rows.elements()) out[row.get("id").str()] = row.get("index").integer();
    return out;
}

template <typename T>
std::vector<T> pad(const std::vector<T>& values, std::size_t width, T fill = T{}) {
    std::vector<T> out(width, fill);
    std::copy_n(values.begin(), std::min(values.size(), width), out.begin());
    return out;
}

TensorView view(DType type, std::vector<int64_t> shape, const void* data) { return TensorView{type, std::move(shape), data}; }

double elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

double round1(double value) { return std::nearbyint(value * 10.0) / 10.0; }

uint64_t splitmix64(uint64_t& state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

}  // namespace

void gaussian_noise(uint64_t seed, int64_t count, float temperature, float* out) {
    uint64_t s[4];
    uint64_t state = seed;
    for (uint64_t& word : s) word = splitmix64(state);
    auto next = [&s]() {
        auto rotl = [](uint64_t x, int k) { return (x << k) | (x >> (64 - k)); };
        const uint64_t result = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return result;
    };
    auto uniform = [&]() { return (static_cast<double>(next() >> 11) + 1.0) * (1.0 / 9007199254740992.0); };  // (0, 1]
    const double two_pi = 6.283185307179586;
    for (int64_t i = 0; i < count; i += 2) {
        const double radius = std::sqrt(-2.0 * std::log(uniform())), angle = two_pi * uniform();
        out[i] = static_cast<float>(radius * std::cos(angle)) * temperature;
        if (i + 1 < count) out[i + 1] = static_cast<float>(radius * std::sin(angle)) * temperature;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Engine

Engine::Engine(const BackendOptions& options) : backend_(make_backend(options)), bundle_dir_(options.bundle_dir) {
    const std::filesystem::path manifest_path = std::filesystem::path(bundle_dir_) / "manifest.json";
    if (!std::filesystem::is_regular_file(manifest_path)) throw std::invalid_argument("No manifest.json in " + bundle_dir_);
    manifest_ = Json::parse_file(manifest_path.string());
    validate();
    const Json& assets = manifest_.get("assets");
    const Json& controls = manifest_.get("controls");
    auto asset = [&](const std::string& key) { return bundle_dir_ + "/" + assets.get(key).str(); };

    Json vocab = Json::parse_file(asset("phone_vocab"));
    const Json& token_to_id = vocab.has("token_to_id") ? vocab.get("token_to_id") : vocab;
    for (const auto& [token, id] : token_to_id.items()) {
        phone_to_id_[token] = static_cast<int>(id.integer());
        if (id.integer() >= static_cast<int64_t>(id_to_token_.size())) id_to_token_.resize(static_cast<std::size_t>(id.integer()) + 1);
        id_to_token_[static_cast<std::size_t>(id.integer())] = token;
    }
    voice_to_id_ = load_index(asset("voice_index"));
    language_to_id_ = load_index(asset("language_index"));
    bits_ = modifier_bits(phone_to_id_);

    const Json& span = controls.get("span_conditioning");
    span_width_ = static_cast<std::size_t>(span.get("context_max_phones").integer(512));
    context_phones_ = static_cast<std::size_t>(span.get("context_phones_each_side").integer(180));
    prefix_frames_ = static_cast<int>(controls.get("prefix_conditioning").get("max_frames").integer(96));
    decode_context_ = static_cast<int>(controls.get("decoding").get("context_frames").integer(48));
    latent_dim_ = static_cast<int>(manifest_.get("audio").get("latent_dim").integer(24));
    latent_hop_ = static_cast<int>(manifest_.get("audio").get("latent_hop_length").integer(512));
    sample_rate_ = static_cast<int>(manifest_.get("audio").get("sample_rate").integer(24000));
    default_steps_ = static_cast<int>(controls.get("steps").get("default").integer(8));
    default_sampler_ = controls.get("sampler").get("default").str("heun");
    for (const Json& b : controls.get("target_buckets").get("buckets").elements()) {
        buckets_.push_back(Bucket{b.get("latent_frames").integer(), b.get("vector_estimator").str(), b.get("vocoder").str()});
    }
    std::sort(buckets_.begin(), buckets_.end(), [](const Bucket& a, const Bucket& b) { return a.frames < b.frames; });
    for (const Json& v : manifest_.get("voices").elements()) {
        Voice voice{v.get("id").str(), {}, v.get("default_language").str()};
        for (const Json& language : v.get("languages").elements()) voice.languages.push_back(language.str());
        voices_.push_back(std::move(voice));
    }
    g2p_ = std::make_unique<G2P>(bundle_dir_, manifest_, phone_to_id_, [this](const std::vector<int64_t>& ids, int64_t& symbols) {
        int64_t length = static_cast<int64_t>(ids.size());
        GraphOutput out = run("g2p", {view(DType::i64, {1, length}, ids.data())});
        symbols = out.shape.empty() ? 0 : out.shape.back();
        return std::move(out.values);
    });
}

Engine::~Engine() {
    graphs_.clear();
    g2p_.reset();
}

void Engine::validate() const {
    if (manifest_.get("contract_version").str() != kContractVersion) {
        throw std::invalid_argument("Bundle contract_version '" + manifest_.get("contract_version").str() + "' != '" + kContractVersion + "'");
    }
    const Json& controls = manifest_.get("controls");
    if (controls.get("graph_input_contract").str() != kGraphContract) {
        throw std::invalid_argument("Bundle graph contract '" + controls.get("graph_input_contract").str() +
                                    "' is not supported by this runtime (expected '" + kGraphContract + "'). Download the current model.");
    }
    const Json& assets = manifest_.get("assets");
    for (const std::string& key : kRequiredAssets) {
        const std::string rel = assets.get(key).str();
        if (rel.empty() || !std::filesystem::is_regular_file(std::filesystem::path(bundle_dir_) / rel)) {
            throw std::invalid_argument("Bundle asset '" + key + "' is missing (" + rel + ")");
        }
    }
    const Json& buckets = controls.get("target_buckets").get("buckets");
    if (buckets.size() == 0) throw std::invalid_argument("Bundle declares no latent-frame buckets");
    std::vector<std::pair<std::string, const std::vector<std::string>*>> expected = {
        {"g2p", nullptr}, {"duration_predictor", &kDurationInputs}, {"vector_context_encoder", &kContextInputs}};
    for (const Json& b : buckets.elements()) {
        expected.emplace_back(b.get("vector_estimator").str(), &kVectorInputs);
        expected.emplace_back(b.get("vocoder").str(), &kVocoderInputs);
    }
    const Json& components = manifest_.get("components");
    const std::string backend = backend_->name();
    for (const auto& [name, inputs] : expected) {
        const Json& spec = components.get(name);
        if (spec.is_null()) throw std::invalid_argument("Bundle component '" + name + "' is missing");
        std::vector<std::string> declared;
        for (const Json& input : spec.get("inputs").elements()) declared.push_back(input.str());
        const std::vector<std::string> want = inputs ? *inputs : std::vector<std::string>{"text"};
        if (declared != want) throw std::invalid_argument("Bundle component '" + name + "' inputs differ from the contract");
        const Json& artifact = spec.get("artifacts").get(backend);
        if (artifact.is_null()) {
            throw std::invalid_argument("Bundle component '" + name + "' has no " + backend + " artifact; this runtime was built for " + backend +
                                        " bundles");
        }
        if (!std::filesystem::is_regular_file(std::filesystem::path(bundle_dir_) / artifact.get("path").str())) {
            throw std::invalid_argument("Bundle component '" + name + "' has no " + backend + " artifact at " + artifact.get("path").str());
        }
    }
    const Json& voices = manifest_.get("voices");
    if (voices.size() == 0) throw std::invalid_argument("Bundle voices must each declare their languages");
    for (const Json& v : voices.elements()) {
        if (v.get("languages").size() == 0) throw std::invalid_argument("Bundle voices must each declare their languages");
    }
}

std::string Engine::release() const { return manifest_.get("controls").get("delivery").get("release_id").str(); }

std::vector<int64_t> Engine::bucket_frames() const {
    std::vector<int64_t> out;
    for (const Bucket& b : buckets_) out.push_back(b.frames);
    return out;
}

const Voice& Engine::voice(const std::string& id) const {
    for (const Voice& v : voices_) {
        if (v.id == id) return v;
    }
    std::vector<std::string> names;
    for (const Voice& v : voices_) names.push_back(v.id);
    std::sort(names.begin(), names.end());
    std::string list;
    for (const std::string& name : names) list += (list.empty() ? "" : ", ") + name;
    throw std::invalid_argument("Unknown voice '" + id + "'; available: " + list);
}

std::string Engine::resolve_language(const std::string& voice_id, const std::string& language) const {
    const Voice& v = voice(voice_id);
    std::string value = unicode::encode(unicode::lower(unicode::strip(unicode::decode(language))));
    if (value.empty()) return v.default_language;
    std::replace(value.begin(), value.end(), '-', '_');
    if (value == "en" || value == "english") return v.default_language;
    if (std::find(v.languages.begin(), v.languages.end(), value) == v.languages.end()) {
        std::string list;
        for (const std::string& l : v.languages) list += (list.empty() ? "" : ", ") + l;
        throw std::invalid_argument("Voice '" + voice_id + "' supports " + list + "; got '" + language + "'");
    }
    return value;
}

int64_t Engine::voice_index(const std::string& voice) const {
    auto it = voice_to_id_.find(voice);
    if (it == voice_to_id_.end()) throw std::invalid_argument("Unknown voice '" + voice + "'");
    return it->second;
}

int64_t Engine::language_index(const std::string& language) const {
    auto it = language_to_id_.find(language);
    if (it == language_to_id_.end()) throw std::invalid_argument("Unknown language '" + language + "'");
    return it->second;
}

Phonemized Engine::phonemize(const std::string& text, const std::string& language) { return g2p_->phonemize(text, language); }

std::vector<int64_t> Engine::context_ids(const std::vector<std::string>& phones) const {
    std::vector<int64_t> out;
    for (const std::string& phone : phones) {
        if (phone != kSilencePhone) out.push_back(phone_to_id_.at(phone));
    }
    return out;
}

Graph& Engine::graph(const std::string& component) {
    std::lock_guard<std::mutex> lock(graphs_mutex_);
    std::unique_ptr<Graph>& slot = graphs_[component];
    if (!slot) slot = backend_->open(component, manifest_.get("components").get(component));
    return *slot;
}

GraphOutput Engine::run(const std::string& component, const std::vector<TensorView>& inputs) {
    if (cancel_generation_.load() != active_generation_) throw CancelledError();
    return graph(component).run(inputs);
}

const Engine::Bucket& Engine::bucket(int64_t frames) const {
    for (const Bucket& b : buckets_) {
        if (b.frames >= frames) return b;
    }
    throw OverlongError("Predicted " + std::to_string(frames) + " latent frames, but this bundle supports at most " +
                        std::to_string(buckets_.back().frames) + ". Split the text into shorter sentences.");
}

std::vector<int> Engine::durations(const Sentence& sentence, int64_t voice, int64_t language, const DeliveryTensors& delivery, double speed) {
    std::vector<int64_t> target;
    for (const std::string& phone : sentence.phones) target.push_back(phone_to_id_.at(phone));
    if (target.size() > span_width_) {
        throw OverlongError("Sentence has " + std::to_string(target.size()) + " phones, but this bundle supports at most " +
                            std::to_string(span_width_) + ".");
    }
    std::vector<int64_t> before(sentence.before_ids.end() - static_cast<std::ptrdiff_t>(std::min(sentence.before_ids.size(), context_phones_)),
                                sentence.before_ids.end());
    std::vector<int64_t> after(sentence.after_ids.begin(), sentence.after_ids.begin() + static_cast<std::ptrdiff_t>(std::min(sentence.after_ids.size(), context_phones_)));
    auto [span, segments] = balanced_context(before, target, after, span_width_);
    const int64_t width = static_cast<int64_t>(span_width_);
    std::vector<int64_t> ids = pad(span, span_width_), segs = pad(segments, span_width_);
    std::vector<uint8_t> mask = pad(std::vector<uint8_t>(span.size(), 1), span_width_);
    int64_t voice_id = voice, language_id = language;
    GraphOutput frames = run("duration_predictor", {view(DType::i64, {1, width}, ids.data()), view(DType::i64, {1, width}, segs.data()),
                                                    view(DType::b8, {1, width}, mask.data()), view(DType::i64, {1}, &voice_id),
                                                    view(DType::i64, {1}, &language_id), view(DType::f32, {1, 5}, delivery.values),
                                                    view(DType::b8, {1, 5}, delivery.present)});
    std::size_t offset = static_cast<std::size_t>(std::count(segments.begin(), segments.end(), int64_t{0}));
    std::vector<float> values(frames.values.begin() + static_cast<std::ptrdiff_t>(offset),
                              frames.values.begin() + static_cast<std::ptrdiff_t>(offset + target.size()));
    return frames_to_durations(values, sentence.phones, 1.0 / speed);
}

SentenceResult Engine::synthesize_sentence(const Sentence& sentence, const std::string& voice, const std::string& language,
                                           const DeliveryTensors& delivery, const std::vector<float>& prefix, int64_t prefix_frames, int steps,
                                           const std::string& sampler, double speed, const NoiseFn& noise, const std::vector<int>* forced_durations) {
    int64_t voice_id = voice_index(voice), language_id = language_index(language);
    std::vector<int> durations = forced_durations ? *forced_durations : this->durations(sentence, voice_id, language_id, delivery, speed);
    if (durations.size() != sentence.phones.size()) throw std::invalid_argument("Durations must match the phones");
    int64_t length = 0;
    for (int frames : durations) length += frames;
    if (length <= 0) throw std::invalid_argument("Predicted zero latent frames; cannot synthesize");
    const Bucket& chosen = bucket(length);
    const int64_t frames = chosen.frames;
    const std::size_t F = static_cast<std::size_t>(frames), D = static_cast<std::size_t>(latent_dim_), P = static_cast<std::size_t>(prefix_frames_);

    std::vector<int64_t> target;
    for (const std::string& phone : sentence.phones) target.push_back(phone_to_id_.at(phone));
    std::vector<int64_t> after(sentence.after_ids.begin(), sentence.after_ids.begin() + static_cast<std::ptrdiff_t>(std::min(sentence.after_ids.size(), context_phones_)));
    std::vector<int64_t> before(sentence.before_ids.end() - static_cast<std::ptrdiff_t>(std::min(sentence.before_ids.size(), context_phones_)),
                                sentence.before_ids.end());
    std::vector<std::string> following;
    for (int64_t id : after) following.push_back(id_to_token_.at(static_cast<std::size_t>(id)));
    FrameEvents events = frame_events(sentence.phones, durations, sentence.word_starts, following, phone_to_id_, bits_);

    auto [span, segments] = balanced_context(before, target, after, span_width_);
    const int64_t width = static_cast<int64_t>(span_width_);
    std::vector<int64_t> span_ids = pad(span, span_width_), span_segs = pad(segments, span_width_);
    std::vector<uint8_t> span_mask = pad(std::vector<uint8_t>(span.size(), 1), span_width_);
    GraphOutput hidden = run("vector_context_encoder", {view(DType::i64, {1, width}, span_ids.data()), view(DType::i64, {1, width}, span_segs.data()),
                                                        view(DType::b8, {1, width}, span_mask.data())});

    std::vector<float> prefix_latents(D * P, 0.0f);
    std::vector<uint8_t> prefix_mask(P, 0);
    const std::size_t kept = static_cast<std::size_t>(std::min<int64_t>(prefix_frames, prefix_frames_));
    for (std::size_t d = 0; d < D && kept; ++d) {
        for (std::size_t t = 0; t < kept; ++t) {
            prefix_latents[d * P + (P - kept) + t] = prefix[d * static_cast<std::size_t>(prefix_frames) + (static_cast<std::size_t>(prefix_frames) - kept) + t];
        }
    }
    for (std::size_t t = P - kept; t < P; ++t) prefix_mask[t] = 1;

    std::vector<uint8_t> latent_mask(F, 0);
    std::fill_n(latent_mask.begin(), static_cast<std::size_t>(length), uint8_t{1});
    std::vector<int64_t> expanded;
    for (std::size_t i = 0; i < target.size(); ++i) expanded.insert(expanded.end(), static_cast<std::size_t>(durations[i]), target[i]);
    expanded = pad(expanded, F);
    std::vector<int64_t> boundary = pad(events.boundary, F), modifier = pad(events.modifier, F), sentence_type = pad(events.sentence, F);
    std::vector<float> phase = pad(events.phase, F), log_duration = pad(events.log_duration, F);

    std::vector<float> x(D * F, 0.0f), noise_values(D * static_cast<std::size_t>(length));
    noise(length, noise_values.data());
    for (std::size_t d = 0; d < D; ++d) {
        std::copy_n(noise_values.begin() + static_cast<std::ptrdiff_t>(d * static_cast<std::size_t>(length)), static_cast<std::size_t>(length),
                    x.begin() + static_cast<std::ptrdiff_t>(d * F));
    }
    float time_value = 0.0f;
    std::vector<float> probe(D * F);
    auto velocity = [&](const std::vector<float>& state, float time) {
        time_value = time;
        return run(chosen.vector_estimator,
                   {view(DType::f32, {1, latent_dim_, frames}, state.data()), view(DType::f32, {1}, &time_value),
                    view(DType::i64, {1, frames}, expanded.data()), view(DType::i64, {1}, &voice_id), view(DType::i64, {1}, &language_id),
                    view(DType::f32, {1, 5}, delivery.values), view(DType::b8, {1, 5}, delivery.present), view(DType::b8, {1, frames}, latent_mask.data()),
                    view(DType::f32, hidden.shape, hidden.values.data()), view(DType::f32, {1, latent_dim_, prefix_frames_}, prefix_latents.data()),
                    view(DType::b8, {1, prefix_frames_}, prefix_mask.data()), view(DType::i64, {1, frames}, boundary.data()),
                    view(DType::i64, {1, frames}, modifier.data()), view(DType::f32, {1, frames}, phase.data()),
                    view(DType::f32, {1, frames}, log_duration.data()), view(DType::i64, {1, frames}, sentence_type.data())})
            .values;
    };
    const bool heun = sampler == "heun";
    const float n = static_cast<float>(steps), n2 = static_cast<float>(2 * steps);
    for (int step = 0; step < steps; ++step) {
        std::vector<float> a = velocity(x, static_cast<float>(static_cast<double>(step) / steps));
        if (heun) {
            for (std::size_t i = 0; i < x.size(); ++i) probe[i] = x[i] + a[i] / n;
            std::vector<float> b = velocity(probe, static_cast<float>(static_cast<double>(step + 1) / steps));
            for (std::size_t i = 0; i < x.size(); ++i) x[i] = x[i] + (a[i] + b[i]) / n2;
        } else {
            for (std::size_t i = 0; i < x.size(); ++i) x[i] = x[i] + a[i] / n;
        }
        for (std::size_t d = 0; d < D; ++d) {
            for (std::size_t t = 0; t < F; ++t) x[d * F + t] = x[d * F + t] * static_cast<float>(latent_mask[t]);
        }
    }

    SentenceResult result;
    result.frames = length;
    result.bucket = frames;
    result.durations = durations;
    result.latents.resize(D * static_cast<std::size_t>(length));
    for (std::size_t d = 0; d < D; ++d) {
        std::copy_n(x.begin() + static_cast<std::ptrdiff_t>(d * F), static_cast<std::size_t>(length),
                    result.latents.begin() + static_cast<std::ptrdiff_t>(d * static_cast<std::size_t>(length)));
    }
    Json metadata{Json::Object{}};
    metadata.set("phones", Json::array_of(sentence.phones));
    metadata.set("durations", Json::array_of(durations));
    metadata.set("latent_frames", Json(length));
    metadata.set("bucket", Json(frames));
    metadata.set("steps", Json(steps));
    metadata.set("sampler", Json(sampler));
    metadata.set("prefix_frames", Json(static_cast<int64_t>(kept)));
    metadata.set("context_before", Json(std::min(sentence.before_ids.size(), context_phones_)));
    metadata.set("context_after", Json(std::min(sentence.after_ids.size(), context_phones_)));
    result.metadata = std::move(metadata);
    return result;
}

std::vector<float> Engine::decode(const std::vector<float>& latents, int64_t frames, const std::string& voice, const std::string& language,
                                  const std::vector<float>& left, int64_t left_frames) {
    const int64_t max_frames = buckets_.back().frames;
    // Up to decode_context frames of left context, and one spare frame: the vocoder returns frames * hop - hop / 2 samples.
    int64_t context = std::min<int64_t>(left_frames, decode_context_);
    context = std::max<int64_t>(0, std::min(context, max_frames - 1 - frames));
    const int64_t joined = context + frames;
    const Bucket& chosen = bucket(std::min(joined + 1, max_frames));
    const std::size_t F = static_cast<std::size_t>(chosen.frames), D = static_cast<std::size_t>(latent_dim_);
    std::vector<float> padded(D * F, 0.0f);
    for (std::size_t d = 0; d < D; ++d) {
        for (int64_t t = 0; t < context; ++t) {
            padded[d * F + static_cast<std::size_t>(t)] = left[d * static_cast<std::size_t>(left_frames) + static_cast<std::size_t>(left_frames - context + t)];
        }
        std::copy_n(latents.begin() + static_cast<std::ptrdiff_t>(d * static_cast<std::size_t>(frames)), static_cast<std::size_t>(frames),
                    padded.begin() + static_cast<std::ptrdiff_t>(d * F + static_cast<std::size_t>(context)));
    }
    std::vector<uint8_t> mask(F, 0);
    std::fill_n(mask.begin(), static_cast<std::size_t>(joined), uint8_t{1});
    int64_t voice_id = voice_index(voice), language_id = language_index(language), emotion_id = 0;
    GraphOutput audio = run(chosen.vocoder, {view(DType::f32, {1, latent_dim_, chosen.frames}, padded.data()), view(DType::b8, {1, chosen.frames}, mask.data()),
                                             view(DType::i64, {1}, &voice_id), view(DType::i64, {1}, &language_id), view(DType::i64, {1}, &emotion_id)});
    const std::size_t start = static_cast<std::size_t>(context * latent_hop_), count = static_cast<std::size_t>(frames * latent_hop_);
    std::vector<float> out(count, 0.0f);
    for (std::size_t i = 0; i < count && start + i < audio.values.size(); ++i) out[i] = std::clamp(audio.values[start + i], -1.0f, 1.0f);
    return out;
}

// ---------------------------------------------------------------------------------------------------------------
// Planning and streaming

Json PlanChunk::to_json() const {
    Json out{Json::Object{}};
    out.set("chunk_id", Json(chunk_id));
    out.set("index", Json(index));
    out.set("record_index", Json(record_index));
    out.set("chain_id", Json(chain_id));
    out.set("voice", Json(voice));
    out.set("language", Json(language));
    out.set("delivery", delivery.to_json());
    out.set("text", Json(text));
    out.set("source_text", Json(source_text));
    out.set("paragraph_end", Json(paragraph_end));
    return out;
}

Json Plan::to_json() const {
    Json out{Json::Object{}};
    out.set("source_kind", Json("text"));
    Json::Array items;
    std::vector<std::string> chains;
    for (const PlanChunk& chunk : chunks) {
        items.push_back(chunk.to_json());
        if (std::find(chains.begin(), chains.end(), chunk.chain_id) == chains.end()) chains.push_back(chunk.chain_id);
    }
    out.set("chunks", Json(std::move(items)));
    out.set("chains", Json(chains.size()));
    return out;
}

Plan plan_text(Engine& engine, const SynthesisOptions& options) {
    std::string voice = unicode::encode(unicode::strip(unicode::decode(options.voice)));
    if (voice.empty()) throw std::invalid_argument("Record 1 has no voice");
    const std::string language = engine.resolve_language(voice, options.language);
    const Delivery delivery = Delivery::parse(options.has_delivery ? options.delivery_spec.c_str() : nullptr);
    char chain_id[32];
    std::snprintf(chain_id, sizeof(chain_id), "chain_%03d", 1);
    Plan plan;
    std::vector<Text> paragraphs = split_paragraphs(unicode::decode(options.text));
    for (std::size_t p = 0; p < paragraphs.size(); ++p) {
        std::string paragraph = unicode::encode(paragraphs[p]);
        Text spoken = options.normalize ? unicode::decode(normalize_spoken_text(paragraph, language)) : paragraphs[p];
        std::vector<Text> sentences = split_sentences(spoken);
        for (std::size_t s = 0; s < sentences.size(); ++s) {
            PlanChunk chunk;
            chunk.index = static_cast<int>(plan.chunks.size());
            char chunk_id[32];
            std::snprintf(chunk_id, sizeof(chunk_id), "chunk_%04d", chunk.index + 1);
            chunk.chunk_id = chunk_id;
            chunk.chain_id = chain_id;
            chunk.voice = voice;
            chunk.language = language;
            chunk.delivery = delivery;
            chunk.text = unicode::encode(sentences[s]);
            chunk.source_text = unicode::encode(unicode::strip(paragraphs[p]));
            chunk.paragraph_end = s == sentences.size() - 1 && p < paragraphs.size() - 1;
            plan.chunks.push_back(std::move(chunk));
        }
    }
    return plan;
}

Json synthesize_stream(Engine& engine, const SynthesisOptions& options, const std::function<bool(const StreamEvent&)>& emit,
                       const ChunkNoiseFn& chunk_noise) {
    std::string sampler = unicode::encode(unicode::lower(unicode::decode(options.sampler.empty() ? engine.default_sampler() : options.sampler)));
    if (sampler != "heun" && sampler != "euler") throw std::invalid_argument("Unsupported sampler '" + sampler + "'; choose one of heun, euler");
    if (!(options.speed > 0.0) || !std::isfinite(options.speed)) throw std::invalid_argument("speed must be positive");
    if (options.steps < 0) throw std::invalid_argument("steps must be positive");
    if (!std::isfinite(options.temperature)) throw std::invalid_argument("temperature must be finite");
    const int steps = options.steps > 0 ? options.steps : engine.default_steps();
    const Plan plan = plan_text(engine, options);
    if (plan.chunks.empty()) throw std::invalid_argument("Nothing to speak: the text has no words");
    const auto started = std::chrono::steady_clock::now();
    const int chunk_count = static_cast<int>(plan.chunks.size());
    auto send = [&](StreamEvent event) {
        event.chunk_count = chunk_count;
        if (!emit(event)) throw CancelledError();
    };
    send(StreamEvent{StreamEvent::kPlan, nullptr, 0, nullptr, plan.to_json()});

    const std::size_t D = static_cast<std::size_t>(engine.latent_dim());
    const int64_t keep = std::max(engine.prefix_frames(), engine.decode_context());
    std::deque<PlanChunk> pending(plan.chunks.begin(), plan.chunks.end());
    std::vector<PlanChunk> spoken;
    std::map<std::string, std::pair<std::vector<float>, int64_t>> tails;  // chain -> ([D, frames], frames)
    std::optional<double> first_audio_ms;
    Json chunk_metadata{Json::Array{}};
    std::random_device entropy;

    while (!pending.empty()) {
        PlanChunk chunk = pending.front();
        pending.pop_front();
        // The sentence's phones plus up to the context budget of neighbouring phones from the same chain.
        Phonemized phonemized = engine.phonemize(chunk.text, chunk.language);
        Sentence sentence{phonemized.phones, phonemized.word_starts, {}, {}};
        for (auto it = spoken.rbegin(); it != spoken.rend(); ++it) {
            if (it->chain_id != chunk.chain_id) continue;
            if (sentence.before_ids.size() >= engine.context_phones()) break;
            std::vector<int64_t> ids = engine.context_ids(engine.phonemize(it->text, it->language).phones);
            sentence.before_ids.insert(sentence.before_ids.begin(), ids.begin(), ids.end());
        }
        for (const PlanChunk& next : pending) {
            if (next.chain_id != chunk.chain_id) continue;
            if (sentence.after_ids.size() >= engine.context_phones()) break;
            std::vector<int64_t> ids = engine.context_ids(engine.phonemize(next.text, next.language).phones);
            sentence.after_ids.insert(sentence.after_ids.end(), ids.begin(), ids.end());
        }
        DeliveryTensors delivery{};
        chunk.delivery.tensors(delivery.values, delivery.present);
        const std::optional<uint64_t> seed = options.seed ? std::optional<uint64_t>(*options.seed + static_cast<uint64_t>(chunk.index)) : std::nullopt;
        static const std::pair<std::vector<float>, int64_t> no_tail{{}, 0};
        auto tail_it = tails.find(chunk.chain_id);
        const auto& tail = tail_it == tails.end() ? no_tail : tail_it->second;
        send(StreamEvent{StreamEvent::kChunkStarted, &chunk, 0, nullptr, chunk.to_json()});

        const uint64_t draw_seed = seed ? *seed : (static_cast<uint64_t>(entropy()) << 32) ^ entropy();
        NoiseFn noise = [&](int64_t frames, float* out) {
            if (chunk_noise) chunk_noise(chunk.index, seed.value_or(0), seed.has_value(), frames, out);
            else gaussian_noise(draw_seed, static_cast<int64_t>(D) * frames, static_cast<float>(options.temperature), out);
        };
        SentenceResult result;
        try {
            result = engine.synthesize_sentence(sentence, chunk.voice, chunk.language, delivery, tail.first, tail.second, steps, sampler, options.speed, noise);
        } catch (const OverlongError&) {
            std::vector<Text> pieces = split_for_retry(unicode::decode(chunk.text));
            for (std::size_t offset = 0; offset < pieces.size(); ++offset) {
                PlanChunk piece = chunk;
                piece.chunk_id = chunk.chunk_id + "_" + std::to_string(pieces.size() - offset);
                piece.text = unicode::encode(pieces[pieces.size() - 1 - offset]);
                pending.push_front(std::move(piece));
            }
            continue;
        }
        std::vector<float> audio = engine.decode(result.latents, result.frames, chunk.voice, chunk.language, tail.first, tail.second);
        // Keep the last frames of the chain's latents for the next sentence's prefix and decode context.
        const int64_t joined_frames = tail.second + result.frames, kept = std::min(joined_frames, keep);
        std::vector<float> next_tail(D * static_cast<std::size_t>(kept));
        for (std::size_t d = 0; d < D; ++d) {
            for (int64_t t = 0; t < kept; ++t) {
                const int64_t source = joined_frames - kept + t;
                next_tail[d * static_cast<std::size_t>(kept) + static_cast<std::size_t>(t)] =
                    source < tail.second ? tail.first[d * static_cast<std::size_t>(tail.second) + static_cast<std::size_t>(source)]
                                         : result.latents[d * static_cast<std::size_t>(result.frames) + static_cast<std::size_t>(source - tail.second)];
            }
        }
        tails[chunk.chain_id] = {std::move(next_tail), kept};
        spoken.push_back(chunk);
        if (!first_audio_ms) first_audio_ms = elapsed_ms(started);
        Json metadata{Json::Object{}};
        metadata.set("chunk", chunk.to_json());
        for (const auto& [key, value] : result.metadata.items()) metadata.set(key, value);
        metadata.set("seed", seed ? Json(static_cast<int64_t>(*seed)) : Json());
        metadata.set("first_audio_ms", Json(round1(*first_audio_ms)));
        chunk_metadata.push(metadata);
        send(StreamEvent{StreamEvent::kAudio, &chunk, 0, &audio, metadata});
    }
    Json summary{Json::Object{}};
    summary.set("chunks", Json(spoken.size()));
    summary.set("first_audio_ms", first_audio_ms ? Json(round1(*first_audio_ms)) : Json());
    summary.set("elapsed_ms", Json(round1(elapsed_ms(started))));
    send(StreamEvent{StreamEvent::kDone, nullptr, 0, nullptr, summary});
    summary.set("chunks", chunk_metadata);
    return summary;
}

}  // namespace scyllasband
