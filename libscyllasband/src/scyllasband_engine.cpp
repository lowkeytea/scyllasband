#include "scyllasband_engine.h"

#include "scyllasband_text.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

namespace scyllasband {

const char* const kGraphContract = "scyllasband_measured_controls_v3";
const char* const kGraphContractV4 = "scyllasband_measured_controls_v4";

namespace {

const char* const kContractVersion = "1.0.0";
const std::vector<std::string> kDurationInputs = {"span_phone_ids", "span_segment_ids", "span_mask", "voice_id", "language_id",
                                                  "delivery_values", "delivery_present"};
const std::vector<std::string> kContextInputs = {"span_context_phone_ids", "span_context_segment_ids", "span_context_mask"};
const std::vector<std::string> kVectorInputs = {"noise", "time", "expanded_phone_ids", "voice_id", "language_id", "delivery_values",
                                                "delivery_present", "latent_mask", "span_context_hidden", "prefix_latents", "prefix_mask",
                                                "expanded_boundary_event_ids", "expanded_modifier_event_ids", "expanded_phone_phase",
                                                "expanded_phone_log_duration", "expanded_sentence_type_ids", "expanded_melody"};
const std::vector<std::string> kPitchInputs = {"noise", "time", "expanded_phone_ids", "voice_id", "language_id", "delivery_values",
                                               "delivery_present", "latent_mask", "span_context_hidden", "expanded_boundary_event_ids",
                                               "expanded_modifier_event_ids", "expanded_phone_phase", "expanded_phone_log_duration",
                                               "expanded_sentence_type_ids"};
const std::vector<std::string> kVocoderInputs = {"latents", "latent_mask", "voice_id", "language_id", "emotion_id"};
const std::vector<std::string> kFlowInputs = {"noise", "expanded_phone_ids", "voice_id", "language_id", "delivery_values", "delivery_present",
                                              "latent_mask", "span_context_hidden", "prefix_latents", "prefix_mask", "expanded_boundary_event_ids",
                                              "expanded_modifier_event_ids", "expanded_phone_phase", "expanded_phone_log_duration",
                                              "expanded_sentence_type_ids", "expanded_melody"};
// v4: the flow graphs also take the delivery mode id (contract.py VECTOR_INPUTS_V4).
const char* const kDeliveryModeInput = "delivery_mode_id";
std::vector<std::string> with_mode(std::vector<std::string> inputs) {
    inputs.emplace_back(kDeliveryModeInput);
    return inputs;
}
const std::vector<std::string> kVectorInputsV4 = with_mode(kVectorInputs);
const std::vector<std::string> kFlowInputsV4 = with_mode(kFlowInputs);
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

}  // namespace

// ---------------------------------------------------------------------------------------------------------------
// Engine

Engine::Engine(const BackendOptions& options) : Engine(options, make_backend(options)) {}

Engine::Engine(const BackendOptions& options, std::unique_ptr<Backend> backend) : backend_(std::move(backend)), bundle_dir_(options.bundle_dir) {
    const std::filesystem::path manifest_path = std::filesystem::path(bundle_dir_) / "manifest.json";
    if (!std::filesystem::is_regular_file(manifest_path)) throw std::invalid_argument("No manifest.json in " + bundle_dir_);
    manifest_ = Json::parse_file(manifest_path.string());
    validate();
    backend_->configure(manifest_);
    const Json& assets = manifest_.get("assets");
    const Json& controls = manifest_.get("controls");
    graph_contract_ = controls.get("graph_input_contract").str();
    feeds_delivery_mode_ = graph_contract_ == kGraphContractV4;
    const Json& modes = controls.get("delivery_modes");
    if (feeds_delivery_mode_ && modes.get("enabled").boolean(false)) {
        for (const auto& [mode, id] : modes.get("ids").items()) singing_.mode_ids[mode] = id.integer();
        const Json& singing = modes.get("singing");
        for (const Json& mode : singing.get("modes").elements()) singing_.modes.push_back(mode.str());
        singing_.sampler = unicode::encode(unicode::lower(unicode::decode(singing.get("sampler").str("heun"))));
        singing_.steps = static_cast<int>(singing.get("steps").integer(8));
        singing_.decoder = singing.get("decoder").str("vocos");
        singing_.guidance = singing.get("guidance").num(1.0);
        singing_.midi_center = singing.get("melody").get("midi_center").num(60.0);
        singing_.midi_scale = singing.get("melody").get("midi_scale").num(24.0);
        for (const auto& [voice_id, center] : singing.get("voice_midi_center").items()) singing_.voice_center[voice_id] = center.num();
    }
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
        buckets_.push_back(Bucket{b.get("latent_frames").integer(), b.get("vector_estimator").str(), b.get("vocoder").str(),
                                  b.get("pitch_estimator").str(), b.get("vector_flow").str(), b.get("vector_estimator_first").str(),
                                  b.get("vector_flow_first").str()});
    }
    std::sort(buckets_.begin(), buckets_.end(), [](const Bucket& a, const Bucket& b) { return a.frames < b.frames; });
    load_decoders();
    // v2 was trained on targets of 64 to 420 latent frames (about 1.4 to 9 seconds): shorter targets lose their prosody and
    // longer ones their intelligibility. A bundle can state its own as controls.chunking.min/max_target_frames.
    min_target_frames_ = controls.get("chunking").get("min_target_frames").integer(64);
    max_target_frames_ = controls.get("chunking").get("max_target_frames").integer(420);
    if (!buckets_.empty()) max_target_frames_ = std::min(max_target_frames_, buckets_.back().frames);
    // Optional (Apple bundles): fused flow graphs for one sampler and step count, and narrower G2P inputs.
    fused_sampler_ = controls.get("fused_flow").get("sampler").str();
    fused_steps_ = static_cast<int>(controls.get("fused_flow").get("steps").integer(0));
    for (const Json& b : controls.get("g2p_buckets").elements()) g2p_buckets_.emplace_back(b.get("text_tokens").integer(), b.get("component").str());
    std::sort(g2p_buckets_.begin(), g2p_buckets_.end());
    for (const Json& b : controls.get("span_buckets").elements()) {
        span_buckets_.push_back(SpanBucket{b.get("phones").integer(), b.get("duration_predictor").str(), b.get("vector_context_encoder").str(),
                                           b.get("pitch_context_encoder").str()});
    }
    std::sort(span_buckets_.begin(), span_buckets_.end(), [](const SpanBucket& a, const SpanBucket& b) { return a.phones < b.phones; });
    const Json& delivery = controls.get("delivery");
    z_scale_ = delivery.get("z_scale").num(5.0);
    z_limit_ = delivery.get("z_limit").num(2.5);
    min_spoken_phones_ = static_cast<int>(delivery.get("min_spoken_phones").integer(4));
    for (std::size_t i = 0; i < 3; ++i) {
        const Json& range = delivery.get("ranges").get(kDeliveryControls[i]);
        if (range.size() == 2) {
            ranges_[i][0] = range.elements()[0].num(ranges_[i][0]);
            ranges_[i][1] = range.elements()[1].num(ranges_[i][1]);
        }
    }
    for (const auto& [voice_id, values] : delivery.get("defaults").items()) {
        std::array<double, 3> defaults{};
        for (std::size_t i = 0; i < 3; ++i) defaults[i] = values.get(kDeliveryControls[i]).num(0.0);
        voice_defaults_[voice_id] = defaults;
    }
    pace_pause_per_sd_ = controls.get("pace_pauses").get("per_sd").num(0.0);
    const Json& pitch = controls.get("pitch");
    pitch_center_ = pitch.get("center").num(10.0);
    pitch_scale_ = pitch.get("scale").num(6.0);
    melody_center_ = pitch.get("melody_center").num(16.65);
    melody_scale_ = pitch.get("melody_scale").num(24.0);
    voiced_threshold_ = pitch.get("voiced_threshold").num(0.5);
    knee_ = pitch.get("knee").num(2.0);
    pitch_context_ = pitch.get("context_encoder").str("pitch_context_encoder");
    pitch_sampler_ = unicode::encode(unicode::lower(unicode::decode(pitch.get("sampler").get("default").str("euler"))));
    pitch_steps_ = static_cast<int>(pitch.get("steps").get("default").integer(4));
    for (const auto& [voice_id, edges] : pitch.get("edges").items()) {
        pitch_edges_[voice_id] = {edges.elements().at(0).num(), edges.elements().at(1).num()};
    }
    for (const Json& v : manifest_.get("voices").elements()) {
        Voice voice{v.get("id").str(), {}, v.get("default_language").str()};
        for (const Json& language : v.get("languages").elements()) voice.languages.push_back(language.str());
        voices_.push_back(std::move(voice));
    }
    // Each voice's sung centre: the manifest's voice_midi_center, else the middle of its spoken pitch edges (semitones on the
    // melody scale, whose melody_center is C4) moved onto the singing melody's MIDI scale.
    if (singing_.enabled()) {
        for (const Voice& voice : voices_) {
            auto edges = pitch_edges_.find(voice.id);
            if (singing_.voice_center.count(voice.id) || edges == pitch_edges_.end()) continue;
            singing_.voice_center[voice.id] = (edges->second.first + edges->second.second) / 2 - melody_center_ + singing_.midi_center;
        }
    }
    g2p_ = std::make_unique<G2P>(bundle_dir_, manifest_, phone_to_id_, [this](const std::vector<int64_t>& ids, int64_t& symbols) {
        // Padding is masked, so the narrowest G2P graph that holds the text gives the same logits.
        int64_t length = static_cast<int64_t>(ids.size()), used = length;
        while (used > 1 && ids[static_cast<std::size_t>(used - 1)] == g2p_->text_pad()) --used;
        std::string component = "g2p";
        for (const auto& [tokens, name] : g2p_buckets_) {
            if (tokens >= used && tokens < length) {
                component = name;
                length = tokens;
                break;
            }
        }
        GraphOutput out = run(component, {view(DType::i64, {1, length}, ids.data())});
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
    const std::string contract = controls.get("graph_input_contract").str();
    if (contract != kGraphContract && contract != kGraphContractV4) {
        throw std::invalid_argument("Bundle graph contract '" + contract + "' is not supported by this runtime (expected '" + kGraphContract +
                                    "' or '" + kGraphContractV4 + "'). Download the current model.");
    }
    const bool v4 = contract == kGraphContractV4;
    const std::vector<std::string>& vector_inputs = v4 ? kVectorInputsV4 : kVectorInputs;
    const std::vector<std::string>& flow_inputs = v4 ? kFlowInputsV4 : kFlowInputs;
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
        {"g2p", nullptr}, {"duration_predictor", &kDurationInputs}, {"vector_context_encoder", &kContextInputs},
        {controls.get("pitch").get("context_encoder").str("pitch_context_encoder"), &kContextInputs}};
    for (const Json& b : buckets.elements()) {
        if (b.get("pitch_estimator").str().empty()) throw std::invalid_argument("Bundle bucket has no pitch_estimator");
        expected.emplace_back(b.get("vector_estimator").str(), &vector_inputs);
        expected.emplace_back(b.get("pitch_estimator").str(), &kPitchInputs);
        expected.emplace_back(b.get("vocoder").str(), &kVocoderInputs);
        if (!b.get("vector_estimator_first").str().empty()) expected.emplace_back(b.get("vector_estimator_first").str(), &vector_inputs);
        if (!b.get("vector_flow").str().empty()) expected.emplace_back(b.get("vector_flow").str(), &flow_inputs);
        if (!b.get("vector_flow_first").str().empty()) expected.emplace_back(b.get("vector_flow_first").str(), &flow_inputs);
    }
    for (const Json& b : controls.get("span_buckets").elements()) {
        if (!b.get("duration_predictor").str().empty()) expected.emplace_back(b.get("duration_predictor").str(), &kDurationInputs);
        if (!b.get("vector_context_encoder").str().empty()) expected.emplace_back(b.get("vector_context_encoder").str(), &kContextInputs);
        if (!b.get("pitch_context_encoder").str().empty()) expected.emplace_back(b.get("pitch_context_encoder").str(), &kContextInputs);
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
        if (!std::filesystem::exists(std::filesystem::path(bundle_dir_) / artifact.get("path").str())) {  // Apple assets are directories
            throw std::invalid_argument("Bundle component '" + name + "' has no " + backend + " artifact at " + artifact.get("path").str());
        }
    }
    const Json& voices = manifest_.get("voices");
    if (voices.size() == 0) throw std::invalid_argument("Bundle voices must each declare their languages");
    for (const Json& v : voices.elements()) {
        if (v.get("languages").size() == 0) throw std::invalid_argument("Bundle voices must each declare their languages");
        if (controls.get("pitch").get("edges").get(v.get("id").str()).size() != 2) {
            throw std::invalid_argument("Bundle pitch edges lack voice '" + v.get("id").str() + "'");
        }
    }
}

void Engine::load_decoders() {
    // controls.decoders (v4); a bundle without it has the Vocos vocoder only, as v3 manifests describe it.
    const Json& declared = manifest_.get("controls").get("decoders");
    for (const auto& [name, spec] : declared.items()) {
        DecoderSpec decoder{name, spec.get("component").str("vocoder"), static_cast<int>(spec.get("samples_per_frame").integer(512)), 0};
        // "latent_frames * 512" or "latent_frames * 512 - 256" (engine.py _OUTPUT_RULE).
        std::string rule;
        for (char c : spec.get("output_samples").str()) {
            if (!std::isspace(static_cast<unsigned char>(c))) rule += c;
        }
        const std::string head = "latent_frames*";
        if (rule.rfind(head, 0) == 0) {
            const std::string tail = rule.substr(head.size());
            const std::size_t minus = tail.find('-');
            const std::string per = tail.substr(0, minus), shortfall = minus == std::string::npos ? std::string() : tail.substr(minus + 1);
            auto digits = [](const std::string& text) { return !text.empty() && std::all_of(text.begin(), text.end(), [](char c) { return c >= '0' && c <= '9'; }); };
            if (digits(per) && (minus == std::string::npos || digits(shortfall))) {
                decoder.samples_per_frame = std::stoi(per);
                decoder.short_samples = shortfall.empty() ? 0 : std::stoi(shortfall);
            }
        }
        decoders_[name] = decoder;
    }
    if (!decoders_.count("vocos")) decoders_["vocos"] = DecoderSpec{"vocos", "vocoder", latent_hop_, latent_hop_ / 2};
    const Json& components = manifest_.get("components");
    const std::string backend = backend_->name();
    for (const auto& [name, decoder] : decoders_) {
        std::vector<std::pair<int64_t, std::string>>& graphs = decoder_graphs_[name];
        for (const Json& b : manifest_.get("controls").get("target_buckets").get("buckets").elements()) {
            const std::string component = b.get(decoder.component).str();
            if (component.empty()) continue;
            if (decoder.component != "vocoder") {  // an optional decoder: checked here, used where this backend has it
                const Json& spec = components.get(component);
                if (spec.is_null()) throw std::invalid_argument("Bundle component '" + component + "' is named by a bucket but missing");
                std::vector<std::string> inputs;
                for (const Json& input : spec.get("inputs").elements()) inputs.push_back(input.str());
                if (inputs != kVocoderInputs) throw std::invalid_argument("Bundle component '" + component + "' inputs differ from the contract");
                const Json& artifact = spec.get("artifacts").get(backend);
                if (artifact.is_null()) continue;
                if (!std::filesystem::exists(std::filesystem::path(bundle_dir_) / artifact.get("path").str())) {
                    throw std::invalid_argument("Bundle component '" + component + "' has no " + backend + " artifact at " + artifact.get("path").str());
                }
            }
            graphs.emplace_back(b.get("latent_frames").integer(), component);
        }
        std::sort(graphs.begin(), graphs.end());
    }
    load_decoder_windows(declared);
}

namespace {

int64_t window_number(const std::string& text, const std::string& entry) {
    if (text.empty() || !std::all_of(text.begin(), text.end(), [](char c) { return c >= '0' && c <= '9'; }) || text.size() > 6) {
        throw std::invalid_argument("SCYLLASBAND_DECODER_WINDOW entry '" + entry + "' is not decoder=frames[:context]");
    }
    return std::stoll(text);
}

}  // namespace

void Engine::load_decoder_windows(const Json& declared) {
    for (auto& [name, decoder] : decoders_) {
        const Json& window = declared.get(name).get("window");
        decoder.window_frames = window.get("frames").integer(0);
        decoder.window_context = window.get("context").integer(0);
    }
    // SCYLLASBAND_DECODER_WINDOW overrides the manifest: "bigvgan=256:32" (frames, context), "bigvgan=0" (off).
    const char* value = std::getenv("SCYLLASBAND_DECODER_WINDOW");
    const std::string text = value ? value : "";
    for (std::size_t start = 0; start < text.size();) {
        const std::size_t comma = std::min(text.find(',', start), text.size());
        const std::string entry = text.substr(start, comma - start);
        start = comma + 1;
        if (entry.empty()) continue;
        const std::size_t equals = entry.find('='), colon = entry.find(':');
        if (equals == std::string::npos) throw std::invalid_argument("SCYLLASBAND_DECODER_WINDOW entry '" + entry + "' is not decoder=frames[:context]");
        auto decoder = decoders_.find(entry.substr(0, equals));
        if (decoder == decoders_.end()) throw std::invalid_argument("SCYLLASBAND_DECODER_WINDOW names no decoder of this bundle: '" + entry + "'");
        decoder->second.window_frames = window_number(entry.substr(equals + 1, colon == std::string::npos ? std::string::npos : colon - equals - 1), entry);
        decoder->second.window_context = colon == std::string::npos ? 0 : window_number(entry.substr(colon + 1), entry);
    }
    for (auto& [name, decoder] : decoders_) {
        if (decoder.window_frames <= 0) {
            decoder.window_frames = decoder.window_context = 0;
            continue;
        }
        // Windows stitch whole frames, so only a decoder that returns frames * hop samples can window (BigVGAN, not Vocos).
        if (decoder.short_samples != 0) throw std::invalid_argument("The " + name + " decoder cannot decode in windows (its output is not whole frames)");
        if (has_decoder(name)) decoder.window_frames = decoder_bucket(name, std::min(decoder.window_frames, decoder_max_frames(name))).first;
        if (decoder.window_context < 0 || 2 * decoder.window_context >= decoder.window_frames) {
            throw std::invalid_argument("The " + name + " decoder's window context must leave frames to keep: " + std::to_string(decoder.window_frames) +
                                        " frames, " + std::to_string(decoder.window_context) + " context");
        }
    }
}

bool Engine::has_decoder(const std::string& decoder) const {
    auto found = decoder_graphs_.find(decoder);
    return found != decoder_graphs_.end() && !found->second.empty();
}

std::vector<std::string> Engine::available_decoders() const {
    std::vector<std::string> out;
    for (const auto& [name, graphs] : decoder_graphs_) {
        if (!graphs.empty()) out.push_back(name);
    }
    return out;
}

int64_t Engine::decoder_max_frames(const std::string& decoder) const {
    auto found = decoder_graphs_.find(decoder);
    return found == decoder_graphs_.end() || found->second.empty() ? 0 : found->second.back().first;
}

std::pair<int64_t, std::string> Engine::decoder_bucket(const std::string& decoder, int64_t frames) const {
    auto found = decoder_graphs_.find(decoder);
    if (found == decoder_graphs_.end()) {
        std::string list;
        for (const auto& [name, spec] : decoders_) list += (list.empty() ? "" : ", ") + name;
        throw std::invalid_argument("Unknown decoder '" + decoder + "'; this bundle declares " + list);
    }
    if (found->second.empty()) throw std::invalid_argument("This bundle has no " + decoder + " decoder graphs");
    for (const auto& [size, component] : found->second) {
        if (size >= frames) return {size, component};
    }
    throw OverlongError(std::to_string(frames) + " latent frames to decode, but the " + decoder + " decoder supports at most " +
                        std::to_string(found->second.back().first) + ".");
}

DeliveryTensors Engine::delivery_tensors(const std::string& voice_id, const Delivery& delivery, Json* resolved) const {
    voice(voice_id);
    std::array<double, 3> value{};
    auto found = voice_defaults_.find(voice_id);
    for (std::size_t i = 0; i < 3; ++i) {
        const double fallback = found != voice_defaults_.end() ? found->second[i] : 0.0;
        value[i] = delivery.controls[i] ? std::min(std::max(*delivery.controls[i], ranges_[i][0]), ranges_[i][1]) : fallback;
    }
    DeliveryTensors out{};
    for (std::size_t i = 0; i < 2; ++i) {
        out.values[i] = static_cast<float>(std::min(std::max(value[i], -z_limit_), z_limit_) / z_scale_);
        out.present[i] = 1;
    }
    out.pitch = value[2];
    if (resolved) {
        Json json{Json::Object{}};
        for (std::size_t i = 0; i < 3; ++i) json.set(kDeliveryControls[i], Json(value[i]));
        *resolved = std::move(json);
    }
    return out;
}

Json Engine::delivery_info() const {
    Json ranges{Json::Object{}}, defaults{Json::Object{}}, units{Json::Object{}};
    for (std::size_t i = 0; i < 3; ++i) ranges.set(kDeliveryControls[i], Json(Json::Array{Json(ranges_[i][0]), Json(ranges_[i][1])}));
    for (const Voice& v : voices_) {
        Json value{Json::Object{}};
        auto found = voice_defaults_.find(v.id);
        for (std::size_t i = 0; i < 3; ++i) value.set(kDeliveryControls[i], Json(found != voice_defaults_.end() ? found->second[i] : 0.0));
        defaults.set(v.id, value);
    }
    units.set("intensity", Json("sd"));
    units.set("pace", Json("sd"));
    units.set("pitch", Json("semitones"));
    Json out{Json::Object{}};
    out.set("ranges", ranges);
    out.set("defaults", defaults);
    out.set("units", units);
    return out;
}

void Engine::target_delivery(const std::vector<std::string>& phones, const DeliveryTensors& delivery, float values[5], uint8_t present[5]) const {
    int spoken = 0;
    for (const std::string& phone : phones) spoken += phone.rfind("<", 0) != 0 && !is_non_acoustic_modifier(phone);
    for (int i = 0; i < 5; ++i) {
        values[i] = delivery.values[i];
        present[i] = spoken < min_spoken_phones_ ? 0 : delivery.present[i];
    }
}

double Engine::pause_scale(const DeliveryTensors& delivery) const {
    const double pace = delivery.present[1] ? static_cast<double>(delivery.values[1]) * z_scale_ : 0.0;
    return std::exp(pace_pause_per_sd_ * pace);
}

std::vector<float> Engine::melody(const std::vector<float>& contour, int64_t frames, const std::string& voice, double shift) const {
    const auto edges = pitch_edges_.at(voice);
    const double low = edges.first, high = edges.second, knee = std::min(knee_, (high - low) / 4.0);
    const double top = high - knee, bottom = low + knee;
    const std::size_t T = static_cast<std::size_t>(frames);
    std::vector<float> out(T * 3);
    for (std::size_t t = 0; t < T; ++t) {
        double st = static_cast<double>(contour[t]) * pitch_scale_ + pitch_center_ + shift;
        if (st > top) st = top + knee * std::tanh((st - top) / knee);
        if (st < bottom) st = bottom - knee * std::tanh((bottom - st) / knee);
        const double probability = std::min(std::max((static_cast<double>(contour[T + t]) + 1.0) / 2.0, 0.0), 1.0);
        const double voiced = probability > voiced_threshold_ ? 1.0 : 0.0;
        out[t * 3] = static_cast<float>((st - melody_center_) / melody_scale_ * voiced);
        out[t * 3 + 1] = static_cast<float>(probability);
        out[t * 3 + 2] = static_cast<float>(voiced);
    }
    return out;
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
    // SCYLLASBAND_TRACE_GRAPHS=1 prints each graph call's component and milliseconds to stderr (content-free).
    static const bool trace = [] {
        const char* value = std::getenv("SCYLLASBAND_TRACE_GRAPHS");
        return value != nullptr && value[0] == '1';
    }();
    if (!trace) return graph(component).run(inputs);
    const auto started = std::chrono::steady_clock::now();
    GraphOutput out = graph(component).run(inputs);
    std::fprintf(stderr, "scyllasband graph %s %.2f ms\n", component.c_str(), elapsed_ms(started));
    return out;
}

const Engine::Bucket& Engine::bucket(int64_t frames) const {
    for (const Bucket& b : buckets_) {
        if (b.frames >= frames) return b;
    }
    throw OverlongError("Predicted " + std::to_string(frames) + " latent frames, but this bundle supports at most " +
                        std::to_string(buckets_.back().frames) + ". Split the text into shorter sentences.");
}

const Engine::SpanBucket* Engine::span_bucket(std::size_t phones) const {
    for (const SpanBucket& b : span_buckets_) {
        if (static_cast<std::size_t>(b.phones) >= phones && static_cast<std::size_t>(b.phones) < span_width_) return &b;
    }
    return nullptr;
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
    const SpanBucket* narrow = span_bucket(span.size());
    const std::size_t padded = narrow ? static_cast<std::size_t>(narrow->phones) : span_width_;
    const int64_t width = static_cast<int64_t>(padded);
    std::vector<int64_t> ids = pad(span, padded), segs = pad(segments, padded);
    std::vector<uint8_t> mask = pad(std::vector<uint8_t>(span.size(), 1), padded);
    int64_t voice_id = voice, language_id = language;
    float values[5];
    uint8_t present[5];
    target_delivery(sentence.phones, delivery, values, present);
    GraphOutput frames = run(narrow && !narrow->duration.empty() ? narrow->duration : std::string("duration_predictor"), {view(DType::i64, {1, width}, ids.data()), view(DType::i64, {1, width}, segs.data()),
                                                    view(DType::b8, {1, width}, mask.data()), view(DType::i64, {1}, &voice_id),
                                                    view(DType::i64, {1}, &language_id), view(DType::f32, {1, 5}, values),
                                                    view(DType::b8, {1, 5}, present)});
    std::size_t offset = static_cast<std::size_t>(std::count(segments.begin(), segments.end(), int64_t{0}));
    std::vector<float> predicted(frames.values.begin() + static_cast<std::ptrdiff_t>(offset),
                                 frames.values.begin() + static_cast<std::ptrdiff_t>(offset + target.size()));
    return frames_to_durations(predicted, sentence.phones, 1.0 / speed, pause_scale(delivery));
}

SentenceResult Engine::synthesize_sentence(const Sentence& sentence, const SentenceRequest& request) {
    const std::string& voice = request.voice;
    const std::string& language = request.language;
    const DeliveryTensors& delivery = request.delivery;
    static const std::vector<float> kNoPrefix;
    const std::vector<float>& prefix = request.prefix ? *request.prefix : kNoPrefix;
    const int64_t prefix_frames = request.prefix ? request.prefix_frames : 0;
    const int steps = request.steps;
    const std::string& sampler = request.sampler;
    const double speed = request.speed;
    const NoiseFn& noise = request.noise;
    const std::vector<int>* forced_durations = request.durations;
    const SungConditioning* sung = request.sung;
    int64_t voice_id = voice_index(voice), language_id = language_index(language);
    const std::vector<float>* score_melody = sung ? sung->melody : nullptr;
    int64_t mode_id = sung ? sung->mode : 0;
    const double guidance = sung ? sung->guidance : 1.0;
    if (mode_id != 0 && !feeds_delivery_mode_) throw std::invalid_argument("This bundle has no delivery modes (a v3 bundle); singing needs a v4 bundle");
    std::vector<int> durations = forced_durations ? *forced_durations : this->durations(sentence, voice_id, language_id, delivery, speed);
    if (durations.size() != sentence.phones.size()) throw std::invalid_argument("Durations must match the phones");
    int64_t length = 0;
    for (int frames : durations) length += frames;
    if (length <= 0) throw std::invalid_argument("Predicted zero latent frames; cannot synthesize");
    if (score_melody && score_melody->size() != static_cast<std::size_t>(length) * 3) {
        throw std::invalid_argument("melody must be [" + std::to_string(length) + ", 3] (one row per latent frame); got " +
                                    std::to_string(score_melody->size()) + " values");
    }
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
    const SpanBucket* narrow = span_bucket(span.size());
    const bool narrow_context = narrow && !narrow->context.empty();
    const std::size_t padded = narrow_context ? static_cast<std::size_t>(narrow->phones) : span_width_;
    const int64_t width = static_cast<int64_t>(padded);
    std::vector<int64_t> span_ids = pad(span, padded), span_segs = pad(segments, padded);
    std::vector<uint8_t> span_mask = pad(std::vector<uint8_t>(span.size(), 1), padded);
    GraphOutput hidden = run(narrow_context ? narrow->context : std::string("vector_context_encoder"), {view(DType::i64, {1, width}, span_ids.data()), view(DType::i64, {1, width}, span_segs.data()),
                                                        view(DType::b8, {1, width}, span_mask.data())});
    // The pitch predictor's own span encoder, at its narrowest graph that holds the span (not needed when a score gives the melody).
    GraphOutput pitch_hidden;
    if (!score_melody) {
        const bool narrow_pitch = narrow && !narrow->pitch_context.empty();
        const std::size_t pitch_padded = narrow_pitch ? static_cast<std::size_t>(narrow->phones) : span_width_;
        const int64_t pitch_width = static_cast<int64_t>(pitch_padded);
        std::vector<int64_t> pitch_span_ids = pad(span, pitch_padded), pitch_span_segs = pad(segments, pitch_padded);
        std::vector<uint8_t> pitch_span_mask = pad(std::vector<uint8_t>(span.size(), 1), pitch_padded);
        pitch_hidden = run(narrow_pitch ? narrow->pitch_context : pitch_context_,
                           {view(DType::i64, {1, pitch_width}, pitch_span_ids.data()), view(DType::i64, {1, pitch_width}, pitch_span_segs.data()),
                            view(DType::b8, {1, pitch_width}, pitch_span_mask.data())});
    }
    float delivery_values[5];
    uint8_t delivery_present[5];
    target_delivery(sentence.phones, delivery, delivery_values, delivery_present);

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

    const std::size_t L = static_cast<std::size_t>(length);
    std::vector<float> x(D * F, 0.0f), noise_values(D * L), pitch_noise(2 * L);
    noise(length, pitch_noise.data(), noise_values.data());

    // Pitch contour: the predictor integrates its own noise over the same frames and events (no audio prefix, no melody).
    // A sung target takes its melody from the score instead: no pitch predictor, no pitch shift, no soft limit.
    std::vector<float> expanded_melody(F * 3, 0.0f);
    const int pitch_steps = pitch_steps_;
    if (score_melody) {
        std::copy(score_melody->begin(), score_melody->end(), expanded_melody.begin());
    } else {
    std::vector<float> c(2 * F, 0.0f), c_probe(2 * F);
    for (std::size_t r = 0; r < 2; ++r) std::copy_n(pitch_noise.begin() + static_cast<std::ptrdiff_t>(r * L), L, c.begin() + static_cast<std::ptrdiff_t>(r * F));
    float pitch_time = 0.0f;
    auto pitch_velocity = [&](const std::vector<float>& state, float time) {
        pitch_time = time;
        return run(chosen.pitch_estimator,
                   {view(DType::f32, {1, 2, frames}, state.data()), view(DType::f32, {1}, &pitch_time),
                    view(DType::i64, {1, frames}, expanded.data()), view(DType::i64, {1}, &voice_id), view(DType::i64, {1}, &language_id),
                    view(DType::f32, {1, 5}, delivery_values), view(DType::b8, {1, 5}, delivery_present), view(DType::b8, {1, frames}, latent_mask.data()),
                    view(DType::f32, pitch_hidden.shape, pitch_hidden.values.data()), view(DType::i64, {1, frames}, boundary.data()),
                    view(DType::i64, {1, frames}, modifier.data()), view(DType::f32, {1, frames}, phase.data()),
                    view(DType::f32, {1, frames}, log_duration.data()), view(DType::i64, {1, frames}, sentence_type.data())})
            .values;
    };
    const float pn = static_cast<float>(pitch_steps), pn2 = static_cast<float>(2 * pitch_steps);
    for (int step = 0; step < pitch_steps; ++step) {
        std::vector<float> a = pitch_velocity(c, static_cast<float>(static_cast<double>(step) / pitch_steps));
        if (pitch_sampler_ == "heun") {
            for (std::size_t i = 0; i < c.size(); ++i) c_probe[i] = c[i] + a[i] / pn;
            std::vector<float> b = pitch_velocity(c_probe, static_cast<float>(static_cast<double>(step + 1) / pitch_steps));
            for (std::size_t i = 0; i < c.size(); ++i) c[i] = c[i] + (a[i] + b[i]) / pn2;
        } else {
            for (std::size_t i = 0; i < c.size(); ++i) c[i] = c[i] + a[i] / pn;
        }
        for (std::size_t r = 0; r < 2; ++r) {
            for (std::size_t t = 0; t < F; ++t) c[r * F + t] = c[r * F + t] * static_cast<float>(latent_mask[t]);
        }
    }
    std::vector<float> contour(2 * L);
    for (std::size_t r = 0; r < 2; ++r) std::copy_n(c.begin() + static_cast<std::ptrdiff_t>(r * F), L, contour.begin() + static_cast<std::ptrdiff_t>(r * L));
    std::vector<float> melody_values = melody(contour, length, voice, delivery.pitch);
    std::copy(melody_values.begin(), melody_values.end(), expanded_melody.begin());
    }

    for (std::size_t d = 0; d < D; ++d) {
        std::copy_n(noise_values.begin() + static_cast<std::ptrdiff_t>(d * static_cast<std::size_t>(length)), static_cast<std::size_t>(length),
                    x.begin() + static_cast<std::ptrdiff_t>(d * F));
    }
    // A target without a prefix (a chain's first) uses the bucket's prefix-less graphs when the bundle has them.
    const bool no_prefix = kept == 0;
    const std::string& estimator = no_prefix && !chosen.vector_estimator_first.empty() ? chosen.vector_estimator_first : chosen.vector_estimator;
    const std::string& flow = no_prefix && !chosen.vector_flow_first.empty() ? chosen.vector_flow_first : chosen.vector_flow;
    float time_value = 0.0f;
    std::vector<float> probe(D * F);
    // Guidance contrasts the conditioned velocity with the melody and mode switched off; it needs the stepped graph.
    const bool guided = guidance != 1.0 && (score_melody != nullptr || mode_id != 0);
    const std::vector<float> no_melody(guided ? F * 3 : 0, 0.0f);
    int64_t no_mode = 0;
    const float scale = static_cast<float>(guidance);
    auto conditioned = [&](const std::vector<float>& state, float time, const std::vector<float>& melody_input, const int64_t* mode) {
        time_value = time;
        std::vector<TensorView> inputs = {
            view(DType::f32, {1, latent_dim_, frames}, state.data()), view(DType::f32, {1}, &time_value),
            view(DType::i64, {1, frames}, expanded.data()), view(DType::i64, {1}, &voice_id), view(DType::i64, {1}, &language_id),
            view(DType::f32, {1, 5}, delivery_values), view(DType::b8, {1, 5}, delivery_present), view(DType::b8, {1, frames}, latent_mask.data()),
            view(DType::f32, hidden.shape, hidden.values.data()), view(DType::f32, {1, latent_dim_, prefix_frames_}, prefix_latents.data()),
            view(DType::b8, {1, prefix_frames_}, prefix_mask.data()), view(DType::i64, {1, frames}, boundary.data()),
            view(DType::i64, {1, frames}, modifier.data()), view(DType::f32, {1, frames}, phase.data()),
            view(DType::f32, {1, frames}, log_duration.data()), view(DType::i64, {1, frames}, sentence_type.data()),
            view(DType::f32, {1, frames, 3}, melody_input.data())};
        if (feeds_delivery_mode_) inputs.push_back(view(DType::i64, {1}, mode));
        return run(estimator, inputs).values;
    };
    auto velocity = [&](const std::vector<float>& state, float time) {
        std::vector<float> on = conditioned(state, time, expanded_melody, &mode_id);
        if (!guided) return on;
        const std::vector<float> base = conditioned(state, time, no_melody, &no_mode);
        for (std::size_t i = 0; i < on.size(); ++i) on[i] = base[i] + scale * (on[i] - base[i]);
        return on;
    };
    const bool heun = sampler == "heun";
    const bool fused = !flow.empty() && sampler == fused_sampler_ && steps == fused_steps_ && !guided;
    if (fused) {
        std::vector<TensorView> inputs = {
            view(DType::f32, {1, latent_dim_, frames}, x.data()), view(DType::i64, {1, frames}, expanded.data()), view(DType::i64, {1}, &voice_id),
            view(DType::i64, {1}, &language_id), view(DType::f32, {1, 5}, delivery_values), view(DType::b8, {1, 5}, delivery_present),
            view(DType::b8, {1, frames}, latent_mask.data()), view(DType::f32, hidden.shape, hidden.values.data()),
            view(DType::f32, {1, latent_dim_, prefix_frames_}, prefix_latents.data()), view(DType::b8, {1, prefix_frames_}, prefix_mask.data()),
            view(DType::i64, {1, frames}, boundary.data()), view(DType::i64, {1, frames}, modifier.data()), view(DType::f32, {1, frames}, phase.data()),
            view(DType::f32, {1, frames}, log_duration.data()), view(DType::i64, {1, frames}, sentence_type.data()),
            view(DType::f32, {1, frames, 3}, expanded_melody.data())};
        if (feeds_delivery_mode_) inputs.push_back(view(DType::i64, {1}, &mode_id));
        x = run(flow, inputs).values;
    }
    const float n = static_cast<float>(steps), n2 = static_cast<float>(2 * steps);
    for (int step = 0; step < (fused ? 0 : steps); ++step) {
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
    metadata.set("pitch_steps", Json(pitch_steps));
    metadata.set("pitch_sampler", Json(pitch_sampler_));
    metadata.set("pitch_shift", Json(delivery.pitch));
    metadata.set("delivery_set", Json(delivery_present[0] != 0));
    metadata.set("fused_flow", Json(fused));
    metadata.set("flow_graph", Json(fused ? flow : estimator));
    metadata.set("prefix_frames", Json(static_cast<int64_t>(kept)));
    metadata.set("context_before", Json(std::min(sentence.before_ids.size(), context_phones_)));
    metadata.set("context_after", Json(std::min(sentence.after_ids.size(), context_phones_)));
    if (feeds_delivery_mode_ || score_melody) {
        metadata.set("delivery_mode", Json(mode_id));
        metadata.set("melody_source", Json(score_melody ? "score" : "pitch_predictor"));
        metadata.set("guidance", Json(guidance));
    }
    result.metadata = std::move(metadata);
    return result;
}

std::vector<float> Engine::decode(const std::vector<float>& latents, int64_t frames, const DecodeRequest& request) {
    const std::string& voice = request.voice;
    const std::string& language = request.language;
    const std::string& decoder = request.decoder;
    static const std::vector<float> kNoLeft;
    const std::vector<float>& left = request.left ? *request.left : kNoLeft;
    const int64_t left_frames = request.left ? request.left_frames : 0;
    auto spec = decoders_.find(decoder);
    if (spec == decoders_.end()) decoder_bucket(decoder, frames);  // throws: unknown decoder
    if (spec->second.window_frames > 0) return decode_windowed(latents, frames, request, spec->second);
    const int per_frame = spec->second.samples_per_frame;
    auto graphs = decoder_graphs_.find(decoder);
    const int64_t max_frames = graphs->second.empty() ? buckets_.back().frames : graphs->second.back().first;
    // A target longer than the decoder's largest bucket cannot be decoded (BigVGAN may stop below the flow's largest bucket).
    if (frames > max_frames) {
        throw OverlongError(std::to_string(frames) + " latent frames to decode, but the " + decoder + " decoder supports at most " +
                            std::to_string(max_frames) + ".");
    }
    // Up to decode_context frames of left context; left context gives way to the target when both do not fit the largest bucket.
    int64_t context = std::min<int64_t>(left_frames, decode_context_);
    context = std::max<int64_t>(0, std::min(context, max_frames - 1 - frames));
    const int64_t joined = context + frames;
    // Vocos returns frames * hop - hop / 2 samples, so one spare frame lets it emit the target's last frame in full; BigVGAN
    // returns frames * hop and needs none.
    const int64_t spare = spec->second.short_samples > 0 ? 1 : 0;
    const auto [bucket_frames, component] = decoder_bucket(decoder, std::min(joined + spare, max_frames));
    struct {
        int64_t frames;
        const std::string& vocoder;
    } chosen{bucket_frames, component};
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
    const std::size_t start = static_cast<std::size_t>(context * per_frame), count = static_cast<std::size_t>(frames * per_frame);
    std::vector<float> out(count, 0.0f);
    for (std::size_t i = 0; i < count && start + i < audio.values.size(); ++i) out[i] = std::clamp(audio.values[start + i], -1.0f, 1.0f);
    return out;
}

std::vector<float> Engine::decode_windowed(const std::vector<float>& latents, int64_t frames, const DecodeRequest& request, const DecoderSpec& spec) {
    const auto [window, component] = decoder_bucket(spec.name, spec.window_frames);
    const int64_t context = spec.window_context;
    // The sequence to decode: up to decode_context frames of left context, then the target. Each call decodes `window` frames
    // of it (fewer at a short sequence, padded and masked) and keeps those at least `context` frames from either cut inside
    // the sequence; the last window ends at the sequence end and moves left to decode a full window where it can.
    const int64_t left = request.left ? std::min<int64_t>(request.left_frames, decode_context_) : 0;
    const int64_t total = left + frames;
    const std::size_t D = static_cast<std::size_t>(latent_dim_), T = static_cast<std::size_t>(total), W = static_cast<std::size_t>(window);
    std::vector<float> sequence(D * T);
    for (std::size_t d = 0; d < D; ++d) {
        for (int64_t t = 0; t < left; ++t) {
            sequence[d * T + static_cast<std::size_t>(t)] =
                (*request.left)[d * static_cast<std::size_t>(request.left_frames) + static_cast<std::size_t>(request.left_frames - left + t)];
        }
        std::copy_n(latents.begin() + static_cast<std::ptrdiff_t>(d * static_cast<std::size_t>(frames)), static_cast<std::size_t>(frames),
                    sequence.begin() + static_cast<std::ptrdiff_t>(d * T + static_cast<std::size_t>(left)));
    }
    int64_t voice_id = voice_index(request.voice), language_id = language_index(request.language), emotion_id = 0;
    const std::size_t hop = static_cast<std::size_t>(spec.samples_per_frame);
    std::vector<float> out(static_cast<std::size_t>(frames) * hop, 0.0f), padded(D * W);
    std::vector<uint8_t> mask(W);
    for (int64_t start = left; start < total;) {
        const int64_t from = std::max<int64_t>(0, std::min(start - context, total - window));
        const int64_t to = std::min(total, from + window);
        const int64_t keep = to == total ? total : to - context;  // > start: window > 2 * context (load_decoder_windows)
        std::fill(padded.begin(), padded.end(), 0.0f);
        for (std::size_t d = 0; d < D; ++d) {
            std::copy_n(sequence.begin() + static_cast<std::ptrdiff_t>(d * T + static_cast<std::size_t>(from)), static_cast<std::size_t>(to - from),
                        padded.begin() + static_cast<std::ptrdiff_t>(d * W));
        }
        std::fill(mask.begin(), mask.end(), uint8_t{0});
        std::fill_n(mask.begin(), static_cast<std::size_t>(to - from), uint8_t{1});
        GraphOutput audio = run(component, {view(DType::f32, {1, latent_dim_, window}, padded.data()), view(DType::b8, {1, window}, mask.data()),
                                            view(DType::i64, {1}, &voice_id), view(DType::i64, {1}, &language_id), view(DType::i64, {1}, &emotion_id)});
        const std::size_t first = static_cast<std::size_t>(start - from) * hop, last = static_cast<std::size_t>(keep - from) * hop;
        const std::size_t at = static_cast<std::size_t>(start - left) * hop;
        for (std::size_t i = first; i < last && i < audio.values.size(); ++i) out[at + i - first] = std::clamp(audio.values[i], -1.0f, 1.0f);
        start = keep;
    }
    return out;
}


}  // namespace scyllasband
