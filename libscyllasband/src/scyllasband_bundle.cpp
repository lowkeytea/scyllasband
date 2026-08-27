#include "scyllasband_bundle.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace scyllasband_detail {
namespace {

constexpr const char* kContractVersion = "1.0.0";
const char* kRequiredComponents[] = {"g2p", "duration_predictor", "vector_estimator", "vocoder"};
const char* kOptionalComponents[] = {"vector_context_encoder", "vector_estimator_prefix", "vector_estimator_tail"};

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Bundle manifest is missing: " + path.string());
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (char ch : value) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << ch; break;
        }
    }
    return out.str();
}

void append_json_string_or_null(std::ostringstream& out, const std::string& value) {
    if (value.empty()) {
        out << "null";
        return;
    }
    out << "\"" << json_escape(value) << "\"";
}

std::string find_key_token(const std::string& key) {
    return "\"" + key + "\"";
}

std::size_t find_value_start(const std::string& json, const std::string& key) {
    const std::string token = find_key_token(key);
    const std::size_t key_pos = json.find(token);
    if (key_pos == std::string::npos) {
        return std::string::npos;
    }
    const std::size_t colon = json.find(':', key_pos + token.size());
    if (colon == std::string::npos) {
        return std::string::npos;
    }
    std::size_t pos = colon + 1;
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) {
        ++pos;
    }
    return pos;
}

std::size_t matching_delimiter(const std::string& json, std::size_t start, char open, char close) {
    bool in_string = false;
    bool escaped = false;
    int depth = 0;
    for (std::size_t pos = start; pos < json.size(); ++pos) {
        const char ch = json[pos];
        if (escaped) {
            escaped = false;
            continue;
        }
        if (ch == '\\' && in_string) {
            escaped = true;
            continue;
        }
        if (ch == '"') {
            in_string = !in_string;
            continue;
        }
        if (in_string) {
            continue;
        }
        if (ch == open) {
            ++depth;
        } else if (ch == close) {
            --depth;
            if (depth == 0) {
                return pos;
            }
        }
    }
    return std::string::npos;
}

std::string object_for_key(const std::string& json, const std::string& key) {
    const std::size_t start = find_value_start(json, key);
    if (start == std::string::npos || start >= json.size() || json[start] != '{') {
        return {};
    }
    const std::size_t end = matching_delimiter(json, start, '{', '}');
    if (end == std::string::npos) {
        return {};
    }
    return json.substr(start, end - start + 1);
}

std::string array_for_key(const std::string& json, const std::string& key) {
    const std::size_t start = find_value_start(json, key);
    if (start == std::string::npos || start >= json.size() || json[start] != '[') {
        return {};
    }
    const std::size_t end = matching_delimiter(json, start, '[', ']');
    if (end == std::string::npos) {
        return {};
    }
    return json.substr(start, end - start + 1);
}

std::string string_for_key(const std::string& json, const std::string& key, const std::string& fallback = {}) {
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
    std::smatch match;
    if (std::regex_search(json, match, pattern)) {
        return match[1].str();
    }
    return fallback;
}

int int_for_key(const std::string& json, const std::string& key, int fallback = 0) {
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*(-?[0-9]+)");
    std::smatch match;
    if (std::regex_search(json, match, pattern)) {
        return std::stoi(match[1].str());
    }
    return fallback;
}

float float_for_key(const std::string& json, const std::string& key, float fallback = 0.0f) {
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?)");
    std::smatch match;
    if (std::regex_search(json, match, pattern)) {
        return std::stof(match[1].str());
    }
    return fallback;
}

bool bool_for_key(const std::string& json, const std::string& key, bool fallback = false) {
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*(true|false)");
    std::smatch match;
    if (std::regex_search(json, match, pattern)) {
        return match[1].str() == "true";
    }
    return fallback;
}

std::vector<std::string> string_array_for_key(const std::string& json, const std::string& key) {
    std::vector<std::string> values;
    const std::string array = array_for_key(json, key);
    if (array.empty()) {
        return values;
    }
    const std::regex item_pattern("\\\"([^\\\"]*)\\\"");
    for (std::sregex_iterator it(array.begin(), array.end(), item_pattern), end; it != end; ++it) {
        values.push_back((*it)[1].str());
    }
    return values;
}

std::vector<float> float_array_for_key(const std::string& json, const std::string& key) {
    std::vector<float> values;
    const std::string array = array_for_key(json, key);
    if (array.empty()) {
        return values;
    }
    const std::regex item_pattern("-?[0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?");
    for (std::sregex_iterator it(array.begin(), array.end(), item_pattern), end; it != end; ++it) {
        values.push_back(std::stof((*it)[0].str()));
    }
    return values;
}

std::map<std::string, std::vector<float>> float_array_map_for_object(
    const std::string& object_json
) {
    std::map<std::string, std::vector<float>> values;
    const std::regex pair_pattern("\\\"([^\\\"]+)\\\"\\s*:\\s*(\\[[^\\]]*\\])");
    for (std::sregex_iterator it(object_json.begin(), object_json.end(), pair_pattern), end; it != end; ++it) {
        std::vector<float> vector;
        const std::string wrapped = "{\"value\":" + (*it)[2].str() + "}";
        vector = float_array_for_key(wrapped, "value");
        values[(*it)[1].str()] = std::move(vector);
    }
    return values;
}

std::map<std::string, std::string> string_map_for_object(const std::string& object_json) {
    std::map<std::string, std::string> values;
    const std::regex pair_pattern("\\\"([^\\\"]+)\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
    for (std::sregex_iterator it(object_json.begin(), object_json.end(), pair_pattern), end; it != end; ++it) {
        values[(*it)[1].str()] = (*it)[2].str();
    }
    return values;
}

std::map<std::string, int> int_map_for_object(const std::string& object_json) {
    std::map<std::string, int> values;
    const std::regex pair_pattern("\\\"([^\\\"]+)\\\"\\s*:\\s*(-?[0-9]+)");
    for (std::sregex_iterator it(object_json.begin(), object_json.end(), pair_pattern), end; it != end; ++it) {
        values[(*it)[1].str()] = std::stoi((*it)[2].str());
    }
    return values;
}

std::map<std::string, float> float_map_for_object(const std::string& object_json) {
    std::map<std::string, float> values;
    const std::regex pair_pattern(
        "\\\"([^\\\"]+)\\\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
    );
    for (std::sregex_iterator it(object_json.begin(), object_json.end(), pair_pattern), end;
         it != end;
         ++it) {
        values[(*it)[1].str()] = std::stof((*it)[2].str());
    }
    return values;
}

std::string decode_json_string_token(const std::string& token) {
    std::string out;
    bool escaped = false;
    for (std::size_t index = 1; index + 1 < token.size(); ++index) {
        const char ch = token[index];
        if (escaped) {
            switch (ch) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                default:
                    out.push_back(ch);
                    break;
            }
            escaped = false;
        } else if (ch == '\\') {
            escaped = true;
        } else {
            out.push_back(ch);
        }
    }
    return out;
}

std::size_t find_json_string_end(const std::string& json, std::size_t start) {
    bool escaped = false;
    for (std::size_t pos = start + 1; pos < json.size(); ++pos) {
        const char ch = json[pos];
        if (escaped) {
            escaped = false;
            continue;
        }
        if (ch == '\\') {
            escaped = true;
            continue;
        }
        if (ch == '"') {
            return pos;
        }
    }
    return std::string::npos;
}

std::map<std::string, int> json_string_int_map_for_object(const std::string& object_json) {
    std::map<std::string, int> values;
    std::size_t pos = 0;
    while (pos < object_json.size()) {
        const std::size_t key_start = object_json.find('"', pos);
        if (key_start == std::string::npos) {
            break;
        }
        const std::size_t key_end = find_json_string_end(object_json, key_start);
        if (key_end == std::string::npos) {
            break;
        }
        std::size_t colon = object_json.find(':', key_end + 1);
        if (colon == std::string::npos) {
            break;
        }
        std::size_t value_start = colon + 1;
        while (value_start < object_json.size() && std::isspace(static_cast<unsigned char>(object_json[value_start]))) {
            ++value_start;
        }
        std::size_t value_end = value_start;
        if (value_end < object_json.size() && (object_json[value_end] == '-' || std::isdigit(static_cast<unsigned char>(object_json[value_end])))) {
            ++value_end;
            while (value_end < object_json.size() && std::isdigit(static_cast<unsigned char>(object_json[value_end]))) {
                ++value_end;
            }
            values[decode_json_string_token(object_json.substr(key_start, key_end - key_start + 1))] =
                std::stoi(object_json.substr(value_start, value_end - value_start));
        }
        pos = value_end;
    }
    return values;
}

std::map<int, std::string> json_int_string_map_for_object(const std::string& object_json) {
    std::map<int, std::string> values;
    std::size_t pos = 0;
    while (pos < object_json.size()) {
        const std::size_t key_start = object_json.find('"', pos);
        if (key_start == std::string::npos) {
            break;
        }
        const std::size_t key_end = find_json_string_end(object_json, key_start);
        if (key_end == std::string::npos) {
            break;
        }
        const std::string key = decode_json_string_token(object_json.substr(key_start, key_end - key_start + 1));
        std::size_t colon = object_json.find(':', key_end + 1);
        if (colon == std::string::npos) {
            break;
        }
        std::size_t value_start = object_json.find('"', colon + 1);
        if (value_start == std::string::npos) {
            break;
        }
        const std::size_t value_end = find_json_string_end(object_json, value_start);
        if (value_end == std::string::npos) {
            break;
        }
        values[std::stoi(key)] = decode_json_string_token(object_json.substr(value_start, value_end - value_start + 1));
        pos = value_end + 1;
    }
    return values;
}


std::vector<std::string> object_items_from_array(const std::string& array_json) {
    std::vector<std::string> items;
    std::size_t pos = 0;
    while (pos < array_json.size()) {
        const std::size_t start = array_json.find('{', pos);
        if (start == std::string::npos) {
            break;
        }
        const std::size_t end = matching_delimiter(array_json, start, '{', '}');
        if (end == std::string::npos) {
            break;
        }
        items.push_back(array_json.substr(start, end - start + 1));
        pos = end + 1;
    }
    return items;
}

std::map<std::string, int> id_index_map_for_array(const std::string& json) {
    std::map<std::string, int> values;
    for (const std::string& item : object_items_from_array(json)) {
        const std::string id = string_for_key(item, "id");
        if (id.empty()) {
            continue;
        }
        values[id] = int_for_key(item, "index", static_cast<int>(values.size()));
    }
    return values;
}

std::map<std::string, int> load_id_index_asset(const std::filesystem::path& path) {
    return id_index_map_for_array(read_text_file(path));
}

std::map<std::string, int> load_phone_vocab_asset(const std::filesystem::path& path) {
    const std::string json = read_text_file(path);
    const std::string token_to_id = object_for_key(json, "token_to_id");
    return int_map_for_object(token_to_id);
}

bool required_component(const std::string& component_json) {
    return bool_for_key(component_json, "required", true);
}

void require_exists(const std::filesystem::path& path, const std::string& message) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error(message + ": " + path.string());
    }
}

std::string component_artifact_path(
    const std::string& component_json,
    const std::string& backend_name
) {
    const std::string artifacts = object_for_key(component_json, "artifacts");
    if (!artifacts.empty()) {
        const std::string backend_spec = object_for_key(artifacts, backend_name);
        if (!backend_spec.empty()) {
            return string_for_key(backend_spec, "path");
        }
    }
    return string_for_key(component_json, "path");
}

void validate_architecture(const std::string& architecture) {
    if (architecture != "scyllasband-duration-flow") {
        throw std::runtime_error("Unsupported architecture " + architecture + "; expected scyllasband-duration-flow");
    }
}

}  // namespace

std::string scyllasband_backend_name(ScyllasBandBackend backend) {
    switch (backend) {
        case SCYLLASBAND_BACKEND_AUTO:
            return "auto";
        case SCYLLASBAND_BACKEND_LITERT:
            return "litert";
        case SCYLLASBAND_BACKEND_COREML:
            return "coreml";
        case SCYLLASBAND_BACKEND_ONNX:
            return "onnx";
        case SCYLLASBAND_BACKEND_COREAI:
            return "coreai";
    }
    return "unknown";
}

ScyllasBandBackend scyllasband_select_backend(ScyllasBandBackend backend) {
    if (backend != SCYLLASBAND_BACKEND_AUTO) {
        return backend;
    }
#if defined(SCYLLASBAND_WITH_COREAI)
    return SCYLLASBAND_BACKEND_COREAI;
#elif defined(SCYLLASBAND_WITH_ONNXRUNTIME)
    return SCYLLASBAND_BACKEND_ONNX;
#elif defined(SCYLLASBAND_WITH_LITERT)
    return SCYLLASBAND_BACKEND_LITERT;
#else
    return SCYLLASBAND_BACKEND_ONNX;
#endif
}

ScyllasBandBundleInfo load_scyllasband_bundle_info(
    const std::string& bundle_dir,
    ScyllasBandBackend backend
) {
    const std::filesystem::path bundle_path(bundle_dir);
    const std::filesystem::path manifest_path = bundle_path / "manifest.json";
    const std::string manifest = read_text_file(manifest_path);

    ScyllasBandBundleInfo info;
    info.bundle_dir = bundle_path.string();
    info.selected_backend = scyllasband_backend_name(scyllasband_select_backend(backend));
    info.contract_version = string_for_key(manifest, "contract_version");
    if (info.contract_version != kContractVersion) {
        throw std::runtime_error(
            "Unsupported contract_version " + info.contract_version + "; expected " + kContractVersion
        );
    }
    info.model_name = string_for_key(manifest, "model_name");
    info.model_version = string_for_key(manifest, "model_version", "3");
    info.architecture = string_for_key(manifest, "architecture", "scyllasband-duration-flow");
    validate_architecture(info.architecture);
    info.default_language = string_for_key(manifest, "default_language", "en_us");
    info.languages = string_array_for_key(manifest, "languages");
    const std::vector<std::string> voice_specs = object_items_from_array(array_for_key(manifest, "voices"));
    for (const std::string& voice : voice_specs) {
        const std::string voice_id = string_for_key(voice, "id");
        if (voice_id.empty()) {
            continue;
        }
        info.voices.push_back(voice_id);
        info.voice_default_language[voice_id] = string_for_key(voice, "default_language", info.default_language);
        info.voice_languages[voice_id] = string_array_for_key(voice, "languages");
    }
    info.preferred_backends = string_array_for_key(manifest, "preferred_backends");
    if (info.preferred_backends.empty()) {
        throw std::runtime_error("Bundle manifest must declare at least one backend");
    }

    const std::string audio = object_for_key(manifest, "audio");
    info.sample_rate = int_for_key(audio, "sample_rate");
    info.hop_length = int_for_key(audio, "hop_length");
    info.latent_hop_length = int_for_key(audio, "latent_hop_length");
    info.latent_dim = int_for_key(audio, "latent_dim");

    const std::string controls = object_for_key(manifest, "controls");
    const std::string fixed_shapes = object_for_key(controls, "fixed_shapes");
    info.g2p_text_tokens = int_for_key(fixed_shapes, "g2p_text_tokens");
    info.phone_frames = int_for_key(fixed_shapes, "phone_frames");
    info.latent_frames = int_for_key(fixed_shapes, "latent_frames");
    const std::string target_buckets = object_for_key(controls, "target_buckets");
    info.target_bucket_margin_frames = std::max(0, int_for_key(target_buckets, "bucket_margin_frames", 0));
    for (const std::string& item : object_items_from_array(array_for_key(target_buckets, "buckets"))) {
        ScyllasBandTargetBucketInfo bucket;
        bucket.latent_frames = int_for_key(item, "latent_frames");
        bucket.vector_component = string_for_key(item, "vector_estimator", "vector_estimator");
        bucket.vector_prefix_component = string_for_key(item, "vector_estimator_prefix", "vector_estimator_prefix");
        bucket.vector_tail_component = string_for_key(item, "vector_estimator_tail", "vector_estimator_tail");
        bucket.vocoder_component = string_for_key(item, "vocoder", "vocoder");
        if (bucket.latent_frames > 0) {
            info.target_buckets.push_back(bucket);
        }
    }
    std::sort(info.target_buckets.begin(), info.target_buckets.end(), [](const auto& left, const auto& right) {
        return left.latent_frames < right.latent_frames;
    });
    const std::string reference = object_for_key(controls, "reference_packs");
    info.reference_packs_enabled = bool_for_key(reference, "enabled", false);
    info.reference_pack_schema_version = int_for_key(reference, "schema_version");
    info.reference_style_dim = int_for_key(reference, "style_dim");
    info.reference_prosody_dim = int_for_key(reference, "prosody_dim");
    info.reference_identity_dim = int_for_key(reference, "identity_dim");
    info.reference_baseline_dim = int_for_key(reference, "prosody_baseline_dim");
    info.reference_delta_dim = int_for_key(reference, "prosody_delta_dim");
    info.reference_affect_routing_version = string_for_key(
        object_for_key(reference, "affect_routing"), "version"
    );
    info.reference_fallback_weight = float_for_key(reference, "fallback_weight", 0.25f);
    const std::string prefix = object_for_key(controls, "prefix_conditioning");
    info.prefix_conditioning_enabled = bool_for_key(prefix, "enabled", false);
    info.prefix_max_frames = int_for_key(prefix, "max_frames");
    const std::string span = object_for_key(controls, "span_conditioning");
    info.span_conditioning_enabled = bool_for_key(span, "enabled", false);
    info.span_context_hidden_size = int_for_key(span, "hidden_size");
    info.span_context_max_phones = int_for_key(span, "context_max_phones");
    const std::string guidance = object_for_key(controls, "emotion_guidance");
    info.emotion_guidance_enabled = bool_for_key(guidance, "enabled", false);
    const std::string affect = object_for_key(controls, "affect");
    info.affect_enabled = bool_for_key(affect, "enabled", false);
    if (info.affect_enabled) {
        info.affect_graph_input_contract = string_for_key(
            controls,
            "graph_input_contract",
            string_for_key(affect, "graph_input_contract")
        );
        info.affect_axes = string_array_for_key(affect, "axes");
        info.affect_axis_order_version = std::to_string(int_for_key(affect, "axis_order_version", 0));
        info.affect_default_preset = string_for_key(affect, "default_preset");
        info.affect_preset_version = int_for_key(affect, "preset_version", 0);
        info.affect_presets = float_array_map_for_object(object_for_key(affect, "presets"));
        info.affect_legacy_presets = float_array_map_for_object(object_for_key(affect, "legacy_presets"));
        info.affect_partial_defaults = float_map_for_object(
            object_for_key(affect, "partial_defaults")
        );
        info.affect_axis_minimums = float_map_for_object(
            object_for_key(affect, "axis_minimums")
        );
        info.affect_axis_maximums = float_map_for_object(
            object_for_key(affect, "axis_maximums")
        );
        const std::string affect_guidance = object_for_key(affect, "guidance");
        info.affect_guidance_default_scale = float_for_key(affect_guidance, "default_scale", 1.0f);
        const int axis_version = int_for_key(affect, "axis_order_version", 0);
        const std::vector<std::string> expected_axes = axis_version == 1
            ? std::vector<std::string>{
                "calm", "joy", "anger", "sadness", "sarcasm", "questioning"
            }
            : axis_version == 2
            ? std::vector<std::string>{
                "calm", "joy", "anger", "sadness", "sarcasm", "whisper"
            }
            : axis_version == 3
            ? std::vector<std::string>{
                "calm", "joy", "anger", "sadness", "whisper"
            }
            : std::vector<std::string>{};
        if (expected_axes.empty()) {
            throw std::runtime_error(
                "Unsupported affect axis_order_version '" + std::to_string(axis_version) + "'"
            );
        }
        const std::string expected_graph_contract =
            "scyllasband_affect_v" + std::to_string(axis_version);
        if (info.affect_graph_input_contract != expected_graph_contract) {
            throw std::runtime_error(
                "Unsupported affect graph_input_contract '" + info.affect_graph_input_contract + "'"
            );
        }
        if (info.affect_axes != expected_axes) {
            throw std::runtime_error(
                "Affect axes do not match axis_order_version " + std::to_string(axis_version)
            );
        }
        // The affect contract is self-describing: axis_order_version alone fixes
        // both the axis names and the graph input contract, and both are checked
        // above, so v1 'questioning' can never be read as v2 'whisper'.
        // model_version is release metadata and is deliberately NOT cross-checked
        // against the axis order -- doing so hardcodes a release-numbering scheme
        // into the runtime. Keep this in sync with scyllasband/contract.py.
        auto validate_presets = [&](const std::map<std::string, std::vector<float>>& presets) {
            for (const auto& item : presets) {
                if (item.second.size() != expected_axes.size()) {
                    throw std::runtime_error(
                        "Affect preset '" + item.first + "' must contain " +
                        std::to_string(expected_axes.size()) + " values"
                    );
                }
                for (std::size_t index = 0; index < item.second.size(); ++index) {
                    const float value = item.second[index];
                    if (!std::isfinite(value) || value < 0.0f || value > 1.0f) {
                        throw std::runtime_error("Affect preset '" + item.first + "' contains an invalid value");
                    }
                    const std::string& axis = expected_axes[index];
                    const float minimum = info.affect_axis_minimums.count(axis) > 0
                        ? info.affect_axis_minimums.at(axis)
                        : 0.0f;
                    const float maximum = info.affect_axis_maximums.count(axis) > 0
                        ? info.affect_axis_maximums.at(axis)
                        : 1.0f;
                    if (value < minimum || value > maximum) {
                        throw std::runtime_error(
                            "Affect preset '" + item.first + "' violates bounds for '" + axis + "'"
                        );
                    }
                }
            }
        };
        auto validate_axis_policy = [&](const std::map<std::string, float>& policy,
                                        const std::string& label) {
            for (const auto& item : policy) {
                if (std::find(expected_axes.begin(), expected_axes.end(), item.first) == expected_axes.end()) {
                    throw std::runtime_error(
                        "Affect " + label + " contains unknown axis '" + item.first + "'"
                    );
                }
                if (!std::isfinite(item.second) || item.second < 0.0f || item.second > 1.0f) {
                    throw std::runtime_error(
                        "Affect " + label + " contains an invalid value for '" + item.first + "'"
                    );
                }
            }
        };
        validate_axis_policy(info.affect_partial_defaults, "partial_defaults");
        validate_axis_policy(info.affect_axis_minimums, "axis_minimums");
        validate_axis_policy(info.affect_axis_maximums, "axis_maximums");
        for (const std::string& axis : expected_axes) {
            const float minimum = info.affect_axis_minimums.count(axis) > 0
                ? info.affect_axis_minimums.at(axis)
                : 0.0f;
            const float maximum = info.affect_axis_maximums.count(axis) > 0
                ? info.affect_axis_maximums.at(axis)
                : 1.0f;
            if (minimum > maximum) {
                throw std::runtime_error("Affect minimum exceeds maximum for '" + axis + "'");
            }
            const auto partial = info.affect_partial_defaults.find(axis);
            if (partial != info.affect_partial_defaults.end() &&
                (partial->second < minimum || partial->second > maximum)) {
                throw std::runtime_error(
                    "Affect partial default violates bounds for '" + axis + "'"
                );
            }
        }
        validate_presets(info.affect_presets);
        validate_presets(info.affect_legacy_presets);
        if (info.affect_presets.count(info.affect_default_preset) == 0) {
            throw std::runtime_error("Affect default_preset must name a declared preset");
        }
    }
    const std::string word_boundaries = object_for_key(controls, "word_boundaries");
    info.word_boundaries_enabled = bool_for_key(
        word_boundaries, "enabled", false
    );
    info.word_boundary_g2p_output_symbol = string_for_key(
        word_boundaries, "g2p_output_symbol", " "
    );
    info.word_boundary_duration_phone = string_for_key(
        word_boundaries, "duration_phone", "<sil>"
    );
    info.word_boundary_presence_threshold_frames = float_for_key(
        word_boundaries, "presence_threshold_frames", 0.5f
    );
    if (info.word_boundaries_enabled &&
        info.word_boundary_presence_threshold_frames <= 0.0f) {
        throw std::runtime_error(
            "Word-boundary presence threshold must be positive"
        );
    }
    const std::string punctuation_silence = object_for_key(controls, "punctuation_silence");
    info.punctuation_silence_target = string_for_key(
        punctuation_silence,
        "target",
        "merge_into_punctuation"
    );
    if (info.punctuation_silence_target != "explicit_silence" &&
        info.punctuation_silence_target != "merge_into_punctuation") {
        info.punctuation_silence_target = "merge_into_punctuation";
    }
    info.punctuation_pause_floors_calibrated = bool_for_key(
        punctuation_silence,
        "calibrated_floors",
        false
    );
    info.punctuation_pause_floor_table_ms = float_map_for_object(
        object_for_key(punctuation_silence, "floor_table_ms")
    );
    info.punctuation_pause_floor_table_sha256 = string_for_key(
        punctuation_silence,
        "floor_table_sha256"
    );
    if (!info.punctuation_pause_floor_table_ms.empty() &&
        !info.punctuation_pause_floors_calibrated) {
        throw std::runtime_error(
            "Punctuation pause floor table is present but not marked calibrated"
        );
    }
    if (info.punctuation_pause_floors_calibrated &&
        (info.punctuation_pause_floor_table_ms.empty() ||
         info.punctuation_pause_floor_table_sha256.empty())) {
        throw std::runtime_error(
            "Calibrated punctuation pause floors require a table and SHA-256"
        );
    }
    for (const auto& item : info.punctuation_pause_floor_table_ms) {
        if (!std::isfinite(item.second) || item.second < 0.0f) {
            throw std::runtime_error(
                "Invalid punctuation pause floor for '" + item.first + "'"
            );
        }
    }
    const std::string emotions = object_for_key(controls, "emotions");
    info.emotions_enabled = bool_for_key(emotions, "enabled", false);
    const std::string runtime_acceleration = object_for_key(controls, "runtime_acceleration");
    info.runtime_acceleration_metadata_present = !runtime_acceleration.empty();
    if (info.runtime_acceleration_metadata_present) {
        info.runtime_acceleration_backend = string_for_key(runtime_acceleration, "backend");
        info.runtime_acceleration_runtime_version = string_for_key(
            runtime_acceleration,
            "runtime_version"
        );
        const std::string native_gpu = object_for_key(runtime_acceleration, "native_gpu_acceleration");
        if (!native_gpu.empty()) {
            info.runtime_acceleration_cuda_required = bool_for_key(native_gpu, "cuda_required", false);
            if (info.runtime_acceleration_runtime_version.empty()) {
                info.runtime_acceleration_runtime_version = string_for_key(native_gpu, "runtime_version");
            }
        }
        const std::string vector = object_for_key(runtime_acceleration, "vector_estimator");
        if (!vector.empty()) {
            info.runtime_acceleration_vector_execution = string_for_key(vector, "execution");
            info.runtime_acceleration_vector_policy = string_for_key(vector, "policy");
            info.runtime_acceleration_min_gpu_split_vector_latent_frames = int_for_key(
                vector,
                "min_gpu_split_vector_latent_frames",
                0
            );
            info.runtime_acceleration_vector_split_available = bool_for_key(
                vector,
                "split_artifacts_available",
                false
            );
        }
    }

    const std::string components = object_for_key(manifest, "components");
    if (components.empty()) {
        throw std::runtime_error("Bundle manifest is missing components");
    }
    for (const char* component_name : kRequiredComponents) {
        const std::string component = object_for_key(components, component_name);
        if (component.empty()) {
            throw std::runtime_error(std::string("Bundle manifest is missing component: ") + component_name);
        }
        if (!required_component(component)) {
            continue;
        }
        const std::string relative_path = component_artifact_path(component, info.selected_backend);
        if (relative_path.empty()) {
            throw std::runtime_error(
                std::string("Required component '") + component_name + "' has no " + info.selected_backend + " artifact"
            );
        }
        const std::filesystem::path artifact_path = bundle_path / relative_path;
        require_exists(
            artifact_path,
            std::string("Required component '") + component_name + "' artifact '" + info.selected_backend + "' is missing"
        );
        info.component_artifacts[component_name] = relative_path;
        info.component_inputs[component_name] = string_array_for_key(component, "inputs");
        info.component_outputs[component_name] = string_array_for_key(component, "outputs");
    }
    for (const char* component_name : kOptionalComponents) {
        const std::string component = object_for_key(components, component_name);
        if (component.empty()) {
            continue;
        }
        const std::string relative_path = component_artifact_path(component, info.selected_backend);
        if (relative_path.empty()) {
            continue;
        }
        require_exists(
            bundle_path / relative_path,
            std::string("Optional component '") + component_name + "' artifact '" + info.selected_backend + "' is missing"
        );
        info.component_artifacts[component_name] = relative_path;
        info.component_inputs[component_name] = string_array_for_key(component, "inputs");
        info.component_outputs[component_name] = string_array_for_key(component, "outputs");
    }
    auto load_bucket_component = [&](const std::string& component_name, bool required) {
        if (component_name.empty() || info.component_artifacts.count(component_name) > 0) {
            return;
        }
        const std::string component = object_for_key(components, component_name);
        if (component.empty()) {
            if (!required) {
                return;
            }
            throw std::runtime_error("Target bucket component is missing from manifest: " + component_name);
        }
        const std::string relative_path = component_artifact_path(component, info.selected_backend);
        if (relative_path.empty()) {
            if (!required) {
                return;
            }
            throw std::runtime_error("Target bucket component has no " + info.selected_backend + " artifact: " + component_name);
        }
        require_exists(
            bundle_path / relative_path,
            "Target bucket component '" + component_name + "' artifact '" + info.selected_backend + "' is missing"
        );
        info.component_artifacts[component_name] = relative_path;
        info.component_inputs[component_name] = string_array_for_key(component, "inputs");
        info.component_outputs[component_name] = string_array_for_key(component, "outputs");
    };
    const bool require_split_bucket_components = info.selected_backend == "litert";
    for (const ScyllasBandTargetBucketInfo& bucket : info.target_buckets) {
        load_bucket_component(bucket.vector_component, true);
        load_bucket_component(bucket.vocoder_component, true);
        if (!bucket.vector_prefix_component.empty() && !bucket.vector_tail_component.empty()) {
            load_bucket_component(bucket.vector_prefix_component, require_split_bucket_components);
            load_bucket_component(bucket.vector_tail_component, require_split_bucket_components);
        }
    }

    const std::string assets = object_for_key(manifest, "assets");
    info.assets = string_map_for_object(assets);
    for (const auto& item : info.assets) {
        require_exists(
            bundle_path / item.second,
            std::string("Declared asset '") + item.first + "' is missing"
        );
    }
    const auto voice_asset = info.assets.find("voice_index");
    if (voice_asset != info.assets.end()) {
        info.voice_to_id = load_id_index_asset(bundle_path / voice_asset->second);
    }
    const auto language_asset = info.assets.find("language_index");
    if (language_asset != info.assets.end()) {
        info.language_to_id = load_id_index_asset(bundle_path / language_asset->second);
    }
    const auto emotion_asset = info.assets.find("emotion_index");
    if (emotion_asset != info.assets.end()) {
        info.emotion_to_id = load_id_index_asset(bundle_path / emotion_asset->second);
    }
    const auto phone_asset = info.assets.find("phone_vocab");
    if (phone_asset != info.assets.end()) {
        info.phone_to_id = load_phone_vocab_asset(bundle_path / phone_asset->second);
    }
    const auto g2p_tokenizer_asset = info.assets.find("g2p_tokenizer");
    if (g2p_tokenizer_asset != info.assets.end()) {
        const std::string tokenizer_json = read_text_file(bundle_path / g2p_tokenizer_asset->second);
        info.g2p_text_to_id = json_string_int_map_for_object(object_for_key(tokenizer_json, "text_symbols"));
        info.g2p_phoneme_symbols = json_int_string_map_for_object(object_for_key(tokenizer_json, "phoneme_symbols"));
        info.g2p_char_repeats = std::max(1, int_for_key(tokenizer_json, "char_repeats", 1));
        info.g2p_lowercase = bool_for_key(tokenizer_json, "lowercase", true);
        info.g2p_text_pad_index = int_for_key(tokenizer_json, "text_pad_index", 0);
        info.g2p_phoneme_pad_index = int_for_key(tokenizer_json, "phoneme_pad_index", 0);
        info.g2p_phoneme_end_index = int_for_key(tokenizer_json, "phoneme_end_index", 0);
    }
    const auto g2p_config_asset = info.assets.find("g2p_config");
    if (g2p_config_asset != info.assets.end()) {
        const std::string g2p_config_json = read_text_file(bundle_path / g2p_config_asset->second);
        if (info.g2p_text_tokens <= 0) {
            info.g2p_text_tokens = int_for_key(g2p_config_json, "fixed_text_tokens", 0);
        }
        if (info.punctuation_silence_target == "merge_into_punctuation") {
            const std::string target = string_for_key(g2p_config_json, "punctuation_silence_target");
            if (target == "explicit_silence") {
                info.punctuation_silence_target = target;
            }
        }
    }
    const auto g2p_normalization_asset = info.assets.find("g2p_normalization");
    if (g2p_normalization_asset != info.assets.end()) {
        const std::string normalization_json = read_text_file(
            bundle_path / g2p_normalization_asset->second
        );
        info.g2p_punctuation_token_remap = string_map_for_object(
            object_for_key(normalization_json, "punctuation_token_remap")
        );
        info.g2p_punctuation_token_remap_scope = string_for_key(
            normalization_json,
            "punctuation_token_remap_scope",
            "all_boundaries"
        );
        if (info.g2p_punctuation_token_remap_scope != "all_boundaries" &&
            info.g2p_punctuation_token_remap_scope != "continuation_only") {
            throw std::runtime_error("Unsupported punctuation_token_remap_scope in G2P normalization asset");
        }
        for (const auto& item : info.g2p_punctuation_token_remap) {
            if (info.phone_to_id.find(item.first) == info.phone_to_id.end()) {
                throw std::runtime_error(
                    "Punctuation remap source is absent from phone vocabulary: " + item.first
                );
            }
            if (info.phone_to_id.find(item.second) == info.phone_to_id.end()) {
                throw std::runtime_error(
                    "Punctuation remap target is absent from phone vocabulary: " + item.second
                );
            }
        }
    }
    const auto g2p_language_asset = info.assets.find("g2p_language_map");
    if (g2p_language_asset != info.assets.end()) {
        info.g2p_language_map = string_map_for_object(read_text_file(bundle_path / g2p_language_asset->second));
    }
    if (info.g2p_text_tokens <= 0) {
        info.g2p_text_tokens = 512;
    }
    if (info.voice_to_id.empty()) {
        for (std::size_t index = 0; index < info.voices.size(); ++index) {
            info.voice_to_id[info.voices[index]] = static_cast<int>(index);
        }
    }
    if (info.language_to_id.empty()) {
        for (std::size_t index = 0; index < info.languages.size(); ++index) {
            info.language_to_id[info.languages[index]] = static_cast<int>(index);
        }
    }
    if (info.emotion_to_id.empty()) {
        info.emotion_to_id["neutral"] = 0;
    }

    const std::regex embedding_pattern("\\\"embedding_path\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"");
    for (std::sregex_iterator it(manifest.begin(), manifest.end(), embedding_pattern), end; it != end; ++it) {
        require_exists(bundle_path / (*it)[1].str(), "Voice embedding is missing");
    }

    return info;
}

std::string target_bucket_frames_json(const std::vector<ScyllasBandTargetBucketInfo>& buckets) {
    std::ostringstream out;
    out << "[";
    for (std::size_t index = 0; index < buckets.size(); ++index) {
        if (index > 0) {
            out << ",";
        }
        out << buckets[index].latent_frames;
    }
    out << "]";
    return out.str();
}

std::string bundle_summary_json(const ScyllasBandBundleInfo& bundle) {
    const bool split_vector_available =
        bundle.component_artifacts.count("vector_estimator_prefix") > 0 &&
        bundle.component_artifacts.count("vector_estimator_tail") > 0;
    const bool coreai_gpu_ready =
        bundle.runtime_acceleration_metadata_present &&
        bundle.runtime_acceleration_backend == "coreai";
    const bool gpu_ready = coreai_gpu_ready ||
        (split_vector_available && !bundle.runtime_acceleration_cuda_required);
    std::string vector_execution = bundle.runtime_acceleration_vector_execution;
    if (vector_execution.empty()) {
        vector_execution = split_vector_available ? "split_prefix_tail" : "single_graph_or_unknown";
    }

    std::ostringstream metadata;
    metadata << "{"
             << "\"contract_version\":\"" << json_escape(bundle.contract_version) << "\","
             << "\"architecture\":\"" << json_escape(bundle.architecture) << "\","
             << "\"model_name\":\"" << json_escape(bundle.model_name) << "\","
             << "\"model_version\":\"" << json_escape(bundle.model_version) << "\","
             << "\"backend\":\"" << json_escape(bundle.selected_backend) << "\","
             << "\"sample_rate\":" << bundle.sample_rate << ","
             << "\"hop_length\":" << bundle.hop_length << ","
             << "\"latent_hop_length\":" << bundle.latent_hop_length << ","
             << "\"latent_dim\":" << bundle.latent_dim << ","
             << "\"phone_frames\":" << bundle.phone_frames << ","
             << "\"latent_frames\":" << bundle.latent_frames << ","
             << "\"target_bucket_count\":" << bundle.target_buckets.size() << ","
             << "\"target_bucket_margin_frames\":" << bundle.target_bucket_margin_frames << ","
             << "\"target_buckets\":" << target_bucket_frames_json(bundle.target_buckets) << ","
             << "\"voices\":" << bundle.voice_to_id.size() << ","
             << "\"languages\":" << bundle.language_to_id.size() << ","
             << "\"emotions\":" << bundle.emotion_to_id.size() << ","
             << "\"emotions_enabled\":" << (bundle.emotions_enabled ? "true" : "false") << ","
             << "\"components\":" << bundle.component_artifacts.size() << ","
             << "\"vector_split_available\":"
             << (split_vector_available ? "true" : "false") << ","
             << "\"runtime_acceleration\":{"
             << "\"metadata_present\":"
             << (bundle.runtime_acceleration_metadata_present ? "true" : "false") << ","
             << "\"backend\":";
    append_json_string_or_null(metadata, bundle.runtime_acceleration_backend);
    metadata << ",\"runtime_version\":";
    append_json_string_or_null(metadata, bundle.runtime_acceleration_runtime_version);
    metadata << ",\"gpu_ready\":" << (gpu_ready ? "true" : "false")
             << ",\"native_gpu_acceleration\":{"
             << "\"cuda_required\":"
             << (bundle.runtime_acceleration_cuda_required ? "true" : "false")
             << "},\"vector_estimator\":{"
             << "\"execution\":\"" << json_escape(vector_execution) << "\","
             << "\"policy\":";
    append_json_string_or_null(metadata, bundle.runtime_acceleration_vector_policy);
    metadata << ",\"metadata_split_artifacts_available\":"
             << (bundle.runtime_acceleration_vector_split_available ? "true" : "false")
             << ",\"min_gpu_split_vector_latent_frames\":"
             << bundle.runtime_acceleration_min_gpu_split_vector_latent_frames
             << ",\"split_artifacts_available\":"
             << (split_vector_available ? "true" : "false")
             << "}},"
             << "\"reference_packs_enabled\":" << (bundle.reference_packs_enabled ? "true" : "false") << ","
             << "\"reference_style_dim\":" << bundle.reference_style_dim << ","
             << "\"reference_prosody_dim\":" << bundle.reference_prosody_dim << ","
             << "\"prefix_conditioning_enabled\":" << (bundle.prefix_conditioning_enabled ? "true" : "false") << ","
             << "\"prefix_max_frames\":" << bundle.prefix_max_frames << ","
             << "\"span_conditioning_enabled\":" << (bundle.span_conditioning_enabled ? "true" : "false") << ","
             << "\"span_context_hidden_size\":" << bundle.span_context_hidden_size << ","
             << "\"span_context_max_phones\":" << bundle.span_context_max_phones << ","
             << "\"emotion_guidance_enabled\":" << (bundle.emotion_guidance_enabled ? "true" : "false") << ","
             << "\"affect_enabled\":" << (bundle.affect_enabled ? "true" : "false") << ","
             << "\"affect_graph_input_contract\":\"" << json_escape(bundle.affect_graph_input_contract) << "\","
             << "\"affect_axis_count\":" << bundle.affect_axes.size() << ","
             << "\"g2p_punctuation_token_remap_scope\":\""
             << json_escape(bundle.g2p_punctuation_token_remap_scope) << "\","
             << "\"g2p_punctuation_token_remap_entries\":"
             << bundle.g2p_punctuation_token_remap.size() << ","
             << "\"word_boundaries_enabled\":" << (bundle.word_boundaries_enabled ? "true" : "false") << ","
             << "\"word_boundary_duration_phone\":\"" << json_escape(bundle.word_boundary_duration_phone) << "\","
             << "\"word_boundary_presence_threshold_frames\":" << bundle.word_boundary_presence_threshold_frames << ","
             << "\"punctuation_silence_target\":\"" << json_escape(bundle.punctuation_silence_target) << "\","
             << "\"punctuation_pause_floors_calibrated\":"
             << (bundle.punctuation_pause_floors_calibrated ? "true" : "false") << ","
             << "\"punctuation_pause_floor_entries\":"
             << bundle.punctuation_pause_floor_table_ms.size() << ","
             << "\"punctuation_pause_floor_table_sha256\":";
    append_json_string_or_null(metadata, bundle.punctuation_pause_floor_table_sha256);
    metadata
             << "}";
    return metadata.str();
}

}  // namespace scyllasband_detail
