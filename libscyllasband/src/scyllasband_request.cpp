#include "scyllasband_request.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
#include <sstream>
#include <stdexcept>

namespace scyllasband_detail {
namespace {

std::string nullable_string(const char* value) {
    return value == nullptr ? std::string() : std::string(value);
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

std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::string trim(const std::string& value) {
    std::size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) {
        ++start;
    }
    std::size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    return value.substr(start, end - start);
}

std::string normalize_token(std::string value) {
    value = lower_ascii(trim(value));
    for (char& ch : value) {
        if (ch == '-' || ch == ' ') {
            ch = '_';
        }
    }
    return value;
}

std::string normalize_language(const char* raw) {
    std::string language = normalize_token(nullable_string(raw));
    if (language.empty() || language == "auto") {
        return {};
    }
    if (language == "en" || language == "eng" || language == "english") {
        return "en";
    }
    if (language == "en_us" || language == "eng_us" || language == "english_us" ||
        language == "us" || language == "american" || language == "american_english") {
        return "en_us";
    }
    if (language == "en_gb" || language == "en_uk" || language == "eng_gb" ||
        language == "english_gb" || language == "english_uk" || language == "gb" ||
        language == "uk" || language == "british" || language == "british_english") {
        return "en_gb";
    }
    if (language == "es" || language == "es_mx" || language == "es_es" ||
        language == "spa" || language == "spanish") {
        return "es";
    }
    if (language == "it" || language == "it_it" || language == "ita" ||
        language == "italian") {
        return "it";
    }
    if (language == "fr" || language == "fr_fr" || language == "fra" ||
        language == "fre" || language == "french") {
        return "fr";
    }
    if (language == "de" || language == "de_de" || language == "deu" ||
        language == "ger" || language == "german") {
        return "de";
    }
    if (language == "vi" || language == "vi_vn" || language == "vi_hn" ||
        language == "vie" || language == "vietnamese") {
        return "vi";
    }
    return language;
}

std::string normalize_emotion(const std::string& raw) {
    std::string emotion = normalize_token(raw.empty() ? "neutral" : raw);
    if (emotion.empty() || emotion == "auto" || emotion == "default" || emotion == "none") {
        return "neutral";
    }
    return emotion;
}

std::string sorted_keys(const std::map<std::string, int>& values) {
    std::ostringstream out;
    bool first = true;
    for (const auto& item : values) {
        if (!first) {
            out << ", ";
        }
        first = false;
        out << item.first;
    }
    return out.str();
}

int lookup_index(const std::map<std::string, int>& values, const std::string& key, const std::string& label) {
    if (values.empty()) {
        return -1;
    }
    const auto it = values.find(key);
    if (it != values.end()) {
        return it->second;
    }
    throw std::runtime_error("Unknown " + label + " '" + key + "'; available: " + sorted_keys(values));
}

bool contains_string(const std::vector<std::string>& values, const std::string& target) {
    return std::find(values.begin(), values.end(), target) != values.end();
}

std::vector<std::string> split_commas(const std::string& value) {
    std::vector<std::string> parts;
    std::size_t start = 0;
    while (start <= value.size()) {
        const std::size_t comma = value.find(',', start);
        const std::size_t end = comma == std::string::npos ? value.size() : comma;
        parts.push_back(value.substr(start, end - start));
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return parts;
}

std::vector<ScyllasBandEmotionGuidanceTerm> parse_emotion_guidance(
    const ScyllasBandBundleInfo& bundle,
    const char* raw_spec
) {
    std::vector<ScyllasBandEmotionGuidanceTerm> terms;
    const std::string spec = trim(nullable_string(raw_spec));
    if (spec.empty()) {
        return terms;
    }
    for (const std::string& raw_part : split_commas(spec)) {
        const std::string part = trim(raw_part);
        if (part.empty()) {
            continue;
        }
        const std::size_t colon = part.find(':');
        const std::string emotion_name = colon == std::string::npos ? part : part.substr(0, colon);
        const std::string scale_text = colon == std::string::npos ? std::string() : trim(part.substr(colon + 1));
        float scale = 1.0f;
        if (!scale_text.empty()) {
            std::size_t consumed = 0;
            scale = std::stof(scale_text, &consumed);
            if (consumed != scale_text.size() || !std::isfinite(scale)) {
                throw std::runtime_error("Invalid emotion guidance scale '" + scale_text + "'");
            }
            if (scale < 0.0f) {
                throw std::runtime_error("Emotion guidance scale must be non-negative: '" + scale_text + "'");
            }
        }
        const std::string emotion = normalize_emotion(emotion_name);
        terms.push_back(ScyllasBandEmotionGuidanceTerm{
            emotion,
            lookup_index(bundle.emotion_to_id, emotion, "emotion"),
            scale,
        });
    }
    return terms;
}

float emotion_guidance_null_weight(const std::vector<ScyllasBandEmotionGuidanceTerm>& terms) {
    if (terms.empty()) {
        return 0.0f;
    }
    float total = 0.0f;
    for (const auto& term : terms) {
        total += term.scale;
    }
    return 1.0f - total;
}

struct AffectResolution {
    std::string requested;
    std::string preset;
    std::vector<float> values;
    float guidance_scale = 1.0f;
};

std::string preset_keys(const ScyllasBandBundleInfo& bundle) {
    std::map<std::string, int> names;
    for (const auto& item : bundle.affect_presets) names[item.first] = 1;
    for (const auto& item : bundle.affect_legacy_presets) names[item.first] = 1;
    return sorted_keys(names);
}

AffectResolution resolve_affect(
    const ScyllasBandBundleInfo& bundle,
    const ScyllasBandSynthesisRequest& request
) {
    const std::string explicit_affect = trim(nullable_string(request.affect));
    if (!bundle.affect_enabled) {
        if (!explicit_affect.empty() ||
            (request.has_affect_guidance_scale != 0 && request.affect_guidance_scale != 1.0f)) {
            throw std::runtime_error("This bundle does not support six-axis affect conditioning");
        }
        return AffectResolution{};
    }
    if (!trim(nullable_string(request.emotion_guidance)).empty()) {
        throw std::runtime_error(
            "Categorical emotion guidance is not supported by six-axis affect bundles"
        );
    }

    const std::string raw_emotion = normalize_emotion(nullable_string(request.emotion));
    if (!explicit_affect.empty() && raw_emotion != "neutral") {
        throw std::runtime_error("Pass either affect values or a legacy emotion preset, not both");
    }

    AffectResolution resolved;
    std::string spec = explicit_affect;
    resolved.requested = explicit_affect;
    if (spec.empty()) {
        if (raw_emotion != "neutral") {
            spec = raw_emotion;
            resolved.requested = raw_emotion;
        } else {
            spec = bundle.affect_default_preset;
        }
    }

    if (spec.find('=') == std::string::npos) {
        const std::string name = normalize_token(spec);
        auto preset = bundle.affect_presets.find(name);
        if (preset != bundle.affect_presets.end()) {
            resolved.preset = name;
            resolved.values = preset->second;
        } else {
            const auto legacy = bundle.affect_legacy_presets.find(name);
            if (legacy == bundle.affect_legacy_presets.end()) {
                throw std::runtime_error(
                    "Unknown affect preset '" + name + "'; available: " + preset_keys(bundle)
                );
            }
            resolved.preset = name;
            resolved.values = legacy->second;
        }
    } else {
        resolved.values.assign(bundle.affect_axes.size(), 0.0f);
        std::map<std::string, bool> seen;
        for (const std::string& raw_part : split_commas(spec)) {
            const std::string part = trim(raw_part);
            const std::size_t equals = part.find('=');
            if (equals == std::string::npos || equals == 0 || equals + 1 >= part.size()) {
                throw std::runtime_error("Invalid affect term '" + part + "'; expected axis=value");
            }
            const std::string axis = normalize_token(part.substr(0, equals));
            if (seen[axis]) {
                throw std::runtime_error("Duplicate affect axis '" + axis + "'");
            }
            seen[axis] = true;
            const auto axis_it = std::find(bundle.affect_axes.begin(), bundle.affect_axes.end(), axis);
            if (axis_it == bundle.affect_axes.end()) {
                throw std::runtime_error("Unknown affect axis '" + axis + "'");
            }
            const std::string value_text = trim(part.substr(equals + 1));
            std::size_t consumed = 0;
            const float value = std::stof(value_text, &consumed);
            if (consumed != value_text.size() || !std::isfinite(value) || value < 0.0f || value > 1.0f) {
                throw std::runtime_error(
                    "Affect value for '" + axis + "' must be finite and within [0, 1]"
                );
            }
            resolved.values[static_cast<std::size_t>(axis_it - bundle.affect_axes.begin())] = value;
        }
    }
    if (resolved.values.size() != bundle.affect_axes.size()) {
        throw std::runtime_error("Resolved affect vector does not match the bundle axis count");
    }
    resolved.guidance_scale = request.has_affect_guidance_scale != 0
        ? request.affect_guidance_scale
        : bundle.affect_guidance_default_scale;
    if (!std::isfinite(resolved.guidance_scale) || resolved.guidance_scale < 0.0f) {
        throw std::runtime_error("affect_guidance_scale must be finite and non-negative");
    }
    return resolved;
}

std::string resolve_language(
    const ScyllasBandBundleInfo& bundle,
    const std::string& voice_id,
    const char* raw_language
) {
    const std::string requested = normalize_language(raw_language);
    std::string voice_default = bundle.default_language;
    const auto default_it = bundle.voice_default_language.find(voice_id);
    if (default_it != bundle.voice_default_language.end() && !default_it->second.empty()) {
        voice_default = default_it->second;
    }

    std::string resolved;
    if (requested.empty() || requested == "en") {
        resolved = voice_default;
    } else {
        resolved = requested;
    }
    if (resolved.empty()) {
        resolved = requested.empty() ? "auto" : requested;
    }
    return resolved;
}

std::pair<std::string, int> resolve_emotion(
    const ScyllasBandBundleInfo& bundle,
    const ScyllasBandSynthesisRequest& request,
    const std::vector<ScyllasBandEmotionGuidanceTerm>& guidance_terms
) {
    if (!guidance_terms.empty()) {
        const auto primary = std::max_element(
            guidance_terms.begin(),
            guidance_terms.end(),
            [](const ScyllasBandEmotionGuidanceTerm& lhs, const ScyllasBandEmotionGuidanceTerm& rhs) {
                return lhs.scale < rhs.scale;
            }
        );
        return {primary->emotion, primary->emotion_id};
    }
    const std::string emotion = normalize_emotion(nullable_string(request.emotion));
    return {emotion, lookup_index(bundle.emotion_to_id, emotion, "emotion")};
}

}  // namespace

ScyllasBandResolvedRequest resolve_scyllasband_request_context(
    const ScyllasBandBundleInfo& bundle,
    const ScyllasBandSynthesisRequest& request
) {
    ScyllasBandResolvedRequest resolved;
    resolved.voice_id = nullable_string(request.voice_id);
    resolved.voice_index = lookup_index(bundle.voice_to_id, resolved.voice_id, "voice");

    resolved.language = resolve_language(bundle, resolved.voice_id, request.language);
    resolved.language_index = lookup_index(bundle.language_to_id, resolved.language, "language");
    const auto voice_languages = bundle.voice_languages.find(resolved.voice_id);
    if (voice_languages != bundle.voice_languages.end() &&
        !voice_languages->second.empty() &&
        !contains_string(voice_languages->second, resolved.language)) {
        std::ostringstream options;
        for (std::size_t index = 0; index < voice_languages->second.size(); ++index) {
            if (index > 0) {
                options << ", ";
            }
            options << voice_languages->second[index];
        }
        throw std::runtime_error(
            "Unsupported language '" + resolved.language + "' for voice '" + resolved.voice_id +
            "'; available: " + options.str()
        );
    }

    if (bundle.affect_enabled) {
        const AffectResolution affect = resolve_affect(bundle, request);
        resolved.affect_requested = affect.requested;
        resolved.affect_preset = affect.preset;
        resolved.affect_values = affect.values;
        resolved.affect_condition_mask = 1.0f;
        resolved.affect_guidance_scale = affect.guidance_scale;
        resolved.emotion = "neutral";
        const auto neutral = bundle.emotion_to_id.find("neutral");
        resolved.emotion_index = neutral == bundle.emotion_to_id.end() ? 0 : neutral->second;
    } else {
        resolve_affect(bundle, request);
        resolved.emotion_guidance = parse_emotion_guidance(bundle, request.emotion_guidance);
        const auto emotion = resolve_emotion(bundle, request, resolved.emotion_guidance);
        resolved.emotion = emotion.first;
        resolved.emotion_index = emotion.second;
        resolved.emotion_guidance_null_weight = emotion_guidance_null_weight(resolved.emotion_guidance);
    }
    resolved.guidance_null_reference = request.guidance_null_reference != 0;
    resolved.emotion_embed_scale = request.emotion_embed_scale;
    resolved.prefix_latent_dim = request.prefix_latent_dim;
    resolved.prefix_latent_frames = request.prefix_latent_frames;
    if (request.prefix_latent_frames > 0 || request.prefix_latent_dim > 0) {
        if (bundle.latent_dim > 0 && request.prefix_latent_dim != bundle.latent_dim) {
            throw std::runtime_error(
                "prefix_latent_dim " + std::to_string(request.prefix_latent_dim) +
                " does not match bundle latent_dim " + std::to_string(bundle.latent_dim)
            );
        }
        if (bundle.prefix_conditioning_enabled && bundle.prefix_max_frames > 0) {
            resolved.effective_prefix_frames = std::min(request.prefix_latent_frames, bundle.prefix_max_frames);
            resolved.prefix_truncated = request.prefix_latent_frames > bundle.prefix_max_frames;
        }
    }
    resolved.context_before = nullable_string(request.context_before);
    resolved.context_after = nullable_string(request.context_after);
    resolved.chunk_index = request.chunk_index;
    resolved.chunk_count = request.chunk_count;
    resolved.boundary_before = nullable_string(request.boundary_before);
    resolved.boundary_after = nullable_string(request.boundary_after);
    resolved.min_sentence_pause_ms = std::max(0.0f, request.min_sentence_pause_ms);
    resolved.min_clause_pause_ms = std::max(0.0f, request.min_clause_pause_ms);
    resolved.has_explicit_phones = request.explicit_phones != nullptr && request.explicit_phones[0] != '\0';
    return resolved;
}

std::string request_context_json(const ScyllasBandResolvedRequest& request) {
    std::ostringstream metadata;
    metadata << "{"
             << "\"resolved_voice\":\"" << json_escape(request.voice_id) << "\","
             << "\"voice_index\":" << request.voice_index << ","
             << "\"resolved_language\":\"" << json_escape(request.language) << "\","
             << "\"language_index\":" << request.language_index << ","
             << "\"resolved_emotion\":\"" << json_escape(request.emotion) << "\","
             << "\"emotion_index\":" << request.emotion_index << ","
             << "\"emotion_guidance\":";
    if (request.emotion_guidance.empty()) {
        metadata << "null";
    } else {
        metadata << "[";
        for (std::size_t index = 0; index < request.emotion_guidance.size(); ++index) {
            const auto& term = request.emotion_guidance[index];
            if (index > 0) {
                metadata << ",";
            }
            metadata << "{"
                     << "\"emotion\":\"" << json_escape(term.emotion) << "\","
                     << "\"scale\":" << term.scale
                     << "}";
        }
        metadata << "]";
    }
    metadata << ",\"affect_requested\":";
    if (request.affect_requested.empty()) {
        metadata << "null";
    } else {
        metadata << "\"" << json_escape(request.affect_requested) << "\"";
    }
    metadata << ",\"affect_preset\":";
    if (request.affect_preset.empty()) {
        metadata << "null";
    } else {
        metadata << "\"" << json_escape(request.affect_preset) << "\"";
    }
    metadata << ",\"affect_vector\":[";
    for (std::size_t index = 0; index < request.affect_values.size(); ++index) {
        if (index > 0) metadata << ",";
        metadata << request.affect_values[index];
    }
    metadata << "]"
             << ",\"affect_condition_mask\":" << request.affect_condition_mask
             << ",\"affect_guidance_scale\":" << request.affect_guidance_scale;
    metadata << ","
             << "\"emotion_guidance_null_weight\":" << request.emotion_guidance_null_weight << ","
             << "\"guidance_null_reference\":" << (request.guidance_null_reference ? "true" : "false") << ","
             << "\"emotion_embed_scale\":" << request.emotion_embed_scale << ","
             << "\"prefix_latent_dim\":" << request.prefix_latent_dim << ","
             << "\"prefix_latent_frames\":" << request.prefix_latent_frames << ","
             << "\"effective_prefix_frames\":" << request.effective_prefix_frames << ","
             << "\"prefix_truncated\":" << (request.prefix_truncated ? "true" : "false") << ","
             << "\"context_before\":\"" << json_escape(request.context_before) << "\","
             << "\"context_after\":\"" << json_escape(request.context_after) << "\","
             << "\"chunk_index\":" << request.chunk_index << ","
             << "\"chunk_count\":" << request.chunk_count << ","
             << "\"boundary_before\":\"" << json_escape(request.boundary_before) << "\","
             << "\"boundary_after\":\"" << json_escape(request.boundary_after) << "\","
             << "\"min_sentence_pause_ms\":" << request.min_sentence_pause_ms << ","
             << "\"min_clause_pause_ms\":" << request.min_clause_pause_ms << ","
             << "\"has_explicit_phones\":" << (request.has_explicit_phones ? "true" : "false")
             << "}";
    return metadata.str();
}

}  // namespace scyllasband_detail
