#include "scyllasband_execution_plan.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <sstream>
#include <stdexcept>

namespace scyllasband_detail {
namespace {

const char* kExecutionComponents[] = {"g2p", "duration_predictor", "vector_estimator", "vocoder"};

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

std::string normalize_boundary_token(std::string value) {
    value = lower_ascii(trim(value));
    for (char& ch : value) {
        if (ch == '-' || ch == ' ') {
            ch = '_';
        }
    }
    return value;
}

std::vector<std::string> split_whitespace(const std::string& value) {
    std::vector<std::string> parts;
    std::istringstream input(value);
    std::string part;
    while (input >> part) {
        parts.push_back(part);
    }
    return parts;
}

int64_t phone_id_for_token(const ScyllasBandBundleInfo& bundle, const std::string& phone) {
    const auto it = bundle.phone_to_id.find(phone);
    if (it != bundle.phone_to_id.end()) {
        return static_cast<int64_t>(it->second);
    }
    throw std::runtime_error("Explicit phone token '" + phone + "' is not in the bundle phone vocabulary");
}

int64_t pad_phone_id(const ScyllasBandBundleInfo& bundle) {
    const auto it = bundle.phone_to_id.find("<pad>");
    return it == bundle.phone_to_id.end() ? 0 : static_cast<int64_t>(it->second);
}

int64_t boundary_id_for_value(
    std::string value,
    const std::vector<std::string>& ordered_values,
    const std::string& fallback
) {
    value = normalize_boundary_token(value);
    if (value.empty() || value == "auto" || value == "none") {
        value = fallback;
    } else if (value == "start" || value == "utterance_start") {
        value = "paragraph_start";
    } else if (value == "end" || value == "utterance_end") {
        value = "sentence_end";
    } else if (value == "paragraph" || value == "document_start" || value == "doc_start") {
        value = "paragraph_start";
    } else if (value == "document_end" || value == "doc_end") {
        value = "paragraph_end";
    } else if (value == "sentence") {
        value = fallback.find("_start") != std::string::npos ? "sentence_start" : "sentence_end";
    } else if (value == "clause" || value == "continuation" || value == "continue") {
        value = "clause_continue";
    } else if (value == "chunk" || value == "artificial_continue" || value == "budget_continue") {
        value = "chunk_continue";
    }
    for (std::size_t index = 0; index < ordered_values.size(); ++index) {
        if (ordered_values[index] == value) {
            return static_cast<int64_t>(index);
        }
    }
    for (std::size_t index = 0; index < ordered_values.size(); ++index) {
        if (ordered_values[index] == fallback) {
            return static_cast<int64_t>(index);
        }
    }
    return 0;
}

std::string int64_array_json(const std::vector<int64_t>& values) {
    std::ostringstream out;
    out << "[";
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index > 0) {
            out << ",";
        }
        out << values[index];
    }
    out << "]";
    return out.str();
}

bool contains_string(const std::vector<std::string>& values, const std::string& target) {
    return std::find(values.begin(), values.end(), target) != values.end();
}

std::string join_strings(const std::vector<std::string>& values) {
    std::ostringstream out;
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index > 0) {
            out << ", ";
        }
        out << values[index];
    }
    return out.str();
}

const std::vector<std::string>& component_inputs(
    const ScyllasBandBundleInfo& bundle,
    const std::string& component_name
) {
    const auto it = bundle.component_inputs.find(component_name);
    if (it == bundle.component_inputs.end()) {
        throw std::runtime_error("Bundle component '" + component_name + "' does not declare inputs");
    }
    return it->second;
}

const std::vector<std::string>& component_outputs(
    const ScyllasBandBundleInfo& bundle,
    const std::string& component_name
) {
    const auto it = bundle.component_outputs.find(component_name);
    if (it == bundle.component_outputs.end()) {
        static const std::vector<std::string> empty;
        return empty;
    }
    return it->second;
}

void require_component(const ScyllasBandBundleInfo& bundle, const std::string& component_name) {
    if (bundle.component_artifacts.find(component_name) == bundle.component_artifacts.end()) {
        throw std::runtime_error("Bundle is missing required execution component '" + component_name + "'");
    }
}

void require_inputs(
    const ScyllasBandBundleInfo& bundle,
    const std::string& component_name,
    const std::vector<std::string>& required_inputs
) {
    const auto& inputs = component_inputs(bundle, component_name);
    std::vector<std::string> missing;
    for (const std::string& input : required_inputs) {
        if (!contains_string(inputs, input)) {
            missing.push_back(input);
        }
    }
    if (!missing.empty()) {
        throw std::runtime_error(
            "Bundle component '" + component_name + "' is missing required input(s): " +
            join_strings(missing)
        );
    }
}

ScyllasBandComponentExecutionPlan component_plan(
    const ScyllasBandBundleInfo& bundle,
    const std::string& component_name
) {
    const auto artifact = bundle.component_artifacts.find(component_name);
    if (artifact == bundle.component_artifacts.end()) {
        throw std::runtime_error("Bundle is missing required execution component '" + component_name + "'");
    }
    return ScyllasBandComponentExecutionPlan{
        component_name,
        artifact->second,
        component_inputs(bundle, component_name),
        component_outputs(bundle, component_name),
    };
}

void require_fixed_shapes(const ScyllasBandBundleInfo& bundle) {
    if (bundle.sample_rate <= 0 || bundle.hop_length <= 0 || bundle.latent_dim <= 0 ||
        bundle.phone_frames <= 0 || bundle.latent_frames <= 0) {
        throw std::runtime_error("Bundle is missing fixed audio or graph shape metadata");
    }
}

std::string string_array_json(const std::vector<std::string>& values) {
    std::ostringstream out;
    out << "[";
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index > 0) {
            out << ",";
        }
        out << "\"" << json_escape(values[index]) << "\"";
    }
    out << "]";
    return out.str();
}

}  // namespace

ScyllasBandDurationFlowExecutionPlan build_scyllasband_duration_flow_plan(
    const ScyllasBandBundleInfo& bundle,
    const ScyllasBandResolvedRequest& request
) {
    require_fixed_shapes(bundle);
    for (const char* component_name : kExecutionComponents) {
        require_component(bundle, component_name);
    }

    const auto& g2p_inputs = component_inputs(bundle, "g2p");
    if (!contains_string(g2p_inputs, "text_tokens") &&
        !(contains_string(g2p_inputs, "text") && contains_string(g2p_inputs, "language"))) {
        throw std::runtime_error(
            "Bundle component 'g2p' must declare text_tokens or text and language inputs"
        );
    }
    std::vector<std::string> duration_inputs = {
        "phone_ids",
        "voice_id",
        "language_id",
        "boundary_before_id",
        "boundary_after_id",
        "phone_mask",
    };
    std::vector<std::string> vector_inputs = {
        "noise",
        "time",
        "expanded_phone_ids",
        "voice_id",
        "language_id",
        "boundary_before_id",
        "boundary_after_id",
        "latent_mask",
    };
    if (bundle.affect_enabled) {
        duration_inputs.push_back("affect_values");
        duration_inputs.push_back("affect_condition_mask");
        vector_inputs.push_back("affect_values");
        vector_inputs.push_back("affect_condition_mask");
    } else {
        duration_inputs.push_back("emotion_id");
        vector_inputs.push_back("emotion_id");
    }
    require_inputs(bundle, "duration_predictor", duration_inputs);
    require_inputs(bundle, "vector_estimator", vector_inputs);
    require_inputs(bundle, "vocoder", {"latents", "voice_id", "language_id", "emotion_id"});

    if (bundle.reference_packs_enabled && bundle.reference_pack_schema_version == 4) {
        const std::vector<std::string> reference_inputs = {
            "identity_reference",
            "identity_reference_mask",
            "prosody_baseline",
            "prosody_delta",
            "prosody_feature_mask",
            "prosody_confidence",
        };
        require_inputs(bundle, "duration_predictor", reference_inputs);
        require_inputs(bundle, "vector_estimator", reference_inputs);
    } else if (bundle.reference_packs_enabled) {
        for (const std::string& component_name : {"duration_predictor", "vector_estimator", "vocoder"}) {
            require_inputs(bundle, component_name, {"reference_style", "reference_prosody", "reference_mask"});
        }
    }
    if (!bundle.affect_enabled && (bundle.emotion_guidance_enabled || !request.emotion_guidance.empty())) {
        require_inputs(bundle, "duration_predictor", {"emotion_condition_mask"});
        require_inputs(bundle, "vector_estimator", {"emotion_condition_mask"});
        if (bundle.reference_packs_enabled) {
            require_inputs(bundle, "duration_predictor", {"reference_condition_mask"});
            require_inputs(bundle, "vector_estimator", {"reference_condition_mask"});
        }
    }
    if (bundle.prefix_conditioning_enabled || request.effective_prefix_frames > 0) {
        require_inputs(bundle, "vector_estimator", {"prefix_latents", "prefix_mask"});
    }
    if (bundle.span_conditioning_enabled) {
        require_inputs(bundle, "vector_estimator", {"span_context_hidden"});
        require_component(bundle, "vector_context_encoder");
    }

    ScyllasBandDurationFlowExecutionPlan plan;
    plan.sample_rate = bundle.sample_rate;
    plan.hop_length = bundle.hop_length;
    plan.latent_dim = bundle.latent_dim;
    plan.phone_frames = bundle.phone_frames;
    plan.latent_frames = bundle.latent_frames;
    plan.prefix_max_frames = bundle.prefix_max_frames;
    plan.reference_inputs_required = bundle.reference_packs_enabled;
    plan.prefix_inputs_required = bundle.prefix_conditioning_enabled;
    plan.span_inputs_required = bundle.span_conditioning_enabled;
    plan.emotion_guidance_inputs_required = bundle.emotion_guidance_enabled;
    plan.affect_inputs_required = bundle.affect_enabled;
    plan.has_emotion_guidance_terms = !request.emotion_guidance.empty();
    for (const char* component_name : kExecutionComponents) {
        plan.components.push_back(component_plan(bundle, component_name));
    }
    return plan;
}

std::string execution_plan_json(const ScyllasBandDurationFlowExecutionPlan& plan) {
    std::ostringstream metadata;
    metadata << "{"
             << "\"sample_rate\":" << plan.sample_rate << ","
             << "\"hop_length\":" << plan.hop_length << ","
             << "\"latent_dim\":" << plan.latent_dim << ","
             << "\"phone_frames\":" << plan.phone_frames << ","
             << "\"latent_frames\":" << plan.latent_frames << ","
             << "\"prefix_max_frames\":" << plan.prefix_max_frames << ","
             << "\"reference_inputs_required\":" << (plan.reference_inputs_required ? "true" : "false") << ","
             << "\"prefix_inputs_required\":" << (plan.prefix_inputs_required ? "true" : "false") << ","
             << "\"span_inputs_required\":" << (plan.span_inputs_required ? "true" : "false") << ","
             << "\"emotion_guidance_inputs_required\":" << (plan.emotion_guidance_inputs_required ? "true" : "false") << ","
             << "\"affect_inputs_required\":" << (plan.affect_inputs_required ? "true" : "false") << ","
             << "\"has_emotion_guidance_terms\":" << (plan.has_emotion_guidance_terms ? "true" : "false") << ","
             << "\"components\":[";
    for (std::size_t index = 0; index < plan.components.size(); ++index) {
        const auto& component = plan.components[index];
        if (index > 0) {
            metadata << ",";
        }
        metadata << "{"
                 << "\"name\":\"" << json_escape(component.name) << "\","
                 << "\"artifact_path\":\"" << json_escape(component.artifact_path) << "\","
                 << "\"inputs\":" << string_array_json(component.inputs) << ","
                 << "\"outputs\":" << string_array_json(component.outputs)
                 << "}";
    }
    metadata << "]}";
    return metadata.str();
}


ScyllasBandDurationFlowPreparedInputs prepare_scyllasband_duration_flow_inputs(
    const ScyllasBandBundleInfo& bundle,
    const ScyllasBandResolvedRequest& resolved_request,
    const ScyllasBandSynthesisRequest& request
) {
    require_fixed_shapes(bundle);
    if (request.explicit_phones == nullptr || request.explicit_phones[0] == '\0') {
        throw std::runtime_error("Native duration-flow input preparation currently requires explicit_phones");
    }

    ScyllasBandDurationFlowPreparedInputs inputs;
    inputs.phones = split_whitespace(request.explicit_phones);
    inputs.phone_count = static_cast<int>(inputs.phones.size());
    if (inputs.phone_count <= 0) {
        throw std::runtime_error("explicit_phones did not contain any phone tokens");
    }
    if (inputs.phone_count > bundle.phone_frames) {
        throw std::runtime_error(
            "Explicit phone sequence has " + std::to_string(inputs.phone_count) +
            " phones, but this bundle supports at most " + std::to_string(bundle.phone_frames)
        );
    }

    const int64_t pad_id = pad_phone_id(bundle);
    inputs.phone_ids.assign(static_cast<std::size_t>(bundle.phone_frames), pad_id);
    inputs.phone_mask.assign(static_cast<std::size_t>(bundle.phone_frames), 0);
    inputs.active_phone_ids.reserve(inputs.phones.size());
    for (std::size_t index = 0; index < inputs.phones.size(); ++index) {
        const int64_t phone_id = phone_id_for_token(bundle, inputs.phones[index]);
        inputs.active_phone_ids.push_back(phone_id);
        inputs.phone_ids[index] = phone_id;
        inputs.phone_mask[index] = 1;
    }

    inputs.voice_id = resolved_request.voice_index;
    inputs.language_id = resolved_request.language_index;
    inputs.emotion_id = resolved_request.emotion_index;
    inputs.affect_values = resolved_request.affect_values;
    inputs.affect_condition_mask = resolved_request.affect_condition_mask;
    const std::size_t affect_mask_size = (
        bundle.affect_enabled && bundle.affect_axis_order_version == "3"
    ) ? inputs.affect_values.size() : 1U;
    inputs.affect_condition_mask_values.assign(
        affect_mask_size, inputs.affect_condition_mask
    );
    inputs.boundary_before_id = boundary_id_for_value(
        resolved_request.boundary_before,
        {"sentence_start", "paragraph_start", "clause_continue", "chunk_continue"},
        "sentence_start"
    );
    inputs.boundary_after_id = boundary_id_for_value(
        resolved_request.boundary_after,
        {"sentence_end", "paragraph_end", "clause_continue", "chunk_continue"},
        "sentence_end"
    );

    inputs.reference_style.assign(static_cast<std::size_t>(std::max(0, bundle.reference_style_dim)), 0.0f);
    inputs.reference_prosody.assign(static_cast<std::size_t>(std::max(0, bundle.reference_prosody_dim)), 0.0f);
    inputs.reference_mask = 0.0f;
    inputs.identity_reference.assign(static_cast<std::size_t>(std::max(0, bundle.reference_identity_dim)), 0.0f);
    inputs.identity_reference_mask = 0.0f;
    inputs.prosody_baseline.assign(static_cast<std::size_t>(std::max(0, bundle.reference_baseline_dim)), 0.0f);
    inputs.prosody_delta.assign(static_cast<std::size_t>(std::max(0, bundle.reference_delta_dim)), 0.0f);
    inputs.prosody_feature_mask.assign(static_cast<std::size_t>(std::max(0, bundle.reference_prosody_dim)), 0.0f);
    inputs.prosody_confidence = 0.0f;

    const int prefix_frames = bundle.prefix_conditioning_enabled ? std::max(0, bundle.prefix_max_frames) : 0;
    inputs.prefix_latents.assign(static_cast<std::size_t>(std::max(0, bundle.latent_dim) * prefix_frames), 0.0f);
    inputs.prefix_mask.assign(static_cast<std::size_t>(prefix_frames), 0);
    inputs.prefix_frames_used = std::min(resolved_request.effective_prefix_frames, prefix_frames);
    inputs.prefix_truncated = resolved_request.prefix_truncated;
    if (request.prefix_latents != nullptr && inputs.prefix_frames_used > 0 && bundle.latent_dim > 0) {
        const int source_frames = request.prefix_latent_frames;
        const int keep = inputs.prefix_frames_used;
        for (int dim = 0; dim < bundle.latent_dim; ++dim) {
            for (int frame = 0; frame < keep; ++frame) {
                const int source_frame = source_frames - keep + frame;
                const int target_frame = prefix_frames - keep + frame;
                inputs.prefix_latents[static_cast<std::size_t>(dim * prefix_frames + target_frame)] =
                    request.prefix_latents[static_cast<std::size_t>(dim * source_frames + source_frame)];
            }
        }
        for (int frame = prefix_frames - keep; frame < prefix_frames; ++frame) {
            inputs.prefix_mask[static_cast<std::size_t>(frame)] = 1;
        }
    }
    return inputs;
}

std::string prepared_inputs_json(const ScyllasBandDurationFlowPreparedInputs& inputs) {
    std::ostringstream metadata;
    metadata << "{"
             << "\"phone_source\":\"" << json_escape(inputs.phone_source) << "\","
             << "\"phone_count\":" << inputs.phone_count << ","
             << "\"phones\":" << string_array_json(inputs.phones) << ","
             << "\"active_phone_ids\":" << int64_array_json(inputs.active_phone_ids) << ","
             << "\"fixed_phone_frames\":" << inputs.phone_ids.size() << ","
             << "\"voice_id\":" << inputs.voice_id << ","
             << "\"language_id\":" << inputs.language_id << ","
             << "\"emotion_id\":" << inputs.emotion_id << ","
             << "\"affect_vector\":[";
    for (std::size_t index = 0; index < inputs.affect_values.size(); ++index) {
        if (index > 0) metadata << ",";
        metadata << inputs.affect_values[index];
    }
    metadata << "],\"affect_condition_mask\":" << inputs.affect_condition_mask << ","
             << "\"boundary_before_id\":" << inputs.boundary_before_id << ","
             << "\"boundary_after_id\":" << inputs.boundary_after_id << ","
             << "\"reference_style_dim\":" << inputs.reference_style.size() << ","
             << "\"reference_prosody_dim\":" << inputs.reference_prosody.size() << ","
             << "\"reference_mask\":" << inputs.reference_mask << ","
             << "\"native_reference_mask\":" << inputs.native_reference_mask << ","
             << "\"fallback_reference_mask\":" << inputs.fallback_reference_mask << ","
             << "\"reference_key\":\"" << json_escape(inputs.reference_key) << "\","
             << "\"reference_pack_path\":\"" << json_escape(inputs.reference_pack_path) << "\","
             << "\"prefix_frames_used\":" << inputs.prefix_frames_used << ","
             << "\"prefix_mask_frames\":" << inputs.prefix_mask.size() << ","
             << "\"prefix_truncated\":" << (inputs.prefix_truncated ? "true" : "false") << ","
             << "\"span_context\":" << inputs.span_context_metadata
             << "}";
    return metadata.str();
}


}  // namespace scyllasband_detail
