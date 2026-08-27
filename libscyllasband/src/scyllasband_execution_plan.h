#pragma once

#include "scyllasband_bundle.h"
#include "scyllasband_request.h"

#include <cstdint>
#include <string>
#include <vector>

namespace scyllasband_detail {

struct ScyllasBandComponentExecutionPlan {
    std::string name;
    std::string artifact_path;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
};


struct ScyllasBandDurationFlowPreparedInputs {
    std::string phone_source = "explicit";
    std::vector<std::string> phones;
    std::vector<int64_t> active_phone_ids;
    std::vector<std::string> punctuation_floor_phones;
    std::vector<uint8_t> word_boundary_candidate_mask;
    std::vector<int64_t> phone_ids;
    std::vector<uint8_t> phone_mask;
    int phone_count = 0;
    int64_t voice_id = -1;
    int64_t language_id = -1;
    int64_t emotion_id = -1;
    std::vector<float> affect_values;
    float affect_condition_mask = 0.0f;
    std::vector<float> affect_condition_mask_values;
    int64_t boundary_before_id = 0;
    int64_t boundary_after_id = 0;
    std::vector<float> reference_style;
    std::vector<float> reference_prosody;
    float reference_mask = 0.0f;
    std::vector<float> identity_reference;
    float identity_reference_mask = 0.0f;
    std::vector<float> prosody_baseline;
    std::vector<float> prosody_delta;
    std::vector<float> prosody_feature_mask;
    float prosody_confidence = 0.0f;
    float native_reference_mask = 0.0f;
    float fallback_reference_mask = 0.0f;
    std::string reference_key;
    std::string reference_pack_path;
    std::vector<float> prefix_latents;
    std::vector<uint8_t> prefix_mask;
    int prefix_frames_used = 0;
    bool prefix_truncated = false;
    std::vector<float> span_context_hidden;
    std::string span_context_metadata = "null";
};

struct ScyllasBandDurationFlowExecutionPlan {
    int sample_rate = 0;
    int hop_length = 0;
    int latent_dim = 0;
    int phone_frames = 0;
    int latent_frames = 0;
    int prefix_max_frames = 0;
    bool reference_inputs_required = false;
    bool prefix_inputs_required = false;
    bool span_inputs_required = false;
    bool emotion_guidance_inputs_required = false;
    bool affect_inputs_required = false;
    bool has_emotion_guidance_terms = false;
    std::vector<ScyllasBandComponentExecutionPlan> components;
};

ScyllasBandDurationFlowExecutionPlan build_scyllasband_duration_flow_plan(
    const ScyllasBandBundleInfo& bundle,
    const ScyllasBandResolvedRequest& request
);

std::string execution_plan_json(const ScyllasBandDurationFlowExecutionPlan& plan);

ScyllasBandDurationFlowPreparedInputs prepare_scyllasband_duration_flow_inputs(
    const ScyllasBandBundleInfo& bundle,
    const ScyllasBandResolvedRequest& resolved_request,
    const ScyllasBandSynthesisRequest& request
);

std::string prepared_inputs_json(const ScyllasBandDurationFlowPreparedInputs& inputs);

}  // namespace scyllasband_detail
