#pragma once

#include "scyllasband.h"
#include "scyllasband_bundle.h"

#include <string>
#include <vector>

namespace scyllasband_detail {

struct ScyllasBandEmotionGuidanceTerm {
    std::string emotion;
    int emotion_id = -1;
    float scale = 1.0f;
};

struct ScyllasBandResolvedRequest {
    std::string voice_id;
    int voice_index = -1;
    std::string language;
    int language_index = -1;
    std::string emotion;
    int emotion_index = -1;
    std::vector<ScyllasBandEmotionGuidanceTerm> emotion_guidance;
    float emotion_guidance_null_weight = 0.0f;
    bool guidance_null_reference = true;
    float emotion_embed_scale = 1.0f;
    std::string affect_requested;
    std::string affect_preset;
    std::vector<float> affect_values;
    float affect_condition_mask = 0.0f;
    float affect_guidance_scale = 1.0f;
    int prefix_latent_dim = 0;
    int prefix_latent_frames = 0;
    int effective_prefix_frames = 0;
    bool prefix_truncated = false;
    std::string context_before;
    std::string context_after;
    int chunk_index = 0;
    int chunk_count = 0;
    std::string boundary_before;
    std::string boundary_after;
    float min_sentence_pause_ms = 0.0f;
    float min_clause_pause_ms = 0.0f;
    bool has_explicit_phones = false;
};

ScyllasBandResolvedRequest resolve_scyllasband_request_context(
    const ScyllasBandBundleInfo& bundle,
    const ScyllasBandSynthesisRequest& request
);

std::string request_context_json(const ScyllasBandResolvedRequest& request);

}  // namespace scyllasband_detail
