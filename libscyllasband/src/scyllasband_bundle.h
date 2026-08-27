#pragma once

#include "scyllasband.h"

#include <map>
#include <string>
#include <vector>

namespace scyllasband_detail {

struct ScyllasBandTargetBucketInfo {
    int latent_frames = 0;
    std::string vector_component = "vector_estimator";
    std::string vector_prefix_component = "vector_estimator_prefix";
    std::string vector_tail_component = "vector_estimator_tail";
    std::string vocoder_component = "vocoder";
};

struct ScyllasBandBundleInfo {
    std::string bundle_dir;
    std::string contract_version;
    std::string architecture;
    std::string model_name;
    std::string model_version = "3";
    std::string selected_backend;
    std::vector<std::string> preferred_backends;
    std::map<std::string, std::string> assets;
    std::map<std::string, std::string> component_artifacts;
    std::map<std::string, std::vector<std::string>> component_inputs;
    std::map<std::string, std::vector<std::string>> component_outputs;
    std::vector<std::string> languages;
    std::string default_language;
    std::vector<std::string> voices;
    std::map<std::string, int> voice_to_id;
    std::map<std::string, int> language_to_id;
    std::map<std::string, int> emotion_to_id;
    std::map<std::string, int> phone_to_id;
    std::map<std::string, int> g2p_text_to_id;
    std::map<int, std::string> g2p_phoneme_symbols;
    std::map<std::string, std::string> g2p_language_map;
    std::map<std::string, std::string> g2p_punctuation_token_remap;
    std::string g2p_punctuation_token_remap_scope = "all_boundaries";
    bool word_boundaries_enabled = false;
    std::string word_boundary_g2p_output_symbol = " ";
    std::string word_boundary_duration_phone = "<sil>";
    float word_boundary_presence_threshold_frames = 0.5f;
    std::map<std::string, std::string> voice_default_language;
    std::map<std::string, std::vector<std::string>> voice_languages;
    int sample_rate = 0;
    int hop_length = 0;
    int latent_hop_length = 0;
    int latent_dim = 0;
    int g2p_text_tokens = 0;
    int g2p_char_repeats = 1;
    int g2p_text_pad_index = 0;
    int g2p_phoneme_pad_index = 0;
    int g2p_phoneme_end_index = 0;
    int phone_frames = 0;
    int latent_frames = 0;
    int target_bucket_margin_frames = 0;
    std::vector<ScyllasBandTargetBucketInfo> target_buckets;
    bool g2p_lowercase = true;
    bool emotions_enabled = false;
    bool reference_packs_enabled = false;
    int reference_pack_schema_version = 0;
    int reference_style_dim = 0;
    int reference_prosody_dim = 0;
    int reference_identity_dim = 0;
    int reference_baseline_dim = 0;
    int reference_delta_dim = 0;
    std::string reference_affect_routing_version;
    float reference_fallback_weight = 0.25f;
    bool prefix_conditioning_enabled = false;
    int prefix_max_frames = 0;
    bool span_conditioning_enabled = false;
    int span_context_hidden_size = 0;
    int span_context_max_phones = 0;
    bool emotion_guidance_enabled = false;
    bool affect_enabled = false;
    std::string affect_graph_input_contract;
    std::vector<std::string> affect_axes;
    std::string affect_axis_order_version;
    std::string affect_default_preset;
    int affect_preset_version = 0;
    std::map<std::string, std::vector<float>> affect_presets;
    std::map<std::string, std::vector<float>> affect_legacy_presets;
    std::map<std::string, float> affect_partial_defaults;
    std::map<std::string, float> affect_axis_minimums;
    std::map<std::string, float> affect_axis_maximums;
    float affect_guidance_default_scale = 1.0f;
    std::string punctuation_silence_target = "merge_into_punctuation";
    bool punctuation_pause_floors_calibrated = false;
    std::map<std::string, float> punctuation_pause_floor_table_ms;
    std::string punctuation_pause_floor_table_sha256;
    bool runtime_acceleration_metadata_present = false;
    std::string runtime_acceleration_backend;
    std::string runtime_acceleration_runtime_version;
    bool runtime_acceleration_cuda_required = false;
    std::string runtime_acceleration_vector_execution;
    std::string runtime_acceleration_vector_policy;
    int runtime_acceleration_min_gpu_split_vector_latent_frames = 0;
    bool runtime_acceleration_vector_split_available = false;
};

std::string scyllasband_backend_name(ScyllasBandBackend backend);
ScyllasBandBackend scyllasband_select_backend(ScyllasBandBackend backend);
ScyllasBandBundleInfo load_scyllasband_bundle_info(
    const std::string& bundle_dir,
    ScyllasBandBackend backend
);
std::string bundle_summary_json(const ScyllasBandBundleInfo& bundle);

}  // namespace scyllasband_detail
