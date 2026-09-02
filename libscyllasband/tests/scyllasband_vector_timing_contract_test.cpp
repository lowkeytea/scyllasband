#include "scyllasband_bundle.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

constexpr const char* kPhoneVocab =
    "{\"token_to_id\":{\"<pad>\":0,\"<sil>\":1,\"<end_stmt>\":2,\"ˈ\":3,\"a\":4}}";
constexpr const char* kPhoneVocabSha256 =
    "b64ad8390901411e8b446b5ed7665dd883b6e48d1b0cea8ac4bc33442ee15b35";

void write_file(const std::filesystem::path& path, const std::string& value) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    output << value;
}

std::string timing_control(
    const std::string& features,
    const std::string& digest,
    int punctuation_phone_id
) {
    std::ostringstream out;
    out << "\"vector_timing_conditioning\":{";
    out << "\"enabled\":true,";
    out << "\"schema\":\"scyllasband_vector_timing_conditioning_v1\",";
    out << "\"inputs\":[\"expanded_boundary_event_ids\",\"expanded_modifier_event_ids\","
           "\"expanded_phone_phase\",\"expanded_phone_log_duration\"],";
    out << "\"features\":" << features << ",";
    out << "\"phone_vocab\":{\"asset\":\"assets/phone_vocab.json\","
           "\"sha256\":\"" << digest << "\",\"size\":5},";
    out << "\"boundary_events\":{\"enabled\":true,"
           "\"encoding\":\"phone_vocab_id_plus_synthetic_word_boundary_v1\","
           "\"synthetic_word_boundary_id\":5,"
           "\"punctuation_symbols\":[\"<end_stmt>\"],"
           "\"punctuation_phone_ids\":[" << punctuation_phone_id << "],"
           "\"owners\":{"
           "\"punctuation\":\"following_silence_then_first_surviving_frame\","
           "\"word_boundary\":\"candidate_silence_then_first_surviving_frame\"}},";
    out << "\"modifier_events\":{\"enabled\":true,\"encoding\":\"bitmask_v1\","
           "\"bits\":1,\"symbols\":[\"ˈ\"],\"phone_ids\":[3],\"bit_masks\":[1],"
           "\"ownership\":\"stress_bidirectional_postfix_exact_base_segment_bounded_v2\","
           "\"zero_quantized_fallback\":\"audit_unrepresented_zero_frame_modifier_v1\"},";
    out << "\"local_timing\":{\"enabled\":true,"
           "\"phase\":\"(frame_index_plus_0_5)/phone_duration_frames\","
           "\"log_duration\":\"log1p(phone_duration_frames)\","
           "\"duration_source\":\"final_post_pause_presence_and_floor_frames\"}}";
    return out.str();
}

std::string manifest_json(
    const std::string& features,
    const std::string& digest = kPhoneVocabSha256,
    int punctuation_phone_id = 2,
    bool include_control = true,
    bool include_timing_inputs = true
) {
    const std::string suffix = include_timing_inputs
        ? ",\"expanded_boundary_event_ids\",\"expanded_modifier_event_ids\","
          "\"expanded_phone_phase\",\"expanded_phone_log_duration\""
        : "";
    std::ostringstream out;
    out << "{\"contract_version\":\"1.0.0\","
           "\"model_name\":\"native-vector-timing-test\","
           "\"model_version\":\"2\","
           "\"architecture\":\"scyllasband-duration-flow\","
           "\"preferred_backends\":[\"onnx\"],"
           "\"languages\":[\"en_us\"],\"default_language\":\"en_us\",\"voices\":[],"
           "\"audio\":{\"sample_rate\":24000,\"hop_length\":256,"
           "\"latent_hop_length\":512,\"latent_dim\":4},"
           "\"components\":{"
           "\"g2p\":{\"path\":\"g2p.onnx\",\"inputs\":[],\"outputs\":[\"logits\"]},"
           "\"duration_predictor\":{\"path\":\"duration.onnx\",\"inputs\":[],"
           "\"outputs\":[\"durations\"]},"
           "\"vector_estimator\":{\"path\":\"vector.onnx\",\"inputs\":[\"noise\""
        << suffix
        << "],\"outputs\":[\"velocity\"]},"
           "\"vocoder\":{\"path\":\"vocoder.onnx\",\"inputs\":[],\"outputs\":[\"audio\"]}},"
           "\"controls\":{\"fixed_shapes\":{\"g2p_text_tokens\":8,"
           "\"phone_frames\":8,\"latent_frames\":8}";
    if (include_control) {
        out << "," << timing_control(features, digest, punctuation_phone_id);
    }
    out << "},\"assets\":{\"phone_vocab\":\"assets/phone_vocab.json\"}}";
    return out.str();
}

bool load_fails_with(
    const std::filesystem::path& root,
    const std::string& manifest,
    const std::string& expected
) {
    write_file(root / "manifest.json", manifest);
    try {
        (void)scyllasband_detail::load_scyllasband_bundle_info(
            root.string(),
            SCYLLASBAND_BACKEND_ONNX
        );
    } catch (const std::exception& exc) {
        return std::string(exc.what()).find(expected) != std::string::npos;
    }
    return false;
}

}  // namespace

int main() {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path root = std::filesystem::temp_directory_path() /
        ("scyllasband-vector-timing-" + std::to_string(nonce));
    std::filesystem::create_directories(root / "assets");
    write_file(root / "assets/phone_vocab.json", kPhoneVocab);
    for (const char* name : {"g2p.onnx", "duration.onnx", "vector.onnx", "vocoder.onnx"}) {
        write_file(root / name, "fixture");
    }
    const std::string valid_features =
        "{\"boundary_events\":true,\"modifier_events\":true,\"local_timing\":true}";
    try {
        write_file(root / "manifest.json", manifest_json(valid_features));
        const auto bundle = scyllasband_detail::load_scyllasband_bundle_info(
            root.string(),
            SCYLLASBAND_BACKEND_ONNX
        );
        if (!bundle.vector_timing_enabled || !bundle.vector_boundary_events_enabled ||
            !bundle.vector_modifier_events_enabled || !bundle.vector_local_timing_enabled) {
            throw std::runtime_error("Valid timing feature flags were not loaded");
        }
        if (!load_fails_with(
                root,
                manifest_json("{\"boundary_events\":true,\"modifier_events\":true}"),
                "feature flags"
            ) ||
            !load_fails_with(
                root,
                manifest_json(
                    "{\"boundary_events\":true,\"modifier_events\":true,"
                    "\"local_timing\":true,\"typo\":false}"
                ),
                "feature flags"
            ) ||
            !load_fails_with(
                root,
                manifest_json(
                    "{\"boundary_events\":true,\"modifier_events\":true,"
                    "\"local_timing\":1}"
                ),
                "feature flags"
            )) {
            throw std::runtime_error("Malformed feature objects were not rejected");
        }
        if (!load_fails_with(
                root,
                manifest_json(valid_features, kPhoneVocabSha256, 2, false, true),
                "without controls"
            )) {
            throw std::runtime_error("Timing input leakage without controls was not rejected");
        }
        if (!load_fails_with(
                root,
                manifest_json(valid_features, std::string(64, '0')),
                "SHA-256 mismatch"
            )) {
            throw std::runtime_error("Bad vocabulary hash was not rejected");
        }
        if (!load_fails_with(
                root,
                manifest_json(valid_features, kPhoneVocabSha256, 4),
                "punctuation mapping mismatch"
            )) {
            throw std::runtime_error("Bad punctuation mapping was not rejected");
        }
    } catch (const std::exception& exc) {
        std::filesystem::remove_all(root);
        std::cerr << exc.what() << std::endl;
        return 1;
    }
    std::filesystem::remove_all(root);
    return 0;
}
