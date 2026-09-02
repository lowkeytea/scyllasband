#include "scyllasband_backend.h"
#include "scyllasband_bundle.h"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void write_file(const std::filesystem::path& path, const std::string& value) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    output << value;
}

std::string manifest(const std::string& phrase_boundary) {
    return std::string(
        "{\"contract_version\":\"1.0.0\","
        "\"model_name\":\"duration-hierarchy-test\",\"model_version\":\"2\","
        "\"architecture\":\"scyllasband-duration-flow\","
        "\"preferred_backends\":[\"onnx\"],"
        "\"languages\":[\"en_gb\"],\"default_language\":\"en_gb\","
        "\"voices\":[],\"audio\":{\"sample_rate\":24000,\"hop_length\":256,"
        "\"latent_hop_length\":512,\"latent_dim\":4},"
        "\"components\":{"
        "\"g2p\":{\"path\":\"g2p.onnx\",\"inputs\":[],\"outputs\":[\"logits\"]},"
        "\"duration_predictor\":{\"path\":\"duration.onnx\",\"inputs\":[],"
        "\"outputs\":[\"durations\",\"pause_presence_logits\",\"duration_quantiles\"]},"
        "\"vector_estimator\":{\"path\":\"vector.onnx\",\"inputs\":[],"
        "\"outputs\":[\"velocity\"]},"
        "\"vocoder\":{\"path\":\"vocoder.onnx\",\"inputs\":[],"
        "\"outputs\":[\"audio\"]}},"
        "\"controls\":{\"fixed_shapes\":{\"g2p_text_tokens\":8,"
        "\"phone_frames\":8,\"latent_frames\":8},"
        "\"duration_pause_presence\":{\"enabled\":true,"
        "\"schema\":\"scyllasband_duration_pause_presence_v1\","
        "\"activation\":\"sigmoid\",\"threshold_probability\":0.5,"
        "\"outputs\":[\"durations\",\"pause_presence_logits\"],"
        "\"eligible_pause_owners\":{\"punctuation\":\"following_silence_only\","
        "\"word_boundary\":\"explicit_candidate_mask_only\"}},"
        "\"duration_hierarchy_sampling\":{\"enabled\":true,"
        "\"schema\":\"scyllasband_duration_hierarchy_sampling_v1\","
        "\"policy\":\"coherent_utterance_phrase_quantile_with_presence_hurdle_v1\","
        "\"outputs\":[\"durations\",\"pause_presence_logits\",\"duration_quantiles\"],"
        "\"quantiles\":{\"output\":\"duration_quantiles\","
        "\"levels\":[0.1,0.5,0.9],\"domain\":\"positive_duration_frames\","
        "\"ordering\":\"monotonic_p10_p50_p90\"},"
        "\"modes\":[\"sampled\",\"p50\"],\"default_mode\":\"sampled\","
        "\"seed\":{\"source\":\"request_seed\",\"missing_request_seed\":0,"
        "\"algorithm\":\"sha256_box_muller_v1\"},"
        "\"sampling_key\":\"utf8_byte_length_prefixed_language_voice_phones_v1\","
        "\"defaults\":{\"pause_strength\":1.0,\"speech_strength\":0.0,"
        "\"sample_presence\":true,\"max_abs_z\":2.0},"
        "\"phrase_boundary\":\""
    ) + phrase_boundary +
        "\",\"pause_owners\":{\"punctuation\":"
        "\"following_or_remapped_silence_only\",\"word_boundary\":"
        "\"explicit_candidate_mask_only\"}}},\"assets\":{}}";
}

}  // namespace

int main() {
    const std::vector<std::string> phones = {
        "AA", "<pause_comma>", "<sil>", "ɜ"
    };
    const std::string key = scyllasband_detail::duration_hierarchy_sampling_key(
        "en_gb", "Tuesday", phones
    );
    const std::string expected_key =
        "5:en_gb|7:Tuesday|1:4|2:AA|13:<pause_comma>|5:<sil>|2:ɜ";
    if (key != expected_key ||
        scyllasband_detail::sha256_hex(key) !=
            "d272697a6ebdc48b84f52d3d562caf1527d23e19ac461a5c62e3a6b7472a4c8d") {
        std::cerr << "Portable hierarchy key or SHA mismatch" << std::endl;
        return 1;
    }
    const std::string scope = "duration-hierarchy:" +
        scyllasband_detail::sha256_hex(key);
    const double value = scyllasband_detail::duration_hierarchy_hash_normal(
        2027, scope + ":utterance", 2.0
    );
    if (std::fabs(value - (-0.5283969377841734)) > 1.0e-12) {
        std::cerr << "Portable hierarchy normal mismatch" << std::endl;
        return 1;
    }
    if (scyllasband_detail::duration_hierarchy_round_nonnegative(2.5) != 2 ||
        scyllasband_detail::duration_hierarchy_round_nonnegative(3.5) != 4) {
        std::cerr << "Portable hierarchy rounding is not ties-to-even" << std::endl;
        return 1;
    }
    struct SampleVector {
        uint64_t seed;
        int phrase;
        double lower;
        double median;
        double upper;
        double presence_logit;
        double duration_scale;
        bool present;
        int64_t frames;
    };
    const std::vector<SampleVector> vectors = {
        {0, 0, 1.0, 3.0, 7.0, 1.25, 1.0, true, 4},
        {2027, 0, 1.0, 3.0, 7.0, 1.25, 1.0, false, 0},
        {2028, 1, 0.0, 2.0, 9.0, -1.5, 1.25, true, 11},
        {2029, 2, 2.5, 2.5, 2.5, 20.0, 1.0, true, 2},
    };
    const std::string key_sha256 = scyllasband_detail::sha256_hex(key);
    for (const auto& expected : vectors) {
        const auto sample = scyllasband_detail::duration_hierarchy_sample_pause(
            expected.seed,
            key_sha256,
            expected.phrase,
            expected.lower,
            expected.median,
            expected.upper,
            expected.presence_logit,
            expected.duration_scale,
            true,
            1.0,
            2.0
        );
        if (sample.present != expected.present || sample.frames != expected.frames) {
            std::cerr << "Portable hierarchy seeded sample mismatch for seed "
                      << expected.seed << std::endl;
            return 1;
        }
    }

    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path root = std::filesystem::temp_directory_path() /
        ("scyllasband-duration-hierarchy-" + std::to_string(nonce));
    std::filesystem::create_directories(root);
    for (const char* name : {"g2p.onnx", "duration.onnx", "vector.onnx", "vocoder.onnx"}) {
        write_file(root / name, "fixture");
    }
    try {
        write_file(
            root / "manifest.json",
            manifest("punctuation_owned_silence_after_boundary_v1")
        );
        const auto bundle = scyllasband_detail::load_scyllasband_bundle_info(
            root.string(), SCYLLASBAND_BACKEND_ONNX
        );
        if (!bundle.duration_hierarchy_sampling_enabled ||
            !bundle.duration_hierarchy_default_sampled ||
            !bundle.duration_hierarchy_sample_presence) {
            throw std::runtime_error("Valid duration hierarchy control was not loaded");
        }
        write_file(root / "manifest.json", manifest("wrong"));
        try {
            (void)scyllasband_detail::load_scyllasband_bundle_info(
                root.string(), SCYLLASBAND_BACKEND_ONNX
            );
            throw std::runtime_error("Invalid phrase boundary was accepted");
        } catch (const std::runtime_error& exc) {
            if (std::string(exc.what()) == "Invalid phrase boundary was accepted") throw;
        }
    } catch (const std::exception& exc) {
        std::filesystem::remove_all(root);
        std::cerr << exc.what() << std::endl;
        return 1;
    }
    std::filesystem::remove_all(root);
    return 0;
}
