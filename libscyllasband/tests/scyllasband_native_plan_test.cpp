#include "scyllasband.h"
#include "scyllasband_bundle.h"
#include "scyllasband_execution_plan.h"

#include <iostream>
#include <string>

namespace {

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

int fail(const std::string& message) {
    std::cerr << message << "\n";
    return 1;
}

std::string plan_for_text(const char* text, int max_chars, int min_chars) {
    ScyllasBandChunkPlanRequest request{};
    request.text = text;
    request.max_chunk_chars = max_chars;
    request.min_chunk_chars = min_chars;
    ScyllasBandChunkPlanResult result{};
    const ScyllasBandStatus status = scyllasband_plan_long_form_chunks(&request, &result);
    if (status != SCYLLASBAND_STATUS_OK) {
        std::cerr << "scyllasband_plan_long_form_chunks failed: "
                  << scyllasband_status_message(status) << ": "
                  << (scyllasband_last_error() == nullptr ? "" : scyllasband_last_error()) << "\n";
        return {};
    }
    std::string payload = result.metadata_json == nullptr ? std::string() : result.metadata_json;
    scyllasband_chunk_plan_result_free(&result);
    return payload;
}

int expect_contains(const std::string& payload, const std::string& needle) {
    if (!contains(payload, needle)) {
        return fail("missing expected JSON field or value: " + needle + "\n" + payload);
    }
    return 0;
}

int validate_text_only_g2p_execution_plan() {
    scyllasband_detail::ScyllasBandBundleInfo bundle;
    bundle.sample_rate = 24000;
    bundle.hop_length = 256;
    bundle.latent_dim = 128;
    bundle.phone_frames = 384;
    bundle.latent_frames = 384;
    for (const std::string& name : {"g2p", "duration_predictor", "vector_estimator", "vocoder"}) {
        bundle.component_artifacts[name] = name + ".onnx";
    }
    // ONNX manifests expose logical raw text. The native frontend adds the
    // language token while encoding the graph's fixed text-token tensor.
    bundle.component_inputs["g2p"] = {"text"};
    bundle.component_inputs["duration_predictor"] = {"phone_ids", "voice_id", "language_id", "boundary_before_id", "boundary_after_id", "phone_mask", "emotion_id"};
    bundle.component_inputs["vector_estimator"] = {"noise", "time", "expanded_phone_ids", "voice_id", "language_id", "boundary_before_id", "boundary_after_id", "latent_mask", "emotion_id"};
    bundle.component_inputs["vocoder"] = {"latents", "voice_id", "language_id", "emotion_id"};
    try {
        const auto plan = scyllasband_detail::build_scyllasband_duration_flow_plan(bundle, scyllasband_detail::ScyllasBandResolvedRequest{});
        if (plan.components.size() != 4) {
            return fail("text-only ONNX G2P plan did not retain all execution components");
        }
    } catch (const std::exception& exc) {
        return fail(std::string("text-only ONNX G2P plan was rejected: ") + exc.what());
    }
    return 0;
}

}  // namespace

int main() {
    if (scyllasband_detail::scyllasband_select_backend(SCYLLASBAND_BACKEND_AUTO) ==
        SCYLLASBAND_BACKEND_AUTO) {
        return fail("SCYLLASBAND_BACKEND_AUTO did not resolve to a compiled backend");
    }
    if (scyllasband_detail::scyllasband_select_backend(SCYLLASBAND_BACKEND_LITERT) != SCYLLASBAND_BACKEND_LITERT) {
        return fail("explicit LiteRT backend selection was not preserved");
    }
    if (scyllasband_detail::scyllasband_select_backend(SCYLLASBAND_BACKEND_COREAI) != SCYLLASBAND_BACKEND_COREAI) {
        return fail("explicit Core AI backend selection was not preserved");
    }
    if (std::string(scyllasband_detail::scyllasband_backend_name(SCYLLASBAND_BACKEND_COREAI)) != "coreai") {
        return fail("Core AI backend name does not match the bundle contract");
    }
    const int execution_plan_status = validate_text_only_g2p_execution_plan();
    if (execution_plan_status != 0) {
        return execution_plan_status;
    }

    const char* text =
        "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu. "
        "Nu xi omicron pi rho sigma tau upsilon phi chi psi omega.";
    const std::string first = plan_for_text(text, 48, 16);
    if (first.empty()) {
        return 1;
    }
    const std::string second = plan_for_text(text, 48, 16);
    if (first != second) {
        return fail("native chunk planner output is not deterministic");
    }

    for (const std::string& needle : {
             "\"version\":\"scyllasband.streaming.plan.v1\"",
             "\"backend\":\"onnx\"",
             "\"source_kind\":\"text\"",
             "\"fixed_shape_budget\"",
             "\"records\"",
             "\"scheduler_hints\"",
             "\"parallelizable_chains\":[\"chain-0000\"]",
             "\"chunk_id\":\"chunk-0000\"",
             "\"chain_id\":\"chain-0000\"",
             "\"record_id\":\"record-0000\"",
             "\"order_index\":0",
             "\"prefix_policy\":\"none\"",
             "\"prefix_policy\":\"previous_chunk\"",
             "\"context_after\"",
             "\"context_before\"",
             "\"reference_key\":null",
             "\"predicted_latent_frames\":null",
             "\"chunk_count\":",
         }) {
        const int status = expect_contains(first, needle);
        if (status != 0) {
            return status;
        }
    }
    if (!contains(first, "\"chunk_id\":\"chunk-0001\"")) {
        return fail("small budget did not produce multiple stable chunk ids\n" + first);
    }
    return 0;
}
