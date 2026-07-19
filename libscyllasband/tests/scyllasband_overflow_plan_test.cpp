#include "scyllasband.h"
#include "scyllasband_backend.h"
#include "scyllasband_runtime_internal.h"

#include <iostream>
#include <memory>
#include <string>
#include <utility>

namespace {

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

int fail(const std::string& message) {
    std::cerr << message << "\n";
    return 1;
}

class OverflowEstimateBackend final : public scyllasband_detail::ScyllasBandBackendEngine {
public:
    explicit OverflowEstimateBackend(std::string overlong_text)
        : overlong_text_(std::move(overlong_text)) {}

    ScyllasBandStatus synthesize(
        const ScyllasBandSynthesisRequest&,
        ScyllasBandSynthesisResult*
    ) override {
        return SCYLLASBAND_STATUS_NOT_IMPLEMENTED;
    }

    ScyllasBandStatus estimate_latent_frames(
        const ScyllasBandSynthesisRequest& request,
        scyllasband_detail::ScyllasBandDurationEstimate* out_estimate
    ) override {
        if (out_estimate == nullptr || request.text == nullptr) {
            return SCYLLASBAND_STATUS_INVALID_ARGUMENT;
        }
        ++estimate_calls;
        const std::string text(request.text);
        out_estimate->predicted_latent_frames = text == overlong_text_
            ? 641
            : static_cast<int>(text.size());
        out_estimate->fixed_latent_frames = 640;
        out_estimate->metadata_json.clear();
        return SCYLLASBAND_STATUS_OK;
    }

    const char* name() const override {
        return "fake-overflow";
    }

    int estimate_calls = 0;

private:
    std::string overlong_text_;
};

}  // namespace

int main() {
    const std::string text =
        "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu; "
        "Nu xi omicron pi rho sigma tau upsilon phi chi psi omega.";
    auto backend = std::make_unique<OverflowEstimateBackend>(text);
    OverflowEstimateBackend* backend_ptr = backend.get();
    ScyllasBandRuntime runtime;
    runtime.backend = SCYLLASBAND_BACKEND_LITERT;
    runtime.engine = std::move(backend);

    ScyllasBandLongFormSynthesisRequest request{};
    request.request.text = text.c_str();
    request.request.voice_id = "test-voice";
    request.request.language = "en_us";
    request.request.steps = 1;
    request.request.sampler = SCYLLASBAND_SAMPLER_EULER;
    request.request.speed = 1.0f;
    request.max_chunk_chars = 1000;
    request.min_chunk_chars = 0;
    request.preflight_chunks = 1;

    ScyllasBandChunkPlanResult result{};
    const ScyllasBandStatus status = scyllasband_runtime_plan_long_form(&runtime, &request, &result);
    if (status != SCYLLASBAND_STATUS_OK || result.metadata_json == nullptr) {
        return fail(std::string("native overflow plan failed: ") +
            (scyllasband_last_error() == nullptr ? "" : scyllasband_last_error()));
    }
    const std::string payload(result.metadata_json);
    scyllasband_chunk_plan_result_free(&result);

    if (backend_ptr->estimate_calls < 3) {
        return fail("native planner did not re-estimate both overflow replacements");
    }
    for (const std::string& needle : {
             "\"chunk_count\":2",
             "\"split_reason\":\"latent_budget\"",
             "\"split_reason\":\"latent_budget_tail\"",
             "\"boundary_after\":\"clause_continue\"",
             "\"latent_retry_predicted_frames\":641",
             "\"latent_retry_fixed_frames\":640",
         }) {
        if (!contains(payload, needle)) {
            return fail("native overflow plan is missing expected linguistic replan metadata: " + needle + "\n" + payload);
        }
    }
    if (contains(payload, "\"predicted_latent_frames\":641")) {
        return fail("native saved plan retained an unsplit over-640 chunk\n" + payload);
    }
    return 0;
}
