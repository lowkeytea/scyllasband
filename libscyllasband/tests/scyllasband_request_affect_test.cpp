#include "scyllasband_request.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int fail(const std::string& message) {
    std::cerr << message << "\n";
    return 1;
}

bool matches(const std::vector<float>& actual, const std::vector<float>& expected) {
    if (actual.size() != expected.size()) {
        return false;
    }
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (std::fabs(actual[index] - expected[index]) > 1e-6f) {
            return false;
        }
    }
    return true;
}

scyllasband_detail::ScyllasBandBundleInfo test_bundle() {
    scyllasband_detail::ScyllasBandBundleInfo bundle;
    bundle.affect_enabled = true;
    bundle.affect_axes = {"calm", "joy", "anger", "sadness", "whisper"};
    bundle.affect_axis_order_version = "3";
    bundle.affect_default_preset = "neutral";
    bundle.affect_presets["neutral"] = {0.5f, 0.25f, 0.0f, 0.0f, 0.0f};
    bundle.affect_partial_defaults["calm"] = 0.5f;
    bundle.affect_axis_minimums["calm"] = 0.25f;
    bundle.affect_axis_maximums["calm"] = 0.75f;
    bundle.voice_to_id["scylla"] = 0;
    bundle.language_to_id["en_us"] = 0;
    bundle.emotion_to_id["neutral"] = 0;
    bundle.default_language = "en_us";
    bundle.voice_default_language["scylla"] = "en_us";
    bundle.voice_languages["scylla"] = {"en_us"};
    return bundle;
}

ScyllasBandSynthesisRequest request_with_affect(const char* affect) {
    ScyllasBandSynthesisRequest request{};
    request.voice_id = "scylla";
    request.language = "en_us";
    request.affect = affect;
    return request;
}

}  // namespace

int main() {
    const auto bundle = test_bundle();

    const auto omitted = scyllasband_detail::resolve_scyllasband_request_context(
        bundle,
        request_with_affect(nullptr)
    );
    if (!matches(omitted.affect_values, {0.5f, 0.25f, 0.0f, 0.0f, 0.0f})) {
        return fail("omitted affect did not resolve to the neutral preset");
    }

    const auto partial = scyllasband_detail::resolve_scyllasband_request_context(
        bundle,
        request_with_affect("anger=0.25")
    );
    if (!matches(partial.affect_values, {0.5f, 0.0f, 0.25f, 0.0f, 0.0f})) {
        return fail("partial affect did not overlay the calm default");
    }

    try {
        (void)scyllasband_detail::resolve_scyllasband_request_context(
            bundle,
            request_with_affect("calm=0")
        );
        return fail("explicit calm=0 was accepted");
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()).find("calm") == std::string::npos) {
            return fail("calm bound failure did not identify the affected axis");
        }
    }
    return 0;
}
