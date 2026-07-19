#include "scyllasband_target_bucket_policy.h"

#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

int fail(const char* message) {
    std::cerr << message << "\n";
    return 1;
}

}  // namespace

int main() {
    const std::vector<int> buckets{256, 384, 512, 640};
    const std::vector<std::pair<int, int>> thresholds{
        {1, 256},
        {256, 256},
        {257, 384},
        {384, 384},
        {385, 512},
        {512, 512},
        {513, 640},
        {640, 640},
    };
    for (const auto& item : thresholds) {
        const auto decision = scyllasband_detail::select_smallest_target_bucket(
            item.first,
            0,
            640,
            buckets
        );
        if (decision.latent_frames != item.second ||
            decision.max_latent_frames != 640 ||
            !decision.fits ||
            scyllasband_detail::target_bucket_requires_replan(
                item.first,
                decision.latent_frames
            )) {
            return fail("exact target-bucket threshold mismatch");
        }
    }

    const auto overflow = scyllasband_detail::select_smallest_target_bucket(
        641,
        0,
        640,
        buckets
    );
    if (overflow.latent_frames != 640 ||
        overflow.max_latent_frames != 640 ||
        overflow.fits ||
        !scyllasband_detail::target_bucket_requires_replan(641, overflow.latent_frames)) {
        return fail("overflow must preserve the maximum bucket and require replanning");
    }

    try {
        (void)scyllasband_detail::select_smallest_target_bucket(0, 0, 640, buckets);
        return fail("nonpositive predictions must fail closed");
    } catch (const std::invalid_argument&) {
    }
    try {
        (void)scyllasband_detail::select_smallest_target_bucket(1, 0, 0, {});
        return fail("an empty bucket ladder must fail closed");
    } catch (const std::invalid_argument&) {
    }
    return 0;
}
