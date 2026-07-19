#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace scyllasband_detail {

struct ScyllasBandTargetBucketDecision {
    int latent_frames = 0;
    int max_latent_frames = 0;
    bool fits = false;
};

inline bool target_bucket_requires_replan(
    int64_t predicted_latent_frames,
    int fixed_latent_frames
) {
    return fixed_latent_frames > 0 &&
        predicted_latent_frames > static_cast<int64_t>(fixed_latent_frames);
}

inline ScyllasBandTargetBucketDecision select_smallest_target_bucket(
    int64_t predicted_latent_frames,
    int bucket_margin_frames,
    int default_latent_frames,
    std::vector<int> target_bucket_frames
) {
    if (predicted_latent_frames <= 0) {
        throw std::invalid_argument("Predicted latent frame count must be positive");
    }
    target_bucket_frames.erase(
        std::remove_if(
            target_bucket_frames.begin(),
            target_bucket_frames.end(),
            [](int frames) { return frames <= 0; }
        ),
        target_bucket_frames.end()
    );
    if (default_latent_frames > 0) {
        target_bucket_frames.push_back(default_latent_frames);
    }
    std::sort(target_bucket_frames.begin(), target_bucket_frames.end());
    target_bucket_frames.erase(
        std::unique(target_bucket_frames.begin(), target_bucket_frames.end()),
        target_bucket_frames.end()
    );
    if (target_bucket_frames.empty()) {
        throw std::invalid_argument("At least one positive target bucket is required");
    }

    const int64_t required = predicted_latent_frames + std::max(0, bucket_margin_frames);
    int selected = target_bucket_frames.back();
    for (int candidate : target_bucket_frames) {
        if (static_cast<int64_t>(candidate) >= required) {
            selected = candidate;
            break;
        }
    }
    return ScyllasBandTargetBucketDecision{
        selected,
        target_bucket_frames.back(),
        predicted_latent_frames <= static_cast<int64_t>(selected),
    };
}

}  // namespace scyllasband_detail
