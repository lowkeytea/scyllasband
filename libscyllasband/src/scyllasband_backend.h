#pragma once

#include "scyllasband.h"

#include <memory>
#include <string>
#include <vector>

namespace scyllasband_detail {

struct ScyllasBandDurationEstimate {
    int predicted_latent_frames = 0;
    int fixed_latent_frames = 0;
    std::string metadata_json;
};

std::string duration_hierarchy_sampling_key(
    const std::string& language,
    const std::string& voice,
    const std::vector<std::string>& phones
);
double duration_hierarchy_hash_normal(
    uint64_t seed,
    const std::string& scope,
    double max_abs_z
);
int64_t duration_hierarchy_round_nonnegative(double value);

std::string g2p_lowercase_text(const std::string& value);

int64_t duration_hierarchy_apply_nonempty_phone_floor(
    int64_t frame_count,
    bool hierarchy_sampled,
    bool pause_presence_eligible,
    bool phone_empty
);

struct ScyllasBandDurationHierarchyPauseSample {
    int64_t frames = 0;
    bool present = false;
    double unit_probability = 0.5;
    double duration_quantile = 0.5;
};

ScyllasBandDurationHierarchyPauseSample duration_hierarchy_sample_pause(
    uint64_t seed,
    const std::string& sampling_key_sha256,
    int phrase_index,
    double lower,
    double median,
    double upper,
    double presence_logit,
    double duration_scale,
    bool sample_presence,
    double pause_strength,
    double max_abs_z
);

class ScyllasBandBackendEngine {
public:
    virtual ~ScyllasBandBackendEngine() = default;
    virtual ScyllasBandStatus synthesize(
        const ScyllasBandSynthesisRequest& request,
        ScyllasBandSynthesisResult* out_result
    ) = 0;
    virtual ScyllasBandStatus estimate_latent_frames(
        const ScyllasBandSynthesisRequest& request,
        ScyllasBandDurationEstimate* out_estimate
    ) = 0;
    virtual void set_target_bucket_cache_capacity(int capacity) {
        (void)capacity;
    }
    virtual const char* name() const = 0;
};

std::unique_ptr<ScyllasBandBackendEngine> create_backend_engine(
    ScyllasBandBackend backend,
    std::string bundle_dir,
    bool validate_bundle,
    ScyllasBandLiteRtAccelerator litert_accelerator,
    int litert_max_threads
);

}  // namespace scyllasband_detail
