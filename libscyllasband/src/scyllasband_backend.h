#pragma once

#include "scyllasband.h"

#include <memory>
#include <string>

namespace scyllasband_detail {

struct ScyllasBandDurationEstimate {
    int predicted_latent_frames = 0;
    int fixed_latent_frames = 0;
    std::string metadata_json;
};

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
