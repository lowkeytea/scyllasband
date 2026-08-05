#pragma once

#include "scyllasband_backend.h"

#include <memory>
#include <string>

struct ScyllasBandRuntime {
    std::string bundle_dir;
    ScyllasBandBackend backend = SCYLLASBAND_BACKEND_AUTO;
    bool validate_bundle = true;
    ScyllasBandLiteRtAccelerator litert_accelerator = SCYLLASBAND_LITERT_ACCELERATOR_CPU;
    int litert_max_threads = 0;
    int target_bucket_cache_capacity = 0;
    std::unique_ptr<scyllasband_detail::ScyllasBandBackendEngine> engine;
};
