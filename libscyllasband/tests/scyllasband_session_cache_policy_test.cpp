#include "scyllasband_session_cache_policy.h"

#include <iostream>
#include <string>
#include <vector>

namespace {

int fail(const char* message) {
    std::cerr << message << "\n";
    return 1;
}

bool equals(
    const std::vector<std::string>& actual,
    const std::vector<std::string>& expected
) {
    return actual == expected;
}

}  // namespace

int main() {
    scyllasband_detail::ScyllasBandSessionCachePolicy policy;

    if (!policy.touch("256").empty() || !policy.touch("384").empty() ||
        !equals(policy.lru_keys(), {"256", "384"})) {
        return fail("unbounded policy must retain all touched buckets");
    }

    if (!equals(policy.set_capacity(1), {"256"}) ||
        !equals(policy.lru_keys(), {"384"})) {
        return fail("lowering capacity must evict the least-recent bucket");
    }

    if (!policy.touch("384").empty() ||
        !equals(policy.lru_keys(), {"384"})) {
        return fail("touching the active bucket must not evict it");
    }

    if (!equals(policy.touch("512"), {"384"}) ||
        !equals(policy.lru_keys(), {"512"})) {
        return fail("capacity-one policy must replace the previous bucket");
    }

    if (!policy.set_capacity(0).empty() || !policy.touch("640").empty() ||
        !equals(policy.lru_keys(), {"512", "640"})) {
        return fail("capacity zero must restore unbounded retention");
    }

    return 0;
}
