// Tests of the frontend library alone (no graph runtime): run as `scyllasband_frontend_test <suite> <fixtures dir>`.
// Configure with -DSCYLLASBAND_SANITIZE=ON to run them under AddressSanitizer and UndefinedBehaviorSanitizer.
#include "scyllasband_noise.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL %s\n", what.c_str());
    }
}

// The generator's draw order is part of reproducibility (a seed gives the same take on every platform and release), so
// these values are frozen. Platform math libraries may differ in the last bit of log/cos/sin, hence the tolerance.
void noise_suite(const std::string&) {
    struct Case {
        uint64_t seed;
        int64_t count;
        float temperature;
        std::vector<float> expected;
    };
    const Case cases[] = {
        {7ULL, 6, 1.0f, {-0.151572749f, 0.829897106f, 0.587099552f, -0.0700574666f, 0.0944718644f, -0.0971522331f}},
        {0ULL, 5, 1.0f, {-0.0141067971f, -1.00858653f, -1.84589505f, 1.06692827f, 0.788179159f}},
        {31415ULL, 4, 0.5f, {0.883161068f, 0.470940083f, -0.904917359f, 0.130011484f}},
        {18446744073709551615ULL, 3, 1.0f, {0.117751814f, -1.0705862f, -0.0172506422f}},
    };
    for (const Case& c : cases) {
        std::vector<float> got(static_cast<std::size_t>(c.count), 99.0f);
        scyllasband::gaussian_noise(c.seed, c.count, c.temperature, got.data());
        for (std::size_t i = 0; i < got.size(); ++i) {
            expect(std::fabs(got[i] - c.expected[i]) <= 1e-6f * std::max(1.0f, std::fabs(c.expected[i])),
                   "noise seed " + std::to_string(c.seed) + " [" + std::to_string(i) + "] = " + std::to_string(got[i]) + ", expected " +
                       std::to_string(c.expected[i]));
        }
    }
    // A longer draw starts with the shorter one: callers that split one draw into rows rely on it.
    std::vector<float> few(4), many(10);
    scyllasband::gaussian_noise(7, 4, 1.0f, few.data());
    scyllasband::gaussian_noise(7, 10, 1.0f, many.data());
    expect(std::memcmp(few.data(), many.data(), sizeof(float) * 4) == 0, "noise prefix stability");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: scyllasband_frontend_test <suite> [fixtures dir]\n");
        return 2;
    }
    const std::map<std::string, std::function<void(const std::string&)>> suites = {
        {"noise", noise_suite},
    };
    const auto suite = suites.find(argv[1]);
    if (suite == suites.end()) {
        std::fprintf(stderr, "unknown suite %s\n", argv[1]);
        return 2;
    }
    suite->second(argc > 2 ? argv[2] : "");
    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("%s: ok\n", argv[1]);
    return 0;
}
