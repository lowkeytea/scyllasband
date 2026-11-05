#include "scyllasband_noise.h"

#include <cmath>

namespace scyllasband {

namespace {

uint64_t splitmix64(uint64_t& state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

}  // namespace

void gaussian_noise(uint64_t seed, int64_t count, float temperature, float* out) {
    uint64_t s[4];
    uint64_t state = seed;
    for (uint64_t& word : s) word = splitmix64(state);
    auto next = [&s]() {
        auto rotl = [](uint64_t x, int k) { return (x << k) | (x >> (64 - k)); };
        const uint64_t result = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return result;
    };
    auto uniform = [&]() { return (static_cast<double>(next() >> 11) + 1.0) * (1.0 / 9007199254740992.0); };  // (0, 1]
    const double two_pi = 6.283185307179586;
    for (int64_t i = 0; i < count; i += 2) {
        const double radius = std::sqrt(-2.0 * std::log(uniform())), angle = two_pi * uniform();
        out[i] = static_cast<float>(radius * std::cos(angle)) * temperature;
        if (i + 1 < count) out[i + 1] = static_cast<float>(radius * std::sin(angle)) * temperature;
    }
}

}  // namespace scyllasband
