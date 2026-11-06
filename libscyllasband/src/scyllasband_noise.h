// Seeded Gaussian noise: the native runtime's own generator (xoshiro256** seeded by splitmix64, Box-Muller pairs).
// The draw order is part of the runtime's reproducibility: a target of speech draws (2 + latent_dim) x frames values (the
// pitch predictor's two rows, then the flow's, scaled by the temperature), a sung phrase latent_dim x frames. The Python
// runtime draws from numpy instead, so a seed gives different (equally valid) takes there.
#pragma once

#include <cstdint>

namespace scyllasband {

// Normal noise from a seeded xoshiro256** generator (Box-Muller), scaled by `temperature`.
void gaussian_noise(uint64_t seed, int64_t count, float temperature, float* out);

}  // namespace scyllasband
