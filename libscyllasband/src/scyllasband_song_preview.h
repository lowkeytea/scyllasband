// A score as plain tones, for hearing a melody instantly before it is sung: one soft tone per note on the song's own
// time base (silence on rests). Pass the score as it will be sung (after transpose and the octave register) so the
// preview's pitch is the voice's.
#pragma once

#include "scyllasband_score.h"

#include <vector>

namespace scyllasband {

// Mono samples in [-1, 1] at `sample_rate`, ceil(score.seconds() * sample_rate) long: per note a sine with a little second
// harmonic, a short attack and release so notes do not click, at `amplitude`.
std::vector<float> preview_tones(const Score& score, int sample_rate, float amplitude = 0.25f);

}  // namespace scyllasband
