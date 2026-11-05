#include "scyllasband_song_preview.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace scyllasband {

std::vector<float> preview_tones(const Score& score, int sample_rate, float amplitude) {
    if (sample_rate <= 0) throw std::invalid_argument("sample_rate must be positive");
    const double rate = static_cast<double>(sample_rate);
    const std::size_t length = static_cast<std::size_t>(std::ceil(score.seconds() * rate));
    std::vector<float> out(length, 0.0f);
    const double two_pi = 6.283185307179586;
    for (const ScoreNote& note : score.notes) {
        if (!note.midi || note.end <= note.start) continue;
        const double frequency = 440.0 * std::pow(2.0, (*note.midi - 69) / 12.0);
        const std::size_t first = static_cast<std::size_t>(std::llround(note.start * rate));
        const std::size_t last = std::min(length, static_cast<std::size_t>(std::llround(note.end * rate)));
        if (last <= first) continue;
        const double duration = static_cast<double>(last - first) / rate;
        const double attack = std::min(0.01, duration / 4), release = std::min(0.05, duration / 3);
        for (std::size_t i = first; i < last; ++i) {
            const double t = static_cast<double>(i - first) / rate;
            double envelope = 1.0;
            if (t < attack) envelope = t / attack;
            if (duration - t < release) envelope = std::min(envelope, (duration - t) / release);
            const double phase = two_pi * frequency * t;
            const double tone = std::sin(phase) + 0.3 * std::sin(2.0 * phase);
            out[i] += static_cast<float>(amplitude * envelope * tone / 1.3);
        }
    }
    for (float& sample : out) sample = std::clamp(sample, -1.0f, 1.0f);
    return out;
}

}  // namespace scyllasband
