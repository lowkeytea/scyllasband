// Integer durations, per-frame text events, span context and delivery requests, exactly as the acoustic
// model expects them (see events.py, engine.balanced_context and delivery.py).
#pragma once

#include "scyllasband_json.h"

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace scyllasband {

extern const char* const kSilencePhone;

bool is_non_acoustic_modifier(const std::string& phone);  // stress, length and combining marks own no frames
bool is_punctuation_pause(const std::string& phone);
bool is_punctuation_phone(const std::string& phone);

// Rounded frames (half to even): punctuation and modifiers 0, <sil> may be 0, every other phone at least 1.
std::vector<int> frames_to_durations(const std::vector<float>& values, const std::vector<std::string>& phones, double scale);

// Modifier tokens in id order, each with its own bit.
std::map<std::string, int64_t> modifier_bits(const std::map<std::string, int>& token_to_id);

// Per phone: the type of the next sentence end at or after it, else that of the following context, else 0.
std::vector<int64_t> sentence_types(const std::vector<std::string>& phones, const std::vector<std::string>& following);

struct FrameEvents {
    std::vector<int64_t> boundary, modifier, sentence;
    std::vector<float> phase, log_duration;
};

FrameEvents frame_events(const std::vector<std::string>& phones, const std::vector<int>& durations, const std::vector<int>& word_starts,
                         const std::vector<std::string>& following, const std::map<std::string, int>& token_to_id,
                         const std::map<std::string, int64_t>& bits);

// Span ids and segments (0 before, 1 target, 2 after), context split evenly with spare room given to either side.
std::pair<std::vector<int64_t>, std::vector<int64_t>> balanced_context(const std::vector<int64_t>& before, const std::vector<int64_t>& target,
                                                                       const std::vector<int64_t>& after, std::size_t max_phones);

// Delivery: energy, tension, valence and assertiveness on 0-4 (2 = neutral, nullopt = auto). The graphs' fifth delivery
// input (whisper, from training) is always sent as off.
struct Delivery {
    std::array<std::optional<double>, 4> axes{2.0, 2.0, 2.0, 2.0};

    // NULL, "" or "neutral": neutral; "auto": every channel unconditioned; else "axis=value,..." pairs.
    static Delivery parse(const char* spec);
    void tensors(float values[5], uint8_t present[5]) const;
    Json to_json() const;
    std::string spec() const;  // canonical text, used to group sentences into chains
};

extern const char* const kDeliveryAxes[4];

}  // namespace scyllasband
