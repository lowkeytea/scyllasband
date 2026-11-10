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

// A target's phones, its words, and the neighbouring text (as phone ids) its span context is drawn from.
struct Sentence {
    std::vector<std::string> phones;
    std::vector<int> word_starts;
    std::vector<int64_t> before_ids;  // preceding passage phones, silences excluded
    std::vector<int64_t> after_ids;   // following passage phones, silences excluded
    std::vector<int> min_pauses;      // silences held at least the tight-dash pause (dashes typed with no spaces)
};

bool is_non_acoustic_modifier(const std::string& phone);  // stress, length and combining marks own no frames
bool is_punctuation_pause(const std::string& phone);
bool is_punctuation_phone(const std::string& phone);

// Rounded frames (half to even): punctuation and modifiers 0, <sil> may be 0, every other phone at least 1. `pause_scale`
// multiplies <sil> frames on top of `scale` (pace moves pauses with the words). The <sil> at each index in `min_pauses`
// lasts at least `min_pause_frames` x `scale` frames.
std::vector<int> frames_to_durations(const std::vector<float>& values, const std::vector<std::string>& phones, double scale,
                                     double pause_scale = 1.0, const std::vector<int>& min_pauses = {},
                                     double min_pause_frames = 0.0);

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

// Delivery (see delivery.py): intensity (calm to intense) and pace (slower to faster, words and pauses), both in the voice's
// own standard deviations, and pitch, a shift in semitones. nullopt takes the voice's default. Values are clamped into the
// default ranges here (kDeliveryRanges) and into the bundle's own ranges by the engine.
struct Delivery {
    std::array<std::optional<double>, 3> controls{};  // intensity, pace, pitch

    // NULL, "", "default" or "neutral": every control at the voice's default; else "control=value,..." pairs.
    static Delivery parse(const char* spec);
    // delivery.resolve_delivery: null, a control=value string (as parse), or a mapping {"intensity": 1, "pace": "-0.5"} whose
    // null, "", "default" and "auto" values take the voice's default.
    static Delivery from_json(const Json& value);
    Json to_json() const;
    std::string spec() const;  // canonical text, used to group sentences into chains
};

extern const char* const kDeliveryControls[3];
extern const double kDeliveryRanges[3][2];  // [low, high] per control, the same as the Python runtime's DEFAULT_RANGES

}  // namespace scyllasband
