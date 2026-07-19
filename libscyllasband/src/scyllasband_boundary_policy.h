#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

namespace scyllasband_detail {

inline bool is_sentence_punctuation_phone(const std::string& phone) {
    return phone == "<end_stmt>" ||
           phone == "<end_question>" ||
           phone == "<end_exclaim>" ||
           phone == "<ellipsis>" ||
           phone == "<ctx_sentence_end>";
}

inline bool is_clause_punctuation_phone(const std::string& phone) {
    return phone == "<pause_comma>" ||
           phone == "<pause_semicolon>" ||
           phone == "<pause_colon>" ||
           phone == "<pause_dash>" ||
           phone == "<ctx_continuation>";
}

inline std::string trailing_boundary_pause_phone(
    const std::string& boundary_after,
    bool has_pause_comma,
    bool already_ends_in_boundary_phone
) {
    if (boundary_after == "clause_continue" &&
        has_pause_comma &&
        !already_ends_in_boundary_phone) {
        return "<pause_comma>";
    }
    return {};
}

inline int64_t pause_ms_to_latent_frames(
    float pause_ms,
    int sample_rate,
    int latent_hop_length
) {
    if (!std::isfinite(pause_ms) || pause_ms <= 0.0f ||
        sample_rate <= 0 || latent_hop_length <= 0) {
        return 0;
    }
    const double frames = static_cast<double>(pause_ms) *
                          static_cast<double>(sample_rate) /
                          (1000.0 * static_cast<double>(latent_hop_length));
    return std::max<int64_t>(1, static_cast<int64_t>(std::llround(frames)));
}

inline int64_t punctuation_duration_floor_frames(
    const std::string& phone,
    int64_t sentence_pause_floor,
    int64_t clause_pause_floor
) {
    if (sentence_pause_floor > 0 && is_sentence_punctuation_phone(phone)) {
        return sentence_pause_floor;
    }
    if (clause_pause_floor > 0 && is_clause_punctuation_phone(phone)) {
        return clause_pause_floor;
    }
    return 0;
}

inline bool is_terminal_pause_target(int phone_index, int phone_count) {
    // Host assembly owns silence after the request. Keep the model-predicted
    // duration for the final phone instead of asking the flow model to extend it.
    return phone_index < 0 || phone_index + 1 >= phone_count;
}

inline int host_pause_after_ms(
    const std::string& boundary_after,
    bool is_last,
    int sentence_pause_ms,
    int continuation_pause_ms
) {
    if (is_last || boundary_after == "chunk_continue") {
        return 0;
    }
    if (boundary_after == "sentence_end" || boundary_after == "paragraph_end") {
        return std::max(0, sentence_pause_ms);
    }
    return std::max(0, continuation_pause_ms);
}

}  // namespace scyllasband_detail
