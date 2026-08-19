#include "scyllasband_boundary_policy.h"

#include <iostream>
#include <string>

namespace {

int fail(const std::string& message) {
    std::cerr << message << "\n";
    return 1;
}

}  // namespace

int main() {
    using scyllasband_detail::host_pause_after_ms;
    using scyllasband_detail::is_terminal_pause_target;
    using scyllasband_detail::is_non_acoustic_modifier_phone;
    using scyllasband_detail::is_zero_duration_punctuation_phone;
    using scyllasband_detail::pause_ms_to_latent_frames;
    using scyllasband_detail::punctuation_duration_floor_frames;
    using scyllasband_detail::trailing_boundary_pause_phone;

    if (!trailing_boundary_pause_phone("chunk_continue", true, false).empty()) {
        return fail("chunk_continue inserted a linguistic phone");
    }
    if (trailing_boundary_pause_phone("clause_continue", true, false) != "<pause_comma>") {
        return fail("clause_continue lost its comma fallback");
    }
    if (!trailing_boundary_pause_phone("clause_continue", true, true).empty()) {
        return fail("clause_continue duplicated an existing boundary phone");
    }

    const int64_t clause_floor = pause_ms_to_latent_frames(160.0f, 24000, 512);
    if (clause_floor != 8) {
        return fail("160 ms clause floor did not resolve to eight latent frames");
    }
    for (const std::string phone : {
             "<pause_comma>",
             "<pause_semicolon>",
             "<pause_colon>",
             "<pause_dash>",
         }) {
        if (punctuation_duration_floor_frames(phone, 0, clause_floor) != clause_floor) {
            return fail("natural clause punctuation lost its duration floor: " + phone);
        }
    }
    if (punctuation_duration_floor_frames("<ctx_chunk_continue>", 0, clause_floor) != 0) {
        return fail("artificial chunk context received a clause duration floor");
    }
    if (punctuation_duration_floor_frames("<ellipsis>", 0, 0, 12) != 12) {
        return fail("calibrated ellipsis floor was not applied");
    }
    if (punctuation_duration_floor_frames("<end_stmt>", 15, 0, 12) != 15) {
        return fail("explicit sentence override did not exceed calibrated floor");
    }
    if (punctuation_duration_floor_frames("<pause_colon>", 0, 8, 12) != 12) {
        return fail("calibrated colon floor was not retained over broad clause floor");
    }
    if (!is_zero_duration_punctuation_phone("<end_question>") ||
        !is_zero_duration_punctuation_phone("<pause_comma>") ||
        is_zero_duration_punctuation_phone("<ctx_sentence_end>") ||
        is_zero_duration_punctuation_phone("<sil>")) {
        return fail("explicit-silence punctuation classification is inconsistent");
    }
    if (!is_non_acoustic_modifier_phone(u8"ˈ") ||
        !is_non_acoustic_modifier_phone(u8"ː") ||
        !is_non_acoustic_modifier_phone(u8"̃") ||
        is_non_acoustic_modifier_phone("t") ||
        is_non_acoustic_modifier_phone("<sil>")) {
        return fail("non-acoustic modifier classification is inconsistent");
    }
    if (!is_terminal_pause_target(2, 3)) {
        return fail("final explicit silence was not recognized as terminal");
    }
    if (!is_terminal_pause_target(1, 2)) {
        return fail("merged terminal punctuation was not recognized as terminal");
    }
    if (is_terminal_pause_target(2, 4)) {
        return fail("internal punctuation silence was incorrectly recognized as terminal");
    }

    if (host_pause_after_ms("chunk_continue", false, 320, 160) != 0) {
        return fail("artificial chunk context received host assembly silence");
    }
    if (host_pause_after_ms("clause_continue", false, 320, 160) != 160) {
        return fail("linguistic continuation lost host assembly silence");
    }
    if (host_pause_after_ms("sentence_end", false, 320, 160) != 320) {
        return fail("sentence boundary lost host assembly silence");
    }
    if (host_pause_after_ms("sentence_end", true, 320, 160) != 0) {
        return fail("last chunk retained host assembly silence");
    }
    return 0;
}
