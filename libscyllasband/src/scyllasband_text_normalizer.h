#pragma once

#include <string>

namespace scyllasband_detail {

// Converts raw display text into the spoken-form text expected by ScyllasBand's
// G2P model. Model language identifiers such as en_us and en_gb are accepted.
std::string normalize_spoken_text(
    const std::string& text,
    const std::string& language
);

}  // namespace scyllasband_detail
