// Text frontend shared by the planner and the G2P: spoken-form normalization, punctuation folding,
// punctuated phrase segmentation and sentence splitting, all mirroring the Python runtime.
#pragma once

#include "scyllasband_unicode.h"

#include <string>
#include <utility>
#include <vector>

namespace scyllasband {

using unicode::Text;

// Version and SHA-256 of the spoken-text normalizer contract this runtime implements; a bundle that
// declares a different contract is rejected.
extern const char* const kSpokenTextNormalizerVersion;
extern const char* const kSpokenTextNormalizerSha256;

// text_normalizer.normalize_spoken_text
std::string normalize_spoken_text(const std::string& text, const std::string& language);

// text_normalizer.nonverbal_spellings (v5 bundles): typed variants of a non-verbal word written the way a v5 model's training
// rows are (the language's nonverbal_spellings rules; English only): "Uhuh" "Uh-uh", "Hehheh" "Heh heh", "A-hem" "Ahem",
// "Pst" "Psst". Run on the typed text, before normalize_spoken_text. A capital first letter is kept, and a variant in
// capitals stays in capitals.
std::string nonverbal_spellings(const std::string& text, const std::string& language);

// The phones a v5 model was trained on for the English non-verbal sounds (text_data/en.json "nonverbal_phones"), in the
// table's order.
struct NonverbalPhoneTable {
    std::vector<std::pair<std::string, std::string>> words;    // word -> symbols separated by spaces, " | " between its parts
    std::vector<std::pair<std::string, std::string>> phrases;  // the words that spell a sound -> the word whose parts they take
    std::vector<std::string> locales;                          // the G2P languages the table covers
};
const NonverbalPhoneTable& nonverbal_phone_table();

// g2p_phrases.normalize_g2p_phrase_punctuation
Text normalize_phrase_punctuation(const Text& text);

// g2p_phrases.punctuation_run_phone_token; empty when the run maps to no token.
std::string punctuation_run_phone_token(const Text& run);

// g2p.punctuated_segments: phrases with the punctuation tokens that follow them.
std::vector<std::pair<Text, std::vector<std::string>>> punctuated_segments(const Text& text, std::size_t max_chars);

// The marker the normalizers write for a dash typed with a letter directly on each side ("a\u2014dream").
constexpr char32_t kTightDash = 0x2E3A;

// punctuated_segments plus whether the segment's punctuation holds such a dash (g2p._marked_segments).
struct MarkedSegment {
    Text text;
    std::vector<std::string> tokens;
    bool tight_dash = false;
};
std::vector<MarkedSegment> marked_segments(const Text& text, std::size_t max_chars);

// planner.split_sentences / split_for_retry / paragraph split
std::vector<Text> split_sentences(const Text& text);
std::vector<Text> split_for_retry(const Text& text, double min_share = 0.0);
std::vector<Text> split_paragraphs(const Text& text);

}  // namespace scyllasband
