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

// g2p_phrases.normalize_g2p_phrase_punctuation
Text normalize_phrase_punctuation(const Text& text);

// g2p_phrases.punctuation_run_phone_token; empty when the run maps to no token.
std::string punctuation_run_phone_token(const Text& run);

// g2p.punctuated_segments: phrases with the punctuation tokens that follow them.
std::vector<std::pair<Text, std::vector<std::string>>> punctuated_segments(const Text& text, std::size_t max_chars);

// planner.split_sentences / split_for_retry / paragraph split
std::vector<Text> split_sentences(const Text& text);
std::vector<Text> split_for_retry(const Text& text);
std::vector<Text> split_paragraphs(const Text& text);

}  // namespace scyllasband
