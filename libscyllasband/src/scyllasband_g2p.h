// Text -> Scylla's Band phones in the layout the acoustic model expects (see g2p.py):
// a leading <sil>; a word-boundary <sil> between the words of a phrase; a <sil> after every punctuation
// token; a trailing <sil>. `word_starts` lists the first phone of every word after the first.
#pragma once

#include "scyllasband_json.h"
#include "scyllasband_text.h"

#include <cstdint>
#include <functional>
#include <list>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace scyllasband {

struct Phonemized {
    std::vector<std::string> phones;
    std::vector<int> word_starts;
    std::vector<int> word_boundary_candidates;
    Json punctuation{Json::Array{}};
    std::vector<std::string> segments;
    double confidence = 0.0;
    Json pronunciation_overrides{Json::Array{}};
    Json terminal_tail_repairs{Json::Array{}};
};

class G2P {
public:
    int64_t text_pad() const { return text_pad_; }
    // Runs the G2P graph on padded text ids [1, tokens]; returns logits [1, frames, symbols] and sets `symbols`.
    using Infer = std::function<std::vector<float>(const std::vector<int64_t>& ids, int64_t& symbols)>;

    G2P(const std::string& bundle_dir, const Json& manifest, const std::map<std::string, int>& phone_to_id, Infer infer);

    std::string model_language(const std::string& language) const;
    Phonemized phonemize(const std::string& text, const std::string& language);

private:
    struct Prediction {
        std::vector<std::string> phones;
        double confidence = 0.0;
        Json overrides{Json::Array{}};
        Json repairs{Json::Array{}};
    };

    Phonemized run(const Text& text, const std::string& language);
    Prediction predict_segment(const Text& text, const std::string& language);
    Prediction predict_raw(const Text& text, const std::string& language);
    void apply_overrides(Prediction& prediction, const Text& text, const std::string& language) const;
    std::vector<int64_t> encode_text(const Text& text, const std::string& language) const;

    std::map<std::string, int> phone_to_id_;
    Infer infer_;
    Json language_map_;
    Json overrides_;
    std::map<std::string, int64_t> text_symbols_;
    std::map<int64_t, std::string> phoneme_symbols_;
    std::string word_boundary_symbol_ = " ";
    int64_t text_tokens_ = 512;
    std::size_t chunk_max_chars_ = 140;
    int char_repeats_ = 1;
    bool lowercase_ = true;
    int64_t text_pad_ = 0, phoneme_pad_ = 0, phoneme_end_ = 0;

    std::mutex mutex_;
    std::list<std::pair<std::string, Phonemized>> cache_;  // most recent first
    std::list<std::pair<std::string, std::vector<std::string>>> word_cache_;
};

}  // namespace scyllasband
