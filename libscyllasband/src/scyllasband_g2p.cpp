#include "scyllasband_g2p.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace scyllasband {
namespace {

using namespace unicode;
constexpr std::size_t kCacheSize = 256;
const std::string kWordBoundary = "<word_boundary>";
const std::string kSilence = "<sil>";

bool is_silence(const std::string& phone) { return phone == "<sil>" || phone == "sil" || phone == "sp" || phone == "<sp>"; }

bool skip_output_symbol(const std::string& symbol) {
    if (symbol.empty() || symbol[0] == '<' || symbol == "_") return true;
    Text text = decode(symbol);
    if (strip(text).empty()) return true;
    return std::all_of(text.begin(), text.end(), [](char32_t cp) { return is_punct_or_space(cp); });
}

bool is_internal_word_boundary(const std::vector<std::string>& phones, std::size_t index) {
    if (phones[index] != kWordBoundary || (index > 0 && phones[index - 1] == kWordBoundary)) return false;
    auto acoustic = [](const std::string& p) { return p != kWordBoundary && !is_silence(p); };
    return std::any_of(phones.begin(), phones.begin() + static_cast<std::ptrdiff_t>(index), acoustic) &&
           std::any_of(phones.begin() + static_cast<std::ptrdiff_t>(index) + 1, phones.end(), acoustic);
}

// [^\W\d_]: a word character that is neither a decimal digit nor the underscore.
bool word_letter(char32_t cp) { return is_alnum(cp) && !is_decimal(cp); }

// The last match of [^\W\d_]+(?:['’\-][^\W\d_]+)* in text.lower()
Text terminal_word(const Text& text) {
    Text lowered = lower(text), last;
    std::size_t pos = 0;
    while (pos < lowered.size()) {
        if (!word_letter(lowered[pos])) {
            ++pos;
            continue;
        }
        std::size_t end = pos;
        while (end < lowered.size() && word_letter(lowered[end])) ++end;
        while (end + 1 < lowered.size() && (lowered[end] == '\'' || lowered[end] == 0x2019 || lowered[end] == '-') && word_letter(lowered[end + 1])) {
            ++end;
            while (end < lowered.size() && word_letter(lowered[end])) ++end;
        }
        last = lowered.substr(pos, end - pos);
        pos = end;
    }
    return last;
}

std::vector<std::string> phone_list(const Json& spec, const std::string& key) {
    const Json& value = spec.is_object() ? spec.get(key) : (key == "phones" ? spec : Json());
    std::vector<std::string> out;
    if (value.is_string()) {
        for (const Text& item : split_whitespace(decode(value.str()))) out.push_back(encode(item));
    } else if (value.is_array()) {
        for (const Json& item : value.elements()) {
            std::string text = item.is_string() ? item.str() : item.dump();
            if (!text.empty()) out.push_back(text);
        }
    }
    return out;
}

int replace_subsequence(std::vector<std::string>& phones, const std::vector<std::string>& source, const std::vector<std::string>& target) {
    std::vector<std::string> out;
    int count = 0;
    std::size_t index = 0;
    while (index < phones.size()) {
        if (index + source.size() <= phones.size() && std::equal(source.begin(), source.end(), phones.begin() + static_cast<std::ptrdiff_t>(index))) {
            out.insert(out.end(), target.begin(), target.end());
            index += source.size();
            ++count;
        } else {
            out.push_back(phones[index++]);
        }
    }
    phones = std::move(out);
    return count;
}

double prob_product(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    double total = 0.0;
    for (double value : values) total += std::log(std::max(value, 1e-12));
    return std::exp(total);
}

}  // namespace

G2P::G2P(const std::string& bundle_dir, const Json& manifest, const std::map<std::string, int>& phone_to_id, Infer infer)
    : phone_to_id_(phone_to_id), infer_(std::move(infer)) {
    const Json& assets = manifest.get("assets");
    const Json& controls = manifest.get("controls");
    auto asset = [&](const std::string& key) { return bundle_dir + "/" + assets.get(key).str(); };
    Json config = Json::parse_file(asset("g2p_config"));
    Json tokenizer = Json::parse_file(asset("g2p_tokenizer"));
    language_map_ = Json::parse_file(asset("g2p_language_map"));
    if (assets.has("g2p_normalization") && !assets.get("g2p_normalization").str().empty()) {
        std::string declared = Json::parse_file(asset("g2p_normalization")).get("contract_sha256").str();
        if (!declared.empty() && declared != kSpokenTextNormalizerSha256) {
            throw std::runtime_error("Bundle spoken-text normalizer contract does not match this runtime: bundle=" + declared +
                                     " runtime=" + kSpokenTextNormalizerSha256);
        }
    }
    if (assets.has("g2p_pronunciation_overrides")) {
        try {
            overrides_ = Json::parse_file(asset("g2p_pronunciation_overrides"));
        } catch (const std::exception&) {
            overrides_ = Json(Json::Object{});  // optional asset
        }
    }
    const Json& boundaries = controls.get("word_boundaries");
    if (boundaries.has("g2p_output_symbol")) word_boundary_symbol_ = boundaries.get("g2p_output_symbol").str(" ");
    int64_t fixed = controls.get("fixed_shapes").get("g2p_text_tokens").integer(0);
    if (fixed <= 0) fixed = config.get("fixed_text_tokens").integer(0);
    text_tokens_ = fixed > 0 ? fixed : 512;
    char_repeats_ = std::max<int>(1, static_cast<int>(tokenizer.get("char_repeats").integer(1)));
    lowercase_ = tokenizer.get("lowercase").boolean(true);
    text_pad_ = tokenizer.get("text_pad_index").integer(0);
    phoneme_pad_ = tokenizer.get("phoneme_pad_index").integer(0);
    phoneme_end_ = tokenizer.get("phoneme_end_index").integer(0);
    int64_t configured = config.get("chunk_max_chars").integer(config.get("phrase_chunk_max_chars").integer(0));
    if (configured <= 0) configured = 140;
    chunk_max_chars_ = static_cast<std::size_t>(std::min<int64_t>(configured, std::max<int64_t>(1, (text_tokens_ - 2) / char_repeats_)));
    for (const auto& [symbol, id] : tokenizer.get("text_symbols").items()) text_symbols_[symbol] = id.integer();
    for (const auto& [id, symbol] : tokenizer.get("phoneme_symbols").items()) phoneme_symbols_[std::stoll(id)] = symbol.str();
}

std::string G2P::model_language(const std::string& language) const {
    std::string normalized = encode(lower(strip(decode(language))));
    std::replace(normalized.begin(), normalized.end(), '-', '_');
    const Json& mapped = language_map_.get(normalized);
    return mapped.is_string() ? mapped.str() : normalized;
}

Phonemized G2P::phonemize(const std::string& text, const std::string& language) {
    std::string model = model_language(language);
    std::string key = model + '\x1f' + text;
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = cache_.begin(); it != cache_.end(); ++it) {
        if (it->first == key) {
            cache_.splice(cache_.begin(), cache_, it);
            return cache_.front().second;
        }
    }
    Phonemized result = run(decode(text), model);
    cache_.emplace_front(key, result);
    if (cache_.size() > kCacheSize) cache_.pop_back();
    return result;
}

Phonemized G2P::run(const Text& text, const std::string& language) {
    Phonemized out;
    std::vector<std::string>& phones = out.phones;
    phones.push_back(kSilence);
    std::vector<double> confidences;
    auto segments = punctuated_segments(text, chunk_max_chars_);
    for (std::size_t index = 0; index < segments.size(); ++index) {
        const auto& [segment, tokens] = segments[index];
        Prediction prediction = predict_segment(segment, language);
        confidences.push_back(prediction.confidence);
        for (const Json& item : prediction.overrides.elements()) out.pronunciation_overrides.push(item);
        for (const Json& item : prediction.repairs.elements()) out.terminal_tail_repairs.push(item);
        bool word_open = false;
        for (std::size_t position = 0; position < prediction.phones.size(); ++position) {
            const std::string& phone = prediction.phones[position];
            if (phone == kWordBoundary) {
                if (is_internal_word_boundary(prediction.phones, position)) word_open = false;
                continue;
            }
            if (is_silence(phone)) continue;
            if (!word_open) {
                if (!out.word_starts.empty() || phones.size() > 1) {
                    if (phones.back() != kSilence) {
                        out.word_boundary_candidates.push_back(static_cast<int>(phones.size()));
                        phones.push_back(kSilence);
                    }
                    out.word_starts.push_back(static_cast<int>(phones.size()));
                }
                word_open = true;
            }
            phones.push_back(phone);
        }
        if (phones.size() > 1) {
            for (const std::string& token : tokens) {
                if (!phone_to_id_.count(token)) continue;
                Json event{Json::Object{}};
                event.set("segment_index", Json(index));
                event.set("phone_index", Json(phones.size()));
                event.set("phone", Json(token));
                out.punctuation.push(std::move(event));
                phones.push_back(token);
                phones.push_back(kSilence);
            }
        }
        out.segments.push_back(encode(segment));
    }
    if (phones.back() != kSilence) phones.push_back(kSilence);
    if (phones.size() <= 2 && std::all_of(phones.begin(), phones.end(), is_silence)) {
        throw std::invalid_argument("G2P produced no usable Scylla's Band phone symbols");
    }
    std::set<std::string> unknown;
    for (const std::string& phone : phones) {
        if (!phone_to_id_.count(phone)) unknown.insert(phone);
    }
    if (!unknown.empty()) {
        std::string list;
        for (const std::string& phone : unknown) list += (list.empty() ? "" : ", ") + phone;
        throw std::invalid_argument("G2P produced symbols outside the phone vocabulary: " + list);
    }
    out.confidence = confidences.empty() ? 0.0 : *std::min_element(confidences.begin(), confidences.end());
    return out;
}

G2P::Prediction G2P::predict_segment(const Text& text, const std::string& language) {
    const std::string key = language + '\x1f' + encode(text);
    for (auto it = segment_cache_.begin(); it != segment_cache_.end(); ++it) {
        if (it->first == key) {
            segment_cache_.splice(segment_cache_.begin(), segment_cache_, it);
            return segment_cache_.front().second;
        }
    }
    Prediction prediction = predict_repaired(text, language);
    segment_cache_.emplace_front(key, prediction);
    if (segment_cache_.size() > kCacheSize) segment_cache_.pop_back();
    return prediction;
}

G2P::Prediction G2P::predict_repaired(const Text& text, const std::string& language) {
    Prediction prediction = predict_raw(text, language);
    apply_overrides(prediction, text, language);
    // Trim a short CTC tail hallucination after a correctly decoded final word.
    Text word = terminal_word(text);
    if (word.empty()) return prediction;
    std::string key = language + '\x1f' + encode(word);
    std::vector<std::string> isolated;
    bool cached = false;
    for (const auto& entry : word_cache_) {
        if (entry.first == key) {
            isolated = entry.second;
            cached = true;
            break;
        }
    }
    if (!cached) {
        Prediction alone = predict_raw(word, language);
        apply_overrides(alone, word, language);
        isolated = alone.phones;
        word_cache_.emplace_front(key, isolated);
        if (word_cache_.size() > kCacheSize) word_cache_.pop_back();
    }
    std::vector<std::string>& phrase = prediction.phones;
    if (phrase.empty() || isolated.empty()) return prediction;
    const std::ptrdiff_t count = static_cast<std::ptrdiff_t>(isolated.size()), size = static_cast<std::ptrdiff_t>(phrase.size());
    for (std::ptrdiff_t start = size - count; start >= std::max<std::ptrdiff_t>(0, size - count - 2); --start) {
        if (start < 0 || !std::equal(isolated.begin(), isolated.end(), phrase.begin() + start)) continue;
        std::size_t removed = static_cast<std::size_t>(size - (start + count));
        if (removed > 0 && removed <= 2) {
            std::vector<std::string> tail(phrase.begin() + start + count, phrase.end());
            phrase.resize(static_cast<std::size_t>(start + count));
            Json repair{Json::Object{}};
            repair.set("word", Json(encode(word)));
            repair.set("phones", Json::array_of(isolated));
            repair.set("removed", Json::array_of(tail));
            prediction.repairs.push(std::move(repair));
            break;
        }
    }
    return prediction;
}

G2P::Prediction G2P::predict_raw(const Text& text, const std::string& language) {
    std::vector<int64_t> ids = encode_text(text, language);
    int64_t symbols = 0;
    std::vector<float> logits = infer_(ids, symbols);
    if (symbols <= 0 || logits.size() % static_cast<std::size_t>(symbols) != 0) throw std::runtime_error("G2P graph returned malformed logits");
    const std::size_t classes = static_cast<std::size_t>(symbols), frames = logits.size() / classes;
    Prediction out;
    std::vector<double> emitted;
    int64_t previous = -1;
    for (std::size_t frame = 0; frame < frames; ++frame) {
        const float* row = logits.data() + frame * classes;
        std::size_t best = 0;
        for (std::size_t k = 1; k < classes; ++k) {
            if (row[k] > row[best]) best = k;
        }
        const int64_t token = static_cast<int64_t>(best);
        if (token == previous) continue;
        previous = token;
        if (token == phoneme_pad_) continue;
        if (token == phoneme_end_) break;
        auto it = phoneme_symbols_.find(token);
        if (it == phoneme_symbols_.end()) continue;
        double total = 0.0;
        for (std::size_t k = 0; k < classes; ++k) total += std::exp(static_cast<double>(row[k]) - row[best]);
        double probability = 1.0 / total;
        if (it->second == word_boundary_symbol_) {
            out.phones.push_back(kWordBoundary);
            emitted.push_back(probability);
            continue;
        }
        if (skip_output_symbol(it->second)) continue;
        out.phones.push_back(it->second);
        emitted.push_back(probability);
    }
    out.confidence = prob_product(emitted);
    return out;
}

void G2P::apply_overrides(Prediction& prediction, const Text& text, const std::string& language) const {
    const Json& languages = overrides_.has("languages") ? overrides_.get("languages") : overrides_;
    const Json& raw = languages.get(language);
    if (!raw.is_object() || raw.size() == 0) return;
    std::map<std::string, Json> table;  // sorted like sorted(overrides.items())
    for (const auto& [word, spec] : raw.items()) {
        std::string key = encode(lower(strip(decode(word))));
        if (!key.empty()) table[key] = spec;
    }
    std::set<std::string> words;  // [A-Za-z]+(?:'[A-Za-z]+)?, lowercased
    for (std::size_t pos = 0; pos < text.size();) {
        if (!is_ascii_letter(text[pos])) {
            ++pos;
            continue;
        }
        std::size_t end = pos;
        while (end < text.size() && is_ascii_letter(text[end])) ++end;
        if (end + 1 < text.size() && text[end] == '\'' && is_ascii_letter(text[end + 1])) {
            ++end;
            while (end < text.size() && is_ascii_letter(text[end])) ++end;
        }
        words.insert(encode(lower(text.substr(pos, end - pos))));
        pos = end;
    }
    if (table.empty() || words.empty()) return;
    for (const auto& [word, spec] : table) {
        if (!words.count(word)) continue;
        std::vector<std::string> target = phone_list(spec, "phones"), source = phone_list(spec, "replace");
        if (target.empty() || source.empty()) continue;
        int count = replace_subsequence(prediction.phones, source, target);
        if (!count) continue;
        Json applied{Json::Object{}};
        applied.set("word", Json(word));
        applied.set("replace", Json::array_of(source));
        applied.set("phones", Json::array_of(target));
        applied.set("count", Json(count));
        prediction.overrides.push(std::move(applied));
    }
}

std::vector<int64_t> G2P::encode_text(const Text& text, const std::string& language) const {
    auto language_token = text_symbols_.find("<" + language + ">");
    if (language_token == text_symbols_.end()) throw std::invalid_argument("G2P language '" + language + "' is not supported by this bundle");
    Text work = lowercase_ ? lower(text) : text;
    std::vector<int64_t> sequence{language_token->second};
    int emitted = 0;
    std::string key;
    for (char32_t cp : work) {
        key.clear();
        append_utf8(key, cp);
        auto it = text_symbols_.find(key);
        if (it == text_symbols_.end()) continue;
        sequence.insert(sequence.end(), static_cast<std::size_t>(char_repeats_), it->second);
        ++emitted;
    }
    auto end = text_symbols_.find("<end>");
    sequence.push_back(end == text_symbols_.end() ? text_pad_ : end->second);
    if (emitted <= 0) throw std::invalid_argument("G2P text has no characters supported by this bundle");
    if (static_cast<int64_t>(sequence.size()) > text_tokens_) {
        throw std::invalid_argument("G2P input encodes to " + std::to_string(sequence.size()) + " tokens, but this bundle supports at most " +
                                    std::to_string(text_tokens_) + ". Split the text into shorter chunks.");
    }
    sequence.resize(static_cast<std::size_t>(text_tokens_), text_pad_);
    return sequence;
}

}  // namespace scyllasband
