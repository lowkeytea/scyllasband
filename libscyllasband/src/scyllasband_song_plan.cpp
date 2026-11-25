#include "scyllasband_song_plan.h"

#include "scyllasband_numeric.h"
#include "scyllasband_unicode.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace scyllasband {

namespace {

constexpr int kMaxConsonantFrames = 8;
// Vowel symbols of the phone vocabulary (the first character of a phone decides): the same set the trainer uses.
const unicode::Text kVowels = unicode::decode("aeiouyɑɐɒæəɚɛɜɝɪɔʊʌɨʉøœɯɤɵɘɞɶ");
// v5: the vowel set its sung rows were timed with (the first character of a phone once its marks are stripped).
const unicode::Text kVowelsV5 = kVowels + unicode::decode("ᵻ");
const char* const kNeedsV4 = "Singing needs a v4 bundle with delivery modes (controls.delivery_modes); this bundle has none";

// Median share of a word's acoustic frames its vowels take, measured by forced alignment on the training corpus: sung 0.51,
// rap 0.42 (rap at a higher share stretched vowels and squeezed consonants).
double vowel_share_of(const std::string& mode) {
    if (mode == "rap") return 0.42;
    return 0.51;   // sung, ballad, spoken
}

std::string strip(const std::string& text) { return unicode::encode(unicode::strip(unicode::decode(text))); }
std::string lower(const std::string& text) { return unicode::encode(unicode::lower(unicode::decode(text))); }

std::string join(const std::vector<std::string>& parts, const std::string& separator) {
    std::string out;
    for (std::size_t i = 0; i < parts.size(); ++i) out += (i ? separator : "") + parts[i];
    return out;
}

std::string quote(const std::string& text) { return "'" + text + "'"; }

Json note_json(const ScoreNote& note) {
    Json out{Json::Object{}};
    out.set("label", Json(note.label));
    out.set("note", Json(note.name()));
    out.set("start", Json(round_to(note.start, 4)));
    out.set("end", Json(round_to(note.end, 4)));
    return out;
}

// The delivery mode the bundle sings `score` in; throws when its style is not one of the bundle's.
std::string check_style(const SingingControls& controls, const Score& score) {
    for (const auto& [style, mode] : style_table(controls)) {
        if (style == score.style) return mode;
    }
    throw ScoreError("Style " + quote(score.style) + " (" + score.mode + ") is not a singing style of this bundle; choose one of " +
                     join(singing_styles(controls), ", "));
}

// A phone without its stress, length and combining marks; empty for a token that is only marks.
unicode::Text sounding(const std::string& phone) {
    unicode::Text out;
    for (char32_t cp : unicode::decode(phone)) {
        if (!is_non_acoustic_modifier(unicode::encode(unicode::Text(1, cp)))) out.push_back(cp);
    }
    return out;
}

bool is_vowel_v5(const std::string& phone) {
    const unicode::Text core = sounding(phone);
    return !core.empty() && kVowelsV5.find(core[0]) != unicode::Text::npos;
}

int64_t edit_distance(const std::vector<unicode::Text>& a, const std::vector<unicode::Text>& b) {
    std::vector<int64_t> row(b.size() + 1);
    for (std::size_t j = 0; j <= b.size(); ++j) row[j] = static_cast<int64_t>(j);
    for (std::size_t i = 1; i <= a.size(); ++i) {
        int64_t previous = row[0];
        row[0] = static_cast<int64_t>(i);
        for (std::size_t j = 1; j <= b.size(); ++j) {
            const int64_t above = row[j];
            row[j] = std::min({row[j] + 1, row[j - 1] + 1, previous + (a[i - 1] != b[j - 1] ? 1 : 0)});
            previous = above;
        }
    }
    return row.back();
}

// The phones of `text` as the G2P groups them into words, silences and punctuation dropped.
std::vector<std::vector<std::string>> phone_groups(const std::string& text, SongFrontend& frontend, const std::string& language) {
    const Phonemized result = frontend.phonemize(text, language);
    std::vector<std::size_t> starts = {0};
    for (int start : result.word_starts) starts.push_back(static_cast<std::size_t>(start));
    starts.push_back(result.phones.size());
    std::vector<std::vector<std::string>> out;
    for (std::size_t k = 0; k + 1 < starts.size(); ++k) {
        std::vector<std::string> piece;
        for (std::size_t i = starts[k]; i < starts[k + 1] && i < result.phones.size(); ++i) {
            if (result.phones[i] != kSilencePhone && !is_punctuation_pause(result.phones[i])) piece.push_back(result.phones[i]);
        }
        if (!piece.empty()) out.push_back(std::move(piece));
    }
    return out;
}

// The marks-free phones of a piece (line_phones compares these).
std::vector<unicode::Text> sounding_phones(const std::vector<std::string>& phones, std::size_t from, std::size_t to) {
    std::vector<unicode::Text> out;
    for (std::size_t i = from; i < to; ++i) {
        unicode::Text core = sounding(phones[i]);
        if (!core.empty()) out.push_back(std::move(core));
    }
    return out;
}

std::optional<double> voice_center(const SingingControls& controls, const std::string& voice) {
    auto found = controls.voice_center.find(voice);
    return found == controls.voice_center.end() ? std::nullopt : std::optional<double>(found->second);
}

}  // namespace

std::vector<std::string> singing_modes(const SingingControls& controls) {
    std::vector<std::string> listed;
    for (const std::string& mode : controls.modes) {
        if (controls.mode_ids.count(mode)) listed.push_back(mode);
    }
    if (!listed.empty()) return listed;
    for (const auto& [mode, id] : controls.mode_ids) {   // std::map: sorted, as Python's sorted()
        if (mode != "spoken") listed.push_back(mode);
    }
    return listed;
}

StyleTable style_table(const SingingControls& controls) {
    const std::vector<std::string> modes = singing_modes(controls);
    auto sung = [&](const std::string& mode) { return std::find(modes.begin(), modes.end(), mode) != modes.end(); };
    StyleTable out;
    if (!controls.v5) {
        for (const auto& [style, mode] : style_modes()) {
            if (sung(mode)) out.emplace_back(style, mode);
        }
        return out;
    }
    for (const std::string& mode : modes) out.emplace_back(mode, mode);
    for (const auto& [alias, style] : controls.aliases) {
        const bool named = std::any_of(out.begin(), out.end(), [&](const auto& entry) { return entry.first == alias; });
        if (sung(style) && !named) out.emplace_back(alias, style);
    }
    return out;
}

std::vector<std::string> singing_styles(const SingingControls& controls) {
    std::vector<std::string> out;
    for (const auto& [style, mode] : style_table(controls)) out.push_back(style);
    return out;
}

Score parse_bundle_score(const std::string& text, const SingingControls& controls) {
    if (!controls.v5) return parse_score_text(text);
    const StyleTable styles = style_table(controls);
    return parse_score_text(text, &styles, true);
}

int bundle_register_shift(const SingingControls& controls, const Score& score, const std::string& voice) {
    int shift = register_shift(score, voice_center(controls, voice));
    const std::vector<int> pitched = score.pitched();
    if (controls.register_floor && !pitched.empty()) {
        const int lowest = *std::min_element(pitched.begin(), pitched.end());
        while (lowest + shift < *controls.register_floor) shift += 12;
    }
    return shift;
}

// --- phones, durations and melody ----------------------------------------------------------------------------------------------

std::pair<std::vector<std::string>, std::vector<int>> word_phones(const std::vector<std::string>& words, SongFrontend& frontend,
                                                                  const std::string& language) {
    // The phrase is phonemized as one text so the G2P sees its context; when its words do not line up one-to-one with the
    // score's, each word is phonemized on its own, so the count always matches.
    const Phonemized result = frontend.phonemize(join(words, " "), language);
    // Punctuation phones are dropped (a score has none; the G2P may still produce some), with their silence when it duplicates a
    // neighbouring one, keeping the word starts on the same phones.
    std::vector<std::string> kept;
    std::vector<int> starts;
    const std::set<int> drop(result.word_starts.begin(), result.word_starts.end());
    for (std::size_t index = 0; index < result.phones.size(); ++index) {
        const std::string& phone = result.phones[index];
        if (is_punctuation_pause(phone) || (phone == kSilencePhone && !kept.empty() && kept.back() == kSilencePhone)) continue;
        if (drop.count(static_cast<int>(index))) starts.push_back(static_cast<int>(kept.size()));
        kept.push_back(phone);
    }
    if (starts.size() + 1 == words.size()) return {kept, starts};
    std::vector<std::string> phones = {kSilencePhone};
    starts.clear();
    for (const std::string& word : words) {
        const Phonemized single = frontend.phonemize(word, language);
        std::vector<std::string> inner;
        for (const std::string& phone : single.phones) {
            if (phone != kSilencePhone && !is_punctuation_pause(phone)) inner.push_back(phone);
        }
        if (inner.empty()) throw ScoreError("No phones for the word " + quote(word));
        if (phones.size() > 1) {
            phones.push_back(kSilencePhone);
            starts.push_back(static_cast<int>(phones.size()));
        }
        phones.insert(phones.end(), inner.begin(), inner.end());
    }
    phones.push_back(kSilencePhone);
    return {phones, starts};
}

std::vector<std::string> stressed_pronoun(const std::vector<std::string>& phones) {
    if (std::find(phones.begin(), phones.end(), "ˈ") != phones.end()) return phones;
    const auto vowel = std::find_if(phones.begin(), phones.end(), is_vowel_v5);
    if (vowel == phones.end()) return phones;
    std::vector<std::string> out(phones.begin(), vowel);
    if (!out.empty() && out.back() == "ˌ") out.pop_back();  // a secondary stress on the vowel becomes the primary one
    out.push_back("ˈ");
    out.insert(out.end(), vowel, phones.end());
    return out;
}

std::vector<std::vector<std::string>> line_phones(const std::vector<std::string>& words, SongFrontend& frontend, const std::string& language) {
    // The pronoun "I" as its own word (not "I'm", "I'll", "I'd", "I've") is stressed in English lyrics.
    auto sung = [&](std::vector<std::vector<std::string>> pieces) {
        if (language != "en_us" && language != "en_gb") return pieces;
        for (std::size_t i = 0; i < pieces.size() && i < words.size(); ++i) {
            if (words[i] == "i" || words[i] == "I") pieces[i] = stressed_pronoun(pieces[i]);
        }
        return pieces;
    };
    std::vector<std::vector<std::string>> found = phone_groups(join(words, " "), frontend, language);
    if (found.size() == words.size()) return sung(std::move(found));
    std::vector<std::vector<std::string>> alone;
    for (const std::string& word : words) {
        std::vector<std::string> phones;
        for (const std::vector<std::string>& group : phone_groups(word, frontend, language)) phones.insert(phones.end(), group.begin(), group.end());
        alone.push_back(std::move(phones));
    }
    std::vector<std::string> flat;
    for (const std::vector<std::string>& group : found) flat.insert(flat.end(), group.begin(), group.end());
    const std::size_t count = words.size(), size = flat.size();
    const int64_t far = 1000000000;
    std::vector<std::vector<int64_t>> best(count + 1, std::vector<int64_t>(size + 1, far));
    std::vector<std::vector<std::size_t>> back(count + 1, std::vector<std::size_t>(size + 1, 0));
    best[0][0] = 0;
    for (std::size_t i = 1; i <= count; ++i) {
        const std::vector<unicode::Text> wanted = sounding_phones(alone[i - 1], 0, alone[i - 1].size());
        for (std::size_t b = i; b <= size; ++b) {
            const int64_t reach = static_cast<int64_t>(b) - static_cast<int64_t>(alone[i - 1].size()) - 6;
            for (std::size_t a = static_cast<std::size_t>(std::max<int64_t>(static_cast<int64_t>(i) - 1, reach)); a < b; ++a) {
                if (best[i - 1][a] >= far) continue;
                const std::vector<unicode::Text> piece = sounding_phones(flat, a, b);
                if (piece.empty()) continue;
                const int64_t cost = best[i - 1][a] + edit_distance(piece, wanted) + (flat[a] == "ː" || flat[a] == "ˑ" ? 2 : 0);
                if (cost < best[i][b]) {
                    best[i][b] = cost;
                    back[i][b] = a;
                }
            }
        }
    }
    if (best[count][size] >= far) throw ScoreError("Could not cut the phrase's phones into its words: " + join(words, " "));
    std::vector<std::vector<std::string>> out(count);
    std::size_t b = size;
    for (std::size_t i = count; i > 0; --i) {
        const std::size_t a = back[i][b];
        out[i - 1].assign(flat.begin() + static_cast<std::ptrdiff_t>(a), flat.begin() + static_cast<std::ptrdiff_t>(b));
        b = a;
    }
    return sung(std::move(out));
}

std::pair<std::vector<std::string>, std::vector<int>> word_layout(const std::vector<std::vector<std::string>>& groups) {
    std::vector<std::string> phones = {kSilencePhone};
    std::vector<int> starts;
    for (std::size_t index = 0; index < groups.size(); ++index) {
        if (index) {
            phones.push_back(kSilencePhone);
            starts.push_back(static_cast<int>(phones.size()));
        }
        phones.insert(phones.end(), groups[index].begin(), groups[index].end());
    }
    phones.push_back(kSilencePhone);
    return {phones, starts};
}

std::vector<int> score_durations(const std::vector<std::string>& phones, const std::vector<int>& word_starts, const std::vector<ScoreWord>& words,
                                 int64_t total_frames, double vowel_share, int hop, int sample_rate) {
    std::size_t first = 0;
    while (first < phones.size() && phones[first] == kSilencePhone) ++first;
    std::vector<std::size_t> starts = {first};
    for (int start : word_starts) starts.push_back(static_cast<std::size_t>(start));
    starts.push_back(phones.size());
    if (starts.size() - 1 != words.size()) {
        std::vector<std::string> texts;
        for (const ScoreWord& word : words) texts.push_back(word.text);
        throw ScoreError(std::to_string(starts.size() - 1) + " phonemized words for " + std::to_string(words.size()) + " score words: " + join(texts, " "));
    }
    std::vector<std::pair<int64_t, int64_t>> bounds;
    for (const ScoreWord& word : words) {
        bounds.emplace_back(round_half_even(word.start * sample_rate / hop), round_half_even(word.end * sample_rate / hop));
    }
    std::vector<int64_t> durations(phones.size(), 0);
    int64_t cursor = bounds[0].first;
    durations[0] = cursor;
    for (std::size_t gi = 0; gi + 1 < starts.size(); ++gi) {
        const std::size_t g0 = starts[gi], g1 = starts[gi + 1];
        const bool word_between_silences = g1 > g0 && g1 <= phones.size() && phones[g1 - 1] == kSilencePhone &&
                                           std::find(phones.begin() + static_cast<std::ptrdiff_t>(g0), phones.begin() + static_cast<std::ptrdiff_t>(g1 - 1),
                                                     kSilencePhone) == phones.begin() + static_cast<std::ptrdiff_t>(g1 - 1);
        if (!word_between_silences) throw ScoreError("The phones of " + quote(words[gi].text) + " are not one word between two silences");
        const std::size_t last = g1 - 1;  // the <sil> that closes the word
        const int64_t f0 = std::max(bounds[gi].first, cursor), f1 = bounds[gi].second;
        std::vector<std::size_t> vowels, consonants;
        for (std::size_t i = g0; i < last; ++i) {
            if (sounding(phones[i]).empty()) continue;
            (is_vowel_v5(phones[i]) ? vowels : consonants).push_back(i);
        }
        const int64_t vowel_count = static_cast<int64_t>(vowels.size()), consonant_count = static_cast<int64_t>(consonants.size());
        const int64_t n = std::max(f1 - f0, vowel_count + consonant_count);
        int64_t c = 0;
        if (!vowels.empty()) {
            c = std::min<int64_t>(kMaxConsonantFrames, std::max<int64_t>(1, round_half_even(static_cast<double>(n) * (1 - vowel_share) /
                                                                                            static_cast<double>(std::max<int64_t>(1, consonant_count)))));
            while (consonant_count > 0 && n - c * consonant_count < vowel_count && c > 1) --c;
        } else {
            c = std::max<int64_t>(1, floor_div(n, std::max<int64_t>(1, consonant_count)));
        }
        for (std::size_t i : consonants) durations[i] = c;
        const int64_t left = n - c * consonant_count;
        if (!vowels.empty()) {
            const int64_t shared = std::max(left, vowel_count);
            const int64_t share = floor_div(shared, vowel_count), extra = floor_mod(shared, vowel_count);
            for (std::size_t k = 0; k < vowels.size(); ++k) durations[vowels[k]] = share + (static_cast<int64_t>(k) < extra ? 1 : 0);
        } else if (!consonants.empty()) {
            durations[consonants.back()] += std::max<int64_t>(0, left);
        }
        int64_t end = f0;
        for (std::size_t i = g0; i < last; ++i) end += durations[i];
        const int64_t following = gi + 1 < bounds.size() ? bounds[gi + 1].first : std::max(end + 2, total_frames);
        durations[last] = std::max<int64_t>(0, following - end);
        cursor = end + durations[last];
    }
    return std::vector<int>(durations.begin(), durations.end());
}

bool is_vowel(const std::string& phone) {
    if (phone.empty() || phone[0] == '<') return false;
    const unicode::Text text = unicode::decode(phone);
    return !text.empty() && kVowels.find(text[0]) != unicode::Text::npos;
}

std::vector<int> phone_durations(const std::vector<std::string>& phones, const std::vector<int>& word_starts, const std::vector<ScoreWord>& words,
                                 int64_t total_frames, double vowel_share, int hop, int sample_rate) {
    std::size_t first = 0;
    while (first < phones.size() && phones[first] == kSilencePhone) ++first;
    std::vector<std::size_t> starts = {first};
    for (int start : word_starts) starts.push_back(static_cast<std::size_t>(start));
    starts.push_back(phones.size());
    if (starts.size() - 1 != words.size()) {
        std::vector<std::string> texts;
        for (const ScoreWord& word : words) texts.push_back(word.text);
        throw ScoreError(std::to_string(starts.size() - 1) + " phonemized words for " + std::to_string(words.size()) + " score words: " + join(texts, " "));
    }
    std::vector<std::pair<int64_t, int64_t>> bounds;
    for (const ScoreWord& word : words) {
        bounds.emplace_back(round_half_even(word.start * sample_rate / hop), round_half_even(word.end * sample_rate / hop));
    }
    std::vector<int64_t> durations(phones.size(), 0);
    for (std::size_t i = 0; i < first; ++i) {
        if (phones[i] == kSilencePhone) durations[i] = bounds[0].first;
    }
    for (std::size_t gi = 0; gi + 1 < starts.size(); ++gi) {
        const std::size_t g0 = starts[gi], g1 = starts[gi + 1];
        const int64_t f0 = bounds[gi].first, f1 = bounds[gi].second;
        std::vector<std::size_t> vowels, consonants;
        int64_t acoustic = 0;
        for (std::size_t i = g0; i < g1; ++i) {
            const std::string& phone = phones[i];
            if (phone == kSilencePhone || is_punctuation_pause(phone) || is_non_acoustic_modifier(phone)) continue;
            ++acoustic;
            (is_vowel(phone) ? vowels : consonants).push_back(i);
        }
        const int64_t n = std::max(f1 - f0, acoustic);
        const int64_t consonant_count = static_cast<int64_t>(consonants.size());
        int64_t c = 0;
        if (!vowels.empty()) {
            c = std::min<int64_t>(kMaxConsonantFrames, std::max<int64_t>(1, round_half_even(static_cast<double>(n) * (1 - vowel_share) /
                                                                                            static_cast<double>(std::max<int64_t>(1, consonant_count)))));
        } else {
            c = std::max<int64_t>(1, floor_div(n, std::max<int64_t>(1, consonant_count)));
        }
        for (std::size_t i : consonants) durations[i] = c;
        const int64_t left = n - c * consonant_count;
        if (!vowels.empty()) {
            const int64_t count = static_cast<int64_t>(vowels.size());
            const int64_t share = floor_div(left, count), extra = floor_mod(left, count);
            for (std::size_t k = 0; k < vowels.size(); ++k) durations[vowels[k]] = std::max<int64_t>(1, share + (static_cast<int64_t>(k) < extra ? 1 : 0));
        } else if (!consonants.empty()) {
            durations[consonants.back()] += left;
        }
        const int64_t following = gi + 1 < bounds.size() ? bounds[gi + 1].first : std::max(f1 + 2, total_frames);
        std::optional<std::size_t> last_silence;
        for (std::size_t i = g0; i < g1; ++i) {
            if (phones[i] == kSilencePhone) last_silence = i;
        }
        if (last_silence) durations[*last_silence] = std::max<int64_t>(0, following - (f0 + n));
    }
    return std::vector<int>(durations.begin(), durations.end());
}

std::vector<float> frame_melody(const std::vector<ScoreNote>& notes, int64_t frames, int hop, int sample_rate, double midi_center, double midi_scale) {
    const std::size_t T = static_cast<std::size_t>(frames);
    std::vector<float> out(T * 3, 0.0f);
    std::vector<double> centres(T);
    for (std::size_t t = 0; t < T; ++t) centres[t] = (static_cast<double>(t) * hop + static_cast<double>(hop / 2)) / sample_rate;
    for (const ScoreNote& note : notes) {
        if (!note.midi || note.end <= note.start) continue;
        const float pitch = static_cast<float>((*note.midi - midi_center) / midi_scale);
        for (std::size_t t = 0; t < T; ++t) {
            if (centres[t] >= note.start && centres[t] < note.end) {
                out[t * 3] = pitch;
                out[t * 3 + 1] = static_cast<float>((centres[t] - note.start) / (note.end - note.start));
                out[t * 3 + 2] = 1.0f;
            }
        }
    }
    return out;
}

// --- the plan ----------------------------------------------------------------------------------------------------------------

Json PhrasePlan::to_json(bool with_melody) const {
    Json out{Json::Object{}};
    out.set("index", Json(index));
    out.set("start", Json(round_to(phrase.start, 4)));
    out.set("end", Json(round_to(phrase.end, 4)));
    out.set("text", Json(phrase.text));
    out.set("frames", Json(frames));
    out.set("phones", Json::array_of(sentence.phones));
    out.set("durations", Json::array_of(durations));
    Json::Array note_items;
    for (const ScoreNote& note : phrase.notes) note_items.push_back(note_json(note));
    out.set("notes", Json(std::move(note_items)));
    if (with_melody) {
        Json::Array values;
        values.reserve(melody.size());
        for (float value : melody) values.emplace_back(static_cast<double>(value));
        out.set("melody", Json(std::move(values)));
    }
    return out;
}

double SongPlan::seconds() const {
    double out = 0.0;
    for (const PhrasePlan& phrase : phrases) out = std::max(out, phrase.phrase.end);
    return out;
}

Json SongPlan::to_json(bool with_melody) const {
    Json out{Json::Object{}};
    out.set("voice", Json(voice));
    out.set("language", Json(language));
    out.set("mode", Json(mode));
    out.set("mode_id", Json(mode_id));
    out.set("register_shift", Json(register_shift));
    out.set("transpose", Json(transpose));
    out.set("bpm", Json(score.bpm));
    out.set("style", Json(score.style));
    out.set("text", Json(score.text));
    out.set("seconds", Json(round_to(seconds(), 3)));
    Json::Array items;
    for (const PhrasePlan& phrase : phrases) items.push_back(phrase.to_json(with_melody));
    out.set("phrases", Json(std::move(items)));
    return out;
}

SongPlan plan_song(const Score& input, SongFrontend& frontend, const std::string& voice, const std::string& language, const PlanSettings& settings) {
    const SingingControls& controls = frontend.singing();
    if (!controls.enabled()) throw std::invalid_argument(kNeedsV4);
    const std::string mode = check_style(controls, input);
    const int sample_rate = frontend.sample_rate(), hop = frontend.latent_hop();
    const int64_t max_frames = frontend.max_phrase_frames();
    Score score = input.transposed(settings.transpose);
    const int shift = settings.use_register ? bundle_register_shift(controls, score, voice) : 0;
    score = score.transposed(shift);
    const double longest = static_cast<double>((max_frames - 2) * hop) / sample_rate - 2 * settings.pad;
    const double max_seconds = std::min(settings.max_seconds.value_or(kMaxPhraseSeconds), longest);
    const double share = settings.vowel_share.value_or(vowel_share_of(mode));
    SongPlan plan;
    plan.voice = voice;
    plan.language = language;
    plan.mode = mode;
    plan.mode_id = controls.mode_ids.at(mode);
    plan.register_shift = shift;
    plan.transpose = settings.transpose;
    const std::vector<Phrase> phrases = split_phrases(score, settings.phrase_rest, max_seconds, settings.pad);
    for (std::size_t index = 0; index < phrases.size(); ++index) {
        const Phrase& phrase = phrases[index];
        std::vector<std::string> words;
        for (const ScoreWord& word : phrase.words) words.push_back(word.text);
        const int64_t total = round_half_even(phrase.notes.back().end * sample_rate / hop) + 2;
        // v5: read and timed as the model's sung rows were (the phrase read once, no non-verbal table).
        auto [phones, starts] = controls.v5 ? word_layout(line_phones(words, frontend, language)) : word_phones(words, frontend, language);
        std::vector<int> durations = controls.v5 ? score_durations(phones, starts, phrase.words, total, share, hop, sample_rate)
                                                 : phone_durations(phones, starts, phrase.words, total, share, hop, sample_rate);
        int64_t frames = 0;
        for (int value : durations) frames += value;
        if (frames > max_frames) {
            throw OverlongError("Phrase " + std::to_string(index + 1) + " ('" + unicode::encode(unicode::decode(phrase.text).substr(0, 40)) + "') needs " +
                                      std::to_string(frames) + " latent frames, but this bundle holds at most " + std::to_string(max_frames) +
                                      "; add a rest inside it or shorten its notes");
        }
        PhrasePlan item;
        item.index = static_cast<int>(index);
        item.phrase = phrase;
        item.sentence.phones = std::move(phones);
        item.sentence.word_starts = std::move(starts);
        item.durations = std::move(durations);
        item.frames = frames;
        item.melody = frame_melody(phrase.notes, frames, hop, sample_rate, controls.midi_center, controls.midi_scale);
        plan.phrases.push_back(std::move(item));
    }
    plan.score = std::move(score);
    return plan;
}

// --- requests ------------------------------------------------------------------------------------------------------------------

SingSettings resolve_sing_settings(const SongFrontend& frontend, const SingRequest& request) {
    const SingingControls& singing = frontend.singing();
    SingSettings out;
    if (request.steps < 0) throw std::invalid_argument("steps must be positive");
    out.steps = request.steps > 0 ? request.steps : (singing.steps > 0 ? singing.steps : 8);
    out.sampler = lower(request.sampler.empty() ? (singing.sampler.empty() ? std::string("heun") : singing.sampler) : request.sampler);
    if (out.sampler != "heun" && out.sampler != "euler") throw std::invalid_argument("Unsupported sampler '" + out.sampler + "'; choose heun or euler");
    const std::string preferred = singing.decoder.empty() ? std::string("vocos") : singing.decoder;
    // A preferred decoder the bundle does not carry gives way to vocos; on a v5 bundle only when the manifest marks that
    // decoder "required": false.
    out.decoder = request.decoder.empty() ? (frontend.has_decoder(preferred) || singing.decoder_required ? preferred : std::string("vocos"))
                                          : lower(request.decoder);
    if (!frontend.has_decoder(out.decoder)) {
        throw std::invalid_argument("This bundle has no '" + out.decoder + "' decoder; it has " + join(frontend.available_decoders(), ", "));
    }
    if (!request.guidance && singing.v5 && !singing.guidance_stated) {
        throw std::invalid_argument("This v5 bundle states no singing guidance (controls.delivery_modes.singing.guidance); pass guidance");
    }
    out.guidance = request.guidance.value_or(singing.guidance);
    if (!std::isfinite(out.guidance)) throw std::invalid_argument("guidance must be finite");
    out.guidance_steps = singing.guidance_steps;
    if (out.guidance_steps == "first_half") {
        out.guidance_until = 0.5;
    } else if (out.guidance_steps != "all") {
        throw std::invalid_argument("Unknown controls.delivery_modes.singing.guidance_steps '" + out.guidance_steps + "'; this runtime knows all, first_half");
    }
    return out;
}

ResolvedSing resolve_sing_request(SongFrontend& frontend, const Score& score, const SingRequest& request) {
    if (!frontend.singing().enabled()) throw std::invalid_argument(kNeedsV4);
    ResolvedSing out;
    out.voice = strip(request.voice);
    if (out.voice.empty()) throw std::invalid_argument("A voice is required");
    out.language = frontend.resolve_language(out.voice, request.language);
    check_style(frontend.singing(), score);
    out.score = score;
    out.settings = resolve_sing_settings(frontend, request);
    return out;
}

// --- fit -----------------------------------------------------------------------------------------------------------------------

Json SongFit::to_json() const {
    Json out{Json::Object{}};
    out.set("voice_center", voice_center ? Json(*voice_center) : Json());
    Json band{Json::Array{}};
    if (voice_center) {
        band.push(Json(band_low));
        band.push(Json(band_high));
    }
    out.set("band", band);
    out.set("transpose", Json(transpose));
    out.set("register_shift", Json(register_shift));
    out.set("mean_midi", Json(round_to(mean_midi, 3)));
    out.set("low_midi", Json(low_midi));
    out.set("high_midi", Json(high_midi));
    out.set("low_note", Json(midi_to_note(low_midi)));
    out.set("high_note", Json(midi_to_note(high_midi)));
    out.set("outside_semitones", Json(round_to(outside_semitones, 3)));
    out.set("suggested_transpose", Json(suggested_transpose));
    return out;
}

SongFit song_fit(const SingingControls& controls, const Score& score, const std::string& voice, int transpose, bool use_register) {
    SongFit fit;
    fit.voice_center = voice_center(controls, voice);
    fit.transpose = transpose;
    auto sung_mean = [&](int t, int* shift_out) {
        const Score moved = score.transposed(t);
        const int shift = use_register ? bundle_register_shift(controls, moved, voice) : 0;
        if (shift_out) *shift_out = shift;
        const std::vector<int> pitched = moved.pitched();
        long long sum = 0;
        for (int midi : pitched) sum += midi;
        return pitched.empty() ? 0.0 : static_cast<double>(sum) / static_cast<double>(pitched.size()) + shift;
    };
    fit.mean_midi = sung_mean(transpose, &fit.register_shift);
    const std::vector<int> pitched = score.transposed(transpose + fit.register_shift).pitched();
    if (!pitched.empty()) {
        fit.low_midi = *std::min_element(pitched.begin(), pitched.end());
        fit.high_midi = *std::max_element(pitched.begin(), pitched.end());
    }
    fit.suggested_transpose = transpose;
    if (fit.voice_center && !pitched.empty()) {
        fit.band_low = *fit.voice_center - kRegisterBelow;
        fit.band_high = *fit.voice_center + kRegisterAbove;
        auto outside = [&](double mean) { return std::max({0.0, fit.band_low - mean, mean - fit.band_high}); };
        fit.outside_semitones = outside(fit.mean_midi);
        if (fit.outside_semitones > 0.0) {
            double best = INFINITY;
            for (int step = 0; step <= 12; ++step) {   // 0, 1, -1, 2, -2, ... 6, -6: the first that fits, else the nearest
                const int candidate = step % 2 ? (step + 1) / 2 : -(step / 2);
                const double distance = outside(sung_mean(candidate, nullptr));
                if (distance < best - 1e-9) {
                    best = distance;
                    fit.suggested_transpose = candidate;
                }
                if (distance == 0.0) break;
            }
        }
    }
    return fit;
}

}  // namespace scyllasband
