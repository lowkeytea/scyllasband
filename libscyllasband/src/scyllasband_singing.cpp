#include "scyllasband_singing.h"

#include "scyllasband_events.h"
#include "scyllasband_noise.h"
#include "scyllasband_unicode.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <map>
#include <random>
#include <set>

namespace scyllasband {

namespace {

constexpr double kMidiCenter = 60.0, kMidiScale = 24.0;
// The band, in semitones around the voice's sung centre, that the song's mean pitch is moved into by octaves. Pitch accuracy
// held within +-5 semitones of the centre and collapsed at +-8 in the register test; +7.5 sang flat.
constexpr double kRegisterBelow = 3.0, kRegisterAbove = 5.0;
constexpr int kMaxConsonantFrames = 8;
// Vowel symbols of the phone vocabulary (the first character of a phone decides): the same set the trainer uses.
const unicode::Text kVowels = unicode::decode("aeiouyɑɐɒæəɚɛɜɝɪɔʊʌɨʉøœɯɤɵɘɞɶ");
const char* const kNoteNames[12] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};

// Style -> delivery mode, in the order styles are listed.
const std::vector<std::pair<std::string, std::string>> kStyleModes = {{"pop", "sung"}, {"sung", "sung"}, {"ballad", "ballad"}, {"rap", "rap"}};

// Median share of a word's acoustic frames its vowels take, measured by forced alignment on the training corpus.
double vowel_share_of(const std::string& mode) {
    if (mode == "rap") return 0.42;
    return 0.51;   // sung, ballad, spoken
}

std::string style_mode(const std::string& style) {
    for (const auto& [name, mode] : kStyleModes) {
        if (name == style) return mode;
    }
    return {};
}

int64_t floor_div(int64_t a, int64_t b) {
    int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}

int64_t floor_mod(int64_t a, int64_t b) { return a - floor_div(a, b) * b; }

// Python round() to `digits` decimals, for metadata.
double round_to(double value, int digits) {
    const double scale = std::pow(10.0, digits);
    return std::nearbyint(value * scale) / scale;
}

// int(round(x)): half to even, as Python rounds.
int64_t round_half_even(double value) { return static_cast<int64_t>(std::nearbyint(value)); }

// str(value) for a JSON scalar, as Python prints it.
std::string py_str(const Json& value) {
    switch (value.type()) {
        case Json::Type::string: return value.str();
        case Json::Type::null: return "None";
        case Json::Type::boolean: return value.boolean() ? "True" : "False";
        case Json::Type::number: {
            const double number = value.num();
            if (std::nearbyint(number) == number && std::fabs(number) < 1e15 && value.dump().find('.') == std::string::npos &&
                value.dump().find('e') == std::string::npos) {
                return std::to_string(static_cast<long long>(number));
            }
            return format_float(number);
        }
        default: return value.dump();
    }
}

std::string strip(const std::string& text) { return unicode::encode(unicode::strip(unicode::decode(text))); }
std::string lower(const std::string& text) { return unicode::encode(unicode::lower(unicode::decode(text))); }

// float(value) for a JSON number or numeric string; nullopt when Python's float() would raise.
std::optional<double> py_float(const Json& value) {
    if (value.is_number()) return value.num();
    if (value.type() == Json::Type::boolean) return value.boolean() ? 1.0 : 0.0;
    if (!value.is_string()) return std::nullopt;
    std::string text = strip(value.str());
    text.erase(std::remove(text.begin(), text.end(), '_'), text.end());
    if (text.empty()) return std::nullopt;
    const std::string folded = lower(text);
    for (const char* special : {"inf", "+inf", "-inf", "infinity", "+infinity", "-infinity", "nan", "+nan", "-nan"}) {
        if (folded == special) return folded[0] == '-' ? -INFINITY : folded.find("nan") != std::string::npos ? NAN : INFINITY;
    }
    char* end = nullptr;
    const double number = std::strtod(text.c_str(), &end);
    if (end == nullptr || *end != '\0') return std::nullopt;
    return number;
}

// int(text) for a stripped string: an optional sign and decimal digits.
std::optional<int> py_int(const std::string& text) {
    const unicode::Text value = unicode::strip(unicode::decode(text));
    std::size_t i = 0;
    bool negative = false;
    if (i < value.size() && (value[i] == U'+' || value[i] == U'-')) negative = value[i++] == U'-';
    if (i >= value.size()) return std::nullopt;
    long long number = 0;
    bool last_underscore = false;
    for (; i < value.size(); ++i) {
        if (value[i] == U'_' && !last_underscore && number >= 0 && i + 1 < value.size()) {
            last_underscore = true;
            continue;
        }
        const int digit = unicode::decimal_value(value[i]);
        if (digit < 0) return std::nullopt;
        number = number * 10 + digit;
        if (number > 1000000000LL) return std::nullopt;
        last_underscore = false;
    }
    return static_cast<int>(negative ? -number : number);
}

std::string norm_word(const std::string& word) {
    std::string out;
    for (char32_t cp : unicode::lower(unicode::decode(word))) {
        if (unicode::is_alnum(cp) || cp == U'\'') unicode::append_utf8(out, cp);
    }
    return out;
}

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

double elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace

// --- score ---------------------------------------------------------------------------------------------------------------

std::string ScoreNote::name() const { return midi ? midi_to_note(*midi) : "rest"; }

std::vector<int> Score::pitched() const {
    std::vector<int> out;
    for (const ScoreNote& note : notes) {
        if (note.midi) out.push_back(*note.midi);
    }
    return out;
}

Score Score::transposed(int semitones) const {
    if (semitones == 0) return *this;
    Score out = *this;
    for (ScoreNote& note : out.notes) {
        if (note.midi) note.midi = *note.midi + semitones;
    }
    return out;
}

Json Score::to_json() const {
    Json out{Json::Object{}};
    out.set("text", Json(text));
    out.set("bpm", Json(bpm));
    out.set("style", Json(style));
    out.set("mode", Json(mode));
    Json::Array note_items, word_items;
    for (const ScoreNote& note : notes) note_items.push_back(note_json(note));
    for (const ScoreWord& word : words) {
        Json item{Json::Object{}};
        item.set("word", Json(word.text));
        item.set("start", Json(round_to(word.start, 4)));
        item.set("end", Json(round_to(word.end, 4)));
        word_items.push_back(std::move(item));
    }
    out.set("notes", Json(std::move(note_items)));
    out.set("words", Json(std::move(word_items)));
    return out;
}

std::optional<int> note_to_midi(const std::string& name) {
    const std::string value = strip(name);
    const std::string folded = lower(value);
    if (folded == "rest" || folded == "-" || folded == "r" || folded.empty()) return std::nullopt;
    const std::string error = "Unknown note '" + name + "': use a pitch name with octave (C4, F#3, Bb3) or rest";
    const char letter = static_cast<char>(std::toupper(static_cast<unsigned char>(value[0])));
    static const std::map<char, int> index = {{'C', 0}, {'D', 2}, {'E', 4}, {'F', 5}, {'G', 7}, {'A', 9}, {'B', 11}};
    if (static_cast<unsigned char>(value[0]) >= 0x80 || !index.count(letter)) throw ScoreError(error);
    int semis = index.at(letter);
    unicode::Text rest = unicode::decode(value.substr(1));
    while (!rest.empty() && (rest[0] == U'#' || rest[0] == U'b' || rest[0] == U'♯' || rest[0] == U'♭')) {
        semis += rest[0] == U'#' || rest[0] == U'♯' ? 1 : -1;
        rest.erase(rest.begin());
    }
    const std::optional<int> octave = py_int(unicode::encode(rest));
    if (!octave) throw ScoreError(error);
    return 12 * (*octave + 1) + semis;
}

std::string midi_to_note(int midi) {
    return std::string(kNoteNames[floor_mod(midi, 12)]) + std::to_string(floor_div(midi, 12) - 1);
}

Score parse_score(const Json& data) {
    if (!data.is_object()) throw ScoreError("A score is a JSON object with text, bpm, style and notes");
    const std::optional<double> bpm = data.has("bpm") ? py_float(data.get("bpm")) : std::nullopt;
    if (!bpm) throw ScoreError("Score needs a numeric bpm");
    if (!std::isfinite(*bpm) || *bpm <= 0) throw ScoreError("Score bpm must be positive");
    const Json& style_value = data.get("style");
    const bool falsy = style_value.is_null() || (style_value.is_string() && style_value.str().empty()) ||
                       (style_value.type() == Json::Type::boolean && !style_value.boolean()) ||
                       (style_value.is_number() && style_value.num() == 0.0) || ((style_value.is_array() || style_value.is_object()) && style_value.size() == 0);
    const std::string style = lower(strip(falsy ? std::string("pop") : py_str(style_value)));
    if (style_mode(style).empty()) throw ScoreError("Unknown style " + quote(style) + "; choose one of ballad, pop, rap, sung");
    const Json& rows = data.get("notes");
    if (!rows.is_array() || rows.size() == 0) throw ScoreError("Score notes must be a non-empty list of [label, note, beats]");
    const double beat = 60.0 / *bpm;
    Score score;
    score.bpm = *bpm;
    score.style = style;
    score.mode = style_mode(style);
    std::vector<std::string> pieces;  // syllable spellings of the open word
    int open_part = 0;                // the last part number of the open word (0: no open word)
    double cursor = 0.0;
    for (std::size_t index = 0; index < rows.size(); ++index) {
        const Json& row = rows[index];
        const std::string number_text = std::to_string(index + 1);
        if (!row.is_array() || row.size() != 3) throw ScoreError("Note " + number_text + " must be [label, note, beats]; got " + row.dump());
        const std::string label = strip(py_str(row[0])), name = py_str(row[1]);
        const std::optional<double> beats = py_float(row[2]);
        if (!beats) throw ScoreError("Note " + number_text + " (" + quote(label) + ") has a non-numeric length " + row[2].dump());
        if (!std::isfinite(*beats) || *beats <= 0) throw ScoreError("Note " + number_text + " (" + quote(label) + ") must last a positive number of beats");
        const std::optional<int> midi = note_to_midi(name);
        const double start = cursor, end = cursor + *beats * beat;
        cursor = end;
        const bool rest = !midi || label == "-" || label.empty();
        score.notes.push_back(ScoreNote{rest ? std::string("-") : label, rest ? std::nullopt : midi, start, end});
        if (rest) {
            open_part = 0;
            pieces.clear();
            continue;
        }
        const std::size_t tilde = label.find('~');
        const std::string base = label.substr(0, tilde), part = tilde == std::string::npos ? std::string() : label.substr(tilde + 1);
        if (base.empty()) throw ScoreError("Note " + number_text + " has an empty word label");
        const unicode::Text part_text = unicode::decode(part);
        const bool digits = !part_text.empty() && std::all_of(part_text.begin(), part_text.end(), [](char32_t cp) { return unicode::decimal_value(cp) >= 0; });
        if (!part.empty() && !digits) throw ScoreError("Note " + number_text + ": " + quote(label) + " should be word~n with a part number");
        const int number = part.empty() ? 1 : py_int(part).value_or(0);
        if (number > 1) {
            if (number != open_part + 1) {
                throw ScoreError("Note " + number_text + ": " + quote(label) + " does not follow part " + std::to_string(number - 1) +
                                 " of its word (numbered parts must be consecutive, with no rest between them)");
            }
            pieces.push_back(base);
            const ScoreWord previous = score.words.back();
            const std::string text = base == pieces.front() ? previous.text : join(pieces, "");
            score.words.back() = ScoreWord{text, previous.start, end, previous.parts + 1};
            open_part = number;
            continue;
        }
        score.words.push_back(ScoreWord{base, start, end, 1});
        pieces = {base};
        open_part = 1;
    }
    if (score.words.empty()) throw ScoreError("Score has no sung notes");
    const Json& text_value = data.get("text");
    const bool text_falsy = text_value.is_null() || (text_value.is_string() && text_value.str().empty()) ||
                            (text_value.type() == Json::Type::boolean && !text_value.boolean()) || (text_value.is_number() && text_value.num() == 0.0);
    std::string text = text_falsy ? std::string() : strip(py_str(text_value));
    std::vector<std::string> word_texts;
    for (const ScoreWord& word : score.words) word_texts.push_back(word.text);
    if (text.empty()) text = join(word_texts, " ");
    std::vector<std::string> given_words;
    for (const unicode::Text& token : unicode::split_whitespace(unicode::decode(text))) given_words.push_back(unicode::encode(token));
    std::vector<std::string> expected, given;
    for (const std::string& word : word_texts) {
        const std::string n = norm_word(word);
        if (!n.empty()) expected.push_back(n);
    }
    for (const std::string& word : given_words) {
        const std::string n = norm_word(word);
        if (!n.empty()) given.push_back(n);
    }
    if (expected != given) {
        std::vector<std::string> first_given(given_words.begin(), given_words.begin() + static_cast<std::ptrdiff_t>(std::min<std::size_t>(8, given_words.size())));
        std::vector<std::string> first_words(word_texts.begin(), word_texts.begin() + static_cast<std::ptrdiff_t>(std::min<std::size_t>(8, word_texts.size())));
        throw ScoreError("Score text does not match the note labels: text has " + std::to_string(given_words.size()) + " words (" +
                         join(first_given, " ") + " ...), the notes spell " + std::to_string(word_texts.size()) + " (" + join(first_words, " ") + " ...)");
    }
    score.text = text;
    return score;
}

// --- bundle singing settings -------------------------------------------------------------------------------------------------

std::vector<std::string> singing_modes(const Engine& engine) {
    std::vector<std::string> listed;
    for (const std::string& mode : engine.singing().modes) {
        if (engine.delivery_modes().count(mode)) listed.push_back(mode);
    }
    if (!listed.empty()) return listed;
    for (const auto& [mode, id] : engine.delivery_modes()) {   // std::map: sorted, as Python's sorted()
        if (mode != "spoken") listed.push_back(mode);
    }
    return listed;
}

std::vector<std::string> singing_styles(const Engine& engine) {
    const std::vector<std::string> modes = singing_modes(engine);
    std::vector<std::string> out;
    for (const auto& [style, mode] : kStyleModes) {
        if (std::find(modes.begin(), modes.end(), mode) != modes.end()) out.push_back(style);
    }
    return out;
}

std::optional<double> voice_sung_center(const Engine& engine, const std::string& voice) {
    const auto& centers = engine.singing().voice_midi_center;
    auto found = centers.find(voice);
    if (found != centers.end()) return found->second;
    auto edges = engine.pitch_edges().find(voice);
    if (edges == engine.pitch_edges().end()) return std::nullopt;
    const double low = edges->second.first, high = edges->second.second;
    return (low + high) / 2 - engine.melody_center() + engine.singing().midi_center;
}

int register_shift(const Score& score, std::optional<double> center) {
    const std::vector<int> pitched = score.pitched();
    if (!center || pitched.empty()) return 0;
    long long sum = 0;
    for (int midi : pitched) sum += midi;
    const double mean = static_cast<double>(sum) / static_cast<double>(pitched.size());
    const double low = *center - kRegisterBelow, high = *center + kRegisterAbove;
    auto distance = [&](int shift) { return std::max({0.0, low - (mean + shift), (mean + shift) - high}); };
    int best = 0;
    bool first = true;
    double best_distance = 0.0;
    for (int shift : {-24, -12, 0, 12, 24}) {   // min by (distance rounded to 3 decimals, -shift): ties go up
        const double d = round_to(distance(shift), 3);
        if (first || d < best_distance || (d == best_distance && -shift < -best)) {
            best = shift;
            best_distance = d;
            first = false;
        }
    }
    return best;
}

// --- phrases -----------------------------------------------------------------------------------------------------------------

std::vector<Phrase> split_phrases(const Score& score, double phrase_rest, double max_seconds, double pad) {
    std::vector<ScoreNote> sung;
    for (const ScoreNote& note : score.notes) {
        if (note.midi) sung.push_back(note);
    }
    std::vector<std::vector<ScoreNote>> runs;
    std::vector<ScoreNote> current;
    for (const ScoreNote& note : sung) {
        if (!current.empty() && note.start - current.back().end >= phrase_rest) {
            runs.push_back(current);
            current.clear();
        }
        current.push_back(note);
    }
    if (!current.empty()) runs.push_back(current);
    std::set<double> word_starts;
    for (const ScoreWord& word : score.words) word_starts.insert(word.start);

    std::function<std::vector<std::vector<ScoreNote>>(const std::vector<ScoreNote>&)> cut = [&](const std::vector<ScoreNote>& run) {
        if (run.back().end - run.front().start <= max_seconds || run.size() < 2) return std::vector<std::vector<ScoreNote>>{run};
        // The longest internal rest (the last of equal ones), else the word boundary nearest the middle.
        double gap = -INFINITY;
        std::size_t index = 0;
        for (std::size_t i = 0; i + 1 < run.size(); ++i) {
            const double g = run[i + 1].start - run[i].end;
            if (g > gap || (g == gap && i > index)) {
                gap = g;
                index = i;
            }
        }
        if (gap <= 0.02) {
            const double middle = (run.front().start + run.back().end) / 2;
            std::vector<std::size_t> candidates;
            for (std::size_t k = 1; k < run.size(); ++k) {
                if (word_starts.count(run[k].start)) candidates.push_back(k);
            }
            if (candidates.empty()) candidates.push_back(run.size() / 2);
            std::size_t best = candidates.front();
            for (std::size_t k : candidates) {
                if (std::fabs(run[k].start - middle) < std::fabs(run[best].start - middle)) best = k;
            }
            index = best - 1;
        }
        std::vector<ScoreNote> head(run.begin(), run.begin() + static_cast<std::ptrdiff_t>(index + 1)), tail(run.begin() + static_cast<std::ptrdiff_t>(index + 1), run.end());
        std::vector<std::vector<ScoreNote>> out = cut(head), later = cut(tail);
        out.insert(out.end(), later.begin(), later.end());
        return out;
    };
    std::vector<std::vector<ScoreNote>> pieces;
    for (const auto& run : runs) {
        for (auto& piece : cut(run)) pieces.push_back(std::move(piece));
    }
    std::vector<std::vector<ScoreNote>> packed;
    current.clear();
    for (const auto& run : pieces) {
        if (!current.empty() && run.back().end - current.front().start > max_seconds) {
            packed.push_back(current);
            current.clear();
        }
        current.insert(current.end(), run.begin(), run.end());
    }
    if (!current.empty()) packed.push_back(current);

    std::vector<Phrase> phrases;
    for (const auto& group : packed) {
        const double t0 = group.front().start, t1 = group.back().end;
        Phrase phrase;
        std::vector<std::string> texts;
        for (const ScoreWord& word : score.words) {
            if (word.start >= t0 - 1e-6 && word.end <= t1 + 1e-6) {
                phrase.words.push_back(ScoreWord{word.text, word.start - t0 + pad, word.end - t0 + pad, word.parts});
                texts.push_back(word.text);
            }
        }
        for (const ScoreNote& note : group) phrase.notes.push_back(ScoreNote{note.label, note.midi, note.start - t0 + pad, note.end - t0 + pad});
        phrase.notes.push_back(ScoreNote{"-", std::nullopt, t1 - t0 + pad, t1 - t0 + 2 * pad});
        phrase.start = t0 - pad;
        phrase.end = t1 + pad;
        phrase.text = join(texts, " ");
        phrases.push_back(std::move(phrase));
    }
    return phrases;
}

// --- phones, durations and melody ----------------------------------------------------------------------------------------------

std::pair<std::vector<std::string>, std::vector<int>> word_phones(const std::vector<std::string>& words, Engine& engine, const std::string& language) {
    // The phrase is phonemized as one text so the G2P sees its context; when its words do not line up one-to-one with the
    // score's, each word is phonemized on its own, so the count always matches.
    const Phonemized result = engine.phonemize(join(words, " "), language);
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
        const Phonemized single = engine.phonemize(word, language);
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

SongPlan plan_song(const Score& input, Engine& engine, const std::string& voice, const std::string& language, const PlanSettings& settings) {
    if (!engine.supports_singing()) {
        throw std::invalid_argument("Singing needs a v4 bundle with delivery modes (controls.delivery_modes); this bundle has none");
    }
    const std::vector<std::string> allowed = singing_styles(engine);
    if (std::find(allowed.begin(), allowed.end(), input.style) == allowed.end()) {
        throw ScoreError("Style " + quote(input.style) + " (" + input.mode + ") is not a singing style of this bundle; choose one of " + join(allowed, ", "));
    }
    const int sample_rate = engine.sample_rate(), hop = engine.latent_hop();
    const double midi_center = engine.singing().midi_center, midi_scale = engine.singing().midi_scale;
    Score score = input.transposed(settings.transpose);
    const int shift = settings.use_register ? register_shift(score, voice_sung_center(engine, voice)) : 0;
    score = score.transposed(shift);
    const double longest = static_cast<double>((engine.max_frames() - 2) * hop) / sample_rate - 2 * settings.pad;
    const double max_seconds = std::min(settings.max_seconds.value_or(kMaxPhraseSeconds), longest);
    const double share = settings.vowel_share.value_or(vowel_share_of(score.mode));
    SongPlan plan;
    plan.voice = voice;
    plan.language = language;
    plan.mode = score.mode;
    plan.mode_id = engine.delivery_modes().at(score.mode);
    plan.register_shift = shift;
    plan.transpose = settings.transpose;
    const std::vector<Phrase> phrases = split_phrases(score, settings.phrase_rest, max_seconds, settings.pad);
    for (std::size_t index = 0; index < phrases.size(); ++index) {
        const Phrase& phrase = phrases[index];
        std::vector<std::string> words;
        for (const ScoreWord& word : phrase.words) words.push_back(word.text);
        auto [phones, starts] = word_phones(words, engine, language);
        const int64_t total = round_half_even(phrase.notes.back().end * sample_rate / hop) + 2;
        std::vector<int> durations = phone_durations(phones, starts, phrase.words, total, share, hop, sample_rate);
        int64_t frames = 0;
        for (int value : durations) frames += value;
        if (frames > engine.max_frames()) {
            throw OverlongError("Phrase " + std::to_string(index + 1) + " ('" + unicode::encode(unicode::decode(phrase.text).substr(0, 40)) + "') needs " +
                                std::to_string(frames) + " latent frames, but this bundle holds at most " + std::to_string(engine.max_frames()) +
                                "; add a rest inside it or shorten its notes");
        }
        PhrasePlan item;
        item.index = static_cast<int>(index);
        item.phrase = phrase;
        item.sentence.phones = std::move(phones);
        item.sentence.word_starts = std::move(starts);
        item.durations = std::move(durations);
        item.frames = frames;
        item.melody = frame_melody(phrase.notes, frames, hop, sample_rate, midi_center, midi_scale);
        plan.phrases.push_back(std::move(item));
    }
    plan.score = std::move(score);
    return plan;
}

// --- rendering ---------------------------------------------------------------------------------------------------------------

SingSettings sing_settings(const Engine& engine, const SingOptions& options) {
    const SingingSettings& singing = engine.singing();
    SingSettings out;
    if (options.steps < 0) throw std::invalid_argument("steps must be positive");
    out.steps = options.steps > 0 ? options.steps : (singing.steps > 0 ? singing.steps : 8);
    out.sampler = lower(options.sampler.empty() ? (singing.sampler.empty() ? std::string("heun") : singing.sampler) : options.sampler);
    if (out.sampler != "heun" && out.sampler != "euler") throw std::invalid_argument("Unsupported sampler '" + out.sampler + "'; choose heun or euler");
    const std::string preferred = singing.decoder.empty() ? std::string("vocos") : singing.decoder;
    out.decoder = options.decoder.empty() ? (engine.has_decoder(preferred) ? preferred : std::string("vocos")) : lower(options.decoder);
    if (!engine.has_decoder(out.decoder)) {
        throw std::invalid_argument("This bundle has no '" + out.decoder + "' decoder; it has " + join(engine.available_decoders(), ", "));
    }
    out.guidance = options.guidance.value_or(singing.guidance);
    if (!std::isfinite(out.guidance)) throw std::invalid_argument("guidance must be finite");
    return out;
}

Json sing_stream(Engine& engine, const Json& score_json, const SingOptions& options, const std::function<bool(const SingEvent&)>& emit,
                 const PhraseNoiseFn& phrase_noise) {
    if (!engine.supports_singing()) {
        throw std::invalid_argument("Singing needs a v4 bundle with delivery modes; this bundle is spoken only. Download a current model.");
    }
    const std::string voice = unicode::encode(unicode::strip(unicode::decode(options.voice)));
    if (voice.empty()) throw std::invalid_argument("A voice is required");
    const std::string language = engine.resolve_language(voice, options.language);
    const Score score = parse_score(score_json);
    const SongPlan plan = plan_song(score, engine, voice, language, options.plan);
    const SingSettings settings = sing_settings(engine, options);
    const auto started = std::chrono::steady_clock::now();
    const int sample_rate = engine.sample_rate();
    const int phrase_count = static_cast<int>(plan.phrases.size());
    auto send = [&](SingEvent event) {
        event.phrase_count = phrase_count;
        if (!emit(event)) throw CancelledError();
    };
    const int64_t song_samples = static_cast<int64_t>(std::ceil(plan.seconds() * sample_rate));
    Json plan_json = plan.to_json();
    plan_json.set("samples", Json(song_samples));
    send(SingEvent{SingEvent::kPlan, nullptr, 0, nullptr, plan_json});

    // No delivery request: sung clips were trained without intensity or pace, and the score sets the timing.
    DeliveryTensors delivery{};
    const std::size_t D = static_cast<std::size_t>(engine.latent_dim());
    std::random_device entropy;
    std::optional<double> first_audio_ms;
    Json::Array rendered;
    for (const PhrasePlan& item : plan.phrases) {
        Json started_json = item.to_json();
        send(SingEvent{SingEvent::kPhraseStarted, &item, 0, nullptr, started_json});
        const std::optional<uint64_t> seed = options.seed ? std::optional<uint64_t>(*options.seed + static_cast<uint64_t>(item.index)) : std::nullopt;
        const uint64_t draw_seed = seed ? *seed : (static_cast<uint64_t>(entropy()) << 32) ^ entropy();
        NoiseFn noise = [&](int64_t frames, float* pitch, float* flow) {
            const std::size_t T = static_cast<std::size_t>(frames);
            std::fill_n(pitch, 2 * T, 0.0f);   // unused: the score gives the melody
            if (phrase_noise) {
                std::vector<float> all((2 + D) * T);
                phrase_noise(item.index, seed.value_or(0), seed.has_value(), frames, all.data());
                std::copy_n(all.begin() + static_cast<std::ptrdiff_t>(2 * T), D * T, flow);
            } else {
                gaussian_noise(draw_seed, static_cast<int64_t>(D * T), 1.0f, flow);
            }
        };
        const SungConditioning sung{&item.melody, plan.mode_id, settings.guidance};
        SentenceRequest request;
        request.voice = voice;
        request.language = language;
        request.delivery = delivery;
        request.steps = settings.steps;
        request.sampler = settings.sampler;
        request.noise = noise;
        request.durations = &item.durations;
        request.sung = &sung;
        SentenceResult result = engine.synthesize_sentence(item.sentence, request);
        DecodeRequest decode;
        decode.voice = voice;
        decode.language = language;
        decode.decoder = settings.decoder;
        std::vector<float> audio = engine.decode(result.latents, result.frames, decode);
        int64_t offset = round_half_even(item.phrase.start * sample_rate);
        if (offset < 0) {
            audio.erase(audio.begin(), audio.begin() + static_cast<std::ptrdiff_t>(std::min<int64_t>(-offset, static_cast<int64_t>(audio.size()))));
            offset = 0;
        }
        if (!first_audio_ms) first_audio_ms = elapsed_ms(started);
        Json phrase_json = item.to_json();
        phrase_json.set("bucket", Json(result.bucket));
        phrase_json.set("seconds", Json(round_to(static_cast<double>(audio.size()) / sample_rate, 3)));
        phrase_json.set("seed", seed ? Json(static_cast<int64_t>(*seed)) : Json());
        rendered.push_back(phrase_json);
        Json metadata = phrase_json;
        metadata.set("offset_samples", Json(offset));
        metadata.set("song_samples", Json(song_samples));
        metadata.set("decoder", Json(settings.decoder));
        metadata.set("first_audio_ms", Json(round_to(*first_audio_ms, 1)));
        metadata.set("elapsed_ms", Json(round_to(elapsed_ms(started), 1)));
        send(SingEvent{SingEvent::kAudio, &item, 0, &audio, metadata});
    }
    Json summary{Json::Object{}};
    summary.set("phrases", Json(phrase_count));
    summary.set("first_audio_ms", first_audio_ms ? Json(round_to(*first_audio_ms, 1)) : Json());
    summary.set("elapsed_ms", Json(round_to(elapsed_ms(started), 1)));
    summary.set("samples", Json(song_samples));
    summary.set("seconds", Json(round_to(plan.seconds(), 3)));
    send(SingEvent{SingEvent::kDone, nullptr, 0, nullptr, summary});

    Json metadata = plan.to_json();
    metadata.set("phrases", Json(std::move(rendered)));
    metadata.set("steps", Json(settings.steps));
    metadata.set("sampler", Json(settings.sampler));
    metadata.set("decoder", Json(settings.decoder));
    metadata.set("guidance", Json(settings.guidance));
    metadata.set("seed", options.seed ? Json(static_cast<int64_t>(*options.seed)) : Json());
    metadata.set("graph_contract", Json(engine.graph_contract()));
    metadata.set("samples", Json(song_samples));
    metadata.set("first_audio_ms", summary.get("first_audio_ms"));
    metadata.set("elapsed_ms", summary.get("elapsed_ms"));
    return metadata;
}

std::vector<float> sing(Engine& engine, const Json& score, const SingOptions& options, Json* metadata, const PhraseNoiseFn& phrase_noise) {
    std::vector<float> song;
    const int sample_rate = engine.sample_rate();
    Json result = sing_stream(engine, score, options, [&](const SingEvent& event) {
        if (event.type == SingEvent::kPlan) song.assign(static_cast<std::size_t>(event.metadata.get("samples").integer() + sample_rate / 10), 0.0f);
        if (event.type == SingEvent::kAudio) {
            const std::size_t offset = static_cast<std::size_t>(event.metadata.get("offset_samples").integer());
            const std::size_t count = offset < song.size() ? std::min(event.audio->size(), song.size() - offset) : 0;
            for (std::size_t i = 0; i < count; ++i) song[offset + i] += (*event.audio)[i];
        }
        return true;
    }, phrase_noise);
    song.resize(static_cast<std::size_t>(result.get("samples").integer()));
    for (float& sample : song) sample = std::clamp(sample, -1.0f, 1.0f);
    if (metadata) *metadata = std::move(result);
    return song;
}

Json singing_info(const Engine& engine) {
    Json out{Json::Object{}};
    out.set("supported", Json(engine.supports_singing()));
    out.set("graph_contract", Json(engine.graph_contract()));
    out.set("styles", Json::array_of(engine.supports_singing() ? singing_styles(engine) : std::vector<std::string>{}));
    out.set("decoders", Json::array_of(engine.available_decoders()));
    if (engine.supports_singing()) {
        const SingingSettings& singing = engine.singing();
        out.set("default_decoder", Json(engine.has_decoder(singing.decoder) ? singing.decoder : std::string("vocos")));
        out.set("sampler", Json(singing.sampler));
        out.set("steps", Json(singing.steps));
        out.set("guidance", Json(singing.guidance));
        Json centers{Json::Object{}};
        for (const Voice& voice : engine.voices()) {
            const std::optional<double> center = voice_sung_center(engine, voice.id);
            if (center) centers.set(voice.id, Json(*center));
        }
        out.set("voice_midi_center", centers);
    }
    return out;
}

}  // namespace scyllasband
