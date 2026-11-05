#include "scyllasband_score.h"

#include "scyllasband_numeric.h"
#include "scyllasband_unicode.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <map>
#include <set>

namespace scyllasband {

namespace {

const char* const kNoteNames[12] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};

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

}  // namespace

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

const std::vector<std::pair<std::string, std::string>>& style_modes() {
    static const std::vector<std::pair<std::string, std::string>> modes = {{"pop", "sung"}, {"sung", "sung"}, {"ballad", "ballad"}, {"rap", "rap"}};
    return modes;
}

std::string style_mode(const std::string& style) {
    for (const auto& [name, mode] : style_modes()) {
        if (name == style) return mode;
    }
    return {};
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

Score parse_score_text(const std::string& text) {
    Json data;
    try {
        data = Json::parse(text);
    } catch (const std::exception& error) {
        throw ScoreError(std::string("The score is not valid JSON: ") + error.what());
    }
    return parse_score(data);
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
        const int at = static_cast<int>(index);
        const std::string number_text = std::to_string(index + 1);
        if (!row.is_array() || row.size() != 3) throw ScoreError("Note " + number_text + " must be [label, note, beats]; got " + row.dump(), at);
        const std::string label = strip(py_str(row[0])), name = py_str(row[1]);
        const std::optional<double> beats = py_float(row[2]);
        if (!beats) throw ScoreError("Note " + number_text + " (" + quote(label) + ") has a non-numeric length " + row[2].dump(), at);
        if (!std::isfinite(*beats) || *beats <= 0) throw ScoreError("Note " + number_text + " (" + quote(label) + ") must last a positive number of beats", at);
        std::optional<int> midi;
        try {
            midi = note_to_midi(name);
        } catch (const ScoreError& error) {
            throw ScoreError(error.what(), at);
        }
        const double start = cursor, end = cursor + *beats * beat;
        cursor = end;
        const bool rest = !midi || label == "-" || label.empty();
        score.notes.push_back(ScoreNote{rest ? std::string("-") : label, rest ? std::nullopt : midi, start, end, *beats, -1, 0});
        if (rest) {
            open_part = 0;
            pieces.clear();
            continue;
        }
        const std::size_t tilde = label.find('~');
        const std::string base = label.substr(0, tilde), part = tilde == std::string::npos ? std::string() : label.substr(tilde + 1);
        if (base.empty()) throw ScoreError("Note " + number_text + " has an empty word label", at);
        const unicode::Text part_text = unicode::decode(part);
        const bool digits = !part_text.empty() && std::all_of(part_text.begin(), part_text.end(), [](char32_t cp) { return unicode::decimal_value(cp) >= 0; });
        if (!part.empty() && !digits) throw ScoreError("Note " + number_text + ": " + quote(label) + " should be word~n with a part number", at);
        const int number = part.empty() ? 1 : py_int(part).value_or(0);
        if (number > 1) {
            if (number != open_part + 1) {
                throw ScoreError("Note " + number_text + ": " + quote(label) + " does not follow part " + std::to_string(number - 1) +
                                     " of its word (numbered parts must be consecutive, with no rest between them)",
                                 at);
            }
            pieces.push_back(base);
            const ScoreWord previous = score.words.back();
            const std::string text = base == pieces.front() ? previous.text : join(pieces, "");
            score.words.back() = ScoreWord{text, previous.start, end, previous.parts + 1};
            open_part = number;
            score.notes.back().word = static_cast<int>(score.words.size()) - 1;
            score.notes.back().part = number;
            continue;
        }
        score.notes.back().word = static_cast<int>(score.words.size());
        score.notes.back().part = 1;
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
        for (const ScoreNote& note : group) {
            ScoreNote shifted = note;
            shifted.start = note.start - t0 + pad;
            shifted.end = note.end - t0 + pad;
            phrase.notes.push_back(std::move(shifted));
        }
        phrase.notes.push_back(ScoreNote{"-", std::nullopt, t1 - t0 + pad, t1 - t0 + 2 * pad, 0.0, -1, 0});
        phrase.start = t0 - pad;
        phrase.end = t1 + pad;
        phrase.text = join(texts, " ");
        phrases.push_back(std::move(phrase));
    }
    return phrases;
}

std::vector<std::pair<int, int>> score_lines(const Score& score, double phrase_rest) {
    std::vector<std::pair<int, int>> lines;
    int first = 0;
    std::optional<double> last_sung_end;  // the end of the current line's last sung note
    for (std::size_t i = 0; i < score.notes.size(); ++i) {
        const ScoreNote& note = score.notes[i];
        if (!note.midi) continue;
        if (last_sung_end && note.start - *last_sung_end >= phrase_rest) {
            lines.emplace_back(first, static_cast<int>(i) - 1);   // the rests before this note close the previous line
            first = static_cast<int>(i);
        }
        last_sung_end = note.end;
    }
    if (!score.notes.empty()) lines.emplace_back(first, static_cast<int>(score.notes.size()) - 1);
    return lines;
}

Json score_summary(const Score& input, int transpose) {
    const Score score = input.transposed(transpose);
    Json out{Json::Object{}};
    out.set("text", Json(score.text));
    out.set("bpm", Json(score.bpm));
    out.set("style", Json(score.style));
    out.set("mode", Json(score.mode));
    out.set("seconds", Json(round_to(score.seconds(), 4)));
    out.set("transpose", Json(transpose));
    Json::Array notes, words, lines;
    for (std::size_t i = 0; i < score.notes.size(); ++i) {
        const ScoreNote& note = score.notes[i];
        Json item{Json::Object{}};
        item.set("index", Json(static_cast<int64_t>(i)));
        item.set("label", Json(note.label));
        item.set("word_index", note.word >= 0 ? Json(note.word) : Json());
        item.set("part", Json(note.part));
        item.set("note", Json(note.name()));
        item.set("midi", note.midi ? Json(*note.midi) : Json());
        item.set("beats", Json(note.beats));
        item.set("start", Json(round_to(note.start, 4)));
        item.set("end", Json(round_to(note.end, 4)));
        notes.push_back(std::move(item));
    }
    for (const ScoreWord& word : score.words) {
        Json item{Json::Object{}};
        item.set("word", Json(word.text));
        item.set("start", Json(round_to(word.start, 4)));
        item.set("end", Json(round_to(word.end, 4)));
        item.set("parts", Json(word.parts));
        words.push_back(std::move(item));
    }
    for (const auto& [first, last] : score_lines(score)) {
        Json item{Json::Object{}};
        item.set("first", Json(first));
        item.set("last", Json(last));
        lines.push_back(std::move(item));
    }
    out.set("notes", Json(std::move(notes)));
    out.set("words", Json(std::move(words)));
    const std::vector<int> pitched = score.pitched();
    Json range{Json::Object{}};
    if (!pitched.empty()) {
        const int low = *std::min_element(pitched.begin(), pitched.end()), high = *std::max_element(pitched.begin(), pitched.end());
        long long sum = 0;
        for (int midi : pitched) sum += midi;
        range.set("low", Json(low));
        range.set("high", Json(high));
        range.set("mean", Json(round_to(static_cast<double>(sum) / static_cast<double>(pitched.size()), 3)));
        range.set("low_note", Json(midi_to_note(low)));
        range.set("high_note", Json(midi_to_note(high)));
    }
    out.set("range", range);
    out.set("lines", Json(std::move(lines)));
    return out;
}

}  // namespace scyllasband
