#include "scyllasband_text.h"

#include "scyllasband_json.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>

namespace scyllasband {

const char* const kSpokenTextNormalizerVersion = "scyllasband_spoken_text_v2_2026_08_05";
const char* const kSpokenTextNormalizerSha256 = "d8fd1b2415d0d100d1c70237a4469117ee6b73d0e4da703e4654c83192779cc4";

namespace {

using namespace unicode;
constexpr std::size_t npos = std::u32string::npos;

// ---------------------------------------------------------------------------------------------------------------
// Regex-substitution scaffolding. Every pattern is matched by hand at a start position against the input string,
// as re.sub does: left to right, lookarounds see the unmodified input, scanning resumes after each match.

struct Match {
    std::size_t start = 0, end = 0;
    std::vector<std::pair<std::size_t, std::size_t>> groups;  // {npos, npos} when a group did not participate
    Text group(const Text& s, std::size_t index) const {
        const auto& g = groups.at(index);
        return g.first == npos ? Text() : s.substr(g.first, g.second - g.first);
    }
    bool matched(std::size_t index) const { return groups.at(index).first != npos; }
};

using Matcher = std::function<std::optional<Match>(const Text&, std::size_t)>;

Text substitute(const Text& s, const Matcher& matcher, const std::function<Text(const Text&, const Match&)>& replace) {
    Text out;
    std::size_t last = 0, pos = 0;
    while (pos <= s.size()) {
        std::optional<Match> m = pos < s.size() ? matcher(s, pos) : std::nullopt;
        if (m && m->end > m->start) {
            out.append(s, last, m->start - last);
            out += replace(s, *m);
            last = pos = m->end;
        } else {
            ++pos;
        }
    }
    out.append(s, last, npos);
    return out;
}

Text replace_char(const Text& value, char32_t from, const Text& to) {
    Text out;
    for (char32_t cp : value) {
        if (cp == from) out += to;
        else out.push_back(cp);
    }
    return out;
}

std::size_t run_end(const Text& s, std::size_t pos, const std::function<bool(char32_t)>& predicate) {
    while (pos < s.size() && predicate(s[pos])) ++pos;
    return pos;
}

char32_t at(const Text& s, std::size_t pos) { return pos < s.size() ? s[pos] : 0; }

bool in_set(char32_t cp, std::u32string_view set) { return cp != 0 && set.find(cp) != std::u32string_view::npos; }

// [A-Za-zÀ-ÖØ-öø-ÿÑñ]
bool latin_letter(char32_t cp) {
    return is_ascii_letter(cp) || (cp >= 0xC0 && cp <= 0xD6) || (cp >= 0xD8 && cp <= 0xF6) || (cp >= 0xF8 && cp <= 0xFF);
}
// [A-Za-zÑñ]
bool initial_letter(char32_t cp) { return is_ascii_letter(cp) || cp == 0xD1 || cp == 0xF1; }
// [A-ZÀ-ÖØ-Þ]
bool upper_latin(char32_t cp) { return (cp >= 'A' && cp <= 'Z') || (cp >= 0xC0 && cp <= 0xD6) || (cp >= 0xD8 && cp <= 0xDE); }
// [A-Za-z0-9] under re.IGNORECASE
bool ascii_alnum_ci(char32_t cp) {
    return is_ascii_letter(cp) || is_ascii_digit(cp) || cp == 0x130 || cp == 0x131 || cp == 0x17F || cp == 0x212A;
}
bool ci_literal(const Text& s, std::size_t pos, std::string_view literal) {
    if (pos + literal.size() > s.size()) return false;
    for (std::size_t i = 0; i < literal.size(); ++i) {
        if (!ascii_ci_equal(s[pos + i], literal[i])) return false;
    }
    return true;
}

// g2p_phrases.mark_tight_dashes: (?<=[^\W\d_])(?:[\u2012-\u2015\u2212\ufe58\ufe63\uff0d]|-{2,})(?=[^\W\d_]) -> " \u2e3a "
// (a dash typed with a letter directly on each side, "a\u2014dream": an interruption, always paused).
Text mark_tight_dashes(const Text& value) {
    auto letter = [](char32_t c) { return c != 0 && c != '_' && is_word(c) && !is_decimal(c); };
    auto dash = [](char32_t c) { return (c >= 0x2012 && c <= 0x2015) || c == 0x2212 || c == 0xFE58 || c == 0xFE63 || c == 0xFF0D; };
    Text out;
    std::size_t i = 0;
    while (i < value.size()) {
        std::size_t end = i;
        if (dash(value[i])) end = i + 1;
        else if (value[i] == '-') end = run_end(value, i, [](char32_t c) { return c == '-'; });
        const bool run = end > i && (dash(value[i]) || end - i >= 2);
        if (run && i > 0 && letter(value[i - 1]) && end < value.size() && letter(value[end])) {
            out += U" \u2e3a ";
            i = end;
        } else if (end > i) {
            out.append(value, i, end - i);
            i = end;
        } else {
            out.push_back(value[i++]);
        }
    }
    return out;
}

// [A-Z]
bool ascii_upper(char32_t cp) { return cp >= 'A' && cp <= 'Z'; }
// [a-zß-öø-ÿ'’]
bool name_tail(char32_t cp) {
    return (cp >= 'a' && cp <= 'z') || (cp >= 0xDF && cp <= 0xF6) || (cp >= 0xF8 && cp <= 0xFF) || cp == '\'' || cp == 0x2019;
}
// [^\W\d_]
bool letter_char(char32_t cp) { return cp != 0 && is_alnum(cp) && !is_decimal(cp); }
// (?<![A-Za-zÀ-ÖØ-öø-ÿ.'’]) fails at p
bool after_name_char(const Text& s, std::size_t p) { return p > 0 && (latin_letter(s[p - 1]) || in_set(s[p - 1], U".'’")); }
// (?<![\w.'’]) fails at p
bool after_word_char(const Text& s, std::size_t p) { return p > 0 && (is_word(s[p - 1]) || in_set(s[p - 1], U".'’")); }
// (?<![\w.,]) fails at p
bool after_number_char(const Text& s, std::size_t p) { return p > 0 && (is_word(s[p - 1]) || in_set(s[p - 1], U".,")); }

// _SENTENCE_FOLLOWS_RE.match(s, pos): \s*$|\s+["'“‘(\[]?[A-ZÀ-ÖØ-ÞÑ]
bool sentence_follows(const Text& s, std::size_t pos) {
    std::size_t j = run_end(s, pos, is_space);
    if (j == s.size()) return true;
    if (j == pos) return false;
    return upper_latin(s[j]) || (in_set(s[j], U"\"'“‘([") && upper_latin(at(s, j + 1)));
}

// s[:end].split()
std::vector<Text> words_before(const Text& s, std::size_t end) { return split_whitespace(std::u32string_view(s).substr(0, end)); }

// str.strip(chars)
Text strip_chars(const Text& value, std::u32string_view chars) {
    std::size_t start = 0, end = value.size();
    while (start < end && in_set(value[start], chars)) ++start;
    while (end > start && in_set(value[end - 1], chars)) --end;
    return value.substr(start, end - start);
}

Text dashes_to_pause(const Text& value) {
    Text out;
    for (char32_t cp : value) {
        if (cp == 0x2010 || cp == 0x2011) out.push_back('-');
        else if (cp == 0x2012 || cp == 0x2013 || cp == 0x2014 || cp == 0x2015 || cp == 0x2212 || cp == 0xFE58 || cp == 0xFE63 || cp == 0xFF0D) out += U" — ";
        else out.push_back(cp);
    }
    return out;
}

// re.sub(r"-{2,}", " — ", v) then re.sub(r"(?:(?<=\s)-+|-+(?=\s))", " — ", v)
Text hyphen_runs_to_pause(const Text& value) {
    Text step = substitute(value, [](const Text& s, std::size_t p) -> std::optional<Match> {
        if (s[p] != '-') return std::nullopt;
        std::size_t e = run_end(s, p, [](char32_t c) { return c == '-'; });
        if (e - p < 2) return std::nullopt;
        return Match{p, e, {}};
    }, [](const Text&, const Match&) { return Text(U" — "); });
    return substitute(step, [](const Text& s, std::size_t p) -> std::optional<Match> {
        if (s[p] != '-') return std::nullopt;
        std::size_t e = run_end(s, p, [](char32_t c) { return c == '-'; });
        if ((p > 0 && is_space(s[p - 1])) || (e < s.size() && is_space(s[e]))) return Match{p, e, {}};
        return std::nullopt;
    }, [](const Text&, const Match&) { return Text(U" — "); });
}

// re.sub(r"\.{2,}", "...", v) then re.sub(r"([!?])[!?]+", r"\1", v)
Text fold_terminal_runs(const Text& value) {
    Text step = substitute(value, [](const Text& s, std::size_t p) -> std::optional<Match> {
        if (s[p] != '.') return std::nullopt;
        std::size_t e = run_end(s, p, [](char32_t c) { return c == '.'; });
        if (e - p < 2) return std::nullopt;
        return Match{p, e, {}};
    }, [](const Text&, const Match&) { return Text(U"..."); });
    return substitute(step, [](const Text& s, std::size_t p) -> std::optional<Match> {
        if (s[p] != '!' && s[p] != '?') return std::nullopt;
        std::size_t e = run_end(s, p + 1, [](char32_t c) { return c == '!' || c == '?'; });
        if (e - p < 2) return std::nullopt;
        return Match{p, e, {{p, p + 1}}};
    }, [](const Text& s, const Match& m) { return m.group(s, 0); });
}

// ---------------------------------------------------------------------------------------------------------------
// Number words (the text_normalizer fallbacks), on decimal digit strings so any length works like Python ints.

struct Num {
    std::string d = "0";  // no leading zeros

    static Num of(const std::string& digits) {
        Num n;
        std::size_t first = digits.find_first_not_of('0');
        n.d = first == std::string::npos ? "0" : digits.substr(first);
        return n;
    }
    static Num of(int64_t value) { return of(std::to_string(value)); }
    int cmp(int64_t value) const {
        std::string other = std::to_string(value);
        if (d.size() != other.size()) return d.size() < other.size() ? -1 : 1;
        return d < other ? -1 : (d == other ? 0 : 1);
    }
    bool lt(int64_t value) const { return cmp(value) < 0; }
    bool eq(int64_t value) const { return cmp(value) == 0; }
    int64_t small() const { return std::stoll(d.size() > 18 ? d.substr(d.size() - 18) : d); }
    Num div_pow10(std::size_t k) const { return d.size() <= k ? Num::of(std::string("0")) : Num::of(d.substr(0, d.size() - k)); }
    Num mod_pow10(std::size_t k) const { return d.size() <= k ? *this : Num::of(d.substr(d.size() - k)); }
    bool zero() const { return d == "0"; }
};

std::string digits_of(const Text& text) {  // "".join(ch for ch in text if ch.isdigit()), as ASCII
    std::string out;
    for (char32_t cp : text) {
        int value = decimal_value(cp);
        if (value >= 0) out.push_back(static_cast<char>('0' + value));
    }
    return out;
}

const char* const kEnUnder20[] = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                                  "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"};
const char* const kEnTens[] = {"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};
const char* const kEnOrdinals[] = {"", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
                                   "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
                                   "eighteenth", "nineteenth", "twentieth", "twenty first", "twenty second", "twenty third",
                                   "twenty fourth", "twenty fifth", "twenty sixth", "twenty seventh", "twenty eighth", "twenty ninth",
                                   "thirtieth", "thirty first"};
const char* const kEsUnder30[] = {"cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez", "once",
                                  "doce", "trece", "catorce", "quince", "dieciséis", "diecisiete", "dieciocho", "diecinueve", "veinte",
                                  "veintiuno", "veintidós", "veintitrés", "veinticuatro", "veinticinco", "veintiséis", "veintisiete",
                                  "veintiocho", "veintinueve"};
const char* const kEsTens[] = {"", "", "", "treinta", "cuarenta", "cincuenta", "sesenta", "setenta", "ochenta", "noventa"};
const char* const kEsHundreds[] = {"", "cien", "doscientos", "trescientos", "cuatrocientos", "quinientos", "seiscientos", "setecientos",
                                   "ochocientos", "novecientos"};
const char* const kItUnder20[] = {"zero", "uno", "due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove", "dieci", "undici",
                                  "dodici", "tredici", "quattordici", "quindici", "sedici", "diciassette", "diciotto", "diciannove"};
const char* const kItTens[] = {"", "", "venti", "trenta", "quaranta", "cinquanta", "sessanta", "settanta", "ottanta", "novanta"};
const char* const kFrUnder20[] = {"zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix", "onze",
                                  "douze", "treize", "quatorze", "quinze", "seize", "dix-sept", "dix-huit", "dix-neuf"};
const char* const kFrTens[] = {"", "", "vingt", "trente", "quarante", "cinquante", "soixante"};
const char* const kDeUnder20[] = {"null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun", "zehn", "elf",
                                  "zwölf", "dreizehn", "vierzehn", "fünfzehn", "sechzehn", "siebzehn", "achtzehn", "neunzehn"};
const char* const kDeTens[] = {"", "", "zwanzig", "dreißig", "vierzig", "fünfzig", "sechzig", "siebzig", "achtzig", "neunzig"};
const char* const kViDigits[] = {"không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"};

struct Scale {
    std::size_t zeros;
    const char* singular;
    const char* plural;
};

std::string english_number(const Num& n) {
    if (n.lt(20)) return kEnUnder20[n.small()];
    if (n.lt(100)) {
        int64_t v = n.small(), rest = v % 10;
        return rest == 0 ? kEnTens[v / 10] : std::string(kEnTens[v / 10]) + " " + english_number(Num::of(rest));
    }
    if (n.lt(1000)) {
        Num rest = n.mod_pow10(2);
        std::string head = english_number(n.div_pow10(2)) + " hundred";
        return rest.zero() ? head : head + " " + english_number(rest);
    }
    for (const Scale& scale : {Scale{9, "billion", ""}, Scale{6, "million", ""}, Scale{3, "thousand", ""}}) {
        if (n.d.size() > scale.zeros) {
            Num rest = n.mod_pow10(scale.zeros);
            std::string head = english_number(n.div_pow10(scale.zeros)) + " " + scale.singular;
            return rest.zero() ? head : head + " " + english_number(rest);
        }
    }
    return n.d;
}

std::string spanish_number(const Num& n) {
    if (n.lt(30)) return kEsUnder30[n.small()];
    if (n.lt(100)) {
        int64_t v = n.small(), rest = v % 10;
        return rest == 0 ? kEsTens[v / 10] : std::string(kEsTens[v / 10]) + " y " + spanish_number(Num::of(rest));
    }
    if (n.lt(1000)) {
        int64_t v = n.small();
        if (v % 100 == 0) return kEsHundreds[v / 100];
        if (v < 200) return "ciento " + spanish_number(Num::of(v % 100));
        return std::string(kEsHundreds[v / 100]) + " " + spanish_number(Num::of(v % 100));
    }
    if (n.lt(1000000)) {
        Num rest = n.mod_pow10(3), count = n.div_pow10(3);
        std::string head = count.eq(1) ? "mil" : spanish_number(count) + " mil";
        return rest.zero() ? head : head + " " + spanish_number(rest);
    }
    Num rest = n.mod_pow10(6), count = n.div_pow10(6);
    std::string head = count.eq(1) ? "un millón" : spanish_number(count) + " millones";
    return rest.zero() ? head : head + " " + spanish_number(rest);
}

// Italian writes numbers below a million as one word ("trentacinque", "duemilacinquecento").
std::string italian_words(const Num& n) {
    if (n.lt(20)) return kItUnder20[n.small()];
    if (n.lt(100)) {
        int64_t v = n.small(), rest = v % 10;
        std::string tens = kItTens[v / 10];
        if (rest == 0) return tens;
        if (rest == 1 || rest == 8) tens.pop_back();  // "ventuno", "trentotto": the tens lose their vowel
        return tens + kItUnder20[rest];
    }
    if (n.lt(1000)) {
        Num rest = n.mod_pow10(2), count = n.div_pow10(2);
        std::string head = count.eq(1) ? std::string("cento") : std::string(kItUnder20[count.small()]) + "cento";
        if (rest.zero()) return head;
        std::string tail = italian_words(rest);
        if (tail.rfind("ott", 0) == 0) head.pop_back();  // "centotto", "duecentottanta"
        return head + tail;
    }
    if (n.lt(1000000)) {
        Num rest = n.mod_pow10(3), count = n.div_pow10(3);
        std::string head = count.eq(1) ? std::string("mille") : italian_words(count) + "mila";
        return rest.zero() ? head : head + italian_words(rest);
    }
    if (!n.lt(1000000000)) {
        Num rest = n.mod_pow10(9), count = n.div_pow10(9);
        std::string head = count.eq(1) ? std::string("un miliardo") : italian_words(count) + " miliardi";
        return rest.zero() ? head : head + " " + italian_words(rest);
    }
    Num rest = n.mod_pow10(6), count = n.div_pow10(6);
    std::string head = count.eq(1) ? std::string("un milione") : italian_words(count) + " milioni";
    return rest.zero() ? head : head + " " + italian_words(rest);
}

// A final "tre" takes an accent ("ventitré"), unless it stands alone.
std::string italian_number(const Num& n) {
    std::string words = italian_words(n);
    if (words.size() > 3 && words.compare(words.size() - 3, 3, "tre") == 0 && (words.size() < 4 || words[words.size() - 4] != ' ')) {
        return words.substr(0, words.size() - 3) + "tré";
    }
    return words;
}

std::string french_number(const Num& n) {
    if (n.lt(20)) return kFrUnder20[n.small()];
    if (n.lt(70)) {
        int64_t v = n.small(), rest = v % 10;
        if (rest == 0) return kFrTens[v / 10];
        return std::string(kFrTens[v / 10]) + (rest == 1 ? " et " : "-") + french_number(Num::of(rest));
    }
    if (n.lt(80)) {
        int64_t rest = n.small() - 60;
        return std::string("soixante") + (rest == 11 ? " et " : "-") + french_number(Num::of(rest));
    }
    if (n.lt(100)) {
        int64_t rest = n.small() - 80;
        return rest == 0 ? "quatre-vingts" : "quatre-vingt-" + french_number(Num::of(rest));
    }
    if (n.lt(1000)) {
        Num count = n.div_pow10(2), rest = n.mod_pow10(2);
        std::string head = count.eq(1) ? "cent" : french_number(count) + " cent";
        if (rest.zero()) return count.eq(1) ? head : head + "s";
        return head + " " + french_number(rest);
    }
    for (const Scale& scale : {Scale{9, "milliard", "milliards"}, Scale{6, "million", "millions"}, Scale{3, "mille", "mille"}}) {
        if (n.d.size() > scale.zeros) {
            Num count = n.div_pow10(scale.zeros), rest = n.mod_pow10(scale.zeros);
            std::string count_words = french_number(count);
            auto plural_s = [&](const char* tail) {
                const std::size_t n = std::char_traits<char>::length(tail);
                return count_words.size() >= n && count_words.compare(count_words.size() - n, n, tail) == 0;
            };
            if (scale.zeros == 3 && (plural_s("cents") || plural_s("vingts"))) {
                count_words.pop_back();  // no plural s before "mille": "deux cent mille", "quatre-vingt mille"
            }
            std::string head = scale.zeros == 3 && count.eq(1)
                ? std::string(scale.singular)
                : count_words + " " + (count.eq(1) ? scale.singular : scale.plural);
            return rest.zero() ? head : head + " " + french_number(rest);
        }
    }
    return n.d;
}

std::string german_number(const Num& n) {
    if (n.lt(20)) return kDeUnder20[n.small()];
    if (n.lt(100)) {
        int64_t v = n.small(), rest = v % 10;
        if (rest == 0) return kDeTens[v / 10];
        return (rest == 1 ? std::string("ein") : german_number(Num::of(rest))) + "und" + kDeTens[v / 10];
    }
    if (n.lt(1000)) {
        Num count = n.div_pow10(2), rest = n.mod_pow10(2);
        std::string head = count.eq(1) ? "einhundert" : german_number(count) + "hundert";
        return rest.zero() ? head : head + german_number(rest);
    }
    if (n.lt(1000000)) {
        Num count = n.div_pow10(3), rest = n.mod_pow10(3);
        std::string head = count.eq(1) ? "eintausend" : german_number(count) + "tausend";
        return rest.zero() ? head : head + german_number(rest);
    }
    for (const Scale& scale : {Scale{9, "Milliarde", "Milliarden"}, Scale{6, "Million", "Millionen"}}) {
        if (n.d.size() > scale.zeros) {
            Num count = n.div_pow10(scale.zeros), rest = n.mod_pow10(scale.zeros);
            std::string head = count.eq(1) ? std::string("eine ") + scale.singular : german_number(count) + " " + scale.plural;
            return rest.zero() ? head : head + " " + german_number(rest);
        }
    }
    return n.d;
}

std::string vietnamese_below_thousand(int64_t n, bool force_hundreds = false) {
    int64_t hundreds = n / 100, rest = n % 100;
    std::vector<std::string> words;
    if (hundreds || force_hundreds) {
        words.push_back(kViDigits[hundreds]);
        words.push_back("trăm");
        if (rest > 0 && rest < 10) words.push_back("linh");
    }
    if (rest >= 10) {
        int64_t tens = rest / 10, unit = rest % 10;
        words.push_back(tens == 1 ? std::string("mười") : std::string(kViDigits[tens]) + " mươi");
        if (unit) {
            if (unit == 1 && tens > 1) words.push_back("mốt");
            else if (unit == 4 && tens > 1) words.push_back("tư");
            else if (unit == 5) words.push_back("lăm");
            else words.push_back(kViDigits[unit]);
        }
    } else if (rest) {
        words.push_back(kViDigits[rest]);
    }
    if (words.empty()) return kViDigits[0];
    std::string out;
    for (std::size_t i = 0; i < words.size(); ++i) out += (i ? " " : "") + words[i];
    return out;
}

std::string vietnamese_number(const Num& n) {
    if (n.lt(1000)) return vietnamese_below_thousand(n.small());
    for (const Scale& scale : {Scale{9, "tỷ", ""}, Scale{6, "triệu", ""}, Scale{3, "nghìn", ""}}) {
        if (n.d.size() > scale.zeros) {
            Num count = n.div_pow10(scale.zeros), rest = n.mod_pow10(scale.zeros);
            std::string head = vietnamese_number(count) + " " + scale.singular;
            if (rest.zero()) return head;
            std::string tail = rest.lt(100) ? vietnamese_below_thousand(rest.small(), true) : vietnamese_number(rest);
            return head + " " + tail;
        }
    }
    return n.d;
}

std::string cardinal(const Num& n, const std::string& lang) {
    if (lang == "es") return spanish_number(n);
    if (lang == "it") return italian_number(n);
    if (lang == "fr") return french_number(n);
    if (lang == "de") return german_number(n);
    if (lang == "vi") return vietnamese_number(n);
    return english_number(n);
}

bool ends_with(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// _ES_SCALE_APOCOPE_RE: \b(veinti)?uno(?= (?:mil|millones|billones)\b) -> "veintiún" / "un": "veintiún mil", "treinta y un millones".
std::string spanish_scale_apocope(const std::string& words) {
    auto scale_follows = [&](std::size_t end) {
        for (const std::string scale : {" mil", " millones", " billones"}) {
            if (words.compare(end, scale.size(), scale) == 0 && (end + scale.size() == words.size() || words[end + scale.size()] == ' ')) return true;
        }
        return false;
    };
    std::string out;
    std::size_t i = 0;
    while (i < words.size()) {
        const bool boundary = i == 0 || words[i - 1] == ' ';
        if (boundary && words.compare(i, 9, "veintiuno") == 0 && scale_follows(i + 9)) {
            out += "veintiún";
            i += 9;
        } else if (boundary && words.compare(i, 3, "uno") == 0 && scale_follows(i + 3)) {
            out += "un";
            i += 3;
        } else {
            out.push_back(words[i++]);
        }
    }
    return out;
}

std::string to_words(const Num& n, const std::string& lang, bool ordinal = false) {
    if (ordinal && lang == "en" && !n.lt(1) && n.lt(32)) return kEnOrdinals[n.small()];
    if (ordinal && lang == "fr") {
        if (n.eq(1)) return "premier";
        if (n.eq(2)) return "deuxième";
        if (n.eq(5)) return "cinquième";
        if (n.eq(9)) return "neuvième";
        std::string word = french_number(n);
        if (ends_with(word, "e") || ends_with(word, "vingts") || ends_with(word, "cents") || ends_with(word, "millions") ||
            ends_with(word, "milliards")) {
            word.pop_back();
        }
        return word + "ième";
    }
    if (ordinal && lang == "de") {
        if (n.eq(1)) return "erste";
        if (n.eq(2)) return "zweite";
        if (n.eq(3)) return "dritte";
        if (n.eq(7)) return "siebte";
        if (n.eq(8)) return "achte";
        return german_number(n) + (n.lt(20) ? "te" : "ste");
    }
    if (ordinal && lang == "vi") {
        if (n.eq(1)) return "thứ nhất";
        if (n.eq(4)) return "thứ tư";
        return "thứ " + vietnamese_number(n);
    }
    return lang == "es" ? spanish_scale_apocope(cardinal(n, lang)) : cardinal(n, lang);
}

Text words(const Text& digits, const std::string& lang, bool ordinal = false) {
    return u(to_words(Num::of(digits_of(digits)), lang, ordinal));
}

// ---------------------------------------------------------------------------------------------------------------
// Spoken-form tables: scyllasband/text_data/<language>.json, embedded by tools/generate_text_data.py and parsed once.

#include "scyllasband_text_data.inc"

enum class Casing { sentence, any, exact };

struct TableEntry {
    Text spoken, after_name, otherwise, one, many;
    std::string gender = "m";
    bool ends = false;
};

struct TableForm {
    std::size_t entry;
    Text form;  // as written
    Casing casing;
};

// One section of a language's tables (_FormTable): every written form, longest first, and the forms by their lower-case
// spelling with spaces removed.
struct FormTable {
    std::vector<TableEntry> entries;
    std::vector<Text> written;
    std::map<Text, std::vector<TableForm>> keys;
};

// One letter run of a non-verbal word's shape (_nonverbal_runs): '-' is the place for one hyphen or space.
struct NonverbalRun {
    char32_t letter;
    std::size_t least;
    bool open_ended;
};

struct NonverbalWord {
    Text spoken;
    std::vector<std::vector<NonverbalRun>> shapes;
    std::string when;
};

// one_before_noun: the forms of "one" before a noun, and how the noun's gender is told (_expand_one_before_noun, _feminine).
struct OneBeforeNoun {
    bool present = false, compounds = false, adjective_rule = false;
    Text one, masculine, feminine, feminine_before_vowel, vowels;
    std::vector<std::pair<Text, Text>> compound_masculine;
    std::vector<Text> feminine_endings, adjective_feminine_endings, keep_before;
    std::set<Text> masculine_words, feminine_words, not_nouns, labels, dative_after;  // in lower case
    Text dative_masculine, dative_feminine;
};

struct LanguageData {
    std::map<Text, Text> letters, currency_scales, ordinal_leaders;
    std::map<std::string, Text> words, one;
    std::map<Text, std::vector<Text>> currency;  // symbol: one, many, minor one, minor many
    std::map<Text, std::string> ordinal_suffixes;
    std::vector<Text> months, ordinals, year_cues, scale_order, suffix_order;
    std::vector<std::pair<Text, Text>> symbols;
    std::set<Text> sentence_openers, name_leaders, feminine_titles, regnal_titles, regnal_names, unit_words, spell, capitals_only;
    Text currency_symbols, vowels;
    int64_t unlisted_min = 0, unlisted_max = 0;
    FormTable abbreviations, titles, before_number, units;
    std::vector<NonverbalWord> nonverbal;
    OneBeforeNoun one_before_noun;
};

Text compact(const Text& value) {  // re.sub(r"\s+", "", value)
    Text out;
    for (char32_t cp : value) {
        if (!is_space(cp)) out.push_back(cp);
    }
    return out;
}

void sort_longest_first(std::vector<Text>& values) {
    std::sort(values.begin(), values.end(), [](const Text& a, const Text& b) { return a.size() != b.size() ? a.size() > b.size() : a < b; });
}

std::vector<Text> texts_of(const Json& value) {  // text_data.forms: one string or a list
    std::vector<Text> out;
    if (value.is_string()) out.push_back(u(value.str()));
    for (const Json& item : value.elements()) out.push_back(u(item.str()));
    return out;
}

std::set<Text> text_set(const Json& value) {
    std::vector<Text> items = texts_of(value);
    return std::set<Text>(items.begin(), items.end());
}

FormTable form_table(const Json& entries, Casing default_casing) {
    FormTable table;
    for (const Json& item : entries.elements()) {
        TableEntry entry;
        entry.spoken = u(item.get("spoken").str());
        entry.after_name = u(item.get("after_name").str());
        entry.otherwise = u(item.get("otherwise").str());
        entry.one = u(item.get("one").str());
        entry.many = u(item.get("many").str());
        entry.gender = item.get("gender").str("m");
        entry.ends = item.get("ends").boolean();
        const std::string policy = item.get("case").str();
        const Casing casing = policy == "any"        ? Casing::any
                              : policy == "exact"    ? Casing::exact
                              : policy == "sentence" ? Casing::sentence
                                                     : default_casing;
        table.entries.push_back(std::move(entry));
        for (const Text& form : texts_of(item.get("written"))) {
            table.written.push_back(form);
            table.keys[lower(compact(form))].push_back(TableForm{table.entries.size() - 1, form, casing});
        }
    }
    sort_longest_first(table.written);
    return table;
}

// _nonverbal_runs: "mm+" is two or more "m"; "he+h+" one "h", "e"s, "h"s.
std::vector<NonverbalRun> nonverbal_runs(const Text& shape) {
    std::vector<NonverbalRun> runs;
    std::size_t index = 0;
    while (index < shape.size()) {
        const char32_t letter = shape[index];
        const bool open_ended = index + 1 < shape.size() && shape[index + 1] == '+';
        index += open_ended ? 2 : 1;
        if (letter != '-' && !runs.empty() && runs.back().letter == letter) {
            runs.back().least += 1;
            runs.back().open_ended = runs.back().open_ended || open_ended;
        } else {
            runs.push_back(NonverbalRun{letter, letter == '-' ? std::size_t{0} : std::size_t{1}, open_ended});
        }
    }
    return runs;
}

LanguageData language_data(const Json& json) {
    LanguageData data;
    for (const auto& [key, value] : json.get("letters").items()) data.letters[u(key)] = u(value.str());
    for (const auto& [key, value] : json.get("words").items()) data.words[key] = u(value.str());
    for (const auto& [key, value] : json.get("one").items()) data.one[key] = u(value.str());
    for (const auto& [key, value] : json.get("currency").items()) {
        data.currency[u(key)] = texts_of(value);
        data.currency_symbols += u(key);
    }
    for (const auto& [key, value] : json.get("currency_scales").items()) {
        data.currency_scales[u(key)] = u(value.str());
        data.scale_order.push_back(u(key));
    }
    sort_longest_first(data.scale_order);
    for (const auto& [key, value] : json.get("ordinal_suffixes").items()) {
        data.ordinal_suffixes[u(key)] = value.str();
        data.suffix_order.push_back(u(key));
    }
    sort_longest_first(data.suffix_order);
    for (const auto& [key, value] : json.get("ordinal_leaders").items()) data.ordinal_leaders[u(key)] = u(value.str());
    for (const auto& [key, value] : json.get("symbols").items()) data.symbols.emplace_back(u(key), u(value.str()));
    data.months = texts_of(json.get("months"));
    data.ordinals = texts_of(json.get("ordinals"));
    data.year_cues = texts_of(json.get("year_cues"));
    data.sentence_openers = text_set(json.get("sentence_openers"));
    data.name_leaders = text_set(json.get("name_leaders"));
    data.feminine_titles = text_set(json.get("feminine_titles"));
    data.regnal_titles = text_set(json.get("regnal_titles"));
    data.regnal_names = text_set(json.get("regnal_names"));
    data.spell = text_set(json.get("initialisms").get("spell"));
    data.capitals_only = text_set(json.get("initialisms").get("capitals_only"));
    const Json& unlisted = json.get("spell_unlisted");
    data.unlisted_min = unlisted.get("min").integer();
    data.unlisted_max = unlisted.get("max").integer();
    data.vowels = u(unlisted.get("vowels").str());
    data.abbreviations = form_table(json.get("abbreviations"), Casing::sentence);
    data.titles = form_table(json.get("titles"), Casing::sentence);
    data.before_number = form_table(json.get("before_number"), Casing::any);
    data.units = form_table(json.get("units"), Casing::any);
    const Json& nonverbal = json.get("nonverbal");
    for (const Json& item : nonverbal.get("words").elements()) {
        NonverbalWord word;
        word.spoken = u(item.get("spoken").str());
        for (const Text& shape : texts_of(item.get("shapes"))) word.shapes.push_back(nonverbal_runs(shape));
        word.when = item.get("when").str();
        data.nonverbal.push_back(std::move(word));
    }
    const Json& one_before = json.get("one_before_noun");
    if (one_before.is_object()) {
        OneBeforeNoun& table = data.one_before_noun;
        auto lowered = [](const Json& list) {
            std::set<Text> out;
            for (const Text& word : texts_of(list)) out.insert(lower(word));
            return out;
        };
        table.present = true;
        table.compounds = one_before.get("compounds").boolean();
        table.one = u(one_before.get("one").str());
        table.masculine = u(one_before.get("masculine").str());
        table.feminine = u(one_before.get("feminine").str());
        table.feminine_before_vowel = u(one_before.get("feminine_before_vowel").str());
        table.vowels = u(one_before.get("vowels").str());
        for (const auto& [key, value] : one_before.get("compound_masculine").items()) table.compound_masculine.emplace_back(u(key), u(value.str()));
        table.feminine_endings = texts_of(one_before.get("feminine_endings"));
        table.adjective_rule = one_before.has("adjective_feminine_endings");
        table.adjective_feminine_endings = texts_of(one_before.get("adjective_feminine_endings"));
        table.keep_before = texts_of(one_before.get("keep_before"));
        table.masculine_words = lowered(one_before.get("masculine_words"));
        table.feminine_words = lowered(one_before.get("feminine_words"));
        table.not_nouns = lowered(one_before.get("not_nouns"));
        table.labels = lowered(one_before.get("labels"));
        table.dative_after = lowered(one_before.get("dative_after"));
        table.dative_masculine = u(one_before.get("dative_masculine").str());
        table.dative_feminine = u(one_before.get("dative_feminine").str());
    }
    for (const Json& item : json.get("units").elements()) {  // _unit_words_of: a unit in capitals ("300 MPH") is no initialism
        for (const Text& form : texts_of(item.get("written"))) {
            Text word = lower(form);
            while (!word.empty() && word.back() == '.') word.pop_back();
            data.unit_words.insert(word);
        }
    }
    return data;
}

const std::string kEnglish = "en";

const LanguageData& text_data(const std::string& lang) {
    static const std::map<std::string, LanguageData> tables = [] {
        std::map<std::string, LanguageData> out;
        for (const TextDataFile& file : kTextData) out.emplace(file.language, language_data(Json::parse(file.json)));
        return out;
    }();
    auto found = tables.find(lang);
    return found == tables.end() ? tables.at(kEnglish) : found->second;
}

// _word: a word the rules insert, from the language's `words` table (English when the language has none).
Text spoken_word(const std::string& lang, const std::string& key) {
    const LanguageData& data = text_data(lang);
    auto found = data.words.find(key);
    if (found != data.words.end() && !found->second.empty()) return found->second;
    return text_data(kEnglish).words.at(key);
}

// _letter_name
Text letter_name(char32_t letter, const std::string& lang) {
    const Text key = upper(Text(1, letter));
    const LanguageData& data = text_data(lang);
    auto found = data.letters.find(key);
    if (found != data.letters.end() && !found->second.empty()) return found->second;
    const LanguageData& english = text_data(kEnglish);
    found = english.letters.find(key);
    return found == english.letters.end() ? Text(1, letter) : found->second;
}

// ---------------------------------------------------------------------------------------------------------------
// Matching shared by the table rules (_form_pattern, _case_matches, ...).

// One written form at p, as _form_pattern matches it under re.IGNORECASE: a space after a dot is optional, other spaces
// match any run of whitespace, and a form that starts or ends with a letter or digit is a whole token. The end, or npos.
std::size_t match_form(const Text& s, std::size_t p, const Text& form, bool left_boundary) {
    if (left_boundary && is_alnum(form[0]) && after_word_char(s, p)) return npos;
    std::size_t i = p;
    for (std::size_t k = 0; k < form.size(); ++k) {
        if (form[k] == ' ') {
            std::size_t j = run_end(s, i, is_space);
            if (j == i && !(k > 0 && form[k - 1] == '.')) return npos;
            i = j;
        } else if (i < s.size() && re_ignorecase_equal(s[i], form[k])) {
            ++i;
        } else {
            return npos;
        }
    }
    if (is_alnum(form.back()) && is_word(at(s, i))) return npos;
    return i;
}

// The table's forms tried longest first at p, as one alternation: the first that matches.
std::size_t match_table(const Text& s, std::size_t p, const FormTable& table, bool left_boundary) {
    for (const Text& form : table.written) {
        std::size_t e = match_form(s, p, form, left_boundary);
        if (e != npos) return e;
    }
    return npos;
}

// _sentence_start
bool sentence_start(const Text& s, std::size_t start) {
    std::size_t index = start;
    while (index > 0 && (is_space(s[index - 1]) || in_set(s[index - 1], U"\"'“‘«(¿¡["))) --index;
    return index == 0 || in_set(s[index - 1], U".!?…") || s.find('\n', index) < start;
}

// _case_matches, on spellings with spaces removed.
bool case_matches(const Text& matched, const Text& form, Casing casing, const Text& s, std::size_t start, bool capitals) {
    if (matched == form || casing == Casing::any) return true;
    if (casing == Casing::exact) return capitals && matched == upper(form);
    if (matched == lower(form) || (capitals && matched == upper(form))) return true;
    const Text head = upper(form.substr(0, 1)), rest = form.substr(std::min<std::size_t>(1, form.size()));
    if (matched == head + rest || matched == head + lower(rest)) return sentence_start(s, start);
    return false;
}

// _FormTable.find: the form whose casing policy accepts the matched text, or nullptr.
const TableForm* find_form(const FormTable& table, const Text& s, const Text& matched, std::size_t start, bool capitals) {
    const Text spelled = compact(matched);
    auto found = table.keys.find(lower(spelled));
    if (found == table.keys.end()) return nullptr;
    for (const TableForm& form : found->second) {
        if (case_matches(spelled, compact(form.form), form.casing, s, start, capitals)) return &form;
    }
    return nullptr;
}

bool ends_with_text(const Text& value, std::u32string_view suffix) {
    return value.size() >= suffix.size() && std::u32string_view(value).substr(value.size() - suffix.size()) == suffix;
}

// _cased: the replacement with a capital first letter when the written form had one.
Text cased(const Text& spoken, const Text& matched) {
    if (!matched.empty() && is_upper(matched[0]) && !spoken.empty() && is_lower(spoken[0])) return upper(spoken.substr(0, 1)) + spoken.substr(1);
    return spoken;
}

// str.rstrip(chars)
Text rstrip_chars(const Text& value, std::u32string_view chars) {
    std::size_t end = value.size();
    while (end > 0 && in_set(value[end - 1], chars)) --end;
    return value.substr(0, end);
}

// re.search(r"\d\s?$", s[:end]): a digit, with at most one whitespace character, before end ($ also before a final newline).
bool digit_before(const Text& s, std::size_t end) {
    for (std::size_t e : {end, end > 0 && s[end - 1] == '\n' ? end - 1 : npos}) {
        if (e == npos) continue;
        if (e >= 1 && is_decimal(s[e - 1])) return true;
        if (e >= 2 && is_space(s[e - 1]) && is_decimal(s[e - 2])) return true;
    }
    return false;
}

// _NEXT_WORD_RE.match(s, pos): \s+["'“‘«(\[]?([^\W\d_][\w'’-]*), the span of the word.
std::optional<std::pair<std::size_t, std::size_t>> next_word(const Text& s, std::size_t pos) {
    std::size_t j = run_end(s, pos, is_space);
    if (j == pos) return std::nullopt;
    std::size_t k = in_set(at(s, j), U"\"'“‘«([") ? j + 1 : j;
    if (!letter_char(at(s, k))) return std::nullopt;
    return std::make_pair(k, run_end(s, k + 1, [](char32_t c) { return is_word(c) || c == '\'' || c == 0x2019 || c == '-'; }));
}

// _after_capitalised_word
bool after_capitalised_word(const Text& s, std::size_t start) {
    std::vector<Text> before = words_before(s, start);
    if (before.empty()) return false;
    Text word = strip_chars(before.back(), U"\"'“‘«(");
    return !word.empty() && is_upper(word[0]) && is_alpha(word.back());
}

// _sentence_follows: a sentence ends at `end` when the text ends there or a capitalised word follows that is not another
// abbreviation (in German, where every noun is capitalised, a word from `sentence_openers`).
bool sentence_ends(const Text& s, std::size_t end, const std::string& lang) {
    if (strip_chars(s.substr(end), U" \t\r\n\"'”’»)]").empty()) return true;
    auto name = next_word(s, end);
    if (!name || !is_upper(s[name->first])) return false;
    const Text token = s.substr(name->first, run_end(s, name->first, [](char32_t c) { return !is_space(c); }) - name->first);
    const Text following = lower(rstrip_chars(token, U",;:!?)\"'”’»"));
    const LanguageData& data = text_data(lang);
    if (data.abbreviations.keys.count(following) || data.titles.keys.count(following)) return false;
    return lang != "de" || data.sentence_openers.count(s.substr(name->first, name->second - name->first)) > 0;
}

Text replace_all(const Text& value, const Text& from, const Text& to) {  // str.replace
    if (from.empty()) return value;
    Text out;
    std::size_t pos = 0;
    while (true) {
        std::size_t found = value.find(from, pos);
        if (found == npos) break;
        out.append(value, pos, found - pos);
        out += to;
        pos = found + from.size();
    }
    out.append(value, pos, npos);
    return out;
}

// ---------------------------------------------------------------------------------------------------------------
// Spoken-text normalizer.

std::string normalizer_language(const std::string& language) {
    std::string key = language.empty() ? "en" : language;
    std::replace(key.begin(), key.end(), '-', '_');
    key = encode(lower(decode(key)));
    static const std::map<std::string, std::string> table = {
        {"en", "en"}, {"en_us", "en"}, {"en_gb", "en"}, {"es", "es"}, {"es_mx", "es"}, {"es_es", "es"}, {"it", "it"}, {"it_it", "it"},
        {"fr", "fr"}, {"fr_fr", "fr"}, {"de", "de"}, {"de_de", "de"}, {"vi", "vi"}, {"vi_vn", "vi"}, {"vi_hn", "vi"}};
    auto it = table.find(key);
    return it == table.end() ? "en" : it->second;
}

bool one_of(const std::string& lang, std::initializer_list<const char*> languages) {
    return std::any_of(languages.begin(), languages.end(), [&](const char* item) { return lang == item; });
}

constexpr char32_t kMinusMark = 0xE000;  // _MINUS_MARK

Text normalize_punctuation_step(const Text& input, const std::string& lang) {
    Text value;
    for (char32_t cp : input) {
        if (!((cp >= 0x200B && cp <= 0x200D) || cp == 0xFEFF)) value.push_back(cp);
    }
    value = replace_char(value, '*', U" ");  // scene breaks, *emphasis* and footnote stars are not spoken
    // _MINUS_RE: a hyphen or minus sign right before a digit, not after a letter, digit, dot, comma or another dash ("1939-1945",
    // "COVID-19") and not after a number and a space ("10 -12" is a range): (?<![\w.,\-−])(?<!\d )[-−](?=\d). It is marked
    // here and written as the language's word once the text is known to be in capitals or not.
    value = substitute(value, [](const Text& s, std::size_t p) -> std::optional<Match> {
        if ((s[p] != '-' && s[p] != 0x2212) || !is_decimal(at(s, p + 1))) return std::nullopt;
        if (p > 0 && (is_word(s[p - 1]) || in_set(s[p - 1], U".,-\u2212"))) return std::nullopt;
        if (p > 1 && s[p - 1] == ' ' && is_decimal(s[p - 2])) return std::nullopt;
        return Match{p, p + 1, {}};
    }, [](const Text&, const Match&) { return Text(1, kMinusMark) + U" "; });
    // _normalize_dash_contract: tight dashes first, then an en dash between numbers is a range ("1939–1945"):
    // (?<=\d)\s?[‒–]\s?(?=\d)
    value = mark_tight_dashes(value);
    value = substitute(value, [](const Text& s, std::size_t p) -> std::optional<Match> {
        if (p == 0 || !is_decimal(s[p - 1])) return std::nullopt;
        std::size_t dash = is_space(s[p]) ? p + 1 : p;
        if (at(s, dash) != 0x2012 && at(s, dash) != 0x2013) return std::nullopt;
        if (is_space(at(s, dash + 1)) && is_decimal(at(s, dash + 2))) return Match{p, dash + 2, {}};
        if (is_decimal(at(s, dash + 1))) return Match{p, dash + 1, {}};
        return std::nullopt;
    }, [](const Text&, const Match&) { return Text(U"-"); });
    value = collapse_spaces(hyphen_runs_to_pause(dashes_to_pause(value)));
    // _UNICODE_TRANSLATION
    Text translated;
    for (char32_t cp : value) {
        switch (cp) {
            case 0x00A0: case 0x1680: case 0x2000: case 0x2001: case 0x2002: case 0x2003: case 0x2004: case 0x2005: case 0x2006:
            case 0x2007: case 0x2008: case 0x2009: case 0x200A: case 0x202F: case 0x205F: case 0x3000:
                translated.push_back(' ');
                break;
            case 0x02BC: case 0x2018: case 0x2019: case 0x201A: case 0x201B: case 0x2032: case 0xFF07:
                translated.push_back('\'');
                break;
            case 0x00AB: case 0x00BB: case 0x201C: case 0x201D: case 0x201E: case 0x201F: case 0x2033: case 0xFF02:
                translated.push_back('"');
                break;
            case 0x2026: translated += U"..."; break;
            case 0x2044: case 0x2215: translated.push_back('/'); break;
            case 0x00BC: translated += U" 1/4 "; break;
            case 0x00BD: translated += U" 1/2 "; break;
            case 0x00BE: translated += U" 3/4 "; break;
            default: translated.push_back(cp);
        }
    }
    value = fold_terminal_runs(translated);
    value = replace_char(value, '=', U" " + spoken_word(lang, "equals") + U" ");
    const Text degrees = spoken_word(lang, "degrees");
    // "30 °C" is read "thirty degrees Celsius": _DEGREE_SCALE_RE, \s?°\s?([CF])(?![A-Za-z])
    value = substitute(value, [](const Text& s, std::size_t p) -> std::optional<Match> {
        std::size_t q = is_space(s[p]) ? p + 1 : p;
        if (at(s, q) != 0xB0) return std::nullopt;
        std::size_t k = is_space(at(s, q + 1)) ? q + 2 : q + 1;
        if ((at(s, k) != 'C' && at(s, k) != 'F') || is_ascii_letter(at(s, k + 1))) return std::nullopt;
        return Match{p, k + 1, {{k, k + 1}}};
    }, [&](const Text& s, const Match& m) {
        return U" " + degrees + U" " + spoken_word(lang, s[m.end - 1] == 'C' ? "celsius" : "fahrenheit");
    });
    // _DEGREE_RE: a degree sign right after a letter is a number sign ("n° 5"), not degrees.
    value = substitute(value, [](const Text& s, std::size_t p) -> std::optional<Match> {
        if (s[p] != 0xB0 || (p > 0 && letter_char(s[p - 1]))) return std::nullopt;
        return Match{p, p + 1, {}};
    }, [&](const Text&, const Match&) { return U" " + degrees + U" "; });
    return collapse_spaces(value);
}

Text identifier_fragment(const Text& fragment, const std::string& lang) {
    Text out = replace_char(fragment, '+', U" " + spoken_word(lang, "plus") + U" ");
    out = replace_char(out, '_', U" ");
    out = replace_char(out, '-', U" ");
    return replace_char(out, '.', U" " + spoken_word(lang, "dot") + U" ");
}

Text normalize_at_symbols(const Text& value, const std::string& lang) {
    const Text at_word = spoken_word(lang, "at");
    auto local_char = [](char32_t c) { return is_ascii_letter(c) || is_ascii_digit(c) || in_set(c, U"._%+-"); };
    auto domain_char = [](char32_t c) { return is_ascii_letter(c) || is_ascii_digit(c) || c == '.' || c == '-'; };
    // \b(?P<local>[A-Za-z0-9._%+\-]+)@(?P<domain>[A-Za-z0-9.\-]+\.[A-Za-z]{2,})\b
    Text step = substitute(value, [&](const Text& s, std::size_t p) -> std::optional<Match> {
        if (!local_char(s[p]) || !word_boundary(s, p)) return std::nullopt;
        std::size_t at_pos = run_end(s, p, local_char);
        if (at(s, at_pos) != '@') return std::nullopt;
        std::size_t start = at_pos + 1;
        std::size_t r_end = run_end(s, start, domain_char);
        for (std::size_t k = r_end; k-- > start + 1;) {  // X = s[start, k) nonempty, s[k] == '.'
            if (s[k] != '.') continue;
            std::size_t e = run_end(s, k + 1, [](char32_t c) { return is_ascii_letter(c); });
            if (e - (k + 1) >= 2 && word_boundary(s, e)) return Match{p, e, {{p, at_pos}, {start, e}}};
        }
        return std::nullopt;
    }, [&](const Text& s, const Match& m) {
        return identifier_fragment(m.group(s, 0), lang) + U" " + at_word + U" " + identifier_fragment(m.group(s, 1), lang);
    });
    return replace_char(step, '@', U" " + at_word + U" ");
}

// _expand_dotted_initialisms
Text expand_dotted_initialisms(const Text& value, const std::string& lang) {
    // (?<![A-Za-zÀ-ÖØ-öø-ÿÑñ])(?:[A-Za-zÑñ]\.){2,}
    return substitute(value, [](const Text& s, std::size_t p) -> std::optional<Match> {
        if (p > 0 && latin_letter(s[p - 1])) return std::nullopt;
        std::size_t i = p, count = 0;
        while (initial_letter(at(s, i)) && at(s, i + 1) == '.') {
            i += 2;
            ++count;
        }
        if (count < 2) return std::nullopt;
        return Match{p, i, {}};
    }, [&](const Text& s, const Match& m) {
        Text spoken;
        for (std::size_t i = m.start; i < m.end; ++i) {
            if (!initial_letter(s[i])) continue;
            if (!spoken.empty()) spoken.push_back(' ');
            spoken += letter_name(s[i], lang);
        }
        // The last dot also ends the sentence at the end of the text, and after a lower-case initialism followed by
        // _SENTENCE_FOLLOWS_RE = \s+["'“‘(\[]?[A-ZÀ-ÖØ-ÞÑ]; an upper-case one before a capital is part of a name.
        std::size_t j = run_end(s, m.end, is_space);
        bool ends_sentence = j == s.size();
        if (!ends_sentence && j > m.end) {
            std::size_t k = in_set(s[j], U"\"'“‘([") ? j + 1 : j;
            bool lower_case = true;
            for (std::size_t i = m.start; i < m.end; ++i) lower_case = lower_case && (!initial_letter(s[i]) || !upper_latin(s[i]));
            ends_sentence = lower_case && (upper_latin(at(s, k)) || (k != j && upper_latin(at(s, j))));
        }
        return ends_sentence ? spoken + U"." : spoken;
    });
}

// _ascii_lower
char32_t ascii_lower(char32_t cp) { return cp >= 'A' && cp <= 'Z' ? cp + 32 : cp; }

// _nonverbal_match: where the shape ends when it matches at start (ASCII letters in either case), or npos.
std::size_t nonverbal_match(const std::vector<NonverbalRun>& runs, const Text& s, std::size_t start) {
    std::size_t pos = start, index = 0;
    const NonverbalRun* previous = nullptr;  // the last letter run, and the count found for it
    std::size_t previous_count = 0;
    while (index < runs.size()) {
        const NonverbalRun& run = runs[index];
        if (run.letter == '-') {
            const NonverbalRun& following = runs.at(index + 1);
            if (pos + 1 < s.size() && (s[pos] == '-' || s[pos] == ' ') && ascii_lower(s[pos + 1]) == following.letter) {
                ++pos;
            } else if (previous != nullptr && previous->letter == following.letter && previous->open_ended) {
                // Closed up, with the same letter on both sides ("uhhuh"): one run that holds both.
                if (previous_count < previous->least + following.least) return npos;
                ++index;
            }
            ++index;
            continue;
        }
        std::size_t count = 0;
        while (pos + count < s.size() && ascii_lower(s[pos + count]) == run.letter) ++count;
        if (count < run.least) return npos;
        previous = &run;
        previous_count = run.open_ended ? count : run.least;
        pos += previous_count;
        ++index;
    }
    return pos;
}

// _nonverbal_token: s[start:end] stands as a token of its own, not inside a word and not tied to one by a hyphen,
// apostrophe, slash or dot.
bool nonverbal_token(const Text& s, std::size_t start, std::size_t end) {
    static const std::u32string_view ties = U"-'\u2019/.";
    if (start > 0) {
        const char32_t before = s[start - 1];
        if (is_word(before) || (in_set(before, ties) && start > 1 && is_word(s[start - 2]))) return false;
    }
    if (end < s.size()) {
        const char32_t after = s[end];
        if (is_word(after) || (in_set(after, ties) && end + 1 < s.size() && is_word(s[end + 1]))) return false;
    }
    return true;
}

// _normalize_nonverbal: non-verbal words (the language's `nonverbal` table; English only) recognised by shape and
// rewritten to one spelling, in lower case (in capitals in text written all in capitals). Whole tokens in any case, never
// right after a number. A listed initialism in capitals inside mixed-case text stays ("the ER.").
Text normalize_nonverbal(const Text& s, const std::string& lang, bool capitals) {
    const LanguageData& data = text_data(lang);
    if (data.nonverbal.empty()) return s;
    auto punctuated = [&](std::size_t end) { return in_set(at(s, end), U",.!?\u2026"); };
    auto applies = [&](const std::string& when, std::size_t start, std::size_t end) {
        if ((start > 0 && is_decimal(s[start - 1])) || (start > 1 && is_space(s[start - 1]) && is_decimal(s[start - 2]))) return false;
        if (when == "before_punctuation" && !punctuated(end)) return false;
        if (when == "sentence_start_before_punctuation" && !(punctuated(end) && sentence_start(s, start))) return false;
        Text letters;
        for (std::size_t i = start; i < end; ++i) {
            if (s[i] != '-' && s[i] != ' ') letters.push_back(s[i]);
        }
        return capitals || !(data.spell.count(letters) || data.capitals_only.count(letters));
    };
    Text out;
    std::size_t last = 0, pos = 0;
    while (pos < s.size()) {
        std::size_t found_end = npos;
        const NonverbalWord* found = nullptr;
        if (is_ascii_letter(s[pos])) {
            for (const NonverbalWord& word : data.nonverbal) {
                for (const std::vector<NonverbalRun>& runs : word.shapes) {
                    const std::size_t end = nonverbal_match(runs, s, pos);
                    if (end == npos || (found != nullptr && end <= found_end)) continue;
                    if (nonverbal_token(s, pos, end) && applies(word.when, pos, end)) {
                        found_end = end;
                        found = &word;
                    }
                }
            }
        }
        if (found == nullptr) {
            ++pos;
            continue;
        }
        out.append(s, last, pos - last);
        out += capitals ? upper(found->spoken) : found->spoken;
        last = pos = found_end;
    }
    out.append(s, last, npos);
    return out;
}

// _expand_abbreviations: replaced wherever they stand; `ends` keeps a sentence-ending dot where the sentence ends.
Text expand_abbreviations(const Text& value, const std::string& lang, bool capitals) {
    const FormTable& table = text_data(lang).abbreviations;
    if (table.written.empty()) return value;
    return substitute(value, [&](const Text& s, std::size_t p) -> std::optional<Match> {
        std::size_t e = match_table(s, p, table, true);
        if (e == npos) return std::nullopt;
        return Match{p, e, {}};
    }, [&](const Text& s, const Match& m) {
        const Text matched = s.substr(m.start, m.end - m.start);
        const TableForm* form = find_form(table, s, matched, m.start, capitals);
        if (form == nullptr) return matched;
        const TableEntry& entry = table.entries[form->entry];
        Text spoken = cased(entry.spoken, matched);
        if (entry.ends && form->form.back() == '.' && sentence_ends(s, m.end, lang)) spoken += U".";
        return spoken;
    });
}

// _expand_symbols
Text expand_symbols(const Text& value, const std::string& lang) {
    Text out = value;
    for (const auto& [symbol, spoken] : text_data(lang).symbols) out = replace_all(out, symbol, U" " + spoken + U" ");
    return out;
}

// _expand_titles: replaced before a capitalised word (a name); `after_name` after a capitalised word, `otherwise` elsewhere.
Text expand_titles(const Text& value, const std::string& lang, bool capitals) {
    const LanguageData& data = text_data(lang);
    const FormTable& table = data.titles;
    if (table.written.empty()) return value;
    return substitute(value, [&](const Text& s, std::size_t p) -> std::optional<Match> {
        std::size_t e = match_table(s, p, table, true);
        if (e == npos) return std::nullopt;
        return Match{p, e, {}};
    }, [&](const Text& s, const Match& m) {
        const Text matched = s.substr(m.start, m.end - m.start);
        const TableForm* form = find_form(table, s, matched, m.start, capitals);
        if (form == nullptr || digit_before(s, m.start)) return matched;  // "7 FT", "5 MS": a unit after a number, not a title
        const TableEntry& entry = table.entries[form->entry];
        auto name = next_word(s, m.end);
        if (name && is_upper(s[name->first]) && !data.sentence_openers.count(s.substr(name->first, name->second - name->first))) {
            return cased(entry.spoken, matched);
        }
        Text spoken = after_capitalised_word(s, m.start) ? entry.after_name : Text();
        if (spoken.empty()) spoken = entry.otherwise;
        if (spoken.empty()) return matched;
        const bool dot = form->form.back() == '.' && sentence_ends(s, m.end, lang);
        return cased(spoken, matched) + (dot ? U"." : U"");
    });
}

// _expand_before_number: replaced when a number follows ("No. 5", "§ 3", "~5").
Text expand_before_number(const Text& value, const std::string& lang, bool capitals) {
    const FormTable& table = text_data(lang).before_number;
    if (table.written.empty()) return value;
    return substitute(value, [&](const Text& s, std::size_t p) -> std::optional<Match> {
        std::size_t e = match_table(s, p, table, true);
        if (e == npos) return std::nullopt;
        return Match{p, e, {}};
    }, [&](const Text& s, const Match& m) {
        const Text matched = s.substr(m.start, m.end - m.start);
        const TableForm* form = find_form(table, s, matched, m.start, capitals);
        if (form == nullptr || !is_decimal(at(s, run_end(s, m.end, is_space)))) return matched;  // _NUMBER_FOLLOWS_RE: \s*\d
        return cased(table.entries[form->entry].spoken, matched) + U" ";
    });
}

// _unit_words: the unit in the singular for 1 (French: below 2), and the word for 1 agreeing with the unit's gender ("eine
// Tonne", "veintiún kilómetros", "vingt et une tonnes").
Text unit_words(const Text& raw, const TableEntry& entry, const std::string& lang);

// _expand_units: a unit after a number, with or without one space: (?<![\w.,])(\d+(?:[.,]\d+)*)( ?)(forms)
Text expand_units(const Text& value, const std::string& lang, bool capitals) {
    const FormTable& table = text_data(lang).units;
    if (table.written.empty()) return value;
    return substitute(value, [&](const Text& s, std::size_t p) -> std::optional<Match> {
        if (!is_decimal(s[p]) || after_number_char(s, p)) return std::nullopt;
        std::size_t e = run_end(s, p, is_decimal);
        while (in_set(at(s, e), U".,") && is_decimal(at(s, e + 1))) e = run_end(s, e + 1, is_decimal);
        std::size_t q = at(s, e) == ' ' ? e + 1 : e;
        std::size_t end = match_table(s, q, table, false);
        if (end == npos) return std::nullopt;
        return Match{p, end, {{p, e}, {q, end}}};
    }, [&](const Text& s, const Match& m) {
        const TableForm* form = find_form(table, s, m.group(s, 1), m.groups[1].first, capitals);
        if (form == nullptr) return s.substr(m.start, m.end - m.start);
        const bool dot = form->form.back() == '.' && sentence_ends(s, m.end, lang);
        return unit_words(m.group(s, 0), table.entries[form->entry], lang) + (dot ? U"." : U"");
    });
}

// _NAME_INITIAL_RE and _expand_name_initial
Text expand_name_initials(const Text& value, const std::string& lang) {
    const LanguageData& data = text_data(lang);
    // (?<![A-Za-zÀ-ÖØ-öø-ÿ.'’])([A-Z])\.(?=\s+(?:[A-Z]\.\s+)*([A-ZÀ-ÖØ-Þ][a-zß-öø-ÿ'’]+))
    return substitute(value, [](const Text& s, std::size_t p) -> std::optional<Match> {
        if (!ascii_upper(s[p]) || at(s, p + 1) != '.' || after_name_char(s, p)) return std::nullopt;
        std::size_t w = run_end(s, p + 2, is_space);
        if (w == p + 2) return std::nullopt;
        while (ascii_upper(at(s, w)) && at(s, w + 1) == '.') {
            std::size_t next = run_end(s, w + 2, is_space);
            if (next == w + 2) break;
            w = next;
        }
        if (!upper_latin(at(s, w))) return std::nullopt;
        std::size_t e = run_end(s, w + 1, name_tail);
        if (e == w + 1) return std::nullopt;
        return Match{p, p + 2, {{p, p + 1}, {w, e}}};
    }, [&](const Text& s, const Match& m) {
        Text original = s.substr(m.start, m.end - m.start);
        if (data.sentence_openers.count(m.group(s, 1))) return original;
        std::vector<Text> before = words_before(s, m.start);
        Text previous = before.empty() ? Text() : before.back();
        bool another_initial = at(s, run_end(s, m.end, is_space) + 1) == '.';
        bool leader = data.name_leaders.count(lower(previous)) > 0;
        bool capital = !previous.empty() && is_upper(previous[0]);
        bool in_name = previous.empty() || capital || leader || another_initial;
        if (!in_name || (lang == "en" && s[m.start] == 'I' && !capital)) return original;
        return letter_name(s[m.start], lang);
    });
}

// _SPACED_INITIALS_RE and _expand_spaced_initials
Text expand_spaced_initials(const Text& value, const std::string& lang) {
    // (?<![A-Za-zÀ-ÖØ-öø-ÿ.'’])[A-Z]\.(?: [A-Z]\.){1,2}(?![A-Za-zÀ-ÖØ-öø-ÿ])
    return substitute(value, [](const Text& s, std::size_t p) -> std::optional<Match> {
        if (!ascii_upper(s[p]) || at(s, p + 1) != '.' || after_name_char(s, p)) return std::nullopt;
        std::size_t count = 0;
        while (count < 2 && at(s, p + 2 + 3 * count) == ' ' && ascii_upper(at(s, p + 3 + 3 * count)) && at(s, p + 4 + 3 * count) == '.') ++count;
        for (; count >= 1; --count) {
            std::size_t e = p + 2 + 3 * count;
            if (!latin_letter(at(s, e))) return Match{p, e, {}};
        }
        return std::nullopt;
    }, [&](const Text& s, const Match& m) {
        Text spoken;
        for (std::size_t i = m.start; i < m.end; i += 3) {
            if (!spoken.empty()) spoken.push_back(' ');
            spoken += letter_name(s[i], lang);
        }
        return sentence_follows(s, m.end) ? spoken + U"." : spoken;
    });
}

// X{0,3}(?:IX|IV|V?I{0,3}) at p: the ends of a numeral, in the order the pattern tries them.
std::vector<std::size_t> roman_numeral_ends(const Text& s, std::size_t p) {
    std::vector<std::size_t> ends;
    std::size_t x_run = std::min<std::size_t>(run_end(s, p, [](char32_t c) { return c == 'X'; }) - p, 3);
    for (std::size_t x = x_run + 1; x-- > 0;) {
        std::size_t q = p + x;
        if (at(s, q) == 'I' && at(s, q + 1) == 'X') ends.push_back(q + 2);
        if (at(s, q) == 'I' && at(s, q + 1) == 'V') ends.push_back(q + 2);
        for (std::size_t v : {std::size_t{1}, std::size_t{0}}) {
            if (v == 1 && at(s, q) != 'V') continue;
            std::size_t i_run = std::min<std::size_t>(run_end(s, q + v, [](char32_t c) { return c == 'I'; }) - (q + v), 3);
            for (std::size_t i = i_run + 1; i-- > 0;) ends.push_back(q + v + i);
        }
    }
    return ends;
}

// _roman_value
int roman_value(const Text& numeral) {
    auto digit = [](char32_t c) { return c == 'I' ? 1 : c == 'V' ? 5 : 10; };
    int total = 0;
    for (std::size_t i = 0; i < numeral.size(); ++i) {
        int d = digit(numeral[i]);
        total += i + 1 < numeral.size() && digit(numeral[i + 1]) > d ? -d : d;
    }
    return total;
}

// A capitalised word of three letters or more, the rest in lower case: the word a ruler's numeral follows.
bool title_word(const Text& word) {
    return word.size() > 2 && is_upper(word[0]) && all_lower(std::u32string_view(word).substr(1));
}

// _ordinal_word: `kind` is the gender ("m", "f"), "short" for the Spanish forms before a noun ("primer", "tercer"), or a
// French "m1"/"f1"/"m2"/"f2" form for "premier"/"première"/"second"/"seconde".
Text ordinal_word(const Num& value, const std::string& lang, const std::string& kind = "m") {
    if (lang == "fr") {
        if (value.eq(1) && kind == "m1") return U"premier";
        if (value.eq(1) && kind == "f1") return U"première";
        if (value.eq(2) && kind == "m2") return U"second";
        if (value.eq(2) && kind == "f2") return U"seconde";
        return u(to_words(value, lang, true));
    }
    const std::vector<Text>& ordinals = text_data(lang).ordinals;
    if ((lang == "es" || lang == "it") && !ordinals.empty()) {
        if (value.zero()) return u(to_words(value, lang));
        Text word;
        if (value.lt(static_cast<int64_t>(ordinals.size()) + 1)) {
            word = ordinals[static_cast<std::size_t>(value.small() - 1)];
        } else if (lang == "it") {
            Text cardinal = u(to_words(value, lang));
            if (!ends_with_text(cardinal, U"tre") && !ends_with_text(cardinal, U"tré") && !ends_with_text(cardinal, U"sei")) cardinal.pop_back();
            word = replace_all(cardinal, U"tré", U"tre") + U"esimo";
        } else {
            return u(to_words(value, lang));
        }
        if (kind == "short" && (value.eq(1) || value.eq(3))) return word.substr(0, word.size() - 1);
        if (kind == "f" && !word.empty() && word.back() == 'o') return word.substr(0, word.size() - 1) + U"a";
        return word;
    }
    return u(to_words(value, lang, true));
}

struct RomanContext {
    Text previous, earlier;
    bool named = false;
};

// _expand_roman_numerals.context: the word before the numeral, the two before it, and whether the first is a name.
RomanContext roman_context(const Text& s, std::size_t start) {
    std::vector<Text> before = words_before(s, start);
    std::vector<Text> tokens;
    for (std::size_t i = before.size() > 3 ? before.size() - 3 : 0; i < before.size(); ++i) {
        tokens.push_back(strip_chars(before[i], U"\"'“‘«(,"));
    }
    RomanContext context;
    if (!tokens.empty()) context.previous = tokens.back();
    for (std::size_t i = 0; i + 1 < tokens.size(); ++i) context.earlier += (i ? U" " : U"") + tokens[i];
    context.named = title_word(context.previous);
    return context;
}

// _expand_roman_numerals: English numerals after a name (rulers read "the second"); French, Spanish and Italian centuries and
// rulers. Only I, V and X.
Text expand_roman_numerals(const Text& value, const std::string& lang) {
    const LanguageData& data = text_data(lang);
    if (lang == "en") {
        // _EN_ROMAN_RE: (?<![\w.'’])(X{0,3}(?:IX|IV|V?I{0,3}))(?=(?:['’]s)?(?![\w'’]))
        return substitute(value, [](const Text& s, std::size_t p) -> std::optional<Match> {
            if (!in_set(s[p], U"XVI") || after_word_char(s, p)) return std::nullopt;
            auto word_char = [&](std::size_t k) { return is_word(at(s, k)) || in_set(at(s, k), U"'’"); };
            for (std::size_t e : roman_numeral_ends(s, p)) {
                bool possessive = in_set(at(s, e), U"'’") && at(s, e + 1) == 's' && !word_char(e + 2);
                if (e > p && (possessive || !word_char(e))) return Match{p, e, {}};
            }
            return std::nullopt;
        }, [&](const Text& s, const Match& m) {
            Text numeral = s.substr(m.start, m.end - m.start);
            RomanContext context = roman_context(s, m.start);
            bool regnal = data.regnal_names.count(context.previous) > 0;
            for (const Text& word : split_whitespace(context.earlier)) regnal = regnal || data.regnal_titles.count(word) > 0;
            if (!context.named || (numeral.size() < 2 && !regnal)) return numeral;
            if (numeral == U"I") {
                // A lone "I" after a ruler's name is nearly always the pronoun: "the first" only with a title before the
                // name, or when it closes the phrase (_EN_ROMAN_ONE_FOLLOW_RE: ['’]s(?![\w'’])|\s*(?:$|[.,;:!?)"”]|of(?![\w'’]))).
                bool titled = false;
                for (const Text& word : split_whitespace(context.earlier)) titled = titled || data.regnal_titles.count(word) > 0;
                if (!titled) {
                    std::size_t b = m.start;
                    while (b > 0 && is_space(s[b - 1])) --b;
                    std::size_t k = m.end;
                    const bool possessive = in_set(at(s, k), U"'’") && at(s, k + 1) == 's' && !is_word(at(s, k + 2)) && !in_set(at(s, k + 2), U"'’");
                    while (k < s.size() && is_space(s[k])) ++k;
                    const bool closes = possessive || k >= s.size() || in_set(s[k], U".,;:!?)\"”") ||
                        (s[k] == 'o' && at(s, k + 1) == 'f' && !is_word(at(s, k + 2)) && !in_set(at(s, k + 2), U"'’"));
                    if ((b > 0 && s[b - 1] == ',') || !closes) return numeral;
                }
            }
            const Num number = Num::of(roman_value(numeral));
            return regnal ? U"the " + ordinal_word(number, lang) : u(to_words(number, lang));
        });
    }
    const auto century = data.words.find("century");
    const Text century_word = century == data.words.end() ? Text() : century->second;
    // _ROMAN_RE: (?<![\w.'’])(X{0,3}(?:IX|IV|V?I{0,3}))(er|re|e|ème)?(?![\w'’]), with a numeral
    static const Text suffixes[] = {U"er", U"re", U"e", U"ème", U""};
    return substitute(value, [](const Text& s, std::size_t p) -> std::optional<Match> {
        if (!in_set(s[p], U"XVI") || after_word_char(s, p)) return std::nullopt;
        for (std::size_t e : roman_numeral_ends(s, p)) {
            if (e == p) continue;
            for (const Text& suffix : suffixes) {
                std::size_t end = e + suffix.size();
                if (s.compare(e, suffix.size(), suffix) != 0 || is_word(at(s, end)) || in_set(at(s, end), U"'’")) continue;
                return Match{p, end, {{p, e}, suffix.empty() ? std::make_pair(npos, npos) : std::make_pair(e, end)}};
            }
        }
        return std::nullopt;
    }, [&](const Text& s, const Match& m) -> Text {
        Text original = s.substr(m.start, m.end - m.start), numeral = m.group(s, 0), suffix = m.group(s, 1);
        if (numeral.size() == 1 && suffix.empty()) return original;
        const Num number = Num::of(roman_value(numeral));
        RomanContext context = roman_context(s, m.start);
        std::size_t j = run_end(s, m.end, is_space);
        const Text following = strip_chars(s.substr(j, std::min(s.find(' ', j), s.size()) - j), U".,;:!?\"'()");
        if (lang == "fr") {
            if (!suffix.empty()) {
                auto kind = data.ordinal_suffixes.find(suffix);
                return ordinal_word(number, lang, kind == data.ordinal_suffixes.end() ? "m" : kind->second);
            }
            return context.named ? u(to_words(number, lang)) : original;
        }
        if (!suffix.empty()) return original;
        if (lower(context.previous) == century_word) return lang == "es" ? u(to_words(number, lang)) : ordinal_word(number, lang);
        if (lang == "it" && lower(following) == century_word) return ordinal_word(number, lang);
        if (!context.named) return original;
        if (lang == "es" && !number.lt(11)) return u(to_words(number, lang));  // Spanish reads rulers above ten with cardinals
        std::vector<Text> earlier = split_whitespace(context.earlier);
        bool feminine = (!context.previous.empty() && context.previous.back() == 'a') ||
                        (!earlier.empty() && data.feminine_titles.count(lower(earlier.back())) > 0);
        return ordinal_word(number, lang, feminine ? "f" : "m");
    });
}

// _LETTER_WORD_RE and _expand_initialism: initialisms outside English, spelled with letter names. `spell` entries are never
// words, so they match in capitals or in lower case, also in text written all in capitals; `capitals_only` entries and
// unlisted ones (`spell_unlisted`: capitals, no vowel) only in capitals inside mixed-case text.
Text expand_initialisms(const Text& value, const std::string& lang, bool capitals) {
    const LanguageData& data = text_data(lang);
    // (?<![\w.])([^\W\d_]{2,8})(?![\w'’]|-?\d)
    return substitute(value, [](const Text& s, std::size_t p) -> std::optional<Match> {
        if (!letter_char(s[p]) || (p > 0 && (is_word(s[p - 1]) || s[p - 1] == '.'))) return std::nullopt;
        std::size_t e = run_end(s, p, letter_char);
        if (e - p < 2 || e - p > 8) return std::nullopt;
        char32_t next = at(s, e);
        if (is_word(next) || next == '\'' || next == 0x2019 || (next == '-' && is_decimal(at(s, e + 1)))) return std::nullopt;
        return Match{p, e, {}};
    }, [&](const Text& s, const Match& m) {
        const Text word = s.substr(m.start, m.end - m.start), capitals_word = upper(word);
        if (!(data.spell.count(capitals_word) && (word == capitals_word || word == lower(word)))) {
            if (word != capitals_word || capitals) return word;
            bool spelled = data.capitals_only.count(capitals_word) > 0;
            if (!spelled) {
                const int64_t length = static_cast<int64_t>(word.size());
                spelled = data.unlisted_min <= length && length <= data.unlisted_max &&
                          std::none_of(word.begin(), word.end(), [&](char32_t c) { return in_set(c, data.vowels); });
            }
            if (!spelled || (digit_before(s, m.start) && data.unit_words.count(lower(word)))) return word;
        }
        Text spoken;
        for (char32_t letter : capitals_word) {
            if (!spoken.empty()) spoken.push_back(' ');
            spoken += letter_name(letter, lang);
        }
        return spoken;
    });
}

// _EN_NUMBER_RANGE_RE (English) or _NUMBER_RANGE_RE, and _expand_number_range.
Text year_words(const Num& year);
Text german_year(const Num& year);

Text expand_number_ranges(const Text& value, const std::string& lang) {
    const char32_t separator = lang == "en" ? '.' : ',';
    // \d+(?:SEP\d+)? at p: the end of the number
    auto number_end = [separator](const Text& s, std::size_t p) {
        std::size_t e = run_end(s, p, is_decimal);
        if (at(s, e) == separator && is_decimal(at(s, e + 1))) e = run_end(s, e + 1, is_decimal);
        return e;
    };
    const Text range = spoken_word(lang, "range");
    // (?<![\w.,-])(\d+(?:\.\d+)?)\s?-\s?(\d+(?:\.\d+)?)(?![\w-])(?!\.\d); outside English the decimal mark is a comma and
    // the last lookahead is (?![.,]\d)
    return substitute(value, [&](const Text& s, std::size_t p) -> std::optional<Match> {
        if (!is_decimal(s[p]) || (p > 0 && (is_word(s[p - 1]) || in_set(s[p - 1], U".,-")))) return std::nullopt;
        std::size_t first_end = number_end(s, p);
        std::size_t dash = is_space(at(s, first_end)) ? first_end + 1 : first_end;
        if (at(s, dash) != '-') return std::nullopt;
        std::size_t q = is_space(at(s, dash + 1)) ? dash + 2 : dash + 1;
        if (!is_decimal(at(s, q))) return std::nullopt;
        std::size_t e = number_end(s, q);
        char32_t next = at(s, e);
        if (is_word(next) || next == '-') return std::nullopt;
        if ((next == '.' || (next == ',' && separator == ',')) && is_decimal(at(s, e + 1))) return std::nullopt;
        return Match{p, e, {{p, first_end}, {q, e}}};
    }, [&](const Text& s, const Match& m) {
        Text first = m.group(s, 0), second = m.group(s, 1);
        if (first.find(separator) == npos && second.find(separator) == npos) {
            // Left alone: descending whole numbers, a three-digit group before a four-digit one, and leading zeros.
            Num low = Num::of(digits_of(first)), high = Num::of(digits_of(second));
            bool rising = high.d.size() != low.d.size() ? high.d.size() > low.d.size() : high.d > low.d;
            if (!rising || (first.size() == 3 && second.size() == 4) || first[0] == '0' || second[0] == '0') return s.substr(m.start, m.end - m.start);
            if (lang == "en" && first.size() == 4 && second.size() == 4 && !low.lt(1100) && high.lt(2100)) {
                return year_words(low) + U" to " + year_words(high);  // "1939-1945"
            }
            if (lang == "de" && first.size() == 4 && second.size() == 4 && !low.lt(1100) && high.lt(2100)) {
                return german_year(low) + U" " + range + U" " + german_year(high);  // "neunzehnhundertneununddreißig bis ..."
            }
        }
        return first + U" " + range + U" " + second;
    });
}

bool grouped_integer(const Text& raw, char32_t separator) {  // re.fullmatch(r"\d{1,3}(?:SEP\d{3})+", raw)
    std::size_t first = run_end(raw, 0, is_decimal);
    if (first < 1 || first > 3 || first == raw.size()) return false;
    std::size_t pos = first;
    while (pos < raw.size()) {
        if (raw[pos] != separator || pos + 4 > raw.size()) return false;
        for (std::size_t k = 1; k <= 3; ++k) {
            if (!is_decimal(raw[pos + k])) return false;
        }
        pos += 4;
    }
    return true;
}

bool is_non_english(const std::string& lang) { return lang == "es" || lang == "it" || lang == "fr" || lang == "de" || lang == "vi"; }

// _split_decimal: (whole, fraction digits or nullopt)
std::pair<Text, std::optional<Text>> split_decimal(const Text& raw, const std::string& lang, bool currency) {
    Text value;
    for (char32_t cp : strip(raw)) {
        if (cp != ' ') value.push_back(cp);
    }
    if (value.empty()) return {U"0", std::nullopt};
    std::size_t comma = value.rfind(','), dot = value.rfind('.');
    char32_t separator = 0;
    if (comma != npos && dot != npos) {
        separator = comma > dot ? ',' : '.';
    } else if (comma != npos) {
        // In French a single comma is the decimal sign even before three digits ("2,125 kg"); money keeps "1,000 €" a thousand.
        const bool grouped = grouped_integer(value, ',');
        const bool french = lang == "fr" && std::count(value.begin(), value.end(), U',') == 1 && !(currency && grouped);
        if (french || (!grouped && (is_non_english(lang) || currency))) separator = ',';
    } else if (dot != npos) {
        if (!grouped_integer(value, '.') && (lang == "en" || currency || value.size() - dot - 1 != 3)) separator = '.';
    }
    if (!separator) return {value, std::nullopt};
    std::size_t position = value.rfind(separator);
    Text fraction;
    for (std::size_t i = position + 1; i < value.size(); ++i) {
        if (is_decimal(value[i])) fraction.push_back(value[i]);
    }
    return {value.substr(0, position), fraction};
}

Text unit_words(const Text& raw, const TableEntry& entry, const std::string& lang) {
    auto [whole, fraction] = split_decimal(raw, lang, false);
    const Num value = Num::of(digits_of(whole));
    const bool singular = (value.eq(1) && !fraction) || (lang == "fr" && value.lt(2));
    Text number = raw;
    if (!fraction && one_of(lang, {"de", "fr", "es", "it"})) {
        if (value.eq(1)) {
            number = text_data(lang).one.at(entry.gender);
        } else if (value.mod_pow10(1).eq(1) && !value.mod_pow10(2).eq(11) && (lang == "es" || lang == "fr")) {
            Text words = u(to_words(value, lang));
            if (lang == "es" && ends_with_text(words, U"uno")) {
                number = words.substr(0, words.size() - 3) + (entry.gender == "f" ? U"una" : ends_with_text(words, U"veintiuno") ? U"ún" : U"un");
            } else if (lang == "fr" && entry.gender == "f" && ends_with_text(words, U"un")) {
                number = words + U"e";
            }
        }
    }
    return number + U" " + (singular ? entry.one : entry.many);
}

Text decimal_words(const Text& raw, const std::string& lang) {
    auto [whole, fraction] = split_decimal(raw, lang, false);
    Text out = words(whole, lang);
    if (!fraction) return out;
    out += U" " + spoken_word(lang, "point");
    if (one_of(lang, {"fr", "es", "it", "vi"}) && !fraction->empty() && fraction->size() <= 3) {
        // _DECIMAL_AS_NUMBER_LANGUAGES: a short decimal part read as a number, each leading zero on its own ("3,14" "trois
        // virgule quatorze", "tres coma catorce").
        std::size_t zeros = 0;
        while (zeros < fraction->size() && (*fraction)[zeros] == '0') ++zeros;
        for (std::size_t k = 0; k < zeros; ++k) out += U" " + words(U"0", lang);
        if (zeros < fraction->size()) out += U" " + words(fraction->substr(zeros), lang);
        return out;
    }
    for (char32_t digit : *fraction) out += U" " + words(Text(1, digit), lang);
    return out;
}

// _currency_to_words
Text currency_words(const Text& raw, const Text& symbol, const std::string& lang) {
    const LanguageData& data = text_data(lang);
    auto found = data.currency.find(symbol);
    const std::vector<Text>& units = found != data.currency.end() ? found->second : text_data(kEnglish).currency.at(U"$");
    auto [whole, cents] = split_decimal(raw, lang, true);
    Num major = Num::of(digits_of(whole));
    Text major_words = u(to_words(major, lang)) + U" " + (major.eq(1) ? units.at(0) : units.at(1));
    if (!cents) return major_words;
    std::string padded = digits_of(*cents) + "00";
    Num minor = Num::of(padded.substr(0, 2));
    if (minor.zero()) return major_words;
    return major_words + U" " + spoken_word(lang, "and") + U" " + u(to_words(minor, lang)) + U" " + (minor.eq(1) ? units.at(2) : units.at(3));
}

// Greedy [0-9](?:[0-9.,]*[0-9])? from p (p is an ASCII digit): end of the longest run that ends in a digit.
std::size_t amount_end(const Text& s, std::size_t p) {
    std::size_t e = run_end(s, p, [](char32_t c) { return is_ascii_digit(c) || c == '.' || c == ','; });
    while (e > p + 1 && !is_ascii_digit(s[e - 1])) --e;
    return e;
}

// _expand_currency: "$3.50", "£12", "40€", "50.000 ₫"; with a scale word "$5bn" and "$3.5 million".
Text expand_currency(const Text& value, const std::string& lang) {
    const LanguageData& data = text_data(lang);
    if (data.currency.empty()) return value;
    const Text& symbols = data.currency_symbols;
    // ([symbols])\s*([0-9](?:[0-9.,]*[0-9])?)(?:\s?(scales)(?!\w))?
    Text step = substitute(value, [&](const Text& s, std::size_t p) -> std::optional<Match> {
        if (!in_set(s[p], symbols)) return std::nullopt;
        std::size_t q = run_end(s, p + 1, is_space);
        if (!is_ascii_digit(at(s, q))) return std::nullopt;
        std::size_t e = amount_end(s, q);
        std::size_t t = is_space(at(s, e)) ? e + 1 : e;
        for (const Text& scale : data.scale_order) {
            if (s.compare(t, scale.size(), scale) == 0 && !is_word(at(s, t + scale.size()))) {
                return Match{p, t + scale.size(), {{p, p + 1}, {q, e}, {t, t + scale.size()}}};
            }
        }
        return Match{p, e, {{p, p + 1}, {q, e}, {npos, npos}}};
    }, [&](const Text& s, const Match& m) {
        const Text symbol = m.group(s, 0), raw = m.group(s, 1), scale = m.group(s, 2);
        if (!scale.empty()) return decimal_words(raw, lang) + U" " + data.currency_scales.at(scale) + U" " + data.currency.at(symbol).at(1);
        return currency_words(raw, symbol, lang);
    });
    // (?<![\w.,])([0-9](?:[0-9.,]*[0-9])?)\s*([symbols]|EUR)(?!\w), IGNORECASE
    return substitute(step, [&](const Text& s, std::size_t p) -> std::optional<Match> {
        if (!is_ascii_digit(s[p]) || after_number_char(s, p)) return std::nullopt;
        std::size_t e = amount_end(s, p);
        std::size_t q = run_end(s, e, is_space);
        std::size_t symbol_end = npos;
        if (in_set(at(s, q), symbols)) symbol_end = q + 1;
        else if (q + 3 <= s.size() && re_ignorecase_equal(s[q], 'E') && re_ignorecase_equal(s[q + 1], 'U') && re_ignorecase_equal(s[q + 2], 'R'))
            symbol_end = q + 3;
        if (symbol_end == npos || is_word(at(s, symbol_end))) return std::nullopt;
        return Match{p, symbol_end, {{p, e}, {q, symbol_end}}};
    }, [&](const Text& s, const Match& m) {
        const Text symbol = m.group(s, 1);
        return currency_words(m.group(s, 0), upper(symbol) == U"EUR" ? Text(U"€") : symbol, lang);
    });
}

Text month_name(int month, const std::string& lang) { return text_data(lang).months.at(static_cast<std::size_t>(month - 1)); }

// _year_words: an English year as spoken: 1990 "nineteen ninety", 1905 "nineteen oh five", 1900 "nineteen hundred", 2005 "two
// thousand five", 2024 "twenty twenty four". Years outside 1100-2099 are read as numbers.
std::string year_words_utf8(const Num& year) {
    if (year.d.size() > 4) return to_words(year, "en");
    int64_t y = year.small();
    if (y >= 2000 && y <= 2009) return y == 2000 ? std::string("two thousand") : "two thousand " + to_words(Num::of(y - 2000), "en");
    if (y >= 1100 && y <= 2099) {
        int64_t high = y / 100, low = y % 100;
        std::string tail = low == 0 ? std::string("hundred") : low < 10 ? "oh " + to_words(Num::of(low), "en") : to_words(Num::of(low), "en");
        return to_words(Num::of(high), "en") + " " + tail;
    }
    return to_words(year, "en");
}

Text year_words(const Num& year) { return u(year_words_utf8(year)); }

// _german_year: German years 1100-1999 are read in hundreds, 1990 "neunzehnhundertneunzig".
Text german_year(const Num& year) {
    if (!year.lt(1100) && year.lt(2000)) {
        Num high = year.div_pow10(2), low = year.mod_pow10(2);
        return u(to_words(high, "de") + "hundert" + (low.zero() ? std::string() : to_words(low, "de")));
    }
    return u(to_words(year, "de"));
}

// _clock_hour: the hour of a clock time with the language's word for it ("quinze heures", "une heure", "fünfzehn Uhr").
Text clock_hour(const Num& hour, const std::string& lang) {
    Text spoken = u(to_words(hour, lang));
    if (lang == "fr" && ends_with_text(spoken, U"un")) spoken += U"e";  // "une heure", "vingt et une heures"
    else if (lang == "de" && spoken == U"eins") spoken = U"ein";      // "ein Uhr"
    return spoken + U" " + spoken_word(lang, hour.lt(2) ? "hour" : "hours");
}

// _ampm_words: "AM" and "PM" spelled ("ay em"); the G2P reads a bare "AM" as the word "am".
Text ampm_words(char32_t letter, const std::string& lang) { return letter_name(letter, lang) + U" " + letter_name('M', lang); }

// _plural_words: "twenty" -> "twenties", "six" -> "sixes", "nineteen ninety" -> "nineteen nineties".
std::string plural_words(std::string words) {
    std::replace(words.begin(), words.end(), '-', ' ');
    if (ends_with(words, "y")) return words.substr(0, words.size() - 1) + "ies";
    return words + (ends_with(words, "x") ? "es" : "s");
}

// _plural_number_words: "4s" fours, "100s" hundreds, "1000s" thousands.
std::string plural_number_words(const Num& value) {
    if (value.eq(100)) return "hundreds";
    if (value.eq(1000)) return "thousands";
    if (value.eq(1000000)) return "millions";
    if (value.eq(1000000000)) return "billions";
    return plural_words(to_words(value, "en"));
}

// _split_alphanumeric: letter and digit runs of a token such as "V3" or "MP3"; a single letter, or up to three letters that are
// all capitals or have no vowel, spelled with letter names; digits left for the number rules. In English a final "s" after
// digits makes the number plural ("MP3s").
Text split_alphanumeric(const Text& token, const std::string& lang) {
    std::vector<std::pair<Text, bool>> runs;  // run, is digits
    for (std::size_t k = 0; k < token.size();) {
        bool digit = is_ascii_digit(token[k]);
        std::size_t e = run_end(token, k, [digit](char32_t c) { return digit ? is_ascii_digit(c) : is_ascii_letter(c); });
        runs.emplace_back(token.substr(k, e - k), digit);
        k = e;
    }
    bool plural = lang == "en" && runs.size() > 1 && runs.back().first == U"s" && runs[runs.size() - 2].second;
    if (plural) runs.pop_back();
    Text out;
    auto add = [&](const Text& part) {
        if (!out.empty()) out += U" ";
        out += part;
    };
    for (std::size_t r = 0; r < runs.size(); ++r) {
        const auto& [run, digit] = runs[r];
        bool upper_case = true, vowel = false;
        for (char32_t c : run) {
            upper_case = upper_case && c >= 'A' && c <= 'Z';
            vowel = vowel || in_set(c, U"AEIOUYaeiouy");
        }
        if (digit && plural && r + 1 == runs.size()) {
            add(u(plural_number_words(Num::of(digits_of(run)))));
        } else if (!digit && (run.size() == 1 || (run.size() <= 3 && (upper_case || !vowel)))) {
            for (char32_t c : run) add(letter_name(c, lang));
        } else {
            add(run);
        }
    }
    return out;
}

// \d+(?:\.\d+)? at p: the end of the number, or p.
std::size_t decimal_number_end(const Text& s, std::size_t p) {
    std::size_t e = run_end(s, p, is_decimal);
    if (e > p && at(s, e) == '.' && is_decimal(at(s, e + 1))) e = run_end(s, e + 1, is_decimal);
    return e;
}

// _EN_NUMBER, \d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)? at p: the end of the number, or p.
std::size_t english_number_end(const Text& s, std::size_t p) {
    std::size_t e = run_end(s, p, is_decimal);
    if (e > p && e - p <= 3) {
        std::size_t g = e;
        while (at(s, g) == ',' && is_decimal(at(s, g + 1)) && is_decimal(at(s, g + 2)) && is_decimal(at(s, g + 3))) g += 4;
        if (g > e) return at(s, g) == '.' && is_decimal(at(s, g + 1)) ? run_end(s, g + 1, is_decimal) : g;
    }
    return decimal_number_end(s, p);
}

// _expand_date
Text expand_date(const Text& month_s, const Text& day_s, const Text& year_s, const std::string& lang, bool british = false) {
    Num month = Num::of(digits_of(month_s)), day = Num::of(digits_of(day_s)), year = Num::of(digits_of(year_s));
    if (month.lt(1) || !month.lt(13) || day.lt(1) || !day.lt(32)) return month_s + U"/" + day_s + U"/" + year_s;
    Text name = month_name(static_cast<int>(month.small()), lang);
    if (lang == "en" && british) return U"the " + u(to_words(day, lang, true)) + U" of " + name + U" " + year_words(year);
    if (lang == "en") return name + U" " + u(to_words(day, lang, true)) + U", " + year_words(year);
    if (lang == "vi") return u(day.d) + U" " + name + U" " + spoken_word(lang, "year") + U" " + u(year.d);
    if (lang == "es") return u(day.d) + U" de " + name + U" de " + u(year.d);
    if (lang == "de") return u(to_words(day, lang, true)) + U"n " + name + U" " + german_year(year);
    if (day.eq(1) && (lang == "fr" || lang == "it")) return ordinal_word(day, lang) + U" " + name + U" " + u(year.d);  // "premier mai"
    return u(day.d) + U" " + name + U" " + u(year.d);
}

Text expand_fraction(const Text& numerator_s, const Text& denominator_s, const std::string& lang) {
    Num numerator = Num::of(digits_of(numerator_s)), denominator = Num::of(digits_of(denominator_s));
    std::string head = to_words(numerator, lang);
    if (lang == "en") {
        static const char* const names[][2] = {{"half", "halves"}, {"third", "thirds"}, {"fourth", "fourths"}, {"fifth", "fifths"},
                                               {"sixth", "sixths"}, {"seventh", "sevenths"}, {"eighth", "eighths"}, {"ninth", "ninths"},
                                               {"tenth", "tenths"}};
        if (!denominator.lt(2) && denominator.lt(11)) return u(head + " " + names[denominator.small() - 2][numerator.eq(1) ? 0 : 1]);
        return u(head + " over " + to_words(denominator, lang));
    }
    return u(head) + U" " + spoken_word(lang, "over") + U" " + u(to_words(denominator, lang));
}

std::size_t digits_end(const Text& s, std::size_t p) { return run_end(s, p, is_decimal); }

// \b\d{1,3}(?:SEP\d{3}){min_groups,}\b at p
std::optional<Match> grouped_number_at(const Text& s, std::size_t p, char32_t separator, std::size_t min_groups = 1) {
    if (!is_decimal(s[p]) || !word_boundary(s, p)) return std::nullopt;
    std::size_t run = digits_end(s, p) - p;
    if (run > 3) return std::nullopt;
    std::vector<std::size_t> ends;
    std::size_t pos = p + run;
    while (at(s, pos) == separator && pos + 4 <= s.size() && is_decimal(s[pos + 1]) && is_decimal(s[pos + 2]) && is_decimal(s[pos + 3])) {
        pos += 4;
        ends.push_back(pos);
    }
    for (std::size_t k = ends.size(); k-- > 0 && k + 1 >= min_groups;) {
        if (word_boundary(s, ends[k])) return Match{p, ends[k], {}};
    }
    return std::nullopt;
}

// A year after a cue word: \b(cues)\s+(year)(?![\d.,]\d|\w). `four_digits` tells whether the year at w matches.
Text expand_cued_years(const Text& value, const std::vector<Text>& cues, bool ignore_case,
                       const std::function<bool(const Text&, std::size_t)>& four_digits, const std::function<Text(const Num&)>& spoken) {
    return substitute(value, [&](const Text& s, std::size_t p) -> std::optional<Match> {
        if (!word_boundary(s, p)) return std::nullopt;
        for (const Text& cue : cues) {
            if (cue.empty() || p + cue.size() > s.size()) continue;
            bool same = true;
            for (std::size_t k = 0; k < cue.size() && same; ++k) same = ignore_case ? re_ignorecase_equal(s[p + k], cue[k]) : s[p + k] == cue[k];
            if (!same) continue;
            std::size_t cue_end = p + cue.size(), w = run_end(s, cue_end, is_space);
            if (w == cue_end || !four_digits(s, w)) continue;
            std::size_t y = w + 4;
            if (((is_decimal(at(s, y)) || in_set(at(s, y), U".,")) && is_decimal(at(s, y + 1))) || is_word(at(s, y))) continue;
            return Match{p, y, {{p, cue_end}, {w, y}}};
        }
        return std::nullopt;
    }, [&](const Text& s, const Match& m) { return m.group(s, 0) + U" " + spoken(Num::of(digits_of(m.group(s, 1)))); });
}

// _expand_one_before_noun: a number read with "one" takes the form the noun after it needs ("un mensaje", "una alarma",
// "veintiún años", "uno studente", "un'ora", "ein Termin", "eine Stunde"); a label before the number keeps it a number.
Text one_before_noun(const Text& s, const Match& m, const OneBeforeNoun& table, const std::string& lang) {
    const Text original = s.substr(m.start, m.end - m.start);
    const Text digits = m.group(s, 0), space = m.group(s, 1), noun = m.group(s, 2);
    const Text words = u(to_words(Num::of(digits_of(digits)), lang));
    if (words != table.one && !(table.compounds && ends_with_text(words, table.one))) return original;
    const Text low = lower(noun);
    if (table.not_nouns.count(low)) return original;
    std::size_t index = m.start;  // _previous_word
    while (index > 0 && is_space(s[index - 1])) --index;
    std::size_t start = index;
    while (start > 0 && (is_alnum(s[start - 1]) || s[start - 1] == '\'' || s[start - 1] == 0x2019)) --start;
    const Text previous = lower(s.substr(start, index - start));
    if (start < index && table.labels.count(previous)) return original;
    auto any_ending = [&](const std::vector<Text>& endings) {
        return std::any_of(endings.begin(), endings.end(), [&](const Text& ending) { return ends_with_text(low, ending); });
    };
    bool feminine;  // _feminine
    if (table.masculine_words.count(low)) feminine = false;
    else if (table.feminine_words.count(low)) feminine = true;
    else if (table.adjective_rule && is_lower(noun[0])) feminine = any_ending(table.adjective_feminine_endings);
    else feminine = any_ending(table.feminine_endings);
    if (words == table.one && start < index && table.dative_after.count(previous)) {
        if (is_lower(noun[0])) {
            // A dative adjective ends in -en whatever the gender ("mit einer neuen Nachricht"): the noun after it decides (_next_word).
            std::size_t q = run_end(s, m.end, is_space), e = run_end(s, q, is_alnum);
            const Text following = s.substr(q, e - q);
            const Text low_following = lower(following);
            bool f = false;
            if (!following.empty() && is_upper(following[0])) {
                if (table.masculine_words.count(low_following)) f = false;
                else if (table.feminine_words.count(low_following)) f = true;
                else f = std::any_of(table.feminine_endings.begin(), table.feminine_endings.end(),
                                     [&](const Text& ending) { return ends_with_text(low_following, ending); });
            }
            feminine = f;
        }
        return (feminine ? table.dative_feminine : table.dative_masculine) + space + noun;  // "in einer Stunde", "seit einem Jahr"
    }
    if (words == table.one) {
        if (feminine) {
            if (!table.feminine_before_vowel.empty() && in_set(low[0], table.vowels)) return table.feminine_before_vowel + noun;
            return table.feminine + space + noun;
        }
        for (const Text& prefix : table.keep_before) {
            if (low.compare(0, prefix.size(), prefix) == 0) return table.one + space + noun;  // "uno studente"
        }
        return table.masculine + space + noun;
    }
    const Text stem = words.substr(0, words.size() - table.one.size());
    if (feminine) return stem + table.feminine + space + noun;  // "veintiuna casas"
    for (const auto& [ending, form] : table.compound_masculine) {
        if (ends_with_text(stem, ending)) return stem.substr(0, stem.size() - ending.size()) + form + space + noun;  // "veintiún años"
    }
    return stem + table.masculine + space + noun;  // "treinta y un días"
}

Text normalize_text(const Text& text, const std::string& language) {
    const std::string lang = normalizer_language(language);
    const LanguageData& data = text_data(lang);
    std::string dialect = language;
    for (char& c : dialect) c = c == '-' ? '_' : static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    const bool british = dialect == "en_gb";  // dates day first ("1/9/2026" the first of September)
    Text value = normalize_punctuation_step(nfc(text), lang);
    if (lang == "en") {
        // (?<![A-Za-z0-9])ok(?![A-Za-z0-9]), IGNORECASE
        value = substitute(value, [](const Text& s, std::size_t p) -> std::optional<Match> {
            if (!ci_literal(s, p, "ok")) return std::nullopt;
            if ((p > 0 && ascii_alnum_ci(s[p - 1])) || ascii_alnum_ci(at(s, p + 2))) return std::nullopt;
            return Match{p, p + 2, {}};
        }, [](const Text&, const Match&) { return Text(U"okay"); });
    }
    value = normalize_at_symbols(value, lang);
    // Text written all in capitals: casing tells nothing about names or initialisms, and the result stays in capitals.
    const bool capitals = std::any_of(value.begin(), value.end(), is_upper) && std::none_of(value.begin(), value.end(), is_lower);
    value = replace_char(value, kMinusMark, capitals ? upper(spoken_word(lang, "minus")) : spoken_word(lang, "minus"));
    value = expand_abbreviations(value, lang, capitals);
    value = expand_symbols(value, lang);
    value = expand_dotted_initialisms(value, lang);
    if (one_of(lang, {"fr", "es", "it", "de", "vi"})) {
        // Thousands set apart by spaces: (?<![\w.,])(?<!\d )[1-9]\d{0,2}(?: \d{3})+(?![\w])(?! \d)
        value = substitute(value, [](const Text& s, std::size_t p) -> std::optional<Match> {
            if (s[p] < '1' || s[p] > '9' || after_number_char(s, p) || (p > 1 && s[p - 1] == ' ' && is_decimal(s[p - 2]))) return std::nullopt;
            std::size_t e = run_end(s, p + 1, is_decimal);
            if (e - p > 3) return std::nullopt;
            std::size_t g = e;
            while (at(s, g) == ' ' && is_decimal(at(s, g + 1)) && is_decimal(at(s, g + 2)) && is_decimal(at(s, g + 3))) g += 4;
            if (g == e || is_word(at(s, g)) || (at(s, g) == ' ' && is_decimal(at(s, g + 1)))) return std::nullopt;
            return Match{p, g, {}};
        }, [](const Text& s, const Match& m) { return replace_char(s.substr(m.start, m.end - m.start), ' ', Text()); });
    }
    value = expand_currency(value, lang);
    // Units before titles and initialisms: "7 SQ FT" and "300 MPH" are units, not "Fort" or a spelled initialism.
    value = expand_units(value, lang, capitals);
    value = expand_titles(value, lang, capitals);
    value = expand_before_number(value, lang, capitals);
    value = expand_name_initials(value, lang);
    value = expand_spaced_initials(value, lang);
    if (one_of(lang, {"en", "fr", "es", "it"}) && !capitals) value = expand_roman_numerals(value, lang);
    // Non-verbal words take one spelling: after the rules above, which tell a sentence start by its capital, and before the
    // lexicon would spell them as initialisms and the hyphen becomes a space.
    value = normalize_nonverbal(value, lang, capitals);
    if (one_of(lang, {"de", "fr", "es", "it"})) value = expand_initialisms(value, lang, capitals);
    if (lang == "de") {
        // _GERMAN_DATE_RE: (?<![\w.,])(\d{1,2})\.\s?(\d{1,2})\.\s?(\d{4})(?!\w|\.\d)
        value = substitute(value, [](const Text& s, std::size_t p) -> std::optional<Match> {
            if (!is_decimal(s[p]) || after_number_char(s, p)) return std::nullopt;
            std::size_t day_end = is_decimal(at(s, p + 1)) ? p + 2 : p + 1;
            if (at(s, day_end) != '.') return std::nullopt;
            std::size_t q = is_space(at(s, day_end + 1)) ? day_end + 2 : day_end + 1;
            if (!is_decimal(at(s, q))) return std::nullopt;
            std::size_t month_end = is_decimal(at(s, q + 1)) ? q + 2 : q + 1;
            if (at(s, month_end) != '.') return std::nullopt;
            std::size_t r = is_space(at(s, month_end + 1)) ? month_end + 2 : month_end + 1;
            if (digits_end(s, r) < r + 4) return std::nullopt;
            std::size_t y = r + 4;
            if (is_word(at(s, y)) || (at(s, y) == '.' && is_decimal(at(s, y + 1)))) return std::nullopt;
            return Match{p, y, {{p, day_end}, {q, month_end}, {r, y}}};
        }, [&](const Text& s, const Match& m) {
            Num day = Num::of(digits_of(m.group(s, 0))), month = Num::of(digits_of(m.group(s, 1)));
            if (day.lt(1) || !day.lt(32) || month.lt(1) || !month.lt(13)) return s.substr(m.start, m.end - m.start);
            return u(to_words(day, "de", true)) + U"n " + month_name(static_cast<int>(month.small()), "de") + U" " +
                   german_year(Num::of(digits_of(m.group(s, 2))));
        });
    }
    value = expand_number_ranges(value, lang);

    // Times: \b(\d{1,2}):(\d{2})(?:\s*([AaPp][Mm]))?\b; in the languages that name the hour (_time_pattern), with the hour
    // word written after them: (?:\s*(?:hour words)(?!\w))? ("15:00 Uhr").
    std::vector<Text> hour_words;
    if (one_of(lang, {"fr", "vi", "de"})) {
        hour_words = {spoken_word(lang, "hour")};
        if (spoken_word(lang, "hours") != hour_words[0]) hour_words.push_back(spoken_word(lang, "hours"));
        sort_longest_first(hour_words);
    }
    auto with_hour_word = [&hour_words](const Text& s, Match m) {
        std::size_t q = run_end(s, m.end, is_space);
        for (const Text& word : hour_words) {
            if (s.compare(q, word.size(), word) == 0 && !is_word(at(s, q + word.size()))) {
                m.end = q + word.size();
                break;
            }
        }
        return m;
    };
    value = substitute(value, [&](const Text& s, std::size_t p) -> std::optional<Match> {
        if (!is_decimal(s[p]) || !word_boundary(s, p)) return std::nullopt;
        std::size_t colon = is_decimal(at(s, p + 1)) && at(s, p + 2) == ':' ? p + 2 : at(s, p + 1) == ':' ? p + 1 : npos;
        if (colon == npos || !is_decimal(at(s, colon + 1)) || !is_decimal(at(s, colon + 2))) return std::nullopt;
        std::size_t minutes_end = colon + 3;
        std::size_t q = run_end(s, minutes_end, is_space);
        if (in_set(at(s, q), U"AaPp") && in_set(at(s, q + 1), U"Mm") && word_boundary(s, q + 2)) {
            return with_hour_word(s, Match{p, q + 2, {{p, colon}, {colon + 1, minutes_end}, {q, q + 2}}});
        }
        if (!word_boundary(s, minutes_end)) return std::nullopt;
        return with_hour_word(s, Match{p, minutes_end, {{p, colon}, {colon + 1, minutes_end}, {npos, npos}}});
    }, [&](const Text& s, const Match& m) {
        Num hour = Num::of(digits_of(m.group(s, 0))), minute = Num::of(digits_of(m.group(s, 1)));
        Text ampm = m.group(s, 2);
        for (char32_t& c : ampm) c = (c >= 'a' && c <= 'z') ? c - 32 : c;
        if (one_of(lang, {"fr", "vi", "de"}) && hour.lt(25) && minute.lt(60)) {
            // "15:30" as "quinze heures trente", "fünfzehn Uhr dreißig"; "15:00" as the hour alone.
            Text spoken = clock_hour(hour, lang);
            if (!minute.zero()) spoken += U" " + u(to_words(minute, lang));
            return strip(spoken + U" " + ampm);
        }
        if (minute.zero() && one_of(lang, {"es", "it"}) && hour.lt(25)) return strip(words(m.group(s, 0), lang) + U" " + ampm);
        if (lang == "it" && hour.lt(25) && minute.lt(60)) {
            return strip(words(m.group(s, 0), lang) + U" e " + u(to_words(minute, lang)) + U" " + ampm);  // "le quindici e trenta"
        }
        if (lang == "en" && minute.zero() && ampm.empty() && (hour.zero() || hour.eq(24))) return Text(U"midnight");
        if (lang == "en" && minute.zero() && ampm.empty() && !hour.lt(13) && hour.lt(24)) {
            return u(to_words(hour, lang)) + U" hundred";  // a 24-hour time on the hour: "fifteen hundred"
        }
        Text minute_words;
        if (minute.zero() && lang == "en") minute_words = U"o'clock";
        else if (minute.lt(10) && lang == "en") minute_words = U"oh " + u(to_words(minute, lang));
        else minute_words = u(to_words(minute, lang));
        if (!ampm.empty() && lang == "en") ampm = ampm_words(ampm[0], lang);
        return strip(words(m.group(s, 0), lang) + U" " + minute_words + U" " + ampm);
    });
    if (lang == "en") {
        // Clock times without minutes: (?<![\w.,:])(\d{1,2})\s*([AaPp])[Mm](?!\w)
        value = substitute(value, [](const Text& s, std::size_t p) -> std::optional<Match> {
            if (!is_decimal(s[p]) || (p > 0 && (is_word(s[p - 1]) || in_set(s[p - 1], U".,:")))) return std::nullopt;
            for (std::size_t length : {std::size_t{2}, std::size_t{1}}) {
                if (length == 2 && !is_decimal(at(s, p + 1))) continue;
                std::size_t q = run_end(s, p + length, is_space);
                if (in_set(at(s, q), U"AaPp") && in_set(at(s, q + 1), U"Mm") && !is_word(at(s, q + 2))) {
                    return Match{p, q + 2, {{p, p + length}, {q, q + 1}}};
                }
            }
            return std::nullopt;
        }, [&](const Text& s, const Match& m) {
            Num hour = Num::of(digits_of(m.group(s, 0)));
            if (hour.lt(1) || !hour.lt(13)) return s.substr(m.start, m.end - m.start);
            return u(to_words(hour, lang)) + U" " + ampm_words(s[m.groups[1].first], lang);
        });
    }
    if (lang == "fr" || lang == "vi") {
        // Clock times written with "h", _CLOCK_RE: (?<![\w.,])(\d{1,2}) ?h(?: ?(\d{2}))?(?![\w])
        value = substitute(value, [](const Text& s, std::size_t p) -> std::optional<Match> {
            if (!is_decimal(s[p]) || after_number_char(s, p)) return std::nullopt;
            std::size_t e = is_decimal(at(s, p + 1)) ? p + 2 : p + 1;
            std::size_t h = at(s, e) == ' ' ? e + 1 : e;
            if (at(s, h) != 'h') return std::nullopt;
            std::size_t q = at(s, h + 1) == ' ' ? h + 2 : h + 1;
            if (is_decimal(at(s, q)) && is_decimal(at(s, q + 1)) && !is_word(at(s, q + 2))) return Match{p, q + 2, {{p, e}, {q, q + 2}}};
            if (is_word(at(s, h + 1))) return std::nullopt;
            return Match{p, h + 1, {{p, e}, {npos, npos}}};
        }, [&](const Text& s, const Match& m) {
            Num hour = Num::of(digits_of(m.group(s, 0))), minutes = Num::of(digits_of(m.group(s, 1)));
            if (!hour.lt(25) || !minutes.lt(60)) return s.substr(m.start, m.end - m.start);
            Text spoken = clock_hour(hour, lang);
            return m.matched(1) && !minutes.zero() ? spoken + U" " + u(minutes.d) : spoken;  // "15 h 00" is "quinze heures"
        });
    }
    // "31%" and, as French and German print it, "31 %"; "0.5 ‰": (?<![\d.,])(\d[\d.,]*) ?SIGN
    struct Sign {
        char32_t mark;
        const char* key;
    };
    for (const Sign sign : {Sign{'%', "percent"}, Sign{0x2030, "permille"}}) {
        const char32_t mark = sign.mark;
        const Text spoken = spoken_word(lang, sign.key);
        value = substitute(value, [mark](const Text& s, std::size_t p) -> std::optional<Match> {
            if (!is_decimal(s[p]) || (p > 0 && (is_decimal(s[p - 1]) || s[p - 1] == '.' || s[p - 1] == ','))) return std::nullopt;
            std::size_t e = run_end(s, p, [](char32_t c) { return is_decimal(c) || c == '.' || c == ','; });
            std::size_t q = at(s, e) == ' ' ? e + 1 : e;
            if (at(s, q) != mark) return std::nullopt;
            return Match{p, q + 1, {{p, e}}};
        }, [&](const Text& s, const Match& m) { return decimal_words(m.group(s, 0), lang) + U" " + spoken; });
    }
    if (lang == "en") {
        // _english_year_pattern: a year 1100-2099 after a year word ("in 1776", "July 1969"), IGNORECASE
        value = expand_cued_years(value, data.year_cues, true, [](const Text& s, std::size_t w) {
            bool digits = is_decimal(at(s, w + 2)) && is_decimal(at(s, w + 3));
            return digits && ((at(s, w) == '1' && at(s, w + 1) >= '1' && at(s, w + 1) <= '9') || (at(s, w) == '2' && at(s, w + 1) == '0'));
        }, year_words);
    }
    if (lang == "en") {
        // _EN_DAY_MONTH_RE: (?<![\w.,])(?:([Tt]he)\s+)?(\d{1,2})(?:st|nd|rd|th)?\s+(January|...|December)\b, months in capitals
        static const std::vector<Text> kMonths = {U"January", U"February", U"March", U"April", U"May", U"June", U"July", U"August",
                                                  U"September", U"October", U"November", U"December"};
        value = substitute(value, [](const Text& s, std::size_t p) -> std::optional<Match> {
            if (p > 0 && (is_word(s[p - 1]) || in_set(s[p - 1], U".,"))) return std::nullopt;
            std::size_t d = p;
            if ((s[p] == 't' || s[p] == 'T') && at(s, p + 1) == 'h' && at(s, p + 2) == 'e' && is_space(at(s, p + 3))) d = run_end(s, p + 3, is_space);
            if (!is_decimal(at(s, d))) return std::nullopt;
            std::size_t e = is_decimal(at(s, d + 1)) ? d + 2 : d + 1;
            if (is_decimal(at(s, e))) return std::nullopt;
            std::size_t after = e;
            for (const char32_t* suffix : {U"st", U"nd", U"rd", U"th"}) {
                if (at(s, e) == suffix[0] && at(s, e + 1) == suffix[1] && is_space(at(s, e + 2))) after = e + 2;
            }
            std::size_t q = run_end(s, after, is_space);
            if (q == after) return std::nullopt;
            for (const Text& month : kMonths) {
                if (s.compare(q, month.size(), month) == 0 && !is_word(at(s, q + month.size()))) {
                    return Match{p, q + month.size(), {{p, d == p ? p : p + 3}, {d, e}, {q, q + month.size()}}};
                }
            }
            return std::nullopt;
        }, [&](const Text& s, const Match& m) {
            const Num day = Num::of(digits_of(m.group(s, 1)));
            if (day.lt(1) || !day.lt(32)) return s.substr(m.start, m.end - m.start);
            const Text the = m.groups[0].second > m.groups[0].first ? m.group(s, 0) : Text(U"the");
            return the + U" " + u(to_words(day, lang, true)) + U" of " + m.group(s, 2);
        });
    } else if (lang == "fr" || lang == "it") {
        // _first_of_month_pattern: (?<![\w.,\-])1(?=\s+(?:months)(?!\w)), IGNORECASE: "le 1 septembre" "le premier septembre"
        value = substitute(value, [&](const Text& s, std::size_t p) -> std::optional<Match> {
            if (s[p] != '1' || (p > 0 && (is_word(s[p - 1]) || in_set(s[p - 1], U".,-")))) return std::nullopt;
            std::size_t q = run_end(s, p + 1, is_space);
            if (q == p + 1) return std::nullopt;
            for (const Text& month : data.months) {
                if (q + month.size() <= s.size() && lower(s.substr(q, month.size())) == lower(month) && !is_word(at(s, q + month.size()))) {
                    return Match{p, p + 1, {}};
                }
            }
            return std::nullopt;
        }, [&](const Text&, const Match&) { return ordinal_word(Num::of("1"), lang); });
    }
    if (!data.suffix_order.empty()) {
        // _expand_ordinal_suffixes: (?<![\w.,])(\d+)(suffixes)(?!\w), IGNORECASE: "2nd", "1er", "1.º", "3ª"
        value = substitute(value, [&](const Text& s, std::size_t p) -> std::optional<Match> {
            if (!is_decimal(s[p]) || after_number_char(s, p)) return std::nullopt;
            std::size_t e = digits_end(s, p);
            for (const Text& suffix : data.suffix_order) {
                std::size_t end = match_form(s, e, suffix, false);
                if (end == npos || is_word(at(s, end))) continue;
                return Match{p, end, {{p, e}, {e, end}}};
            }
            return std::nullopt;
        }, [&](const Text& s, const Match& m) {
            const Text suffix = m.group(s, 1);
            auto kind = data.ordinal_suffixes.find(lower(suffix));
            if (kind == data.ordinal_suffixes.end()) kind = data.ordinal_suffixes.find(suffix);
            return ordinal_word(Num::of(digits_of(m.group(s, 0))), lang, kind == data.ordinal_suffixes.end() ? "m" : kind->second);
        });
    }
    if (lang == "de") {
        // German ordinals are written with a dot: (?<![\w.,])(\d{1,3})\.(?=\s+([^\W\d_][\w-]*))
        value = substitute(value, [](const Text& s, std::size_t p) -> std::optional<Match> {
            if (!is_decimal(s[p]) || after_number_char(s, p)) return std::nullopt;
            std::size_t e = digits_end(s, p);
            if (e - p > 3 || at(s, e) != '.') return std::nullopt;
            std::size_t w = run_end(s, e + 1, is_space);
            if (w == e + 1 || !letter_char(at(s, w))) return std::nullopt;
            std::size_t g = run_end(s, w + 1, [](char32_t c) { return is_word(c) || c == '-'; });
            return Match{p, e + 1, {{p, e}, {w, g}}};
        }, [&](const Text& s, const Match& m) {
            std::vector<Text> before = words_before(s, m.start);
            if (before.empty()) return s.substr(m.start, m.end - m.start);  // a numbered list item ("1. Einleitung")
            const Text previous = lower(strip_chars(before.back(), U"\"'„“("));
            const Text following = m.group(s, 1);
            bool month = std::any_of(data.months.begin(), data.months.end(), [&](const Text& name) { return lower(name) == lower(following); });
            auto leader = data.ordinal_leaders.find(previous);
            if (leader == data.ordinal_leaders.end() && !month && !is_lower(following[0])) return s.substr(m.start, m.end - m.start);
            const Text ending = leader == data.ordinal_leaders.end() ? Text(U"n") : leader->second;
            return u(to_words(Num::of(digits_of(m.group(s, 0))), "de", true)) + ending;
        });
        // _german_year_pattern: a year 1100-1999 after a German month or another year word ("Oktober 1990", "im Jahr 1759")
        value = expand_cued_years(value, data.year_cues, false, [](const Text& s, std::size_t w) {
            return at(s, w) == '1' && at(s, w + 1) >= '1' && at(s, w + 1) <= '9' && is_decimal(at(s, w + 2)) && is_decimal(at(s, w + 3));
        }, german_year);
    }

    if (lang == "en") {
        // Multipliers and sizes: (?<![\w.,])(_EN_NUMBER)[xX×](\d+(?:\.\d+)?)?(?!\w)
        value = substitute(value, [](const Text& s, std::size_t p) -> std::optional<Match> {
            if (!is_decimal(s[p]) || (p > 0 && (is_word(s[p - 1]) || in_set(s[p - 1], U".,")))) return std::nullopt;
            std::size_t e = english_number_end(s, p);
            if (!in_set(at(s, e), U"xX×")) return std::nullopt;
            std::size_t q = e + 1;
            if (is_decimal(at(s, q))) {
                std::size_t whole = run_end(s, q, is_decimal), full = decimal_number_end(s, q);
                for (std::size_t end : {full, whole}) {
                    if (!is_word(at(s, end))) return Match{p, end, {{p, e}, {q, end}}};
                }
                return std::nullopt;
            }
            if (is_word(at(s, q))) return std::nullopt;
            return Match{p, q, {{p, e}, {npos, npos}}};
        }, [&](const Text& s, const Match& m) {
            return m.matched(1) ? m.group(s, 0) + U" by " + m.group(s, 1) : m.group(s, 0) + U" times";
        });
        // Plural numbers: (?<![\w'’.,])(['’]?)(_EN_NUMBER)(['’]?)s(?!\w)
        value = substitute(value, [](const Text& s, std::size_t p) -> std::optional<Match> {
            auto apostrophe = [](char32_t c) { return c == '\'' || c == 0x2019; };
            if (p > 0 && (is_word(s[p - 1]) || apostrophe(s[p - 1]) || in_set(s[p - 1], U".,"))) return std::nullopt;
            std::size_t q = apostrophe(s[p]) ? p + 1 : p;
            if (!is_decimal(at(s, q))) return std::nullopt;
            std::size_t e = english_number_end(s, q);
            std::size_t j = apostrophe(at(s, e)) ? e + 1 : e;
            if (at(s, j) != 's' || is_word(at(s, j + 1))) return std::nullopt;
            return Match{p, j + 1, {{p, q}, {q, e}, {e, j}}};
        }, [&](const Text& s, const Match& m) {
            Text original = s.substr(m.start, m.end - m.start), digits = m.group(s, 1);
            bool lead = !m.group(s, 0).empty(), apostrophe = !m.group(s, 2).empty();
            if (digits.find('.') != npos) return lead || apostrophe ? original : digits + U" seconds";
            Num number = Num::of(digits_of(digits));
            bool decade = (digits.size() == 2 || digits.size() == 4) && number.mod_pow10(1).zero();
            if (((lead || apostrophe) && !decade) || (lead && digits.size() != 2)) return original;
            if (digits.size() == 4 && !number.lt(1100) && number.lt(2100)) return u(plural_words(year_words_utf8(number)));
            return u(plural_number_words(number));
        });
    }
    if (data.one_before_noun.present) {
        // _ONE_BEFORE_NOUN_RE: (?<![\w.,\-])(\d+)(?![.,]\d)(\s+)([^\W\d_]+). "One" agrees with a noun written after it, before
        // letters glued to digits are split ("1K" is no noun).
        auto letter = [](char32_t c) { return c != 0 && c != '_' && is_word(c) && !is_decimal(c); };
        value = substitute(value, [&](const Text& s, std::size_t p) -> std::optional<Match> {
            if (!is_decimal(s[p]) || (p > 0 && (is_word(s[p - 1]) || in_set(s[p - 1], U".,-")))) return std::nullopt;
            std::size_t e = digits_end(s, p);
            if ((at(s, e) == '.' || at(s, e) == ',') && is_decimal(at(s, e + 1))) return std::nullopt;
            std::size_t q = run_end(s, e, is_space);
            if (q == e || !letter(at(s, q))) return std::nullopt;
            std::size_t end = run_end(s, q, letter);
            return Match{p, end, {{p, e}, {e, q}, {q, end}}};
        }, [&](const Text& s, const Match& m) { return one_before_noun(s, m, data.one_before_noun, lang); });
    }
    // Letters glued to digits: (?<!\w)[A-Za-z0-9]*(?:[A-Za-z][0-9]|[0-9][A-Za-z])[A-Za-z0-9]*(?!\w)
    value = substitute(value, [](const Text& s, std::size_t p) -> std::optional<Match> {
        auto alnum = [](char32_t c) { return is_ascii_letter(c) || is_ascii_digit(c); };
        if (!alnum(s[p]) || (p > 0 && is_word(s[p - 1]))) return std::nullopt;
        std::size_t e = run_end(s, p, alnum);
        if (is_word(at(s, e))) return std::nullopt;
        for (std::size_t k = p + 1; k < e; ++k) {
            if (is_ascii_digit(s[k]) != is_ascii_digit(s[k - 1])) return Match{p, e, {}};
        }
        return std::nullopt;
    }, [&](const Text& s, const Match& m) { return split_alphanumeric(s.substr(m.start, m.end - m.start), lang); });

    // Named-month dates (English only; other languages keep the text).
    value = substitute(value, [](const Text& s, std::size_t p) -> std::optional<Match> {
        if (!is_alnum(s[p]) || !word_boundary(s, p)) return std::nullopt;
        static const std::vector<std::vector<std::string_view>> alternatives = {
            {"january", "jan"}, {"february", "feb"}, {"march", "mar"}, {"april", "apr"}, {"may"}, {"june", "jun"}, {"july", "jul"},
            {"august", "aug"}, {"september", "sept", "sep"}, {"october", "oct"}, {"november", "nov"}, {"december", "dec"}};
        for (const auto& options : alternatives) {
            for (std::string_view name : options) {
                if (!ci_literal(s, p, name)) continue;
                std::size_t j = p + name.size();
                std::vector<std::size_t> dots;
                if (at(s, j) == '.') dots.push_back(j + 1);
                dots.push_back(j);
                for (std::size_t q : dots) {
                    std::size_t w = run_end(s, q, is_space);
                    if (w == q || !is_decimal(at(s, w))) continue;
                    std::vector<std::size_t> day_ends;
                    if (is_decimal(at(s, w + 1))) day_ends.push_back(w + 2);
                    day_ends.push_back(w + 1);
                    for (std::size_t day_end : day_ends) {
                        if (at(s, day_end) == ',') {
                            std::size_t y = run_end(s, day_end + 1, is_space);
                            std::size_t run = std::min<std::size_t>(digits_end(s, y) - y, 4);
                            for (std::size_t length = run; length >= 2; --length) {
                                if (word_boundary(s, y + length)) {
                                    return Match{p, y + length, {{p, j}, {w, day_end}, {y, y + length}}};
                                }
                            }
                        }
                        if (word_boundary(s, day_end)) return Match{p, day_end, {{p, j}, {w, day_end}, {npos, npos}}};
                    }
                }
            }
        }
        return std::nullopt;
    }, [&](const Text& s, const Match& m) {
        Text original = s.substr(m.start, m.end - m.start);
        if (lang != "en") return original;
        static const std::map<Text, int> aliases = {
            {U"jan", 1}, {U"january", 1}, {U"feb", 2}, {U"february", 2}, {U"mar", 3}, {U"march", 3}, {U"apr", 4}, {U"april", 4}, {U"may", 5},
            {U"jun", 6}, {U"june", 6}, {U"jul", 7}, {U"july", 7}, {U"aug", 8}, {U"august", 8}, {U"sep", 9}, {U"sept", 9}, {U"september", 9},
            {U"oct", 10}, {U"october", 10}, {U"nov", 11}, {U"november", 11}, {U"dec", 12}, {U"december", 12}};
        auto it = aliases.find(lower(m.group(s, 0)));
        if (it == aliases.end()) return original;
        Num day = Num::of(digits_of(m.group(s, 1)));
        if (day.lt(1) || !day.lt(32)) return original;
        Text out = month_name(it->second, "en") + U" " + u(to_words(day, lang, true));
        if (!m.matched(2)) return out;
        return out + U", " + year_words(Num::of(digits_of(m.group(s, 2))));
    });
    // \b(\d{1,2})/(\d{1,2})/(\d{2,4})\b
    value = substitute(value, [](const Text& s, std::size_t p) -> std::optional<Match> {
        if (!is_decimal(s[p]) || !word_boundary(s, p)) return std::nullopt;
        auto one_or_two = [&](std::size_t q) -> std::size_t {
            if (!is_decimal(at(s, q))) return npos;
            if (is_decimal(at(s, q + 1)) && at(s, q + 2) == '/') return q + 2;
            return at(s, q + 1) == '/' ? q + 1 : npos;
        };
        std::size_t first = one_or_two(p);
        if (first == npos) return std::nullopt;
        std::size_t second = one_or_two(first + 1);
        if (second == npos) return std::nullopt;
        std::size_t y = second + 1;
        std::size_t run = std::min<std::size_t>(digits_end(s, y) - y, 4);
        for (std::size_t length = run; length >= 2; --length) {
            if (word_boundary(s, y + length)) return Match{p, y + length, {{p, first}, {first + 1, second}, {y, y + length}}};
        }
        return std::nullopt;
    }, [&](const Text& s, const Match& m) {
        // _expand_slash_date: American month first, unless British or the first number can only be a day.
        const Num first = Num::of(digits_of(m.group(s, 0))), second = Num::of(digits_of(m.group(s, 1)));
        if (lang == "en" && !british && !(!first.lt(13) && second.lt(13))) return expand_date(m.group(s, 0), m.group(s, 1), m.group(s, 2), lang);
        return expand_date(m.group(s, 1), m.group(s, 0), m.group(s, 2), lang, lang == "en");
    });
    // \b(\d{4})-(\d{1,2})-(\d{1,2})\b
    value = substitute(value, [](const Text& s, std::size_t p) -> std::optional<Match> {
        if (!is_decimal(s[p]) || !word_boundary(s, p)) return std::nullopt;
        for (std::size_t k = 0; k < 4; ++k) {
            if (!is_decimal(at(s, p + k))) return std::nullopt;
        }
        if (at(s, p + 4) != '-' || !is_decimal(at(s, p + 5))) return std::nullopt;
        std::size_t month_end = is_decimal(at(s, p + 6)) && at(s, p + 7) == '-' ? p + 7 : at(s, p + 6) == '-' ? p + 6 : npos;
        if (month_end == npos || !is_decimal(at(s, month_end + 1))) return std::nullopt;
        std::size_t d = month_end + 1;
        for (std::size_t day_end : {d + 2, d + 1}) {
            if (day_end == d + 2 && !is_decimal(at(s, d + 1))) continue;
            if (word_boundary(s, day_end)) return Match{p, day_end, {{p, p + 4}, {p + 5, month_end}, {d, day_end}}};
        }
        return std::nullopt;
    }, [&](const Text& s, const Match& m) { return expand_date(m.group(s, 1), m.group(s, 2), m.group(s, 0), lang, british); });

    // Fractions: \b(\d+)\s+(\d+/\d+)\b, then \b(\d+)\s*/\s*(\d+)\b
    value = substitute(value, [](const Text& s, std::size_t p) -> std::optional<Match> {
        if (!is_decimal(s[p]) || !word_boundary(s, p)) return std::nullopt;
        std::size_t e = digits_end(s, p);
        std::size_t q = run_end(s, e, is_space);
        if (q == e || !is_decimal(at(s, q))) return std::nullopt;
        std::size_t slash = digits_end(s, q);
        if (at(s, slash) != '/' || !is_decimal(at(s, slash + 1))) return std::nullopt;
        std::size_t end = digits_end(s, slash + 1);
        if (!word_boundary(s, end)) return std::nullopt;
        return Match{p, end, {{p, e}, {q, end}}};
    }, [&](const Text& s, const Match& m) { return m.group(s, 0) + U" " + spoken_word(lang, "and") + U" " + m.group(s, 1); });
    value = substitute(value, [](const Text& s, std::size_t p) -> std::optional<Match> {
        if (!is_decimal(s[p]) || !word_boundary(s, p)) return std::nullopt;
        std::size_t e = digits_end(s, p);
        std::size_t slash = run_end(s, e, is_space);
        if (at(s, slash) != '/') return std::nullopt;
        std::size_t q = run_end(s, slash + 1, is_space);
        if (!is_decimal(at(s, q))) return std::nullopt;
        std::size_t end = digits_end(s, q);
        if (!word_boundary(s, end)) return std::nullopt;
        return Match{p, end, {{p, e}, {q, end}}};
    }, [&](const Text& s, const Match& m) { return expand_fraction(m.group(s, 0), m.group(s, 1), lang); });

    if (is_non_english(lang)) {
        // \b\d{1,3}(?:,\d{3}){2,}\b: two or more groups after commas are a whole number written the English way ("1,000,000")
        value = substitute(value, [](const Text& s, std::size_t p) { return grouped_number_at(s, p, ',', 2); },
                           [&](const Text& s, const Match& m) { return words(s.substr(m.start, m.end - m.start), lang); });
        // \b\d[\d.]*,\d+\b
        value = substitute(value, [](const Text& s, std::size_t p) -> std::optional<Match> {
            if (!is_decimal(s[p]) || !word_boundary(s, p)) return std::nullopt;
            std::size_t e = run_end(s, p + 1, [](char32_t c) { return is_decimal(c) || c == '.'; });
            if (at(s, e) != ',' || !is_decimal(at(s, e + 1))) return std::nullopt;
            std::size_t end = digits_end(s, e + 1);
            if (!word_boundary(s, end)) return std::nullopt;
            return Match{p, end, {}};
        }, [&](const Text& s, const Match& m) { return decimal_words(s.substr(m.start, m.end - m.start), lang); });
        // \b\d{1,3}(?:\.\d{3})+\b
        value = substitute(value, [](const Text& s, std::size_t p) { return grouped_number_at(s, p, '.'); },
                           [&](const Text& s, const Match& m) { return words(s.substr(m.start, m.end - m.start), lang); });
    }
    // \b\d[\d,]*\.\d+\b(?!\.\d)
    value = substitute(value, [](const Text& s, std::size_t p) -> std::optional<Match> {
        if (!is_decimal(s[p]) || !word_boundary(s, p)) return std::nullopt;
        std::size_t e = run_end(s, p + 1, [](char32_t c) { return is_decimal(c) || c == ','; });
        if (at(s, e) != '.' || !is_decimal(at(s, e + 1))) return std::nullopt;
        std::size_t end = digits_end(s, e + 1);
        if (!word_boundary(s, end) || (at(s, end) == '.' && is_decimal(at(s, end + 1)))) return std::nullopt;
        return Match{p, end, {}};
    }, [&](const Text& s, const Match& m) { return decimal_words(s.substr(m.start, m.end - m.start), lang); });
    // \b\d{1,3}(?:,\d{3})+\b
    value = substitute(value, [](const Text& s, std::size_t p) { return grouped_number_at(s, p, ','); },
                       [&](const Text& s, const Match& m) { return words(s.substr(m.start, m.end - m.start), lang); });
    // \b\d+\b
    value = substitute(value, [](const Text& s, std::size_t p) -> std::optional<Match> {
        if (!is_decimal(s[p]) || !word_boundary(s, p)) return std::nullopt;
        std::size_t e = digits_end(s, p);
        if (!word_boundary(s, e)) return std::nullopt;
        return Match{p, e, {}};
    }, [&](const Text& s, const Match& m) { return words(s.substr(m.start, m.end - m.start), lang); });

    // _WORD_SLASH_RE: (?<=[^\W\d_])\s?/\s?(?=[^\W\d_])
    value = substitute(value, [](const Text& s, std::size_t p) -> std::optional<Match> {
        std::size_t slash = is_space(s[p]) ? p + 1 : p;
        if (at(s, slash) != '/' || p == 0 || !letter_char(s[p - 1])) return std::nullopt;
        std::size_t e = is_space(at(s, slash + 1)) ? slash + 2 : slash + 1;
        if (!letter_char(at(s, e))) return std::nullopt;
        return Match{p, e, {}};
    }, [](const Text&, const Match&) { return Text(U" "); });

    // _expand_dash_letter_names
    auto dash_word = [](char32_t c) { return c != 0 && (is_alnum(c) || is_mark(c)); };
    value = substitute(value, [](const Text& s, std::size_t p) -> std::optional<Match> {  // ([A-Za-zÑñ])\s*-\s*
        if (!initial_letter(s[p])) return std::nullopt;
        std::size_t q = run_end(s, p + 1, is_space);
        if (at(s, q) != '-') return std::nullopt;
        return Match{p, run_end(s, q + 1, is_space), {{p, p + 1}}};
    }, [&](const Text& s, const Match& m) {
        char32_t before = m.start ? s[m.start - 1] : 0, after = at(s, m.end);
        if (before == '_' || dash_word(before) || !dash_word(after)) return s.substr(m.start, m.end - m.start);
        return letter_name(s[m.start], lang) + U" ";
    });
    value = substitute(value, [](const Text& s, std::size_t p) -> std::optional<Match> {  // \s*-\s*([A-Za-zÑñ])
        std::size_t q = run_end(s, p, is_space);
        if (at(s, q) != '-') return std::nullopt;
        std::size_t k = run_end(s, q + 1, is_space);
        if (!initial_letter(at(s, k))) return std::nullopt;
        return Match{p, k + 1, {{k, k + 1}}};
    }, [&](const Text& s, const Match& m) {
        char32_t before = m.start ? s[m.start - 1] : 0, after = at(s, m.end);
        if (!dash_word(before) || after == '_' || dash_word(after)) return s.substr(m.start, m.end - m.start);
        return U" " + letter_name(s[m.end - 1], lang);
    });
    value = replace_char(value, '-', U" ");
    if (capitals) value = upper(value);
    return collapse_spaces(value);
}

}  // namespace

std::string normalize_spoken_text(const std::string& text, const std::string& language) {
    return encode(normalize_text(decode(text), language));
}

// ---------------------------------------------------------------------------------------------------------------
// Non-verbal spellings and phones of the v5 models.

namespace {

// [A-Za-z'’\-‑]: what a typed variant may not touch on either side.
bool spelling_neighbour(char32_t cp) { return is_ascii_letter(cp) || cp == '\'' || cp == 0x2019 || cp == '-' || cp == 0x2011; }

bool spelling_hyphen(char32_t cp) { return cp == '-' || cp == 0x2011; }  // [‑\-]

// The end of `letters` at `pos` (npos when they are not there); `ignore_case` as re.IGNORECASE.
std::size_t spelling_letters(const Text& s, std::size_t pos, std::string_view letters, bool ignore_case = true) {
    if (pos == npos || pos + letters.size() > s.size()) return npos;
    for (std::size_t i = 0; i < letters.size(); ++i) {
        const char32_t cp = s[pos + i];
        if (ignore_case ? !ascii_ci_equal(cp, letters[i]) : cp != static_cast<char32_t>(letters[i])) return npos;
    }
    return pos + letters.size();
}

std::size_t spelling_one(const Text& s, std::size_t pos, bool (*accepts)(char32_t)) {
    return pos != npos && pos < s.size() && accepts(s[pos]) ? pos + 1 : npos;
}

bool literal_space(char32_t cp) { return cp == ' '; }

// (?:ah?[‑\-]ha[‑\-]ha|ah ha ha), ignoring case.
std::size_t spelling_ahaha(const Text& s, std::size_t pos) {
    for (const char* opening : {"ah", "a"}) {   // "ah?" is greedy: with the h first, then without it
        std::size_t end = spelling_one(s, spelling_letters(s, pos, opening), spelling_hyphen);
        end = spelling_one(s, spelling_letters(s, end, "ha"), spelling_hyphen);
        end = spelling_letters(s, end, "ha");
        if (end != npos) return end;
    }
    std::size_t end = spelling_one(s, spelling_letters(s, pos, "ah"), literal_space);
    end = spelling_one(s, spelling_letters(s, end, "ha"), literal_space);
    return spelling_letters(s, end, "ha");
}

// (?=\s*(?:[,.!?…;:]|$)): punctuation or the end of the text follows, after any spaces.
bool spelling_before_punctuation(const Text& s, std::size_t pos) {
    while (pos < s.size() && is_space(s[pos])) ++pos;
    return pos == s.size() || in_set(s[pos], U",.!?…;:");
}

// The rules of text_data/en.json "nonverbal_spellings", matched by hand (std::regex has neither lookbehind nor inline flags).
// `pattern` is the rule's regular expression as the table writes it: a rule whose expression changed, or a rule this
// list does not know, is an error, never skipped. `body` returns the end of the variant at a position; `follows` is a further
// condition on what comes after it.
struct SpellingMatcher {
    const char* id;
    const char* pattern;
    std::size_t (*body)(const Text& s, std::size_t pos);
    bool (*follows)(const Text& s, std::size_t end);
};

const SpellingMatcher kSpellingMatchers[] = {
    {"uhuh_joined", R"re((?i)(?<![A-Za-z'’\-‑])uhuh(?![A-Za-z'’\-‑]))re", [](const Text& s, std::size_t p) { return spelling_letters(s, p, "uhuh"); }, nullptr},
    {"nuhuh_joined", R"re((?i)(?<![A-Za-z'’\-‑])nuhuh(?![A-Za-z'’\-‑]))re", [](const Text& s, std::size_t p) { return spelling_letters(s, p, "nuhuh"); }, nullptr},
    {"hehheh_joined", R"re((?i)(?<![A-Za-z'’\-‑])hehheh(?![A-Za-z'’\-‑]))re", [](const Text& s, std::size_t p) { return spelling_letters(s, p, "hehheh"); }, nullptr},
    {"ahaha_parts", R"re((?i)(?<![A-Za-z'’\-‑])(?:ah?[‑\-]ha[‑\-]ha|ah ha ha)(?![A-Za-z'’\-‑]))re", spelling_ahaha, nullptr},
    {"ahem_hyphen", R"re((?i)(?<![A-Za-z'’\-‑])a[‑\-]hem(?![A-Za-z'’\-‑]))re",
     [](const Text& s, std::size_t p) { return spelling_letters(s, spelling_one(s, spelling_letters(s, p, "a"), spelling_hyphen), "hem"); }, nullptr},
    {"pst_short", R"re((?<![A-Za-z'’\-‑])[Pp]st(?![A-Za-z'’\-‑]))re",
     [](const Text& s, std::size_t p) { return spelling_letters(s, spelling_one(s, p, [](char32_t cp) { return cp == 'P' || cp == 'p'; }), "st", false); }, nullptr},
    {"sh_short", R"re((?<![A-Za-z'’\-‑])[Ss]h(?![A-Za-z'’\-‑])(?=\s*(?:[,.!?…;:]|$)))re",
     [](const Text& s, std::size_t p) { return spelling_letters(s, spelling_one(s, p, [](char32_t cp) { return cp == 'S' || cp == 's'; }), "h", false); },
     spelling_before_punctuation},
};

struct SpellingRule {
    const SpellingMatcher* matcher;
    Text spelling;
};

const Json& text_data_json(const std::string& lang) {
    static const std::map<std::string, Json> files = [] {
        std::map<std::string, Json> out;
        for (const TextDataFile& file : kTextData) out.emplace(file.language, Json::parse(file.json));
        return out;
    }();
    static const Json none;
    auto found = files.find(lang);
    return found == files.end() ? none : found->second;
}

const std::vector<SpellingRule>& spelling_rules(const std::string& lang) {
    static const std::map<std::string, std::vector<SpellingRule>> tables = [] {
        std::map<std::string, std::vector<SpellingRule>> out;
        for (const TextDataFile& file : kTextData) {
            std::vector<SpellingRule>& rules = out[file.language];
            for (const Json& rule : text_data_json(file.language).get("nonverbal_spellings").elements()) {
                const std::string id = rule.get("id").str();
                auto matcher = std::find_if(std::begin(kSpellingMatchers), std::end(kSpellingMatchers), [&](const SpellingMatcher& m) { return id == m.id; });
                if (matcher == std::end(kSpellingMatchers) || std::string(file.language) != kEnglish || rule.get("pattern").str() != matcher->pattern) {
                    throw std::logic_error("The non-verbal spelling rule '" + id + "' (" + file.language + ") is not one this runtime matches: " + rule.get("pattern").str());
                }
                rules.push_back(SpellingRule{matcher, u(rule.get("replace").str())});
            }
        }
        return out;
    }();
    static const std::vector<SpellingRule> none;
    auto found = tables.find(lang);
    return found == tables.end() ? none : found->second;
}

// The spelling in the case the variant was typed in: capitals stay capitals, a capital first letter is kept.
Text spelling_as_typed(const Text& typed, const Text& spelling) {
    const bool capitals = std::any_of(typed.begin(), typed.end(), is_upper) && std::none_of(typed.begin(), typed.end(), is_lower);  // str.isupper
    if (capitals && typed.size() > 1) return upper(spelling);
    if (!typed.empty() && is_upper(typed[0])) return upper(spelling.substr(0, 1)) + spelling.substr(1);
    return spelling;
}

}  // namespace

std::string nonverbal_spellings(const std::string& text, const std::string& language) {
    Text value = decode(text);
    for (const SpellingRule& rule : spelling_rules(normalizer_language(language))) {
        Text out;
        for (std::size_t pos = 0; pos < value.size();) {
            const std::size_t end = pos > 0 && spelling_neighbour(value[pos - 1]) ? npos : rule.matcher->body(value, pos);
            if (end == npos || (end < value.size() && spelling_neighbour(value[end])) || (rule.matcher->follows && !rule.matcher->follows(value, end))) {
                out.push_back(value[pos++]);
                continue;
            }
            out += spelling_as_typed(value.substr(pos, end - pos), rule.spelling);
            pos = end;
        }
        value = std::move(out);
    }
    return encode(value);
}

const NonverbalPhoneTable& nonverbal_phone_table() {
    static const NonverbalPhoneTable table = [] {
        NonverbalPhoneTable out;
        const Json& json = text_data_json(kEnglish).get("nonverbal_phones");
        for (const auto& [word, phones] : json.get("words").items()) out.words.emplace_back(word, phones.str());
        for (const auto& [phrase, word] : json.get("phrases").items()) out.phrases.emplace_back(phrase, word.str());
        for (const Json& locale : json.get("locales").elements()) out.locales.push_back(locale.str());
        return out;
    }();
    return table;
}

// ---------------------------------------------------------------------------------------------------------------
// G2P phrase punctuation and segmentation.

// g2p_phrases._WORD_HYPHEN_RE: a hyphen inside a word ("dix-sept", "well-known") is a space without a pause, as the text
// normalizer writes it: (?<=[^\W\d_])-(?=[^\W\d_])
Text word_hyphens_to_space(const Text& value) {
    auto letter = [](char32_t c) { return c != 0 && c != '_' && is_word(c) && !is_decimal(c); };
    Text out = value;
    for (std::size_t i = 1; i + 1 < value.size(); ++i) {
        if (value[i] == '-' && letter(value[i - 1]) && letter(value[i + 1])) out[i] = ' ';
    }
    return out;
}

Text normalize_phrase_punctuation(const Text& text) {
    Text value = hyphen_runs_to_pause(word_hyphens_to_space(dashes_to_pause(mark_tight_dashes(text))));
    Text folded;
    for (char32_t cp : value) {
        switch (cp) {
            case 0x00A0: case 0x1680: case 0x2000: case 0x2001: case 0x2002: case 0x2003: case 0x2004: case 0x2005: case 0x2006:
            case 0x2007: case 0x2008: case 0x2009: case 0x200A: case 0x202F: case 0x205F: case 0x3000:
                folded.push_back(' ');
                break;
            case 0x02BC: case 0x2018: case 0x2019: case 0x201A: case 0x201B: case 0xFF07:
                folded.push_back('\'');
                break;
            case 0x00AB: case 0x00BB: case 0x201C: case 0x201D: case 0x201E: case 0x201F: case 0xFF02:
                folded.push_back('"');
                break;
            case 0x2026: folded += U"..."; break;
            default: folded.push_back(cp);
        }
    }
    return collapse_spaces(fold_terminal_runs(folded));
}

std::string punctuation_run_phone_token(const Text& run) {
    if (run.empty()) return {};
    for (char32_t cp : run) {
        if (cp == '?') return "<end_question>";
        if (cp == '!') return "<end_exclaim>";
    }
    if (run.find(0x2026) != npos || std::count(run.begin(), run.end(), U'.') >= 2) return "<ellipsis>";
    if (run.find('.') != npos) return "<end_stmt>";
    if (run.find(',') != npos) return "<pause_comma>";
    if (run.find(';') != npos) return "<pause_semicolon>";
    if (run.find(':') != npos) return "<pause_colon>";
    if (run.find('-') != npos || run.find(0x2014) != npos || run.find(kTightDash) != npos) return "<pause_dash>";
    return {};
}

std::vector<std::pair<Text, std::vector<std::string>>> punctuated_segments(const Text& text, std::size_t max_chars) {
    std::vector<std::pair<Text, std::vector<std::string>>> out;
    for (auto& segment : marked_segments(text, max_chars)) out.emplace_back(std::move(segment.text), std::move(segment.tokens));
    return out;
}

std::vector<MarkedSegment> marked_segments(const Text& text, std::size_t max_chars) {
    static const std::u32string_view punctuation = U",.!?;:-…—\u2e3a";
    // _BRACKETED_NOTE_RE = \[[^\[\]\n]{1,80}\]
    Text value = substitute(normalize_phrase_punctuation(text), [](const Text& s, std::size_t p) -> std::optional<Match> {
        if (s[p] != '[') return std::nullopt;
        std::size_t e = run_end(s, p + 1, [](char32_t c) { return c != '[' && c != ']' && c != '\n'; });
        std::size_t length = e - p - 1;
        if (length < 1 || length > 80 || at(s, e) != ']') return std::nullopt;
        return Match{p, e + 1, {}};
    }, [](const Text&, const Match&) { return Text(U" "); });
    value = collapse_spaces(value);

    std::vector<MarkedSegment> out;
    Text current;
    std::size_t index = 0;
    while (index < value.size()) {
        if (in_set(value[index], punctuation)) {
            std::size_t end = run_end(value, index, [](char32_t c) { return in_set(c, punctuation); });
            const Text run = value.substr(index, end - index);
            std::string token = punctuation_run_phone_token(run);
            const bool tight = run.find(kTightDash) != npos;
            Text segment = strip(current);
            if (has_alnum(segment)) {
                out.push_back({segment, token.empty() ? std::vector<std::string>{} : std::vector<std::string>{token}, tight});
            } else if (!out.empty() && !token.empty()) {
                out.back().tokens.push_back(token);
                out.back().tight_dash = out.back().tight_dash || tight;
            }
            current.clear();
            index = end;
            continue;
        }
        current.push_back(value[index]);
        ++index;
    }
    if (has_alnum(current)) out.push_back({strip(current), {}, false});

    std::vector<MarkedSegment> result;
    static const char* const terminal_priority[] = {"<end_question>", "<end_exclaim>", "<ellipsis>", "<end_stmt>"};
    for (auto& [segment, tokens, tight] : out) {
        for (const char* terminal : terminal_priority) {
            if (std::find(tokens.begin(), tokens.end(), terminal) != tokens.end()) {
                tokens = {terminal};
                break;
            }
        }
        std::vector<Text> parts;
        if (segment.size() <= max_chars) {
            parts.push_back(segment);
        } else {
            Text part;
            for (const Text& word : split_whitespace(segment)) {
                Text candidate = part.empty() ? word : strip(part + U" " + word);
                if (!part.empty() && candidate.size() > max_chars) {
                    parts.push_back(part);
                    part = word;
                } else {
                    part = candidate;
                }
            }
            if (!part.empty()) parts.push_back(part);
        }
        for (std::size_t i = 0; i + 1 < parts.size(); ++i) result.push_back({parts[i], {}, false});
        result.push_back({parts.back(), tokens, tight});
    }
    return result;
}

// ---------------------------------------------------------------------------------------------------------------
// Planner text splitting.

std::vector<Text> split_sentences(const Text& text) {
    Text value = collapse_spaces(text);
    std::vector<Text> out;
    if (value.empty()) return out;
    // (?:[.!?…]+|\.\.\.)["'”’»)\]]*\s+
    auto terminal = [](char32_t c) { return c == '.' || c == '!' || c == '?' || c == 0x2026; };
    auto closer = [](char32_t c) { return in_set(c, U"\"'”’»)]"); };
    std::size_t start = 0, pos = 0;
    while (pos < value.size()) {
        if (!terminal(value[pos])) {
            ++pos;
            continue;
        }
        std::size_t e = run_end(value, run_end(value, pos, terminal), closer);
        std::size_t w = run_end(value, e, is_space);
        if (w == e) {
            ++pos;
            continue;
        }
        Text piece = strip(value.substr(start, w - start));
        if (has_alnum(piece)) {
            out.push_back(piece);
            start = w;
        }
        pos = w;
    }
    Text tail = strip(value.substr(start));
    if (has_alnum(tail)) out.push_back(tail);
    else if (!tail.empty() && !out.empty()) out.back() += U" " + tail;
    return out;
}

std::vector<Text> split_for_retry(const Text& text, double min_share) {
    Text value = strip(text);
    double middle = static_cast<double>(value.size()) / 2.0;
    const double least = min_share * static_cast<double>(value.size());
    std::vector<std::size_t> cuts;
    for (std::size_t p = 0; p < value.size(); ++p) {  // [,;:—-]\s+, leaving each piece at least min_share of the text
        if (!in_set(value[p], U",;:—-")) continue;
        std::size_t e = run_end(value, p + 1, is_space);
        const double end = static_cast<double>(e);
        if (e > p + 1 && e > 0 && e < value.size() && least <= end && end <= static_cast<double>(value.size()) - least) cuts.push_back(e);
        if (e > p + 1) p = e - 1;
    }
    if (cuts.empty()) {
        for (std::size_t p = 0; p < value.size(); ++p) {  // \s+
            if (!is_space(value[p])) continue;
            std::size_t e = run_end(value, p, is_space);
            cuts.push_back(e);
            p = e - 1;
        }
    }
    if (cuts.empty()) throw std::runtime_error("Sentence cannot be split further");
    std::size_t cut = cuts.front();
    for (std::size_t candidate : cuts) {
        if (std::fabs(static_cast<double>(candidate) - middle) < std::fabs(static_cast<double>(cut) - middle)) cut = candidate;
    }
    return {strip(value.substr(0, cut)), strip(value.substr(cut))};
}

std::vector<Text> split_paragraphs(const Text& text) {
    // re.split(r"\n\s*\n+", text), keeping pieces with content
    std::vector<Text> pieces;
    std::size_t start = 0, pos = 0;
    while (pos < text.size()) {
        if (text[pos] != '\n') {
            ++pos;
            continue;
        }
        std::size_t e = run_end(text, pos + 1, is_space);
        std::size_t last_newline = npos;
        for (std::size_t k = pos + 1; k < e; ++k) {
            if (text[k] == '\n') last_newline = k;
        }
        if (last_newline == npos) {
            ++pos;
            continue;
        }
        pieces.push_back(text.substr(start, pos - start));
        start = pos = last_newline + 1;
    }
    pieces.push_back(text.substr(start));
    std::vector<Text> out;
    for (Text& piece : pieces) {
        if (!strip(piece).empty()) out.push_back(std::move(piece));
    }
    return out;
}

}  // namespace scyllasband
