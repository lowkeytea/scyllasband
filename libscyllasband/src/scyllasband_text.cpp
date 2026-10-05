#include "scyllasband_text.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
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

std::string italian_number(const Num& n) {
    if (n.lt(20)) return kItUnder20[n.small()];
    if (n.lt(100)) {
        int64_t v = n.small(), rest = v % 10;
        return rest == 0 ? kItTens[v / 10] : std::string(kItTens[v / 10]) + " " + italian_number(Num::of(rest));
    }
    if (n.lt(1000)) {
        Num rest = n.mod_pow10(2), count = n.div_pow10(2);
        std::string head = count.eq(1) ? "cento" : italian_number(count) + " cento";
        return rest.zero() ? head : head + " " + italian_number(rest);
    }
    if (n.lt(1000000)) {
        Num rest = n.mod_pow10(3), count = n.div_pow10(3);
        std::string head = count.eq(1) ? "mille" : italian_number(count) + " mila";
        return rest.zero() ? head : head + " " + italian_number(rest);
    }
    Num rest = n.mod_pow10(6), count = n.div_pow10(6);
    std::string head = count.eq(1) ? "un milione" : italian_number(count) + " milioni";
    return rest.zero() ? head : head + " " + italian_number(rest);
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
            std::string head = scale.zeros == 3 && count.eq(1)
                ? std::string(scale.singular)
                : french_number(count) + " " + (count.eq(1) ? scale.singular : scale.plural);
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
    return cardinal(n, lang);
}

Text words(const Text& digits, const std::string& lang, bool ordinal = false) {
    return u(to_words(Num::of(digits_of(digits)), lang, ordinal));
}

const char* by_lang(const std::string& lang, const char* en, const char* es, const char* it, const char* fr, const char* de, const char* vi) {
    if (lang == "es") return es;
    if (lang == "it") return it;
    if (lang == "fr") return fr;
    if (lang == "de") return de;
    if (lang == "vi") return vi;
    return en;
}

const char* and_word(const std::string& lang) { return by_lang(lang, "and", "y", "e", "et", "und", "và"); }

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

const char* const* letter_names(const std::string& lang) {
    static const char* const en[] = {"ay", "bee", "see", "dee", "ee", "eff", "gee", "aitch", "eye", "jay", "kay", "ell", "em", "en", "oh",
                                     "pee", "cue", "are", "ess", "tee", "you", "vee", "double you", "ex", "why", "zee", nullptr};
    static const char* const es[] = {"a", "be", "ce", "de", "e", "efe", "ge", "hache", "i", "jota", "ka", "ele", "eme", "ene", "o", "pe",
                                     "cu", "erre", "ese", "te", "u", "uve", "doble uve", "equis", "ye", "zeta", "eñe"};
    static const char* const it[] = {"a", "bi", "ci", "di", "e", "effe", "gi", "acca", "i", "i lunga", "cappa", "elle", "emme", "enne", "o",
                                     "pi", "cu", "erre", "esse", "ti", "u", "vu", "doppia vu", "ics", "ipsilon", "zeta", nullptr};
    static const char* const fr[] = {"a", "bé", "cé", "dé", "e", "effe", "gé", "ache", "i", "ji", "ka", "elle", "emme", "enne", "o", "pé",
                                     "ku", "erre", "esse", "té", "u", "vé", "double vé", "iks", "i grec", "zède", nullptr};
    static const char* const de[] = {"a", "be", "tse", "de", "e", "eff", "ge", "ha", "i", "jot", "ka", "ell", "emm", "enn", "o", "pe",
                                     "ku", "err", "ess", "te", "u", "fau", "we", "iks", "ypsilon", "tset", nullptr};
    static const char* const vi[] = {"a", "bê", "xê", "dê", "e", "ép", "giê", "hát", "i", "giây", "ca", "e lờ", "e mờ", "e nờ", "o", "pê",
                                     "quy", "e rờ", "ét", "tê", "u", "vê", "vê kép", "ích", "i dài", "dét", nullptr};
    if (lang == "es") return es;
    if (lang == "it") return it;
    if (lang == "fr") return fr;
    if (lang == "de") return de;
    if (lang == "vi") return vi;
    return en;
}

Text letter_name(char32_t letter, const std::string& lang) {
    const char* const* names = letter_names(lang);
    char32_t upper = (letter >= 'a' && letter <= 'z') ? letter - 32 : (letter == 0xF1 ? 0xD1 : letter);
    if (upper >= 'A' && upper <= 'Z') return u(names[upper - 'A']);
    if (upper == 0xD1 && names[26] != nullptr) return u(names[26]);
    return Text(1, letter);
}

Text normalize_punctuation_step(const Text& input, const std::string& lang) {
    Text value;
    for (char32_t cp : input) {
        if (!((cp >= 0x200B && cp <= 0x200D) || cp == 0xFEFF)) value.push_back(cp);
    }
    // _normalize_dash_contract
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
    value = replace_char(value, '=', u(std::string(" ") + by_lang(lang, "equals", "igual", "uguale", "égal", "gleich", "bằng") + " "));
    value = replace_char(value, 0xB0, u(std::string(" ") + by_lang(lang, "degrees", "grados", "gradi", "degrés", "Grad", "độ") + " "));
    return collapse_spaces(value);
}

Text identifier_fragment(const Text& fragment, const std::string& lang) {
    Text out = replace_char(fragment, '+', u(std::string(" ") + by_lang(lang, "plus", "más", "più", "plus", "plus", "cộng") + " "));
    out = replace_char(out, '_', U" ");
    out = replace_char(out, '-', U" ");
    return replace_char(out, '.', u(std::string(" ") + by_lang(lang, "dot", "punto", "punto", "point", "Punkt", "chấm") + " "));
}

Text normalize_at_symbols(const Text& value, const std::string& lang) {
    const Text word = u(by_lang(lang, "at", "arroba", "chiocciola", "arobase", "at", "a còng"));
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
        return identifier_fragment(m.group(s, 0), lang) + U" " + word + U" " + identifier_fragment(m.group(s, 1), lang);
    });
    return replace_char(step, '@', U" " + word + U" ");
}

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
        // _SENTENCE_FOLLOWS_RE = \s*$|\s+["'“‘(\[]?[A-ZÀ-ÖØ-ÞÑ]  (matched at the initialism's end)
        std::size_t j = run_end(s, m.end, is_space);
        bool ends_sentence = j == s.size();
        if (!ends_sentence && j > m.end) {
            std::size_t k = in_set(s[j], U"\"'“‘([") ? j + 1 : j;
            ends_sentence = upper_latin(at(s, k)) || (k != j && upper_latin(at(s, j)));
        }
        return ends_sentence ? spoken + U"." : spoken;
    });
}

Text expand_title_abbreviations(const Text& value, const std::string& lang) {
    if (lang != "en") return value;
    // (?<![A-Za-zÀ-ÖØ-öø-ÿÑñ])([A-Za-z]{2,4})\.?\s+(?=[A-ZÀ-ÖØ-Þ])
    return substitute(value, [](const Text& s, std::size_t p) -> std::optional<Match> {
        if (p > 0 && latin_letter(s[p - 1])) return std::nullopt;
        for (std::size_t length = 4; length >= 2; --length) {
            if (p + length > s.size()) continue;
            bool letters = true;
            for (std::size_t i = p; i < p + length; ++i) letters = letters && is_ascii_letter(s[i]);
            if (!letters) continue;
            std::size_t j = p + length;
            std::vector<std::size_t> options;
            if (at(s, j) == '.') options.push_back(j + 1);
            options.push_back(j);
            for (std::size_t q : options) {
                std::size_t w = run_end(s, q, is_space);
                if (w > q && upper_latin(at(s, w))) return Match{p, w, {{p, p + length}}};
            }
        }
        return std::nullopt;
    }, [](const Text& s, const Match& m) {
        Text key = lower(m.group(s, 0));
        if (key == U"mrs") return Text(U"misses ");
        if (key == U"dr") return Text(U"doctor ");
        return s.substr(m.start, m.end - m.start);
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
        if (!grouped_integer(value, ',') && (is_non_english(lang) || currency)) separator = ',';
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

Text decimal_words(const Text& raw, const std::string& lang) {
    auto [whole, fraction] = split_decimal(raw, lang, false);
    Text out = words(whole, lang);
    if (!fraction) return out;
    out += U" " + u(by_lang(lang, "point", "coma", "virgola", "virgule", "Komma", "phẩy"));
    for (char32_t digit : *fraction) out += U" " + words(Text(1, digit), lang);
    return out;
}

Text currency_words(const Text& raw, char32_t symbol, const std::string& lang) {
    struct Units {
        const char *major, *majors, *minor, *minors;
    };
    auto units_for = [&]() -> Units {
        int column = symbol == 0xA3 ? 1 : symbol == 0x20AC ? 2 : 0;  // $, £, €
        static const std::map<std::string, std::vector<Units>> table = {
            {"en", {{"dollar", "dollars", "cent", "cents"}, {"pound", "pounds", "penny", "pence"}, {"euro", "euros", "cent", "cents"}}},
            {"es", {{"dólar", "dólares", "centavo", "centavos"}, {"libra", "libras", "penique", "peniques"}, {"euro", "euros", "céntimo", "céntimos"}}},
            {"it", {{"dollaro", "dollari", "cent", "cent"}, {"sterlina", "sterline", "penny", "pence"}, {"euro", "euro", "centesimo", "centesimi"}}},
            {"fr", {{"dollar", "dollars", "centime", "centimes"}, {"livre", "livres", "penny", "pence"}, {"euro", "euros", "centime", "centimes"}}},
            {"de", {{"Dollar", "Dollar", "Cent", "Cent"}, {"Pfund", "Pfund", "Penny", "Pence"}, {"Euro", "Euro", "Cent", "Cent"}}},
            {"vi", {{"đô la", "đô la", "xu", "xu"}, {"bảng Anh", "bảng Anh", "xu", "xu"}, {"euro", "euro", "xu", "xu"}}}};
        auto it = table.find(lang);
        return (it == table.end() ? table.at("en") : it->second)[static_cast<std::size_t>(column)];
    };
    Units units = units_for();
    auto [whole, cents] = split_decimal(raw, lang, true);
    Num major = Num::of(digits_of(whole));
    Text major_words = u(to_words(major, lang) + " " + (major.eq(1) ? units.major : units.majors));
    if (!cents) return major_words;
    std::string padded = digits_of(*cents) + "00";
    Num minor = Num::of(padded.substr(0, 2));
    if (minor.zero()) return major_words;
    return major_words + U" " + u(and_word(lang)) + U" " + u(to_words(minor, lang) + " " + (minor.eq(1) ? units.minor : units.minors));
}

Text month_name(int month, const std::string& lang) {
    static const std::map<std::string, std::vector<const char*>> months = {
        {"en", {"January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"}},
        {"es", {"enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"}},
        {"it", {"gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"}},
        {"fr", {"janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"}},
        {"de", {"Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober", "November", "Dezember"}},
        {"vi", {"tháng một", "tháng hai", "tháng ba", "tháng tư", "tháng năm", "tháng sáu", "tháng bảy", "tháng tám", "tháng chín",
                "tháng mười", "tháng mười một", "tháng mười hai"}}};
    auto it = months.find(lang);
    return u((it == months.end() ? months.at("en") : it->second)[static_cast<std::size_t>(month - 1)]);
}

Text expand_date(const Text& month_s, const Text& day_s, const Text& year_s, const std::string& lang) {
    Num month = Num::of(digits_of(month_s)), day = Num::of(digits_of(day_s)), year = Num::of(digits_of(year_s));
    if (month.lt(1) || !month.lt(13) || day.lt(1) || !day.lt(32)) return month_s + U"/" + day_s + U"/" + year_s;
    Text name = month_name(static_cast<int>(month.small()), lang);
    if (lang == "en") return name + U" " + u(to_words(day, lang, true)) + U", " + u(to_words(year, lang));
    if (lang == "es") return u(day.d) + U" de " + name + U" de " + u(year.d);
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
    if (lang == "vi") return u(head + " phần " + to_words(denominator, lang));
    return u(head + " " + by_lang(lang, "over", "sobre", "su", "sur", "durch", "over") + " " + to_words(denominator, lang));
}

// Greedy [0-9](?:[0-9.,]*[0-9])? from p (p is an ASCII digit): end of the longest run that ends in a digit.
std::size_t amount_end(const Text& s, std::size_t p) {
    std::size_t e = run_end(s, p, [](char32_t c) { return is_ascii_digit(c) || c == '.' || c == ','; });
    while (e > p + 1 && !is_ascii_digit(s[e - 1])) --e;
    return e;
}

std::size_t digits_end(const Text& s, std::size_t p) { return run_end(s, p, is_decimal); }

// \b\d{1,3}(?:SEP\d{3})+\b at p
std::optional<Match> grouped_number_at(const Text& s, std::size_t p, char32_t separator) {
    if (!is_decimal(s[p]) || !word_boundary(s, p)) return std::nullopt;
    std::size_t run = digits_end(s, p) - p;
    if (run > 3) return std::nullopt;
    std::vector<std::size_t> ends;
    std::size_t pos = p + run;
    while (at(s, pos) == separator && pos + 4 <= s.size() && is_decimal(s[pos + 1]) && is_decimal(s[pos + 2]) && is_decimal(s[pos + 3])) {
        pos += 4;
        ends.push_back(pos);
    }
    for (auto it = ends.rbegin(); it != ends.rend(); ++it) {
        if (word_boundary(s, *it)) return Match{p, *it, {}};
    }
    return std::nullopt;
}

Text normalize_text(const Text& text, const std::string& language) {
    const std::string lang = normalizer_language(language);
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
    value = expand_dotted_initialisms(value, lang);
    value = expand_title_abbreviations(value, lang);

    // Currency: ([$£€])\s*([0-9](?:[0-9.,]*[0-9])?)
    value = substitute(value, [](const Text& s, std::size_t p) -> std::optional<Match> {
        if (s[p] != '$' && s[p] != 0xA3 && s[p] != 0x20AC) return std::nullopt;
        std::size_t q = run_end(s, p + 1, is_space);
        if (!is_ascii_digit(at(s, q))) return std::nullopt;
        std::size_t e = amount_end(s, q);
        return Match{p, e, {{p, p + 1}, {q, e}}};
    }, [&](const Text& s, const Match& m) { return currency_words(m.group(s, 1), s[m.start], lang); });
    // (?<![\w.,])([0-9](?:[0-9.,]*[0-9])?)\s*(€|EUR)(?!\w), IGNORECASE
    value = substitute(value, [](const Text& s, std::size_t p) -> std::optional<Match> {
        if (!is_ascii_digit(s[p]) || (p > 0 && (is_word(s[p - 1]) || s[p - 1] == '.' || s[p - 1] == ','))) return std::nullopt;
        std::size_t e = amount_end(s, p);
        std::size_t q = run_end(s, e, is_space);
        std::size_t symbol_end = at(s, q) == 0x20AC ? q + 1 : ci_literal(s, q, "eur") ? q + 3 : npos;
        if (symbol_end == npos || is_word(at(s, symbol_end))) return std::nullopt;
        return Match{p, symbol_end, {{p, e}}};
    }, [&](const Text& s, const Match& m) { return currency_words(m.group(s, 0), 0x20AC, lang); });

    // Times: \b(\d{1,2}):(\d{2})(?:\s*([AaPp][Mm]))?\b
    value = substitute(value, [](const Text& s, std::size_t p) -> std::optional<Match> {
        if (!is_decimal(s[p]) || !word_boundary(s, p)) return std::nullopt;
        std::size_t colon = is_decimal(at(s, p + 1)) && at(s, p + 2) == ':' ? p + 2 : at(s, p + 1) == ':' ? p + 1 : npos;
        if (colon == npos || !is_decimal(at(s, colon + 1)) || !is_decimal(at(s, colon + 2))) return std::nullopt;
        std::size_t minutes_end = colon + 3;
        std::size_t q = run_end(s, minutes_end, is_space);
        if (in_set(at(s, q), U"AaPp") && in_set(at(s, q + 1), U"Mm") && word_boundary(s, q + 2)) {
            return Match{p, q + 2, {{p, colon}, {colon + 1, minutes_end}, {q, q + 2}}};
        }
        if (!word_boundary(s, minutes_end)) return std::nullopt;
        return Match{p, minutes_end, {{p, colon}, {colon + 1, minutes_end}, {npos, npos}}};
    }, [&](const Text& s, const Match& m) {
        Num minute = Num::of(digits_of(m.group(s, 1)));
        Text minute_words;
        if (minute.zero() && lang == "en") minute_words = U"o'clock";
        else if (minute.lt(10) && lang == "en") minute_words = U"oh " + u(to_words(minute, lang));
        else minute_words = u(to_words(minute, lang));
        Text ampm = m.group(s, 2);
        for (char32_t& c : ampm) c = (c >= 'a' && c <= 'z') ? c - 32 : c;
        return strip(words(m.group(s, 0), lang) + U" " + minute_words + U" " + ampm);
    });
    // Percentages: (?<![\d.,])(\d[\d.,]*)%
    value = substitute(value, [](const Text& s, std::size_t p) -> std::optional<Match> {
        if (!is_decimal(s[p]) || (p > 0 && (is_decimal(s[p - 1]) || s[p - 1] == '.' || s[p - 1] == ','))) return std::nullopt;
        std::size_t e = run_end(s, p, [](char32_t c) { return is_decimal(c) || c == '.' || c == ','; });
        if (at(s, e) != '%') return std::nullopt;
        return Match{p, e + 1, {{p, e}}};
    }, [&](const Text& s, const Match& m) {
        return decimal_words(m.group(s, 0), lang) + U" " + u(by_lang(lang, "percent", "por ciento", "per cento", "pour cent", "Prozent", "phần trăm"));
    });
    // Ordinals: \b(\d+)(st|nd|rd|th|o|a)\b (IGNORECASE), then \b(\d+)(?:º|ª)\b
    value = substitute(value, [](const Text& s, std::size_t p) -> std::optional<Match> {
        if (!is_decimal(s[p]) || !word_boundary(s, p)) return std::nullopt;
        std::size_t e = digits_end(s, p);
        for (std::string_view suffix : {"st", "nd", "rd", "th", "o", "a"}) {
            if (ci_literal(s, e, suffix) && word_boundary(s, e + suffix.size())) return Match{p, e + suffix.size(), {{p, e}}};
        }
        return std::nullopt;
    }, [&](const Text& s, const Match& m) { return words(m.group(s, 0), lang, true); });
    value = substitute(value, [](const Text& s, std::size_t p) -> std::optional<Match> {
        if (!is_decimal(s[p]) || !word_boundary(s, p)) return std::nullopt;
        std::size_t e = digits_end(s, p);
        if ((at(s, e) != 0xBA && at(s, e) != 0xAA) || !word_boundary(s, e + 1)) return std::nullopt;
        return Match{p, e + 1, {{p, e}}};
    }, [&](const Text& s, const Match& m) { return words(m.group(s, 0), lang, true); });

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
        return out + U", " + words(m.group(s, 2), lang);
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
        if (lang == "en") return expand_date(m.group(s, 0), m.group(s, 1), m.group(s, 2), lang);
        return expand_date(m.group(s, 1), m.group(s, 0), m.group(s, 2), lang);
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
    }, [&](const Text& s, const Match& m) { return expand_date(m.group(s, 1), m.group(s, 2), m.group(s, 0), lang); });

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
    }, [&](const Text& s, const Match& m) { return m.group(s, 0) + U" " + u(and_word(lang)) + U" " + m.group(s, 1); });
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
    return collapse_spaces(replace_char(value, '-', U" "));
}

}  // namespace

std::string normalize_spoken_text(const std::string& text, const std::string& language) {
    return encode(normalize_text(decode(text), language));
}

// ---------------------------------------------------------------------------------------------------------------
// G2P phrase punctuation and segmentation.

Text normalize_phrase_punctuation(const Text& text) {
    Text value = hyphen_runs_to_pause(dashes_to_pause(text));
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
    if (run.find('-') != npos || run.find(0x2014) != npos) return "<pause_dash>";
    return {};
}

std::vector<std::pair<Text, std::vector<std::string>>> punctuated_segments(const Text& text, std::size_t max_chars) {
    static const std::u32string_view punctuation = U",.!?;:-…—";
    // _BRACKETED_NOTE_RE = \[[^\[\]\n]{1,80}\]
    Text value = substitute(normalize_phrase_punctuation(text), [](const Text& s, std::size_t p) -> std::optional<Match> {
        if (s[p] != '[') return std::nullopt;
        std::size_t e = run_end(s, p + 1, [](char32_t c) { return c != '[' && c != ']' && c != '\n'; });
        std::size_t length = e - p - 1;
        if (length < 1 || length > 80 || at(s, e) != ']') return std::nullopt;
        return Match{p, e + 1, {}};
    }, [](const Text&, const Match&) { return Text(U" "); });
    value = collapse_spaces(value);

    std::vector<std::pair<Text, std::vector<std::string>>> out;
    Text current;
    std::size_t index = 0;
    while (index < value.size()) {
        if (in_set(value[index], punctuation)) {
            std::size_t end = run_end(value, index, [](char32_t c) { return in_set(c, punctuation); });
            std::string token = punctuation_run_phone_token(value.substr(index, end - index));
            Text segment = strip(current);
            if (has_alnum(segment)) {
                out.emplace_back(segment, token.empty() ? std::vector<std::string>{} : std::vector<std::string>{token});
            } else if (!out.empty() && !token.empty()) {
                out.back().second.push_back(token);
            }
            current.clear();
            index = end;
            continue;
        }
        current.push_back(value[index]);
        ++index;
    }
    if (has_alnum(current)) out.emplace_back(strip(current), std::vector<std::string>{});

    std::vector<std::pair<Text, std::vector<std::string>>> result;
    static const char* const terminal_priority[] = {"<end_question>", "<end_exclaim>", "<ellipsis>", "<end_stmt>"};
    for (auto& [segment, tokens] : out) {
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
        for (std::size_t i = 0; i + 1 < parts.size(); ++i) result.emplace_back(parts[i], std::vector<std::string>{});
        result.emplace_back(parts.back(), tokens);
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
