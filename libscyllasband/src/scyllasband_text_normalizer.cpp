#include "scyllasband_text_normalizer.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <regex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace scyllasband_detail {
namespace {

using RegexMatch = std::smatch;

std::string trim(const std::string& value) {
    std::size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) {
        ++start;
    }
    std::size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    return value.substr(start, end - start);
}

std::string collapse_ws(const std::string& value) {
    std::ostringstream out;
    bool pending_space = false;
    for (unsigned char ch : value) {
        if (std::isspace(ch)) {
            pending_space = out.tellp() > 0;
            continue;
        }
        if (pending_space) {
            out << ' ';
            pending_space = false;
        }
        out << static_cast<char>(ch);
    }
    return trim(out.str());
}

void replace_all(std::string& value, const std::string& from, const std::string& to) {
    if (from.empty()) {
        return;
    }
    std::size_t position = 0;
    while ((position = value.find(from, position)) != std::string::npos) {
        value.replace(position, from.size(), to);
        position += to.size();
    }
}

template <typename Replacer>
std::string replace_regex(
    const std::string& value,
    const std::regex& pattern,
    Replacer replacer
) {
    std::string out;
    std::size_t cursor = 0;
    for (std::sregex_iterator iterator(value.begin(), value.end(), pattern), end;
         iterator != end;
         ++iterator) {
        const RegexMatch& match = *iterator;
        const std::size_t position = static_cast<std::size_t>(match.position());
        out.append(value, cursor, position - cursor);
        out += replacer(match);
        cursor = position + static_cast<std::size_t>(match.length());
    }
    out.append(value, cursor, std::string::npos);
    return out;
}

std::string normalizer_language(std::string language) {
    std::transform(language.begin(), language.end(), language.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    std::replace(language.begin(), language.end(), '-', '_');
    if (language == "es" || language == "it") {
        return language;
    }
    return "en";
}

std::string english_number(int64_t number) {
    static const std::vector<std::string> under_twenty = {
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen",
    };
    static const std::vector<std::string> tens = {
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    };
    if (number < 0) {
        return "minus " + english_number(-number);
    }
    if (number < 20) {
        return under_twenty[static_cast<std::size_t>(number)];
    }
    if (number < 100) {
        const int64_t rest = number % 10;
        const std::string head = tens[static_cast<std::size_t>(number / 10)];
        return rest == 0 ? head : head + " " + english_number(rest);
    }
    if (number < 1000) {
        const int64_t rest = number % 100;
        const std::string head = english_number(number / 100) + " hundred";
        return rest == 0 ? head : head + " " + english_number(rest);
    }
    for (const auto& scale : std::vector<std::pair<int64_t, const char*>>{
             {1'000'000'000'000LL, "trillion"},
             {1'000'000'000LL, "billion"},
             {1'000'000LL, "million"},
             {1'000LL, "thousand"},
         }) {
        if (number >= scale.first) {
            const int64_t rest = number % scale.first;
            const std::string head = english_number(number / scale.first) + " " + scale.second;
            return rest == 0 ? head : head + " " + english_number(rest);
        }
    }
    return std::to_string(number);
}

std::string spanish_number(int64_t number) {
    static const std::vector<std::string> under_thirty = {
        "cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve",
        "diez", "once", "doce", "trece", "catorce", "quince", "dieciséis", "diecisiete",
        "dieciocho", "diecinueve", "veinte", "veintiuno", "veintidós", "veintitrés",
        "veinticuatro", "veinticinco", "veintiséis", "veintisiete", "veintiocho", "veintinueve",
    };
    static const std::vector<std::string> tens = {
        "", "", "", "treinta", "cuarenta", "cincuenta", "sesenta", "setenta", "ochenta", "noventa",
    };
    static const std::vector<std::string> hundreds = {
        "", "cien", "doscientos", "trescientos", "cuatrocientos", "quinientos",
        "seiscientos", "setecientos", "ochocientos", "novecientos",
    };
    if (number < 0) {
        return "menos " + spanish_number(-number);
    }
    if (number < 30) {
        return under_thirty[static_cast<std::size_t>(number)];
    }
    if (number < 100) {
        const int64_t rest = number % 10;
        const std::string head = tens[static_cast<std::size_t>(number / 10)];
        return rest == 0 ? head : head + " y " + spanish_number(rest);
    }
    if (number < 1000) {
        const int64_t rest = number % 100;
        std::string head;
        if (number < 200 && number != 100) {
            head = "ciento";
        } else {
            head = hundreds[static_cast<std::size_t>(number / 100)];
        }
        return rest == 0 ? head : head + " " + spanish_number(rest);
    }
    if (number < 1'000'000) {
        const int64_t rest = number % 1000;
        const std::string head = number / 1000 == 1 ? "mil" : spanish_number(number / 1000) + " mil";
        return rest == 0 ? head : head + " " + spanish_number(rest);
    }
    if (number < 1'000'000'000'000LL) {
        const int64_t rest = number % 1'000'000;
        const std::string head = number / 1'000'000 == 1
            ? "un millón"
            : spanish_number(number / 1'000'000) + " millones";
        return rest == 0 ? head : head + " " + spanish_number(rest);
    }
    return std::to_string(number);
}

std::string italian_number(int64_t number) {
    static const std::vector<std::string> under_twenty = {
        "zero", "uno", "due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove",
        "dieci", "undici", "dodici", "tredici", "quattordici", "quindici", "sedici",
        "diciassette", "diciotto", "diciannove",
    };
    static const std::vector<std::string> tens = {
        "", "", "venti", "trenta", "quaranta", "cinquanta", "sessanta", "settanta", "ottanta", "novanta",
    };
    if (number < 0) {
        return "meno " + italian_number(-number);
    }
    if (number < 20) {
        return under_twenty[static_cast<std::size_t>(number)];
    }
    if (number < 100) {
        const int64_t rest = number % 10;
        const std::string head = tens[static_cast<std::size_t>(number / 10)];
        return rest == 0 ? head : head + " " + italian_number(rest);
    }
    if (number < 1000) {
        const int64_t rest = number % 100;
        const std::string head = number / 100 == 1 ? "cento" : italian_number(number / 100) + " cento";
        return rest == 0 ? head : head + " " + italian_number(rest);
    }
    if (number < 1'000'000) {
        const int64_t rest = number % 1000;
        const std::string head = number / 1000 == 1 ? "mille" : italian_number(number / 1000) + " mila";
        return rest == 0 ? head : head + " " + italian_number(rest);
    }
    if (number < 1'000'000'000'000LL) {
        const int64_t rest = number % 1'000'000;
        const std::string head = number / 1'000'000 == 1
            ? "un milione"
            : italian_number(number / 1'000'000) + " milioni";
        return rest == 0 ? head : head + " " + italian_number(rest);
    }
    return std::to_string(number);
}

std::string cardinal(int64_t value, const std::string& language) {
    if (language == "es") {
        return spanish_number(value);
    }
    if (language == "it") {
        return italian_number(value);
    }
    return english_number(value);
}

std::string ordinal(int64_t value, const std::string& language) {
    static const std::vector<std::string> english_ordinals = {
        "", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
        "ninth", "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth",
        "sixteenth", "seventeenth", "eighteenth", "nineteenth", "twentieth", "twenty first",
        "twenty second", "twenty third", "twenty fourth", "twenty fifth", "twenty sixth",
        "twenty seventh", "twenty eighth", "twenty ninth", "thirtieth", "thirty first",
    };
    if (language == "en" && value > 0 && value < static_cast<int64_t>(english_ordinals.size())) {
        return english_ordinals[static_cast<std::size_t>(value)];
    }
    return cardinal(value, language);
}

int64_t parse_digits(const std::string& value) {
    std::string digits;
    for (unsigned char ch : value) {
        if (std::isdigit(ch)) {
            digits.push_back(static_cast<char>(ch));
        }
    }
    if (digits.empty()) {
        return 0;
    }
    try {
        return std::stoll(digits);
    } catch (...) {
        return 0;
    }
}

bool grouped_integer(const std::string& value, char separator) {
    const std::size_t first = value.find(separator);
    if (first == std::string::npos || first < 1 || first > 3) {
        return false;
    }
    std::size_t position = first;
    while (position < value.size()) {
        if (value[position] != separator || position + 4 > value.size()) {
            return false;
        }
        for (std::size_t offset = 1; offset <= 3; ++offset) {
            if (!std::isdigit(static_cast<unsigned char>(value[position + offset]))) {
                return false;
            }
        }
        position += 4;
    }
    return position == value.size();
}

std::pair<std::string, std::string> split_decimal(
    const std::string& raw,
    const std::string& language,
    bool currency
) {
    std::string value;
    for (unsigned char ch : raw) {
        if (!std::isspace(ch)) {
            value.push_back(static_cast<char>(ch));
        }
    }
    const std::size_t comma = value.rfind(',');
    const std::size_t dot = value.rfind('.');
    char separator = '\0';
    if (comma != std::string::npos && dot != std::string::npos) {
        separator = comma > dot ? ',' : '.';
    } else if (comma != std::string::npos) {
        if (!grouped_integer(value, ',') && (language == "es" || language == "it" || currency)) {
            separator = ',';
        }
    } else if (dot != std::string::npos) {
        const std::size_t fractional_length = value.size() - dot - 1;
        if (!grouped_integer(value, '.') &&
            (language == "en" || currency || fractional_length != 3)) {
            separator = '.';
        }
    }
    if (separator == '\0') {
        return {value, std::string()};
    }
    const std::size_t position = value.rfind(separator);
    std::string fraction;
    for (std::size_t index = position + 1; index < value.size(); ++index) {
        if (std::isdigit(static_cast<unsigned char>(value[index]))) {
            fraction.push_back(value[index]);
        }
    }
    return {value.substr(0, position), fraction};
}

std::string decimal_words(const std::string& raw, const std::string& language) {
    const auto parts = split_decimal(raw, language, false);
    const std::string whole = cardinal(parse_digits(parts.first), language);
    if (parts.second.empty()) {
        return whole;
    }
    const std::string point = language == "es" ? "coma" : language == "it" ? "virgola" : "point";
    std::string out = whole + " " + point;
    for (char digit : parts.second) {
        out += " " + cardinal(digit - '0', language);
    }
    return out;
}

std::string and_word(const std::string& language) {
    return language == "es" ? "y" : language == "it" ? "e" : "and";
}

std::string letter_name(char letter, const std::string& language) {
    static const std::vector<std::string> english = {
        "ay", "bee", "see", "dee", "ee", "eff", "gee", "aitch", "eye", "jay", "kay", "ell", "em",
        "en", "oh", "pee", "cue", "are", "ess", "tee", "you", "vee", "double you", "ex", "why", "zee",
    };
    static const std::vector<std::string> spanish = {
        "a", "be", "ce", "de", "e", "efe", "ge", "hache", "i", "jota", "ka", "ele", "eme",
        "ene", "o", "pe", "cu", "erre", "ese", "te", "u", "uve", "doble uve", "equis", "ye", "zeta",
    };
    static const std::vector<std::string> italian = {
        "a", "bi", "ci", "di", "e", "effe", "gi", "acca", "i", "i lunga", "cappa", "elle", "emme",
        "enne", "o", "pi", "cu", "erre", "esse", "ti", "u", "vu", "doppia vu", "ics", "ipsilon", "zeta",
    };
    const unsigned char upper = static_cast<unsigned char>(std::toupper(static_cast<unsigned char>(letter)));
    if (upper < 'A' || upper > 'Z') {
        return std::string(1, letter);
    }
    const auto& names = language == "es" ? spanish : language == "it" ? italian : english;
    return names[static_cast<std::size_t>(upper - 'A')];
}

std::string expand_dotted_initialisms(const std::string& value, const std::string& language) {
    static const std::regex pattern(R"(\b((?:[A-Za-z]\.){2,}))");
    return replace_regex(value, pattern, [&](const RegexMatch& match) {
        std::string out;
        for (char ch : match[1].str()) {
            if (!std::isalpha(static_cast<unsigned char>(ch))) {
                continue;
            }
            if (!out.empty()) {
                out.push_back(' ');
            }
            out += letter_name(ch, language);
        }
        return out;
    });
}

std::string expand_dash_letter_names(const std::string& value, const std::string& language) {
    static const std::regex left_pattern(R"(\b([A-Za-z])\s*-\s*(?=[A-Za-z0-9]))");
    std::string out = replace_regex(value, left_pattern, [&](const RegexMatch& match) {
        return letter_name(match[1].str()[0], language) + " ";
    });
    static const std::regex right_pattern(R"(([A-Za-z0-9])\s*-\s*([A-Za-z])\b)");
    return replace_regex(out, right_pattern, [&](const RegexMatch& match) {
        return match[1].str() + " " + letter_name(match[2].str()[0], language);
    });
}

std::string currency_words(
    const std::string& raw,
    const std::string& symbol,
    const std::string& language
) {
    const auto parts = split_decimal(raw, language, true);
    const int64_t major = parse_digits(parts.first);
    std::string major_singular;
    std::string major_plural;
    std::string minor_singular;
    std::string minor_plural;
    if (language == "es") {
        if (symbol == "£") {
            major_singular = "libra"; major_plural = "libras"; minor_singular = "penique"; minor_plural = "peniques";
        } else if (symbol == "€") {
            major_singular = "euro"; major_plural = "euros"; minor_singular = "céntimo"; minor_plural = "céntimos";
        } else {
            major_singular = "dólar"; major_plural = "dólares"; minor_singular = "centavo"; minor_plural = "centavos";
        }
    } else if (language == "it") {
        if (symbol == "£") {
            major_singular = "sterlina"; major_plural = "sterline"; minor_singular = "penny"; minor_plural = "pence";
        } else if (symbol == "€") {
            major_singular = "euro"; major_plural = "euro"; minor_singular = "centesimo"; minor_plural = "centesimi";
        } else {
            major_singular = "dollaro"; major_plural = "dollari"; minor_singular = "cent"; minor_plural = "cent";
        }
    } else if (symbol == "£") {
        major_singular = "pound"; major_plural = "pounds"; minor_singular = "penny"; minor_plural = "pence";
    } else if (symbol == "€") {
        major_singular = "euro"; major_plural = "euros"; minor_singular = "cent"; minor_plural = "cents";
    } else {
        major_singular = "dollar"; major_plural = "dollars"; minor_singular = "cent"; minor_plural = "cents";
    }
    std::string out = cardinal(major, language) + " " + (major == 1 ? major_singular : major_plural);
    if (!parts.second.empty()) {
        const int64_t minor = parse_digits((parts.second + "00").substr(0, 2));
        if (minor > 0) {
            out += " " + and_word(language) + " " + cardinal(minor, language) + " " +
                (minor == 1 ? minor_singular : minor_plural);
        }
    }
    return out;
}

std::string month_name(int month, const std::string& language) {
    static const std::vector<std::string> english = {
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    };
    static const std::vector<std::string> spanish = {
        "enero", "febrero", "marzo", "abril", "mayo", "junio",
        "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
    };
    static const std::vector<std::string> italian = {
        "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
        "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre",
    };
    const auto& names = language == "es" ? spanish : language == "it" ? italian : english;
    return month >= 1 && month <= 12 ? names[static_cast<std::size_t>(month - 1)] : std::string();
}

std::string date_words(int month, int day, int year, const std::string& language) {
    if (month < 1 || month > 12 || day < 1 || day > 31) {
        return {};
    }
    if (language == "es") {
        return cardinal(day, language) + " de " + month_name(month, language) + " de " + cardinal(year, language);
    }
    if (language == "it") {
        return cardinal(day, language) + " " + month_name(month, language) + " " + cardinal(year, language);
    }
    return month_name(month, language) + " " + ordinal(day, language) + ", " + cardinal(year, language);
}

std::string fraction_words(int64_t numerator, int64_t denominator, const std::string& language) {
    if (language == "es") {
        return cardinal(numerator, language) + " sobre " + cardinal(denominator, language);
    }
    if (language == "it") {
        return cardinal(numerator, language) + " su " + cardinal(denominator, language);
    }
    static const std::vector<std::pair<std::string, std::string>> names = {
        {"", ""}, {"", ""}, {"half", "halves"}, {"third", "thirds"},
        {"fourth", "fourths"}, {"fifth", "fifths"}, {"sixth", "sixths"},
        {"seventh", "sevenths"}, {"eighth", "eighths"}, {"ninth", "ninths"},
        {"tenth", "tenths"},
    };
    if (denominator >= 2 && denominator < static_cast<int64_t>(names.size())) {
        const auto& name = names[static_cast<std::size_t>(denominator)];
        return cardinal(numerator, language) + " " + (numerator == 1 ? name.first : name.second);
    }
    return cardinal(numerator, language) + " over " + cardinal(denominator, language);
}

std::string normalize_punctuation(std::string value) {
    for (const std::string& zero_width : {u8"\u200b", u8"\u200c", u8"\u200d", u8"\ufeff"}) {
        replace_all(value, zero_width, "");
    }
    for (const std::string& space : {
             u8"\u00a0", u8"\u1680", u8"\u2000", u8"\u2001", u8"\u2002", u8"\u2003",
             u8"\u2004", u8"\u2005", u8"\u2006", u8"\u2007", u8"\u2008", u8"\u2009",
             u8"\u200a", u8"\u202f", u8"\u205f", u8"\u3000",
         }) {
        replace_all(value, space, " ");
    }
    for (const std::string& apostrophe : {
             u8"\u02bc", u8"\u2018", u8"\u2019", u8"\u201a", u8"\u201b", u8"\u2032", u8"\uff07",
         }) {
        replace_all(value, apostrophe, "'");
    }
    for (const std::string& quote : {
             u8"\u00ab", u8"\u00bb", u8"\u201c", u8"\u201d", u8"\u201e", u8"\u201f", u8"\u2033", u8"\uff02",
         }) {
        replace_all(value, quote, "\"");
    }
    for (const std::string& dash : {
             u8"\u2010", u8"\u2011", u8"\u2012", u8"\u2013", u8"\u2014", u8"\u2015", u8"\u2212",
         }) {
        replace_all(value, dash, "-");
    }
    replace_all(value, u8"\u2026", "...");
    replace_all(value, u8"\u2044", "/");
    replace_all(value, u8"\u2215", "/");
    replace_all(value, u8"\u00bc", " 1/4 ");
    replace_all(value, u8"\u00bd", " 1/2 ");
    replace_all(value, u8"\u00be", " 3/4 ");
    replace_all(value, "=", " equals ");
    replace_all(value, u8"\u00b0", " degrees ");
    for (std::size_t index = 0; index < value.size(); ++index) {
        if (value[index] != ':') {
            continue;
        }
        const bool digit_left = index > 0 && std::isdigit(static_cast<unsigned char>(value[index - 1]));
        const bool digit_right = index + 1 < value.size() && std::isdigit(static_cast<unsigned char>(value[index + 1]));
        if (!digit_left || !digit_right) {
            value[index] = ',';
        }
    }
    return collapse_ws(value);
}

}  // namespace

std::string normalize_spoken_text(const std::string& text, const std::string& language) {
    const std::string lang = normalizer_language(language);
    std::string value = normalize_punctuation(text);
    value = expand_dotted_initialisms(value, lang);

    // Currency is expanded before generic decimals and integers.
    value = replace_regex(value, std::regex(R"(\$\s*([0-9](?:[0-9.,]*[0-9])?))"), [&](const RegexMatch& match) {
        return currency_words(match[1].str(), "$", lang);
    });
    value = replace_regex(value, std::regex(u8R"(£\s*([0-9](?:[0-9.,]*[0-9])?))"), [&](const RegexMatch& match) {
        return currency_words(match[1].str(), "£", lang);
    });
    value = replace_regex(value, std::regex(u8R"(€\s*([0-9](?:[0-9.,]*[0-9])?))"), [&](const RegexMatch& match) {
        return currency_words(match[1].str(), "€", lang);
    });
    value = replace_regex(value, std::regex(R"(\b([0-9](?:[0-9.,]*[0-9])?)\s*(€|EUR)\b)", std::regex::icase), [&](const RegexMatch& match) {
        return currency_words(match[1].str(), "€", lang);
    });

    // Times retain an optional AM/PM suffix, matching the Python runtime.
    value = replace_regex(value, std::regex(R"(\b(\d{1,2}):(\d{2})(?:\s*([AaPp][Mm]))?\b)"), [&](const RegexMatch& match) {
        const int hour = std::stoi(match[1].str());
        const int minute = std::stoi(match[2].str());
        std::string minute_words;
        if (lang == "en" && minute == 0) {
            minute_words = "o'clock";
        } else if (lang == "en" && minute < 10) {
            minute_words = "oh " + cardinal(minute, lang);
        } else {
            minute_words = cardinal(minute, lang);
        }
        std::string suffix = match[3].str();
        std::transform(suffix.begin(), suffix.end(), suffix.begin(), [](unsigned char ch) {
            return static_cast<char>(std::toupper(ch));
        });
        return trim(cardinal(hour, lang) + " " + minute_words + " " + suffix);
    });
    value = replace_regex(value, std::regex(R"(\b(\d[\d.,]*)%)"), [&](const RegexMatch& match) {
        const std::string percent = lang == "es" ? "por ciento" : lang == "it" ? "per cento" : "percent";
        return decimal_words(match[1].str(), lang) + " " + percent;
    });
    value = replace_regex(value, std::regex(R"(\b(\d+)(st|nd|rd|th|o|a)\b)", std::regex::icase), [&](const RegexMatch& match) {
        return ordinal(parse_digits(match[1].str()), lang);
    });
    value = replace_regex(value, std::regex(u8R"(\b(\d+)(º|ª))"), [&](const RegexMatch& match) {
        return ordinal(parse_digits(match[1].str()), lang);
    });

    // Named English dates are handled before bare numbers.
    static const std::regex named_date(
        R"(\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+(\d{1,2})(?:,\s*(\d{2,4}))?\b)",
        std::regex::icase
    );
    value = replace_regex(value, named_date, [&](const RegexMatch& match) {
        if (lang != "en") {
            return match[0].str();
        }
        std::string month = match[1].str();
        std::transform(month.begin(), month.end(), month.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        const std::vector<std::vector<std::string>> aliases = {
            {"jan", "january"}, {"feb", "february"}, {"mar", "march"}, {"apr", "april"},
            {"may"}, {"jun", "june"}, {"jul", "july"}, {"aug", "august"},
            {"sep", "sept", "september"}, {"oct", "october"}, {"nov", "november"}, {"dec", "december"},
        };
        int month_number = 0;
        for (std::size_t index = 0; index < aliases.size(); ++index) {
            if (std::find(aliases[index].begin(), aliases[index].end(), month) != aliases[index].end()) {
                month_number = static_cast<int>(index + 1);
                break;
            }
        }
        const int day = std::stoi(match[2].str());
        if (month_number == 0 || day < 1 || day > 31) {
            return match[0].str();
        }
        std::string out = month_name(month_number, lang) + " " + ordinal(day, lang);
        if (match[3].matched) {
            out += ", " + cardinal(parse_digits(match[3].str()), lang);
        }
        return out;
    });
    value = replace_regex(value, std::regex(R"(\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b)"), [&](const RegexMatch& match) {
        const int first = std::stoi(match[1].str());
        const int second = std::stoi(match[2].str());
        const int year = std::stoi(match[3].str());
        const int month = lang == "en" ? first : second;
        const int day = lang == "en" ? second : first;
        const std::string expanded = date_words(month, day, year, lang);
        return expanded.empty() ? match[0].str() : expanded;
    });
    value = replace_regex(value, std::regex(R"(\b(\d{4})-(\d{1,2})-(\d{1,2})\b)"), [&](const RegexMatch& match) {
        const std::string expanded = date_words(
            std::stoi(match[2].str()),
            std::stoi(match[3].str()),
            std::stoi(match[1].str()),
            lang
        );
        return expanded.empty() ? match[0].str() : expanded;
    });

    value = replace_regex(value, std::regex(R"(\b(\d+)\s+(\d+)\s*/\s*(\d+)\b)"), [&](const RegexMatch& match) {
        return cardinal(parse_digits(match[1].str()), lang) + " " + and_word(lang) + " " +
            fraction_words(parse_digits(match[2].str()), parse_digits(match[3].str()), lang);
    });
    value = replace_regex(value, std::regex(R"(\b(\d+)\s*/\s*(\d+)\b)"), [&](const RegexMatch& match) {
        return fraction_words(parse_digits(match[1].str()), parse_digits(match[2].str()), lang);
    });

    if (lang == "es" || lang == "it") {
        value = replace_regex(value, std::regex(R"(\b\d[\d.]*,\d+\b)"), [&](const RegexMatch& match) {
            return decimal_words(match[0].str(), lang);
        });
        value = replace_regex(value, std::regex(R"(\b\d{1,3}(?:\.\d{3})+\b)"), [&](const RegexMatch& match) {
            return cardinal(parse_digits(match[0].str()), lang);
        });
    }
    value = replace_regex(value, std::regex(R"(\b\d[\d,]*\.\d+\b)"), [&](const RegexMatch& match) {
        return decimal_words(match[0].str(), lang);
    });
    value = replace_regex(value, std::regex(R"(\b\d{1,3}(?:,\d{3})+\b)"), [&](const RegexMatch& match) {
        return cardinal(parse_digits(match[0].str()), lang);
    });
    value = replace_regex(value, std::regex(R"(\b\d+\b)"), [&](const RegexMatch& match) {
        return cardinal(parse_digits(match[0].str()), lang);
    });

    value = expand_dash_letter_names(value, lang);
    replace_all(value, "-", " ");
    return collapse_ws(value);
}

}  // namespace scyllasband_detail
