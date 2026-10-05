// Unicode helpers with Python string semantics (str.isalnum, str.lower, str.split, re's \d \w \s \b, NFC).
#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace scyllasband::unicode {

using Text = std::u32string;

Text decode(std::string_view utf8);  // invalid bytes become U+FFFD
std::string encode(std::u32string_view text);
void append_utf8(std::string& out, char32_t cp);

bool is_alnum(char32_t cp);     // str.isalnum
bool is_decimal(char32_t cp);   // str.isdecimal, re \d
int decimal_value(char32_t cp); // 0-9, or -1
bool is_word(char32_t cp);      // re \w
bool is_space(char32_t cp);     // str.isspace, re \s
bool is_mark(char32_t cp);      // general category M*
bool is_punct_or_space(char32_t cp);  // general category P* or Z*
bool is_ascii_letter(char32_t cp);
bool is_ascii_digit(char32_t cp);

Text lower(std::u32string_view text);  // str.lower
Text nfc(std::u32string_view text);    // unicodedata.normalize("NFC", ...)
Text strip(std::u32string_view text);  // str.strip()
Text collapse_spaces(std::u32string_view text);  // re.sub(r"\s+", " ", text).strip()
std::vector<Text> split_whitespace(std::u32string_view text);  // str.split()
bool has_alnum(std::u32string_view text);
bool word_boundary(std::u32string_view text, std::size_t pos);  // re \b at pos

// Case-insensitive comparison of `cp` with an ASCII letter, as re.IGNORECASE does (including the
// dotted/dotless i, long s and Kelvin sign equivalences).
bool ascii_ci_equal(char32_t cp, char ascii_letter);

inline Text from_ascii(std::string_view text) { return Text(text.begin(), text.end()); }
inline Text u(std::string_view utf8) { return decode(utf8); }

}  // namespace scyllasband::unicode
