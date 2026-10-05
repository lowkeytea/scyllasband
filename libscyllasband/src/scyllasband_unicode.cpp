#include "scyllasband_unicode.h"

#include <algorithm>
#include <cstdint>
#include <unordered_map>

namespace scyllasband::unicode {
namespace {

struct CodepointRange {
    char32_t lo, hi;
};
struct LowerRun {
    char32_t lo, hi;
    int stride;
    int delta;
};
struct LowerSpecial {
    char32_t cp;
    const char* utf8;
};
struct Decomposition {
    char32_t cp, first, second;
    int composes;  // 1 when the pair recomposes under NFC (a primary composite)
};
struct CombiningClass {
    char32_t lo, hi;
    int value;
};

#include "scyllasband_unicode_tables.inc"

template <std::size_t N>
bool in_ranges(const CodepointRange (&table)[N], char32_t cp) {
    std::size_t lo = 0, hi = N;
    while (lo < hi) {
        std::size_t mid = (lo + hi) / 2;
        if (cp < table[mid].lo) hi = mid;
        else if (cp > table[mid].hi) lo = mid + 1;
        else return true;
    }
    return false;
}

const CodepointRange* find_range(const CodepointRange* begin, const CodepointRange* end, char32_t cp) {
    auto it = std::upper_bound(begin, end, cp, [](char32_t value, const CodepointRange& range) { return value < range.lo; });
    if (it == begin) return nullptr;
    --it;
    return cp <= it->hi ? it : nullptr;
}

int combining_class(char32_t cp) {
    std::size_t lo = 0, hi = sizeof(kCombiningClass) / sizeof(kCombiningClass[0]);
    while (lo < hi) {
        std::size_t mid = (lo + hi) / 2;
        if (cp < kCombiningClass[mid].lo) hi = mid;
        else if (cp > kCombiningClass[mid].hi) lo = mid + 1;
        else return kCombiningClass[mid].value;
    }
    return 0;
}

const Decomposition* find_decomposition(char32_t cp) {
    const Decomposition* begin = kDecomposition;
    const Decomposition* end = kDecomposition + sizeof(kDecomposition) / sizeof(kDecomposition[0]);
    auto it = std::lower_bound(begin, end, cp, [](const Decomposition& entry, char32_t value) { return entry.cp < value; });
    return it != end && it->cp == cp ? it : nullptr;
}

const std::unordered_map<uint64_t, char32_t>& composition_map() {
    static const std::unordered_map<uint64_t, char32_t> map = [] {
        std::unordered_map<uint64_t, char32_t> out;
        for (const Decomposition& entry : kDecomposition) {
            if (entry.composes) out[(static_cast<uint64_t>(entry.first) << 32) | entry.second] = entry.cp;
        }
        return out;
    }();
    return map;
}

constexpr char32_t kHangulSBase = 0xAC00, kHangulLBase = 0x1100, kHangulVBase = 0x1161, kHangulTBase = 0x11A7;
constexpr int kHangulLCount = 19, kHangulVCount = 21, kHangulTCount = 28, kHangulNCount = kHangulVCount * kHangulTCount;
constexpr int kHangulSCount = kHangulLCount * kHangulNCount;

void decompose(char32_t cp, Text& out) {
    if (cp >= kHangulSBase && cp < kHangulSBase + kHangulSCount) {
        int index = static_cast<int>(cp - kHangulSBase);
        out.push_back(kHangulLBase + index / kHangulNCount);
        out.push_back(kHangulVBase + (index % kHangulNCount) / kHangulTCount);
        if (index % kHangulTCount) out.push_back(kHangulTBase + index % kHangulTCount);
        return;
    }
    const Decomposition* entry = find_decomposition(cp);
    if (entry == nullptr) {
        out.push_back(cp);
        return;
    }
    decompose(entry->first, out);
    if (entry->second) decompose(entry->second, out);
}

char32_t compose_pair(char32_t first, char32_t second) {
    if (first >= kHangulLBase && first < kHangulLBase + kHangulLCount && second >= kHangulVBase && second < kHangulVBase + kHangulVCount) {
        return kHangulSBase + ((first - kHangulLBase) * kHangulVCount + (second - kHangulVBase)) * kHangulTCount;
    }
    if (first >= kHangulSBase && first < kHangulSBase + kHangulSCount && (first - kHangulSBase) % kHangulTCount == 0 &&
        second > kHangulTBase && second < kHangulTBase + kHangulTCount) {
        return first + (second - kHangulTBase);
    }
    const auto& map = composition_map();
    auto it = map.find((static_cast<uint64_t>(first) << 32) | second);
    return it == map.end() ? 0 : it->second;
}

}  // namespace

Text decode(std::string_view utf8) {
    Text out;
    out.reserve(utf8.size());
    std::size_t i = 0;
    while (i < utf8.size()) {
        unsigned char b0 = static_cast<unsigned char>(utf8[i]);
        if (b0 < 0x80) {
            out.push_back(b0);
            ++i;
            continue;
        }
        int length = (b0 & 0xE0) == 0xC0 ? 2 : (b0 & 0xF0) == 0xE0 ? 3 : (b0 & 0xF8) == 0xF0 ? 4 : 0;
        char32_t cp = length == 2 ? (b0 & 0x1F) : length == 3 ? (b0 & 0x0F) : (b0 & 0x07);
        bool valid = length > 0 && i + length <= utf8.size();
        for (int k = 1; valid && k < length; ++k) {
            unsigned char b = static_cast<unsigned char>(utf8[i + k]);
            if ((b & 0xC0) != 0x80) valid = false;
            else cp = (cp << 6) | (b & 0x3F);
        }
        if (valid && ((length == 2 && cp < 0x80) || (length == 3 && cp < 0x800) || (length == 4 && (cp < 0x10000 || cp > 0x10FFFF)))) {
            valid = false;
        }
        if (!valid) {
            out.push_back(0xFFFD);
            ++i;
            continue;
        }
        out.push_back(cp);
        i += static_cast<std::size_t>(length);
    }
    return out;
}

void append_utf8(std::string& out, char32_t cp) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

std::string encode(std::u32string_view text) {
    std::string out;
    out.reserve(text.size());
    for (char32_t cp : text) append_utf8(out, cp);
    return out;
}

bool is_alnum(char32_t cp) {
    if (cp < 0x80) return (cp >= '0' && cp <= '9') || is_ascii_letter(cp);
    return in_ranges(kAlnum, cp);
}

bool is_decimal(char32_t cp) {
    if (cp < 0x80) return cp >= '0' && cp <= '9';
    return in_ranges(kDecimal, cp);
}

int decimal_value(char32_t cp) {
    if (cp >= '0' && cp <= '9') return static_cast<int>(cp - '0');
    const CodepointRange* range = find_range(std::begin(kDecimal), std::end(kDecimal), cp);
    return range == nullptr ? -1 : static_cast<int>((cp - range->lo) % 10);
}

bool is_word(char32_t cp) { return cp == '_' || is_alnum(cp); }

bool is_space(char32_t cp) {
    switch (cp) {
        case 0x09: case 0x0A: case 0x0B: case 0x0C: case 0x0D: case 0x1C: case 0x1D: case 0x1E: case 0x1F: case 0x20:
        case 0x85: case 0xA0: case 0x1680: case 0x2028: case 0x2029: case 0x202F: case 0x205F: case 0x3000:
            return true;
        default:
            return cp >= 0x2000 && cp <= 0x200A;
    }
}

bool is_mark(char32_t cp) { return cp >= 0x300 && in_ranges(kMark, cp); }
bool is_punct_or_space(char32_t cp) { return in_ranges(kPunctOrSpace, cp); }
bool is_ascii_letter(char32_t cp) { return (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z'); }
bool is_ascii_digit(char32_t cp) { return cp >= '0' && cp <= '9'; }

// str.lower() maps a capital sigma to the final form after a cased letter (case-ignorable characters between)
// when no cased letter follows (again skipping case-ignorable characters).
static bool final_sigma(std::u32string_view text, std::size_t index) {
    auto ignorable = [](char32_t cp) { return in_ranges(kCaseIgnorable, cp); };
    auto cased = [](char32_t cp) { return in_ranges(kCased, cp); };
    std::size_t j = index;
    while (j > 0 && ignorable(text[j - 1])) --j;
    if (j == 0 || !cased(text[j - 1])) return false;
    std::size_t k = index + 1;
    while (k < text.size() && ignorable(text[k])) ++k;
    return k == text.size() || !cased(text[k]);
}

Text lower(std::u32string_view text) {
    Text out;
    out.reserve(text.size());
    for (std::size_t index = 0; index < text.size(); ++index) {
        const char32_t cp = text[index];
        if (cp == 0x3A3) {
            out.push_back(final_sigma(text, index) ? 0x3C2 : 0x3C3);
            continue;
        }
        if (cp < 0x80) {
            out.push_back(cp >= 'A' && cp <= 'Z' ? cp + 32 : cp);
            continue;
        }
        bool special = false;
        for (const LowerSpecial& entry : kLowerSpecial) {
            if (entry.cp == cp) {
                Text mapped = decode(entry.utf8);
                out.append(mapped);
                special = true;
                break;
            }
        }
        if (special) continue;
        char32_t mapped = cp;
        auto it = std::upper_bound(std::begin(kLower), std::end(kLower), cp, [](char32_t value, const LowerRun& run) { return value < run.lo; });
        if (it != std::begin(kLower)) {
            --it;
            if (cp <= it->hi && (cp - it->lo) % static_cast<char32_t>(it->stride) == 0) {
                mapped = static_cast<char32_t>(static_cast<int64_t>(cp) + it->delta);
            }
        }
        out.push_back(mapped);
    }
    return out;
}

Text nfc(std::u32string_view text) {
    Text decomposed;
    decomposed.reserve(text.size());
    for (char32_t cp : text) {
        if (cp < 0xC0) decomposed.push_back(cp);
        else decompose(cp, decomposed);
    }
    // Canonical ordering of combining marks.
    for (std::size_t i = 0; i < decomposed.size();) {
        if (combining_class(decomposed[i]) == 0) {
            ++i;
            continue;
        }
        std::size_t j = i;
        while (j < decomposed.size() && combining_class(decomposed[j]) != 0) ++j;
        std::stable_sort(decomposed.begin() + static_cast<std::ptrdiff_t>(i), decomposed.begin() + static_cast<std::ptrdiff_t>(j),
                         [](char32_t a, char32_t b) { return combining_class(a) < combining_class(b); });
        i = j;
    }
    // Canonical composition.
    Text out;
    out.reserve(decomposed.size());
    std::size_t starter = std::u32string::npos;
    int last_class = 0;
    for (char32_t cp : decomposed) {
        int cc = combining_class(cp);
        if (starter != std::u32string::npos) {
            // Marks between the starter and cp are in canonical order, so the last one decides blocking.
            bool adjacent = out.size() - 1 == starter;
            if (adjacent || (last_class != 0 && last_class < cc)) {
                char32_t composite = compose_pair(out[starter], cp);
                if (composite) {
                    out[starter] = composite;
                    continue;
                }
            }
        }
        if (cc == 0) starter = out.size();
        last_class = cc;
        out.push_back(cp);
    }
    return out;
}

Text strip(std::u32string_view text) {
    std::size_t start = 0, end = text.size();
    while (start < end && is_space(text[start])) ++start;
    while (end > start && is_space(text[end - 1])) --end;
    return Text(text.substr(start, end - start));
}

Text collapse_spaces(std::u32string_view text) {
    Text out;
    out.reserve(text.size());
    bool pending = false;
    for (char32_t cp : text) {
        if (is_space(cp)) {
            pending = true;
            continue;
        }
        if (pending && !out.empty()) out.push_back(' ');
        pending = false;
        out.push_back(cp);
    }
    return out;
}

std::vector<Text> split_whitespace(std::u32string_view text) {
    std::vector<Text> out;
    Text current;
    for (char32_t cp : text) {
        if (is_space(cp)) {
            if (!current.empty()) out.push_back(std::move(current));
            current.clear();
        } else {
            current.push_back(cp);
        }
    }
    if (!current.empty()) out.push_back(std::move(current));
    return out;
}

bool has_alnum(std::u32string_view text) {
    return std::any_of(text.begin(), text.end(), [](char32_t cp) { return is_alnum(cp); });
}

bool word_boundary(std::u32string_view text, std::size_t pos) {
    bool before = pos > 0 && pos <= text.size() && is_word(text[pos - 1]);
    bool after = pos < text.size() && is_word(text[pos]);
    return before != after;
}

bool ascii_ci_equal(char32_t cp, char ascii_letter) {
    char lower_letter = static_cast<char>(ascii_letter | 0x20);
    if (cp < 0x80) return static_cast<char>(cp | 0x20) == lower_letter && is_ascii_letter(cp);
    switch (lower_letter) {
        case 'i': return cp == 0x130 || cp == 0x131;
        case 's': return cp == 0x17F;
        case 'k': return cp == 0x212A;
        default: return false;
    }
}

}  // namespace scyllasband::unicode
