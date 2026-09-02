#include "scyllasband_bundle.h"
#include "scyllasband_boundary_policy.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace scyllasband_detail {
namespace {

constexpr const char* kContractVersion = "1.0.0";
const char* kRequiredComponents[] = {"g2p", "duration_predictor", "vector_estimator", "vocoder"};
const char* kOptionalComponents[] = {"vector_context_encoder", "vector_estimator_prefix", "vector_estimator_tail"};

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Bundle manifest is missing: " + path.string());
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

uint32_t rotate_right(uint32_t value, uint32_t count) {
    return (value >> count) | (value << (32U - count));
}

std::string sha256_bytes(const std::string& input) {
    static constexpr std::array<uint32_t, 64> k = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
        0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
        0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
        0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
    };
    std::vector<uint8_t> message(input.begin(), input.end());
    const uint64_t bit_length = static_cast<uint64_t>(message.size()) * 8U;
    message.push_back(0x80U);
    while (message.size() % 64U != 56U) {
        message.push_back(0U);
    }
    for (int shift = 56; shift >= 0; shift -= 8) {
        message.push_back(static_cast<uint8_t>((bit_length >> shift) & 0xffU));
    }
    std::array<uint32_t, 8> hash = {
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
    };
    for (std::size_t chunk = 0; chunk < message.size(); chunk += 64U) {
        std::array<uint32_t, 64> words{};
        for (std::size_t index = 0; index < 16U; ++index) {
            const std::size_t offset = chunk + index * 4U;
            words[index] = (static_cast<uint32_t>(message[offset]) << 24U) |
                (static_cast<uint32_t>(message[offset + 1U]) << 16U) |
                (static_cast<uint32_t>(message[offset + 2U]) << 8U) |
                static_cast<uint32_t>(message[offset + 3U]);
        }
        for (std::size_t index = 16U; index < 64U; ++index) {
            const uint32_t s0 = rotate_right(words[index - 15U], 7U) ^
                rotate_right(words[index - 15U], 18U) ^
                (words[index - 15U] >> 3U);
            const uint32_t s1 = rotate_right(words[index - 2U], 17U) ^
                rotate_right(words[index - 2U], 19U) ^
                (words[index - 2U] >> 10U);
            words[index] = words[index - 16U] + s0 +
                words[index - 7U] + s1;
        }
        uint32_t a = hash[0];
        uint32_t b = hash[1];
        uint32_t c = hash[2];
        uint32_t d = hash[3];
        uint32_t e = hash[4];
        uint32_t f = hash[5];
        uint32_t g = hash[6];
        uint32_t h = hash[7];
        for (std::size_t index = 0; index < 64U; ++index) {
            const uint32_t s1 = rotate_right(e, 6U) ^ rotate_right(e, 11U) ^
                rotate_right(e, 25U);
            const uint32_t choice = (e & f) ^ ((~e) & g);
            const uint32_t temp1 = h + s1 + choice + k[index] + words[index];
            const uint32_t s0 = rotate_right(a, 2U) ^ rotate_right(a, 13U) ^
                rotate_right(a, 22U);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = s0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        hash[0] += a;
        hash[1] += b;
        hash[2] += c;
        hash[3] += d;
        hash[4] += e;
        hash[5] += f;
        hash[6] += g;
        hash[7] += h;
    }
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (uint32_t value : hash) {
        output << std::setw(8) << value;
    }
    return output.str();
}

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (char ch : value) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << ch; break;
        }
    }
    return out.str();
}

void append_json_string_or_null(std::ostringstream& out, const std::string& value) {
    if (value.empty()) {
        out << "null";
        return;
    }
    out << "\"" << json_escape(value) << "\"";
}

std::string find_key_token(const std::string& key) {
    return "\"" + key + "\"";
}

std::size_t find_value_start(const std::string& json, const std::string& key) {
    const std::string token = find_key_token(key);
    const std::size_t key_pos = json.find(token);
    if (key_pos == std::string::npos) {
        return std::string::npos;
    }
    const std::size_t colon = json.find(':', key_pos + token.size());
    if (colon == std::string::npos) {
        return std::string::npos;
    }
    std::size_t pos = colon + 1;
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) {
        ++pos;
    }
    return pos;
}

std::size_t matching_delimiter(const std::string& json, std::size_t start, char open, char close) {
    bool in_string = false;
    bool escaped = false;
    int depth = 0;
    for (std::size_t pos = start; pos < json.size(); ++pos) {
        const char ch = json[pos];
        if (escaped) {
            escaped = false;
            continue;
        }
        if (ch == '\\' && in_string) {
            escaped = true;
            continue;
        }
        if (ch == '"') {
            in_string = !in_string;
            continue;
        }
        if (in_string) {
            continue;
        }
        if (ch == open) {
            ++depth;
        } else if (ch == close) {
            --depth;
            if (depth == 0) {
                return pos;
            }
        }
    }
    return std::string::npos;
}

std::string object_for_key(const std::string& json, const std::string& key) {
    const std::string token = find_key_token(key);
    std::size_t search_from = 0;
    while (search_from < json.size()) {
        const std::size_t key_pos = json.find(token, search_from);
        if (key_pos == std::string::npos) {
            break;
        }
        const std::size_t colon = json.find(':', key_pos + token.size());
        if (colon == std::string::npos) {
            break;
        }
        std::size_t start = colon + 1;
        while (start < json.size() &&
               std::isspace(static_cast<unsigned char>(json[start]))) {
            ++start;
        }
        if (start < json.size() && json[start] == '{') {
            const std::size_t end = matching_delimiter(json, start, '{', '}');
            if (end == std::string::npos) {
                return {};
            }
            return json.substr(start, end - start + 1);
        }
        search_from = key_pos + token.size();
    }
    return {};
}

std::string direct_object_for_key(const std::string& json, const std::string& key) {
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*\\{");
    for (std::sregex_iterator it(json.begin(), json.end(), pattern), end;
         it != end;
         ++it) {
        const std::size_t key_pos = static_cast<std::size_t>((*it).position());
        bool in_string = false;
        bool escaped = false;
        int depth = 0;
        for (std::size_t pos = 0; pos < key_pos; ++pos) {
            const char ch = json[pos];
            if (escaped) {
                escaped = false;
                continue;
            }
            if (ch == '\\' && in_string) {
                escaped = true;
                continue;
            }
            if (ch == '"') {
                in_string = !in_string;
                continue;
            }
            if (!in_string) {
                if (ch == '{') {
                    ++depth;
                } else if (ch == '}') {
                    --depth;
                }
            }
        }
        if (depth != 1) {
            continue;
        }
        const std::size_t start = json.find(
            '{',
            key_pos + static_cast<std::size_t>((*it).length()) - 1
        );
        if (start == std::string::npos) {
            return {};
        }
        const std::size_t object_end = matching_delimiter(json, start, '{', '}');
        if (object_end == std::string::npos) {
            return {};
        }
        return json.substr(start, object_end - start + 1);
    }
    return {};
}

std::string array_for_key(const std::string& json, const std::string& key) {
    const std::size_t start = find_value_start(json, key);
    if (start == std::string::npos || start >= json.size() || json[start] != '[') {
        return {};
    }
    const std::size_t end = matching_delimiter(json, start, '[', ']');
    if (end == std::string::npos) {
        return {};
    }
    return json.substr(start, end - start + 1);
}

std::string string_for_key(const std::string& json, const std::string& key, const std::string& fallback = {}) {
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
    std::smatch match;
    if (std::regex_search(json, match, pattern)) {
        return match[1].str();
    }
    return fallback;
}

int int_for_key(const std::string& json, const std::string& key, int fallback = 0) {
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*(-?[0-9]+)");
    std::smatch match;
    if (std::regex_search(json, match, pattern)) {
        return std::stoi(match[1].str());
    }
    return fallback;
}

float float_for_key(const std::string& json, const std::string& key, float fallback = 0.0f) {
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?)");
    std::smatch match;
    if (std::regex_search(json, match, pattern)) {
        return std::stof(match[1].str());
    }
    return fallback;
}

bool bool_for_key(const std::string& json, const std::string& key, bool fallback = false) {
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*(true|false)");
    std::smatch match;
    if (std::regex_search(json, match, pattern)) {
        return match[1].str() == "true";
    }
    return fallback;
}

bool exact_boolean_object_keys(
    const std::string& json,
    const std::vector<std::string>& expected_keys
) {
    if (json.empty()) {
        return false;
    }
    std::vector<std::string> keys;
    const std::regex key_pattern("\\\"([^\\\"]+)\\\"\\s*:");
    for (std::sregex_iterator it(json.begin(), json.end(), key_pattern), end;
         it != end;
         ++it) {
        keys.push_back((*it)[1].str());
    }
    std::sort(keys.begin(), keys.end());
    std::vector<std::string> expected = expected_keys;
    std::sort(expected.begin(), expected.end());
    if (keys != expected) {
        return false;
    }
    for (const std::string& key : expected_keys) {
        const std::regex boolean_pattern(
            "\\\"" + key + "\\\"\\s*:\\s*(true|false)\\s*[,}]"
        );
        if (!std::regex_search(json, boolean_pattern)) {
            return false;
        }
    }
    return true;
}

std::size_t find_json_string_end(const std::string& json, std::size_t start);
std::string decode_json_string_token(const std::string& token);

std::vector<std::string> direct_object_keys(const std::string& json) {
    std::vector<std::string> keys;
    if (json.size() < 2 || json.front() != '{' || json.back() != '}') {
        return keys;
    }
    bool in_string = false;
    bool escaped = false;
    int object_depth = 0;
    int array_depth = 0;
    for (std::size_t pos = 1; pos + 1 < json.size(); ++pos) {
        const char ch = json[pos];
        if (escaped) {
            escaped = false;
            continue;
        }
        if (ch == '\\' && in_string) {
            escaped = true;
            continue;
        }
        if (ch == '"') {
            if (!in_string && object_depth == 0 && array_depth == 0) {
                const std::size_t end = find_json_string_end(json, pos);
                if (end == std::string::npos) {
                    return {};
                }
                std::size_t colon = end + 1;
                while (colon < json.size() &&
                       std::isspace(static_cast<unsigned char>(json[colon]))) {
                    ++colon;
                }
                if (colon < json.size() && json[colon] == ':') {
                    keys.push_back(decode_json_string_token(
                        json.substr(pos, end - pos + 1)
                    ));
                }
            }
            in_string = !in_string;
            continue;
        }
        if (in_string) {
            continue;
        }
        if (ch == '{') ++object_depth;
        else if (ch == '}') --object_depth;
        else if (ch == '[') ++array_depth;
        else if (ch == ']') --array_depth;
    }
    return keys;
}

bool exact_direct_object_keys(
    const std::string& json,
    std::vector<std::string> expected
) {
    std::vector<std::string> actual = direct_object_keys(json);
    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());
    return actual == expected;
}

std::vector<std::string> string_array_for_key(const std::string& json, const std::string& key) {
    std::vector<std::string> values;
    const std::string array = array_for_key(json, key);
    if (array.empty()) {
        return values;
    }
    const std::regex item_pattern("\\\"([^\\\"]*)\\\"");
    for (std::sregex_iterator it(array.begin(), array.end(), item_pattern), end; it != end; ++it) {
        values.push_back((*it)[1].str());
    }
    return values;
}

std::vector<float> float_array_for_key(const std::string& json, const std::string& key) {
    std::vector<float> values;
    const std::string array = array_for_key(json, key);
    if (array.empty()) {
        return values;
    }
    const std::regex item_pattern("-?[0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?");
    for (std::sregex_iterator it(array.begin(), array.end(), item_pattern), end; it != end; ++it) {
        values.push_back(std::stof((*it)[0].str()));
    }
    return values;
}

std::vector<int64_t> int64_array_for_key(
    const std::string& json,
    const std::string& key
) {
    std::vector<int64_t> values;
    const std::string array = array_for_key(json, key);
    if (array.empty()) {
        return values;
    }
    const std::regex item_pattern("-?[0-9]+");
    for (std::sregex_iterator it(array.begin(), array.end(), item_pattern), end;
         it != end;
         ++it) {
        values.push_back(std::stoll((*it)[0].str()));
    }
    return values;
}

std::map<std::string, std::vector<float>> float_array_map_for_object(
    const std::string& object_json
) {
    std::map<std::string, std::vector<float>> values;
    const std::regex pair_pattern("\\\"([^\\\"]+)\\\"\\s*:\\s*(\\[[^\\]]*\\])");
    for (std::sregex_iterator it(object_json.begin(), object_json.end(), pair_pattern), end; it != end; ++it) {
        std::vector<float> vector;
        const std::string wrapped = "{\"value\":" + (*it)[2].str() + "}";
        vector = float_array_for_key(wrapped, "value");
        values[(*it)[1].str()] = std::move(vector);
    }
    return values;
}

std::map<std::string, std::string> string_map_for_object(const std::string& object_json) {
    std::map<std::string, std::string> values;
    const std::regex pair_pattern("\\\"([^\\\"]+)\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
    for (std::sregex_iterator it(object_json.begin(), object_json.end(), pair_pattern), end; it != end; ++it) {
        values[(*it)[1].str()] = (*it)[2].str();
    }
    return values;
}

std::map<std::string, int> int_map_for_object(const std::string& object_json) {
    std::map<std::string, int> values;
    const std::regex pair_pattern("\\\"([^\\\"]+)\\\"\\s*:\\s*(-?[0-9]+)");
    for (std::sregex_iterator it(object_json.begin(), object_json.end(), pair_pattern), end; it != end; ++it) {
        values[(*it)[1].str()] = std::stoi((*it)[2].str());
    }
    return values;
}

std::map<std::string, float> float_map_for_object(const std::string& object_json) {
    std::map<std::string, float> values;
    const std::regex pair_pattern(
        "\\\"([^\\\"]+)\\\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
    );
    for (std::sregex_iterator it(object_json.begin(), object_json.end(), pair_pattern), end;
         it != end;
         ++it) {
        values[(*it)[1].str()] = std::stof((*it)[2].str());
    }
    return values;
}

std::string decode_json_string_token(const std::string& token) {
    std::string out;
    bool escaped = false;
    for (std::size_t index = 1; index + 1 < token.size(); ++index) {
        const char ch = token[index];
        if (escaped) {
            switch (ch) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                default:
                    out.push_back(ch);
                    break;
            }
            escaped = false;
        } else if (ch == '\\') {
            escaped = true;
        } else {
            out.push_back(ch);
        }
    }
    return out;
}

std::size_t find_json_string_end(const std::string& json, std::size_t start) {
    bool escaped = false;
    for (std::size_t pos = start + 1; pos < json.size(); ++pos) {
        const char ch = json[pos];
        if (escaped) {
            escaped = false;
            continue;
        }
        if (ch == '\\') {
            escaped = true;
            continue;
        }
        if (ch == '"') {
            return pos;
        }
    }
    return std::string::npos;
}

std::map<std::string, int> json_string_int_map_for_object(const std::string& object_json) {
    std::map<std::string, int> values;
    std::size_t pos = 0;
    while (pos < object_json.size()) {
        const std::size_t key_start = object_json.find('"', pos);
        if (key_start == std::string::npos) {
            break;
        }
        const std::size_t key_end = find_json_string_end(object_json, key_start);
        if (key_end == std::string::npos) {
            break;
        }
        std::size_t colon = object_json.find(':', key_end + 1);
        if (colon == std::string::npos) {
            break;
        }
        std::size_t value_start = colon + 1;
        while (value_start < object_json.size() && std::isspace(static_cast<unsigned char>(object_json[value_start]))) {
            ++value_start;
        }
        std::size_t value_end = value_start;
        if (value_end < object_json.size() && (object_json[value_end] == '-' || std::isdigit(static_cast<unsigned char>(object_json[value_end])))) {
            ++value_end;
            while (value_end < object_json.size() && std::isdigit(static_cast<unsigned char>(object_json[value_end]))) {
                ++value_end;
            }
            values[decode_json_string_token(object_json.substr(key_start, key_end - key_start + 1))] =
                std::stoi(object_json.substr(value_start, value_end - value_start));
        }
        pos = value_end;
    }
    return values;
}

std::map<int, std::string> json_int_string_map_for_object(const std::string& object_json) {
    std::map<int, std::string> values;
    std::size_t pos = 0;
    while (pos < object_json.size()) {
        const std::size_t key_start = object_json.find('"', pos);
        if (key_start == std::string::npos) {
            break;
        }
        const std::size_t key_end = find_json_string_end(object_json, key_start);
        if (key_end == std::string::npos) {
            break;
        }
        const std::string key = decode_json_string_token(object_json.substr(key_start, key_end - key_start + 1));
        std::size_t colon = object_json.find(':', key_end + 1);
        if (colon == std::string::npos) {
            break;
        }
        std::size_t value_start = object_json.find('"', colon + 1);
        if (value_start == std::string::npos) {
            break;
        }
        const std::size_t value_end = find_json_string_end(object_json, value_start);
        if (value_end == std::string::npos) {
            break;
        }
        values[std::stoi(key)] = decode_json_string_token(object_json.substr(value_start, value_end - value_start + 1));
        pos = value_end + 1;
    }
    return values;
}


std::vector<std::string> object_items_from_array(const std::string& array_json) {
    std::vector<std::string> items;
    std::size_t pos = 0;
    while (pos < array_json.size()) {
        const std::size_t start = array_json.find('{', pos);
        if (start == std::string::npos) {
            break;
        }
        const std::size_t end = matching_delimiter(array_json, start, '{', '}');
        if (end == std::string::npos) {
            break;
        }
        items.push_back(array_json.substr(start, end - start + 1));
        pos = end + 1;
    }
    return items;
}

std::map<std::string, int> id_index_map_for_array(const std::string& json) {
    std::map<std::string, int> values;
    for (const std::string& item : object_items_from_array(json)) {
        const std::string id = string_for_key(item, "id");
        if (id.empty()) {
            continue;
        }
        values[id] = int_for_key(item, "index", static_cast<int>(values.size()));
    }
    return values;
}

std::map<std::string, int> load_id_index_asset(const std::filesystem::path& path) {
    return id_index_map_for_array(read_text_file(path));
}

std::map<std::string, int> load_phone_vocab_asset(const std::filesystem::path& path) {
    const std::string json = read_text_file(path);
    const std::string token_to_id = object_for_key(json, "token_to_id");
    return int_map_for_object(token_to_id);
}

bool required_component(const std::string& component_json) {
    return bool_for_key(component_json, "required", true);
}

void require_exists(const std::filesystem::path& path, const std::string& message) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error(message + ": " + path.string());
    }
}

std::string component_artifact_path(
    const std::string& component_json,
    const std::string& backend_name
) {
    const std::string artifacts = object_for_key(component_json, "artifacts");
    if (!artifacts.empty()) {
        const std::string backend_spec = object_for_key(artifacts, backend_name);
        if (!backend_spec.empty()) {
            return string_for_key(backend_spec, "path");
        }
    }
    return string_for_key(component_json, "path");
}

void validate_architecture(const std::string& architecture) {
    if (architecture != "scyllasband-duration-flow") {
        throw std::runtime_error("Unsupported architecture " + architecture + "; expected scyllasband-duration-flow");
    }
}

}  // namespace

std::string sha256_hex(const std::string& input) {
    return sha256_bytes(input);
}

std::string scyllasband_backend_name(ScyllasBandBackend backend) {
    switch (backend) {
        case SCYLLASBAND_BACKEND_AUTO:
            return "auto";
        case SCYLLASBAND_BACKEND_LITERT:
            return "litert";
        case SCYLLASBAND_BACKEND_COREML:
            return "coreml";
        case SCYLLASBAND_BACKEND_ONNX:
            return "onnx";
        case SCYLLASBAND_BACKEND_COREAI:
            return "coreai";
    }
    return "unknown";
}

ScyllasBandBackend scyllasband_select_backend(ScyllasBandBackend backend) {
    if (backend != SCYLLASBAND_BACKEND_AUTO) {
        return backend;
    }
#if defined(SCYLLASBAND_WITH_COREAI)
    return SCYLLASBAND_BACKEND_COREAI;
#elif defined(SCYLLASBAND_WITH_ONNXRUNTIME)
    return SCYLLASBAND_BACKEND_ONNX;
#elif defined(SCYLLASBAND_WITH_LITERT)
    return SCYLLASBAND_BACKEND_LITERT;
#else
    return SCYLLASBAND_BACKEND_ONNX;
#endif
}

ScyllasBandBundleInfo load_scyllasband_bundle_info(
    const std::string& bundle_dir,
    ScyllasBandBackend backend
) {
    const std::filesystem::path bundle_path(bundle_dir);
    const std::filesystem::path manifest_path = bundle_path / "manifest.json";
    const std::string manifest = read_text_file(manifest_path);

    ScyllasBandBundleInfo info;
    info.bundle_dir = bundle_path.string();
    info.selected_backend = scyllasband_backend_name(scyllasband_select_backend(backend));
    info.contract_version = string_for_key(manifest, "contract_version");
    if (info.contract_version != kContractVersion) {
        throw std::runtime_error(
            "Unsupported contract_version " + info.contract_version + "; expected " + kContractVersion
        );
    }
    info.model_name = string_for_key(manifest, "model_name");
    info.model_version = string_for_key(manifest, "model_version", "3");
    info.architecture = string_for_key(manifest, "architecture", "scyllasband-duration-flow");
    validate_architecture(info.architecture);
    info.default_language = string_for_key(manifest, "default_language", "en_us");
    info.languages = string_array_for_key(manifest, "languages");
    const std::vector<std::string> voice_specs = object_items_from_array(array_for_key(manifest, "voices"));
    for (const std::string& voice : voice_specs) {
        const std::string voice_id = string_for_key(voice, "id");
        if (voice_id.empty()) {
            continue;
        }
        info.voices.push_back(voice_id);
        info.voice_default_language[voice_id] = string_for_key(voice, "default_language", info.default_language);
        info.voice_languages[voice_id] = string_array_for_key(voice, "languages");
    }
    info.preferred_backends = string_array_for_key(manifest, "preferred_backends");
    if (info.preferred_backends.empty()) {
        throw std::runtime_error("Bundle manifest must declare at least one backend");
    }

    const std::string audio = object_for_key(manifest, "audio");
    info.sample_rate = int_for_key(audio, "sample_rate");
    info.hop_length = int_for_key(audio, "hop_length");
    info.latent_hop_length = int_for_key(audio, "latent_hop_length");
    info.latent_dim = int_for_key(audio, "latent_dim");

    const std::string controls = object_for_key(manifest, "controls");
    const std::string fixed_shapes = object_for_key(controls, "fixed_shapes");
    info.g2p_text_tokens = int_for_key(fixed_shapes, "g2p_text_tokens");
    info.phone_frames = int_for_key(fixed_shapes, "phone_frames");
    info.latent_frames = int_for_key(fixed_shapes, "latent_frames");
    const std::string target_buckets = object_for_key(controls, "target_buckets");
    info.target_bucket_margin_frames = std::max(0, int_for_key(target_buckets, "bucket_margin_frames", 0));
    for (const std::string& item : object_items_from_array(array_for_key(target_buckets, "buckets"))) {
        ScyllasBandTargetBucketInfo bucket;
        bucket.latent_frames = int_for_key(item, "latent_frames");
        bucket.vector_component = string_for_key(item, "vector_estimator", "vector_estimator");
        bucket.vector_prefix_component = string_for_key(item, "vector_estimator_prefix", "vector_estimator_prefix");
        bucket.vector_tail_component = string_for_key(item, "vector_estimator_tail", "vector_estimator_tail");
        bucket.vocoder_component = string_for_key(item, "vocoder", "vocoder");
        if (bucket.latent_frames > 0) {
            info.target_buckets.push_back(bucket);
        }
    }
    std::sort(info.target_buckets.begin(), info.target_buckets.end(), [](const auto& left, const auto& right) {
        return left.latent_frames < right.latent_frames;
    });
    const std::string reference = object_for_key(controls, "reference_packs");
    info.reference_packs_enabled = bool_for_key(reference, "enabled", false);
    info.reference_pack_schema_version = int_for_key(reference, "schema_version");
    info.reference_style_dim = int_for_key(reference, "style_dim");
    info.reference_prosody_dim = int_for_key(reference, "prosody_dim");
    info.reference_identity_dim = int_for_key(reference, "identity_dim");
    info.reference_baseline_dim = int_for_key(reference, "prosody_baseline_dim");
    info.reference_delta_dim = int_for_key(reference, "prosody_delta_dim");
    info.reference_affect_routing_version = string_for_key(
        object_for_key(reference, "affect_routing"), "version"
    );
    info.reference_fallback_weight = float_for_key(reference, "fallback_weight", 0.25f);
    const std::string prefix = object_for_key(controls, "prefix_conditioning");
    info.prefix_conditioning_enabled = bool_for_key(prefix, "enabled", false);
    info.prefix_max_frames = int_for_key(prefix, "max_frames");
    const std::string span = object_for_key(controls, "span_conditioning");
    info.span_conditioning_enabled = bool_for_key(span, "enabled", false);
    info.span_context_hidden_size = int_for_key(span, "hidden_size");
    info.span_context_max_phones = int_for_key(span, "context_max_phones");
    const std::string guidance = object_for_key(controls, "emotion_guidance");
    info.emotion_guidance_enabled = bool_for_key(guidance, "enabled", false);
    const std::string affect = object_for_key(controls, "affect");
    info.affect_enabled = bool_for_key(affect, "enabled", false);
    if (info.affect_enabled) {
        info.affect_graph_input_contract = string_for_key(
            controls,
            "graph_input_contract",
            string_for_key(affect, "graph_input_contract")
        );
        info.affect_axes = string_array_for_key(affect, "axes");
        info.affect_axis_order_version = std::to_string(int_for_key(affect, "axis_order_version", 0));
        info.affect_default_preset = string_for_key(affect, "default_preset");
        info.affect_preset_version = int_for_key(affect, "preset_version", 0);
        info.affect_presets = float_array_map_for_object(object_for_key(affect, "presets"));
        info.affect_legacy_presets = float_array_map_for_object(object_for_key(affect, "legacy_presets"));
        info.affect_partial_defaults = float_map_for_object(
            object_for_key(affect, "partial_defaults")
        );
        info.affect_axis_minimums = float_map_for_object(
            object_for_key(affect, "axis_minimums")
        );
        info.affect_axis_maximums = float_map_for_object(
            object_for_key(affect, "axis_maximums")
        );
        const std::string affect_guidance = object_for_key(affect, "guidance");
        info.affect_guidance_default_scale = float_for_key(affect_guidance, "default_scale", 1.0f);
        const int axis_version = int_for_key(affect, "axis_order_version", 0);
        const std::vector<std::string> expected_axes = axis_version == 1
            ? std::vector<std::string>{
                "calm", "joy", "anger", "sadness", "sarcasm", "questioning"
            }
            : axis_version == 2
            ? std::vector<std::string>{
                "calm", "joy", "anger", "sadness", "sarcasm", "whisper"
            }
            : axis_version == 3
            ? std::vector<std::string>{
                "calm", "joy", "anger", "sadness", "whisper"
            }
            : std::vector<std::string>{};
        if (expected_axes.empty()) {
            throw std::runtime_error(
                "Unsupported affect axis_order_version '" + std::to_string(axis_version) + "'"
            );
        }
        const std::string expected_graph_contract =
            "scyllasband_affect_v" + std::to_string(axis_version);
        if (info.affect_graph_input_contract != expected_graph_contract) {
            throw std::runtime_error(
                "Unsupported affect graph_input_contract '" + info.affect_graph_input_contract + "'"
            );
        }
        if (info.affect_axes != expected_axes) {
            throw std::runtime_error(
                "Affect axes do not match axis_order_version " + std::to_string(axis_version)
            );
        }
        // The affect contract is self-describing: axis_order_version alone fixes
        // both the axis names and the graph input contract, and both are checked
        // above, so v1 'questioning' can never be read as v2 'whisper'.
        // model_version is release metadata and is deliberately NOT cross-checked
        // against the axis order -- doing so hardcodes a release-numbering scheme
        // into the runtime. Keep this in sync with scyllasband/contract.py.
        auto validate_presets = [&](const std::map<std::string, std::vector<float>>& presets) {
            for (const auto& item : presets) {
                if (item.second.size() != expected_axes.size()) {
                    throw std::runtime_error(
                        "Affect preset '" + item.first + "' must contain " +
                        std::to_string(expected_axes.size()) + " values"
                    );
                }
                for (std::size_t index = 0; index < item.second.size(); ++index) {
                    const float value = item.second[index];
                    if (!std::isfinite(value) || value < 0.0f || value > 1.0f) {
                        throw std::runtime_error("Affect preset '" + item.first + "' contains an invalid value");
                    }
                    const std::string& axis = expected_axes[index];
                    const float minimum = info.affect_axis_minimums.count(axis) > 0
                        ? info.affect_axis_minimums.at(axis)
                        : 0.0f;
                    const float maximum = info.affect_axis_maximums.count(axis) > 0
                        ? info.affect_axis_maximums.at(axis)
                        : 1.0f;
                    if (value < minimum || value > maximum) {
                        throw std::runtime_error(
                            "Affect preset '" + item.first + "' violates bounds for '" + axis + "'"
                        );
                    }
                }
            }
        };
        auto validate_axis_policy = [&](const std::map<std::string, float>& policy,
                                        const std::string& label) {
            for (const auto& item : policy) {
                if (std::find(expected_axes.begin(), expected_axes.end(), item.first) == expected_axes.end()) {
                    throw std::runtime_error(
                        "Affect " + label + " contains unknown axis '" + item.first + "'"
                    );
                }
                if (!std::isfinite(item.second) || item.second < 0.0f || item.second > 1.0f) {
                    throw std::runtime_error(
                        "Affect " + label + " contains an invalid value for '" + item.first + "'"
                    );
                }
            }
        };
        validate_axis_policy(info.affect_partial_defaults, "partial_defaults");
        validate_axis_policy(info.affect_axis_minimums, "axis_minimums");
        validate_axis_policy(info.affect_axis_maximums, "axis_maximums");
        for (const std::string& axis : expected_axes) {
            const float minimum = info.affect_axis_minimums.count(axis) > 0
                ? info.affect_axis_minimums.at(axis)
                : 0.0f;
            const float maximum = info.affect_axis_maximums.count(axis) > 0
                ? info.affect_axis_maximums.at(axis)
                : 1.0f;
            if (minimum > maximum) {
                throw std::runtime_error("Affect minimum exceeds maximum for '" + axis + "'");
            }
            const auto partial = info.affect_partial_defaults.find(axis);
            if (partial != info.affect_partial_defaults.end() &&
                (partial->second < minimum || partial->second > maximum)) {
                throw std::runtime_error(
                    "Affect partial default violates bounds for '" + axis + "'"
                );
            }
        }
        validate_presets(info.affect_presets);
        validate_presets(info.affect_legacy_presets);
        if (info.affect_presets.count(info.affect_default_preset) == 0) {
            throw std::runtime_error("Affect default_preset must name a declared preset");
        }
    }
    const std::string word_boundaries = object_for_key(controls, "word_boundaries");
    info.word_boundaries_enabled = bool_for_key(
        word_boundaries, "enabled", false
    );
    info.word_boundary_g2p_output_symbol = string_for_key(
        word_boundaries, "g2p_output_symbol", " "
    );
    info.word_boundary_duration_phone = string_for_key(
        word_boundaries, "duration_phone", "<sil>"
    );
    info.word_boundary_presence_threshold_frames = float_for_key(
        word_boundaries, "presence_threshold_frames", 0.5f
    );
    if (info.word_boundaries_enabled &&
        info.word_boundary_presence_threshold_frames <= 0.0f) {
        throw std::runtime_error(
            "Word-boundary presence threshold must be positive"
        );
    }
    const std::string duration_pause_presence = object_for_key(
        controls, "duration_pause_presence"
    );
    info.duration_pause_presence_enabled = bool_for_key(
        duration_pause_presence, "enabled", false
    );
    info.duration_pause_presence_threshold_probability = float_for_key(
        duration_pause_presence, "threshold_probability", 0.5f
    );
    if (info.duration_pause_presence_enabled) {
        if (string_for_key(duration_pause_presence, "schema") !=
                "scyllasband_duration_pause_presence_v1" ||
            string_for_key(duration_pause_presence, "activation") != "sigmoid") {
            throw std::runtime_error(
                "Unsupported duration pause-presence contract"
            );
        }
        if (!std::isfinite(info.duration_pause_presence_threshold_probability) ||
            info.duration_pause_presence_threshold_probability <= 0.0f ||
            info.duration_pause_presence_threshold_probability > 1.0f) {
            throw std::runtime_error(
                "Duration pause-presence threshold must be finite and in (0, 1]"
            );
        }
        if (string_array_for_key(duration_pause_presence, "outputs") !=
            std::vector<std::string>{"durations", "pause_presence_logits"}) {
            throw std::runtime_error(
                "Duration pause-presence controls must declare both duration outputs"
            );
        }
        const std::string owners = object_for_key(
            duration_pause_presence, "eligible_pause_owners"
        );
        if (string_for_key(owners, "punctuation") != "following_silence_only" ||
            string_for_key(owners, "word_boundary") !=
                "explicit_candidate_mask_only") {
            throw std::runtime_error(
                "Duration pause-presence ownership contract is invalid"
            );
        }
    }
    const std::string duration_hierarchy = direct_object_for_key(
        controls, "duration_hierarchy_sampling"
    );
    info.duration_hierarchy_sampling_enabled = bool_for_key(
        duration_hierarchy, "enabled", false
    );
    if (info.duration_hierarchy_sampling_enabled) {
        const std::vector<std::string> hierarchy_outputs =
            info.duration_pause_presence_enabled
                ? std::vector<std::string>{
                      "durations", "pause_presence_logits", "duration_quantiles"
                  }
                : std::vector<std::string>{"durations", "duration_quantiles"};
        if (!exact_direct_object_keys(
                duration_hierarchy,
                {"enabled", "schema", "policy", "outputs", "quantiles", "modes",
                 "default_mode", "seed", "sampling_key", "defaults",
                 "phrase_boundary", "pause_owners"}
            ) ||
            string_for_key(duration_hierarchy, "schema") !=
                "scyllasband_duration_hierarchy_sampling_v1" ||
            string_for_key(duration_hierarchy, "policy") !=
                "coherent_utterance_phrase_quantile_with_presence_hurdle_v1" ||
            string_array_for_key(duration_hierarchy, "outputs") != hierarchy_outputs ||
            string_array_for_key(duration_hierarchy, "modes") !=
                std::vector<std::string>{"sampled", "p50"}) {
            throw std::runtime_error("Unsupported duration hierarchy sampling contract");
        }
        const std::string default_mode = string_for_key(
            duration_hierarchy, "default_mode"
        );
        if (default_mode != "sampled" && default_mode != "p50") {
            throw std::runtime_error("Duration hierarchy default mode is invalid");
        }
        info.duration_hierarchy_default_sampled = default_mode == "sampled";
        const std::string quantiles = direct_object_for_key(
            duration_hierarchy, "quantiles"
        );
        if (!exact_direct_object_keys(
                quantiles, {"output", "levels", "domain", "ordering"}
            ) ||
            string_for_key(quantiles, "output") != "duration_quantiles" ||
            float_array_for_key(quantiles, "levels") !=
                std::vector<float>{0.1f, 0.5f, 0.9f} ||
            string_for_key(quantiles, "domain") != "positive_duration_frames" ||
            string_for_key(quantiles, "ordering") !=
                "monotonic_p10_p50_p90") {
            throw std::runtime_error("Duration hierarchy quantile contract is invalid");
        }
        const std::string seed = direct_object_for_key(duration_hierarchy, "seed");
        if (!exact_direct_object_keys(
                seed, {"source", "missing_request_seed", "algorithm"}
            ) ||
            string_for_key(seed, "source") != "request_seed" ||
            int_for_key(seed, "missing_request_seed", -1) != 0 ||
            string_for_key(seed, "algorithm") != "sha256_box_muller_v1") {
            throw std::runtime_error("Duration hierarchy seed contract is invalid");
        }
        if (string_for_key(duration_hierarchy, "sampling_key") !=
            "utf8_byte_length_prefixed_language_voice_phones_v1") {
            throw std::runtime_error("Duration hierarchy sampling key is invalid");
        }
        const std::string defaults = direct_object_for_key(
            duration_hierarchy, "defaults"
        );
        info.duration_hierarchy_pause_strength = float_for_key(
            defaults, "pause_strength", -1.0f
        );
        info.duration_hierarchy_speech_strength = float_for_key(
            defaults, "speech_strength", -1.0f
        );
        info.duration_hierarchy_sample_presence = bool_for_key(
            defaults, "sample_presence", false
        );
        info.duration_hierarchy_max_abs_z = float_for_key(
            defaults, "max_abs_z", -1.0f
        );
        if (!exact_direct_object_keys(
                defaults,
                {"pause_strength", "speech_strength", "sample_presence", "max_abs_z"}
            ) ||
            info.duration_hierarchy_pause_strength != 1.0f ||
            info.duration_hierarchy_speech_strength != 0.0f ||
            info.duration_hierarchy_sample_presence !=
                info.duration_pause_presence_enabled ||
            info.duration_hierarchy_max_abs_z != 2.0f) {
            throw std::runtime_error("Duration hierarchy defaults are invalid");
        }
        const std::string pause_owners = direct_object_for_key(
            duration_hierarchy, "pause_owners"
        );
        if (string_for_key(duration_hierarchy, "phrase_boundary") !=
                "punctuation_owned_silence_after_boundary_v1" ||
            !exact_direct_object_keys(
                pause_owners, {"punctuation", "word_boundary"}
            ) ||
            string_for_key(pause_owners, "punctuation") !=
                "following_or_remapped_silence_only" ||
            string_for_key(pause_owners, "word_boundary") !=
                "explicit_candidate_mask_only") {
            throw std::runtime_error("Duration hierarchy ownership contract is invalid");
        }
    } else if (!duration_hierarchy.empty()) {
        throw std::runtime_error(
            "Disabled duration hierarchy controls must be omitted"
        );
    }
    const std::string vector_timing = object_for_key(
        controls, "vector_timing_conditioning"
    );
    info.vector_timing_enabled = bool_for_key(vector_timing, "enabled", false);
    if (info.vector_timing_enabled) {
        const std::vector<std::string> timing_inputs = {
            "expanded_boundary_event_ids",
            "expanded_modifier_event_ids",
            "expanded_phone_phase",
            "expanded_phone_log_duration",
        };
        if (string_for_key(vector_timing, "schema") !=
                "scyllasband_vector_timing_conditioning_v1" ||
            string_array_for_key(vector_timing, "inputs") != timing_inputs) {
            throw std::runtime_error("Unsupported vector timing conditioning contract");
        }
        const std::string features = object_for_key(vector_timing, "features");
        if (!exact_boolean_object_keys(
                features,
                {"boundary_events", "modifier_events", "local_timing"}
            )) {
            throw std::runtime_error("Vector timing feature flags are invalid");
        }
        info.vector_boundary_events_enabled = bool_for_key(
            features, "boundary_events", false
        );
        info.vector_modifier_events_enabled = bool_for_key(
            features, "modifier_events", false
        );
        info.vector_local_timing_enabled = bool_for_key(
            features, "local_timing", false
        );
        if (!info.vector_boundary_events_enabled &&
            !info.vector_modifier_events_enabled &&
            !info.vector_local_timing_enabled) {
            throw std::runtime_error("Enabled vector timing contract has no feature");
        }
        const std::string phone_vocab = object_for_key(vector_timing, "phone_vocab");
        info.vector_timing_phone_vocab_asset = string_for_key(phone_vocab, "asset");
        info.vector_timing_phone_vocab_sha256 = string_for_key(phone_vocab, "sha256");
        info.vector_timing_phone_vocab_size = int_for_key(phone_vocab, "size", 0);
        if (info.vector_timing_phone_vocab_asset.empty() ||
            info.vector_timing_phone_vocab_sha256.size() != 64 ||
            info.vector_timing_phone_vocab_size <= 0) {
            throw std::runtime_error("Vector timing phone vocabulary binding is invalid");
        }
        const std::string boundary = direct_object_for_key(
            vector_timing, "boundary_events"
        );
        if (bool_for_key(boundary, "enabled", false) !=
                info.vector_boundary_events_enabled ||
            string_for_key(boundary, "encoding") !=
                "phone_vocab_id_plus_synthetic_word_boundary_v1") {
            throw std::runtime_error("Vector timing boundary-event contract is invalid");
        }
        info.vector_synthetic_word_boundary_id = int_for_key(
            boundary, "synthetic_word_boundary_id", -1
        );
        info.vector_punctuation_symbols = string_array_for_key(
            boundary, "punctuation_symbols"
        );
        info.vector_punctuation_phone_ids = int64_array_for_key(
            boundary, "punctuation_phone_ids"
        );
        const std::string owners = object_for_key(boundary, "owners");
        if (info.vector_synthetic_word_boundary_id !=
                info.vector_timing_phone_vocab_size ||
            string_for_key(owners, "punctuation") !=
                "following_silence_then_first_surviving_frame" ||
            string_for_key(owners, "word_boundary") !=
                "candidate_silence_then_first_surviving_frame") {
            throw std::runtime_error("Vector timing boundary ownership is invalid");
        }
        const std::string modifiers = direct_object_for_key(
            vector_timing, "modifier_events"
        );
        if (bool_for_key(modifiers, "enabled", false) !=
                info.vector_modifier_events_enabled ||
            string_for_key(modifiers, "encoding") != "bitmask_v1") {
            throw std::runtime_error("Vector timing modifier-event contract is invalid");
        }
        info.vector_modifier_event_bits = int_for_key(modifiers, "bits", 0);
        info.vector_modifier_symbols = string_array_for_key(modifiers, "symbols");
        info.vector_modifier_phone_ids = int64_array_for_key(modifiers, "phone_ids");
        info.vector_modifier_bit_masks = int64_array_for_key(modifiers, "bit_masks");
        const std::size_t modifier_count = info.vector_modifier_events_enabled
            ? static_cast<std::size_t>(info.vector_modifier_event_bits)
            : 0;
        if (info.vector_modifier_event_bits < 0 ||
            info.vector_modifier_event_bits > 30 ||
            info.vector_modifier_symbols.size() != modifier_count ||
            info.vector_modifier_phone_ids.size() != modifier_count ||
            info.vector_modifier_bit_masks.size() != modifier_count ||
            string_for_key(modifiers, "ownership") !=
                "stress_bidirectional_postfix_exact_base_segment_bounded_v2" ||
            string_for_key(modifiers, "zero_quantized_fallback") !=
                "audit_unrepresented_zero_frame_modifier_v1") {
            throw std::runtime_error("Vector timing modifier vocabulary is invalid");
        }
        for (std::size_t bit = 0; bit < modifier_count; ++bit) {
            if (info.vector_modifier_bit_masks[bit] !=
                (static_cast<int64_t>(1) << bit)) {
                throw std::runtime_error("Vector timing modifier bit masks are invalid");
            }
        }
        const std::string local = direct_object_for_key(
            vector_timing, "local_timing"
        );
        if (bool_for_key(local, "enabled", false) != info.vector_local_timing_enabled ||
            string_for_key(local, "phase") !=
                "(frame_index_plus_0_5)/phone_duration_frames" ||
            string_for_key(local, "log_duration") !=
                "log1p(phone_duration_frames)" ||
            string_for_key(local, "duration_source") !=
                "final_post_pause_presence_and_floor_frames") {
            throw std::runtime_error("Vector local-timing contract is invalid");
        }
    }
    const std::string punctuation_silence = object_for_key(controls, "punctuation_silence");
    info.punctuation_silence_target = string_for_key(
        punctuation_silence,
        "target",
        "merge_into_punctuation"
    );
    if (info.punctuation_silence_target != "explicit_silence" &&
        info.punctuation_silence_target != "merge_into_punctuation") {
        info.punctuation_silence_target = "merge_into_punctuation";
    }
    info.punctuation_pause_floors_calibrated = bool_for_key(
        punctuation_silence,
        "calibrated_floors",
        false
    );
    info.punctuation_pause_floor_table_ms = float_map_for_object(
        object_for_key(punctuation_silence, "floor_table_ms")
    );
    info.punctuation_pause_floor_table_sha256 = string_for_key(
        punctuation_silence,
        "floor_table_sha256"
    );
    if (!info.punctuation_pause_floor_table_ms.empty() &&
        !info.punctuation_pause_floors_calibrated) {
        throw std::runtime_error(
            "Punctuation pause floor table is present but not marked calibrated"
        );
    }
    if (info.punctuation_pause_floors_calibrated &&
        (info.punctuation_pause_floor_table_ms.empty() ||
         info.punctuation_pause_floor_table_sha256.empty())) {
        throw std::runtime_error(
            "Calibrated punctuation pause floors require a table and SHA-256"
        );
    }
    for (const auto& item : info.punctuation_pause_floor_table_ms) {
        if (!std::isfinite(item.second) || item.second < 0.0f) {
            throw std::runtime_error(
                "Invalid punctuation pause floor for '" + item.first + "'"
            );
        }
    }
    const std::string emotions = object_for_key(controls, "emotions");
    info.emotions_enabled = bool_for_key(emotions, "enabled", false);
    const std::string runtime_acceleration = object_for_key(controls, "runtime_acceleration");
    info.runtime_acceleration_metadata_present = !runtime_acceleration.empty();
    if (info.runtime_acceleration_metadata_present) {
        info.runtime_acceleration_backend = string_for_key(runtime_acceleration, "backend");
        info.runtime_acceleration_runtime_version = string_for_key(
            runtime_acceleration,
            "runtime_version"
        );
        const std::string native_gpu = object_for_key(runtime_acceleration, "native_gpu_acceleration");
        if (!native_gpu.empty()) {
            info.runtime_acceleration_cuda_required = bool_for_key(native_gpu, "cuda_required", false);
            if (info.runtime_acceleration_runtime_version.empty()) {
                info.runtime_acceleration_runtime_version = string_for_key(native_gpu, "runtime_version");
            }
        }
        const std::string vector = object_for_key(runtime_acceleration, "vector_estimator");
        if (!vector.empty()) {
            info.runtime_acceleration_vector_execution = string_for_key(vector, "execution");
            info.runtime_acceleration_vector_policy = string_for_key(vector, "policy");
            info.runtime_acceleration_min_gpu_split_vector_latent_frames = int_for_key(
                vector,
                "min_gpu_split_vector_latent_frames",
                0
            );
            info.runtime_acceleration_vector_split_available = bool_for_key(
                vector,
                "split_artifacts_available",
                false
            );
        }
    }

    const std::string components = object_for_key(manifest, "components");
    if (components.empty()) {
        throw std::runtime_error("Bundle manifest is missing components");
    }
    for (const char* component_name : kRequiredComponents) {
        const std::string component = object_for_key(components, component_name);
        if (component.empty()) {
            throw std::runtime_error(std::string("Bundle manifest is missing component: ") + component_name);
        }
        if (!required_component(component)) {
            continue;
        }
        const std::string relative_path = component_artifact_path(component, info.selected_backend);
        if (relative_path.empty()) {
            throw std::runtime_error(
                std::string("Required component '") + component_name + "' has no " + info.selected_backend + " artifact"
            );
        }
        const std::filesystem::path artifact_path = bundle_path / relative_path;
        require_exists(
            artifact_path,
            std::string("Required component '") + component_name + "' artifact '" + info.selected_backend + "' is missing"
        );
        info.component_artifacts[component_name] = relative_path;
        info.component_inputs[component_name] = string_array_for_key(component, "inputs");
        info.component_outputs[component_name] = string_array_for_key(component, "outputs");
    }
    for (const char* component_name : kOptionalComponents) {
        const std::string component = object_for_key(components, component_name);
        if (component.empty()) {
            continue;
        }
        const std::string relative_path = component_artifact_path(component, info.selected_backend);
        if (relative_path.empty()) {
            continue;
        }
        require_exists(
            bundle_path / relative_path,
            std::string("Optional component '") + component_name + "' artifact '" + info.selected_backend + "' is missing"
        );
        info.component_artifacts[component_name] = relative_path;
        info.component_inputs[component_name] = string_array_for_key(component, "inputs");
        info.component_outputs[component_name] = string_array_for_key(component, "outputs");
    }
    const std::vector<std::string>& duration_outputs =
        info.component_outputs.at("duration_predictor");
    std::vector<std::string> expected_duration_outputs = {"durations"};
    if (info.duration_pause_presence_enabled) {
        expected_duration_outputs.push_back("pause_presence_logits");
    }
    if (info.duration_hierarchy_sampling_enabled) {
        expected_duration_outputs.push_back("duration_quantiles");
    }
    if (duration_outputs != expected_duration_outputs) {
        throw std::runtime_error(
            "duration_predictor outputs do not match enabled duration controls"
        );
    }
    if (info.duration_pause_presence_enabled) {
        if (std::find(duration_outputs.begin(), duration_outputs.end(),
                      "pause_presence_logits") == duration_outputs.end()) {
            throw std::runtime_error(
                "duration_predictor outputs must be durations,pause_presence_logits "
                "when duration_pause_presence is enabled"
            );
        }
    } else if (std::find(
                   duration_outputs.begin(),
                   duration_outputs.end(),
                   "pause_presence_logits"
               ) != duration_outputs.end()) {
        throw std::runtime_error(
            "duration_predictor declares pause_presence_logits without "
            "duration_pause_presence controls"
        );
    }
    if (!info.duration_hierarchy_sampling_enabled &&
        std::find(duration_outputs.begin(), duration_outputs.end(),
                  "duration_quantiles") != duration_outputs.end()) {
        throw std::runtime_error(
            "duration_predictor declares duration_quantiles without hierarchy controls"
        );
    }
    auto load_bucket_component = [&](const std::string& component_name, bool required) {
        if (component_name.empty() || info.component_artifacts.count(component_name) > 0) {
            return;
        }
        const std::string component = object_for_key(components, component_name);
        if (component.empty()) {
            if (!required) {
                return;
            }
            throw std::runtime_error("Target bucket component is missing from manifest: " + component_name);
        }
        const std::string relative_path = component_artifact_path(component, info.selected_backend);
        if (relative_path.empty()) {
            if (!required) {
                return;
            }
            throw std::runtime_error("Target bucket component has no " + info.selected_backend + " artifact: " + component_name);
        }
        require_exists(
            bundle_path / relative_path,
            "Target bucket component '" + component_name + "' artifact '" + info.selected_backend + "' is missing"
        );
        info.component_artifacts[component_name] = relative_path;
        info.component_inputs[component_name] = string_array_for_key(component, "inputs");
        info.component_outputs[component_name] = string_array_for_key(component, "outputs");
    };
    const bool require_split_bucket_components = info.selected_backend == "litert";
    for (const ScyllasBandTargetBucketInfo& bucket : info.target_buckets) {
        load_bucket_component(bucket.vector_component, true);
        load_bucket_component(bucket.vocoder_component, true);
        if (!bucket.vector_prefix_component.empty() && !bucket.vector_tail_component.empty()) {
            load_bucket_component(bucket.vector_prefix_component, require_split_bucket_components);
            load_bucket_component(bucket.vector_tail_component, require_split_bucket_components);
        }
    }

    const std::string assets = object_for_key(manifest, "assets");
    info.assets = string_map_for_object(assets);
    for (const auto& item : info.assets) {
        require_exists(
            bundle_path / item.second,
            std::string("Declared asset '") + item.first + "' is missing"
        );
    }
    const auto voice_asset = info.assets.find("voice_index");
    if (voice_asset != info.assets.end()) {
        info.voice_to_id = load_id_index_asset(bundle_path / voice_asset->second);
    }
    const auto language_asset = info.assets.find("language_index");
    if (language_asset != info.assets.end()) {
        info.language_to_id = load_id_index_asset(bundle_path / language_asset->second);
    }
    const auto emotion_asset = info.assets.find("emotion_index");
    if (emotion_asset != info.assets.end()) {
        info.emotion_to_id = load_id_index_asset(bundle_path / emotion_asset->second);
    }
    const auto phone_asset = info.assets.find("phone_vocab");
    if (phone_asset != info.assets.end()) {
        info.phone_to_id = load_phone_vocab_asset(bundle_path / phone_asset->second);
    }
    const std::vector<std::string> vector_timing_inputs = {
        "expanded_boundary_event_ids",
        "expanded_modifier_event_ids",
        "expanded_phone_phase",
        "expanded_phone_log_duration",
    };
    for (const auto& item : info.component_inputs) {
        if (item.first.rfind("vector_estimator", 0) != 0) {
            continue;
        }
        const std::vector<std::string>& inputs = item.second;
        bool has_timing = false;
        for (const std::string& name : vector_timing_inputs) {
            has_timing = has_timing ||
                std::find(inputs.begin(), inputs.end(), name) != inputs.end();
        }
        if (!info.vector_timing_enabled) {
            if (has_timing) {
                throw std::runtime_error(
                    item.first + " exposes vector timing inputs without controls"
                );
            }
            continue;
        }
        if (inputs.size() < vector_timing_inputs.size() ||
            !std::equal(
                vector_timing_inputs.begin(),
                vector_timing_inputs.end(),
                inputs.end() - static_cast<std::ptrdiff_t>(vector_timing_inputs.size())
            )) {
            throw std::runtime_error(
                item.first + " must end with the vector timing input quartet"
            );
        }
        for (const std::string& name : vector_timing_inputs) {
            if (std::count(inputs.begin(), inputs.end(), name) != 1) {
                throw std::runtime_error(item.first + " has duplicate vector timing inputs");
            }
        }
    }
    if (info.vector_timing_enabled) {
        if (phone_asset == info.assets.end() ||
            info.vector_timing_phone_vocab_asset != phone_asset->second) {
            throw std::runtime_error(
                "Vector timing phone vocabulary asset does not match bundle"
            );
        }
        const std::string actual_vocab_sha = sha256_bytes(
            read_text_file(bundle_path / phone_asset->second)
        );
        if (actual_vocab_sha != info.vector_timing_phone_vocab_sha256) {
            throw std::runtime_error("Vector timing phone vocabulary SHA-256 mismatch");
        }
        std::vector<int> ids;
        ids.reserve(info.phone_to_id.size());
        for (const auto& item : info.phone_to_id) {
            ids.push_back(item.second);
        }
        std::sort(ids.begin(), ids.end());
        for (std::size_t index = 0; index < ids.size(); ++index) {
            if (ids[index] != static_cast<int>(index)) {
                throw std::runtime_error("Vector timing phone vocabulary IDs are not contiguous");
            }
        }
        if (static_cast<int>(ids.size()) != info.vector_timing_phone_vocab_size) {
            throw std::runtime_error("Vector timing phone vocabulary size mismatch");
        }
        std::vector<std::pair<int, std::string>> ordered;
        ordered.reserve(info.phone_to_id.size());
        for (const auto& item : info.phone_to_id) {
            ordered.emplace_back(item.second, item.first);
        }
        std::sort(ordered.begin(), ordered.end());
        std::vector<std::string> expected_punctuation;
        std::vector<int64_t> expected_punctuation_ids;
        std::vector<std::string> expected_modifiers;
        std::vector<int64_t> expected_modifier_ids;
        for (const auto& item : ordered) {
            if (scyllasband_detail::is_zero_duration_punctuation_phone(item.second)) {
                expected_punctuation.push_back(item.second);
                expected_punctuation_ids.push_back(item.first);
            }
            if (scyllasband_detail::is_non_acoustic_modifier_phone(item.second)) {
                expected_modifiers.push_back(item.second);
                expected_modifier_ids.push_back(item.first);
            }
        }
        if (info.vector_punctuation_symbols != expected_punctuation ||
            info.vector_punctuation_phone_ids != expected_punctuation_ids) {
            throw std::runtime_error("Vector timing punctuation mapping mismatch");
        }
        if (info.vector_modifier_events_enabled &&
            (info.vector_modifier_symbols != expected_modifiers ||
             info.vector_modifier_phone_ids != expected_modifier_ids)) {
            throw std::runtime_error("Vector timing modifier mapping mismatch");
        }
    }
    const auto g2p_tokenizer_asset = info.assets.find("g2p_tokenizer");
    if (g2p_tokenizer_asset != info.assets.end()) {
        const std::string tokenizer_json = read_text_file(bundle_path / g2p_tokenizer_asset->second);
        info.g2p_text_to_id = json_string_int_map_for_object(object_for_key(tokenizer_json, "text_symbols"));
        info.g2p_phoneme_symbols = json_int_string_map_for_object(object_for_key(tokenizer_json, "phoneme_symbols"));
        info.g2p_char_repeats = std::max(1, int_for_key(tokenizer_json, "char_repeats", 1));
        info.g2p_lowercase = bool_for_key(tokenizer_json, "lowercase", true);
        info.g2p_text_pad_index = int_for_key(tokenizer_json, "text_pad_index", 0);
        info.g2p_phoneme_pad_index = int_for_key(tokenizer_json, "phoneme_pad_index", 0);
        info.g2p_phoneme_end_index = int_for_key(tokenizer_json, "phoneme_end_index", 0);
    }
    const auto g2p_config_asset = info.assets.find("g2p_config");
    if (g2p_config_asset != info.assets.end()) {
        const std::string g2p_config_json = read_text_file(bundle_path / g2p_config_asset->second);
        if (info.g2p_text_tokens <= 0) {
            info.g2p_text_tokens = int_for_key(g2p_config_json, "fixed_text_tokens", 0);
        }
        if (info.punctuation_silence_target == "merge_into_punctuation") {
            const std::string target = string_for_key(g2p_config_json, "punctuation_silence_target");
            if (target == "explicit_silence") {
                info.punctuation_silence_target = target;
            }
        }
    }
    const auto g2p_normalization_asset = info.assets.find("g2p_normalization");
    if (g2p_normalization_asset != info.assets.end()) {
        const std::string normalization_json = read_text_file(
            bundle_path / g2p_normalization_asset->second
        );
        info.g2p_punctuation_token_remap = string_map_for_object(
            object_for_key(normalization_json, "punctuation_token_remap")
        );
        info.g2p_punctuation_token_remap_scope = string_for_key(
            normalization_json,
            "punctuation_token_remap_scope",
            "all_boundaries"
        );
        if (info.g2p_punctuation_token_remap_scope != "all_boundaries" &&
            info.g2p_punctuation_token_remap_scope != "continuation_only") {
            throw std::runtime_error("Unsupported punctuation_token_remap_scope in G2P normalization asset");
        }
        for (const auto& item : info.g2p_punctuation_token_remap) {
            if (info.phone_to_id.find(item.first) == info.phone_to_id.end()) {
                throw std::runtime_error(
                    "Punctuation remap source is absent from phone vocabulary: " + item.first
                );
            }
            if (info.phone_to_id.find(item.second) == info.phone_to_id.end()) {
                throw std::runtime_error(
                    "Punctuation remap target is absent from phone vocabulary: " + item.second
                );
            }
        }
    }
    const auto g2p_language_asset = info.assets.find("g2p_language_map");
    if (g2p_language_asset != info.assets.end()) {
        info.g2p_language_map = string_map_for_object(read_text_file(bundle_path / g2p_language_asset->second));
    }
    if (info.g2p_text_tokens <= 0) {
        info.g2p_text_tokens = 512;
    }
    if (info.voice_to_id.empty()) {
        for (std::size_t index = 0; index < info.voices.size(); ++index) {
            info.voice_to_id[info.voices[index]] = static_cast<int>(index);
        }
    }
    if (info.language_to_id.empty()) {
        for (std::size_t index = 0; index < info.languages.size(); ++index) {
            info.language_to_id[info.languages[index]] = static_cast<int>(index);
        }
    }
    if (info.emotion_to_id.empty()) {
        info.emotion_to_id["neutral"] = 0;
    }

    const std::regex embedding_pattern("\\\"embedding_path\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"");
    for (std::sregex_iterator it(manifest.begin(), manifest.end(), embedding_pattern), end; it != end; ++it) {
        require_exists(bundle_path / (*it)[1].str(), "Voice embedding is missing");
    }

    return info;
}

std::string target_bucket_frames_json(const std::vector<ScyllasBandTargetBucketInfo>& buckets) {
    std::ostringstream out;
    out << "[";
    for (std::size_t index = 0; index < buckets.size(); ++index) {
        if (index > 0) {
            out << ",";
        }
        out << buckets[index].latent_frames;
    }
    out << "]";
    return out.str();
}

std::string bundle_summary_json(const ScyllasBandBundleInfo& bundle) {
    const bool split_vector_available =
        bundle.component_artifacts.count("vector_estimator_prefix") > 0 &&
        bundle.component_artifacts.count("vector_estimator_tail") > 0;
    const bool coreai_gpu_ready =
        bundle.runtime_acceleration_metadata_present &&
        bundle.runtime_acceleration_backend == "coreai";
    const bool gpu_ready = coreai_gpu_ready ||
        (split_vector_available && !bundle.runtime_acceleration_cuda_required);
    std::string vector_execution = bundle.runtime_acceleration_vector_execution;
    if (vector_execution.empty()) {
        vector_execution = split_vector_available ? "split_prefix_tail" : "single_graph_or_unknown";
    }

    std::ostringstream metadata;
    metadata << "{"
             << "\"contract_version\":\"" << json_escape(bundle.contract_version) << "\","
             << "\"architecture\":\"" << json_escape(bundle.architecture) << "\","
             << "\"model_name\":\"" << json_escape(bundle.model_name) << "\","
             << "\"model_version\":\"" << json_escape(bundle.model_version) << "\","
             << "\"backend\":\"" << json_escape(bundle.selected_backend) << "\","
             << "\"sample_rate\":" << bundle.sample_rate << ","
             << "\"hop_length\":" << bundle.hop_length << ","
             << "\"latent_hop_length\":" << bundle.latent_hop_length << ","
             << "\"latent_dim\":" << bundle.latent_dim << ","
             << "\"phone_frames\":" << bundle.phone_frames << ","
             << "\"latent_frames\":" << bundle.latent_frames << ","
             << "\"target_bucket_count\":" << bundle.target_buckets.size() << ","
             << "\"target_bucket_margin_frames\":" << bundle.target_bucket_margin_frames << ","
             << "\"target_buckets\":" << target_bucket_frames_json(bundle.target_buckets) << ","
             << "\"voices\":" << bundle.voice_to_id.size() << ","
             << "\"languages\":" << bundle.language_to_id.size() << ","
             << "\"emotions\":" << bundle.emotion_to_id.size() << ","
             << "\"emotions_enabled\":" << (bundle.emotions_enabled ? "true" : "false") << ","
             << "\"components\":" << bundle.component_artifacts.size() << ","
             << "\"vector_split_available\":"
             << (split_vector_available ? "true" : "false") << ","
             << "\"runtime_acceleration\":{"
             << "\"metadata_present\":"
             << (bundle.runtime_acceleration_metadata_present ? "true" : "false") << ","
             << "\"backend\":";
    append_json_string_or_null(metadata, bundle.runtime_acceleration_backend);
    metadata << ",\"runtime_version\":";
    append_json_string_or_null(metadata, bundle.runtime_acceleration_runtime_version);
    metadata << ",\"gpu_ready\":" << (gpu_ready ? "true" : "false")
             << ",\"native_gpu_acceleration\":{"
             << "\"cuda_required\":"
             << (bundle.runtime_acceleration_cuda_required ? "true" : "false")
             << "},\"vector_estimator\":{"
             << "\"execution\":\"" << json_escape(vector_execution) << "\","
             << "\"policy\":";
    append_json_string_or_null(metadata, bundle.runtime_acceleration_vector_policy);
    metadata << ",\"metadata_split_artifacts_available\":"
             << (bundle.runtime_acceleration_vector_split_available ? "true" : "false")
             << ",\"min_gpu_split_vector_latent_frames\":"
             << bundle.runtime_acceleration_min_gpu_split_vector_latent_frames
             << ",\"split_artifacts_available\":"
             << (split_vector_available ? "true" : "false")
             << "}},"
             << "\"reference_packs_enabled\":" << (bundle.reference_packs_enabled ? "true" : "false") << ","
             << "\"reference_style_dim\":" << bundle.reference_style_dim << ","
             << "\"reference_prosody_dim\":" << bundle.reference_prosody_dim << ","
             << "\"prefix_conditioning_enabled\":" << (bundle.prefix_conditioning_enabled ? "true" : "false") << ","
             << "\"prefix_max_frames\":" << bundle.prefix_max_frames << ","
             << "\"span_conditioning_enabled\":" << (bundle.span_conditioning_enabled ? "true" : "false") << ","
             << "\"span_context_hidden_size\":" << bundle.span_context_hidden_size << ","
             << "\"span_context_max_phones\":" << bundle.span_context_max_phones << ","
             << "\"emotion_guidance_enabled\":" << (bundle.emotion_guidance_enabled ? "true" : "false") << ","
             << "\"affect_enabled\":" << (bundle.affect_enabled ? "true" : "false") << ","
             << "\"affect_graph_input_contract\":\"" << json_escape(bundle.affect_graph_input_contract) << "\","
             << "\"affect_axis_count\":" << bundle.affect_axes.size() << ","
             << "\"g2p_punctuation_token_remap_scope\":\""
             << json_escape(bundle.g2p_punctuation_token_remap_scope) << "\","
             << "\"g2p_punctuation_token_remap_entries\":"
             << bundle.g2p_punctuation_token_remap.size() << ","
             << "\"word_boundaries_enabled\":" << (bundle.word_boundaries_enabled ? "true" : "false") << ","
             << "\"word_boundary_duration_phone\":\"" << json_escape(bundle.word_boundary_duration_phone) << "\","
             << "\"word_boundary_presence_threshold_frames\":" << bundle.word_boundary_presence_threshold_frames << ","
             << "\"duration_pause_presence_enabled\":"
             << (bundle.duration_pause_presence_enabled ? "true" : "false") << ","
             << "\"duration_pause_presence_threshold_probability\":"
             << bundle.duration_pause_presence_threshold_probability << ","
             << "\"punctuation_silence_target\":\"" << json_escape(bundle.punctuation_silence_target) << "\","
             << "\"punctuation_pause_floors_calibrated\":"
             << (bundle.punctuation_pause_floors_calibrated ? "true" : "false") << ","
             << "\"punctuation_pause_floor_entries\":"
             << bundle.punctuation_pause_floor_table_ms.size() << ","
             << "\"punctuation_pause_floor_table_sha256\":";
    append_json_string_or_null(metadata, bundle.punctuation_pause_floor_table_sha256);
    metadata
             << "}";
    return metadata.str();
}

}  // namespace scyllasband_detail
