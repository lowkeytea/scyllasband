#include "scyllasband_json.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace scyllasband {
namespace {

class Parser {
public:
    explicit Parser(const std::string& text) : text_(text) {}

    Json document() {
        Json value = parse_value();
        skip_ws();
        if (pos_ != text_.size()) fail("trailing characters");
        return value;
    }

private:
    [[noreturn]] void fail(const char* what) const {
        throw std::runtime_error(std::string("JSON parse error at offset ") + std::to_string(pos_) + ": " + what);
    }

    void skip_ws() {
        while (pos_ < text_.size() && (text_[pos_] == ' ' || text_[pos_] == '\t' || text_[pos_] == '\n' || text_[pos_] == '\r')) ++pos_;
    }

    bool consume(const char* literal) {
        std::size_t length = std::char_traits<char>::length(literal);
        if (text_.compare(pos_, length, literal) != 0) return false;
        pos_ += length;
        return true;
    }

    Json parse_value() {
        skip_ws();
        if (pos_ >= text_.size()) fail("unexpected end");
        char ch = text_[pos_];
        if (ch == '{') return parse_object();
        if (ch == '[') return parse_array();
        if (ch == '"') return Json(parse_string());
        if (consume("true")) return Json(true);
        if (consume("false")) return Json(false);
        if (consume("null")) return Json();
        if (consume("NaN")) return Json(std::nan(""));
        if (consume("Infinity")) return Json(HUGE_VAL);
        if (consume("-Infinity")) return Json(-HUGE_VAL);
        return parse_number();
    }

    Json parse_number() {
        std::size_t start = pos_;
        bool integral = true;
        if (text_[pos_] == '-') ++pos_;
        while (pos_ < text_.size()) {
            char ch = text_[pos_];
            if (ch >= '0' && ch <= '9') {
                ++pos_;
            } else if (ch == '.' || ch == 'e' || ch == 'E' || ch == '+' || ch == '-') {
                integral = false;
                ++pos_;
            } else {
                break;
            }
        }
        if (pos_ == start) fail("unexpected character");
        std::string token = text_.substr(start, pos_ - start);
        char* end = nullptr;
        double value = std::strtod(token.c_str(), &end);
        if (end == nullptr || *end != '\0') fail("invalid number");
        if (integral) return Json(static_cast<int64_t>(std::strtoll(token.c_str(), nullptr, 10)));
        return Json(value);
    }

    unsigned hex4() {
        if (pos_ + 4 > text_.size()) fail("short unicode escape");
        unsigned value = 0;
        for (int i = 0; i < 4; ++i) {
            char ch = text_[pos_++];
            value <<= 4;
            if (ch >= '0' && ch <= '9') value |= static_cast<unsigned>(ch - '0');
            else if (ch >= 'a' && ch <= 'f') value |= static_cast<unsigned>(ch - 'a' + 10);
            else if (ch >= 'A' && ch <= 'F') value |= static_cast<unsigned>(ch - 'A' + 10);
            else fail("bad unicode escape");
        }
        return value;
    }

    static void append_utf8(std::string& out, uint32_t cp) {
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

    std::string parse_string() {
        ++pos_;  // opening quote
        std::string out;
        while (true) {
            if (pos_ >= text_.size()) fail("unterminated string");
            char ch = text_[pos_++];
            if (ch == '"') return out;
            if (ch != '\\') {
                out.push_back(ch);
                continue;
            }
            if (pos_ >= text_.size()) fail("bad escape");
            char esc = text_[pos_++];
            switch (esc) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    uint32_t cp = hex4();
                    if (cp >= 0xD800 && cp < 0xDC00 && text_.compare(pos_, 2, "\\u") == 0) {
                        std::size_t saved = pos_;
                        pos_ += 2;
                        uint32_t low = hex4();
                        if (low >= 0xDC00 && low < 0xE000) cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                        else pos_ = saved;
                    }
                    append_utf8(out, cp);
                    break;
                }
                default: fail("bad escape");
            }
        }
    }

    Json parse_array() {
        ++pos_;
        Json::Array out;
        skip_ws();
        if (pos_ < text_.size() && text_[pos_] == ']') {
            ++pos_;
            return Json(std::move(out));
        }
        while (true) {
            out.push_back(parse_value());
            skip_ws();
            if (pos_ >= text_.size()) fail("unterminated array");
            char ch = text_[pos_++];
            if (ch == ']') return Json(std::move(out));
            if (ch != ',') fail("expected , or ]");
        }
    }

    Json parse_object() {
        ++pos_;
        Json out{Json::Object{}};
        skip_ws();
        if (pos_ < text_.size() && text_[pos_] == '}') {
            ++pos_;
            return out;
        }
        while (true) {
            skip_ws();
            if (pos_ >= text_.size() || text_[pos_] != '"') fail("expected key");
            std::string key = parse_string();
            skip_ws();
            if (pos_ >= text_.size() || text_[pos_++] != ':') fail("expected :");
            out.set(key, parse_value());
            skip_ws();
            if (pos_ >= text_.size()) fail("unterminated object");
            char ch = text_[pos_++];
            if (ch == '}') return out;
            if (ch != ',') fail("expected , or }");
        }
    }

    const std::string& text_;
    std::size_t pos_ = 0;
};

void write_string(std::string& out, const std::string& value) {
    out.push_back('"');
    for (unsigned char ch : value) {
        switch (ch) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (ch < 0x20) {
                    char buffer[8];
                    std::snprintf(buffer, sizeof(buffer), "\\u%04x", ch);
                    out += buffer;
                } else {
                    out.push_back(static_cast<char>(ch));
                }
        }
    }
    out.push_back('"');
}

}  // namespace

std::string format_float(double value) {
    if (std::isnan(value)) return "NaN";
    if (std::isinf(value)) return value > 0 ? "Infinity" : "-Infinity";
    if (value == 0.0) return std::signbit(value) ? "-0.0" : "0.0";
    char buffer[64];
    int precision = 1;
    for (; precision <= 17; ++precision) {
        std::snprintf(buffer, sizeof(buffer), "%.*e", precision - 1, value);
        if (std::strtod(buffer, nullptr) == value) break;
    }
    std::string text(buffer);
    bool negative = text[0] == '-';
    if (negative) text.erase(0, 1);
    std::size_t e_pos = text.find('e');
    int exponent = std::atoi(text.c_str() + e_pos + 1);
    std::string digits;
    for (std::size_t i = 0; i < e_pos; ++i) {
        if (text[i] != '.') digits.push_back(text[i]);
    }
    while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
    std::string out = negative ? "-" : "";
    if (exponent >= -5 + 1 && exponent < 16) {
        if (exponent >= 0) {
            std::string whole = digits.substr(0, std::min<std::size_t>(digits.size(), static_cast<std::size_t>(exponent) + 1));
            while (whole.size() < static_cast<std::size_t>(exponent) + 1) whole.push_back('0');
            std::string fraction = digits.size() > whole.size() ? digits.substr(whole.size()) : "";
            out += whole + "." + (fraction.empty() ? "0" : fraction);
        } else {
            out += "0." + std::string(static_cast<std::size_t>(-exponent - 1), '0') + digits;
        }
    } else {
        out += digits.substr(0, 1);
        if (digits.size() > 1) out += "." + digits.substr(1);
        char exp_buffer[16];
        std::snprintf(exp_buffer, sizeof(exp_buffer), "e%c%02d", exponent < 0 ? '-' : '+', std::abs(exponent));
        out += exp_buffer;
    }
    return out;
}

Json Json::parse(const std::string& text) { return Parser(text).document(); }

Json Json::parse_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot read " + path);
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return parse(buffer.str());
}

const Json& Json::get(const std::string& key) const {
    static const Json null_value;
    if (type_ != Type::object) return null_value;
    for (const auto& item : object_) {
        if (item.first == key) return item.second;
    }
    return null_value;
}

bool Json::has(const std::string& key) const {
    if (type_ != Type::object) return false;
    for (const auto& item : object_) {
        if (item.first == key) return true;
    }
    return false;
}

Json& Json::set(const std::string& key, Json value) {
    if (type_ != Type::object) {
        *this = Json(Object{});
    }
    for (auto& item : object_) {
        if (item.first == key) {
            item.second = std::move(value);
            return item.second;
        }
    }
    object_.emplace_back(key, std::move(value));
    return object_.back().second;
}

std::string Json::dump() const {
    std::string out;
    write(out);
    return out;
}

void Json::write(std::string& out) const {
    switch (type_) {
        case Type::null: out += "null"; break;
        case Type::boolean: out += bool_ ? "true" : "false"; break;
        case Type::number:
            if (integral_) out += std::to_string(static_cast<int64_t>(number_));
            else out += format_float(number_);
            break;
        case Type::string: write_string(out, string_); break;
        case Type::array:
            out.push_back('[');
            for (std::size_t i = 0; i < array_.size(); ++i) {
                if (i) out += ", ";
                array_[i].write(out);
            }
            out.push_back(']');
            break;
        case Type::object:
            out.push_back('{');
            for (std::size_t i = 0; i < object_.size(); ++i) {
                if (i) out += ", ";
                write_string(out, object_[i].first);
                out += ": ";
                object_[i].second.write(out);
            }
            out.push_back('}');
            break;
    }
}

}  // namespace scyllasband
