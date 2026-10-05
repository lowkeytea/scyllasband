// Minimal JSON document model: parse bundle assets, write metadata.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace scyllasband {

class Json {
public:
    enum class Type { null, boolean, number, string, array, object };
    using Array = std::vector<Json>;
    using Object = std::vector<std::pair<std::string, Json>>;  // insertion order preserved

    Json() = default;
    Json(std::nullptr_t) {}
    Json(bool value) : type_(Type::boolean), bool_(value) {}
    Json(int value) : type_(Type::number), number_(value), integral_(true) {}
    Json(int64_t value) : type_(Type::number), number_(static_cast<double>(value)), integral_(true) {}
    Json(std::size_t value) : type_(Type::number), number_(static_cast<double>(value)), integral_(true) {}
    Json(double value) : type_(Type::number), number_(value) {}
    Json(const char* value) : type_(Type::string), string_(value) {}
    Json(std::string value) : type_(Type::string), string_(std::move(value)) {}
    Json(Array value) : type_(Type::array), array_(std::move(value)) {}
    Json(Object value) : type_(Type::object), object_(std::move(value)) {}

    static Json parse(const std::string& text);
    static Json parse_file(const std::string& path);
    template <typename T>
    static Json array_of(const std::vector<T>& values) {
        Array out;
        out.reserve(values.size());
        for (const T& value : values) out.emplace_back(value);
        return Json(std::move(out));
    }

    Type type() const { return type_; }
    bool is_null() const { return type_ == Type::null; }
    bool is_object() const { return type_ == Type::object; }
    bool is_array() const { return type_ == Type::array; }
    bool is_string() const { return type_ == Type::string; }
    bool is_number() const { return type_ == Type::number; }

    // Object access: `get` returns a shared null for missing keys or non-objects.
    const Json& get(const std::string& key) const;
    bool has(const std::string& key) const;
    Json& set(const std::string& key, Json value);
    const Object& items() const { return object_; }
    const Array& elements() const { return array_; }
    Array& elements() { return array_; }
    void push(Json value) { array_.push_back(std::move(value)); }
    std::size_t size() const { return type_ == Type::array ? array_.size() : object_.size(); }
    const Json& operator[](std::size_t index) const { return array_.at(index); }

    std::string str(const std::string& fallback = {}) const { return type_ == Type::string ? string_ : fallback; }
    double num(double fallback = 0.0) const { return type_ == Type::number ? number_ : fallback; }
    int64_t integer(int64_t fallback = 0) const { return type_ == Type::number ? static_cast<int64_t>(number_) : fallback; }
    bool boolean(bool fallback = false) const { return type_ == Type::boolean ? bool_ : fallback; }

    std::string dump() const;

private:
    void write(std::string& out) const;

    Type type_ = Type::null;
    bool bool_ = false;
    double number_ = 0.0;
    bool integral_ = false;
    std::string string_;
    Array array_;
    Object object_;
};

// Python repr()-style float text: the shortest form that round-trips, with ".0" on integral values.
std::string format_float(double value);

}  // namespace scyllasband
