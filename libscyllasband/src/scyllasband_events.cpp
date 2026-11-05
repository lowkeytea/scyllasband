#include "scyllasband_events.h"

#include "scyllasband_unicode.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <set>
#include <stdexcept>

namespace scyllasband {

const char* const kSilencePhone = "<sil>";
const char* const kDeliveryControls[3] = {"intensity", "pace", "pitch"};
const double kDeliveryRanges[3][2] = {{-2.0, 2.0}, {-2.0, 2.0}, {-2.0, 2.0}};

namespace {

const std::set<std::string>& punctuation_phones() {
    static const std::set<std::string> phones = {"<pause_comma>", "<pause_semicolon>", "<pause_colon>", "<pause_dash>",
                                                 "<ellipsis>",    "<end_stmt>",        "<end_question>", "<end_exclaim>"};
    return phones;
}

int64_t sentence_type(const std::string& phone) {
    if (phone == "<end_stmt>") return 1;
    if (phone == "<end_question>") return 2;
    if (phone == "<end_exclaim>") return 3;
    if (phone == "<ellipsis>") return 4;
    return -1;
}

using Span = std::optional<std::pair<int64_t, int64_t>>;

bool is_break(const std::string& phone) { return phone == kSilencePhone || is_punctuation_phone(phone); }

Span next_segment_span(const std::vector<std::string>& phones, const std::vector<Span>& spans, std::size_t start) {
    for (std::size_t index = start; index < spans.size(); ++index) {
        if (is_break(phones[index])) return std::nullopt;
        if (spans[index]) return spans[index];
    }
    return std::nullopt;
}

Span next_acoustic_span(const std::vector<std::string>& phones, const std::vector<Span>& spans, std::ptrdiff_t index, std::ptrdiff_t step) {
    while (index >= 0 && index < static_cast<std::ptrdiff_t>(phones.size())) {
        const std::string& phone = phones[static_cast<std::size_t>(index)];
        if (is_break(phone)) return std::nullopt;
        if (spans[static_cast<std::size_t>(index)] && !is_non_acoustic_modifier(phone)) return spans[static_cast<std::size_t>(index)];
        index += step;
    }
    return std::nullopt;
}

Span postfix_owner_span(const std::vector<std::string>& phones, const std::vector<Span>& spans, std::ptrdiff_t modifier_index) {
    std::ptrdiff_t index = modifier_index - 1;
    while (index >= 0 && is_non_acoustic_modifier(phones[static_cast<std::size_t>(index)])) --index;
    if (index < 0 || is_break(phones[static_cast<std::size_t>(index)])) return std::nullopt;
    return spans[static_cast<std::size_t>(index)];
}

bool python_float(const std::string& text, double& value) {
    // float() syntax for plain decimal literals: [+-] digits [. digits] [e [+-] digits]
    std::size_t i = 0, n = text.size();
    if (i < n && (text[i] == '+' || text[i] == '-')) ++i;
    std::size_t digits = 0;
    while (i < n && std::isdigit(static_cast<unsigned char>(text[i]))) ++i, ++digits;
    if (i < n && text[i] == '.') {
        ++i;
        while (i < n && std::isdigit(static_cast<unsigned char>(text[i]))) ++i, ++digits;
    }
    if (digits == 0) return false;
    if (i < n && (text[i] == 'e' || text[i] == 'E')) {
        ++i;
        if (i < n && (text[i] == '+' || text[i] == '-')) ++i;
        std::size_t exponent = 0;
        while (i < n && std::isdigit(static_cast<unsigned char>(text[i]))) ++i, ++exponent;
        if (exponent == 0) return false;
    }
    if (i != n) return false;
    value = std::strtod(text.c_str(), nullptr);
    return true;
}

}  // namespace

bool is_non_acoustic_modifier(const std::string& phone) {
    if (phone == "ˈ" || phone == "ˌ" || phone == "ː" || phone == "ˑ") return true;
    if (phone.empty() || phone[0] == '<') return false;
    unicode::Text text = unicode::decode(phone);
    return std::all_of(text.begin(), text.end(), [](char32_t cp) { return unicode::is_mark(cp); });
}

bool is_punctuation_pause(const std::string& phone) {
    return phone.rfind("<pause", 0) == 0 || phone.rfind("<end_", 0) == 0 || phone == "<ellipsis>";
}

bool is_punctuation_phone(const std::string& phone) { return punctuation_phones().count(phone) > 0; }

std::vector<int> frames_to_durations(const std::vector<float>& values, const std::vector<std::string>& phones, double scale,
                                     double pause_scale) {
    std::vector<int> durations;
    durations.reserve(phones.size());
    for (std::size_t i = 0; i < phones.size() && i < values.size(); ++i) {
        const double factor = scale * (phones[i] == kSilencePhone ? pause_scale : 1.0);
        int frames = static_cast<int>(std::nearbyint(std::max(0.0, static_cast<double>(values[i])) * factor));
        if (phones[i] == kSilencePhone) {
        } else if (is_non_acoustic_modifier(phones[i]) || is_punctuation_pause(phones[i])) {
            frames = 0;
        } else {
            frames = std::max(1, frames);
        }
        durations.push_back(frames);
    }
    return durations;
}

std::map<std::string, int64_t> modifier_bits(const std::map<std::string, int>& token_to_id) {
    std::vector<std::pair<int, std::string>> tokens;
    for (const auto& [token, id] : token_to_id) {
        if (is_non_acoustic_modifier(token)) tokens.emplace_back(id, token);
    }
    std::stable_sort(tokens.begin(), tokens.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    std::map<std::string, int64_t> bits;
    for (std::size_t bit = 0; bit < tokens.size(); ++bit) bits[tokens[bit].second] = int64_t{1} << bit;
    return bits;
}

std::vector<int64_t> sentence_types(const std::vector<std::string>& phones, const std::vector<std::string>& following) {
    int64_t current = 0;
    for (const std::string& token : following) {
        if (sentence_type(token) >= 0) {
            current = sentence_type(token);
            break;
        }
    }
    std::vector<int64_t> types(phones.size(), 0);
    for (std::size_t index = phones.size(); index-- > 0;) {
        if (sentence_type(phones[index]) >= 0) current = sentence_type(phones[index]);
        types[index] = current;
    }
    return types;
}

FrameEvents frame_events(const std::vector<std::string>& phones, const std::vector<int>& durations, const std::vector<int>& word_starts,
                         const std::vector<std::string>& following, const std::map<std::string, int>& token_to_id,
                         const std::map<std::string, int64_t>& bits) {
    const std::size_t count = phones.size();
    int64_t length = 0;
    for (int frames : durations) length += frames;
    FrameEvents out;
    out.boundary.assign(static_cast<std::size_t>(length), 0);
    out.modifier.assign(static_cast<std::size_t>(length), 0);
    out.phase.assign(static_cast<std::size_t>(length), 0.0f);
    out.log_duration.assign(static_cast<std::size_t>(length), 0.0f);
    std::vector<Span> spans;
    int64_t cursor = 0;
    for (int frames : durations) {
        if (frames <= 0) {
            spans.emplace_back(std::nullopt);
            continue;
        }
        spans.emplace_back(std::make_pair(cursor, cursor + frames));
        const float log_frames = static_cast<float>(std::log1p(static_cast<double>(frames)));
        for (int k = 0; k < frames; ++k) {
            out.phase[static_cast<std::size_t>(cursor + k)] = (static_cast<float>(k) + 0.5f) / static_cast<float>(frames);
            out.log_duration[static_cast<std::size_t>(cursor + k)] = log_frames;
        }
        cursor += frames;
    }
    auto fill = [](std::vector<int64_t>& target, const std::pair<int64_t, int64_t>& span, int64_t value) {
        for (int64_t i = span.first; i < span.second; ++i) target[static_cast<std::size_t>(i)] = value;
    };

    for (std::size_t index = 0; index < count; ++index) {
        const std::string& phone = phones[index];
        if (is_punctuation_phone(phone)) {
            std::size_t owner = index + 1;  // stacked marks share the <sil> after the run; the last one names the event
            while (owner < count && is_punctuation_phone(phones[owner])) ++owner;
            if (owner >= count || phones[owner] != kSilencePhone) throw std::invalid_argument("Punctuation " + phone + " lacks its following <sil>");
            const int64_t event = token_to_id.at(phone);
            if (spans[owner]) {
                fill(out.boundary, *spans[owner], event);
            } else if (Span next = next_segment_span(phones, spans, owner + 1)) {
                out.boundary[static_cast<std::size_t>(next->first)] = event;
            }
        }
        if (is_non_acoustic_modifier(phone)) {
            const bool stress = phone == "ˈ" || phone == "ˌ";
            const std::ptrdiff_t i = static_cast<std::ptrdiff_t>(index);
            Span owner = stress ? next_acoustic_span(phones, spans, i + 1, 1) : postfix_owner_span(phones, spans, i);
            if (stress && !owner) owner = next_acoustic_span(phones, spans, i - 1, -1);
            auto bit = bits.find(phone);
            if (owner && bit != bits.end() && bit->second) {
                for (int64_t f = owner->first; f < owner->second; ++f) out.modifier[static_cast<std::size_t>(f)] |= bit->second;
            }
        }
    }

    const int64_t word_start_id = static_cast<int64_t>(token_to_id.size());
    for (int index : word_starts) {
        if (index < 0 || static_cast<std::size_t>(index) >= count) continue;
        const Span& span = spans[static_cast<std::size_t>(index)];
        if (span) {
            bool marked = false;
            for (int64_t f = span->first; f < span->second; ++f) marked = marked || out.boundary[static_cast<std::size_t>(f)] > 0;
            if (!marked) fill(out.boundary, *span, word_start_id);
            continue;
        }
        Span next = next_segment_span(phones, spans, static_cast<std::size_t>(index) + 1);
        if (next && out.boundary[static_cast<std::size_t>(next->first)] == 0) out.boundary[static_cast<std::size_t>(next->first)] = word_start_id;
    }

    std::vector<int64_t> types = sentence_types(phones, following);
    out.sentence.reserve(static_cast<std::size_t>(length));
    for (std::size_t i = 0; i < count; ++i) out.sentence.insert(out.sentence.end(), static_cast<std::size_t>(std::max(0, durations[i])), types[i]);
    return out;
}

std::pair<std::vector<int64_t>, std::vector<int64_t>> balanced_context(const std::vector<int64_t>& before, const std::vector<int64_t>& target_in,
                                                                       const std::vector<int64_t>& after, std::size_t max_phones) {
    std::vector<int64_t> target(target_in.begin(), target_in.begin() + static_cast<std::ptrdiff_t>(std::min(target_in.size(), max_phones)));
    if (target.size() >= max_phones) return {target, std::vector<int64_t>(target.size(), 1)};
    const std::size_t remaining = max_phones - target.size();
    std::size_t before_keep = std::min(before.size(), (remaining + 1) / 2);
    std::size_t after_keep = std::min(after.size(), remaining - before_keep);
    std::size_t spare = remaining - before_keep - after_keep;
    if (spare > 0 && before_keep < before.size()) {
        std::size_t extra = std::min(spare, before.size() - before_keep);
        before_keep += extra;
        spare -= extra;
    }
    if (spare > 0 && after_keep < after.size()) after_keep += std::min(spare, after.size() - after_keep);
    std::vector<int64_t> span(before.end() - static_cast<std::ptrdiff_t>(before_keep), before.end());
    span.insert(span.end(), target.begin(), target.end());
    span.insert(span.end(), after.begin(), after.begin() + static_cast<std::ptrdiff_t>(after_keep));
    std::vector<int64_t> segments(before_keep, 0);
    segments.insert(segments.end(), target.size(), 1);
    segments.insert(segments.end(), after_keep, 2);
    return {span, segments};
}

Delivery Delivery::parse(const char* spec) {
    Delivery out;
    if (spec == nullptr) return out;
    std::string value(spec);
    std::string folded = unicode::encode(unicode::lower(unicode::strip(unicode::decode(value))));
    if (folded.empty() || folded == "default" || folded == "neutral") return out;
    std::vector<std::pair<std::string, std::string>> fields;
    std::size_t start = 0;
    while (true) {
        std::size_t comma = folded.find(',', start);
        std::string part = unicode::encode(unicode::strip(unicode::decode(folded.substr(start, comma == std::string::npos ? std::string::npos : comma - start))));
        std::size_t eq = part.find('=');
        std::string key = eq == std::string::npos ? part : part.substr(0, eq);
        bool duplicate = std::any_of(fields.begin(), fields.end(), [&](const auto& f) { return f.first == key; });
        if (eq == std::string::npos || duplicate) {
            throw std::invalid_argument("Delivery must be unique control=value entries, e.g. intensity=1,pace=-0.5,pitch=0");
        }
        fields.emplace_back(key, unicode::encode(unicode::strip(unicode::decode(part.substr(eq + 1)))));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    static const std::set<std::string> old_keys = {"energy", "tension", "valence", "assertiveness", "whisper", "emotion", "kind"};
    for (const auto& [key, raw] : fields) {
        if (old_keys.count(key)) {
            throw std::invalid_argument("This model takes intensity, pace and pitch (energy, tension, valence, assertiveness and emotion "
                                        "styles belonged to the v2 models)");
        }
    }
    for (const auto& field : fields) {
        const std::string& key = field.first;
        auto control = std::find_if(std::begin(kDeliveryControls), std::end(kDeliveryControls), [&](const char* name) { return key == name; });
        if (control == std::end(kDeliveryControls)) throw std::invalid_argument("Delivery accepts only intensity, pace, pitch; got " + key);
    }
    for (std::size_t index = 0; index < 3; ++index) {
        const std::string key = kDeliveryControls[index];
        for (const auto& [name, raw] : fields) {
            if (name != key) continue;
            if (raw.empty() || raw == "default" || raw == "auto") break;
            double number = 0.0;
            if (!python_float(raw, number)) throw std::invalid_argument("Delivery " + key + " must be a number");
            if (!std::isfinite(number)) throw std::invalid_argument("Delivery " + key + " must be finite");
            out.controls[index] = std::min(std::max(number, kDeliveryRanges[index][0]), kDeliveryRanges[index][1]);
        }
    }
    return out;
}

Delivery Delivery::from_json(const Json& value) {
    if (value.is_null()) return Delivery{};
    if (value.is_string()) return parse(value.str().c_str());
    if (!value.is_object()) throw std::invalid_argument("Delivery must be a mapping, a control=value string or None");
    static const std::set<std::string> old_keys = {"energy", "tension", "valence", "assertiveness", "whisper", "emotion", "kind"};
    std::vector<std::string> extra;
    for (const auto& [key, raw] : value.items()) {
        if (old_keys.count(key)) {
            throw std::invalid_argument("This model takes intensity, pace and pitch (energy, tension, valence, assertiveness and emotion "
                                        "styles belonged to the v2 models)");
        }
    }
    for (const auto& item : value.items()) {
        const std::string& key = item.first;
        if (std::none_of(std::begin(kDeliveryControls), std::end(kDeliveryControls), [&](const char* name) { return key == name; })) {
            extra.push_back(key);
        }
    }
    if (!extra.empty()) {
        std::sort(extra.begin(), extra.end());
        std::string list;
        for (const std::string& key : extra) list += (list.empty() ? "" : ", ") + key;
        throw std::invalid_argument("Delivery accepts only intensity, pace, pitch; got " + list);
    }
    Delivery out;
    for (std::size_t index = 0; index < 3; ++index) {
        const std::string key = kDeliveryControls[index];
        const Json& raw = value.get(key);
        if (raw.is_null()) continue;
        double number = 0.0;
        if (raw.is_string()) {
            const std::string text = unicode::encode(unicode::strip(unicode::decode(raw.str())));
            const std::string folded = unicode::encode(unicode::lower(unicode::decode(text)));
            if (folded.empty() || folded == "default" || folded == "auto") continue;
            if (folded == "inf" || folded == "+inf" || folded == "-inf" || folded == "infinity" || folded == "+infinity" ||
                folded == "-infinity" || folded == "nan" || folded == "+nan" || folded == "-nan") {
                throw std::invalid_argument("Delivery " + key + " must be finite");
            }
            if (!python_float(text, number)) throw std::invalid_argument("Delivery " + key + " must be a number");
        } else if (raw.is_number()) {
            number = raw.num();
        } else {
            throw std::invalid_argument("Delivery " + key + " must be a number");   // booleans, lists and mappings
        }
        if (!std::isfinite(number)) throw std::invalid_argument("Delivery " + key + " must be finite");
        out.controls[index] = std::min(std::max(number, kDeliveryRanges[index][0]), kDeliveryRanges[index][1]);
    }
    return out;
}

Json Delivery::to_json() const {
    Json out{Json::Object{}};
    for (std::size_t i = 0; i < 3; ++i) out.set(kDeliveryControls[i], controls[i] ? Json(*controls[i]) : Json());
    return out;
}

std::string Delivery::spec() const {
    std::string out;
    for (std::size_t i = 0; i < 3; ++i) {
        out += (i ? "," : "") + std::string(kDeliveryControls[i]) + "=" + (controls[i] ? format_float(*controls[i]) : "default");
    }
    return out;
}

}  // namespace scyllasband
