// Unit tests against fixtures produced by the Python reference runtime (tests/generate_fixtures.py).
// Usage: scyllasband_unit_test <suite> <fixtures dir>
#include "scyllasband_events.h"
#include "scyllasband_json.h"
#include "scyllasband_text.h"
#include "scyllasband_unicode.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace scyllasband;

namespace {

int g_failures = 0;
int g_checks = 0;

void expect(bool condition, const std::string& what) {
    ++g_checks;
    if (!condition) {
        ++g_failures;
        if (g_failures <= 20) std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    }
}

std::vector<std::string> strings(const Json& value) {
    std::vector<std::string> out;
    for (const Json& item : value.elements()) out.push_back(item.str());
    return out;
}

std::vector<int64_t> ints(const Json& value) {
    std::vector<int64_t> out;
    for (const Json& item : value.elements()) out.push_back(item.integer());
    return out;
}

std::vector<std::string> encode_all(const std::vector<Text>& values) {
    std::vector<std::string> out;
    for (const Text& value : values) out.push_back(unicode::encode(value));
    return out;
}

std::string label(const Json& text) { return text.dump().substr(0, 80); }

void events_suite(const Json& fixture) {
    std::map<std::string, int> token_to_id;
    for (const auto& [token, id] : fixture.get("token_to_id").items()) token_to_id[token] = static_cast<int>(id.integer());
    const auto bits = modifier_bits(token_to_id);
    for (const Json& c : fixture.get("cases").elements()) {
        std::vector<int> durations, word_starts;
        for (int64_t value : ints(c.get("durations"))) durations.push_back(static_cast<int>(value));
        for (int64_t value : ints(c.get("word_starts"))) word_starts.push_back(static_cast<int>(value));
        FrameEvents events = frame_events(strings(c.get("phones")), durations, word_starts, strings(c.get("following")), token_to_id, bits);
        const Json& expected = c.get("expected");
        const std::string name = label(c.get("phones"));
        expect(events.boundary == ints(expected.get("expanded_boundary_event_ids")), "boundary events " + name);
        expect(events.modifier == ints(expected.get("expanded_modifier_event_ids")), "modifier events " + name);
        expect(events.sentence == ints(expected.get("expanded_sentence_type_ids")), "sentence types " + name);
        const auto& phase = expected.get("expanded_phone_phase").elements();
        const auto& log_duration = expected.get("expanded_phone_log_duration").elements();
        bool same = phase.size() == events.phase.size() && log_duration.size() == events.log_duration.size();
        for (std::size_t i = 0; same && i < phase.size(); ++i) {
            same = static_cast<float>(phase[i].num()) == events.phase[i] && static_cast<float>(log_duration[i].num()) == events.log_duration[i];
        }
        expect(same, "phase and log duration " + name);
    }
}

void durations_suite(const Json& fixture) {
    for (const Json& c : fixture.get("cases").elements()) {
        std::vector<float> values;
        for (const Json& v : c.get("values").elements()) values.push_back(static_cast<float>(v.num()));
        std::vector<int> got = frames_to_durations(values, strings(c.get("phones")), c.get("scale").num());
        std::vector<int64_t> want = ints(c.get("expected"));
        expect(std::vector<int64_t>(got.begin(), got.end()) == want, "durations " + label(c.get("values")));
    }
}

void context_suite(const Json& fixture) {
    for (const Json& c : fixture.get("cases").elements()) {
        auto [span, segments] = balanced_context(ints(c.get("before")), ints(c.get("target")), ints(c.get("after")),
                                                 static_cast<std::size_t>(c.get("width").integer()));
        expect(span == ints(c.get("expected").get("span")) && segments == ints(c.get("expected").get("segments")),
               "balanced context width " + std::to_string(c.get("width").integer()));
    }
}

void delivery_suite(const Json& fixture) {
    for (const Json& c : fixture.get("cases").elements()) {
        const Json& spec = c.get("spec");
        const std::string text = spec.str();
        // The C ABI reads an empty string as "unset" (neutral); Python rejects it.
        const bool c_neutral = spec.is_string() && text.empty();
        bool error = false;
        Delivery delivery;
        try {
            delivery = Delivery::parse(spec.is_null() ? nullptr : text.c_str());
        } catch (const std::invalid_argument&) {
            error = true;
        }
        const std::string name = "delivery " + spec.dump();
        if (c_neutral) {
            expect(!error && delivery.spec() == Delivery().spec(), name);
            continue;
        }
        expect(error == c.get("error").boolean(), name + " error state");
        if (error || c.get("error").boolean()) continue;
        float values[5];
        uint8_t present[5];
        delivery.tensors(values, present);
        for (int i = 0; i < 5; ++i) {
            expect(values[i] == static_cast<float>(c.get("values")[static_cast<std::size_t>(i)].num()), name + " value " + std::to_string(i));
            expect(present[i] == c.get("present")[static_cast<std::size_t>(i)].integer(), name + " present " + std::to_string(i));
        }
        expect(delivery.to_json().dump() == c.get("resolved").dump(), name + " resolved " + delivery.to_json().dump());
    }
}

void sentences_suite(const Json& fixture) {
    for (const Json& c : fixture.get("sentences").elements()) {
        expect(encode_all(split_sentences(unicode::decode(c.get("text").str()))) == strings(c.get("expected")), "split_sentences " + label(c.get("text")));
    }
    for (const Json& c : fixture.get("paragraphs").elements()) {
        expect(encode_all(split_paragraphs(unicode::decode(c.get("text").str()))) == strings(c.get("expected")), "paragraphs " + label(c.get("text")));
    }
    for (const Json& c : fixture.get("retry").elements()) {
        bool error = false;
        std::vector<std::string> got;
        try {
            got = encode_all(split_for_retry(unicode::decode(c.get("text").str()), c.get("min_share").num(0.0)));
        } catch (const std::exception&) {
            error = true;
        }
        if (c.get("error").boolean()) expect(error, "split_for_retry error " + label(c.get("text")));
        else expect(!error && got == strings(c.get("expected")), "split_for_retry " + label(c.get("text")));
    }
}

void segments_suite(const Json& fixture) {
    for (const Json& c : fixture.get("phrase").elements()) {
        expect(unicode::encode(normalize_phrase_punctuation(unicode::decode(c.get("text").str()))) == c.get("expected").str(),
               "phrase punctuation " + label(c.get("text")));
    }
    for (const Json& c : fixture.get("segments").elements()) {
        auto got = punctuated_segments(unicode::decode(c.get("text").str()), static_cast<std::size_t>(c.get("max_chars").integer()));
        const auto& want = c.get("expected").elements();
        bool same = got.size() == want.size();
        for (std::size_t i = 0; same && i < got.size(); ++i) {
            same = unicode::encode(got[i].first) == want[i][0].str() && got[i].second == strings(want[i][1]);
        }
        expect(same, "punctuated segments " + label(c.get("text")) + " max " + std::to_string(c.get("max_chars").integer()));
    }
}

void normalizer_suite(const Json& fixture) {
    for (const Json& c : fixture.get("cases").elements()) {
        std::string got = normalize_spoken_text(c.get("text").str(), c.get("language").str());
        expect(got == c.get("expected").str(), "normalize[" + c.get("language").str() + "] " + label(c.get("text")) + " -> " + got);
    }
}

void unicode_suite(const Json& fixture) {
    for (const Json& c : fixture.get("cases").elements()) {
        unicode::Text text = unicode::decode(c.get("text").str());
        expect(unicode::encode(unicode::lower(text)) == c.get("lower").str(), "lower " + label(c.get("text")));
        expect(unicode::encode(unicode::nfc(text)) == c.get("nfc").str(), "nfc " + label(c.get("text")));
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <events|durations|context|delivery|sentences|segments|normalizer|unicode> <fixtures dir>\n", argv[0]);
        return 2;
    }
    const std::string suite = argv[1], dir = argv[2];
    const std::map<std::string, std::pair<std::string, std::function<void(const Json&)>>> suites = {
        {"events", {"events.json", events_suite}},       {"durations", {"durations.json", durations_suite}},
        {"context", {"context.json", context_suite}},    {"delivery", {"delivery.json", delivery_suite}},
        {"sentences", {"text.json", sentences_suite}},   {"segments", {"text.json", segments_suite}},
        {"normalizer", {"normalizer.json", normalizer_suite}}, {"unicode", {"unicode.json", unicode_suite}}};
    auto it = suites.find(suite);
    if (it == suites.end()) {
        std::fprintf(stderr, "unknown suite %s\n", suite.c_str());
        return 2;
    }
    try {
        it->second.second(Json::parse_file(dir + "/" + it->second.first));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
    std::printf("%s: %d checks, %d failures\n", suite.c_str(), g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
