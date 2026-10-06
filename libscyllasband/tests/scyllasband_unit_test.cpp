// Unit tests against fixtures produced by the Python reference runtime (tests/generate_fixtures.py).
// Usage: scyllasband_unit_test <suite> <fixtures dir>
#include "scyllasband_engine.h"
#include "scyllasband_events.h"
#include "scyllasband_graph.h"
#include "scyllasband_json.h"
#include "scyllasband_text.h"
#include "scyllasband_unicode.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
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

// Graph stand-ins for the fake bundle of tests/test_runtime.py (_FakeGraphs): G2P characters become phones of the same
// letter, every span position lasts `frames_per_phone`, the flow velocity is minus its input and the vocoder returns
// frames * 512 - 256 samples of 0.1.
class FakeGraph : public Graph {
public:
    FakeGraph(std::string name, float frames_per_phone, const Json& tokenizer) : name_(std::move(name)), frames_per_phone_(frames_per_phone) {
        for (const auto& [symbol, id] : tokenizer.get("text_symbols").items()) text_[id.integer()] = symbol;
        for (const auto& [id, symbol] : tokenizer.get("phoneme_symbols").items()) phonemes_[symbol.str()] = std::stoll(id);
    }

    GraphOutput run(const std::vector<TensorView>& inputs) override {
        const TensorView& first = inputs.front();
        std::size_t count = 1;
        for (int64_t dim : first.shape) count *= static_cast<std::size_t>(dim);
        if (name_.rfind("g2p", 0) == 0) return g2p(static_cast<const int64_t*>(first.data), count);
        if (name_ == "duration_predictor") return GraphOutput{first.shape, std::vector<float>(count, frames_per_phone_)};
        if (name_ == "vector_context_encoder") return GraphOutput{{1, 512}, std::vector<float>(512, 0.0f)};
        if (name_.rfind("vector_estimator", 0) == 0) {
            const float* x = static_cast<const float*>(first.data);
            GraphOutput out{first.shape, std::vector<float>(count)};
            for (std::size_t i = 0; i < count; ++i) out.values[i] = -x[i];
            return out;
        }
        const int64_t samples = first.shape.back() * 512 - 256;
        return GraphOutput{{1, samples}, std::vector<float>(static_cast<std::size_t>(samples), 0.1f)};
    }

private:
    GraphOutput g2p(const int64_t* ids, std::size_t count) const {
        std::vector<int64_t> frames;
        for (std::size_t i = 0; i < count; ++i) {
            auto it = text_.find(ids[i]);
            const std::string symbol = it == text_.end() ? "<pad>" : it->second;
            if (symbol == "<pad>" || symbol == "<en_us>") continue;
            if (symbol == "<end>") {
                frames.push_back(phonemes_.at("<end>"));
                break;
            }
            auto phone = phonemes_.find(symbol == "e" ? "o" : symbol);
            frames.push_back(phone == phonemes_.end() ? phonemes_.at("a") : phone->second);
            frames.push_back(0);  // CTC blank between characters so repeats survive
        }
        const std::size_t symbols = phonemes_.size();
        GraphOutput out{{1, static_cast<int64_t>(frames.size()), static_cast<int64_t>(symbols)}, std::vector<float>(frames.size() * symbols, -10.0f)};
        for (std::size_t i = 0; i < frames.size(); ++i) out.values[i * symbols + static_cast<std::size_t>(frames[i])] = 10.0f;
        return out;
    }

    std::string name_;
    float frames_per_phone_;
    std::map<int64_t, std::string> text_;
    std::map<std::string, int64_t> phonemes_;
};

class FakeBackend : public Backend {
public:
    FakeBackend(float frames_per_phone, Json tokenizer) : frames_per_phone_(frames_per_phone), tokenizer_(std::move(tokenizer)) {}
    const char* name() const override { return "onnx"; }
    std::unique_ptr<Graph> open(const std::string& component, const Json&) override {
        return std::make_unique<FakeGraph>(component, frames_per_phone_, tokenizer_);
    }

private:
    float frames_per_phone_;
    Json tokenizer_;
};

// The fake bundle's files, written to a temporary directory for the engine to load.
struct FakeBundle {
    explicit FakeBundle(const Json& files) {
        dir = std::filesystem::temp_directory_path() /
              ("scyllasband_unit_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        for (const auto& [relative, content] : files.items()) {
            const std::filesystem::path path = dir / relative;
            std::filesystem::create_directories(path.parent_path());
            std::ofstream out(path, std::ios::binary);
            if (!content.is_null()) out << content.dump();
        }
    }
    ~FakeBundle() {
        std::error_code ignored;
        std::filesystem::remove_all(dir, ignored);
    }
    std::filesystem::path dir;
};

TargetCut cut_kind(const std::string& kind) {
    static const std::map<std::string, TargetCut> kinds = {{"end", TargetCut::kEnd}, {"paragraph", TargetCut::kParagraph},
                                                           {"sentence", TargetCut::kSentence}, {"clause", TargetCut::kClause},
                                                           {"word", TargetCut::kWord}};
    return kinds.at(kind);
}

// The streaming loop's choice of target ends, and the targets it speaks with fake graphs, against the Python loop.
void targets_suite(const Json& fixture) {
    for (const Json& c : fixture.get("choose").elements()) {
        std::vector<TargetCandidate> candidates;
        for (const Json& item : c.get("candidates").elements()) {
            candidates.push_back(TargetCandidate{static_cast<std::size_t>(item[0].integer()), cut_kind(item[1].str()), item[2].integer()});
        }
        const std::size_t got = choose_target(candidates, c.get("first").boolean(), c.get("shortest").integer(), c.get("longest").integer());
        expect(got == static_cast<std::size_t>(c.get("expected").integer()), "choose_target " + label(c.get("candidates")) + " -> " + std::to_string(got));
    }
    const Json& streams = fixture.get("streams");
    const FakeBundle bundle(streams.get("bundle"));
    const Json& tokenizer = streams.get("bundle").get("assets/g2p/tokenizer.json");
    for (const Json& c : streams.get("cases").elements()) {
        const std::string name = "stream '" + c.get("name").str() + "'";
        BackendOptions backend;
        backend.bundle_dir = bundle.dir.string();
        Engine engine(backend, std::make_unique<FakeBackend>(static_cast<float>(c.get("frames_per_phone").num()), tokenizer));
        Plan plan;
        for (const Json& item : c.get("chunks").elements()) {
            PlanChunk chunk;
            chunk.chunk_id = item.get("chunk_id").str();
            chunk.index = static_cast<int>(item.get("index").integer());
            chunk.record_index = static_cast<int>(item.get("record_index").integer());
            chunk.chain_id = item.get("chain_id").str();
            chunk.voice = item.get("voice").str();
            chunk.language = item.get("language").str();
            chunk.delivery = Delivery::parse(item.get("delivery").str().c_str());
            chunk.text = item.get("text").str();
            chunk.source_text = item.get("source_text").str();
            chunk.paragraph_end = item.get("paragraph_end").boolean();
            plan.chunks.push_back(std::move(chunk));
        }
        SynthesisOptions options;
        options.steps = 1;
        options.sampler = "heun";
        options.seed = 0;
        std::vector<Json> spoken;
        try {
            synthesize_plan_stream(engine, plan, options, [&](const StreamEvent& event) {
                if (event.type == StreamEvent::kAudio) spoken.push_back(event.metadata);
                return true;
            });
        } catch (const std::exception& error) {
            expect(false, name + ": " + error.what());
            continue;
        }
        const auto& want = c.get("targets").elements();
        expect(spoken.size() == want.size(), name + ": " + std::to_string(spoken.size()) + " targets, expected " + std::to_string(want.size()));
        for (std::size_t i = 0; i < std::min(spoken.size(), want.size()); ++i) {
            const Json& got = spoken[i];
            const Json& chunk = got.get("chunk");
            const bool same = chunk.get("chunk_id").str() == want[i].get("chunk_id").str() && chunk.get("text").str() == want[i].get("text").str() &&
                              chunk.get("index").integer() == want[i].get("index").integer() &&
                              strings(got.get("phones")) == strings(want[i].get("phones")) &&
                              ints(got.get("durations")) == ints(want[i].get("durations")) &&
                              got.get("latent_frames").integer() == want[i].get("latent_frames").integer() &&
                              got.get("context_before").integer() == want[i].get("context_before").integer() &&
                              got.get("context_after").integer() == want[i].get("context_after").integer() &&
                              got.get("prefix_frames").integer() == want[i].get("prefix_frames").integer() &&
                              got.get("seed").integer(-1) == want[i].get("seed").integer(-1);
            expect(same, name + " target " + std::to_string(i) + ": " + chunk.get("chunk_id").str() + " '" + chunk.get("text").str() + "', expected " +
                             want[i].get("chunk_id").str() + " '" + want[i].get("text").str() + "'");
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <events|durations|context|delivery|sentences|segments|normalizer|unicode|targets> <fixtures dir>\n", argv[0]);
        return 2;
    }
    const std::string suite = argv[1], dir = argv[2];
    const std::map<std::string, std::pair<std::string, std::function<void(const Json&)>>> suites = {
        {"events", {"events.json", events_suite}},       {"durations", {"durations.json", durations_suite}},
        {"context", {"context.json", context_suite}},    {"delivery", {"delivery.json", delivery_suite}},
        {"sentences", {"text.json", sentences_suite}},   {"segments", {"text.json", segments_suite}},
        {"normalizer", {"normalizer.json", normalizer_suite}}, {"unicode", {"unicode.json", unicode_suite}},
        {"targets", {"targets.json", targets_suite}}};
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
