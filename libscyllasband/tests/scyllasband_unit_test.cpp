// Unit tests against fixtures produced by the Python reference runtime (tests/generate_fixtures.py).
// Usage: scyllasband_unit_test <suite> <fixtures dir>
#include "scyllasband_engine.h"
#include "scyllasband_events.h"
#include "scyllasband_graph.h"
#include "scyllasband_json.h"
#include "scyllasband_song_render.h"
#include "scyllasband_speech_stream.h"
#include "scyllasband_text.h"
#include "scyllasband_unicode.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <algorithm>
#include <map>
#include <memory>
#include <optional>
#include <set>
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
        std::vector<int> got = frames_to_durations(values, strings(c.get("phones")), c.get("scale").num(), c.get("pause_scale").num(1.0));
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
        bool error = false;
        Delivery delivery;
        try {
            delivery = Delivery::parse(spec.is_null() ? nullptr : text.c_str());
        } catch (const std::invalid_argument&) {
            error = true;
        }
        const std::string name = "delivery " + spec.dump();
        expect(error == c.get("error").boolean(), name + " error state");
        if (error || c.get("error").boolean()) continue;
        expect(delivery.to_json().dump() == c.get("resolved").dump(), name + " resolved " + delivery.to_json().dump());
        expect(delivery.spec() == c.get("spec_text").str(), name + " spec " + delivery.spec());
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
// What the fake graphs were called with: the component, and for flow calls the delivery mode, whether the melody was all zero
// and the time.
struct FakeCall {
    std::string name;
    int64_t mode = -1;
    bool melody_zero = false;
    float time = 0.0f;
};

class FakeGraph : public Graph {
public:
    FakeGraph(std::string name, float frames_per_phone, const Json& tokenizer, std::vector<FakeCall>* log = nullptr)
        : name_(std::move(name)), frames_per_phone_(frames_per_phone), log_(log) {
        for (const auto& [symbol, id] : tokenizer.get("text_symbols").items()) text_[id.integer()] = symbol;
        for (const auto& [id, symbol] : tokenizer.get("phoneme_symbols").items()) phonemes_[symbol.str()] = std::stoll(id);
    }

    GraphOutput run(const std::vector<TensorView>& inputs) override {
        const TensorView& first = inputs.front();
        std::size_t count = 1;
        for (int64_t dim : first.shape) count *= static_cast<std::size_t>(dim);
        if (log_) {
            FakeCall call{name_};
            if (name_.rfind("vector_estimator", 0) == 0) {
                call.time = *static_cast<const float*>(inputs[1].data);
                const TensorView& melody = inputs[16];
                std::size_t values = 1;
                for (int64_t dim : melody.shape) values *= static_cast<std::size_t>(dim);
                const float* data = static_cast<const float*>(melody.data);
                call.melody_zero = std::all_of(data, data + values, [](float v) { return v == 0.0f; });
                call.mode = inputs.size() > 17 ? *static_cast<const int64_t*>(inputs[17].data) : -1;
            }
            log_->push_back(call);
        }
        if (name_.rfind("g2p", 0) == 0) return g2p(static_cast<const int64_t*>(first.data), count);
        if (name_ == "duration_predictor") return GraphOutput{first.shape, std::vector<float>(count, frames_per_phone_)};
        if (name_ == "vector_context_encoder") return GraphOutput{{1, 512}, std::vector<float>(512, 0.0f)};
        if (name_ == "pitch_context_encoder") return GraphOutput{{1, 384}, std::vector<float>(384, 0.0f)};
        if (name_.rfind("vector_estimator", 0) == 0 || name_.rfind("pitch_estimator", 0) == 0) {
            const float* x = static_cast<const float*>(first.data);
            GraphOutput out{first.shape, std::vector<float>(count)};
            for (std::size_t i = 0; i < count; ++i) out.values[i] = -x[i];
            return out;
        }
        if (name_.rfind("vocoder_bigvgan", 0) == 0) {   // BigVGAN returns frames * 512 samples
            const int64_t samples = first.shape.back() * 512;
            return GraphOutput{{1, samples}, std::vector<float>(static_cast<std::size_t>(samples), 0.2f)};
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
    std::vector<FakeCall>* log_;
    std::map<int64_t, std::string> text_;
    std::map<std::string, int64_t> phonemes_;
};

class FakeBackend : public Backend {
public:
    FakeBackend(float frames_per_phone, Json tokenizer, std::vector<FakeCall>* log = nullptr)
        : frames_per_phone_(frames_per_phone), tokenizer_(std::move(tokenizer)), log_(log) {}
    const char* name() const override { return "onnx"; }
    std::unique_ptr<Graph> open(const std::string& component, const Json&) override {
        return std::make_unique<FakeGraph>(component, frames_per_phone_, tokenizer_, log_);
    }

private:
    float frames_per_phone_;
    Json tokenizer_;
    std::vector<FakeCall>* log_;
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

bool same_json(const Json& a, const Json& b, double tolerance = 1e-9);

// Records planning (voices, languages, delivery forms, chains, errors) and pause_ms between chains, against the Python runtime.
void records_suite(const Json& fixture) {
    const FakeBundle bundle(fixture.get("bundle"));
    const Json& tokenizer = fixture.get("bundle").get("assets/g2p/tokenizer.json");
    BackendOptions backend;
    backend.bundle_dir = bundle.dir.string();
    Engine engine(backend, std::make_unique<FakeBackend>(1.0f, tokenizer));
    for (const Json& c : fixture.get("cases").elements()) {
        const std::string name = "records '" + c.get("name").str() + "'";
        Json got;
        std::string error;
        try {
            got = plan_records(engine, c.get("records"), c.get("normalize").boolean()).to_json();
        } catch (const std::invalid_argument& failure) {
            error = failure.what();
        }
        if (c.has("error")) {
            expect(error == c.get("error").str(), name + ": error '" + error + "', expected '" + c.get("error").str() + "'");
        } else {
            expect(error.empty() && same_json(got, c.get("plan")), name + ": " + (error.empty() ? got.dump().substr(0, 400) : error));
        }
    }
    bool refused = false;
    try {
        plan_records(engine, Json(Json::Object{}));
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    expect(refused, "records must be an array");
    const Json& pause = fixture.get("pause");
    SynthesisOptions options;
    options.steps = 1;
    options.sampler = "heun";
    options.seed = 0;
    options.pause_ms = pause.get("pause_ms").num();
    std::vector<std::pair<Json, std::vector<float>>> spoken;
    synthesize_plan_stream(engine, plan_records(engine, pause.get("records")), options, [&](const StreamEvent& event) {
        if (event.type == StreamEvent::kAudio) spoken.emplace_back(event.metadata, *event.audio);
        return true;
    });
    const auto& want = pause.get("targets").elements();
    expect(spoken.size() == want.size(), "pause: " + std::to_string(spoken.size()) + " targets, expected " + std::to_string(want.size()));
    for (std::size_t i = 0; i < std::min(spoken.size(), want.size()); ++i) {
        const Json& chunk = spoken[i].first.get("chunk");
        const std::vector<float>& audio = spoken[i].second;
        int64_t zeros = 0;
        while (zeros < static_cast<int64_t>(audio.size()) && audio[static_cast<std::size_t>(zeros)] == 0.0f) ++zeros;
        expect(chunk.get("chunk_id").str() == want[i].get("chunk_id").str() && chunk.get("chain_id").str() == want[i].get("chain_id").str() &&
                   static_cast<int64_t>(audio.size()) == want[i].get("samples").integer() && zeros == want[i].get("leading_zeros").integer() &&
                   spoken[i].first.get("seed").integer() == want[i].get("seed").integer(),
               "pause target " + std::to_string(i) + ": " + chunk.get("chunk_id").str() + " " + chunk.get("chain_id").str() + " " +
                   std::to_string(audio.size()) + " samples, " + std::to_string(zeros) + " leading zeros");
    }
}

// Structural JSON equality with numbers within `tolerance` (the two sides format floats differently).
bool same_json(const Json& a, const Json& b, double tolerance) {
    if (a.is_number() && b.is_number()) return std::fabs(a.num() - b.num()) <= tolerance;
    if (a.type() != b.type()) return false;
    if (a.is_array()) {
        if (a.size() != b.size()) return false;
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (!same_json(a[i], b[i], tolerance)) return false;
        }
        return true;
    }
    if (a.is_object()) {
        if (a.size() != b.size()) return false;
        for (const auto& [key, value] : a.items()) {
            if (!b.has(key) || !same_json(value, b.get(key), tolerance)) return false;
        }
        return true;
    }
    return a.dump() == b.dump();
}

std::unique_ptr<Engine> fake_engine(const FakeBundle& bundle, const Json& files, std::vector<FakeCall>* log = nullptr, float frames_per_phone = 1.0f) {
    BackendOptions backend;
    backend.bundle_dir = bundle.dir.string();
    return std::make_unique<Engine>(backend, std::make_unique<FakeBackend>(frames_per_phone, files.get("assets/g2p/tokenizer.json"), log));
}

// The singing planner and renderer against scyllasband/singing.py on the fake v4 bundle (generate_fixtures.singing_cases).
void singing_suite(const Json& fixture) {
    for (const Json& c : fixture.get("notes").elements()) {
        const std::string name = c.get("name").str();
        bool error = false;
        std::optional<int> midi;
        try {
            midi = note_to_midi(name);
        } catch (const ScoreError&) {
            error = true;
        }
        expect(error == c.get("error").boolean(), "note_to_midi error state '" + name + "'");
        if (!error && !c.get("error").boolean()) {
            const bool rest = c.get("midi").is_null();
            expect(rest ? !midi.has_value() : (midi && *midi == c.get("midi").integer()), "note_to_midi '" + name + "'");
        }
    }
    for (const Json& c : fixture.get("names").elements()) {
        expect(midi_to_note(static_cast<int>(c.get("midi").integer())) == c.get("name").str(), "midi_to_note " + c.get("midi").dump());
    }
    for (const Json& c : fixture.get("parse").elements()) {
        const std::string name = "parse_score " + c.get("name").str();
        bool error = false;
        Json got;
        try {
            got = parse_score(c.get("score")).to_json();
        } catch (const ScoreError&) {
            error = true;
        }
        expect(error == c.get("error").boolean(), name + " error state");
        if (!error && !c.get("error").boolean()) expect(same_json(got, c.get("expected")), name + ": " + got.dump().substr(0, 200));
    }
    for (const Json& c : fixture.get("registers").elements()) {
        const std::optional<double> center = c.get("center").is_null() ? std::nullopt : std::optional<double>(c.get("center").num());
        const int got = register_shift(parse_score(c.get("score")), center);
        expect(got == c.get("shift").integer(), "register_shift " + c.get("score").get("notes").dump() + " center " + c.get("center").dump() +
                                                     " -> " + std::to_string(got));
    }
    const Json& bundles = fixture.get("bundles");
    const FakeBundle v4(bundles.get("v4")), guided(bundles.get("v4_guided")), speech(bundles.get("v4_speech")), v3(bundles.get("v3"));
    for (const Json& c : fixture.get("plans").elements()) {
        auto engine = fake_engine(v4, bundles.get("v4"));
        PlanSettings settings;
        settings.use_register = c.get("register").boolean();
        settings.transpose = static_cast<int>(c.get("transpose").integer());
        settings.pad = c.get("pad").num();
        if (!c.get("max_seconds").is_null()) settings.max_seconds = c.get("max_seconds").num();
        const std::string name = "plan " + c.get("score").get("notes").dump().substr(0, 60) + " register " + c.get("register").dump() + " transpose " +
                                 c.get("transpose").dump() + " pad " + c.get("pad").dump();
        SongPlan plan;
        bool failed = false;
        try {
            EngineSongFrontend frontend(*engine);
            plan = plan_song(parse_score(c.get("score")), frontend, "scylla", "en_us", settings);
        } catch (const std::exception& error) {
            failed = true;
            expect(!c.get("error").is_null(), name + ": " + error.what());
        }
        if (failed) continue;
        expect(c.get("error").is_null(), name + ": expected " + c.get("error").str());
        if (!c.get("error").is_null()) continue;
        expect(plan.register_shift == c.get("register_shift").integer() && plan.mode_id == c.get("mode_id").integer() &&
                   std::fabs(plan.seconds() - c.get("seconds").num()) < 1e-12, name + ": register, mode or length");
        const auto& want = c.get("phrases").elements();
        expect(plan.phrases.size() == want.size(), name + ": " + std::to_string(plan.phrases.size()) + " phrases, expected " + std::to_string(want.size()));
        for (std::size_t i = 0; i < std::min(plan.phrases.size(), want.size()); ++i) {
            const PhrasePlan& got = plan.phrases[i];
            const Json& w = want[i];
            std::vector<float> melody;
            for (const Json& value : w.get("melody").elements()) melody.push_back(static_cast<float>(value.num()));
            const std::vector<int64_t> durations(got.durations.begin(), got.durations.end()), starts(got.sentence.word_starts.begin(), got.sentence.word_starts.end());
            expect(std::fabs(got.phrase.start - w.get("start").num()) < 1e-12 && std::fabs(got.phrase.end - w.get("end").num()) < 1e-12 &&
                       got.phrase.text == w.get("text").str(), name + " phrase " + std::to_string(i) + ": times or text '" + got.phrase.text + "'");
            expect(got.sentence.phones == strings(w.get("phones")) && starts == ints(w.get("word_starts")), name + " phrase " + std::to_string(i) + ": phones");
            expect(durations == ints(w.get("durations")) && got.frames == w.get("frames").integer(), name + " phrase " + std::to_string(i) + ": durations " +
                                                                                                            Json::array_of(got.durations).dump());
            expect(got.melody == melody, name + " phrase " + std::to_string(i) + ": melody");
        }
    }
    for (const Json& c : fixture.get("sings").elements()) {
        const std::string bundle_name = c.get("bundle").str();
        const FakeBundle& bundle = bundle_name == "v4_guided" ? guided : v4;
        std::vector<FakeCall> log;
        auto engine = fake_engine(bundle, bundles.get(bundle_name), &log);
        const Json& options = c.get("options");
        SingRequest sing_options;
        sing_options.voice = "scylla";
        sing_options.language = "en_us";
        sing_options.seed = 5;
        sing_options.steps = static_cast<int>(options.get("steps").integer(0));
        sing_options.sampler = options.get("sampler").str();
        sing_options.decoder = options.get("decoder").str();
        if (options.has("guidance")) sing_options.guidance = options.get("guidance").num();
        sing_options.plan.pad = 0.02;
        sing_options.plan.transpose = static_cast<int>(options.get("transpose").integer(0));
        const std::string name = "sing " + bundle_name + " " + c.get("score").get("style").str() + " " + options.dump();
        Json metadata;
        std::vector<float> audio;
        try {
            audio = sing(*engine, parse_score(c.get("score")), sing_options, &metadata);
        } catch (const std::exception& error) {
            expect(false, name + ": " + error.what());
            continue;
        }
        // Graph calls apart from the G2P (both sides cache phonemizations their own way).
        std::vector<FakeCall> calls;
        for (const FakeCall& call : log) {
            if (call.name.rfind("g2p", 0) != 0) calls.push_back(call);
        }
        std::vector<const Json*> want_calls;
        for (const Json& call : c.get("calls").elements()) {
            if (call.get("name").str().rfind("g2p", 0) != 0) want_calls.push_back(&call);
        }
        bool same_calls = calls.size() == want_calls.size();
        for (std::size_t i = 0; same_calls && i < calls.size(); ++i) {
            const Json& w = *want_calls[i];
            same_calls = calls[i].name == w.get("name").str();
            if (same_calls && w.has("mode")) {
                same_calls = calls[i].mode == w.get("mode").integer() && calls[i].melody_zero == w.get("melody_zero").boolean() &&
                             calls[i].time == static_cast<float>(w.get("time").num());
            }
        }
        expect(same_calls, name + ": graph calls (" + std::to_string(calls.size()) + " against " + std::to_string(want_calls.size()) + ")");
        Json::Array runs;
        for (float value : audio) {
            if (!runs.empty() && static_cast<float>(runs.back()[0].num()) == value) {
                runs.back().elements()[1] = Json(runs.back()[1].integer() + 1);
            } else {
                runs.push_back(Json(Json::Array{Json(static_cast<double>(value)), Json(int64_t{1})}));
            }
        }
        expect(same_json(Json(runs), c.get("audio"), 1e-7), name + ": audio " + Json(runs).dump().substr(0, 160));
        const Json& m = c.get("metadata");
        bool same_meta = metadata.get("decoder").str() == m.get("decoder").str() && metadata.get("steps").integer() == m.get("steps").integer() &&
                         metadata.get("sampler").str() == m.get("sampler").str() && metadata.get("guidance").num() == m.get("guidance").num() &&
                         metadata.get("mode").str() == m.get("mode").str() && metadata.get("mode_id").integer() == m.get("mode_id").integer() &&
                         metadata.get("register_shift").integer() == m.get("register_shift").integer() &&
                         metadata.get("transpose").integer() == m.get("transpose").integer() && metadata.get("seconds").num() == m.get("seconds").num() &&
                         metadata.get("phrases").size() == m.get("phrases").size();
        for (std::size_t i = 0; same_meta && i < m.get("phrases").size(); ++i) {
            const Json& got = metadata.get("phrases")[i];
            const Json& w = m.get("phrases")[i];
            same_meta = got.get("text").str() == w.get("text").str() && got.get("frames").integer() == w.get("frames").integer() &&
                        got.get("bucket").integer() == w.get("bucket").integer() && got.get("seed").integer() == w.get("seed").integer() &&
                        got.get("seconds").num() == w.get("seconds").num() && ints(got.get("durations")) == ints(w.get("durations"));
        }
        expect(same_meta, name + ": metadata " + metadata.dump().substr(0, 300));
    }
    // Speech on a v4 bundle: the v3 loop's targets, with delivery mode 0 on every flow call.
    for (const Json& c : fixture.get("speech").elements()) {
        std::vector<FakeCall> log;
        auto engine = fake_engine(speech, bundles.get("v4_speech"), &log, static_cast<float>(c.get("frames_per_phone").num()));
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
        synthesize_plan_stream(*engine, plan, options, [&](const StreamEvent& event) {
            if (event.type == StreamEvent::kAudio) spoken.push_back(event.metadata);
            return true;
        });
        const auto& want = c.get("targets").elements();
        expect(spoken.size() == want.size(), "v4 speech targets");
        for (std::size_t i = 0; i < std::min(spoken.size(), want.size()); ++i) {
            const Json& got = spoken[i];
            expect(got.get("chunk").get("chunk_id").str() == want[i].get("chunk_id").str() && strings(got.get("phones")) == strings(want[i].get("phones")) &&
                       ints(got.get("durations")) == ints(want[i].get("durations")) && got.get("latent_frames").integer() == want[i].get("latent_frames").integer() &&
                       got.get("prefix_frames").integer() == want[i].get("prefix_frames").integer() &&
                       got.get("delivery_mode").integer(-1) == want[i].get("delivery_mode").integer(),
                   "v4 speech target " + std::to_string(i));
        }
        std::set<int64_t> modes;
        for (const FakeCall& call : log) {
            if (call.name.rfind("vector_estimator", 0) == 0) modes.insert(call.mode);
        }
        expect(std::vector<int64_t>(modes.begin(), modes.end()) == ints(c.get("flow_modes")), "v4 speech feeds delivery mode 0");
    }
    // Singing info, and a spoken-only (v3) bundle that refuses to sing and never feeds a mode.
    {
        auto engine = fake_engine(v4, bundles.get("v4"));
        const Json info = singing_info(*engine);
        expect(info.get("supported").boolean() && strings(info.get("styles")) == std::vector<std::string>{"pop", "sung", "ballad"} &&
                   strings(info.get("decoders")) == std::vector<std::string>{"bigvgan", "vocos"} && info.get("default_decoder").str() == "bigvgan",
               "singing info " + info.dump());
        std::vector<FakeCall> log;
        auto spoken = fake_engine(v3, bundles.get("v3"), &log);
        expect(!singing_info(*spoken).get("supported").boolean() && strings(singing_info(*spoken).get("decoders")) == std::vector<std::string>{"vocos"},
               "v3 singing info");
        bool refused = false;
        try {
            SingRequest options;
            options.voice = "scylla";
            sing(*spoken, parse_score(fixture.get("sings")[0].get("score")), options);
        } catch (const std::invalid_argument& error) {
            refused = std::string(error.what()).find("v4") != std::string::npos;
        }
        expect(refused, "a v3 bundle refuses to sing");
        SynthesisOptions options;
        options.text = "ab ab.";
        options.voice = "scylla";
        options.steps = 1;
        options.seed = 0;
        synthesize_stream(*spoken, options, [](const StreamEvent&) { return true; });
        expect(std::all_of(log.begin(), log.end(), [](const FakeCall& call) { return call.name.rfind("vector_estimator", 0) != 0 || call.mode == -1; }),
               "v3 speech feeds no delivery mode");
        bool unknown = false;
        try {
            SingRequest options;
            options.voice = "scylla";
            options.decoder = "hifigan";
            options.plan.pad = 0.02;
            sing(*engine, parse_score(fixture.get("sings")[0].get("score")), options);
        } catch (const std::invalid_argument& error) {
            unknown = std::string(error.what()).find("no 'hifigan' decoder") != std::string::npos;
        }
        expect(unknown, "an unknown decoder is refused");
    }
    // A bundle whose BigVGAN stops below the flow's largest bucket, as the Apple bundles carry it (BigVGAN to 512 frames, the flow
    // to 1024): longer decodes are refused (never written past the bucket) and plans keep within it whichever decoder sings.
    {
        Json files = bundles.get("v4");
        Json manifest = files.get("manifest.json"), controls = manifest.get("controls"), targets = controls.get("target_buckets");
        Json buckets = targets.get("buckets");
        for (Json& bucket : buckets.elements()) {
            if (bucket.get("latent_frames").integer() <= 8) continue;
            Json kept{Json::Object{}};
            for (const auto& [key, value] : bucket.items()) {
                if (key != "vocoder_bigvgan") kept.set(key, value);
            }
            bucket = kept;
        }
        targets.set("buckets", buckets);
        controls.set("target_buckets", targets);
        manifest.set("controls", controls);
        files.set("manifest.json", manifest);
        const FakeBundle small(files);
        auto engine = fake_engine(small, files);
        EngineSongFrontend frontend(*engine);
        expect(engine->decoder_max_frames("bigvgan") == 8 && engine->decoder_max_frames("vocos") == 16 && frontend.max_phrase_frames() == 8,
               "decoder limits: bigvgan " + std::to_string(engine->decoder_max_frames("bigvgan")) + ", phrase " + std::to_string(frontend.max_phrase_frames()));
        const std::vector<float> latents(24 * 12, 0.5f);
        DecodeRequest decode;
        decode.voice = "scylla";
        decode.language = "en_us";
        decode.decoder = "bigvgan";
        bool overlong = false;
        try {
            engine->decode(latents, 12, decode);
        } catch (const OverlongError&) {
            overlong = true;
        }
        expect(overlong, "a decode longer than the decoder's largest bucket is refused");
        decode.decoder = "vocos";
        expect(engine->decode(latents, 12, decode).size() == static_cast<std::size_t>(12 * 512), "vocos decodes past BigVGAN's largest bucket");
        const Score score = parse_score(Json::parse(R"({"text": "bob wow", "bpm": 1200, "style": "pop",
            "notes": [["bob", "C4", 1], ["-", "rest", 6], ["wow", "D4", 1]]})"));
        std::vector<Json> phrases;
        for (const char* decoder : {"bigvgan", "vocos"}) {
            SingRequest request;
            request.voice = "scylla";
            request.decoder = decoder;
            request.seed = 3;
            request.plan.pad = 0.02;
            Json metadata;
            sing(*engine, score, request, &metadata);
            phrases.push_back(metadata.get("phrases"));
            for (const Json& phrase : metadata.get("phrases").elements()) {
                expect(phrase.get("frames").integer() <= 8, std::string(decoder) + " phrase within the phrase limit: " + phrase.get("frames").dump());
            }
        }
        std::vector<std::vector<int64_t>> durations[2];
        for (int k = 0; k < 2; ++k) {
            for (const Json& phrase : phrases[k].elements()) durations[k].push_back(ints(phrase.get("durations")));
        }
        expect(!durations[0].empty() && durations[0] == durations[1], "the plan does not depend on the decoder");
    }
    // The planner through a fake frontend: the phrase limit comes from the frontend, styles from its controls, and every
    // request check happens in resolve_sing_request.
    {
        class LetterFrontend : public SongFrontend {
        public:
            Phonemized phonemize(const std::string& text, const std::string&) override {
                Phonemized out;
                out.phones = {kSilencePhone};
                std::size_t word = 0;
                std::string current;
                auto flush = [&]() {
                    if (current.empty()) return;
                    if (word++ > 0) out.word_starts.push_back(static_cast<int>(out.phones.size()));
                    for (char letter : current) out.phones.push_back(std::string(1, letter));
                    out.phones.push_back(kSilencePhone);
                    current.clear();
                };
                for (char c : text) {
                    if (c == ' ') flush();
                    else current += c;
                }
                flush();
                return out;
            }
            std::string resolve_language(const std::string& voice, const std::string& language) const override {
                if (voice != "v") throw std::invalid_argument("Unknown voice '" + voice + "'");
                return language.empty() ? "en_us" : language;
            }
            const SingingControls& singing() const override { return controls; }
            int sample_rate() const override { return 24000; }
            int latent_hop() const override { return 512; }
            int64_t max_phrase_frames() const override { return limit; }
            bool has_decoder(const std::string& decoder) const override { return decoder == "vocos" || decoder == "bigvgan"; }
            std::vector<std::string> available_decoders() const override { return {"bigvgan", "vocos"}; }
            SingingControls controls;
            int64_t limit = 1024;
        };
        LetterFrontend frontend;
        frontend.controls.mode_ids = {{"spoken", 0}, {"sung", 1}, {"ballad", 3}};
        frontend.controls.modes = {"sung", "ballad"};
        frontend.controls.voice_center = {{"v", 60.0}};
        // Two 0.6 s runs a 0.3 s rest apart: one phrase with a roomy limit, two when the limit holds only one run.
        const Score song = parse_score(Json::parse(R"({"text": "bob wow bob wow", "bpm": 100, "style": "pop", "notes": [
            ["bob", "C4", 0.5], ["wow", "D4", 0.5], ["-", "rest", 0.5], ["bob", "E4", 0.5], ["wow", "G4", 0.5]]})"));
        PlanSettings settings;
        settings.max_seconds = 2.0;
        const SongPlan roomy = plan_song(song, frontend, "v", "en_us", settings);
        frontend.limit = 50;   // (50 - 2) * 512 / 24000 - 0.24 = 0.78 s
        const SongPlan tight = plan_song(song, frontend, "v", "en_us", settings);
        expect(roomy.phrases.size() == 1 && tight.phrases.size() == 2, "the frontend's phrase limit packs phrases: " +
               std::to_string(roomy.phrases.size()) + " / " + std::to_string(tight.phrases.size()));
        frontend.limit = 20;
        bool overlong = false;
        try {
            plan_song(song, frontend, "v", "en_us", settings);
        } catch (const OverlongError&) {
            overlong = true;
        }
        expect(overlong, "a phrase longer than the frontend's limit is refused");
        frontend.limit = 1024;
        auto refused = [&](const SingRequest& request, const Score& score, const std::string& needle) {
            try {
                resolve_sing_request(frontend, score, request);
            } catch (const std::invalid_argument& error) {
                return std::string(error.what()).find(needle) != std::string::npos;
            }
            return false;
        };
        SingRequest request;
        request.voice = "  v ";
        const ResolvedSing resolved = resolve_sing_request(frontend, song, request);
        expect(resolved.voice == "v" && resolved.language == "en_us" && resolved.settings.decoder == "vocos" && resolved.settings.steps == 8 &&
                   resolved.settings.sampler == "heun",
               "resolved request: " + resolved.voice + " " + resolved.settings.decoder);
        SingRequest blank = request;
        blank.voice = " ";
        expect(refused(blank, song, "A voice is required"), "a blank voice is refused");
        SingRequest unknown_voice = request;
        unknown_voice.voice = "nobody";
        expect(refused(unknown_voice, song, "Unknown voice"), "an unknown voice is refused");
        SingRequest bad_decoder = request;
        bad_decoder.decoder = "hifigan";
        expect(refused(bad_decoder, song, "no 'hifigan' decoder"), "an unknown decoder is refused");
        SingRequest bad_sampler = request;
        bad_sampler.sampler = "dpm";
        expect(refused(bad_sampler, song, "Unsupported sampler"), "an unknown sampler is refused");
        frontend.controls.modes = {"sung"};
        const Score ballad = parse_score(Json::parse(R"({"bpm": 80, "style": "ballad", "notes": [["wow", "C4", 2]]})"));
        expect(refused(request, ballad, "not a singing style"), "a style the bundle does not sing is refused");
        frontend.controls.modes = {"sung", "ballad"};
        SingingControls spoken_only;
        LetterFrontend speech = frontend;
        speech.controls = spoken_only;
        bool needs_v4 = false;
        try {
            resolve_sing_request(speech, song, request);
        } catch (const std::invalid_argument& error) {
            needs_v4 = std::string(error.what()).find("v4") != std::string::npos;
        }
        expect(needs_v4, "a bundle without delivery modes refuses to sing");
    }
    // Key fit: no graphs, only the controls and the score.
    {
        SingingControls controls;
        controls.mode_ids = {{"spoken", 0}, {"sung", 1}};
        controls.voice_center = {{"v", 60.0}};
        auto score_of = [](const std::vector<std::string>& notes) {
            Json rows{Json::Array{}};
            for (const std::string& note : notes) {
                Json row{Json::Array{}};
                row.push(Json("la"));
                row.push(Json(note));
                row.push(Json(1));
                rows.push(row);
            }
            Json data{Json::Object{}};
            data.set("bpm", Json(120));
            data.set("notes", rows);
            return parse_score(data);
        };
        const SongFit fits = song_fit(controls, score_of({"C5", "C5"}), "v", 0, true);   // mean 72 -> register -12 -> 60
        expect(fits.register_shift == -12 && fits.outside_semitones == 0.0 && fits.suggested_transpose == 0 && fits.band_low == 57.0 &&
                   fits.band_high == 65.0 && fits.low_midi == 60 && fits.high_midi == 60,
               "a song in the voice's range: " + fits.to_json().dump());
        const SongFit high = song_fit(controls, score_of({"F#4", "A4"}), "v", 0, true);   // mean 67.5: -12 -> 55.5, 1.5 below the band
        expect(high.register_shift == -12 && std::fabs(high.outside_semitones - 1.5) < 1e-9 && high.suggested_transpose == 2,
               "a song between octaves: " + high.to_json().dump());
        const SongFit fixed = song_fit(controls, score_of({"C5", "C5"}), "v", 0, false);   // no register: 7 above, -6 comes nearest
        expect(fixed.register_shift == 0 && std::fabs(fixed.outside_semitones - 7.0) < 1e-9 && fixed.suggested_transpose == -6,
               "a song held in its octave: " + fixed.to_json().dump());
        const SongFit unknown = song_fit(controls, score_of({"C5"}), "nobody", 2, true);
        expect(!unknown.voice_center && unknown.register_shift == 0 && unknown.suggested_transpose == 2 && unknown.mean_midi == 74.0,
               "no centre for an unknown voice: " + unknown.to_json().dump());
    }
    // Windowed decoding: a BigVGAN stand-in whose every sample is its frame's first latent channel (0 where masked), so windows
    // stitched from the frames each keeps must equal one whole decode exactly, and run through the window's bucket only.
    {
        struct FrameGraph : Graph {
            FrameGraph(std::string name, std::vector<std::string>* calls) : name(std::move(name)), calls(calls) {}
            GraphOutput run(const std::vector<TensorView>& inputs) override {
                calls->push_back(name);
                const int64_t frames = inputs[0].shape.back();
                const float* latents = static_cast<const float*>(inputs[0].data);   // [1, latent_dim, frames]: channel 0 first
                const uint8_t* mask = static_cast<const uint8_t*>(inputs[1].data);
                GraphOutput out{{1, frames * 512}, std::vector<float>(static_cast<std::size_t>(frames * 512), 0.0f)};
                for (int64_t t = 0; t < frames; ++t) std::fill_n(out.values.begin() + t * 512, 512, mask[t] ? latents[t] : 0.0f);
                return out;
            }
            std::string name;
            std::vector<std::string>* calls;
        };
        struct WindowBackend : FakeBackend {
            WindowBackend(Json tokenizer, std::vector<std::string>* calls) : FakeBackend(1.0f, std::move(tokenizer)), calls(calls) {}
            std::unique_ptr<Graph> open(const std::string& component, const Json& spec) override {
                if (component.rfind("vocoder_bigvgan", 0) == 0) return std::make_unique<FrameGraph>(component, calls);
                return FakeBackend::open(component, spec);
            }
            std::vector<std::string>* calls;
        };
        auto with_window = [&](const char* window) {   // the v4 fake bundle's files, BigVGAN given `window` (JSON) when set
            Json files = bundles.get("v4");
            Json manifest = files.get("manifest.json"), controls = manifest.get("controls"), decoders = controls.get("decoders");
            Json bigvgan = decoders.get("bigvgan");
            if (window) bigvgan.set("window", Json::parse(window));
            decoders.set("bigvgan", bigvgan);
            controls.set("decoders", decoders);
            manifest.set("controls", controls);
            files.set("manifest.json", manifest);
            return files;
        };
        const Json plain_files = with_window(nullptr), windowed_files = with_window(R"({"frames": 8, "context": 2})");
        const FakeBundle plain_bundle(plain_files), windowed_bundle(windowed_files);
        std::vector<std::string> plain_calls, windowed_calls;
        auto open_engine = [](const FakeBundle& bundle, const Json& files, std::vector<std::string>* calls) {
            BackendOptions options;
            options.bundle_dir = bundle.dir.string();
            return std::make_unique<Engine>(options, std::make_unique<WindowBackend>(files.get("assets/g2p/tokenizer.json"), calls));
        };
        auto plain = open_engine(plain_bundle, plain_files, &plain_calls);
        auto windowed = open_engine(windowed_bundle, windowed_files, &windowed_calls);
        expect(windowed->decoders().at("bigvgan").window_frames == 8 && windowed->decoders().at("bigvgan").window_context == 2 &&
                   plain->decoders().at("bigvgan").window_frames == 0,
               "the manifest's decoder window");
        auto latents_of = [](int64_t frames, float base) {   // channel 0 counts frames up from `base`; the rest are 0.5
            std::vector<float> latents(static_cast<std::size_t>(24 * frames), 0.5f);
            for (int64_t t = 0; t < frames; ++t) latents[static_cast<std::size_t>(t)] = base + static_cast<float>(t) / 100.0f;
            return latents;
        };
        DecodeRequest decode;
        decode.voice = "scylla";
        decode.language = "en_us";
        decode.decoder = "bigvgan";
        // 13 frames in windows of 8 with 2 frames of context: keeps [0, 6), [6, 10), [10, 13).
        const std::vector<float> long_latents = latents_of(13, 0.1f);
        const std::vector<float> whole = plain->decode(long_latents, 13, decode), stitched = windowed->decode(long_latents, 13, decode);
        expect(stitched == whole && whole.size() == 13 * 512, "windows stitch to the whole decode");
        expect(windowed_calls == std::vector<std::string>(3, "vocoder_bigvgan_8") && plain_calls == std::vector<std::string>{"vocoder_bigvgan"},
               "windows run through the window's bucket only: " + std::to_string(windowed_calls.size()) + " calls");
        // With left context (speech chunks): only the target's samples come back, still the same.
        const std::vector<float> left = latents_of(5, 0.7f);
        decode.left = &left;
        decode.left_frames = 5;
        expect(windowed->decode(long_latents, 13, decode) == plain->decode(long_latents, 13, decode), "windows with left context");
        decode.left = nullptr;
        decode.left_frames = 0;
        // A decode shorter than the window is one call in the window's bucket.
        windowed_calls.clear();
        const std::vector<float> short_latents = latents_of(5, 0.3f);
        expect(windowed->decode(short_latents, 5, decode) == plain->decode(short_latents, 5, decode) &&
                   windowed_calls == std::vector<std::string>{"vocoder_bigvgan_8"},
               "a short decode in the window's bucket");
        // A bundle carrying BigVGAN only at the window's bucket: windowed, it decodes any length, so its limit is the flow's.
        {
            Json files = with_window(R"({"frames": 8, "context": 2})");
            Json manifest = files.get("manifest.json"), controls = manifest.get("controls"), targets = controls.get("target_buckets");
            Json buckets = targets.get("buckets");
            for (Json& bucket : buckets.elements()) {
                if (bucket.get("latent_frames").integer() <= 8) continue;
                Json kept{Json::Object{}};
                for (const auto& [key, value] : bucket.items()) {
                    if (key != "vocoder_bigvgan") kept.set(key, value);
                }
                bucket = kept;
            }
            targets.set("buckets", buckets);
            controls.set("target_buckets", targets);
            manifest.set("controls", controls);
            files.set("manifest.json", manifest);
            const FakeBundle bundle(files);
            std::vector<std::string> calls;
            auto engine = open_engine(bundle, files, &calls);
            EngineSongFrontend frontend(*engine);
            expect(engine->decoder_max_frames("bigvgan") == 16 && frontend.max_phrase_frames() == 16 &&
                       engine->decode(long_latents, 13, decode) == whole,
                   "a windowed decoder with one bucket decodes past it: limit " + std::to_string(engine->decoder_max_frames("bigvgan")));
        }
        // A context that leaves nothing to keep is refused, and Vocos (frames * 512 - 256 samples) cannot window.
        for (const char* bad : {R"({"frames": 8, "context": 4})"}) {
            const Json files = with_window(bad);
            const FakeBundle bundle(files);
            bool refused = false;
            try {
                std::vector<std::string> calls;
                open_engine(bundle, files, &calls);
            } catch (const std::invalid_argument&) {
                refused = true;
            }
            expect(refused, std::string("a window refused: ") + bad);
        }
#ifndef _WIN32
        setenv("SCYLLASBAND_DECODER_WINDOW", "bigvgan=8:3", 1);
        std::vector<std::string> env_calls;
        auto from_env = open_engine(plain_bundle, plain_files, &env_calls);
        setenv("SCYLLASBAND_DECODER_WINDOW", "vocos=8", 1);
        bool vocos_refused = false;
        try {
            open_engine(plain_bundle, plain_files, &env_calls);
        } catch (const std::invalid_argument&) {
            vocos_refused = true;
        }
        unsetenv("SCYLLASBAND_DECODER_WINDOW");
        expect(from_env->decoders().at("bigvgan").window_frames == 8 && from_env->decoders().at("bigvgan").window_context == 3 &&
                   from_env->decode(long_latents, 13, decode) == whole && vocos_refused,
               "SCYLLASBAND_DECODER_WINDOW sets a window; Vocos cannot window");
#endif
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <events|durations|context|delivery|sentences|segments|normalizer|unicode|targets|singing> <fixtures dir>\n", argv[0]);
        return 2;
    }
    const std::string suite = argv[1], dir = argv[2];
    const std::map<std::string, std::pair<std::string, std::function<void(const Json&)>>> suites = {
        {"events", {"events.json", events_suite}},       {"durations", {"durations.json", durations_suite}},
        {"context", {"context.json", context_suite}},    {"delivery", {"delivery.json", delivery_suite}},
        {"sentences", {"text.json", sentences_suite}},   {"segments", {"text.json", segments_suite}},
        {"normalizer", {"normalizer.json", normalizer_suite}}, {"unicode", {"unicode.json", unicode_suite}},
        {"targets", {"targets.json", targets_suite}}, {"singing", {"singing.json", singing_suite}},
        {"records", {"records.json", records_suite}}};
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
