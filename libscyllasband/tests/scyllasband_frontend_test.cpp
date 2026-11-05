// Tests of the frontend library alone (no graph runtime): run as `scyllasband_frontend_test <suite> <fixtures dir>`.
// Configure with -DSCYLLASBAND_SANITIZE=ON to run them under AddressSanitizer and UndefinedBehaviorSanitizer.
#include "scyllasband_noise.h"
#include "scyllasband_score.h"
#include "scyllasband_song_preview.h"
#include "scyllasband_song_timeline.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL %s\n", what.c_str());
    }
}

// The generator's draw order is part of reproducibility (a seed gives the same take on every platform and release), so
// these values are frozen. Platform math libraries may differ in the last bit of log/cos/sin, hence the tolerance.
void noise_suite(const std::string&) {
    struct Case {
        uint64_t seed;
        int64_t count;
        float temperature;
        std::vector<float> expected;
    };
    const Case cases[] = {
        {7ULL, 6, 1.0f, {-0.151572749f, 0.829897106f, 0.587099552f, -0.0700574666f, 0.0944718644f, -0.0971522331f}},
        {0ULL, 5, 1.0f, {-0.0141067971f, -1.00858653f, -1.84589505f, 1.06692827f, 0.788179159f}},
        {31415ULL, 4, 0.5f, {0.883161068f, 0.470940083f, -0.904917359f, 0.130011484f}},
        {18446744073709551615ULL, 3, 1.0f, {0.117751814f, -1.0705862f, -0.0172506422f}},
    };
    for (const Case& c : cases) {
        std::vector<float> got(static_cast<std::size_t>(c.count), 99.0f);
        scyllasband::gaussian_noise(c.seed, c.count, c.temperature, got.data());
        for (std::size_t i = 0; i < got.size(); ++i) {
            expect(std::fabs(got[i] - c.expected[i]) <= 1e-6f * std::max(1.0f, std::fabs(c.expected[i])),
                   "noise seed " + std::to_string(c.seed) + " [" + std::to_string(i) + "] = " + std::to_string(got[i]) + ", expected " +
                       std::to_string(c.expected[i]));
        }
    }
    // A longer draw starts with the shorter one: callers that split one draw into rows rely on it.
    std::vector<float> few(4), many(10);
    scyllasband::gaussian_noise(7, 4, 1.0f, few.data());
    scyllasband::gaussian_noise(7, 10, 1.0f, many.data());
    expect(std::memcmp(few.data(), many.data(), sizeof(float) * 4) == 0, "noise prefix stability");
}


// singing.py sing(): every phrase added into one buffer at its offset (its head cut when the offset is negative), then the
// song truncated to its length and clipped. The timeline must release exactly this, one run per phrase.
std::vector<float> reference_song(int64_t length, const std::vector<int64_t>& starts, const std::vector<std::vector<float>>& phrases) {
    std::vector<float> song(static_cast<std::size_t>(length) + 2400, 0.0f);
    for (std::size_t i = 0; i < phrases.size(); ++i) {
        std::vector<float> audio = phrases[i];
        int64_t offset = starts[i];
        if (offset < 0) {
            audio.erase(audio.begin(), audio.begin() + static_cast<std::ptrdiff_t>(std::min<int64_t>(-offset, static_cast<int64_t>(audio.size()))));
            offset = 0;
        }
        const std::size_t count = static_cast<std::size_t>(offset) < song.size() ? std::min(audio.size(), song.size() - static_cast<std::size_t>(offset)) : 0;
        for (std::size_t k = 0; k < count; ++k) song[static_cast<std::size_t>(offset) + k] += audio[k];
    }
    song.resize(static_cast<std::size_t>(length));
    for (float& sample : song) sample = std::clamp(sample, -1.0f, 1.0f);
    return song;
}

void timeline_suite(const std::string&) {
    uint64_t state = 12345;
    auto next = [&state]() {   // a small deterministic generator for the cases
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<uint32_t>(state >> 33);
    };
    for (int trial = 0; trial < 400; ++trial) {
        const int count = 1 + static_cast<int>(next() % 6);
        std::vector<int64_t> starts;
        std::vector<std::vector<float>> phrases;
        int64_t cursor = trial % 3 == 0 ? -static_cast<int64_t>(next() % 300) : static_cast<int64_t>(next() % 200);  // some lead-ins start early
        std::size_t longest = 0;
        for (int i = 0; i < count; ++i) {
            starts.push_back(cursor);
            const std::size_t size = 50 + next() % 900;
            longest = std::max(longest, size);
            std::vector<float> audio(size);
            for (float& v : audio) v = (static_cast<float>(next() % 2001) / 1000.0f - 1.0f) * 0.8f;
            phrases.push_back(std::move(audio));
            // The next phrase may overlap this one (no rest between them), even reach back over the one before, or follow a rest.
            cursor += static_cast<int64_t>(next() % 2 ? size / 2 + next() % size : size + next() % 400);
        }
        const int64_t length = std::max<int64_t>(1, starts.back() + static_cast<int64_t>(phrases.back().size()) - static_cast<int64_t>(next() % 120));
        const std::vector<float> want = reference_song(length, starts, phrases);
        scyllasband::SongTimeline timeline(length, starts);
        std::vector<float> got;
        bool contiguous = true, bounded = true;
        for (const std::vector<float>& audio : phrases) {
            const int64_t before = timeline.released();
            const std::vector<float> run = timeline.add(audio);
            contiguous = contiguous && before + static_cast<int64_t>(run.size()) == timeline.released() && before == static_cast<int64_t>(got.size());
            bounded = bounded && timeline.pending() <= 2 * longest;
            got.insert(got.end(), run.begin(), run.end());
        }
        expect(contiguous, "timeline runs are contiguous (trial " + std::to_string(trial) + ")");
        expect(bounded, "timeline holds back at most about a phrase (trial " + std::to_string(trial) + ")");
        expect(got.size() == want.size() && std::memcmp(got.data(), want.data(), sizeof(float) * want.size()) == 0,
               "timeline equals sing()'s mix (trial " + std::to_string(trial) + ", " + std::to_string(got.size()) + " of " +
                   std::to_string(want.size()) + " samples)");
    }
    bool refused = false;
    try {
        scyllasband::SongTimeline backwards(100, {50, 10});
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    expect(refused, "timeline refuses decreasing starts");
}

scyllasband::Score parse(const std::string& text) { return scyllasband::parse_score_text(text); }

void score_suite(const std::string&) {
    using scyllasband::Json;
    // 120 bpm: a beat is 0.5 s. "ba~1 by~2" is one word; the 0.25 s rest is a line break, the 0.125 s one is not.
    const scyllasband::Score score = parse(R"({"text": "give baby me so", "bpm": 120, "style": "ballad", "notes": [
        ["give", "C4", 1], ["-", "rest", 0.25], ["ba~1", "E4", 0.5], ["by~2", "D4", 0.5], ["-", "rest", 0.5], ["me", "G4", 1],
        ["so", "C5", 2]]})");
    expect(score.words.size() == 4 && score.words[1].text == "baby" && score.words[1].parts == 2, "syllable parts join into one word");
    expect(score.notes[2].word == 1 && score.notes[2].part == 1 && score.notes[3].word == 1 && score.notes[3].part == 2 && score.notes[1].word == -1 &&
               score.notes[4].beats == 0.5,
           "notes keep their word, part and beats");
    const auto lines = scyllasband::score_lines(score);
    expect(lines.size() == 2 && lines[0] == std::make_pair(0, 4) && lines[1] == std::make_pair(5, 6), "lines break at rests of 0.25 s or more");
    const Json summary = scyllasband::score_summary(score, 2);
    expect(summary.get("notes")[0].get("note").str() == "D4" && summary.get("notes")[0].get("midi").integer() == 62 &&
               summary.get("notes")[1].get("midi").is_null() && summary.get("notes")[1].get("word_index").is_null() &&
               summary.get("range").get("high_note").str() == "D5" && summary.get("range").get("low").integer() == 62 &&
               summary.get("mode").str() == "ballad" && summary.get("lines").size() == 2 && summary.get("transpose").integer() == 2,
           "summary after transposing: " + summary.dump().substr(0, 300));
    struct Bad {
        const char* score;
        int note_index;
        const char* message;
    };
    const Bad bad[] = {
        {R"({"bpm": 100, "notes": [["la", "C4", 1], ["la", "H4", 1]]})", 1, "Unknown note 'H4'"},
        {R"({"bpm": 100, "notes": [["la", "C4", 1], ["la", "C4", 0]]})", 1, "must last a positive number of beats"},
        {R"({"bpm": 100, "notes": [["ba~2", "C4", 1]]})", 0, "does not follow part 1"},
        {R"({"bpm": 100, "notes": [["la", "C4"]]})", 0, "must be [label, note, beats]"},
        {R"({"bpm": 0, "notes": [["la", "C4", 1]]})", -1, "bpm must be positive"},
        {R"({"bpm": 100, "style": "opera", "notes": [["la", "C4", 1]]})", -1, "Unknown style 'opera'"},
        {R"({"bpm": 100, "text": "la la", "notes": [["la", "C4", 1]]})", -1, "does not match the note labels"},
        {"{not json", -1, "not valid JSON"},
    };
    for (const Bad& b : bad) {
        bool caught = false;
        try {
            parse(b.score);
        } catch (const scyllasband::ScoreError& error) {
            caught = error.note_index() == b.note_index && std::string(error.what()).find(b.message) != std::string::npos;
            if (!caught) std::fprintf(stderr, "  got %d %s\n", error.note_index(), error.what());
        }
        expect(caught, std::string("score error ") + b.message);
    }
}

void preview_suite(const std::string&) {
    // A4 for one second, a half-second rest, then C4: 24 kHz.
    const scyllasband::Score score = parse(R"({"bpm": 60, "notes": [["la", "A4", 1], ["-", "rest", 0.5], ["la", "C4", 1]]})");
    const std::vector<float> tones = scyllasband::preview_tones(score, 24000);
    expect(tones.size() == static_cast<std::size_t>(2.5 * 24000), "preview length is the score's");
    int crossings = 0;
    for (std::size_t i = 1; i < 24000; ++i) crossings += (tones[i - 1] < 0.0f) != (tones[i] < 0.0f);
    expect(std::abs(crossings - 880) <= 4, "preview pitch: A4 crosses zero " + std::to_string(crossings) + " times a second");
    const bool silent = std::all_of(tones.begin() + 24000, tones.begin() + 36000, [](float v) { return v == 0.0f; });
    const float peak = *std::max_element(tones.begin(), tones.end());
    expect(silent && peak > 0.1f && peak <= 0.3f && std::fabs(tones[0]) < 1e-3f, "preview is silent on rests, soft, and starts without a click");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: scyllasband_frontend_test <suite> [fixtures dir]\n");
        return 2;
    }
    const std::map<std::string, std::function<void(const std::string&)>> suites = {
        {"noise", noise_suite},
        {"score", score_suite},
        {"timeline", timeline_suite},
        {"preview", preview_suite},
    };
    const auto suite = suites.find(argv[1]);
    if (suite == suites.end()) {
        std::fprintf(stderr, "unknown suite %s\n", argv[1]);
        return 2;
    }
    suite->second(argc > 2 ? argv[2] : "");
    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("%s: ok\n", argv[1]);
    return 0;
}
