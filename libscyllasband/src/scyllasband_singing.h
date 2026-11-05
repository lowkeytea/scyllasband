// Singing: a melody score to sung audio on a v4 bundle (see singing.py, which this follows step for step).
//
// A score (JSON) is "text", "bpm", "style" and "notes", an ordered list of [label, note, beats]: note a pitch name with
// octave (C4, F#3, Bb3) or "rest" (label "-"); label a word on one note, or "word~n" for the n-th of consecutive parts of
// one word (syllables, or a melisma holding one syllable over several pitches). The style picks the delivery mode: pop (or
// sung) and ballad.
//
// The song is rendered by phrases, the way the model was trained on clips of a few seconds:
// 1. parse_score validates the score, turns notes into MIDI and beats into seconds, and groups labels into words;
// 2. register_shift moves the whole song by octaves so its mean pitch sits in the voice's sung range (after `transpose`);
// 3. split_phrases cuts the song at rests of 0.25 s or more and packs the runs between them up to the longest phrase the
//    bundle can render (at most 10 s), each with 0.12 s of lead-in and release;
// 4. plan_song gives each phrase its phones (the bundle's G2P, one group per score word), integer frames per phone from the
//    words' time spans (phone_durations) and the melody input (frame_melody: per frame (midi - 60) / 24, the phase inside
//    the note and 1 where a note sounds; zeros on rests);
// 5. sing_stream renders each phrase (the score's durations and melody, the delivery mode, the singing sampler, steps and
//    guidance, no delivery request, no acoustic prefix), decodes it with the singing decoder and places it on the song's
//    timeline, so the rests between phrases are silence.
#pragma once

#include "scyllasband_engine.h"

#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace scyllasband {

// The score does not follow the format.
class ScoreError : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

struct ScoreNote {
    std::string label;
    std::optional<int> midi;  // nullopt for a rest
    double start = 0.0, end = 0.0;  // seconds
    std::string name() const;
};

struct ScoreWord {
    std::string text;
    double start = 0.0, end = 0.0;
    int parts = 1;
};

struct Score {
    std::string text;
    double bpm = 0.0;
    std::string style, mode;  // mode: the delivery mode (sung, ballad, rap)
    std::vector<ScoreNote> notes;
    std::vector<ScoreWord> words;

    double seconds() const { return notes.empty() ? 0.0 : notes.back().end; }
    std::vector<int> pitched() const;
    Score transposed(int semitones) const;
    Json to_json() const;
};

// D#4 -> 63, Bb3 -> 58; nullopt for a rest ("rest", "-", "r", ""). Anything else throws ScoreError.
std::optional<int> note_to_midi(const std::string& name);
std::string midi_to_note(int midi);
Score parse_score(const Json& data);

// The delivery modes the bundle sings in, and the score styles it accepts (pop before sung before ballad before rap).
std::vector<std::string> singing_modes(const Engine& engine);
std::vector<std::string> singing_styles(const Engine& engine);
// The MIDI pitch the voice sings around: the manifest's voice_midi_center, else the middle of its spoken pitch edges.
std::optional<double> voice_sung_center(const Engine& engine, const std::string& voice);
// Octaves (as semitones: -24 .. 24) that move the score's mean pitch into [center - 3, center + 5].
int register_shift(const Score& score, std::optional<double> center);

struct Phrase {
    double start = 0.0, end = 0.0;  // on the song's timeline, lead-in and release included
    std::vector<ScoreNote> notes;   // in the phrase's own time (lead-in at 0), plus a closing rest
    std::vector<ScoreWord> words;
    std::string text;
};

constexpr double kPhraseRest = 0.25;        // seconds of rest that end a phrase
constexpr double kMaxPhraseSeconds = 10.0;  // a phrase is never packed beyond this
constexpr double kPhrasePad = 0.12;         // seconds of lead-in and release around each phrase

std::vector<Phrase> split_phrases(const Score& score, double phrase_rest = kPhraseRest, double max_seconds = kMaxPhraseSeconds,
                                  double pad = kPhrasePad);

// Phones in the training layout (<sil>, word, <sil>, word, ..., <sil>) and each later word's first phone.
std::pair<std::vector<std::string>, std::vector<int>> word_phones(const std::vector<std::string>& words, Engine& engine, const std::string& language);
bool is_vowel(const std::string& phone);
std::vector<int> phone_durations(const std::vector<std::string>& phones, const std::vector<int>& word_starts, const std::vector<ScoreWord>& words,
                                 int64_t total_frames, double vowel_share, int hop, int sample_rate);
// [frames, 3] row-major: the note's (midi - midi_center) / midi_scale, the frame's phase inside the note, 1; zeros on rests.
std::vector<float> frame_melody(const std::vector<ScoreNote>& notes, int64_t frames, int hop, int sample_rate, double midi_center, double midi_scale);

struct PhrasePlan {
    int index = 0;
    Phrase phrase;
    Sentence sentence;
    std::vector<int> durations;
    int64_t frames = 0;
    std::vector<float> melody;  // [frames, 3]
    Json to_json(bool with_melody = false) const;
};

struct SongPlan {
    Score score;  // after the transpose and the register shift
    std::string voice, language, mode;
    int64_t mode_id = 0;
    int register_shift = 0;
    int transpose = 0;
    std::vector<PhrasePlan> phrases;
    double seconds() const;
    Json to_json(bool with_melody = false) const;
};

struct PlanSettings {
    bool use_register = true;
    int transpose = 0;
    double phrase_rest = kPhraseRest;
    std::optional<double> max_seconds;  // default: what the bundle's largest bucket holds, at most kMaxPhraseSeconds
    double pad = kPhrasePad;
    std::optional<double> vowel_share;
};

SongPlan plan_song(const Score& score, Engine& engine, const std::string& voice, const std::string& language, const PlanSettings& settings = {});

struct SingOptions {
    std::string voice, language;   // language: empty for the voice's default
    std::optional<uint64_t> seed;  // phrase i draws from seed + i
    int steps = 0;                 // 0: the bundle's singing steps
    std::string sampler;           // empty: the bundle's singing sampler
    std::string decoder;           // empty: the bundle's singing decoder where it has its graphs, else vocos
    std::optional<double> guidance;
    PlanSettings plan;
};

struct SingEvent {
    enum Type { kPlan = 1, kPhraseStarted = 2, kAudio = 3, kDone = 4 };
    Type type;
    const PhrasePlan* phrase = nullptr;
    int phrase_count = 0;
    const std::vector<float>* audio = nullptr;  // the phrase's samples; metadata "offset_samples" places them on the song timeline
    Json metadata;
};

// Per-phrase flow noise for tests: (phrase index, seed, has seed, frames, out [2 + latent_dim, frames]; the first two rows unused).
using PhraseNoiseFn = std::function<void(int, uint64_t, bool, int64_t, float*)>;

// The singing settings a request resolves to (steps, sampler, decoder, guidance), checked against the bundle.
struct SingSettings {
    int steps = 8;
    std::string sampler = "heun", decoder = "vocos";
    double guidance = 1.0;
};
SingSettings sing_settings(const Engine& engine, const SingOptions& options);

// Plans and renders a score phrase by phrase; `emit` returns false to stop. Returns the song's metadata (the plan, the
// rendered phrases, the settings) with "samples": the song's length in samples.
Json sing_stream(Engine& engine, const Json& score, const SingOptions& options, const std::function<bool(const SingEvent&)>& emit,
                 const PhraseNoiseFn& phrase_noise = nullptr);

// The whole song as one waveform (phrases placed on the timeline, clipped to [-1, 1]).
std::vector<float> sing(Engine& engine, const Json& score, const SingOptions& options, Json* metadata = nullptr,
                        const PhraseNoiseFn& phrase_noise = nullptr);

// {"supported", "styles", "decoders", "default_decoder", "sampler", "steps", "guidance", "voice_midi_center"} for
// scyllasband_voices_json.
Json singing_info(const Engine& engine);

}  // namespace scyllasband
