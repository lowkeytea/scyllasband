// Song planning (see singing.py): a score into the phrases the model renders, for one bundle and voice.
//
// The planner sees the bundle only through SongFrontend (its G2P, rates, limits and singing controls), never the engine,
// so it is tested with a fake frontend. For each phrase (split_phrases) it gives the phones (the bundle's G2P, one group
// per score word), integer frames per phone from the words' time spans (phone_durations: consonants share 1 - vowel_share
// of a word's frames, vowels the rest; rests and the lead-in/release become <sil> frames) and the melody input in the
// flow's layout (frame_melody: per frame (midi - midi_center) / midi_scale, the phase inside the note and 1 where a note
// sounds; zeros on rests). resolve_sing_request is the one place a request is checked (voice, language, style, decoder,
// steps, sampler, guidance); song_fit reports how a song sits in a voice's range without any graph.
#pragma once

#include "scyllasband_errors.h"
#include "scyllasband_events.h"
#include "scyllasband_g2p.h"
#include "scyllasband_json.h"
#include "scyllasband_score.h"
#include "scyllasband_singing_controls.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace scyllasband {

// What the planner needs from a bundle.
class SongFrontend {
public:
    virtual ~SongFrontend() = default;
    virtual Phonemized phonemize(const std::string& text, const std::string& language) = 0;
    // The language to sing `voice` in (empty `language`: the voice's default); throws for an unknown voice or language.
    virtual std::string resolve_language(const std::string& voice, const std::string& language) const = 0;
    virtual const SingingControls& singing() const = 0;
    virtual int sample_rate() const = 0;
    virtual int latent_hop() const = 0;
    // The longest phrase, in latent frames, that the bundle can both render and decode with every decoder it has here, so a
    // plan does not depend on the decoder chosen.
    virtual int64_t max_phrase_frames() const = 0;
    virtual bool has_decoder(const std::string& decoder) const = 0;
    virtual std::vector<std::string> available_decoders() const = 0;  // sorted
};

// The delivery modes the bundle sings in (controls.delivery_modes.singing.modes; every non-spoken mode with an id when it
// lists none), and the score styles it accepts, in the order styles are listed.
std::vector<std::string> singing_modes(const SingingControls& controls);
std::vector<std::string> singing_styles(const SingingControls& controls);

// Phones in the training layout (<sil>, word, <sil>, word, ..., <sil>) and each later word's first phone.
std::pair<std::vector<std::string>, std::vector<int>> word_phones(const std::vector<std::string>& words, SongFrontend& frontend,
                                                                  const std::string& language);
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
    std::optional<double> max_seconds;  // default: what the bundle's longest phrase holds, at most kMaxPhraseSeconds
    double pad = kPhrasePad;
    std::optional<double> vowel_share;
};

// A sing request before checking.
struct SingRequest {
    std::string voice, language;   // language: empty for the voice's default
    std::optional<uint64_t> seed;  // phrase i draws from seed + i
    int steps = 0;                 // 0: the bundle's singing steps
    std::string sampler;           // empty: the bundle's singing sampler
    std::string decoder;           // empty: the bundle's singing decoder where it has its graphs, else vocos
    std::optional<double> guidance;
    PlanSettings plan;
};

// The settings a sung phrase renders with.
struct SingSettings {
    int steps = 8;
    std::string sampler = "heun", decoder = "vocos";
    double guidance = 1.0;
};

// A checked request: the voice (stripped), its language, the parsed score (style checked against the bundle) and the
// render settings.
struct ResolvedSing {
    std::string voice, language;
    Score score;
    SingSettings settings;
};

ResolvedSing resolve_sing_request(SongFrontend& frontend, const Score& score, const SingRequest& request);
SingSettings resolve_sing_settings(const SongFrontend& frontend, const SingRequest& request);

SongPlan plan_song(const Score& score, SongFrontend& frontend, const std::string& voice, const std::string& language,
                   const PlanSettings& settings = {});

// How a song sits in a voice's range, after `transpose` and (with `use_register`) the octave register: no G2P, no graphs.
struct SongFit {
    std::optional<double> voice_center;  // MIDI
    double band_low = 0.0, band_high = 0.0;  // the band the register aims for: [center - 3, center + 5]
    int transpose = 0, register_shift = 0;
    double mean_midi = 0.0;
    int low_midi = 0, high_midi = 0;
    double outside_semitones = 0.0;  // how far the sung mean lies outside the band (0 inside it)
    int suggested_transpose = 0;     // `transpose` when the song fits; else the smallest transpose in [-6, 6] that fits (or comes nearest)
    Json to_json() const;
};

SongFit song_fit(const SingingControls& controls, const Score& score, const std::string& voice, int transpose, bool use_register);

}  // namespace scyllasband
