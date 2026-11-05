// Melody scores (see singing.py): the runtime-free part of singing. Nothing here needs a bundle or a graph.
//
// A score (JSON) is "text", "bpm", "style" and "notes", an ordered list of [label, note, beats]: note a pitch name with
// octave (C4, F#3, Bb3) or "rest" (label "-"); label a word on one note, or "word~n" for the n-th of consecutive parts of
// one word (syllables, or a melisma holding one syllable over several pitches). The style picks the delivery mode: pop (or
// sung) and ballad (rap is parsed but only sung when a bundle lists it).
//
// parse_score validates a score (with Python's messages, and the index of the note at fault), turns notes into MIDI and
// beats into seconds, and groups labels into words; register_shift moves a song by octaves into a voice's sung range;
// split_phrases cuts it at rests into the phrases the model renders one at a time; score_lines groups it into the lines an
// editor shows; score_summary is the normalized form apps read back.
#pragma once

#include "scyllasband_json.h"

#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace scyllasband {

// The score does not follow the format. `note_index` (0-based) names the note at fault, -1 when the error is not one note's.
class ScoreError : public std::invalid_argument {
public:
    explicit ScoreError(const std::string& message, int note_index = -1) : std::invalid_argument(message), note_index_(note_index) {}
    int note_index() const { return note_index_; }

private:
    int note_index_;
};

struct ScoreNote {
    std::string label;
    std::optional<int> midi;        // nullopt for a rest
    double start = 0.0, end = 0.0;  // seconds
    double beats = 0.0;
    int word = -1;                  // index into Score::words; -1 for a rest
    int part = 0;                   // the note's part number within its word (1 for a one-note word); 0 for a rest
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

constexpr double kPhraseRest = 0.25;        // seconds of rest that end a phrase (and a line)
constexpr double kMaxPhraseSeconds = 10.0;  // a phrase is never packed beyond this
constexpr double kPhrasePad = 0.12;         // seconds of lead-in and release around each phrase
// The band, in semitones around a voice's sung centre, that a song's mean pitch is moved into by octaves. Pitch accuracy
// held within +-5 semitones of the centre and collapsed at +-8 in the register test; +7.5 sang flat.
constexpr double kRegisterBelow = 3.0, kRegisterAbove = 5.0;

// D#4 -> 63, Bb3 -> 58; nullopt for a rest ("rest", "-", "r", ""). Anything else throws ScoreError.
std::optional<int> note_to_midi(const std::string& name);
std::string midi_to_note(int midi);

// The delivery mode a style sings in (pop and sung -> sung, ballad -> ballad, rap -> rap); empty for an unknown style.
std::string style_mode(const std::string& style);
// Every style the format knows, in the order styles are listed (pop, sung, ballad, rap).
const std::vector<std::pair<std::string, std::string>>& style_modes();

Score parse_score(const Json& data);
// The score in `text` (a JSON document); invalid JSON throws ScoreError.
Score parse_score_text(const std::string& text);

// Octaves (as semitones: -24 .. 24) that move the score's mean pitch into [center - 3, center + 5], or as close as an octave
// step allows (ties go up); 0 without a centre.
int register_shift(const Score& score, std::optional<double> center);

struct Phrase {
    double start = 0.0, end = 0.0;  // on the song's timeline, lead-in and release included
    std::vector<ScoreNote> notes;   // in the phrase's own time (lead-in at 0), plus a closing rest
    std::vector<ScoreWord> words;
    std::string text;
};

// Phrases = runs of notes between rests of at least `phrase_rest` seconds, packed greedily up to `max_seconds` (lead-in and
// release excluded); a single run longer than that is cut at its longest internal rest, else at the word boundary nearest
// its middle.
std::vector<Phrase> split_phrases(const Score& score, double phrase_rest = kPhraseRest, double max_seconds = kMaxPhraseSeconds,
                                  double pad = kPhrasePad);

// The score's notes grouped into lines, [first, last] note indices that cover every note in order: a line ends with the
// rest(s) that separate it from the next sung note when they last `phrase_rest` seconds or more.
std::vector<std::pair<int, int>> score_lines(const Score& score, double phrase_rest = kPhraseRest);

// The normalized score apps read back: {"text", "bpm", "style", "mode", "seconds", "transpose", "notes": [{"index", "label",
// "word_index", "part", "note", "midi", "beats", "start", "end"}], "words": [{"word", "start", "end", "parts"}], "range":
// {"low", "high", "mean", "low_note", "high_note"}, "lines": [{"first", "last"}]}, after transposing by `transpose`.
Json score_summary(const Score& score, int transpose = 0);

}  // namespace scyllasband
