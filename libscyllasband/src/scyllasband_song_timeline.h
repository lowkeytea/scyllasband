// A song's timeline: phrases are mixed in at their offsets and audio is released once no later phrase can change it.
//
// singing.py sing() adds every phrase into one buffer at round(start * sample rate) (dropping the part of a phrase that
// starts before the song), then truncates the song to its length and clips it to [-1, 1]. Phrases arrive in time order and
// a phrase cut without a rest overlaps the previous one, so the samples before the next phrase's start are final once a
// phrase is added: the timeline releases exactly those, clipped, and keeps only the rest (at most about one phrase)
// pending. The released runs, concatenated, are the song sing() returns, sample for sample.
#pragma once

#include <cstdint>
#include <vector>

namespace scyllasband {

// singing.py gate_phrase: the model sometimes voices a breath or a syllable in a phrase's lead-in and closing rest, which is
// not in the score, so a phrase's audio is silenced until kLeadGate seconds before its first note and faded out
// kRelease seconds after its last sung note, with raised-cosine ramps of kGateRamp seconds.
constexpr double kLeadGate = 0.04, kRelease = 0.25, kGateRamp = 0.03;

// The gated copy of a phrase's audio; `first` and `last` are the first note's start and the last sung note's end, in seconds
// from the start of the phrase's audio.
std::vector<float> gate_phrase(const std::vector<float>& audio, int sample_rate, double first, double last);

class SongTimeline {
public:
    // `length`: the song's length in samples; `starts`: each phrase's first sample on the timeline, in phrase order
    // (non-decreasing; the first may be negative when its lead-in starts before the song).
    SongTimeline(int64_t length, std::vector<int64_t> starts);

    // Mixes the next phrase's audio in and returns the samples it settles: [released before the call, released after it),
    // clipped. Exactly one call per phrase, in order; the last call releases the rest of the song.
    std::vector<float> add(const std::vector<float>& audio);

    int64_t length() const { return length_; }
    int64_t released() const { return released_; }
    std::size_t pending() const { return pending_.size(); }  // samples held back, for tests

private:
    int64_t length_;
    std::vector<int64_t> starts_;
    std::size_t next_ = 0;
    int64_t released_ = 0;
    std::vector<float> pending_;  // samples [released_, released_ + pending_.size())
};

}  // namespace scyllasband
