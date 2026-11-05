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
