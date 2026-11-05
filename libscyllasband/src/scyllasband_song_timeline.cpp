#include "scyllasband_song_timeline.h"

#include <algorithm>
#include <stdexcept>

namespace scyllasband {

SongTimeline::SongTimeline(int64_t length, std::vector<int64_t> starts) : length_(std::max<int64_t>(0, length)), starts_(std::move(starts)) {
    for (std::size_t i = 1; i < starts_.size(); ++i) {
        if (starts_[i] < starts_[i - 1]) throw std::invalid_argument("Phrase starts must not decrease");
    }
}

std::vector<float> SongTimeline::add(const std::vector<float>& audio) {
    if (next_ >= starts_.size()) throw std::logic_error("More phrases than the timeline was made for");
    int64_t offset = starts_[next_++];
    std::size_t skip = 0;
    if (offset < 0) {   // a lead-in that starts before the song loses what precedes it
        skip = static_cast<std::size_t>(std::min<int64_t>(-offset, static_cast<int64_t>(audio.size())));
        offset = 0;
    }
    // Mix the phrase into the pending samples (in phrase order, as sing() adds them); nothing past the song's end is kept.
    const int64_t end = std::min<int64_t>(offset + static_cast<int64_t>(audio.size() - skip), length_);
    if (end > released_ + static_cast<int64_t>(pending_.size())) pending_.resize(static_cast<std::size_t>(end - released_), 0.0f);
    for (int64_t t = std::max(offset, released_); t < end; ++t) {
        pending_[static_cast<std::size_t>(t - released_)] += audio[skip + static_cast<std::size_t>(t - offset)];
    }
    // Everything before the next phrase's start is final (later phrases start at or after it).
    const int64_t settle = next_ < starts_.size() ? std::min(std::max<int64_t>(starts_[next_], 0), length_) : length_;
    std::vector<float> out;
    if (settle > released_) {
        const std::size_t count = static_cast<std::size_t>(settle - released_);
        out.assign(count, 0.0f);
        const std::size_t have = std::min(count, pending_.size());
        for (std::size_t i = 0; i < have; ++i) out[i] = std::clamp(pending_[i], -1.0f, 1.0f);
        pending_.erase(pending_.begin(), pending_.begin() + static_cast<std::ptrdiff_t>(have));
        released_ = settle;
    }
    return out;
}

}  // namespace scyllasband
