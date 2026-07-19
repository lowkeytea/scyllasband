#pragma once

#include <algorithm>
#include <string>
#include <vector>

namespace scyllasband_detail {

/**
 * LRU policy for heavyweight target-bucket sessions.
 *
 * A capacity of zero preserves the desktop/server default and retains every
 * bucket encountered. Positive values bound the number of warm buckets.
 */
class ScyllasBandSessionCachePolicy {
public:
    std::vector<std::string> set_capacity(int capacity) {
        capacity_ = std::max(0, capacity);
        return evict_over_capacity();
    }

    std::vector<std::string> touch(const std::string& key) {
        const auto existing = std::find(lru_keys_.begin(), lru_keys_.end(), key);
        if (existing != lru_keys_.end()) {
            lru_keys_.erase(existing);
        }
        lru_keys_.push_back(key);
        return evict_over_capacity();
    }

    int capacity() const { return capacity_; }
    const std::vector<std::string>& lru_keys() const { return lru_keys_; }

private:
    std::vector<std::string> evict_over_capacity() {
        std::vector<std::string> evicted;
        if (capacity_ <= 0) {
            return evicted;
        }
        while (static_cast<int>(lru_keys_.size()) > capacity_) {
            evicted.push_back(lru_keys_.front());
            lru_keys_.erase(lru_keys_.begin());
        }
        return evicted;
    }

    int capacity_ = 0;
    std::vector<std::string> lru_keys_;
};

}  // namespace scyllasband_detail
