// Speech planning: text (paragraphs, normalized, split into sentences) into plan chunks of one chain (see planner.py).
#pragma once

#include "scyllasband_engine.h"

#include <optional>
#include <string>
#include <vector>

namespace scyllasband {

struct PlanChunk {
    std::string chunk_id;
    int index = 0;
    int record_index = 0;
    std::string chain_id, voice, language;
    Delivery delivery;
    std::string text, source_text;
    bool paragraph_end = false;
    Json to_json() const;
};

struct Plan {
    std::vector<PlanChunk> chunks;
    Json to_json() const;
};

struct SynthesisOptions {
    std::string text, voice, language, delivery_spec;
    bool has_delivery = false;
    double speed = 1.0;
    int steps = 0;           // 0: bundle default
    std::string sampler;     // empty: bundle default
    std::optional<uint64_t> seed;
    double temperature = 1.0;
    bool normalize = true;
};

Plan plan_text(Engine& engine, const SynthesisOptions& options);

}  // namespace scyllasband
