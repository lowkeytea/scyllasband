// Speech planning (see planner.py): records of text, each with a voice, language and delivery, into plan chunks.
//
// A record's text is split into paragraphs, normalized and split into sentences. Consecutive records with the same
// voice, language and delivery form a chain, spoken continuously (the speech stream reads a chain as one stream of phones);
// a record that changes any of the three starts a new chain, even when it has no text.
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
    std::string source_kind = "text";
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
    double pause_ms = 0.0;   // silence before the first target of every chain after the first
};

// planner.plan_records: `records` is a JSON array of {"text", "voice", "language"?, "delivery"?, "intensity"?, "pace"?,
// "pitch"?} objects; delivery is a control=value string or a mapping, or the controls are given as fields (not both).
Plan plan_records(Engine& engine, const Json& records, bool normalize = true, const std::string& source_kind = "text");

// One record: options.text with its voice, language and delivery.
Plan plan_text(Engine& engine, const SynthesisOptions& options);

}  // namespace scyllasband
