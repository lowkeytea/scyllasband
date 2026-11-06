// Speech streaming (see streaming.py): within a chain, the plan's sentences are read as one stream of phones and spoken as
// targets, spans of that stream from one silence to another sized by the predicted durations to stay within the lengths
// the model was trained on; each target continues from the latents of the one before it, and decoding with the previous
// latents as left context joins consecutive targets as one continuous waveform.
#pragma once

#include "scyllasband_engine.h"
#include "scyllasband_speech_plan.h"

#include <functional>
#include <vector>

namespace scyllasband {

struct StreamEvent {
    enum Type { kPlan = 1, kChunkStarted = 2, kAudio = 3, kDone = 4 };
    Type type;
    const PlanChunk* chunk = nullptr;
    int chunk_count = 0;
    const std::vector<float>* audio = nullptr;
    Json metadata;
};

// Per-chunk noise source for tests: (chunk index, seed, has seed, frames, out [2 + latent_dim, frames]: the pitch predictor's
// two rows, then the flow's).
using ChunkNoiseFn = std::function<void(int, uint64_t, bool, int64_t, float*)>;

// Plans `options.text` and speaks it target by target; `emit` returns false to stop. Returns the final metadata.
Json synthesize_stream(Engine& engine, const SynthesisOptions& options, const std::function<bool(const StreamEvent&)>& emit,
                       const ChunkNoiseFn& chunk_noise = nullptr);

// Speaks `plan` target by target (the text fields of `options` are unused).
Json synthesize_plan_stream(Engine& engine, const Plan& plan, const SynthesisOptions& options, const std::function<bool(const StreamEvent&)>& emit,
                            const ChunkNoiseFn& chunk_noise = nullptr);

// Silences that can end a target: the end of the chain, a paragraph or record boundary, and the silences after a
// sentence end, after a clause pause and between words.
enum class TargetCut { kEnd, kParagraph, kSentence, kClause, kWord };

struct TargetCandidate {
    std::size_t index;  // the silence's position in the chain's phone stream
    TargetCut kind;
    int64_t frames;     // predicted frames from the target's start through this silence
};

// The end of a target among `candidates` (in stream order): the first target of a request ends at the earliest
// sentence end past the trained minimum (a clause, else a word break, only when the opening sentence does not fit); later targets end at the chain's end or a paragraph if it fits,
// else reach toward the trained maximum, preferring a sentence boundary, then a clause, then a word break; a remainder
// shorter than the minimum is never left before the end or a paragraph.
std::size_t choose_target(const std::vector<TargetCandidate>& candidates, bool first, int64_t shortest, int64_t longest);

}  // namespace scyllasband
