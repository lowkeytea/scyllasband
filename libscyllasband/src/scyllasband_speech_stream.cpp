#include "scyllasband_speech_stream.h"

#include "scyllasband_noise.h"
#include "scyllasband_text.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iterator>
#include <map>
#include <optional>
#include <random>
#include <memory>
#include <set>
#include <stdexcept>

namespace scyllasband {

namespace {

double elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

double round1(double value) { return std::nearbyint(value * 10.0) / 10.0; }

}  // namespace

namespace {

const std::set<std::string> kSentenceEnd = {"<end_stmt>", "<end_question>", "<end_exclaim>", "<ellipsis>"};
const std::set<std::string> kClausePause = {"<pause_comma>", "<pause_semicolon>", "<pause_colon>", "<pause_dash>"};
constexpr std::size_t kWindowPhones = 200;      // phones measured ahead to choose a target's end; past the trained maximum at any speed
constexpr std::size_t kFirstWindowPhones = 48;  // the first target looks only this far ahead at first, so little text is phonemized early
constexpr double kFillShare = 0.5;              // a later target prefers a sentence end in the upper half of the trained range

int64_t total_frames(const std::vector<int>& durations) {
    int64_t frames = 0;
    for (int value : durations) frames += value;
    return frames;
}

// One plan sentence in its chain's phone stream.
struct Piece {
    const PlanChunk* chunk;
    std::size_t start, end;                // stream positions of its leading and trailing silences
    std::vector<std::size_t> word_phones;  // stream position of each word's first phone
    int parts = 0;                         // targets that took part of the sentence so far
};

// One chain's sentences joined as connected text (one shared silence between sentences), phonemized only as far as the
// targets and their context need. `cuts` maps the silences that can end a target to their kind.
class ChainStream {
public:
    ChainStream(Engine& engine, std::vector<const PlanChunk*> chunks) : engine_(engine), chunks_(std::move(chunks)) {}

    const PlanChunk& front() const { return *chunks_.front(); }
    const std::vector<std::string>& phones() const { return phones_; }
    const std::map<std::size_t, TargetCut>& cuts() const { return cuts_; }
    bool complete() const { return pieces_.size() == chunks_.size(); }

    void extend(std::size_t length) {
        while (phones_.size() < length && !complete()) append(*chunks_[pieces_.size()]);
    }

    // Phonemizes until the context budget of phones (silences excluded) follows `end`, or the chain is complete.
    void extend_context(std::size_t end) {
        while (!complete() && spoken_after(end) < engine_.context_phones()) append(*chunks_[pieces_.size()]);
    }

    bool at_end(std::size_t index) {
        extend(index + 2);
        return complete() && index + 1 == phones_.size();
    }

    // The chains spoken just before and after this one by the same voice in the same language (only the delivery changes):
    // their text is context for this chain's targets, and the audio continues across the change.
    void link(ChainStream* previous, ChainStream* following) {
        if (previous != nullptr) previous_ = previous;
        if (following != nullptr) following_ = following;
    }
    bool continues() const { return previous_ != nullptr; }

    // The last `count` context phones of this chain (already spoken), reaching into the chains before it.
    std::vector<int64_t> context_tail(std::size_t count) const {
        std::vector<int64_t> ids = engine_.context_ids(phones_);
        if (ids.size() > count) ids.erase(ids.begin(), ids.end() - static_cast<std::ptrdiff_t>(count));
        if (ids.size() < count && previous_ != nullptr) {
            std::vector<int64_t> lead = previous_->context_tail(count - ids.size());
            ids.insert(ids.begin(), lead.begin(), lead.end());
        }
        return ids;
    }

    // The first `count` context phones of this chain, phonemized only as far as needed, reaching into the next.
    std::vector<int64_t> context_head(std::size_t count) {
        while (!complete() && spoken_from(0) < count) append(*chunks_[pieces_.size()]);
        std::vector<int64_t> ids = engine_.context_ids(phones_);
        if (ids.size() > count) ids.resize(count);
        if (ids.size() < count && following_ != nullptr) {
            const std::vector<int64_t> rest = following_->context_head(count - ids.size());
            ids.insert(ids.end(), rest.begin(), rest.end());
        }
        return ids;
    }

    // The target [start, end] (silence to silence) with its context.
    Sentence sentence(std::size_t start, std::size_t end) {
        const std::size_t reach = engine_.context_phones();
        extend_context(end);
        std::size_t first_word = start;
        while (first_word <= end && phones_[first_word] == kSilencePhone) ++first_word;
        Sentence out;
        out.phones.assign(phones_.begin() + static_cast<std::ptrdiff_t>(start), phones_.begin() + static_cast<std::ptrdiff_t>(end + 1));
        for (std::size_t word : word_starts_) {
            if (start < word && word <= end && word != first_word) out.word_starts.push_back(static_cast<int>(word - start));
        }
        for (auto pause = min_pauses_.upper_bound(start); pause != min_pauses_.end() && *pause <= end; ++pause) {
            out.min_pauses.push_back(static_cast<int>(*pause - start));
        }
        out.before_ids = engine_.context_ids(slice(start > 3 * reach ? start - 3 * reach : 0, start));
        if (out.before_ids.size() > reach) out.before_ids.erase(out.before_ids.begin(), out.before_ids.end() - static_cast<std::ptrdiff_t>(reach));
        if (out.before_ids.size() < reach && previous_ != nullptr) {
            const std::vector<int64_t> lead = previous_->context_tail(reach - out.before_ids.size());
            out.before_ids.insert(out.before_ids.begin(), lead.begin(), lead.end());
        }
        out.after_ids = engine_.context_ids(slice(end + 1, end + 1 + 3 * reach));
        if (out.after_ids.size() > reach) out.after_ids.resize(reach);
        if (out.after_ids.size() < reach && following_ != nullptr) {
            const std::vector<int64_t> rest = following_->context_head(reach - out.after_ids.size());
            out.after_ids.insert(out.after_ids.end(), rest.begin(), rest.end());
        }
        return out;
    }

    // A plan chunk describing the target: its sentences' ids and text (part of a sentence gets `id.n`).
    PlanChunk target_chunk(std::size_t start, std::size_t end) {
        std::vector<std::string> labels, texts;
        const Piece* last = nullptr;
        for (Piece& piece : pieces_) {
            if (piece.start >= end || piece.end <= start) continue;
            last = &piece;
            std::vector<std::size_t> inside;
            for (std::size_t i = 0; i < piece.word_phones.size(); ++i) {
                if (start < piece.word_phones[i] && piece.word_phones[i] < end) inside.push_back(i);
            }
            if (inside.size() == piece.word_phones.size()) {
                labels.push_back(piece.chunk->chunk_id);
                texts.push_back(piece.chunk->text);
                continue;
            }
            piece.parts += 1;
            labels.push_back(piece.chunk->chunk_id + "." + std::to_string(piece.parts));
            const std::vector<Text> words = unicode::split_whitespace(unicode::decode(piece.chunk->text));
            std::size_t from = 0, to = 0;
            if (!inside.empty() && words.size() == piece.word_phones.size()) {
                from = inside.front();
                to = inside.back() + 1;
            } else if (!inside.empty()) {  // the text's words do not line up with the G2P's: share them out evenly
                const double share = static_cast<double>(words.size()) / static_cast<double>(std::max<std::size_t>(1, piece.word_phones.size()));
                from = static_cast<std::size_t>(std::nearbyint(static_cast<double>(inside.front()) * share));
                to = static_cast<std::size_t>(std::nearbyint(static_cast<double>(inside.back() + 1) * share));
            }
            std::string text;
            for (std::size_t i = from; i < std::min(to, words.size()); ++i) text += (text.empty() ? "" : " ") + unicode::encode(words[i]);
            texts.push_back(std::move(text));
        }
        if (last == nullptr) throw std::logic_error("Target outside its chain");
        PlanChunk chunk = *last->chunk;
        chunk.chunk_id = labels.size() == 1 ? labels.front() : labels.front() + "+" + labels.back();
        chunk.text.clear();
        for (const std::string& text : texts) {
            if (!text.empty()) chunk.text += (chunk.text.empty() ? "" : " ") + text;
        }
        return chunk;
    }

private:
    std::vector<std::string> slice(std::size_t from, std::size_t to) const {
        to = std::min(to, phones_.size());
        if (from >= to) return {};
        return std::vector<std::string>(phones_.begin() + static_cast<std::ptrdiff_t>(from), phones_.begin() + static_cast<std::ptrdiff_t>(to));
    }

    std::size_t spoken_from(std::size_t first) const {
        std::size_t count = 0;
        for (std::size_t i = first; i < phones_.size(); ++i) count += phones_[i] != kSilencePhone;
        return count;
    }
    std::size_t spoken_after(std::size_t end) const { return spoken_from(end + 1); }

    void append(const PlanChunk& chunk) {
        const Phonemized result = engine_.phonemize(chunk.text, chunk.language);
        const std::vector<std::string>& phones = result.phones;
        const std::size_t base = phones_.empty() ? 0 : phones_.size() - 1;
        if (!phones_.empty()) {
            const PlanChunk& previous = *pieces_.back().chunk;
            cuts_[base] = previous.paragraph_end || previous.record_index != chunk.record_index ? TargetCut::kParagraph : TargetCut::kSentence;
            phones_.insert(phones_.end(), phones.begin() + 1, phones.end());
        } else {
            phones_.insert(phones_.end(), phones.begin(), phones.end());
        }
        std::size_t first = 0;
        while (first < phones.size() && phones[first] == kSilencePhone) ++first;
        Piece piece{&chunk, base, base + phones.size() - 1, {base + first}, 0};
        if (!pieces_.empty()) word_starts_.push_back(base + first);
        for (int word : result.word_starts) {
            word_starts_.push_back(base + static_cast<std::size_t>(word));
            piece.word_phones.push_back(base + static_cast<std::size_t>(word));
        }
        const std::set<int> candidates(result.word_boundary_candidates.begin(), result.word_boundary_candidates.end());
        for (int pause : result.min_pause_phones) min_pauses_.insert(base + static_cast<std::size_t>(pause));
        for (std::size_t j = 1; j + 1 < phones.size(); ++j) {
            if (phones[j] != kSilencePhone) continue;
            if (kSentenceEnd.count(phones[j - 1])) cuts_[base + j] = TargetCut::kSentence;
            else if (kClausePause.count(phones[j - 1])) cuts_[base + j] = TargetCut::kClause;
            else if (candidates.count(static_cast<int>(j))) cuts_[base + j] = TargetCut::kWord;
        }
        cuts_[base + phones.size() - 1] = TargetCut::kEnd;
        pieces_.push_back(std::move(piece));
    }

    Engine& engine_;
    std::vector<const PlanChunk*> chunks_;
    std::vector<std::string> phones_;
    std::vector<std::size_t> word_starts_;  // every word's first phone but the chain's first
    std::map<std::size_t, TargetCut> cuts_;
    std::set<std::size_t> min_pauses_;      // silences after a dash typed with no spaces around it
    std::vector<Piece> pieces_;
    ChainStream* previous_ = nullptr;
    ChainStream* following_ = nullptr;
};

struct Target {
    std::size_t end = 0;
    Sentence sentence;
    std::vector<int> durations;
};

// The next target of `stream` from `start`: its end, sentence and durations.
// A sentence end, a paragraph or record end, or the end of the chain: where the first target prefers to stop.
bool ends_sentence(TargetCut kind) { return kind == TargetCut::kSentence || kind == TargetCut::kParagraph || kind == TargetCut::kEnd; }

Target next_target(Engine& engine, ChainStream& stream, std::size_t start, bool first, const DeliveryTensors& delivery, double speed) {
    const int64_t shortest = engine.min_target_frames(), longest = engine.max_target_frames();
    const int64_t voice = engine.voice_index(stream.front().voice), language = engine.language_index(stream.front().language);
    std::map<std::size_t, Target> measured;  // the window's measurement is reused when the chosen end is the same silence
    auto measure = [&](std::size_t end) {
        auto found = measured.find(end);
        if (found != measured.end()) return found->second;
        Target target{end, stream.sentence(start, end), {}};
        target.durations = engine.durations(target.sentence, voice, language, delivery, speed);
        measured.emplace(end, target);
        return target;
    };
    std::size_t window = first ? kFirstWindowPhones : kWindowPhones;
    std::vector<TargetCandidate> stops;
    while (true) {
        stream.extend(start + window + 2);
        const std::size_t last = std::min(stream.phones().size() - 1, start + window);
        const auto next = stream.cuts().upper_bound(start);
        if (next == stream.cuts().end()) throw std::logic_error("No silence ends the target");
        stops.clear();
        for (auto cut = next; cut != stream.cuts().end() && cut->first <= last; ++cut) stops.push_back({cut->first, cut->second, 0});
        if (stops.empty()) stops.push_back({next->first, next->second, 0});  // no silence within the window: run to the next one
        const bool no_sentence_end = std::none_of(stops.begin(), stops.end(), [](const TargetCandidate& c) { return ends_sentence(c.kind); });
        if (first && window < kWindowPhones && !stream.at_end(stops.back().index) && no_sentence_end) {
            window *= 2;  // no sentence end in view yet: widen before measuring
            continue;
        }
        const Target measured = measure(stops.back().index);
        int64_t reach = 0;
        std::size_t position = start;
        for (TargetCandidate& stop : stops) {
            for (; position <= stop.index; ++position) reach += measured.durations[position - start];
            stop.frames = reach;
        }
        // The first target widens its window until a sentence end past the minimum is in view.
        if (!first || window >= kWindowPhones || stream.at_end(stops.back().index) ||
            std::any_of(stops.begin(), stops.end(), [&](const TargetCandidate& c) { return ends_sentence(c.kind) && c.frames >= shortest; })) {
            break;
        }
        window *= 2;
    }
    std::vector<TargetCandidate> candidates;
    for (const TargetCandidate& stop : stops) {  // a paragraph or record ends the target once it is long enough
        candidates.push_back(stop);
        if (stop.kind == TargetCut::kEnd || (stop.kind == TargetCut::kParagraph && stop.frames >= shortest)) break;
    }
    while (true) {
        const std::size_t end = choose_target(candidates, first, shortest, longest);
        Target target = measure(end);
        std::vector<TargetCandidate> earlier;
        for (const TargetCandidate& c : candidates) {
            if (c.index < end) earlier.push_back(c);
        }
        if (total_frames(target.durations) <= longest || earlier.empty()) return target;
        candidates = std::move(earlier);  // over the maximum: the furthest earlier boundary that fits
        first = false;
    }
}

struct StreamSettings {
    std::string sampler;
    int steps;
};

StreamSettings stream_settings(const Engine& engine, const SynthesisOptions& options) {
    std::string sampler = unicode::encode(unicode::lower(unicode::decode(options.sampler.empty() ? engine.default_sampler() : options.sampler)));
    if (sampler != "heun" && sampler != "euler") throw std::invalid_argument("Unsupported sampler '" + sampler + "'; choose one of heun, euler");
    if (!(options.speed > 0.0) || !std::isfinite(options.speed)) throw std::invalid_argument("speed must be positive");
    if (options.steps < 0) throw std::invalid_argument("steps must be positive");
    if (!std::isfinite(options.temperature)) throw std::invalid_argument("temperature must be finite");
    if (!std::isfinite(options.pause_ms)) throw std::invalid_argument("pause_ms must be finite");
    return StreamSettings{sampler, options.steps > 0 ? options.steps : engine.default_steps()};
}

template <typename Keep>
std::vector<TargetCandidate> only(const std::vector<TargetCandidate>& candidates, Keep keep) {
    std::vector<TargetCandidate> out;
    std::copy_if(candidates.begin(), candidates.end(), std::back_inserter(out), keep);
    return out;
}

}  // namespace

std::size_t choose_target(const std::vector<TargetCandidate>& candidates, bool first, int64_t shortest, int64_t longest) {
    if (candidates.empty()) throw std::invalid_argument("No target candidates");
    std::vector<TargetCandidate> fit = only(candidates, [&](const TargetCandidate& c) { return c.frames <= longest; });
    if (fit.empty()) fit.push_back(candidates.front());
    const std::vector<TargetCandidate> long_enough = only(fit, [&](const TargetCandidate& c) { return c.frames >= shortest; });
    const TargetCandidate& stop = candidates.back();
    const bool has_stop = stop.kind == TargetCut::kEnd || stop.kind == TargetCut::kParagraph;
    const bool stop_fits = has_stop && std::any_of(fit.begin(), fit.end(), [&](const TargetCandidate& c) { return c.index == stop.index; });
    if (stop_fits && !first) return stop.index;
    const std::vector<TargetCandidate> marks = only(long_enough, [](const TargetCandidate& c) { return c.kind != TargetCut::kWord; });
    TargetCandidate choice{};
    if (first) {
        const std::vector<TargetCandidate> ends = only(long_enough, [](const TargetCandidate& c) { return ends_sentence(c.kind); });
        choice = !ends.empty() ? ends.front() : stop_fits ? stop : !marks.empty() ? marks.front() : (long_enough.empty() ? fit : long_enough).front();
    } else {
        const std::vector<TargetCandidate> full =
            only(marks, [&](const TargetCandidate& c) { return static_cast<double>(c.frames) >= kFillShare * static_cast<double>(longest); });
        const std::vector<TargetCandidate> sentences =
            only(full, [](const TargetCandidate& c) { return c.kind == TargetCut::kSentence || c.kind == TargetCut::kParagraph; });
        const std::vector<TargetCandidate>* preference[] = {&sentences, &full, &marks, &long_enough, &fit};
        for (const std::vector<TargetCandidate>* group : preference) {
            if (!group->empty()) {
                choice = group->back();
                break;
            }
        }
    }
    if (has_stop && choice.index != stop.index && stop.frames - choice.frames < shortest) {  // never leave a short remainder
        if (stop_fits) return stop.index;
        const std::vector<TargetCandidate> balanced = only(long_enough, [&](const TargetCandidate& c) { return stop.frames - c.frames >= shortest; });
        if (!balanced.empty()) {
            const std::vector<TargetCandidate> marked = only(balanced, [](const TargetCandidate& c) { return c.kind != TargetCut::kWord; });
            choice = (marked.empty() ? balanced : marked).back();
        }
    }
    return choice.index;
}

Json synthesize_stream(Engine& engine, const SynthesisOptions& options, const std::function<bool(const StreamEvent&)>& emit,
                       const ChunkNoiseFn& chunk_noise) {
    stream_settings(engine, options);
    return synthesize_plan_stream(engine, plan_text(engine, options), options, emit, chunk_noise);
}

Json synthesize_plan_stream(Engine& engine, const Plan& plan, const SynthesisOptions& options, const std::function<bool(const StreamEvent&)>& emit,
                            const ChunkNoiseFn& chunk_noise) {
    const StreamSettings settings = stream_settings(engine, options);
    if (plan.chunks.empty()) throw std::invalid_argument("Nothing to speak: the text has no words");
    const auto started = std::chrono::steady_clock::now();
    const int chunk_count = static_cast<int>(plan.chunks.size());
    auto send = [&](StreamEvent event) {
        event.chunk_count = chunk_count;
        if (!emit(event)) throw CancelledError();
    };
    send(StreamEvent{StreamEvent::kPlan, nullptr, 0, nullptr, plan.to_json()});

    const std::size_t D = static_cast<std::size_t>(engine.latent_dim());
    const int64_t keep = std::max(engine.prefix_frames(), engine.decode_context());
    int spoken = 0;
    std::string previous_chain;   // the chain of the previous spoken target
    std::optional<double> first_audio_ms;
    Json chunk_metadata{Json::Array{}};
    std::random_device entropy;

    std::vector<std::unique_ptr<ChainStream>> streams;
    for (std::size_t begin = 0; begin < plan.chunks.size();) {
        std::vector<const PlanChunk*> chain;  // consecutive plan sentences that share a chain
        for (; begin < plan.chunks.size() && (chain.empty() || plan.chunks[begin].chain_id == chain.front()->chain_id); ++begin) {
            chain.push_back(&plan.chunks[begin]);
        }
        streams.push_back(std::make_unique<ChainStream>(engine, std::move(chain)));
    }
    // The same voice going on with a new delivery is one passage (unless a pause is asked for between chains).
    for (std::size_t i = 1; i < streams.size(); ++i) {
        const PlanChunk& a = streams[i - 1]->front();
        const PlanChunk& b = streams[i]->front();
        if (options.pause_ms <= 0.0 && a.voice == b.voice && a.language == b.language) {
            streams[i]->link(streams[i - 1].get(), nullptr);
            streams[i - 1]->link(nullptr, streams[i].get());
        }
    }
    std::vector<float> tail;  // [D, tail_frames]: the passage's last latents, the next target's prefix and decode context
    int64_t tail_frames = 0;
    for (const std::unique_ptr<ChainStream>& owned : streams) {
        ChainStream& stream = *owned;
        Json resolved;
        const DeliveryTensors delivery = engine.delivery_tensors(stream.front().voice, stream.front().delivery, &resolved);
        if (!stream.continues()) {  // a new voice or language, or a requested pause: nothing carries over
            tail.clear();
            tail_frames = 0;
        }
        std::size_t start = 0;
        while (true) {
            Target target = next_target(engine, stream, start, spoken == 0, delivery, options.speed);
            const PlanChunk chunk = stream.target_chunk(start, target.end);
            const std::optional<uint64_t> seed = options.seed ? std::optional<uint64_t>(*options.seed + static_cast<uint64_t>(chunk.index)) : std::nullopt;
            send(StreamEvent{StreamEvent::kChunkStarted, &chunk, 0, nullptr, chunk.to_json()});

            const uint64_t draw_seed = seed ? *seed : (static_cast<uint64_t>(entropy()) << 32) ^ entropy();
            // One draw of (2 + latent_dim) x frames per target: the pitch predictor's two rows (unscaled), then the flow's (scaled by
            // the temperature).
            NoiseFn noise = [&](int64_t frames, float* pitch, float* flow) {
                const std::size_t T = static_cast<std::size_t>(frames);
                std::vector<float> all((2 + D) * T);
                if (chunk_noise) {
                    chunk_noise(chunk.index, seed.value_or(0), seed.has_value(), frames, all.data());
                } else {
                    gaussian_noise(draw_seed, static_cast<int64_t>(all.size()), 1.0f, all.data());
                    for (std::size_t i = 2 * T; i < all.size(); ++i) all[i] *= static_cast<float>(options.temperature);
                }
                std::copy_n(all.begin(), 2 * T, pitch);
                std::copy_n(all.begin() + static_cast<std::ptrdiff_t>(2 * T), D * T, flow);
            };
            SentenceRequest request;
            request.voice = chunk.voice;
            request.language = chunk.language;
            request.delivery = delivery;
            request.prefix = &tail;
            request.prefix_frames = tail_frames;
            request.steps = settings.steps;
            request.sampler = settings.sampler;
            request.speed = options.speed;
            request.noise = noise;
            request.durations = &target.durations;
            SentenceResult result = engine.synthesize_sentence(target.sentence, request);
            DecodeRequest decode;
            decode.voice = chunk.voice;
            decode.language = chunk.language;
            decode.left = &tail;
            decode.left_frames = tail_frames;
            std::vector<float> audio = engine.decode(result.latents, result.frames, decode);
            // pause_ms of silence where the voice, language or delivery changes (a new chain), as streaming.py adds it.
            if (!previous_chain.empty() && previous_chain != chunk.chain_id && options.pause_ms > 0.0) {
                const std::size_t pause = static_cast<std::size_t>(static_cast<double>(engine.sample_rate()) * options.pause_ms / 1000.0);
                audio.insert(audio.begin(), pause, 0.0f);
            }
            previous_chain = chunk.chain_id;
            const int64_t joined_frames = tail_frames + result.frames, kept = std::min(joined_frames, keep);
            std::vector<float> next_tail(D * static_cast<std::size_t>(kept));
            for (std::size_t d = 0; d < D; ++d) {
                for (int64_t t = 0; t < kept; ++t) {
                    const int64_t source = joined_frames - kept + t;
                    next_tail[d * static_cast<std::size_t>(kept) + static_cast<std::size_t>(t)] =
                        source < tail_frames ? tail[d * static_cast<std::size_t>(tail_frames) + static_cast<std::size_t>(source)]
                                             : result.latents[d * static_cast<std::size_t>(result.frames) + static_cast<std::size_t>(source - tail_frames)];
                }
            }
            tail = std::move(next_tail);
            tail_frames = kept;
            ++spoken;
            if (!first_audio_ms) first_audio_ms = elapsed_ms(started);
            Json metadata{Json::Object{}};
            metadata.set("chunk", chunk.to_json());
            for (const auto& [key, value] : result.metadata.items()) metadata.set(key, value);
            metadata.set("seed", seed ? Json(static_cast<int64_t>(*seed)) : Json());
            metadata.set("delivery", resolved);
            metadata.set("first_audio_ms", Json(round1(*first_audio_ms)));
            chunk_metadata.push(metadata);
            send(StreamEvent{StreamEvent::kAudio, &chunk, 0, &audio, metadata});
            if (stream.at_end(target.end)) break;
            start = target.end;
        }
    }
    Json summary{Json::Object{}};
    summary.set("chunks", Json(spoken));
    summary.set("first_audio_ms", first_audio_ms ? Json(round1(*first_audio_ms)) : Json());
    summary.set("elapsed_ms", Json(round1(elapsed_ms(started))));
    send(StreamEvent{StreamEvent::kDone, nullptr, 0, nullptr, summary});
    summary.set("chunks", chunk_metadata);
    return summary;
}

}  // namespace scyllasband
