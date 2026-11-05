#include "scyllasband_speech_plan.h"

#include "scyllasband_text.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

namespace scyllasband {

Json PlanChunk::to_json() const {
    Json out{Json::Object{}};
    out.set("chunk_id", Json(chunk_id));
    out.set("index", Json(index));
    out.set("record_index", Json(record_index));
    out.set("chain_id", Json(chain_id));
    out.set("voice", Json(voice));
    out.set("language", Json(language));
    const bool unset = std::none_of(delivery.controls.begin(), delivery.controls.end(), [](const auto& v) { return v.has_value(); });
    out.set("delivery", unset ? Json() : delivery.to_json());   // as the Python plan: null when nothing is set
    out.set("text", Json(text));
    out.set("source_text", Json(source_text));
    out.set("paragraph_end", Json(paragraph_end));
    return out;
}

Json Plan::to_json() const {
    Json out{Json::Object{}};
    out.set("source_kind", Json("text"));
    Json::Array items;
    std::vector<std::string> chains;
    for (const PlanChunk& chunk : chunks) {
        items.push_back(chunk.to_json());
        if (std::find(chains.begin(), chains.end(), chunk.chain_id) == chains.end()) chains.push_back(chunk.chain_id);
    }
    out.set("chunks", Json(std::move(items)));
    out.set("chains", Json(chains.size()));
    return out;
}

Plan plan_text(Engine& engine, const SynthesisOptions& options) {
    std::string voice = unicode::encode(unicode::strip(unicode::decode(options.voice)));
    if (voice.empty()) throw std::invalid_argument("Record 1 has no voice");
    const std::string language = engine.resolve_language(voice, options.language);
    const Delivery delivery = Delivery::parse(options.has_delivery ? options.delivery_spec.c_str() : nullptr);
    char chain_id[32];
    std::snprintf(chain_id, sizeof(chain_id), "chain_%03d", 1);
    Plan plan;
    std::vector<Text> paragraphs = split_paragraphs(unicode::decode(options.text));
    for (std::size_t p = 0; p < paragraphs.size(); ++p) {
        std::string paragraph = unicode::encode(paragraphs[p]);
        Text spoken = options.normalize ? unicode::decode(normalize_spoken_text(paragraph, language)) : paragraphs[p];
        std::vector<Text> sentences = split_sentences(spoken);
        for (std::size_t s = 0; s < sentences.size(); ++s) {
            PlanChunk chunk;
            chunk.index = static_cast<int>(plan.chunks.size());
            char chunk_id[32];
            std::snprintf(chunk_id, sizeof(chunk_id), "chunk_%04d", chunk.index + 1);
            chunk.chunk_id = chunk_id;
            chunk.chain_id = chain_id;
            chunk.voice = voice;
            chunk.language = language;
            chunk.delivery = delivery;
            chunk.text = unicode::encode(sentences[s]);
            chunk.source_text = unicode::encode(unicode::strip(paragraphs[p]));
            chunk.paragraph_end = s == sentences.size() - 1 && p < paragraphs.size() - 1;
            plan.chunks.push_back(std::move(chunk));
        }
    }
    return plan;
}

}  // namespace scyllasband
