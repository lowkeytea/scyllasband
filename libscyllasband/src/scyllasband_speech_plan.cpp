#include "scyllasband_speech_plan.h"

#include "scyllasband_text.h"

#include <algorithm>
#include <cstdio>
#include <optional>
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
    out.set("source_kind", Json(source_kind));
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

namespace {

// str(record.get(key) or ""): strings as they are, null and false as "", anything else as its JSON text.
std::string record_text(const Json& record, const char* key) {
    const Json& value = record.get(key);
    if (value.is_string()) return value.str();
    if (value.is_null() || (value.type() == Json::Type::boolean && !value.boolean())) return "";
    return value.dump();
}

// planner.normalize_request: the record's delivery, from its fields or its delivery (not both); no emotion.
Delivery record_delivery(const Json& record) {
    if (!record.get("emotion").is_null()) {
        throw std::invalid_argument("This model has no emotion choice: the emotion comes from the text. Use intensity, pace and pitch");
    }
    Json value = record.get("delivery");
    Json fields{Json::Object{}};
    bool any = false;
    for (const char* key : kDeliveryControls) {
        if (!record.get(key).is_null()) {
            fields.set(key, record.get(key));
            any = true;
        }
    }
    if (any) {
        if (!value.is_null()) throw std::invalid_argument("Give intensity / pace / pitch either as fields or as a delivery, not both");
        value = fields;
    }
    return Delivery::from_json(value);
}

}  // namespace

Plan plan_records(Engine& engine, const Json& records, bool normalize, const std::string& source_kind) {
    if (!records.is_array()) throw std::invalid_argument("Records must be a JSON array of {text, voice, language, delivery} objects");
    Plan plan;
    plan.source_kind = source_kind;
    std::optional<std::string> chain_key;
    int chain_serial = 0;
    for (std::size_t record_index = 0; record_index < records.size(); ++record_index) {
        const Json& record = records[record_index];
        if (!record.is_object()) throw std::invalid_argument("Record " + std::to_string(record_index + 1) + " must be an object");
        const std::string voice = unicode::encode(unicode::strip(unicode::decode(record_text(record, "voice"))));
        if (voice.empty()) throw std::invalid_argument("Record " + std::to_string(record_index + 1) + " has no voice");
        const std::string language = engine.resolve_language(voice, record_text(record, "language"));
        const Delivery delivery = record_delivery(record);
        const std::string key = voice + "\n" + language + "\n" + delivery.spec();
        if (!chain_key || key != *chain_key) {
            chain_key = key;
            ++chain_serial;
        }
        char chain_id[32];
        std::snprintf(chain_id, sizeof(chain_id), "chain_%03d", chain_serial);
        std::vector<Text> paragraphs = split_paragraphs(unicode::decode(record_text(record, "text")));
        for (std::size_t p = 0; p < paragraphs.size(); ++p) {
            std::string paragraph = unicode::encode(paragraphs[p]);
            Text spoken = normalize ? unicode::decode(normalize_spoken_text(paragraph, language)) : paragraphs[p];
            std::vector<Text> sentences = split_sentences(spoken);
            for (std::size_t s = 0; s < sentences.size(); ++s) {
                PlanChunk chunk;
                chunk.index = static_cast<int>(plan.chunks.size());
                char chunk_id[32];
                std::snprintf(chunk_id, sizeof(chunk_id), "chunk_%04d", chunk.index + 1);
                chunk.chunk_id = chunk_id;
                chunk.record_index = static_cast<int>(record_index);
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
    }
    return plan;
}

Plan plan_text(Engine& engine, const SynthesisOptions& options) {
    Json record{Json::Object{}};
    record.set("text", Json(options.text));
    record.set("voice", Json(options.voice));
    record.set("language", Json(options.language));
    record.set("delivery", options.has_delivery ? Json(options.delivery_spec) : Json());
    Json records{Json::Array{}};
    records.push(std::move(record));
    return plan_records(engine, records, options.normalize);
}

}  // namespace scyllasband
