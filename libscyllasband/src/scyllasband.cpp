#include "scyllasband.h"

#include "scyllasband_backend.h"
#include "scyllasband_bundle.h"
#include "scyllasband_boundary_policy.h"
#include "scyllasband_error.h"
#include "scyllasband_runtime_internal.h"
#include "scyllasband_target_bucket_policy.h"
#include "scyllasband_text_normalizer.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

bool valid_backend(ScyllasBandBackend backend) {
    return backend == SCYLLASBAND_BACKEND_AUTO || backend == SCYLLASBAND_BACKEND_LITERT ||
           backend == SCYLLASBAND_BACKEND_COREML || backend == SCYLLASBAND_BACKEND_ONNX ||
           backend == SCYLLASBAND_BACKEND_COREAI;
}

bool valid_litert_accelerator(ScyllasBandLiteRtAccelerator accelerator) {
    return accelerator == SCYLLASBAND_LITERT_ACCELERATOR_AUTO ||
           accelerator == SCYLLASBAND_LITERT_ACCELERATOR_CPU ||
           accelerator == SCYLLASBAND_LITERT_ACCELERATOR_GPU ||
           accelerator == SCYLLASBAND_LITERT_ACCELERATOR_NPU;
}

bool valid_sampler(ScyllasBandSampler sampler) {
    return sampler == SCYLLASBAND_SAMPLER_EULER || sampler == SCYLLASBAND_SAMPLER_HEUN;
}

bool valid_duration_hierarchy_mode(ScyllasBandDurationHierarchyMode mode) {
    return mode == SCYLLASBAND_DURATION_HIERARCHY_DEFAULT ||
           mode == SCYLLASBAND_DURATION_HIERARCHY_P50 ||
           mode == SCYLLASBAND_DURATION_HIERARCHY_SAMPLED;
}

void reset_result(ScyllasBandSynthesisResult* result) {
    if (result == nullptr) {
        return;
    }
    result->samples = nullptr;
    result->sample_count = 0;
    result->sample_rate = 0;
    result->metadata_json = nullptr;
    result->latents = nullptr;
    result->latent_dim = 0;
    result->latent_frames = 0;
}

void reset_chunk_plan_result(ScyllasBandChunkPlanResult* result) {
    if (result == nullptr) {
        return;
    }
    result->metadata_json = nullptr;
}

void reset_duration_estimate_result(ScyllasBandDurationEstimateResult* result) {
    if (result == nullptr) {
        return;
    }
    result->predicted_latent_frames = 0;
    result->fixed_latent_frames = 0;
    result->metadata_json = nullptr;
}

char* duplicate_c_string_local(const std::string& value) {
    auto* out = static_cast<char*>(std::malloc(value.size() + 1));
    if (out == nullptr) {
        return nullptr;
    }
    std::memcpy(out, value.c_str(), value.size() + 1);
    return out;
}

std::string json_escape_local(const std::string& value) {
    std::ostringstream out;
    for (char ch : value) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << ch; break;
        }
    }
    return out.str();
}

ScyllasBandStatus validate_request(const ScyllasBandSynthesisRequest* request) {
    if (request == nullptr) {
        scyllasband_detail::set_error("scyllasband_runtime_synthesize: request is required");
        return SCYLLASBAND_STATUS_INVALID_ARGUMENT;
    }
    if (request->voice_id == nullptr || request->voice_id[0] == '\0') {
        scyllasband_detail::set_error("scyllasband_runtime_synthesize: voice_id is required");
        return SCYLLASBAND_STATUS_INVALID_ARGUMENT;
    }
    if (request->text == nullptr && request->explicit_phones == nullptr) {
        scyllasband_detail::set_error("scyllasband_runtime_synthesize: text or explicit_phones is required");
        return SCYLLASBAND_STATUS_INVALID_ARGUMENT;
    }
    if (request->steps <= 0 || !valid_sampler(request->sampler)) {
        scyllasband_detail::set_error("scyllasband_runtime_synthesize: invalid sampler or step count");
        return SCYLLASBAND_STATUS_INVALID_ARGUMENT;
    }
    if (!valid_duration_hierarchy_mode(request->duration_hierarchy_mode)) {
        scyllasband_detail::set_error(
            "scyllasband_runtime_synthesize: invalid duration hierarchy mode"
        );
        return SCYLLASBAND_STATUS_INVALID_ARGUMENT;
    }
    if (request->speed <= 0.0f) {
        scyllasband_detail::set_error("scyllasband_runtime_synthesize: speed must be positive");
        return SCYLLASBAND_STATUS_INVALID_ARGUMENT;
    }
    if (request->has_affect_guidance_scale != 0 &&
        (!std::isfinite(request->affect_guidance_scale) || request->affect_guidance_scale < 0.0f)) {
        scyllasband_detail::set_error("scyllasband_runtime_synthesize: affect_guidance_scale must be finite and non-negative");
        return SCYLLASBAND_STATUS_INVALID_ARGUMENT;
    }
    if (request->prefix_latent_dim < 0 || request->prefix_latent_frames < 0) {
        scyllasband_detail::set_error("scyllasband_runtime_synthesize: prefix latent dimensions must be non-negative");
        return SCYLLASBAND_STATUS_INVALID_ARGUMENT;
    }
    if ((request->prefix_latent_dim > 0 || request->prefix_latent_frames > 0) &&
        request->prefix_latents == nullptr) {
        scyllasband_detail::set_error("scyllasband_runtime_synthesize: prefix_latents pointer is required when prefix dimensions are set");
        return SCYLLASBAND_STATUS_INVALID_ARGUMENT;
    }
    return SCYLLASBAND_STATUS_OK;
}

struct NativeChunkRecord {
    std::string text;
    std::string boundary_before;
    std::string boundary_after;
    std::string split_reason;
    bool starts_sentence = false;
    bool ends_sentence = false;
    int paragraph_index = 0;
    int unit_index = 0;
    int piece_index = 0;
    int piece_count = 1;
    int preflight_predicted_latent_frames = 0;
    int preflight_fixed_latent_frames = 0;
    int latent_retry_predicted_frames = 0;
    int latent_retry_fixed_frames = 0;
    int g2p_retry_encoded_tokens = 0;
    int g2p_retry_fixed_tokens = 0;
    int phone_retry_phone_count = 0;
    int phone_retry_fixed_frames = 0;
};

struct OverlongLatentCounts {
    int predicted = 0;
    int fixed = 0;
    bool valid = false;
};

struct OverlongG2PCounts {
    int encoded = 0;
    int fixed = 0;
    bool valid = false;
};

struct OverlongPhoneCounts {
    int phone_count = 0;
    int fixed = 0;
    bool valid = false;
};

std::string trim_copy(const std::string& value) {
    std::size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) {
        ++start;
    }
    std::size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    return value.substr(start, end - start);
}

std::string collapse_ws(const std::string& value) {
    std::string out;
    bool in_space = false;
    for (char ch : value) {
        if (std::isspace(static_cast<unsigned char>(ch))) {
            if (!in_space && !out.empty()) {
                out.push_back(' ');
            }
            in_space = true;
        } else {
            out.push_back(ch);
            in_space = false;
        }
    }
    return trim_copy(out);
}

std::string normalize_spoken_long_form_text(
    const char* text,
    const char* language
) {
    return scyllasband_detail::normalize_spoken_text(
        text == nullptr ? std::string() : std::string(text),
        language == nullptr ? std::string("en") : std::string(language)
    );
}

std::vector<std::string> split_paragraphs(const std::string& text) {
    std::vector<std::string> paragraphs;
    std::string current;
    std::string line;
    std::istringstream stream(text);
    while (std::getline(stream, line)) {
        if (trim_copy(line).empty()) {
            std::string paragraph = collapse_ws(current);
            if (!paragraph.empty()) {
                paragraphs.push_back(paragraph);
            }
            current.clear();
            continue;
        }
        if (!current.empty()) {
            current.push_back(' ');
        }
        current += line;
    }
    std::string paragraph = collapse_ws(current);
    if (!paragraph.empty()) {
        paragraphs.push_back(paragraph);
    }
    if (paragraphs.empty()) {
        paragraph = collapse_ws(text);
        if (!paragraph.empty()) {
            paragraphs.push_back(paragraph);
        }
    }
    return paragraphs;
}

bool terminal_boundary_char(char ch) {
    return ch == '.' || ch == '?' || ch == '!';
}

bool strong_continuation_char(char ch) {
    return ch == ';' || ch == ':';
}

bool sentence_boundary_run_char(char ch) {
    return terminal_boundary_char(ch) || strong_continuation_char(ch);
}

bool ascii_alpha(char ch) {
    return std::isalpha(static_cast<unsigned char>(ch)) != 0;
}

bool decimal_dot_at(const std::string& value, std::size_t index) {
    return index > 0 && index + 1 < value.size() && value[index] == '.' &&
           std::isdigit(static_cast<unsigned char>(value[index - 1])) != 0 &&
           std::isdigit(static_cast<unsigned char>(value[index + 1])) != 0;
}

bool dotted_initialism_dot_at(const std::string& value, std::size_t index) {
    if (index == 0 || index >= value.size() || value[index] != '.') {
        return false;
    }
    std::size_t start = index - 1;
    while (start > 0 && (ascii_alpha(value[start - 1]) || value[start - 1] == '.')) {
        --start;
    }
    if (start > 0 && ascii_alpha(value[start - 1])) {
        return false;
    }
    std::size_t end = index + 1;
    while (end < value.size() && (ascii_alpha(value[end]) || value[end] == '.')) {
        ++end;
    }
    const std::string run = value.substr(start, end - start);
    if (run.size() < 4 || (run.size() % 2) != 0) {
        return false;
    }
    for (std::size_t pos = 0; pos < run.size(); pos += 2) {
        if (!ascii_alpha(run[pos]) || run[pos + 1] != '.') {
            return false;
        }
    }
    return true;
}

bool protected_sentence_dot_at(const std::string& value, std::size_t index) {
    return decimal_dot_at(value, index) || dotted_initialism_dot_at(value, index);
}

std::vector<std::string> sentence_units(const std::string& text) {
    std::vector<std::string> units;
    std::string current;
    for (std::size_t index = 0; index < text.size(); ++index) {
        const char ch = text[index];
        current.push_back(ch);
        const bool protected_dot = ch == '.' && protected_sentence_dot_at(text, index);
        if ((terminal_boundary_char(ch) && !protected_dot) || strong_continuation_char(ch)) {
            while (index + 1 < text.size()) {
                const char next = text[index + 1];
                const bool next_protected_dot = next == '.' && protected_sentence_dot_at(text, index + 1);
                if (sentence_boundary_run_char(next) && !next_protected_dot) {
                    current.push_back(next);
                    ++index;
                    continue;
                }
                if (next == '"' || next == '\'' || next == ')' || next == ']' || next == '}') {
                    current.push_back(next);
                    ++index;
                    continue;
                }
                break;
            }
            std::string unit = collapse_ws(current);
            if (!unit.empty()) {
                units.push_back(unit);
            }
            current.clear();
        }
    }
    std::string tail = collapse_ws(current);
    if (!tail.empty()) {
        units.push_back(tail);
    }
    if (units.empty()) {
        std::string unit = collapse_ws(text);
        if (!unit.empty()) {
            units.push_back(unit);
        }
    }
    return units;
}

std::vector<std::string> rebalance_short_text_pieces(
    const std::vector<std::string>& chunks,
    int max_chars,
    int min_chars
) {
    min_chars = std::max(0, std::min(min_chars, max_chars));
    if (min_chars <= 0 || chunks.size() <= 1) {
        return chunks;
    }
    std::vector<std::string> output;
    for (const std::string& chunk : chunks) {
        if (!chunk.empty()) {
            output.push_back(chunk);
        }
    }
    if (output.size() <= 1 || static_cast<int>(output.back().size()) >= min_chars) {
        return output;
    }
    auto split_local = [](const std::string& value) -> std::vector<std::string> {
        std::vector<std::string> words;
        std::istringstream stream(value);
        std::string word;
        while (stream >> word) {
            words.push_back(word);
        }
        return words;
    };
    auto join_local = [](const std::vector<std::string>& words) -> std::string {
        std::string out;
        for (const std::string& word : words) {
            if (!out.empty()) {
                out.push_back(' ');
            }
            out += word;
        }
        return out;
    };
    auto joined_len = [](const std::vector<std::string>& words) -> int {
        int total = 0;
        for (std::size_t index = 0; index < words.size(); ++index) {
            total += static_cast<int>(words[index].size());
            if (index > 0) {
                ++total;
            }
        }
        return total;
    };
    std::vector<std::string> previous_words = split_local(output[output.size() - 2]);
    std::vector<std::string> last_words = split_local(output.back());
    while (joined_len(last_words) < min_chars && previous_words.size() > 1) {
        last_words.insert(last_words.begin(), previous_words.back());
        previous_words.pop_back();
    }
    if (!previous_words.empty() && joined_len(last_words) >= min_chars) {
        output[output.size() - 2] = join_local(previous_words);
        output.back() = join_local(last_words);
        return output;
    }
    output[output.size() - 2] = trim_copy(output[output.size() - 2] + " " + output.back());
    output.pop_back();
    return output;
}

bool natural_chunk_break_token(std::string value) {
    value = trim_copy(value);
    while (!value.empty()) {
        const char ch = value.back();
        if (ch == '"' || ch == '\'' || ch == ')' || ch == ']' || ch == '}') {
            value.pop_back();
            value = trim_copy(value);
        } else {
            break;
        }
    }
    if (!value.empty() && value.back() == ',') {
        return true;
    }
    static const std::vector<std::string> suffixes = {"--", u8"—", u8"–"};
    for (const std::string& suffix : suffixes) {
        if (value.size() >= suffix.size() &&
            value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0) {
            return true;
        }
    }
    return false;
}

std::vector<std::string> split_oversized_unit(const std::string& text, int max_chars, int min_chars) {
    std::string value = collapse_ws(text);
    if (value.empty()) {
        return {};
    }
    if (max_chars <= 0 || static_cast<int>(value.size()) <= max_chars) {
        return {value};
    }
    std::vector<std::string> words;
    std::istringstream stream(value);
    std::string word;
    while (stream >> word) {
        while (max_chars > 0 && static_cast<int>(word.size()) > max_chars) {
            words.push_back(word.substr(0, static_cast<std::size_t>(max_chars)));
            word.erase(0, static_cast<std::size_t>(max_chars));
        }
        if (!word.empty()) {
            words.push_back(word);
        }
    }
    auto joined_range = [&words](std::size_t start, std::size_t end) -> std::string {
        std::string out;
        for (std::size_t index = start; index < end; ++index) {
            if (!out.empty()) {
                out.push_back(' ');
            }
            out += words[index];
        }
        return out;
    };
    std::vector<std::string> chunks;
    std::size_t start = 0;
    while (start < words.size()) {
        std::size_t hard_end = start;
        while (hard_end < words.size()) {
            const std::string candidate = joined_range(start, hard_end + 1);
            if (static_cast<int>(candidate.size()) > max_chars) {
                break;
            }
            ++hard_end;
        }
        if (hard_end >= words.size()) {
            chunks.push_back(joined_range(start, words.size()));
            break;
        }
        if (hard_end <= start) {
            hard_end = start + 1;
        }
        std::size_t split_at = hard_end;
        const int preferred_min = std::max(1, min_chars);
        for (std::size_t candidate_end = start + 1; candidate_end <= hard_end; ++candidate_end) {
            if (!natural_chunk_break_token(words[candidate_end - 1])) {
                continue;
            }
            const std::string left = joined_range(start, candidate_end);
            const std::string right = joined_range(candidate_end, words.size());
            if (static_cast<int>(left.size()) >= preferred_min &&
                static_cast<int>(right.size()) >= preferred_min) {
                split_at = candidate_end;
            }
        }
        chunks.push_back(joined_range(start, split_at));
        start = split_at;
    }
    return rebalance_short_text_pieces(chunks, max_chars, min_chars);
}


std::vector<std::string> split_words(const std::string& text) {
    std::vector<std::string> words;
    std::istringstream stream(text);
    std::string word;
    while (stream >> word) {
        words.push_back(word);
    }
    return words;
}

std::string join_words(const std::vector<std::string>& words, std::size_t start, std::size_t end) {
    std::string out;
    for (std::size_t index = start; index < end; ++index) {
        if (!out.empty()) {
            out.push_back(' ');
        }
        out += words[index];
    }
    return out;
}

bool retry_pieces_are_usable(const std::vector<std::string>& pieces, int max_chars, int min_chars) {
    if (pieces.size() <= 1) {
        return false;
    }
    max_chars = std::max(1, max_chars);
    const int requested_min = std::max(0, std::min(min_chars, max_chars));
    for (const std::string& piece : pieces) {
        const int length = static_cast<int>(trim_copy(piece).size());
        if (length <= 0 || length > max_chars) {
            return false;
        }
        if (requested_min > 0 && length < requested_min) {
            return false;
        }
    }
    return true;
}

bool retry_pieces_fit_max(const std::vector<std::string>& pieces, int max_chars) {
    return retry_pieces_are_usable(pieces, max_chars, 0);
}

std::pair<int, int> split_path_balance_score(const std::vector<std::string>& pieces) {
    if (pieces.empty()) {
        return {0, 0};
    }
    int min_length = std::numeric_limits<int>::max();
    int max_length = 0;
    for (const std::string& piece : pieces) {
        const int length = static_cast<int>(trim_copy(piece).size());
        min_length = std::min(min_length, length);
        max_length = std::max(max_length, length);
    }
    return {max_length - min_length, max_length};
}


std::vector<std::string> split_words_min_max(const std::string& text, int max_chars, int min_chars) {
    const std::vector<std::string> words = split_words(text);
    if (words.size() <= 1) {
        return {};
    }
    max_chars = std::max(1, max_chars);
    min_chars = std::max(0, std::min(min_chars, max_chars));
    const std::size_t n = words.size();
    std::vector<int> prefix(n + 1, 0);
    for (std::size_t index = 0; index < n; ++index) {
        prefix[index + 1] = prefix[index] + static_cast<int>(words[index].size());
    }
    auto segment_len = [&](std::size_t start, std::size_t end) -> int {
        return prefix[end] - prefix[start] + static_cast<int>(end > start ? end - start - 1 : 0);
    };
    std::vector<std::vector<std::string>> paths(n + 1);
    std::vector<bool> has_path(n + 1, false);
    has_path[0] = true;
    for (std::size_t start = 0; start < n; ++start) {
        if (!has_path[start]) {
            continue;
        }
        for (std::size_t end = start + 1; end <= n; ++end) {
            const int length = segment_len(start, end);
            if (length > max_chars) {
                break;
            }
            if (min_chars > 0 && length < min_chars) {
                continue;
            }
            std::vector<std::string> candidate = paths[start];
            candidate.push_back(join_words(words, start, end));
            if (!has_path[end] ||
                candidate.size() < paths[end].size() ||
                (candidate.size() == paths[end].size() &&
                 split_path_balance_score(candidate) < split_path_balance_score(paths[end]))) {
                paths[end] = std::move(candidate);
                has_path[end] = true;
            }
        }
    }
    if (!has_path[n] || paths[n].size() <= 1) {
        return {};
    }
    return paths[n];
}

std::vector<std::string> split_words_near_half(const std::string& text, int max_chars, int min_chars) {
    const std::vector<std::string> words = split_words(text);
    if (words.size() <= 1) {
        return {};
    }
    const std::size_t midpoint = std::max<std::size_t>(1, words.size() / 2);
    std::vector<std::string> pieces = {
        join_words(words, 0, midpoint),
        join_words(words, midpoint, words.size()),
    };
    return retry_pieces_are_usable(pieces, max_chars, min_chars) ? pieces : std::vector<std::string>{};
}

std::vector<std::string> split_units_min_max(const std::string& text, int max_chars, int min_chars) {
    const std::vector<std::string> units = sentence_units(text);
    if (units.size() <= 1) {
        return {};
    }
    max_chars = std::max(1, max_chars);
    std::vector<std::string> pieces;
    std::string current;
    for (const std::string& unit : units) {
        std::vector<std::string> candidates;
        if (static_cast<int>(unit.size()) <= max_chars) {
            candidates.push_back(unit);
        } else {
            candidates = split_oversized_unit(unit, max_chars, min_chars);
        }
        for (const std::string& candidate_piece : candidates) {
            if (candidate_piece.empty()) {
                continue;
            }
            if (current.empty()) {
                current = candidate_piece;
                continue;
            }
            const std::string candidate = current + " " + candidate_piece;
            if (static_cast<int>(candidate.size()) <= max_chars) {
                current = candidate;
            } else {
                pieces.push_back(current);
                current = candidate_piece;
            }
        }
    }
    if (!current.empty()) {
        pieces.push_back(current);
    }
    return retry_pieces_are_usable(pieces, max_chars, min_chars) ? pieces : std::vector<std::string>{};
}


int retry_split_max_chars(
    const std::string& text,
    int max_chars,
    int min_chars,
    int predicted_latent_frames,
    int fixed_latent_frames
) {
    int requested_max = std::max(1, max_chars);
    const int requested_min = std::max(0, std::min(min_chars, requested_max));
    if (predicted_latent_frames > 0 && fixed_latent_frames > 0 && predicted_latent_frames > fixed_latent_frames) {
        const int text_len = std::max(1, static_cast<int>(trim_copy(text).size()));
        const int scaled = static_cast<int>(
            static_cast<float>(text_len) * static_cast<float>(fixed_latent_frames) /
            static_cast<float>(predicted_latent_frames) * 0.92f
        );
        requested_max = std::min(requested_max, std::max(requested_min > 0 ? requested_min : 1, scaled));
    }
    return std::max(requested_min > 0 ? requested_min : 1, requested_max);
}

std::string boundary_before_for(const std::string& previous_after, bool paragraph_start);
std::string strip_closing_boundary(std::string value);

std::string retry_piece_boundary_after(const std::string& text) {
    const std::string value = strip_closing_boundary(text);
    if (!value.empty() && terminal_boundary_char(value.back())) {
        return "sentence_end";
    }
    if ((!value.empty() && strong_continuation_char(value.back())) ||
        natural_chunk_break_token(value)) {
        return "clause_continue";
    }
    return "chunk_continue";
}

void assign_retry_piece_boundaries(
    NativeChunkRecord& record,
    const NativeChunkRecord& source,
    bool first,
    bool last,
    std::string& previous_after
) {
    record.boundary_before = first
        ? source.boundary_before
        : boundary_before_for(previous_after, false);
    record.boundary_after = last
        ? source.boundary_after
        : retry_piece_boundary_after(record.text);
    record.starts_sentence = first
        ? source.starts_sentence
        : (record.boundary_before == "paragraph_start" || record.boundary_before == "sentence_start");
    record.ends_sentence =
        record.boundary_after == "sentence_end" || record.boundary_after == "paragraph_end";
    previous_after = record.boundary_after;
}

std::vector<NativeChunkRecord> split_chunk_for_latent_retry(
    const NativeChunkRecord& chunk,
    int max_chars,
    int min_chars,
    int predicted_latent_frames,
    int fixed_latent_frames
) {
    const std::string text = trim_copy(chunk.text);
    if (text.empty()) {
        return {};
    }
    const int retry_max_chars = retry_split_max_chars(
        text,
        max_chars,
        min_chars,
        predicted_latent_frames,
        fixed_latent_frames
    );
    std::vector<std::string> pieces = split_units_min_max(text, retry_max_chars, min_chars);
    if (pieces.empty() && min_chars > 0) {
        pieces = split_units_min_max(text, retry_max_chars, 0);
    }
    if (pieces.empty()) {
        pieces = split_words_min_max(text, retry_max_chars, min_chars);
    }
    if (pieces.empty()) {
        pieces = split_oversized_unit(text, retry_max_chars, min_chars);
    }
    if (!retry_pieces_fit_max(pieces, retry_max_chars)) {
        pieces = split_words_near_half(text, retry_max_chars, min_chars);
    }
    if (!retry_pieces_fit_max(pieces, retry_max_chars)) {
        return {};
    }
    std::vector<NativeChunkRecord> out;
    out.reserve(pieces.size());
    std::string previous_after;
    for (std::size_t piece_index = 0; piece_index < pieces.size(); ++piece_index) {
        const bool first = piece_index == 0;
        const bool last = piece_index + 1 == pieces.size();
        NativeChunkRecord record = chunk;
        record.text = pieces[piece_index];
        record.split_reason = last ? "latent_budget_tail" : "latent_budget";
        assign_retry_piece_boundaries(record, chunk, first, last, previous_after);
        record.latent_retry_predicted_frames = predicted_latent_frames;
        record.latent_retry_fixed_frames = fixed_latent_frames;
        out.push_back(std::move(record));
    }
    return out;
}

std::vector<NativeChunkRecord> split_chunk_for_g2p_retry(
    const NativeChunkRecord& chunk,
    int max_chars,
    int min_chars,
    int encoded_tokens,
    int fixed_tokens
) {
    const std::string text = trim_copy(chunk.text);
    if (text.empty()) {
        return {};
    }
    const int retry_max_chars = retry_split_max_chars(
        text,
        max_chars,
        min_chars,
        encoded_tokens,
        fixed_tokens
    );
    std::vector<std::string> pieces = split_units_min_max(text, retry_max_chars, min_chars);
    if (pieces.empty() && min_chars > 0) {
        pieces = split_units_min_max(text, retry_max_chars, 0);
    }
    if (pieces.empty()) {
        pieces = split_words_min_max(text, retry_max_chars, min_chars);
    }
    if (pieces.empty()) {
        pieces = split_oversized_unit(text, retry_max_chars, min_chars);
    }
    if (!retry_pieces_fit_max(pieces, retry_max_chars)) {
        pieces = split_words_near_half(text, retry_max_chars, min_chars);
    }
    if (!retry_pieces_fit_max(pieces, retry_max_chars)) {
        return {};
    }
    std::vector<NativeChunkRecord> out;
    out.reserve(pieces.size());
    std::string previous_after;
    for (std::size_t piece_index = 0; piece_index < pieces.size(); ++piece_index) {
        const bool first = piece_index == 0;
        const bool last = piece_index + 1 == pieces.size();
        NativeChunkRecord record = chunk;
        record.text = pieces[piece_index];
        record.split_reason = last ? "g2p_text_budget_tail" : "g2p_text_budget";
        assign_retry_piece_boundaries(record, chunk, first, last, previous_after);
        record.g2p_retry_encoded_tokens = encoded_tokens;
        record.g2p_retry_fixed_tokens = fixed_tokens;
        out.push_back(std::move(record));
    }
    return out;
}

std::vector<NativeChunkRecord> split_chunk_for_phone_retry(
    const NativeChunkRecord& chunk,
    int max_chars,
    int min_chars,
    int phone_count,
    int fixed_frames
) {
    const std::string text = trim_copy(chunk.text);
    if (text.empty()) {
        return {};
    }
    const int retry_max_chars = retry_split_max_chars(
        text,
        max_chars,
        min_chars,
        phone_count,
        fixed_frames
    );
    std::vector<std::string> pieces = split_units_min_max(text, retry_max_chars, min_chars);
    if (pieces.empty() && min_chars > 0) {
        pieces = split_units_min_max(text, retry_max_chars, 0);
    }
    if (pieces.empty()) {
        pieces = split_words_min_max(text, retry_max_chars, min_chars);
    }
    if (pieces.empty()) {
        pieces = split_oversized_unit(text, retry_max_chars, min_chars);
    }
    if (!retry_pieces_fit_max(pieces, retry_max_chars)) {
        pieces = split_words_near_half(text, retry_max_chars, min_chars);
    }
    if (!retry_pieces_fit_max(pieces, retry_max_chars)) {
        return {};
    }
    std::vector<NativeChunkRecord> out;
    out.reserve(pieces.size());
    std::string previous_after;
    for (std::size_t piece_index = 0; piece_index < pieces.size(); ++piece_index) {
        const bool first = piece_index == 0;
        const bool last = piece_index + 1 == pieces.size();
        NativeChunkRecord record = chunk;
        record.text = pieces[piece_index];
        record.split_reason = last ? "phone_frame_budget_tail" : "phone_frame_budget";
        assign_retry_piece_boundaries(record, chunk, first, last, previous_after);
        record.phone_retry_phone_count = phone_count;
        record.phone_retry_fixed_frames = fixed_frames;
        out.push_back(std::move(record));
    }
    return out;
}

int preflight_split_overlong_chunks(
    ScyllasBandRuntime* runtime,
    std::vector<NativeChunkRecord>& chunks,
    const ScyllasBandSynthesisRequest& base_request,
    int max_chars,
    int min_chars
) {
    if (runtime == nullptr || !runtime->engine) {
        return 0;
    }
    std::size_t index = 0;
    int split_count = 0;
    int split_budget = std::max<int>(1, static_cast<int>(chunks.size()) * 16);
    while (index < chunks.size()) {
        NativeChunkRecord& chunk = chunks[index];
        ScyllasBandSynthesisRequest chunk_request = base_request;
        chunk_request.text = chunk.text.c_str();
        chunk_request.explicit_phones = nullptr;
        chunk_request.context_before = nullptr;
        chunk_request.context_after = nullptr;
        chunk_request.prefix_latents = nullptr;
        chunk_request.prefix_latent_dim = 0;
        chunk_request.prefix_latent_frames = 0;
        chunk_request.boundary_before = chunk.boundary_before.c_str();
        chunk_request.boundary_after = chunk.boundary_after.c_str();

        scyllasband_detail::ScyllasBandDurationEstimate estimate;
        const ScyllasBandStatus status = runtime->engine->estimate_latent_frames(chunk_request, &estimate);
        if (status != SCYLLASBAND_STATUS_OK) {
            scyllasband_detail::clear_error();
            ++index;
            continue;
        }
        chunk.preflight_predicted_latent_frames = estimate.predicted_latent_frames;
        chunk.preflight_fixed_latent_frames = estimate.fixed_latent_frames;
        if (!scyllasband_detail::target_bucket_requires_replan(
                estimate.predicted_latent_frames,
                estimate.fixed_latent_frames
            ) || split_budget <= 0) {
            ++index;
            continue;
        }
        std::vector<NativeChunkRecord> replacements = split_chunk_for_latent_retry(
            chunk,
            max_chars,
            min_chars,
            estimate.predicted_latent_frames,
            estimate.fixed_latent_frames
        );
        if (replacements.empty()) {
            ++index;
            continue;
        }
        --split_budget;
        ++split_count;
        chunks.erase(chunks.begin() + static_cast<std::ptrdiff_t>(index));
        chunks.insert(
            chunks.begin() + static_cast<std::ptrdiff_t>(index),
            std::make_move_iterator(replacements.begin()),
            std::make_move_iterator(replacements.end())
        );
    }
    return split_count;
}

OverlongLatentCounts parse_overlong_latent_error(const std::string& message) {
    constexpr const char* kPrefix = "Predicted ";
    constexpr const char* kMiddle = " latent frames, but this bundle supports at most ";
    const std::size_t prefix_pos = message.find(kPrefix);
    if (prefix_pos == std::string::npos) {
        return {};
    }
    const std::size_t predicted_start = prefix_pos + std::strlen(kPrefix);
    const std::size_t middle_pos = message.find(kMiddle, predicted_start);
    if (middle_pos == std::string::npos) {
        return {};
    }
    const std::string predicted_text = message.substr(predicted_start, middle_pos - predicted_start);
    const std::size_t fixed_start = middle_pos + std::strlen(kMiddle);
    const std::string fixed_text = message.substr(fixed_start);
    char* predicted_end = nullptr;
    char* fixed_end = nullptr;
    const long predicted = std::strtol(predicted_text.c_str(), &predicted_end, 10);
    const long fixed = std::strtol(fixed_text.c_str(), &fixed_end, 10);
    if (predicted_end == predicted_text.c_str() || fixed_end == fixed_text.c_str() || predicted <= 0 || fixed <= 0) {
        return {};
    }
    if (predicted <= fixed || predicted > INT_MAX || fixed > INT_MAX) {
        return {};
    }
    return {static_cast<int>(predicted), static_cast<int>(fixed), true};
}

OverlongPhoneCounts parse_overlong_phone_error(const std::string& message) {
    constexpr const char* kPrefix = "Explicit phone sequence has ";
    constexpr const char* kMiddle = " phones, but this bundle supports at most ";
    const std::size_t prefix_pos = message.find(kPrefix);
    if (prefix_pos == std::string::npos) {
        return {};
    }
    const std::size_t phone_start = prefix_pos + std::strlen(kPrefix);
    const std::size_t middle_pos = message.find(kMiddle, phone_start);
    if (middle_pos == std::string::npos) {
        return {};
    }
    const std::string phone_text = message.substr(phone_start, middle_pos - phone_start);
    const std::size_t fixed_start = middle_pos + std::strlen(kMiddle);
    const std::string fixed_text = message.substr(fixed_start);
    char* phone_end = nullptr;
    char* fixed_end = nullptr;
    const long phone_count = std::strtol(phone_text.c_str(), &phone_end, 10);
    const long fixed = std::strtol(fixed_text.c_str(), &fixed_end, 10);
    if (phone_end == phone_text.c_str() || fixed_end == fixed_text.c_str() || phone_count <= 0 || fixed <= 0) {
        return {};
    }
    if (phone_count <= fixed || phone_count > INT_MAX || fixed > INT_MAX) {
        return {};
    }
    return {static_cast<int>(phone_count), static_cast<int>(fixed), true};
}

OverlongG2PCounts parse_overlong_g2p_error(const std::string& message) {
    constexpr const char* kPrefix = "G2P input encodes to ";
    constexpr const char* kMiddle = " tokens, but this bundle supports at most ";
    const std::size_t prefix_pos = message.find(kPrefix);
    if (prefix_pos == std::string::npos) {
        return {};
    }
    const std::size_t encoded_start = prefix_pos + std::strlen(kPrefix);
    const std::size_t middle_pos = message.find(kMiddle, encoded_start);
    if (middle_pos == std::string::npos) {
        return {};
    }
    const std::string encoded_text = message.substr(encoded_start, middle_pos - encoded_start);
    const std::size_t fixed_start = middle_pos + std::strlen(kMiddle);
    const std::string fixed_text = message.substr(fixed_start);
    char* encoded_end = nullptr;
    char* fixed_end = nullptr;
    const long encoded = std::strtol(encoded_text.c_str(), &encoded_end, 10);
    const long fixed = std::strtol(fixed_text.c_str(), &fixed_end, 10);
    if (encoded_end == encoded_text.c_str() || fixed_end == fixed_text.c_str() || encoded <= 0 || fixed <= 0) {
        return {};
    }
    if (encoded <= fixed || encoded > INT_MAX || fixed > INT_MAX) {
        return {};
    }
    return {static_cast<int>(encoded), static_cast<int>(fixed), true};
}

std::string boundary_before_for(const std::string& previous_after, bool paragraph_start) {
    if (previous_after.empty() || paragraph_start || previous_after == "paragraph_end") {
        return "paragraph_start";
    }
    if (previous_after == "sentence_end") {
        return "sentence_start";
    }
    if (previous_after == "chunk_continue") {
        return "chunk_continue";
    }
    return "clause_continue";
}

std::string strip_closing_boundary(std::string value) {
    value = trim_copy(value);
    while (!value.empty()) {
        const char ch = value.back();
        if (ch == '"' || ch == '\'' || ch == ')' || ch == ']' || ch == '}') {
            value.pop_back();
            value = trim_copy(value);
        } else {
            break;
        }
    }
    return value;
}

std::string boundary_after_for(const std::string& text, bool final_piece, bool paragraph_end) {
    if (!final_piece) {
        if (natural_chunk_break_token(text)) {
            return "clause_continue";
        }
        return "chunk_continue";
    }
    std::string value = strip_closing_boundary(text);
    if (!value.empty() && terminal_boundary_char(value.back())) {
        return paragraph_end ? "paragraph_end" : "sentence_end";
    }
    if (!value.empty() && strong_continuation_char(value.back())) {
        return "clause_continue";
    }
    return paragraph_end ? "paragraph_end" : "clause_continue";
}

bool can_merge_chunk_records(
    const NativeChunkRecord& previous,
    const NativeChunkRecord& current,
    int max_chars
) {
    if (previous.paragraph_index != current.paragraph_index) {
        return false;
    }
    if (previous.boundary_after == "chunk_continue" || current.boundary_before == "chunk_continue") {
        return false;
    }
    const std::string candidate = previous.text + " " + current.text;
    return static_cast<int>(candidate.size()) <= max_chars;
}

std::vector<NativeChunkRecord> merge_adjacent_chunk_records(
    const std::vector<NativeChunkRecord>& records,
    int max_chars
) {
    std::vector<NativeChunkRecord> merged;
    for (const NativeChunkRecord& record : records) {
        if (!merged.empty() && can_merge_chunk_records(merged.back(), record, max_chars)) {
            NativeChunkRecord& previous = merged.back();
            previous.text = previous.text + " " + record.text;
            previous.boundary_after = record.boundary_after;
            previous.ends_sentence = record.ends_sentence;
            previous.split_reason = "merged";
            continue;
        }
        merged.push_back(record);
    }
    return merged;
}

std::vector<NativeChunkRecord> merge_short_chunk_records(
    const std::vector<NativeChunkRecord>& records,
    int max_chars,
    int min_chars
) {
    if (min_chars <= 0 || records.size() <= 1) {
        return records;
    }
    std::vector<NativeChunkRecord> output = records;
    std::size_t index = 0;
    while (index < output.size()) {
        const int text_len = static_cast<int>(output[index].text.size());
        if (text_len >= min_chars) {
            ++index;
            continue;
        }
        bool merged = false;
        if (index > 0 && output[index - 1].paragraph_index == output[index].paragraph_index) {
            const std::string candidate = trim_copy(output[index - 1].text + " " + output[index].text);
            if (static_cast<int>(candidate.size()) <= max_chars || index + 1 == output.size()) {
                NativeChunkRecord& previous = output[index - 1];
                previous.text = candidate;
                previous.boundary_after = output[index].boundary_after;
                previous.ends_sentence = output[index].ends_sentence;
                previous.split_reason = "min_chunk_merged";
                output.erase(output.begin() + static_cast<std::ptrdiff_t>(index));
                merged = true;
            }
        }
        if (merged) {
            continue;
        }
        if (index + 1 < output.size() && output[index + 1].paragraph_index == output[index].paragraph_index) {
            const std::string candidate = trim_copy(output[index].text + " " + output[index + 1].text);
            if (static_cast<int>(candidate.size()) <= max_chars) {
                NativeChunkRecord& current = output[index];
                current.text = candidate;
                current.boundary_after = output[index + 1].boundary_after;
                current.ends_sentence = output[index + 1].ends_sentence;
                current.split_reason = "min_chunk_merged";
                output.erase(output.begin() + static_cast<std::ptrdiff_t>(index + 1));
                continue;
            }
        }
        ++index;
    }
    return output;
}

std::vector<NativeChunkRecord> plan_long_form_chunks(const std::string& text, int max_chars, int min_chars) {
    max_chars = max_chars > 0 ? max_chars : 220;
    min_chars = std::max(0, std::min(min_chars, max_chars));
    std::vector<NativeChunkRecord> records;
    std::string previous_after;
    const std::vector<std::string> paragraphs = split_paragraphs(text);
    for (std::size_t paragraph_index = 0; paragraph_index < paragraphs.size(); ++paragraph_index) {
        const std::vector<std::string> units = sentence_units(paragraphs[paragraph_index]);
        for (std::size_t unit_index = 0; unit_index < units.size(); ++unit_index) {
            const std::vector<std::string> pieces = split_oversized_unit(units[unit_index], max_chars, min_chars);
            for (std::size_t piece_index = 0; piece_index < pieces.size(); ++piece_index) {
                const bool first_piece = piece_index == 0;
                const bool last_piece = piece_index + 1 == pieces.size();
                const bool paragraph_start = unit_index == 0 && first_piece;
                const bool paragraph_end = unit_index + 1 == units.size();
                NativeChunkRecord record;
                record.text = pieces[piece_index];
                record.paragraph_index = static_cast<int>(paragraph_index);
                record.unit_index = static_cast<int>(unit_index);
                record.piece_index = static_cast<int>(piece_index);
                record.piece_count = static_cast<int>(pieces.size());
                record.boundary_before = boundary_before_for(previous_after, paragraph_start);
                record.boundary_after = boundary_after_for(record.text, last_piece, paragraph_end);
                record.split_reason = pieces.size() == 1 ? "unit" : (last_piece ? "unit_tail" : "budget");
                record.starts_sentence = record.boundary_before == "paragraph_start" || record.boundary_before == "sentence_start";
                record.ends_sentence = record.boundary_after == "sentence_end" || record.boundary_after == "paragraph_end";
                previous_after = record.boundary_after;
                records.push_back(std::move(record));
            }
        }
    }
    records = merge_adjacent_chunk_records(records, max_chars);
    return merge_short_chunk_records(records, max_chars, min_chars);
}


int pause_after_ms(const NativeChunkRecord& chunk, bool is_last, int pause_ms, int continuation_pause_ms) {
    return scyllasband_detail::host_pause_after_ms(
        chunk.boundary_after,
        is_last,
        pause_ms,
        continuation_pause_ms
    );
}

std::string context_before_for(const std::vector<NativeChunkRecord>& chunks, std::size_t index) {
    if (index == 0) {
        return {};
    }
    const std::string& text = chunks[index - 1].text;
    constexpr std::size_t kContextChars = 220;
    return text.size() <= kContextChars ? text : text.substr(text.size() - kContextChars);
}

std::string context_after_for(const std::vector<NativeChunkRecord>& chunks, std::size_t index) {
    if (index + 1 >= chunks.size()) {
        return {};
    }
    const std::string& text = chunks[index + 1].text;
    constexpr std::size_t kContextChars = 220;
    return text.substr(0, std::min(kContextChars, text.size()));
}

const char* backend_name_for_plan(ScyllasBandBackend backend) {
    switch (backend) {
        case SCYLLASBAND_BACKEND_LITERT:
            return "litert";
        case SCYLLASBAND_BACKEND_COREML:
            return "coreml";
        case SCYLLASBAND_BACKEND_ONNX:
            return "onnx";
        case SCYLLASBAND_BACKEND_COREAI:
            return "coreai";
        case SCYLLASBAND_BACKEND_AUTO:
#if defined(SCYLLASBAND_WITH_COREAI)
            return "coreai";
#else
            return "onnx";
#endif
    }
    return "onnx";
}

void append_json_string(std::ostringstream& out, const std::string& value) {
    out << "\"" << json_escape_local(value) << "\"";
}

void append_json_optional_string(std::ostringstream& out, const char* value) {
    if (value == nullptr || value[0] == '\0') {
        out << "null";
        return;
    }
    append_json_string(out, value);
}

void append_json_optional_string(std::ostringstream& out, const std::string& value) {
    if (value.empty()) {
        out << "null";
        return;
    }
    append_json_string(out, value);
}

void append_json_nullable_int(std::ostringstream& out, int value) {
    if (value <= 0) {
        out << "null";
        return;
    }
    out << value;
}

std::string reference_key_for_plan(const ScyllasBandSynthesisRequest* request) {
    if (request == nullptr || request->voice_id == nullptr || request->voice_id[0] == '\0') {
        return {};
    }
    return request->voice_id;
}

std::string chunk_id_for_index(std::size_t index);

std::string chunk_plan_json(
    const std::vector<NativeChunkRecord>& chunks,
    int max_chars,
    int min_chars,
    const std::string& normalized_text = std::string(),
    const ScyllasBandSynthesisRequest* request = nullptr,
    ScyllasBandBackend backend = SCYLLASBAND_BACKEND_AUTO,
    bool can_preflight_ahead = false,
    bool preflight_chunks = false
) {
    int latent_frame_budget = 0;
    int phone_frame_budget = 0;
    int g2p_text_budget = 0;
    for (const NativeChunkRecord& chunk : chunks) {
        latent_frame_budget = std::max(latent_frame_budget, chunk.preflight_fixed_latent_frames);
        phone_frame_budget = std::max(phone_frame_budget, chunk.phone_retry_fixed_frames);
        g2p_text_budget = std::max(g2p_text_budget, chunk.g2p_retry_fixed_tokens);
    }

    std::ostringstream out;
    out << "{\"status\":\"ok\","
        << "\"mode\":\"native_chunk_plan\","
        << "\"version\":\"scyllasband.streaming.plan.v1\","
        << "\"source_kind\":\"text\","
        << "\"normalized_text\":";
    append_json_string(out, normalized_text);
    out << ",\"sample_rate\":null,"
        << "\"backend\":\"" << backend_name_for_plan(backend) << "\","
        << "\"fixed_shape_budget\":{"
        << "\"latent_frames\":" << latent_frame_budget << ","
        << "\"phone_frames\":" << phone_frame_budget << ","
        << "\"g2p_text_tokens\":" << g2p_text_budget
        << "},"
        << "\"records\":[{"
        << "\"record_id\":\"record-0000\","
        << "\"voice\":";
    append_json_optional_string(out, request == nullptr ? nullptr : request->voice_id);
    out << ",\"language\":";
    append_json_optional_string(out, request == nullptr ? nullptr : request->language);
    out << ",\"emotion\":";
    append_json_optional_string(out, request == nullptr ? nullptr : request->emotion);
    out << ",\"emotion_guidance\":";
    append_json_optional_string(out, request == nullptr ? nullptr : request->emotion_guidance);
    out << ",\"text\":";
    append_json_string(out, request != nullptr && request->text != nullptr ? request->text : normalized_text);
    out << ",\"normalized_text\":";
    append_json_string(out, normalized_text);
    out << ",\"start_char\":0,"
        << "\"end_char\":" << normalized_text.size()
        << "}],"
        << "\"scheduler_hints\":{"
        << "\"serial_within_chain\":true,"
        << "\"can_preflight_ahead\":" << (can_preflight_ahead ? "true" : "false") << ","
        << "\"parallelizable_chains\":" << (chunks.empty() ? "[]" : "[\"chain-0000\"]")
        << "},"
        << "\"chunk_max_chars\":" << max_chars << ","
        << "\"chunk_min_chars\":" << min_chars << ","
        << "\"preflight_chunks\":" << (preflight_chunks ? "true" : "false") << ","
        << "\"chunk_count\":" << chunks.size() << ","
        << "\"chunks\":[";
    const std::string reference_key = reference_key_for_plan(request);
    for (std::size_t index = 0; index < chunks.size(); ++index) {
        const NativeChunkRecord& chunk = chunks[index];
        if (index > 0) {
            out << ",";
        }
        const std::string context_before = context_before_for(chunks, index);
        const std::string context_after = context_after_for(chunks, index);
        out << "{"
            << "\"index\":" << index << ","
            << "\"chunk_id\":\"" << chunk_id_for_index(index) << "\","
            << "\"record_id\":\"record-0000\","
            << "\"chain_id\":\"chain-0000\","
            << "\"order_index\":" << index << ","
            << "\"text\":\"" << json_escape_local(chunk.text) << "\","
            << "\"voice\":";
        append_json_optional_string(out, request == nullptr ? nullptr : request->voice_id);
        out << ",\"language\":";
        append_json_optional_string(out, request == nullptr ? nullptr : request->language);
        out << ",\"emotion\":";
        append_json_optional_string(out, request == nullptr ? nullptr : request->emotion);
        out << ",\"emotion_guidance\":";
        append_json_optional_string(out, request == nullptr ? nullptr : request->emotion_guidance);
        out << ",\"context_before\":";
        append_json_optional_string(out, context_before);
        out << ",\"context_after\":";
        append_json_optional_string(out, context_after);
        out << ","
            << "\"boundary_before\":\"" << json_escape_local(chunk.boundary_before) << "\","
            << "\"boundary_after\":\"" << json_escape_local(chunk.boundary_after) << "\","
            << "\"starts_sentence\":" << (chunk.starts_sentence ? "true" : "false") << ","
            << "\"ends_sentence\":" << (chunk.ends_sentence ? "true" : "false") << ","
            << "\"g2p_token_count\":";
        append_json_nullable_int(out, chunk.g2p_retry_encoded_tokens);
        out << ",\"phone_count\":";
        append_json_nullable_int(out, chunk.phone_retry_phone_count);
        out << ",\"predicted_latent_frames\":";
        append_json_nullable_int(out, chunk.preflight_predicted_latent_frames);
        out << ",\"fixed_latent_frames\":";
        append_json_nullable_int(out, chunk.preflight_fixed_latent_frames);
        out << ",\"target_duration_seconds\":null,"
            << "\"split_reason\":\"" << json_escape_local(chunk.split_reason) << "\","
            << "\"reference_key\":";
        append_json_optional_string(out, reference_key);
        out << ",\"prefix_policy\":\"" << (index == 0 ? "none" : "previous_chunk") << "\","
            << "\"paragraph_index\":" << chunk.paragraph_index << ","
            << "\"unit_index\":" << chunk.unit_index << ","
            << "\"piece_index\":" << chunk.piece_index << ","
            << "\"piece_count\":" << chunk.piece_count << ","
            << "\"metadata\":{"
            << "\"paragraph_index\":" << chunk.paragraph_index << ","
            << "\"unit_index\":" << chunk.unit_index << ","
            << "\"piece_index\":" << chunk.piece_index << ","
            << "\"piece_count\":" << chunk.piece_count << ","
            << "\"split_reason\":\"" << json_escape_local(chunk.split_reason) << "\","
            << "\"boundary_before\":\"" << json_escape_local(chunk.boundary_before) << "\","
            << "\"boundary_after\":\"" << json_escape_local(chunk.boundary_after) << "\","
            << "\"starts_sentence\":" << (chunk.starts_sentence ? "true" : "false") << ","
            << "\"ends_sentence\":" << (chunk.ends_sentence ? "true" : "false") << ","
            << "\"preflight_predicted_latent_frames\":" << chunk.preflight_predicted_latent_frames << ","
            << "\"preflight_fixed_latent_frames\":" << chunk.preflight_fixed_latent_frames << ","
            << "\"latent_retry_predicted_frames\":" << chunk.latent_retry_predicted_frames << ","
            << "\"latent_retry_fixed_frames\":" << chunk.latent_retry_fixed_frames << ","
            << "\"g2p_retry_encoded_tokens\":" << chunk.g2p_retry_encoded_tokens << ","
            << "\"g2p_retry_fixed_tokens\":" << chunk.g2p_retry_fixed_tokens << ","
            << "\"phone_retry_phone_count\":" << chunk.phone_retry_phone_count << ","
            << "\"phone_retry_fixed_frames\":" << chunk.phone_retry_fixed_frames
            << "}}";
    }
    out << "]}";
    return out.str();
}

constexpr int kLongFormBoundaryFadeMs = 8;

int long_form_boundary_fade_samples(int sample_rate) {
    if (sample_rate <= 0 || kLongFormBoundaryFadeMs <= 0) {
        return 0;
    }
    return std::max(0, static_cast<int>(sample_rate * kLongFormBoundaryFadeMs / 1000.0f + 0.5f));
}

float smoothstep_weight(int index, int count) {
    if (count <= 1) {
        return 1.0f;
    }
    const float t = static_cast<float>(index) / static_cast<float>(count - 1);
    return t * t * (3.0f - 2.0f * t);
}

void fade_audio_edges(std::vector<float>& samples, int fade_samples, bool fade_in, bool fade_out) {
    if (fade_samples <= 0 || samples.empty()) {
        return;
    }
    const int sample_count = static_cast<int>(samples.size());
    const int local = std::min(fade_samples, sample_count / 2);
    if (local <= 0) {
        return;
    }
    if (fade_in) {
        for (int i = 0; i < local; ++i) {
            samples[static_cast<std::size_t>(i)] *= smoothstep_weight(i, local);
        }
    }
    if (fade_out) {
        for (int i = 0; i < local; ++i) {
            samples[static_cast<std::size_t>(sample_count - local + i)] *= 1.0f - smoothstep_weight(i, local);
        }
    }
}

void append_long_form_audio(
    std::vector<float>& rendered,
    const float* samples,
    int sample_count,
    int fade_samples,
    int previous_gap_samples,
    int next_gap_samples,
    std::size_t* out_start_sample,
    std::size_t* out_end_sample
) {
    const std::size_t fallback_start = rendered.size();
    if (out_start_sample != nullptr) {
        *out_start_sample = fallback_start;
    }
    if (out_end_sample != nullptr) {
        *out_end_sample = fallback_start;
    }
    if (sample_count <= 0 || samples == nullptr) {
        return;
    }
    std::vector<float> piece(samples, samples + sample_count);
    if (previous_gap_samples > 0 || next_gap_samples > 0) {
        fade_audio_edges(piece, fade_samples, previous_gap_samples > 0, next_gap_samples > 0);
    }
    if (previous_gap_samples <= 0 && !rendered.empty() && fade_samples > 0 && !piece.empty()) {
        const std::size_t local = std::min(
            std::min(static_cast<std::size_t>(fade_samples), rendered.size()),
            piece.size()
        );
        if (local > 0) {
            const std::size_t start = rendered.size() - local;
            for (std::size_t i = 0; i < local; ++i) {
                const float weight = smoothstep_weight(static_cast<int>(i), static_cast<int>(local));
                rendered[start + i] = rendered[start + i] * (1.0f - weight) + piece[i] * weight;
            }
            rendered.insert(rendered.end(), piece.begin() + static_cast<std::ptrdiff_t>(local), piece.end());
            if (out_start_sample != nullptr) {
                *out_start_sample = start;
            }
            if (out_end_sample != nullptr) {
                *out_end_sample = rendered.size();
            }
            return;
        }
    }
    const std::size_t start = rendered.size();
    rendered.insert(rendered.end(), piece.begin(), piece.end());
    if (out_start_sample != nullptr) {
        *out_start_sample = start;
    }
    if (out_end_sample != nullptr) {
        *out_end_sample = rendered.size();
    }
}

std::string chunk_id_for_index(std::size_t index) {
    std::ostringstream out;
    out << "chunk-";
    const int width = 4;
    std::string digits = std::to_string(index);
    if (static_cast<int>(digits.size()) < width) {
        out << std::string(static_cast<std::size_t>(width - static_cast<int>(digits.size())), '0');
    }
    out << digits;
    return out.str();
}

std::string single_chunk_event_json(
    const NativeChunkRecord& chunk,
    std::size_t index,
    int chunk_count,
    int pause_ms,
    std::size_t start_sample,
    std::size_t end_sample,
    const ScyllasBandSynthesisResult* result
) {
    std::ostringstream out;
    out << "{"
        << "\"index\":" << index << ","
        << "\"chunk_id\":\"" << chunk_id_for_index(index) << "\","
        << "\"chunk_count\":" << chunk_count << ","
        << "\"text\":\"" << json_escape_local(chunk.text) << "\","
        << "\"split_reason\":\"" << json_escape_local(chunk.split_reason) << "\","
        << "\"boundary_before\":\"" << json_escape_local(chunk.boundary_before) << "\","
        << "\"boundary_after\":\"" << json_escape_local(chunk.boundary_after) << "\","
        << "\"starts_sentence\":" << (chunk.starts_sentence ? "true" : "false") << ","
        << "\"ends_sentence\":" << (chunk.ends_sentence ? "true" : "false") << ","
        << "\"pause_after_ms\":" << pause_ms << ","
        << "\"preflight_predicted_latent_frames\":" << chunk.preflight_predicted_latent_frames << ","
        << "\"preflight_fixed_latent_frames\":" << chunk.preflight_fixed_latent_frames << ","
        << "\"start_sample\":" << start_sample << ","
        << "\"end_sample\":" << end_sample;
    if (result != nullptr) {
        out << ","
            << "\"sample_count\":" << result->sample_count << ","
            << "\"sample_rate\":" << result->sample_rate << ","
            << "\"latent_dim\":" << result->latent_dim << ","
            << "\"latent_frames\":" << result->latent_frames << ","
            << "\"metadata\":" << (result->metadata_json == nullptr ? "null" : result->metadata_json);
    }
    out << "}";
    return out.str();
}

ScyllasBandStatus emit_stream_event(
    ScyllasBandStreamingCallback callback,
    void* user_data,
    ScyllasBandStreamingEventType type,
    int chunk_index,
    int chunk_count,
    const std::string& chunk_id,
    const std::string& metadata_json,
    const ScyllasBandSynthesisResult* result
) {
    if (callback == nullptr) {
        scyllasband_detail::set_error("scyllasband_runtime_synthesize_long_form_stream: callback is required");
        return SCYLLASBAND_STATUS_INVALID_ARGUMENT;
    }
    ScyllasBandStreamingEvent event{};
    event.type = type;
    event.chunk_index = static_cast<int32_t>(chunk_index);
    event.chunk_count = static_cast<int32_t>(chunk_count);
    event.chunk_id = chunk_id.empty() ? nullptr : chunk_id.c_str();
    event.metadata_json = metadata_json.empty() ? nullptr : metadata_json.c_str();
    if (result != nullptr) {
        event.samples = result->samples;
        event.sample_count = result->sample_count;
        event.sample_rate = result->sample_rate;
        event.latents = result->latents;
        event.latent_dim = result->latent_dim;
        event.latent_frames = result->latent_frames;
    }
    if (callback(&event, user_data) != 0) {
        scyllasband_detail::set_error("scyllasband_runtime_synthesize_long_form_stream: callback cancelled synthesis");
        return SCYLLASBAND_STATUS_RUNTIME_ERROR;
    }
    return SCYLLASBAND_STATUS_OK;
}

}  // namespace

const char* scyllasband_version(void) {
    return "1.0.0";
}

ScyllasBandStatus scyllasband_runtime_create(
    const ScyllasBandRuntimeOptions* options,
    ScyllasBandRuntime** out_runtime
) {
    if (out_runtime != nullptr) {
        *out_runtime = nullptr;
    }
    if (options == nullptr || out_runtime == nullptr || options->bundle_dir == nullptr ||
        options->bundle_dir[0] == '\0') {
        scyllasband_detail::set_error("scyllasband_runtime_create: options, bundle_dir, and output pointer are required");
        return SCYLLASBAND_STATUS_INVALID_ARGUMENT;
    }
    if (!valid_backend(options->backend)) {
        scyllasband_detail::set_error("scyllasband_runtime_create: unsupported backend");
        return SCYLLASBAND_STATUS_INVALID_ARGUMENT;
    }
    if (!valid_litert_accelerator(options->litert_accelerator) || options->litert_max_threads < 0) {
        scyllasband_detail::set_error("scyllasband_runtime_create: unsupported LiteRT accelerator or thread hint");
        return SCYLLASBAND_STATUS_INVALID_ARGUMENT;
    }

    auto runtime = std::make_unique<ScyllasBandRuntime>();
    runtime->bundle_dir = options->bundle_dir;
    runtime->backend = scyllasband_detail::scyllasband_select_backend(options->backend);
    runtime->validate_bundle = options->validate_bundle != 0;
    runtime->litert_accelerator = options->litert_accelerator;
    runtime->litert_max_threads = options->litert_max_threads;
    runtime->engine = scyllasband_detail::create_backend_engine(
        runtime->backend,
        runtime->bundle_dir,
        runtime->validate_bundle,
        runtime->litert_accelerator,
        runtime->litert_max_threads
    );
    if (!runtime->engine) {
        const char* existing_error = scyllasband_detail::last_error();
        if (existing_error == nullptr || existing_error[0] == '\0') {
            scyllasband_detail::set_error("scyllasband_runtime_create: failed to create backend engine");
        }
        return SCYLLASBAND_STATUS_RUNTIME_ERROR;
    }
    *out_runtime = runtime.release();
    scyllasband_detail::clear_error();
    return SCYLLASBAND_STATUS_OK;
}

void scyllasband_runtime_destroy(ScyllasBandRuntime* runtime) {
    delete runtime;
}

ScyllasBandStatus scyllasband_runtime_set_target_bucket_cache_capacity(
    ScyllasBandRuntime* runtime,
    int32_t capacity
) {
    if (runtime == nullptr || capacity < 0) {
        scyllasband_detail::set_error(
            "scyllasband_runtime_set_target_bucket_cache_capacity: runtime is required and capacity must be non-negative"
        );
        return SCYLLASBAND_STATUS_INVALID_ARGUMENT;
    }
    if (!runtime->engine) {
        scyllasband_detail::set_error(
            "scyllasband_runtime_set_target_bucket_cache_capacity: runtime has no backend engine"
        );
        return SCYLLASBAND_STATUS_RUNTIME_ERROR;
    }
    runtime->target_bucket_cache_capacity = capacity;
    runtime->engine->set_target_bucket_cache_capacity(capacity);
    scyllasband_detail::clear_error();
    return SCYLLASBAND_STATUS_OK;
}

ScyllasBandStatus scyllasband_runtime_synthesize(
    ScyllasBandRuntime* runtime,
    const ScyllasBandSynthesisRequest* request,
    ScyllasBandSynthesisResult* out_result
) {
    reset_result(out_result);
    if (runtime == nullptr || out_result == nullptr) {
        scyllasband_detail::set_error("scyllasband_runtime_synthesize: runtime and output result are required");
        return SCYLLASBAND_STATUS_INVALID_ARGUMENT;
    }
    const ScyllasBandStatus request_status = validate_request(request);
    if (request_status != SCYLLASBAND_STATUS_OK) {
        return request_status;
    }
    if (!runtime->engine) {
        scyllasband_detail::set_error("scyllasband_runtime_synthesize: runtime has no backend engine");
        return SCYLLASBAND_STATUS_RUNTIME_ERROR;
    }
    return runtime->engine->synthesize(*request, out_result);
}

ScyllasBandStatus scyllasband_runtime_estimate_latent_frames(
    ScyllasBandRuntime* runtime,
    const ScyllasBandSynthesisRequest* request,
    ScyllasBandDurationEstimateResult* out_result
) {
    reset_duration_estimate_result(out_result);
    if (runtime == nullptr || request == nullptr || out_result == nullptr) {
        scyllasband_detail::set_error("scyllasband_runtime_estimate_latent_frames: runtime, request, and output result are required");
        return SCYLLASBAND_STATUS_INVALID_ARGUMENT;
    }
    const ScyllasBandStatus request_status = validate_request(request);
    if (request_status != SCYLLASBAND_STATUS_OK) {
        return request_status;
    }
    if (!runtime->engine) {
        scyllasband_detail::set_error("scyllasband_runtime_estimate_latent_frames: runtime has no backend engine");
        return SCYLLASBAND_STATUS_RUNTIME_ERROR;
    }
    scyllasband_detail::ScyllasBandDurationEstimate estimate;
    const ScyllasBandStatus status = runtime->engine->estimate_latent_frames(*request, &estimate);
    if (status != SCYLLASBAND_STATUS_OK) {
        return status;
    }
    out_result->predicted_latent_frames = estimate.predicted_latent_frames;
    out_result->fixed_latent_frames = estimate.fixed_latent_frames;
    if (!estimate.metadata_json.empty()) {
        out_result->metadata_json = duplicate_c_string_local(estimate.metadata_json);
        if (out_result->metadata_json == nullptr) {
            reset_duration_estimate_result(out_result);
            scyllasband_detail::set_error("scyllasband_runtime_estimate_latent_frames: failed to allocate metadata");
            return SCYLLASBAND_STATUS_RUNTIME_ERROR;
        }
    }
    scyllasband_detail::clear_error();
    return SCYLLASBAND_STATUS_OK;
}

ScyllasBandStatus scyllasband_runtime_synthesize_long_form(
    ScyllasBandRuntime* runtime,
    const ScyllasBandLongFormSynthesisRequest* request,
    ScyllasBandSynthesisResult* out_result
) {
    reset_result(out_result);
    if (runtime == nullptr || request == nullptr || out_result == nullptr) {
        scyllasband_detail::set_error("scyllasband_runtime_synthesize_long_form: runtime, request, and output result are required");
        return SCYLLASBAND_STATUS_INVALID_ARGUMENT;
    }
    if (request->request.text == nullptr || request->request.text[0] == '\0') {
        scyllasband_detail::set_error("scyllasband_runtime_synthesize_long_form: request.text is required");
        return SCYLLASBAND_STATUS_INVALID_ARGUMENT;
    }
    if (request->request.explicit_phones != nullptr && request->request.explicit_phones[0] != '\0') {
        scyllasband_detail::set_error("scyllasband_runtime_synthesize_long_form: explicit_phones are not supported for long-form text chunking");
        return SCYLLASBAND_STATUS_INVALID_ARGUMENT;
    }
    const int max_chunk_chars = request->max_chunk_chars > 0 ? request->max_chunk_chars : 220;
    const int min_chunk_chars = request->min_chunk_chars > 0 ? std::min(request->min_chunk_chars, max_chunk_chars) : std::min(48, max_chunk_chars);
    const std::string normalized_text = normalize_spoken_long_form_text(
        request->request.text,
        request->request.language
    );
    std::vector<NativeChunkRecord> chunks = plan_long_form_chunks(normalized_text, max_chunk_chars, min_chunk_chars);
    if (chunks.empty()) {
        scyllasband_detail::set_error("scyllasband_runtime_synthesize_long_form: no speakable chunks after text splitting");
        return SCYLLASBAND_STATUS_INVALID_ARGUMENT;
    }

    const bool auto_split_overlong = request->disable_auto_split_overlong == 0;
    const bool preflight_chunks = request->preflight_chunks != 0;
    const int duration_preflight_split_count = (auto_split_overlong && preflight_chunks)
        ? preflight_split_overlong_chunks(runtime, chunks, request->request, max_chunk_chars, min_chunk_chars)
        : 0;

    std::vector<float> rendered;
    std::vector<float> previous_latents;
    int previous_latent_dim = 0;
    int previous_latent_frames = 0;
    std::string previous_key;
    int sample_rate = 0;
    int previous_pause_samples = 0;
    std::ostringstream chunk_metadata;
    chunk_metadata << "[";
    std::size_t index = 0;
    int retry_split_count = 0;
    int latent_retry_split_count = 0;
    int g2p_retry_split_count = 0;
    int phone_retry_split_count = 0;
    int retry_split_budget = std::max<int>(1, static_cast<int>(chunks.size()) * 16);

    while (index < chunks.size()) {
        const NativeChunkRecord& chunk = chunks[index];
        const std::string context_before = context_before_for(chunks, index);
        const std::string context_after = context_after_for(chunks, index);
        const std::string prefix_key = std::string(request->request.voice_id ? request->request.voice_id : "") + "|" +
                                       std::string(request->request.language ? request->request.language : "") + "|" +
                                       std::string(request->request.emotion ? request->request.emotion : "");
        ScyllasBandSynthesisRequest chunk_request = request->request;
        chunk_request.text = chunk.text.c_str();
        chunk_request.explicit_phones = nullptr;
        chunk_request.context_before = context_before.empty() ? nullptr : context_before.c_str();
        chunk_request.context_after = context_after.empty() ? nullptr : context_after.c_str();
        chunk_request.chunk_index = static_cast<int32_t>(index);
        chunk_request.chunk_count = static_cast<int32_t>(chunks.size());
        chunk_request.boundary_before = chunk.boundary_before.c_str();
        chunk_request.boundary_after = chunk.boundary_after.c_str();
        if (request->request.has_seed) {
            chunk_request.seed = request->request.seed + static_cast<uint64_t>(index);
            chunk_request.has_seed = 1;
        }
        if (request->use_prefix_latents && !previous_latents.empty() && prefix_key == previous_key) {
            chunk_request.prefix_latents = previous_latents.data();
            chunk_request.prefix_latent_dim = previous_latent_dim;
            chunk_request.prefix_latent_frames = previous_latent_frames;
        } else {
            chunk_request.prefix_latents = nullptr;
            chunk_request.prefix_latent_dim = 0;
            chunk_request.prefix_latent_frames = 0;
        }

        ScyllasBandSynthesisResult chunk_result{};
        ScyllasBandStatus status = scyllasband_runtime_synthesize(runtime, &chunk_request, &chunk_result);
        if (status != SCYLLASBAND_STATUS_OK) {
            const std::string error_message = scyllasband_last_error() == nullptr ? "" : scyllasband_last_error();
            const OverlongLatentCounts overlong = parse_overlong_latent_error(error_message);
            const OverlongG2PCounts overlong_g2p = parse_overlong_g2p_error(error_message);
            const OverlongPhoneCounts overlong_phone = parse_overlong_phone_error(error_message);
            scyllasband_synthesis_result_free(&chunk_result);
            if (auto_split_overlong && retry_split_budget > 0 && overlong.valid) {
                std::vector<NativeChunkRecord> replacements = split_chunk_for_latent_retry(
                    chunk,
                    max_chunk_chars,
                    min_chunk_chars,
                    overlong.predicted,
                    overlong.fixed
                );
                if (!replacements.empty()) {
                    --retry_split_budget;
                    ++retry_split_count;
                    ++latent_retry_split_count;
                    chunks.erase(chunks.begin() + static_cast<std::ptrdiff_t>(index));
                    chunks.insert(
                        chunks.begin() + static_cast<std::ptrdiff_t>(index),
                        std::make_move_iterator(replacements.begin()),
                        std::make_move_iterator(replacements.end())
                    );
                    scyllasband_clear_error();
                    continue;
                }
            }
            if (auto_split_overlong && retry_split_budget > 0 && overlong_phone.valid) {
                std::vector<NativeChunkRecord> replacements = split_chunk_for_phone_retry(
                    chunk,
                    max_chunk_chars,
                    min_chunk_chars,
                    overlong_phone.phone_count,
                    overlong_phone.fixed
                );
                if (!replacements.empty()) {
                    --retry_split_budget;
                    ++retry_split_count;
                    ++phone_retry_split_count;
                    chunks.erase(chunks.begin() + static_cast<std::ptrdiff_t>(index));
                    chunks.insert(
                        chunks.begin() + static_cast<std::ptrdiff_t>(index),
                        std::make_move_iterator(replacements.begin()),
                        std::make_move_iterator(replacements.end())
                    );
                    scyllasband_clear_error();
                    continue;
                }
            }
            if (auto_split_overlong && retry_split_budget > 0 && overlong_g2p.valid) {
                std::vector<NativeChunkRecord> replacements = split_chunk_for_g2p_retry(
                    chunk,
                    max_chunk_chars,
                    min_chunk_chars,
                    overlong_g2p.encoded,
                    overlong_g2p.fixed
                );
                if (!replacements.empty()) {
                    --retry_split_budget;
                    ++retry_split_count;
                    ++g2p_retry_split_count;
                    chunks.erase(chunks.begin() + static_cast<std::ptrdiff_t>(index));
                    chunks.insert(
                        chunks.begin() + static_cast<std::ptrdiff_t>(index),
                        std::make_move_iterator(replacements.begin()),
                        std::make_move_iterator(replacements.end())
                    );
                    scyllasband_clear_error();
                    continue;
                }
            }
            return status;
        }
        if (sample_rate == 0) {
            sample_rate = chunk_result.sample_rate;
        }
        const int pause_ms = pause_after_ms(
            chunk,
            index + 1 >= chunks.size(),
            request->pause_ms,
            request->continuation_pause_ms
        );
        const int pause_samples = (pause_ms > 0 && index + 1 < chunks.size() && sample_rate > 0)
            ? std::max(0, static_cast<int>(sample_rate * pause_ms / 1000.0f + 0.5f))
            : 0;
        std::size_t start_sample = rendered.size();
        std::size_t end_sample = rendered.size();
        if (chunk_result.sample_count < 0 || (chunk_result.sample_count > 0 && chunk_result.samples == nullptr)) {
            scyllasband_synthesis_result_free(&chunk_result);
            scyllasband_detail::set_error("scyllasband_runtime_synthesize_long_form: chunk returned invalid audio buffer");
            return SCYLLASBAND_STATUS_RUNTIME_ERROR;
        }
        if (chunk_result.sample_count > 0) {
            append_long_form_audio(
                rendered,
                chunk_result.samples,
                chunk_result.sample_count,
                long_form_boundary_fade_samples(sample_rate),
                previous_pause_samples,
                pause_samples,
                &start_sample,
                &end_sample
            );
        }
        const std::size_t chunk_latent_count =
            static_cast<std::size_t>(std::max(0, chunk_result.latent_dim)) *
            static_cast<std::size_t>(std::max(0, chunk_result.latent_frames));
        if (chunk_result.latents != nullptr && chunk_latent_count > 0) {
            previous_latents.assign(chunk_result.latents, chunk_result.latents + chunk_latent_count);
            previous_latent_dim = chunk_result.latent_dim;
            previous_latent_frames = chunk_result.latent_frames;
            previous_key = prefix_key;
        } else {
            previous_latents.clear();
            previous_latent_dim = 0;
            previous_latent_frames = 0;
            previous_key.clear();
        }

        if (index > 0) {
            chunk_metadata << ",";
        }
        chunk_metadata << "{"
                       << "\"index\":" << index << ","
                       << "\"text\":\"" << json_escape_local(chunk.text) << "\","
                       << "\"split_reason\":\"" << json_escape_local(chunk.split_reason) << "\","
                       << "\"boundary_before\":\"" << json_escape_local(chunk.boundary_before) << "\","
                       << "\"boundary_after\":\"" << json_escape_local(chunk.boundary_after) << "\","
                       << "\"starts_sentence\":" << (chunk.starts_sentence ? "true" : "false") << ","
                       << "\"ends_sentence\":" << (chunk.ends_sentence ? "true" : "false") << ","
                       << "\"pause_after_ms\":" << pause_ms << ","
                       << "\"preflight_predicted_latent_frames\":" << chunk.preflight_predicted_latent_frames << ","
                       << "\"preflight_fixed_latent_frames\":" << chunk.preflight_fixed_latent_frames << ","
                       << "\"start_sample\":" << start_sample << ","
                       << "\"end_sample\":" << end_sample << ","
                       << "\"sample_count\":" << chunk_result.sample_count << ","
                       << "\"latent_dim\":" << chunk_result.latent_dim << ","
                       << "\"latent_frames\":" << chunk_result.latent_frames << ","
                       << "\"latent_retry_predicted_frames\":" << chunk.latent_retry_predicted_frames << ","
                       << "\"latent_retry_fixed_frames\":" << chunk.latent_retry_fixed_frames << ","
                       << "\"g2p_retry_encoded_tokens\":" << chunk.g2p_retry_encoded_tokens << ","
                       << "\"g2p_retry_fixed_tokens\":" << chunk.g2p_retry_fixed_tokens << ","
                       << "\"phone_retry_phone_count\":" << chunk.phone_retry_phone_count << ","
                       << "\"phone_retry_fixed_frames\":" << chunk.phone_retry_fixed_frames << ","
                       << "\"metadata\":" << (chunk_result.metadata_json == nullptr ? "null" : chunk_result.metadata_json)
                       << "}";
        if (pause_samples > 0) {
            rendered.insert(rendered.end(), static_cast<std::size_t>(pause_samples), 0.0f);
            previous_pause_samples = pause_samples;
        } else {
            previous_pause_samples = 0;
        }
        scyllasband_synthesis_result_free(&chunk_result);
        ++index;
    }
    chunk_metadata << "]";

    if (rendered.size() > static_cast<std::size_t>(INT32_MAX)) {
        scyllasband_detail::set_error("scyllasband_runtime_synthesize_long_form: output audio is too large");
        return SCYLLASBAND_STATUS_RUNTIME_ERROR;
    }
    float* samples = nullptr;
    if (!rendered.empty()) {
        samples = static_cast<float*>(std::malloc(rendered.size() * sizeof(float)));
        if (samples == nullptr) {
            scyllasband_detail::set_error("scyllasband_runtime_synthesize_long_form: malloc failed for output audio");
            return SCYLLASBAND_STATUS_RUNTIME_ERROR;
        }
        std::memcpy(samples, rendered.data(), rendered.size() * sizeof(float));
    }
    float* latents = nullptr;
    if (!previous_latents.empty()) {
        latents = static_cast<float*>(std::malloc(previous_latents.size() * sizeof(float)));
        if (latents == nullptr) {
            std::free(samples);
            scyllasband_detail::set_error("scyllasband_runtime_synthesize_long_form: malloc failed for output latents");
            return SCYLLASBAND_STATUS_RUNTIME_ERROR;
        }
        std::memcpy(latents, previous_latents.data(), previous_latents.size() * sizeof(float));
    }
    std::ostringstream metadata;
    metadata << "{"
             << "\"status\":\"ok\","
             << "\"mode\":\"native_chunked\","
             << "\"backend\":\"" << backend_name_for_plan(runtime->backend) << "\","
             << "\"sample_rate\":" << sample_rate << ","
             << "\"sample_count\":" << rendered.size() << ","
             << "\"chunk_count\":" << chunks.size() << ","
             << "\"sampler\":\"" << (request->request.sampler == SCYLLASBAND_SAMPLER_HEUN ? "heun" : "euler") << "\","
             << "\"steps\":" << std::max<int>(1, request->request.steps) << ","
             << "\"speed\":" << request->request.speed << ","
             << "\"emotion_guidance\":";
    if (request->request.emotion_guidance == nullptr || request->request.emotion_guidance[0] == '\0') {
        metadata << "null";
    } else {
        metadata << "\"" << json_escape_local(request->request.emotion_guidance) << "\"";
    }
    metadata << ","
             << "\"guidance_null_reference\":" << (request->request.guidance_null_reference ? "true" : "false") << ","
             << "\"emotion_embed_scale\":" << request->request.emotion_embed_scale << ","
             << "\"normalized_text\":\"" << json_escape_local(normalized_text) << "\","
             << "\"chunk_max_chars\":" << max_chunk_chars << ","
             << "\"chunk_min_chars\":" << min_chunk_chars << ","
             << "\"pause_ms\":" << request->pause_ms << ","
             << "\"continuation_pause_ms\":" << request->continuation_pause_ms << ","
             << "\"min_sentence_pause_ms\":" << request->request.min_sentence_pause_ms << ","
             << "\"min_clause_pause_ms\":" << request->request.min_clause_pause_ms << ","
             << "\"boundary_fade_ms\":" << kLongFormBoundaryFadeMs << ","
             << "\"use_prefix_latents\":" << (request->use_prefix_latents ? "true" : "false") << ","
             << "\"auto_split_overlong\":" << (auto_split_overlong ? "true" : "false") << ","
             << "\"preflight_chunks\":" << (preflight_chunks ? "true" : "false") << ","
             << "\"duration_preflight_split_count\":" << duration_preflight_split_count << ","
             << "\"retry_split_count\":" << retry_split_count << ","
             << "\"latent_retry_split_count\":" << latent_retry_split_count << ","
             << "\"g2p_retry_split_count\":" << g2p_retry_split_count << ","
             << "\"phone_retry_split_count\":" << phone_retry_split_count << ","
             << "\"chunks\":" << chunk_metadata.str()
             << "}";
    char* metadata_json = duplicate_c_string_local(metadata.str());
    if (metadata_json == nullptr) {
        std::free(samples);
        std::free(latents);
        scyllasband_detail::set_error("scyllasband_runtime_synthesize_long_form: malloc failed for metadata");
        return SCYLLASBAND_STATUS_RUNTIME_ERROR;
    }
    out_result->samples = samples;
    out_result->sample_count = static_cast<int32_t>(rendered.size());
    out_result->sample_rate = sample_rate;
    out_result->metadata_json = metadata_json;
    out_result->latents = latents;
    out_result->latent_dim = previous_latent_dim;
    out_result->latent_frames = previous_latent_frames;
    scyllasband_detail::clear_error();
    return SCYLLASBAND_STATUS_OK;
}


ScyllasBandStatus scyllasband_runtime_plan_long_form(
    ScyllasBandRuntime* runtime,
    const ScyllasBandLongFormSynthesisRequest* request,
    ScyllasBandChunkPlanResult* out_result
) {
    reset_chunk_plan_result(out_result);
    if (runtime == nullptr || request == nullptr || out_result == nullptr) {
        scyllasband_detail::set_error("scyllasband_runtime_plan_long_form: runtime, request, and output result are required");
        return SCYLLASBAND_STATUS_INVALID_ARGUMENT;
    }
    if (request->request.text == nullptr || request->request.text[0] == '\0') {
        scyllasband_detail::set_error("scyllasband_runtime_plan_long_form: request.text is required");
        return SCYLLASBAND_STATUS_INVALID_ARGUMENT;
    }
    if (request->request.explicit_phones != nullptr && request->request.explicit_phones[0] != '\0') {
        scyllasband_detail::set_error("scyllasband_runtime_plan_long_form: explicit_phones are not supported for long-form text chunking");
        return SCYLLASBAND_STATUS_INVALID_ARGUMENT;
    }
    const ScyllasBandStatus request_status = validate_request(&request->request);
    if (request_status != SCYLLASBAND_STATUS_OK) {
        return request_status;
    }
    if (!runtime->engine) {
        scyllasband_detail::set_error("scyllasband_runtime_plan_long_form: runtime has no backend engine");
        return SCYLLASBAND_STATUS_RUNTIME_ERROR;
    }
    const int max_chunk_chars = request->max_chunk_chars > 0 ? request->max_chunk_chars : 220;
    const int min_chunk_chars = request->min_chunk_chars > 0 ? std::min(request->min_chunk_chars, max_chunk_chars) : std::min(48, max_chunk_chars);
    const std::string normalized_text = normalize_spoken_long_form_text(
        request->request.text,
        request->request.language
    );
    std::vector<NativeChunkRecord> chunks = plan_long_form_chunks(normalized_text, max_chunk_chars, min_chunk_chars);
    if (chunks.empty()) {
        scyllasband_detail::set_error("scyllasband_runtime_plan_long_form: no speakable chunks after text splitting");
        return SCYLLASBAND_STATUS_INVALID_ARGUMENT;
    }
    const bool auto_split_overlong = request->disable_auto_split_overlong == 0;
    const bool preflight_chunks = request->preflight_chunks != 0;
    if (auto_split_overlong && preflight_chunks) {
        preflight_split_overlong_chunks(runtime, chunks, request->request, max_chunk_chars, min_chunk_chars);
    }
    char* metadata_json = duplicate_c_string_local(chunk_plan_json(chunks, max_chunk_chars, min_chunk_chars, normalized_text, &request->request, runtime->backend, auto_split_overlong, preflight_chunks));
    if (metadata_json == nullptr) {
        scyllasband_detail::set_error("scyllasband_runtime_plan_long_form: failed to allocate metadata");
        return SCYLLASBAND_STATUS_RUNTIME_ERROR;
    }
    out_result->metadata_json = metadata_json;
    scyllasband_detail::clear_error();
    return SCYLLASBAND_STATUS_OK;
}

ScyllasBandStatus scyllasband_runtime_synthesize_long_form_stream(
    ScyllasBandRuntime* runtime,
    const ScyllasBandLongFormSynthesisRequest* request,
    ScyllasBandStreamingCallback callback,
    void* user_data
) {
    if (runtime == nullptr || request == nullptr || callback == nullptr) {
        scyllasband_detail::set_error("scyllasband_runtime_synthesize_long_form_stream: runtime, request, and callback are required");
        return SCYLLASBAND_STATUS_INVALID_ARGUMENT;
    }
    if (request->request.text == nullptr || request->request.text[0] == '\0') {
        scyllasband_detail::set_error("scyllasband_runtime_synthesize_long_form_stream: request.text is required");
        return SCYLLASBAND_STATUS_INVALID_ARGUMENT;
    }
    if (request->request.explicit_phones != nullptr && request->request.explicit_phones[0] != '\0') {
        scyllasband_detail::set_error("scyllasband_runtime_synthesize_long_form_stream: explicit_phones are not supported for long-form text chunking");
        return SCYLLASBAND_STATUS_INVALID_ARGUMENT;
    }
    const ScyllasBandStatus request_status = validate_request(&request->request);
    if (request_status != SCYLLASBAND_STATUS_OK) {
        return request_status;
    }
    if (!runtime->engine) {
        scyllasband_detail::set_error("scyllasband_runtime_synthesize_long_form_stream: runtime has no backend engine");
        return SCYLLASBAND_STATUS_RUNTIME_ERROR;
    }

    const int max_chunk_chars = request->max_chunk_chars > 0 ? request->max_chunk_chars : 220;
    const int min_chunk_chars = request->min_chunk_chars > 0 ? std::min(request->min_chunk_chars, max_chunk_chars) : std::min(48, max_chunk_chars);
    const std::string normalized_text = normalize_spoken_long_form_text(
        request->request.text,
        request->request.language
    );
    std::vector<NativeChunkRecord> chunks = plan_long_form_chunks(normalized_text, max_chunk_chars, min_chunk_chars);
    if (chunks.empty()) {
        scyllasband_detail::set_error("scyllasband_runtime_synthesize_long_form_stream: no speakable chunks after text splitting");
        return SCYLLASBAND_STATUS_INVALID_ARGUMENT;
    }

    const bool auto_split_overlong = request->disable_auto_split_overlong == 0;
    const bool preflight_chunks = request->preflight_chunks != 0;
    const int duration_preflight_split_count = (auto_split_overlong && preflight_chunks)
        ? preflight_split_overlong_chunks(runtime, chunks, request->request, max_chunk_chars, min_chunk_chars)
        : 0;
    const std::string plan_metadata = chunk_plan_json(chunks, max_chunk_chars, min_chunk_chars, normalized_text, &request->request, runtime->backend, auto_split_overlong, preflight_chunks);
    ScyllasBandStatus callback_status = emit_stream_event(
        callback,
        user_data,
        SCYLLASBAND_STREAM_EVENT_PLAN_READY,
        -1,
        static_cast<int>(chunks.size()),
        std::string(),
        plan_metadata,
        nullptr
    );
    if (callback_status != SCYLLASBAND_STATUS_OK) {
        return callback_status;
    }

    std::vector<float> previous_latents;
    int previous_latent_dim = 0;
    int previous_latent_frames = 0;
    std::string previous_key;
    int sample_rate = 0;
    std::size_t rendered_sample_count = 0;
    std::size_t index = 0;
    int retry_split_count = 0;
    int latent_retry_split_count = 0;
    int g2p_retry_split_count = 0;
    int phone_retry_split_count = 0;
    int retry_split_budget = std::max<int>(1, static_cast<int>(chunks.size()) * 16);
    std::vector<std::string> rendered_chunk_metadata;

    while (index < chunks.size()) {
        const NativeChunkRecord& chunk = chunks[index];
        const std::string chunk_id = chunk_id_for_index(index);
        const std::string context_before = context_before_for(chunks, index);
        const std::string context_after = context_after_for(chunks, index);
        const std::string prefix_key = std::string(request->request.voice_id ? request->request.voice_id : "") + "|" +
                                       std::string(request->request.language ? request->request.language : "") + "|" +
                                       std::string(request->request.emotion ? request->request.emotion : "");
        const int pause_ms = pause_after_ms(
            chunk,
            index + 1 >= chunks.size(),
            request->pause_ms,
            request->continuation_pause_ms
        );
        const std::string started_metadata = single_chunk_event_json(
            chunk,
            index,
            static_cast<int>(chunks.size()),
            pause_ms,
            rendered_sample_count,
            rendered_sample_count,
            nullptr
        );
        callback_status = emit_stream_event(
            callback,
            user_data,
            SCYLLASBAND_STREAM_EVENT_CHUNK_STARTED,
            static_cast<int>(index),
            static_cast<int>(chunks.size()),
            chunk_id,
            started_metadata,
            nullptr
        );
        if (callback_status != SCYLLASBAND_STATUS_OK) {
            return callback_status;
        }

        ScyllasBandSynthesisRequest chunk_request = request->request;
        chunk_request.text = chunk.text.c_str();
        chunk_request.explicit_phones = nullptr;
        chunk_request.context_before = context_before.empty() ? nullptr : context_before.c_str();
        chunk_request.context_after = context_after.empty() ? nullptr : context_after.c_str();
        chunk_request.chunk_index = static_cast<int32_t>(index);
        chunk_request.chunk_count = static_cast<int32_t>(chunks.size());
        chunk_request.boundary_before = chunk.boundary_before.c_str();
        chunk_request.boundary_after = chunk.boundary_after.c_str();
        if (request->request.has_seed) {
            chunk_request.seed = request->request.seed + static_cast<uint64_t>(index);
            chunk_request.has_seed = 1;
        }
        if (request->use_prefix_latents && !previous_latents.empty() && prefix_key == previous_key) {
            chunk_request.prefix_latents = previous_latents.data();
            chunk_request.prefix_latent_dim = previous_latent_dim;
            chunk_request.prefix_latent_frames = previous_latent_frames;
        } else {
            chunk_request.prefix_latents = nullptr;
            chunk_request.prefix_latent_dim = 0;
            chunk_request.prefix_latent_frames = 0;
        }

        ScyllasBandSynthesisResult chunk_result{};
        ScyllasBandStatus status = scyllasband_runtime_synthesize(runtime, &chunk_request, &chunk_result);
        if (status != SCYLLASBAND_STATUS_OK) {
            const std::string error_message = scyllasband_last_error() == nullptr ? "" : scyllasband_last_error();
            const OverlongLatentCounts overlong = parse_overlong_latent_error(error_message);
            const OverlongG2PCounts overlong_g2p = parse_overlong_g2p_error(error_message);
            const OverlongPhoneCounts overlong_phone = parse_overlong_phone_error(error_message);
            scyllasband_synthesis_result_free(&chunk_result);
            std::vector<NativeChunkRecord> replacements;
            const char* budget_name = nullptr;
            if (auto_split_overlong && retry_split_budget > 0 && overlong.valid) {
                replacements = split_chunk_for_latent_retry(chunk, max_chunk_chars, min_chunk_chars, overlong.predicted, overlong.fixed);
                budget_name = "latent";
                if (!replacements.empty()) {
                    ++latent_retry_split_count;
                }
            }
            if (replacements.empty() && auto_split_overlong && retry_split_budget > 0 && overlong_phone.valid) {
                replacements = split_chunk_for_phone_retry(chunk, max_chunk_chars, min_chunk_chars, overlong_phone.phone_count, overlong_phone.fixed);
                budget_name = "phone";
                if (!replacements.empty()) {
                    ++phone_retry_split_count;
                }
            }
            if (replacements.empty() && auto_split_overlong && retry_split_budget > 0 && overlong_g2p.valid) {
                replacements = split_chunk_for_g2p_retry(chunk, max_chunk_chars, min_chunk_chars, overlong_g2p.encoded, overlong_g2p.fixed);
                budget_name = "g2p";
                if (!replacements.empty()) {
                    ++g2p_retry_split_count;
                }
            }
            if (!replacements.empty()) {
                --retry_split_budget;
                ++retry_split_count;
                std::ostringstream warning;
                warning << "{\"budget\":\"" << budget_name << "\","
                        << "\"replacement_count\":" << replacements.size() << ","
                        << "\"message\":\"retrying overlong chunk as smaller chunks\"}";
                callback_status = emit_stream_event(
                    callback,
                    user_data,
                    SCYLLASBAND_STREAM_EVENT_WARNING,
                    static_cast<int>(index),
                    static_cast<int>(chunks.size()),
                    chunk_id,
                    warning.str(),
                    nullptr
                );
                if (callback_status != SCYLLASBAND_STATUS_OK) {
                    return callback_status;
                }
                chunks.erase(chunks.begin() + static_cast<std::ptrdiff_t>(index));
                chunks.insert(
                    chunks.begin() + static_cast<std::ptrdiff_t>(index),
                    std::make_move_iterator(replacements.begin()),
                    std::make_move_iterator(replacements.end())
                );
                scyllasband_clear_error();
                continue;
            }
            return status;
        }

        if (sample_rate == 0) {
            sample_rate = chunk_result.sample_rate;
        }
        if (chunk_result.sample_count < 0 || (chunk_result.sample_count > 0 && chunk_result.samples == nullptr)) {
            scyllasband_synthesis_result_free(&chunk_result);
            scyllasband_detail::set_error("scyllasband_runtime_synthesize_long_form_stream: chunk returned invalid audio buffer");
            return SCYLLASBAND_STATUS_RUNTIME_ERROR;
        }
        const int pause_samples = (pause_ms > 0 && index + 1 < chunks.size() && sample_rate > 0)
            ? std::max(0, static_cast<int>(sample_rate * pause_ms / 1000.0f + 0.5f))
            : 0;
        const std::size_t start_sample = rendered_sample_count;
        const std::size_t end_sample = rendered_sample_count + static_cast<std::size_t>(std::max(0, chunk_result.sample_count));
        const std::string audio_metadata = single_chunk_event_json(
            chunk,
            index,
            static_cast<int>(chunks.size()),
            pause_ms,
            start_sample,
            end_sample,
            &chunk_result
        );
        rendered_chunk_metadata.push_back(audio_metadata);
        callback_status = emit_stream_event(
            callback,
            user_data,
            SCYLLASBAND_STREAM_EVENT_AUDIO_CHUNK,
            static_cast<int>(index),
            static_cast<int>(chunks.size()),
            chunk_id,
            audio_metadata,
            &chunk_result
        );
        if (callback_status != SCYLLASBAND_STATUS_OK) {
            scyllasband_synthesis_result_free(&chunk_result);
            return callback_status;
        }
        callback_status = emit_stream_event(
            callback,
            user_data,
            SCYLLASBAND_STREAM_EVENT_CHUNK_FINISHED,
            static_cast<int>(index),
            static_cast<int>(chunks.size()),
            chunk_id,
            audio_metadata,
            nullptr
        );
        if (callback_status != SCYLLASBAND_STATUS_OK) {
            scyllasband_synthesis_result_free(&chunk_result);
            return callback_status;
        }

        const std::size_t chunk_latent_count =
            static_cast<std::size_t>(std::max(0, chunk_result.latent_dim)) *
            static_cast<std::size_t>(std::max(0, chunk_result.latent_frames));
        if (chunk_result.latents != nullptr && chunk_latent_count > 0) {
            previous_latents.assign(chunk_result.latents, chunk_result.latents + chunk_latent_count);
            previous_latent_dim = chunk_result.latent_dim;
            previous_latent_frames = chunk_result.latent_frames;
            previous_key = prefix_key;
        } else {
            previous_latents.clear();
            previous_latent_dim = 0;
            previous_latent_frames = 0;
            previous_key.clear();
        }
        rendered_sample_count = end_sample + static_cast<std::size_t>(pause_samples);
        scyllasband_synthesis_result_free(&chunk_result);
        ++index;
    }

    std::ostringstream done_metadata;
    done_metadata << "{"
                  << "\"status\":\"ok\","
                  << "\"mode\":\"native_chunked_stream\","
                  << "\"backend\":\"" << backend_name_for_plan(runtime->backend) << "\","
                  << "\"sample_rate\":" << sample_rate << ","
                  << "\"sample_count\":" << rendered_sample_count << ","
                  << "\"chunk_count\":" << chunks.size() << ","
                  << "\"preflight_chunks\":" << (preflight_chunks ? "true" : "false") << ","
                  << "\"duration_preflight_split_count\":" << duration_preflight_split_count << ","
                  << "\"retry_split_count\":" << retry_split_count << ","
                  << "\"latent_retry_split_count\":" << latent_retry_split_count << ","
                  << "\"g2p_retry_split_count\":" << g2p_retry_split_count << ","
                  << "\"phone_retry_split_count\":" << phone_retry_split_count << ","
                  << "\"chunks\":[";
    for (std::size_t rendered_index = 0; rendered_index < rendered_chunk_metadata.size(); ++rendered_index) {
        if (rendered_index > 0) {
            done_metadata << ",";
        }
        done_metadata << rendered_chunk_metadata[rendered_index];
    }
    done_metadata << "]"
                  << "}";
    callback_status = emit_stream_event(
        callback,
        user_data,
        SCYLLASBAND_STREAM_EVENT_DONE,
        -1,
        static_cast<int>(chunks.size()),
        std::string(),
        done_metadata.str(),
        nullptr
    );
    if (callback_status != SCYLLASBAND_STATUS_OK) {
        return callback_status;
    }
    scyllasband_detail::clear_error();
    return SCYLLASBAND_STATUS_OK;
}


ScyllasBandStatus scyllasband_plan_long_form_chunks(
    const ScyllasBandChunkPlanRequest* request,
    ScyllasBandChunkPlanResult* out_result
) {
    reset_chunk_plan_result(out_result);
    if (request == nullptr || out_result == nullptr) {
        scyllasband_detail::set_error("scyllasband_plan_long_form_chunks: request and output result are required");
        return SCYLLASBAND_STATUS_INVALID_ARGUMENT;
    }
    if (request->text == nullptr || request->text[0] == '\0') {
        scyllasband_detail::set_error("scyllasband_plan_long_form_chunks: text is required");
        return SCYLLASBAND_STATUS_INVALID_ARGUMENT;
    }
    const int max_chunk_chars = request->max_chunk_chars > 0 ? request->max_chunk_chars : 220;
    const int min_chunk_chars = request->min_chunk_chars > 0 ? std::min(request->min_chunk_chars, max_chunk_chars) : std::min(48, max_chunk_chars);
    const std::string normalized_text = normalize_spoken_long_form_text(request->text, "en");
    std::vector<NativeChunkRecord> chunks = plan_long_form_chunks(normalized_text, max_chunk_chars, min_chunk_chars);
    if (chunks.empty()) {
        scyllasband_detail::set_error("scyllasband_plan_long_form_chunks: no speakable chunks after text splitting");
        return SCYLLASBAND_STATUS_INVALID_ARGUMENT;
    }
    char* metadata_json = duplicate_c_string_local(chunk_plan_json(chunks, max_chunk_chars, min_chunk_chars, normalized_text, nullptr, SCYLLASBAND_BACKEND_AUTO, false));
    if (metadata_json == nullptr) {
        scyllasband_detail::set_error("scyllasband_plan_long_form_chunks: failed to allocate metadata");
        return SCYLLASBAND_STATUS_RUNTIME_ERROR;
    }
    out_result->metadata_json = metadata_json;
    scyllasband_detail::clear_error();
    return SCYLLASBAND_STATUS_OK;
}

void scyllasband_chunk_plan_result_free(ScyllasBandChunkPlanResult* result) {
    if (result == nullptr) {
        return;
    }
    std::free(result->metadata_json);
    result->metadata_json = nullptr;
}

void scyllasband_duration_estimate_result_free(ScyllasBandDurationEstimateResult* result) {
    if (result == nullptr) {
        return;
    }
    std::free(result->metadata_json);
    result->predicted_latent_frames = 0;
    result->fixed_latent_frames = 0;
    result->metadata_json = nullptr;
}

void scyllasband_synthesis_result_free(ScyllasBandSynthesisResult* result) {
    if (result == nullptr) {
        return;
    }
    std::free(result->samples);
    std::free(result->metadata_json);
    std::free(result->latents);
    result->samples = nullptr;
    result->metadata_json = nullptr;
    result->latents = nullptr;
    result->sample_count = 0;
    result->sample_rate = 0;
    result->latent_dim = 0;
    result->latent_frames = 0;
}

const char* scyllasband_status_message(ScyllasBandStatus status) {
    switch (status) {
        case SCYLLASBAND_STATUS_OK:
            return "ok";
        case SCYLLASBAND_STATUS_INVALID_ARGUMENT:
            return "invalid argument";
        case SCYLLASBAND_STATUS_NOT_IMPLEMENTED:
            return "not implemented";
        case SCYLLASBAND_STATUS_RUNTIME_ERROR:
            return "runtime error";
    }
    return "unknown status";
}
