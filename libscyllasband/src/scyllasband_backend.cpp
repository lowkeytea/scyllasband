#include "scyllasband_backend.h"

#include "scyllasband_bundle.h"
#include "scyllasband_boundary_policy.h"
#include "scyllasband_error.h"
#include "scyllasband_execution_plan.h"
#include "scyllasband_request.h"
#include "scyllasband_session_cache_policy.h"
#include "scyllasband_target_bucket_policy.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <limits>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace scyllasband_detail {
namespace {

char* duplicate_c_string(const char* value) {
    if (value == nullptr) {
        return nullptr;
    }
    const std::size_t size = std::strlen(value) + 1;
    auto* out = static_cast<char*>(std::malloc(size));
    if (out == nullptr) {
        return nullptr;
    }
    std::memcpy(out, value, size);
    return out;
}

using ScyllasBandSteadyClock = std::chrono::steady_clock;

constexpr int64_t kMinGpuSplitVectorLatentFrames = 64;

int64_t elapsed_ms_backend(
    ScyllasBandSteadyClock::time_point start,
    ScyllasBandSteadyClock::time_point end
) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

const char* litert_accelerator_name_backend(ScyllasBandLiteRtAccelerator accelerator) {
    switch (accelerator) {
        case SCYLLASBAND_LITERT_ACCELERATOR_AUTO: return "auto";
        case SCYLLASBAND_LITERT_ACCELERATOR_CPU: return "cpu";
        case SCYLLASBAND_LITERT_ACCELERATOR_GPU: return "gpu";
        case SCYLLASBAND_LITERT_ACCELERATOR_NPU: return "npu";
    }
    return "unknown";
}

ScyllasBandLiteRtAccelerator litert_frontend_accelerator_backend(ScyllasBandLiteRtAccelerator accelerator) {
    if (accelerator == SCYLLASBAND_LITERT_ACCELERATOR_AUTO ||
        accelerator == SCYLLASBAND_LITERT_ACCELERATOR_GPU ||
        accelerator == SCYLLASBAND_LITERT_ACCELERATOR_NPU) {
        return SCYLLASBAND_LITERT_ACCELERATOR_CPU;
    }
    return accelerator;
}

const char* litert_frontend_accelerator_policy_backend(ScyllasBandLiteRtAccelerator accelerator) {
    if (litert_frontend_accelerator_backend(accelerator) != accelerator) {
        return "cpu_fallback_for_frontend_control_graphs";
    }
    return "same_as_runtime";
}

ScyllasBandLiteRtAccelerator litert_vector_accelerator_backend(ScyllasBandLiteRtAccelerator accelerator) {
    if (accelerator == SCYLLASBAND_LITERT_ACCELERATOR_AUTO) {
        return SCYLLASBAND_LITERT_ACCELERATOR_CPU;
    }
    return accelerator;
}

const char* litert_vector_accelerator_policy_backend(ScyllasBandLiteRtAccelerator accelerator) {
    if (litert_vector_accelerator_backend(accelerator) != accelerator) {
        return "cpu_fallback_until_vector_gpu_validated";
    }
    return "same_as_runtime";
}

ScyllasBandLiteRtAccelerator litert_vocoder_accelerator_backend(ScyllasBandLiteRtAccelerator accelerator) {
    if (accelerator == SCYLLASBAND_LITERT_ACCELERATOR_AUTO ||
        accelerator == SCYLLASBAND_LITERT_ACCELERATOR_NPU) {
        return SCYLLASBAND_LITERT_ACCELERATOR_CPU;
    }
    return accelerator;
}

const char* litert_vocoder_accelerator_policy_backend(ScyllasBandLiteRtAccelerator accelerator) {
    if (accelerator == SCYLLASBAND_LITERT_ACCELERATOR_AUTO ||
        accelerator == SCYLLASBAND_LITERT_ACCELERATOR_NPU) {
        return "cpu_fallback_for_auto_or_npu_vocoder";
    }
    return "same_as_runtime";
}

bool litert_vocoder_can_fallback_to_cpu_backend(ScyllasBandLiteRtAccelerator accelerator) {
    return accelerator == SCYLLASBAND_LITERT_ACCELERATOR_AUTO ||
           accelerator == SCYLLASBAND_LITERT_ACCELERATOR_GPU ||
           accelerator == SCYLLASBAND_LITERT_ACCELERATOR_NPU;
}

bool is_silence_phone(const std::string& phone);

class NotLinkedBackend final : public ScyllasBandBackendEngine {
public:
    NotLinkedBackend(std::string backend_name, std::string bundle_dir, ScyllasBandBundleInfo bundle_info)
        : backend_name_(std::move(backend_name)),
          bundle_dir_(std::move(bundle_dir)),
          bundle_info_(std::move(bundle_info)) {}

    ScyllasBandStatus synthesize(
        const ScyllasBandSynthesisRequest& request,
        ScyllasBandSynthesisResult* out_result
    ) override {
        if (out_result == nullptr) {
            set_error("scyllasband backend synthesize: output result is required");
            return SCYLLASBAND_STATUS_INVALID_ARGUMENT;
        }
        out_result->samples = nullptr;
        out_result->sample_count = 0;
        out_result->sample_rate = 0;
        out_result->latents = nullptr;
        out_result->latent_dim = 0;
        out_result->latent_frames = 0;
        ScyllasBandResolvedRequest resolved_request;
        std::string execution_plan = "null";
        std::string prepared_inputs = "null";
        try {
            resolved_request = resolve_scyllasband_request_context(bundle_info_, request);
            if (!bundle_info_.component_artifacts.empty()) {
                execution_plan = execution_plan_json(build_scyllasband_duration_flow_plan(bundle_info_, resolved_request));
                if (resolved_request.has_explicit_phones) {
                    prepared_inputs = prepared_inputs_json(
                        prepare_scyllasband_duration_flow_inputs(bundle_info_, resolved_request, request)
                    );
                }
            }
        } catch (const std::exception& exc) {
            set_error(std::string("scyllasband backend synthesize: ") + exc.what());
            return SCYLLASBAND_STATUS_INVALID_ARGUMENT;
        }
        std::ostringstream metadata;
        metadata << "{\"status\":\"not_implemented\","
                 << "\"stage\":\"libscyllasband_backend_scaffold\","
                 << "\"backend\":\"" << backend_name_ << "\","
                 << "\"bundle_dir\":\"" << bundle_dir_ << "\","
                 << "\"bundle\":" << bundle_summary_json(bundle_info_) << ","
                 << "\"request\":" << request_context_json(resolved_request) << ","
                 << "\"execution_plan\":" << execution_plan << ","
                 << "\"prepared_inputs\":" << prepared_inputs << "}";
        out_result->metadata_json = duplicate_c_string(metadata.str().c_str());
        set_error("scyllasband backend synthesize: backend is not linked in this build");
        return SCYLLASBAND_STATUS_NOT_IMPLEMENTED;
    }

    ScyllasBandStatus estimate_latent_frames(
        const ScyllasBandSynthesisRequest& request,
        ScyllasBandDurationEstimate* out_estimate
    ) override {
        if (out_estimate == nullptr) {
            set_error("scyllasband backend estimate_latent_frames: output estimate is required");
            return SCYLLASBAND_STATUS_INVALID_ARGUMENT;
        }
        out_estimate->predicted_latent_frames = 0;
        out_estimate->fixed_latent_frames = 0;
        out_estimate->metadata_json.clear();
        ScyllasBandResolvedRequest resolved_request;
        std::string execution_plan = "null";
        std::string prepared_inputs = "null";
        try {
            resolved_request = resolve_scyllasband_request_context(bundle_info_, request);
            if (!bundle_info_.component_artifacts.empty()) {
                execution_plan = execution_plan_json(build_scyllasband_duration_flow_plan(bundle_info_, resolved_request));
                if (resolved_request.has_explicit_phones) {
                    prepared_inputs = prepared_inputs_json(
                        prepare_scyllasband_duration_flow_inputs(bundle_info_, resolved_request, request)
                    );
                }
            }
        } catch (const std::exception& exc) {
            set_error(std::string("scyllasband backend estimate_latent_frames: ") + exc.what());
            return SCYLLASBAND_STATUS_INVALID_ARGUMENT;
        }
        std::ostringstream metadata;
        metadata << "{\"status\":\"not_implemented\","
                 << "\"stage\":\"libscyllasband_backend_duration_preflight_scaffold\","
                 << "\"backend\":\"" << backend_name_ << "\","
                 << "\"bundle_dir\":\"" << bundle_dir_ << "\","
                 << "\"bundle\":" << bundle_summary_json(bundle_info_) << ","
                 << "\"request\":" << request_context_json(resolved_request) << ","
                 << "\"execution_plan\":" << execution_plan << ","
                 << "\"prepared_inputs\":" << prepared_inputs << ","
                 << "\"predicted_latent_frames\":0,"
                 << "\"fixed_latent_frames\":" << bundle_info_.latent_frames << "}";
        out_estimate->metadata_json = metadata.str();
        set_error("scyllasband backend estimate_latent_frames: backend is not linked in this build");
        return SCYLLASBAND_STATUS_NOT_IMPLEMENTED;
    }

    const char* name() const override {
        return backend_name_.c_str();
    }

private:
    std::string backend_name_;
    std::string bundle_dir_;
    ScyllasBandBundleInfo bundle_info_;
};

#if defined(SCYLLASBAND_WITH_LITERT) || defined(SCYLLASBAND_WITH_ONNXRUNTIME) || defined(SCYLLASBAND_WITH_COREAI)

struct ScyllasBandLiteRtInputStorage {
    std::vector<std::string> names;
    std::vector<int64_t> phone_shape;
    std::vector<int64_t> scalar_shape;
    std::vector<int64_t> affect_shape;
    std::vector<int64_t> affect_condition_mask_shape;
    std::vector<int64_t> reference_style_shape;
    std::vector<int64_t> reference_prosody_shape;
    std::vector<int64_t> identity_reference_shape;
    std::vector<int64_t> prosody_baseline_shape;
    std::vector<int64_t> prosody_delta_shape;
    std::vector<int64_t> prosody_feature_mask_shape;
    std::vector<int64_t> latent_shape;
    std::vector<int64_t> latent_frame_shape;
    std::vector<int64_t> hidden_shape;
    std::vector<int64_t> prefix_latents_shape;
    std::vector<int64_t> prefix_mask_shape;
    std::vector<int64_t> span_context_shape;
    std::vector<int64_t> span_context_hidden_shape;
    std::vector<int64_t> phone_ids;
    std::vector<uint8_t> phone_mask;
    std::vector<float> latents;
    std::vector<float> hidden;
    std::vector<float> time;
    std::vector<int64_t> expanded_phone_ids;
    std::vector<uint8_t> latent_mask;
    std::vector<int64_t> voice_id;
    std::vector<int64_t> language_id;
    std::vector<int64_t> emotion_id;
    std::vector<int64_t> boundary_before_id;
    std::vector<int64_t> boundary_after_id;
    std::vector<float> emotion_condition_mask;
    std::vector<float> affect_values;
    std::vector<float> affect_condition_mask;
    std::vector<float> reference_style;
    std::vector<float> reference_prosody;
    std::vector<float> reference_mask;
    std::vector<float> identity_reference;
    std::vector<float> identity_reference_mask;
    std::vector<float> prosody_baseline;
    std::vector<float> prosody_delta;
    std::vector<float> prosody_feature_mask;
    std::vector<float> prosody_confidence;
    std::vector<float> reference_condition_mask;
    std::vector<float> prefix_latents;
    std::vector<uint8_t> prefix_mask;
    std::vector<int64_t> span_context_phone_ids;
    std::vector<int64_t> span_context_segment_ids;
    std::vector<uint8_t> span_context_mask;
    std::vector<float> span_context_hidden;
    std::vector<ScyllasBandTensorView> views;
};

struct ScyllasBandDurationExpansionMetadata {
    std::vector<float> duration_values;
    std::vector<int64_t> predicted_durations;
    std::vector<int64_t> expanded_phone_ids;
    int64_t predicted_latent_frames = 0;
    float duration_scale = 1.0f;
};

struct ScyllasBandDurationPrediction {
    std::vector<float> values;
    std::string outputs_json;
};

struct ScyllasBandVectorVelocityResult {
    std::vector<float> values;
    std::string outputs_json;
};

struct ScyllasBandVectorPrefixResult {
    std::vector<float> values;
    std::vector<int64_t> shape;
    std::string outputs_json;
};

struct ScyllasBandSpanContextEncoding {
    std::vector<int64_t> phone_ids;
    std::vector<int64_t> segment_ids;
    std::vector<uint8_t> mask;
    int max_context_phones = 0;
    int previous_phone_count = 0;
    int target_phone_count = 0;
    int next_phone_count = 0;
    int context_phone_count = 0;
};

struct ScyllasBandSpanContextResult {
    std::vector<float> hidden;
    std::string outputs_json = "null";
    std::string metadata_json = "null";
};

struct ScyllasBandLatentSamplingResult {
    std::vector<float> latents;
    std::string last_vector_outputs_json = "null";
    int vector_calls = 0;
};

struct ScyllasBandVocoderResult {
    std::vector<float> audio;
    std::string outputs_json = "null";
};

void append_tensor_view(
    ScyllasBandLiteRtInputStorage& storage,
    std::string name,
    int32_t data_type,
    const std::vector<int64_t>& shape,
    const void* data,
    uint64_t byte_length
);

struct ScyllasBandPronunciationOverrideSpec {
    std::string word;
    std::vector<std::string> replace;
    std::vector<std::string> phones;
};

struct ScyllasBandPronunciationOverrideApplied {
    std::string word;
    std::vector<std::string> replace;
    std::vector<std::string> phones;
    int count = 0;
};

struct ScyllasBandG2PTerminalTailRepair {
    std::string word;
    std::vector<std::string> phones;
    std::vector<std::string> removed;
};

using ScyllasBandPronunciationOverrideMap = std::map<std::string, std::vector<ScyllasBandPronunciationOverrideSpec>>;

struct ScyllasBandG2PResult {
    std::vector<std::string> phones;
    std::vector<std::string> segments;
    std::vector<ScyllasBandPronunciationOverrideApplied> pronunciation_overrides;
    std::vector<ScyllasBandG2PTerminalTailRepair> terminal_tail_repairs;
    std::string prediction_text;
    std::string metadata_json = "null";
};


struct ReferenceNormalizationStats {
    std::vector<float> style_mean;
    std::vector<float> style_std;
    std::vector<float> prosody_mean;
    std::vector<float> prosody_std;
};


std::string json_escape_backend(const std::string& value) {
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

std::string string_vector_json_backend(const std::vector<std::string>& values, std::size_t limit = 0) {
    std::ostringstream out;
    out << "[";
    const std::size_t count = limit == 0 ? values.size() : std::min(values.size(), limit);
    for (std::size_t index = 0; index < count; ++index) {
        if (index > 0) {
            out << ",";
        }
        out << "\"" << json_escape_backend(values[index]) << "\"";
    }
    out << "]";
    return out.str();
}

std::string trim_backend(const std::string& value) {
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

std::string lowercase_ascii_backend(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::vector<std::string> utf8_codepoints(const std::string& value) {
    std::vector<std::string> out;
    for (std::size_t index = 0; index < value.size();) {
        const unsigned char ch = static_cast<unsigned char>(value[index]);
        std::size_t width = 1;
        if ((ch & 0x80U) == 0U) {
            width = 1;
        } else if ((ch & 0xE0U) == 0xC0U) {
            width = 2;
        } else if ((ch & 0xF0U) == 0xE0U) {
            width = 3;
        } else if ((ch & 0xF8U) == 0xF0U) {
            width = 4;
        }
        if (index + width > value.size()) {
            width = 1;
        }
        out.push_back(value.substr(index, width));
        index += width;
    }
    return out;
}

std::string resolve_g2p_language(const ScyllasBandBundleInfo& bundle, const std::string& language) {
    const std::string normalized = lowercase_ascii_backend(language);
    const auto it = bundle.g2p_language_map.find(normalized);
    if (it != bundle.g2p_language_map.end()) {
        return it->second;
    }
    return normalized;
}

uint32_t utf8_codepoint_value(const std::string& codepoint) {
    if (codepoint.empty()) {
        return 0;
    }
    const unsigned char b0 = static_cast<unsigned char>(codepoint[0]);
    if ((b0 & 0x80U) == 0U) {
        return b0;
    }
    if ((b0 & 0xE0U) == 0xC0U && codepoint.size() >= 2) {
        const unsigned char b1 = static_cast<unsigned char>(codepoint[1]);
        return ((b0 & 0x1FU) << 6U) | (b1 & 0x3FU);
    }
    if ((b0 & 0xF0U) == 0xE0U && codepoint.size() >= 3) {
        const unsigned char b1 = static_cast<unsigned char>(codepoint[1]);
        const unsigned char b2 = static_cast<unsigned char>(codepoint[2]);
        return ((b0 & 0x0FU) << 12U) | ((b1 & 0x3FU) << 6U) | (b2 & 0x3FU);
    }
    if ((b0 & 0xF8U) == 0xF0U && codepoint.size() >= 4) {
        const unsigned char b1 = static_cast<unsigned char>(codepoint[1]);
        const unsigned char b2 = static_cast<unsigned char>(codepoint[2]);
        const unsigned char b3 = static_cast<unsigned char>(codepoint[3]);
        return ((b0 & 0x07U) << 18U) | ((b1 & 0x3FU) << 12U) |
               ((b2 & 0x3FU) << 6U) | (b3 & 0x3FU);
    }
    return b0;
}

bool is_unicode_space_or_punctuation_codepoint(uint32_t value) {
    if (value <= 0x7FU) {
        return std::isspace(static_cast<unsigned char>(value)) ||
               std::ispunct(static_cast<unsigned char>(value));
    }
    if (value == 0x00A0U || value == 0x00A1U || value == 0x00ADU || value == 0x00BFU) {
        return true;
    }
    if (value >= 0x2000U && value <= 0x206FU) {
        return true;
    }
    if (value >= 0x2E00U && value <= 0x2E7FU) {
        return true;
    }
    if (value >= 0x3000U && value <= 0x303FU) {
        return true;
    }
    if (value >= 0xFE10U && value <= 0xFE6FU) {
        return true;
    }
    if (value >= 0xFF00U && value <= 0xFF65U) {
        return true;
    }
    return false;
}

bool is_unicode_space_or_punctuation(const std::string& symbol) {
    if (symbol.empty()) {
        return true;
    }
    for (const std::string& codepoint : utf8_codepoints(symbol)) {
        if (!is_unicode_space_or_punctuation_codepoint(utf8_codepoint_value(codepoint))) {
            return false;
        }
    }
    return true;
}

bool skip_g2p_output_symbol(const std::string& symbol) {
    if (symbol.empty() || symbol == "_" || symbol[0] == '<') {
        return true;
    }
    return is_unicode_space_or_punctuation(symbol);
}

std::string punctuation_phone_token_backend(const std::string& segment) {
    const std::string value = trim_backend(segment);
    if (value.empty()) {
        return {};
    }
    std::size_t end = value.size();
    while (end > 0 && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    if (end == 0) {
        return {};
    }
    std::size_t start = end;
    while (start > 0) {
        const unsigned char ch = static_cast<unsigned char>(value[start - 1]);
        if (ch == '?' || ch == '!' || ch == '.' || ch == ',' || ch == ';' || ch == ':' || ch == '-') {
            --start;
        } else {
            break;
        }
    }
    const std::string run = value.substr(start, end - start);
    if (run.empty()) {
        return {};
    }
    if (run.find('?') != std::string::npos) {
        return "<end_question>";
    }
    if (run.find('!') != std::string::npos) {
        return "<end_exclaim>";
    }
    if (run.find("...") != std::string::npos) {
        return "<ellipsis>";
    }
    if (run.find('.') != std::string::npos) {
        return "<end_stmt>";
    }
    if (run.find(',') != std::string::npos) {
        return "<pause_comma>";
    }
    if (run.find(';') != std::string::npos) {
        return "<pause_semicolon>";
    }
    if (run.find(':') != std::string::npos) {
        return "<pause_colon>";
    }
    if (run.find('-') != std::string::npos) {
        return "<pause_dash>";
    }
    return {};
}

bool is_g2p_boundary_phone(const std::string& phone) {
    return phone == "<pause_comma>" ||
           phone == "<pause_semicolon>" ||
           phone == "<pause_colon>" ||
           phone == "<pause_dash>" ||
           phone == "<ellipsis>" ||
           phone == "<end_stmt>" ||
           phone == "<end_question>" ||
           phone == "<end_exclaim>" ||
           phone == "<ctx_sentence_start>" ||
           phone == "<ctx_continuation>" ||
           phone == "<ctx_sentence_end>" ||
           phone == "<ctx_chunk_continue>";
}

bool ends_in_g2p_boundary_phone(const std::vector<std::string>& phones) {
    for (auto it = phones.rbegin(); it != phones.rend(); ++it) {
        if (is_silence_phone(*it)) {
            continue;
        }
        return is_g2p_boundary_phone(*it);
    }
    return false;
}

bool g2p_segment_boundary_char(char ch) {
    return ch == '.' || ch == '?' || ch == '!' || ch == ';' || ch == ':';
}

std::size_t utf8_codepoint_count(const std::string& value) {
    return utf8_codepoints(value).size();
}

std::vector<std::string> split_long_g2p_segment(
    const std::string& text,
    std::size_t max_chars
) {
    const std::string value = trim_backend(text);
    if (value.empty()) {
        return {};
    }
    if (max_chars == 0 || utf8_codepoint_count(value) <= max_chars) {
        return {value};
    }

    // Match the Python G2P phrase planner: keep words intact and start a new
    // phrase before adding a word would exceed the model's text-token budget.
    // A single word longer than the budget is deliberately left intact so the
    // caller receives the same explicit fixed-shape error as the Python path.
    std::vector<std::string> pieces;
    std::istringstream words(value);
    std::string current;
    std::string word;
    std::size_t current_chars = 0;
    while (words >> word) {
        const std::size_t word_chars = utf8_codepoint_count(word);
        const std::size_t candidate_chars = current.empty()
            ? word_chars
            : current_chars + 1 + word_chars;
        if (!current.empty() && candidate_chars > max_chars) {
            pieces.push_back(current);
            current = word;
            current_chars = word_chars;
        } else {
            if (!current.empty()) {
                current.push_back(' ');
            }
            current += word;
            current_chars = candidate_chars;
        }
    }
    if (!current.empty()) {
        pieces.push_back(std::move(current));
    }
    return pieces;
}

std::vector<std::string> split_g2p_segments(const std::string& text, std::size_t max_chars = 140) {
    std::vector<std::string> punctuation_segments;
    std::string current;
    auto flush_current = [&]() {
        std::string trimmed = trim_backend(current);
        if (!trimmed.empty()) {
            punctuation_segments.push_back(trimmed);
        }
        current.clear();
    };
    for (std::size_t index = 0; index < text.size(); ++index) {
        const char ch = text[index];
        current.push_back(ch);
        if (g2p_segment_boundary_char(ch)) {
            while (index + 1 < text.size()) {
                const char next = text[index + 1];
                if (g2p_segment_boundary_char(next)) {
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
            flush_current();
        }
    }
    flush_current();
    if (punctuation_segments.empty()) {
        const std::string trimmed = trim_backend(text);
        if (!trimmed.empty()) {
            punctuation_segments.push_back(trimmed);
        }
    }

    std::vector<std::string> segments;
    for (const std::string& segment : punctuation_segments) {
        std::vector<std::string> pieces = split_long_g2p_segment(segment, max_chars);
        segments.insert(
            segments.end(),
            std::make_move_iterator(pieces.begin()),
            std::make_move_iterator(pieces.end())
        );
    }
    return segments;
}

std::vector<int64_t> encode_g2p_text(
    const ScyllasBandBundleInfo& bundle,
    const std::string& text,
    const std::string& language
) {
    if (bundle.g2p_text_to_id.empty()) {
        throw std::runtime_error("Bundle has no G2P tokenizer text_symbols asset loaded");
    }
    const std::string language_token = "<" + language + ">";
    const auto lang_it = bundle.g2p_text_to_id.find(language_token);
    if (lang_it == bundle.g2p_text_to_id.end()) {
        throw std::runtime_error("G2P language '" + language + "' is not supported by this bundle");
    }
    std::string work = bundle.g2p_lowercase ? lowercase_ascii_backend(text) : text;
    std::vector<int64_t> sequence;
    sequence.push_back(static_cast<int64_t>(lang_it->second));
    int emitted_chars = 0;
    for (const std::string& codepoint : utf8_codepoints(work)) {
        const auto token_it = bundle.g2p_text_to_id.find(codepoint);
        if (token_it == bundle.g2p_text_to_id.end()) {
            continue;
        }
        for (int repeat = 0; repeat < std::max(1, bundle.g2p_char_repeats); ++repeat) {
            sequence.push_back(static_cast<int64_t>(token_it->second));
        }
        ++emitted_chars;
    }
    const auto end_it = bundle.g2p_text_to_id.find("<end>");
    sequence.push_back(end_it == bundle.g2p_text_to_id.end() ? bundle.g2p_text_pad_index : end_it->second);
    if (emitted_chars <= 0) {
        throw std::runtime_error("G2P text has no characters supported by this bundle");
    }
    if (sequence.size() > static_cast<std::size_t>(bundle.g2p_text_tokens)) {
        throw std::runtime_error(
            "G2P input encodes to " + std::to_string(sequence.size()) +
            " tokens, but this bundle supports at most " + std::to_string(bundle.g2p_text_tokens)
        );
    }
    std::vector<int64_t> encoded(static_cast<std::size_t>(bundle.g2p_text_tokens), bundle.g2p_text_pad_index);
    std::copy(sequence.begin(), sequence.end(), encoded.begin());
    return encoded;
}

std::vector<std::string> decode_g2p_logits(
    const ScyllasBandBundleInfo& bundle,
    const ScyllasBandOwnedTensor& tensor
) {
    if (tensor.data_type != SCYLLASBAND_TENSOR_FLOAT32 || tensor.data == nullptr) {
        throw std::runtime_error("g2p output_0 must be float32 logits");
    }
    if (bundle.g2p_phoneme_symbols.empty()) {
        throw std::runtime_error("Bundle has no G2P tokenizer phoneme_symbols asset loaded");
    }
    const std::size_t total_values = tensor.byte_length / sizeof(float);
    if (tensor.rank <= 0 || total_values == 0) {
        throw std::runtime_error("g2p output_0 has no logits");
    }
    std::size_t class_count = 0;
    if (tensor.rank >= 2 && tensor.shape != nullptr && tensor.shape[tensor.rank - 1] > 0) {
        class_count = static_cast<std::size_t>(tensor.shape[tensor.rank - 1]);
    }
    if (class_count == 0 || total_values % class_count != 0) {
        throw std::runtime_error("g2p output_0 shape does not expose a valid class dimension");
    }
    const std::size_t frame_count = total_values / class_count;
    const auto* values = static_cast<const float*>(tensor.data);
    std::vector<std::string> phones;
    int64_t previous = std::numeric_limits<int64_t>::min();
    for (std::size_t frame = 0; frame < frame_count; ++frame) {
        const float* row = values + frame * class_count;
        std::size_t argmax = 0;
        float best = row[0];
        for (std::size_t class_index = 1; class_index < class_count; ++class_index) {
            if (row[class_index] > best) {
                best = row[class_index];
                argmax = class_index;
            }
        }
        const int64_t token_id = static_cast<int64_t>(argmax);
        if (token_id == previous) {
            continue;
        }
        previous = token_id;
        if (token_id == bundle.g2p_phoneme_pad_index) {
            continue;
        }
        if (token_id == bundle.g2p_phoneme_end_index) {
            break;
        }
        const auto symbol_it = bundle.g2p_phoneme_symbols.find(static_cast<int>(token_id));
        if (symbol_it == bundle.g2p_phoneme_symbols.end()) {
            continue;
        }
        if (skip_g2p_output_symbol(symbol_it->second)) {
            continue;
        }
        phones.push_back(symbol_it->second);
    }
    return phones;
}

std::vector<std::string> run_g2p_segment(
    ScyllasBandLiteRtSession* session,
    const ScyllasBandBundleInfo& bundle,
    const std::string& segment,
    const std::string& language
) {
    std::vector<int64_t> encoded = encode_g2p_text(bundle, segment, language);
    ScyllasBandLiteRtInputStorage storage;
    storage.names.reserve(1);
    storage.views.reserve(1);
    storage.phone_shape = {1, static_cast<int64_t>(encoded.size())};
    storage.phone_ids = std::move(encoded);
    append_tensor_view(storage, "args_0", SCYLLASBAND_TENSOR_INT64, storage.phone_shape,
                       storage.phone_ids.data(), storage.phone_ids.size() * sizeof(int64_t));

    ScyllasBandOwnedTensor* outputs = nullptr;
    int32_t output_count = 0;
    const int run_status = scyllasband_litert_session_run(
        session,
        "serving_default",
        storage.views.data(),
        static_cast<int32_t>(storage.views.size()),
        &outputs,
        &output_count
    );
    if (run_status != 0) {
        const char* detail = last_error();
        throw std::runtime_error(std::string("g2p failed: ") + (detail == nullptr ? "" : detail));
    }
    if (outputs == nullptr || output_count <= 0) {
        throw std::runtime_error("g2p returned no tensors");
    }
    std::vector<std::string> decoded;
    try {
        decoded = decode_g2p_logits(bundle, outputs[0]);
    } catch (...) {
        scyllasband_tensors_destroy(outputs, output_count);
        throw;
    }
    scyllasband_tensors_destroy(outputs, output_count);
    return decoded;
}

std::string terminal_g2p_word(const std::string& text) {
    std::string value = trim_backend(text);
    const std::string trailing = ".,!?;:\"')]}";
    while (!value.empty() && trailing.find(value.back()) != std::string::npos) {
        value.pop_back();
        value = trim_backend(value);
    }
    const std::size_t separator = value.find_last_of(" \t\r\n");
    std::string word = separator == std::string::npos ? value : value.substr(separator + 1);
    const std::string leading = "\"'([{";
    while (!word.empty() && leading.find(word.front()) != std::string::npos) {
        word.erase(word.begin());
    }
    return lowercase_ascii_backend(word);
}

std::vector<std::string> trim_g2p_terminal_artifacts(
    std::vector<std::string>& phrase_phones,
    const std::vector<std::string>& isolated_word_phones,
    std::size_t max_removed_phones = 2
) {
    if (phrase_phones.empty() || isolated_word_phones.empty() || max_removed_phones == 0) {
        return {};
    }
    for (std::size_t removed_count = 1; removed_count <= max_removed_phones; ++removed_count) {
        if (phrase_phones.size() < isolated_word_phones.size() + removed_count) {
            continue;
        }
        const std::size_t start = phrase_phones.size() - isolated_word_phones.size() - removed_count;
        if (!std::equal(
                isolated_word_phones.begin(),
                isolated_word_phones.end(),
                phrase_phones.begin() + static_cast<std::ptrdiff_t>(start))) {
            continue;
        }
        const std::size_t end = start + isolated_word_phones.size();
        std::vector<std::string> removed(phrase_phones.begin() + static_cast<std::ptrdiff_t>(end), phrase_phones.end());
        phrase_phones.erase(phrase_phones.begin() + static_cast<std::ptrdiff_t>(end), phrase_phones.end());
        return removed;
    }
    return {};
}

std::string join_phone_tokens(const std::vector<std::string>& phones) {
    std::ostringstream out;
    for (std::size_t index = 0; index < phones.size(); ++index) {
        if (index > 0) {
            out << ' ';
        }
        out << phones[index];
    }
    return out.str();
}


uint16_t read_le16(const std::vector<uint8_t>& data, std::size_t offset) {
    if (offset + 2 > data.size()) {
        throw std::runtime_error("unexpected EOF while reading uint16");
    }
    return static_cast<uint16_t>(data[offset]) |
           static_cast<uint16_t>(static_cast<uint16_t>(data[offset + 1]) << 8);
}

uint32_t read_le32(const std::vector<uint8_t>& data, std::size_t offset) {
    if (offset + 4 > data.size()) {
        throw std::runtime_error("unexpected EOF while reading uint32");
    }
    return static_cast<uint32_t>(data[offset]) |
           (static_cast<uint32_t>(data[offset + 1]) << 8) |
           (static_cast<uint32_t>(data[offset + 2]) << 16) |
           (static_cast<uint32_t>(data[offset + 3]) << 24);
}

uint64_t read_le64(const std::vector<uint8_t>& data, std::size_t offset) {
    if (offset + 8 > data.size()) {
        throw std::runtime_error("unexpected EOF while reading uint64");
    }
    uint64_t value = 0;
    for (int byte = 0; byte < 8; ++byte) {
        value |= static_cast<uint64_t>(data[offset + static_cast<std::size_t>(byte)]) << (8 * byte);
    }
    return value;
}

std::vector<uint8_t> read_binary_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("unable to open reference pack: " + path.string());
    }
    input.seekg(0, std::ios::end);
    const std::streamoff size = input.tellg();
    if (size < 0) {
        throw std::runtime_error("unable to size reference pack: " + path.string());
    }
    input.seekg(0, std::ios::beg);
    std::vector<uint8_t> bytes(static_cast<std::size_t>(size));
    if (!bytes.empty()) {
        input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    return bytes;
}

float finite_or_zero(float value) {
    return std::isfinite(value) ? value : 0.0f;
}

std::vector<float> parse_npy_float_values(const std::vector<uint8_t>& payload) {
    if (payload.size() < 10 || payload[0] != 0x93 || payload[1] != 'N' || payload[2] != 'U' ||
        payload[3] != 'M' || payload[4] != 'P' || payload[5] != 'Y') {
        throw std::runtime_error("reference pack entry is not an NPY array");
    }
    const uint8_t major = payload[6];
    std::size_t header_len = 0;
    std::size_t data_offset = 0;
    if (major == 1) {
        header_len = read_le16(payload, 8);
        data_offset = 10 + header_len;
    } else if (major == 2 || major == 3) {
        header_len = read_le32(payload, 8);
        data_offset = 12 + header_len;
    } else {
        throw std::runtime_error("unsupported NPY version in reference pack");
    }
    if (data_offset > payload.size()) {
        throw std::runtime_error("invalid NPY header length in reference pack");
    }
    const std::string header(reinterpret_cast<const char*>(payload.data() + data_offset - header_len), header_len);
    const bool is_float32 = header.find("'descr': '<f4'") != std::string::npos || header.find("\"descr\": \"<f4\"") != std::string::npos;
    const bool is_float64 = header.find("'descr': '<f8'") != std::string::npos || header.find("\"descr\": \"<f8\"") != std::string::npos;
    const bool is_int64 = header.find("'descr': '<i8'") != std::string::npos || header.find("\"descr\": \"<i8\"") != std::string::npos;
    const bool is_int32 = header.find("'descr': '<i4'") != std::string::npos || header.find("\"descr\": \"<i4\"") != std::string::npos;
    const bool is_int8 = header.find("'descr': '|i1'") != std::string::npos || header.find("\"descr\": \"|i1\"") != std::string::npos;
    const bool fortran = header.find("'fortran_order': True") != std::string::npos || header.find("\"fortran_order\": true") != std::string::npos;
    if (fortran) {
        throw std::runtime_error("fortran-order NPY arrays are not supported in reference packs");
    }
    std::size_t element_size = 0;
    if (is_float32 || is_int32) {
        element_size = 4;
    } else if (is_float64 || is_int64) {
        element_size = 8;
    } else if (is_int8) {
        element_size = 1;
    } else {
        throw std::runtime_error("unsupported NPY dtype in reference pack");
    }
    const std::size_t available = payload.size() - data_offset;
    if (available % element_size != 0) {
        throw std::runtime_error("NPY payload byte count is not divisible by element size");
    }
    const std::size_t count = available / element_size;
    std::vector<float> values;
    values.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const std::size_t offset = data_offset + index * element_size;
        if (is_float32) {
            float value = 0.0f;
            std::memcpy(&value, payload.data() + offset, sizeof(float));
            values.push_back(finite_or_zero(value));
        } else if (is_float64) {
            double value = 0.0;
            std::memcpy(&value, payload.data() + offset, sizeof(double));
            values.push_back(finite_or_zero(static_cast<float>(value)));
        } else if (is_int64) {
            values.push_back(static_cast<float>(static_cast<int64_t>(read_le64(payload, offset))));
        } else if (is_int8) {
            values.push_back(static_cast<float>(static_cast<int8_t>(payload[offset])));
        } else if (is_int32) {
            values.push_back(static_cast<float>(static_cast<int32_t>(read_le32(payload, offset))));
        }
    }
    return values;
}
struct ScyllasBandNpyArray {
    std::vector<std::size_t> shape;
    std::vector<float> values;
    std::vector<std::string> strings;
};

using ScyllasBandNpyPack = std::map<std::string, ScyllasBandNpyArray>;

std::size_t npy_data_offset(
    const std::vector<uint8_t>& payload,
    std::string& header
) {
    if (payload.size() < 10 || payload[0] != 0x93 || payload[1] != 'N' || payload[2] != 'U' ||
        payload[3] != 'M' || payload[4] != 'P' || payload[5] != 'Y') {
        throw std::runtime_error("reference pack entry is not an NPY array");
    }
    const uint8_t major = payload[6];
    std::size_t header_len = 0;
    std::size_t data_offset = 0;
    if (major == 1) {
        header_len = read_le16(payload, 8);
        data_offset = 10 + header_len;
    } else if (major == 2 || major == 3) {
        header_len = read_le32(payload, 8);
        data_offset = 12 + header_len;
    } else {
        throw std::runtime_error("unsupported NPY version in reference pack");
    }
    if (data_offset > payload.size() || data_offset < header_len) {
        throw std::runtime_error("invalid NPY header length in reference pack");
    }
    header.assign(
        reinterpret_cast<const char*>(payload.data() + data_offset - header_len),
        header_len
    );
    return data_offset;
}

std::string npy_descriptor(const std::string& header) {
    const std::size_t key = header.find("descr");
    const std::size_t colon = key == std::string::npos ? key : header.find(':', key);
    const std::size_t quote = colon == std::string::npos
        ? colon
        : header.find_first_of("'\"", colon + 1);
    if (quote == std::string::npos) {
        throw std::runtime_error("NPY header lacks a dtype descriptor");
    }
    const std::size_t end = header.find(header[quote], quote + 1);
    if (end == std::string::npos) {
        throw std::runtime_error("NPY dtype descriptor is unterminated");
    }
    return header.substr(quote + 1, end - quote - 1);
}

std::vector<std::size_t> npy_shape(const std::string& header) {
    const std::size_t key = header.find("shape");
    const std::size_t begin = key == std::string::npos ? key : header.find('(', key);
    const std::size_t end = begin == std::string::npos ? begin : header.find(')', begin + 1);
    if (begin == std::string::npos || end == std::string::npos) {
        throw std::runtime_error("NPY header lacks a valid shape");
    }
    std::vector<std::size_t> shape;
    std::size_t cursor = begin + 1;
    while (cursor < end) {
        while (cursor < end && !std::isdigit(static_cast<unsigned char>(header[cursor]))) {
            ++cursor;
        }
        if (cursor >= end) {
            break;
        }
        std::size_t value = 0;
        while (cursor < end && std::isdigit(static_cast<unsigned char>(header[cursor]))) {
            value = value * 10U + static_cast<std::size_t>(header[cursor] - '0');
            ++cursor;
        }
        shape.push_back(value);
    }
    return shape;
}

std::size_t npy_element_count(
    const std::vector<std::size_t>& shape,
    std::size_t available,
    std::size_t element_size
) {
    if (element_size == 0 || available % element_size != 0) {
        throw std::runtime_error("NPY payload byte count is not divisible by element size");
    }
    const std::size_t payload_count = available / element_size;
    if (shape.empty()) {
        if (payload_count != 1) {
            throw std::runtime_error("scalar NPY payload must contain exactly one element");
        }
        return 1;
    }
    std::size_t shape_count = 1;
    for (std::size_t dim : shape) {
        if (dim != 0 && shape_count > std::numeric_limits<std::size_t>::max() / dim) {
            throw std::runtime_error("NPY shape element count overflow");
        }
        shape_count *= dim;
    }
    if (shape_count != payload_count) {
        throw std::runtime_error("NPY shape does not match payload element count");
    }
    return shape_count;
}

void append_npy_utf8_codepoint(std::string& out, uint32_t codepoint) {
    if (codepoint <= 0x7fU) {
        out.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7ffU) {
        out.push_back(static_cast<char>(0xc0U | (codepoint >> 6)));
        out.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    } else if (codepoint <= 0xffffU) {
        out.push_back(static_cast<char>(0xe0U | (codepoint >> 12)));
        out.push_back(static_cast<char>(0x80U | ((codepoint >> 6) & 0x3fU)));
        out.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    } else if (codepoint <= 0x10ffffU) {
        out.push_back(static_cast<char>(0xf0U | (codepoint >> 18)));
        out.push_back(static_cast<char>(0x80U | ((codepoint >> 12) & 0x3fU)));
        out.push_back(static_cast<char>(0x80U | ((codepoint >> 6) & 0x3fU)));
        out.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    }
}

ScyllasBandNpyArray parse_npy_array(const std::vector<uint8_t>& payload) {
    std::string header;
    const std::size_t data_offset = npy_data_offset(payload, header);
    const std::string descriptor = npy_descriptor(header);
    const bool fortran = header.find("'fortran_order': True") != std::string::npos ||
                         header.find("\"fortran_order\": true") != std::string::npos;
    if (fortran) {
        throw std::runtime_error("fortran-order NPY arrays are not supported in reference packs");
    }

    ScyllasBandNpyArray array;
    array.shape = npy_shape(header);
    if (descriptor.rfind("<U", 0) == 0) {
        const std::size_t codepoints = static_cast<std::size_t>(
            std::stoul(descriptor.substr(2))
        );
        const std::size_t element_size = codepoints * sizeof(uint32_t);
        const std::size_t count = npy_element_count(
            array.shape, payload.size() - data_offset, element_size
        );
        array.strings.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            std::string value;
            const std::size_t offset = data_offset + index * element_size;
            for (std::size_t char_index = 0; char_index < codepoints; ++char_index) {
                const uint32_t codepoint = read_le32(
                    payload, offset + char_index * sizeof(uint32_t)
                );
                if (codepoint == 0U) {
                    break;
                }
                append_npy_utf8_codepoint(value, codepoint);
            }
            array.strings.push_back(std::move(value));
        }
        return array;
    }

    array.values = parse_npy_float_values(payload);
    std::size_t expected = 1U;
    for (std::size_t dim : array.shape) {
        expected *= dim;
    }
    if (array.values.size() != expected) {
        throw std::runtime_error("NPY numeric shape does not match decoded values");
    }
    return array;
}

ScyllasBandNpyPack load_uncompressed_npz(const std::filesystem::path& path) {
    std::vector<uint8_t> bytes = read_binary_file(path);
    ScyllasBandNpyPack arrays;
    std::size_t offset = 0;
    while (offset + 30 <= bytes.size()) {
        const uint32_t signature = read_le32(bytes, offset);
        if (signature == 0x02014b50U || signature == 0x06054b50U) {
            break;
        }
        if (signature != 0x04034b50U) {
            break;
        }
        const uint16_t flags = read_le16(bytes, offset + 6);
        const uint16_t method = read_le16(bytes, offset + 8);
        const uint32_t compressed_size32 = read_le32(bytes, offset + 18);
        const uint32_t uncompressed_size32 = read_le32(bytes, offset + 22);
        const uint16_t filename_len = read_le16(bytes, offset + 26);
        const uint16_t extra_len = read_le16(bytes, offset + 28);
        if ((flags & 0x0008U) != 0U) {
            throw std::runtime_error("reference pack uses ZIP data descriptors, which are not supported");
        }
        if (method != 0) {
            throw std::runtime_error("reference pack entries must be uncompressed ZIP_STORED arrays");
        }
        const std::size_t name_start = offset + 30;
        const std::size_t extra_start = name_start + filename_len;
        const std::size_t extra_end = extra_start + extra_len;
        if (extra_end > bytes.size() || extra_end < extra_start) {
            throw std::runtime_error("reference pack local file header exceeds archive size");
        }

        uint64_t compressed_size = compressed_size32;
        uint64_t uncompressed_size = uncompressed_size32;
        if (compressed_size32 == 0xffffffffU || uncompressed_size32 == 0xffffffffU) {
            bool found_zip64 = false;
            std::size_t extra_offset = extra_start;
            while (extra_offset + 4 <= extra_end) {
                const uint16_t header_id = read_le16(bytes, extra_offset);
                const uint16_t data_size = read_le16(bytes, extra_offset + 2);
                extra_offset += 4;
                const std::size_t field_end = extra_offset + data_size;
                if (field_end > extra_end || field_end < extra_offset) {
                    throw std::runtime_error("reference pack ZIP extra field exceeds local header");
                }
                if (header_id == 0x0001U) {
                    std::size_t cursor = extra_offset;
                    if (uncompressed_size32 == 0xffffffffU) {
                        if (cursor + 8 > field_end) {
                            throw std::runtime_error("reference pack ZIP64 extra field lacks uncompressed size");
                        }
                        uncompressed_size = read_le64(bytes, cursor);
                        cursor += 8;
                    }
                    if (compressed_size32 == 0xffffffffU) {
                        if (cursor + 8 > field_end) {
                            throw std::runtime_error("reference pack ZIP64 extra field lacks compressed size");
                        }
                        compressed_size = read_le64(bytes, cursor);
                        cursor += 8;
                    }
                    found_zip64 = true;
                    break;
                }
                extra_offset = field_end;
            }
            if (!found_zip64) {
                throw std::runtime_error("reference pack ZIP64 size extra field is missing");
            }
        }
        if (compressed_size != uncompressed_size) {
            throw std::runtime_error("reference pack ZIP_STORED entry has mismatched stored sizes");
        }
        const std::size_t data_start = extra_end;
        if (compressed_size > static_cast<uint64_t>(bytes.size() - data_start)) {
            throw std::runtime_error("reference pack local file entry exceeds archive size");
        }
        const std::size_t data_end = data_start + static_cast<std::size_t>(compressed_size);
        std::string name(reinterpret_cast<const char*>(bytes.data() + name_start), filename_len);
        std::vector<uint8_t> payload(bytes.begin() + static_cast<std::ptrdiff_t>(data_start), bytes.begin() + static_cast<std::ptrdiff_t>(data_end));
        if (name.size() > 4 && name.substr(name.size() - 4) == ".npy") {
            name = name.substr(0, name.size() - 4);
        }
        arrays[name] = parse_npy_array(payload);
        offset = data_end;
    }
    if (arrays.empty()) {
        throw std::runtime_error("reference pack contained no readable uncompressed NPY arrays: " + path.string());
    }
    return arrays;
}

std::vector<float> array_or_zeros(
    const ScyllasBandNpyPack& arrays,
    const std::string& key,
    int dim
) {
    std::vector<float> out(static_cast<std::size_t>(std::max(0, dim)), 0.0f);
    const auto it = arrays.find(key);
    if (it == arrays.end()) {
        return out;
    }
    const std::size_t count = std::min(out.size(), it->second.values.size());
    for (std::size_t index = 0; index < count; ++index) {
        out[index] = finite_or_zero(it->second.values[index]);
    }
    return out;
}

float mask_value(const ScyllasBandNpyPack& arrays, const std::string& key) {
    const auto it = arrays.find(key);
    if (it == arrays.end() || it->second.values.empty()) {
        return 0.0f;
    }
    return it->second.values[0] > 0.0f ? 1.0f : 0.0f;
}
struct ScyllasBandReferenceRouteV4 {
    std::vector<float> identity;
    std::vector<float> prosody_baseline;
    std::vector<float> prosody_delta;
    std::vector<float> prosody_feature_mask;
    float prosody_confidence = 0.0f;
    int baseline_index = -1;
    int prototype_index = -1;
    std::string route_kind = "global_affect_only";
};

const ScyllasBandNpyArray& required_npy_array(
    const ScyllasBandNpyPack& pack,
    const std::string& key
) {
    const auto it = pack.find(key);
    if (it == pack.end()) {
        throw std::runtime_error("schema-v4 reference pack lacks array '" + key + "'");
    }
    return it->second;
}

const std::vector<float>& required_npy_values(
    const ScyllasBandNpyPack& pack,
    const std::string& key
) {
    const auto& array = required_npy_array(pack, key);
    if (array.values.empty()) {
        throw std::runtime_error("schema-v4 reference array '" + key + "' is not numeric");
    }
    return array.values;
}

const std::vector<std::string>& required_npy_strings(
    const ScyllasBandNpyPack& pack,
    const std::string& key
) {
    const auto& array = required_npy_array(pack, key);
    if (array.strings.empty()) {
        throw std::runtime_error("schema-v4 reference array '" + key + "' is not text");
    }
    return array.strings;
}

void require_npy_shape(
    const ScyllasBandNpyArray& array,
    const std::vector<std::size_t>& expected,
    const std::string& key
) {
    if (array.shape != expected) {
        throw std::runtime_error("schema-v4 reference array '" + key + "' has an invalid shape");
    }
}

std::vector<float> npy_numeric_row(
    const ScyllasBandNpyPack& pack,
    const std::string& key,
    std::size_t row,
    std::size_t width
) {
    const auto& array = required_npy_array(pack, key);
    if (array.shape.size() != 2 || array.shape[1] != width || row >= array.shape[0]) {
        throw std::runtime_error("schema-v4 reference array '" + key + "' has an invalid row shape");
    }
    const std::size_t begin = row * width;
    if (begin + width > array.values.size()) {
        throw std::runtime_error("schema-v4 reference array '" + key + "' row exceeds payload");
    }
    return std::vector<float>(
        array.values.begin() + static_cast<std::ptrdiff_t>(begin),
        array.values.begin() + static_cast<std::ptrdiff_t>(begin + width)
    );
}

ScyllasBandReferenceRouteV4 route_reference_pack_v4_native(
    const ScyllasBandNpyPack& pack,
    const std::string& language,
    const std::vector<float>& affect_values,
    const std::vector<float>& affect_mask
) {
    const auto& schema = required_npy_strings(pack, "schema_version");
    if (schema.size() != 1 || schema[0] != "scyllasband_reference_pack_v4") {
        throw std::runtime_error("reference pack does not implement scyllasband_reference_pack_v4");
    }
    static const std::vector<std::string> expected_axes = {
        "calm", "joy", "anger", "sadness", "whisper",
    };
    if (required_npy_strings(pack, "affect_axis_order") != expected_axes) {
        throw std::runtime_error("schema-v4 reference pack has an invalid affect axis order");
    }
    if (affect_values.size() != 5 || affect_mask.size() != 5) {
        throw std::runtime_error("schema-v4 routing requires five affect values and masks");
    }

    const auto& baseline_languages = required_npy_strings(pack, "baseline_languages");
    std::size_t baseline_index = baseline_languages.size();
    for (std::size_t index = 0; index < baseline_languages.size(); ++index) {
        if (baseline_languages[index] == language) {
            baseline_index = index;
            break;
        }
    }
    if (baseline_index >= baseline_languages.size()) {
        throw std::runtime_error("no native prosody baseline for language '" + language + "'");
    }

    constexpr std::size_t kProsodyDim = 32;
    constexpr std::size_t kIdentityDim = 512;
    const auto& global_location_array = required_npy_array(pack, "prosody_global_locations");
    const auto& global_scale_array = required_npy_array(pack, "prosody_global_scales");
    require_npy_shape(global_location_array, {kProsodyDim}, "prosody_global_locations");
    require_npy_shape(global_scale_array, {kProsodyDim}, "prosody_global_scales");
    const auto& global_location = global_location_array.values;
    const auto& global_scale = global_scale_array.values;
    const std::vector<float> baseline_raw = npy_numeric_row(
        pack, "baseline_locations", baseline_index, kProsodyDim
    );
    const std::vector<float> baseline_mask = npy_numeric_row(
        pack, "baseline_feature_masks", baseline_index, kProsodyDim
    );
    const auto& baseline_confidence = required_npy_values(pack, "baseline_confidence");
    if (baseline_confidence.size() != baseline_languages.size()) {
        throw std::runtime_error("schema-v4 baseline confidence count is invalid");
    }

    ScyllasBandReferenceRouteV4 routed;
    routed.baseline_index = static_cast<int>(baseline_index);
    routed.prosody_baseline.resize(kProsodyDim, 0.0f);
    routed.prosody_delta.assign(kProsodyDim, 0.0f);
    routed.prosody_feature_mask = baseline_mask;
    routed.prosody_confidence = baseline_confidence[baseline_index];
    for (std::size_t index = 0; index < kProsodyDim; ++index) {
        const float scale = global_scale[index];
        if (!std::isfinite(scale) || scale <= 0.0f) {
            throw std::runtime_error("schema-v4 prosody global scale must be positive");
        }
        routed.prosody_baseline[index] = (
            (baseline_raw[index] - global_location[index]) / scale
        ) * baseline_mask[index];
    }

    const auto& prototype_languages = required_npy_strings(pack, "prototype_languages");
    const auto& prototype_baselines = required_npy_values(pack, "prototype_baseline_indices");
    const auto& prototype_centers = required_npy_array(pack, "prototype_affect_centers");
    const auto& prototype_masks = required_npy_array(pack, "prototype_affect_masks");
    const auto& prototype_confidence = required_npy_values(pack, "prototype_confidence");
    const auto& prototype_support = required_npy_values(pack, "prototype_distinct_source_counts");
    const std::size_t prototype_count = prototype_languages.size();
    require_npy_shape(prototype_centers, {prototype_count, 5}, "prototype_affect_centers");
    require_npy_shape(prototype_masks, {prototype_count, 5}, "prototype_affect_masks");
    if (
        prototype_baselines.size() != prototype_count ||
        prototype_confidence.size() != prototype_count ||
        prototype_support.size() != prototype_count
    ) {
        throw std::runtime_error("schema-v4 prototype metadata counts are inconsistent");
    }

    int selected = -1;
    float selected_confidence = -std::numeric_limits<float>::infinity();
    float selected_support = -std::numeric_limits<float>::infinity();
    for (std::size_t row = 0; row < prototype_count; ++row) {
        if (
            prototype_languages[row] != language ||
            static_cast<int>(prototype_baselines[row]) != static_cast<int>(baseline_index)
        ) {
            continue;
        }
        bool exact = true;
        for (std::size_t axis = 0; axis < 5; ++axis) {
            const bool target_active = affect_mask[axis] > 0.5f;
            const bool center_active = prototype_masks.values[row * 5 + axis] > 0.5f;
            if (target_active != center_active) {
                exact = false;
                break;
            }
            if (
                target_active &&
                std::fabs(prototype_centers.values[row * 5 + axis] - affect_values[axis]) > 1.0e-6f
            ) {
                exact = false;
                break;
            }
        }
        if (!exact) {
            continue;
        }
        const float confidence = prototype_confidence[row];
        const float support = prototype_support[row];
        if (
            selected < 0 ||
            confidence > selected_confidence ||
            (confidence == selected_confidence && support > selected_support)
        ) {
            selected = static_cast<int>(row);
            selected_confidence = confidence;
            selected_support = support;
        }
    }

    if (selected >= 0) {
        const auto& reference_indices = required_npy_array(pack, "prototype_reference_indices");
        if (reference_indices.shape.size() != 2 ||
            static_cast<std::size_t>(selected) >= reference_indices.shape[0]) {
            throw std::runtime_error("schema-v4 prototype reference indices have an invalid shape");
        }
        bool has_reference = false;
        const std::size_t width = reference_indices.shape[1];
        for (std::size_t column = 0; column < width; ++column) {
            if (reference_indices.values[static_cast<std::size_t>(selected) * width + column] >= 0.0f) {
                has_reference = true;
                break;
            }
        }
        if (has_reference) {
            routed.prototype_index = selected;
            routed.route_kind = "native_exact";
            routed.prosody_delta = npy_numeric_row(
                pack, "prototype_prosody_deltas", static_cast<std::size_t>(selected), kProsodyDim
            );
            const std::vector<float> prototype_mask = npy_numeric_row(
                pack, "prototype_feature_masks", static_cast<std::size_t>(selected), kProsodyDim
            );
            for (std::size_t index = 0; index < kProsodyDim; ++index) {
                routed.prosody_feature_mask[index] = baseline_mask[index] * prototype_mask[index];
                routed.prosody_delta[index] *= routed.prosody_feature_mask[index];
            }
            routed.prosody_confidence = prototype_confidence[static_cast<std::size_t>(selected)];
        }
    }

    const auto& identity_array = required_npy_array(pack, "identity_embeddings");
    if (
        identity_array.shape.size() != 2 ||
        identity_array.shape[0] == 0 ||
        identity_array.shape[1] != kIdentityDim
    ) {
        throw std::runtime_error("schema-v4 identity embeddings have an invalid shape");
    }
    const auto& identity_weights = required_npy_values(pack, "identity_embedding_weights");
    const auto& identity_references = required_npy_values(pack, "identity_window_reference_indices");
    const std::size_t identity_count = identity_array.shape[0];
    if (identity_weights.size() != identity_count || identity_references.size() != identity_count) {
        throw std::runtime_error("schema-v4 identity metadata counts are inconsistent");
    }
    double weight_sum = 0.0;
    for (float weight : identity_weights) {
        weight_sum += static_cast<double>(weight);
    }
    if (!std::isfinite(weight_sum) || weight_sum <= 0.0) {
        throw std::runtime_error("schema-v4 identity weights must have positive mass");
    }
    routed.identity.assign(kIdentityDim, 0.0f);
    for (std::size_t row = 0; row < identity_count; ++row) {
        const float weight = static_cast<float>(
            static_cast<double>(identity_weights[row]) / weight_sum
        );
        for (std::size_t dim = 0; dim < kIdentityDim; ++dim) {
            routed.identity[dim] += identity_array.values[row * kIdentityDim + dim] * weight;
        }
    }
    double norm_squared = 0.0;
    for (float value : routed.identity) {
        norm_squared += static_cast<double>(value) * static_cast<double>(value);
    }
    const double norm = std::sqrt(norm_squared);
    if (norm > 1.0e-12) {
        for (float& value : routed.identity) {
            value = static_cast<float>(static_cast<double>(value) / norm);
        }
    }
    routed.prosody_confidence = std::max(0.0f, std::min(1.0f, routed.prosody_confidence));
    for (const auto* values : {
        &routed.identity,
        &routed.prosody_baseline,
        &routed.prosody_delta,
        &routed.prosody_feature_mask,
    }) {
        for (float value : *values) {
            if (!std::isfinite(value)) {
                throw std::runtime_error("schema-v4 reference routing produced non-finite values");
            }
        }
    }
    return routed;
}

std::vector<float> transform_prosody(std::vector<float> values) {
    static const int log_indices[] = {1, 2, 5, 6, 7, 8, 9, 10, 11, 13};
    static const int drop_indices[] = {0, 14};
    for (float& value : values) {
        value = finite_or_zero(value);
    }
    for (int index : log_indices) {
        if (index >= 0 && static_cast<std::size_t>(index) < values.size()) {
            values[static_cast<std::size_t>(index)] = std::log1p(std::max(0.0f, values[static_cast<std::size_t>(index)]));
        }
    }
    for (int index : drop_indices) {
        if (index >= 0 && static_cast<std::size_t>(index) < values.size()) {
            values[static_cast<std::size_t>(index)] = 0.0f;
        }
    }
    return values;
}

std::pair<std::vector<float>, std::vector<float>> standardize_rows(
    const std::vector<std::vector<float>>& rows,
    int dim
) {
    const std::size_t width = static_cast<std::size_t>(std::max(0, dim));
    std::vector<float> mean(width, 0.0f);
    std::vector<float> stddev(width, 1.0f);
    if (rows.empty() || width == 0) {
        return {mean, stddev};
    }
    for (const auto& row : rows) {
        for (std::size_t index = 0; index < width; ++index) {
            const float value = index < row.size() ? finite_or_zero(row[index]) : 0.0f;
            mean[index] += value;
        }
    }
    const float denom = static_cast<float>(rows.size());
    for (float& value : mean) {
        value /= denom;
    }
    std::fill(stddev.begin(), stddev.end(), 0.0f);
    for (const auto& row : rows) {
        for (std::size_t index = 0; index < width; ++index) {
            const float value = index < row.size() ? finite_or_zero(row[index]) : 0.0f;
            const float diff = value - mean[index];
            stddev[index] += diff * diff;
        }
    }
    for (float& value : stddev) {
        value = std::sqrt(value / denom);
        if (value < 1e-4f || !std::isfinite(value)) {
            value = 1.0f;
        }
    }
    return {mean, stddev};
}

std::vector<float> normalize_style(
    const std::vector<float>& style,
    const std::vector<float>& mean,
    const std::vector<float>& stddev,
    int dim
) {
    std::vector<float> out(static_cast<std::size_t>(std::max(0, dim)), 0.0f);
    float norm_sq = 0.0f;
    for (std::size_t index = 0; index < out.size(); ++index) {
        const float value = index < style.size() ? finite_or_zero(style[index]) : 0.0f;
        const float mu = index < mean.size() ? mean[index] : 0.0f;
        const float sigma = index < stddev.size() ? std::max(stddev[index], 1e-4f) : 1.0f;
        out[index] = finite_or_zero((value - mu) / sigma);
        norm_sq += out[index] * out[index];
    }
    const float norm = std::max(std::sqrt(norm_sq), 1e-6f);
    for (float& value : out) {
        value = finite_or_zero(value / norm);
    }
    return out;
}

std::vector<float> normalize_prosody(
    const std::vector<float>& prosody,
    const std::vector<float>& mean,
    const std::vector<float>& stddev,
    int dim
) {
    std::vector<float> transformed = transform_prosody(prosody);
    std::vector<float> out(static_cast<std::size_t>(std::max(0, dim)), 0.0f);
    for (std::size_t index = 0; index < out.size(); ++index) {
        const float value = index < transformed.size() ? transformed[index] : 0.0f;
        const float mu = index < mean.size() ? mean[index] : 0.0f;
        const float sigma = index < stddev.size() ? std::max(stddev[index], 1e-4f) : 1.0f;
        out[index] = finite_or_zero((value - mu) / sigma);
    }
    if (!out.empty()) {
        out[0] = 0.0f;
    }
    if (out.size() > 14) {
        out[14] = 0.0f;
    }
    return out;
}

float effective_reference_mask(float reference_mask, float native_mask, float fallback_mask, float fallback_weight) {
    if (reference_mask <= 0.0f) {
        return 0.0f;
    }
    if (native_mask > 0.0f) {
        return 1.0f;
    }
    if (fallback_mask > 0.0f) {
        return std::max(0.0f, fallback_weight);
    }
    return 1.0f;
}

std::string reference_key_from_suffix(const std::string& suffix) {
    const std::size_t pos = suffix.find("__");
    if (pos == std::string::npos) {
        return suffix;
    }
    return suffix.substr(0, pos) + "|" + suffix.substr(pos + 2);
}

std::filesystem::path reference_pack_path_for_voice(const ScyllasBandBundleInfo& bundle, const std::string& voice_id) {
    const auto it = bundle.assets.find("voice_packs");
    if (it == bundle.assets.end() || it->second.empty()) {
        return {};
    }
    return std::filesystem::path(bundle.bundle_dir) / it->second / (voice_id + ".npz");
}

std::string select_reference_suffix(
    const ScyllasBandNpyPack& pack,
    const std::string& language,
    const std::string& emotion
) {
    std::vector<std::string> languages{language};
    if (language == "en_gb") {
        languages.push_back("en_us");
    } else if (language == "en") {
        languages.push_back("en_us");
        languages.push_back("en_gb");
    }
    std::vector<std::string> emotions{emotion};
    if (emotion != "neutral") {
        emotions.push_back("neutral");
    }
    for (const std::string& lang : languages) {
        for (const std::string& emo : emotions) {
            const std::string suffix = lang + "__" + emo;
            if (mask_value(pack, "reference_mask__" + suffix) > 0.0f) {
                return suffix;
            }
        }
    }
    return {};
}

void skip_json_ws(const std::string& json, std::size_t& pos) {
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) {
        ++pos;
    }
}

int hex_digit_value(char ch) {
    if (ch >= '0' && ch <= '9') {
        return ch - '0';
    }
    if (ch >= 'a' && ch <= 'f') {
        return 10 + ch - 'a';
    }
    if (ch >= 'A' && ch <= 'F') {
        return 10 + ch - 'A';
    }
    return -1;
}

void append_utf8_codepoint(std::string& out, uint32_t codepoint) {
    if (codepoint <= 0x7fU) {
        out.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7ffU) {
        out.push_back(static_cast<char>(0xc0U | (codepoint >> 6)));
        out.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    } else if (codepoint <= 0xffffU) {
        out.push_back(static_cast<char>(0xe0U | (codepoint >> 12)));
        out.push_back(static_cast<char>(0x80U | ((codepoint >> 6) & 0x3fU)));
        out.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    } else if (codepoint <= 0x10ffffU) {
        out.push_back(static_cast<char>(0xf0U | (codepoint >> 18)));
        out.push_back(static_cast<char>(0x80U | ((codepoint >> 12) & 0x3fU)));
        out.push_back(static_cast<char>(0x80U | ((codepoint >> 6) & 0x3fU)));
        out.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    }
}

std::string parse_json_string_literal_backend(const std::string& json, std::size_t& pos) {
    skip_json_ws(json, pos);
    if (pos >= json.size() || json[pos] != '"') {
        return {};
    }
    ++pos;
    std::string out;
    while (pos < json.size()) {
        const char ch = json[pos++];
        if (ch == '"') {
            break;
        }
        if (ch == '\\' && pos < json.size()) {
            const char escaped = json[pos++];
            switch (escaped) {
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case 'u': {
                    if (pos + 4 <= json.size()) {
                        uint32_t codepoint = 0;
                        bool valid = true;
                        for (int index = 0; index < 4; ++index) {
                            const int value = hex_digit_value(json[pos + static_cast<std::size_t>(index)]);
                            if (value < 0) {
                                valid = false;
                                break;
                            }
                            codepoint = (codepoint << 4) | static_cast<uint32_t>(value);
                        }
                        if (valid) {
                            append_utf8_codepoint(out, codepoint);
                            pos += 4;
                            break;
                        }
                    }
                    out.push_back('u');
                    break;
                }
                default: out.push_back(escaped); break;
            }
        } else {
            out.push_back(ch);
        }
    }
    return out;
}

std::size_t find_json_matching_delim(
    const std::string& json,
    std::size_t start,
    char open_delim,
    char close_delim
) {
    if (start >= json.size() || json[start] != open_delim) {
        return std::string::npos;
    }
    int depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (std::size_t pos = start; pos < json.size(); ++pos) {
        const char ch = json[pos];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (ch == '\\') {
                escaped = true;
            } else if (ch == '"') {
                in_string = false;
            }
            continue;
        }
        if (ch == '"') {
            in_string = true;
            continue;
        }
        if (ch == open_delim) {
            ++depth;
        } else if (ch == close_delim) {
            --depth;
            if (depth == 0) {
                return pos;
            }
        }
    }
    return std::string::npos;
}

std::string json_value_for_key_backend(const std::string& object_json, const std::string& key) {
    std::size_t pos = object_json.find('{');
    if (pos == std::string::npos) {
        return {};
    }
    ++pos;
    while (pos < object_json.size()) {
        skip_json_ws(object_json, pos);
        if (pos >= object_json.size() || object_json[pos] == '}') {
            break;
        }
        const std::string member_key = parse_json_string_literal_backend(object_json, pos);
        skip_json_ws(object_json, pos);
        if (pos >= object_json.size() || object_json[pos] != ':') {
            break;
        }
        ++pos;
        skip_json_ws(object_json, pos);
        const std::size_t value_start = pos;
        std::size_t value_end = pos;
        if (pos < object_json.size() && object_json[pos] == '{') {
            value_end = find_json_matching_delim(object_json, pos, '{', '}');
            if (value_end == std::string::npos) {
                return {};
            }
            ++value_end;
        } else if (pos < object_json.size() && object_json[pos] == '[') {
            value_end = find_json_matching_delim(object_json, pos, '[', ']');
            if (value_end == std::string::npos) {
                return {};
            }
            ++value_end;
        } else if (pos < object_json.size() && object_json[pos] == '"') {
            parse_json_string_literal_backend(object_json, pos);
            value_end = pos;
        } else {
            while (value_end < object_json.size() && object_json[value_end] != ',' && object_json[value_end] != '}') {
                ++value_end;
            }
        }
        if (member_key == key) {
            return object_json.substr(value_start, value_end - value_start);
        }
        pos = value_end;
        skip_json_ws(object_json, pos);
        if (pos < object_json.size() && object_json[pos] == ',') {
            ++pos;
        }
    }
    return {};
}

std::map<std::string, std::string> json_object_members_backend(const std::string& object_json) {
    std::map<std::string, std::string> members;
    std::size_t pos = object_json.find('{');
    if (pos == std::string::npos) {
        return members;
    }
    ++pos;
    while (pos < object_json.size()) {
        skip_json_ws(object_json, pos);
        if (pos >= object_json.size() || object_json[pos] == '}') {
            break;
        }
        const std::string key = parse_json_string_literal_backend(object_json, pos);
        skip_json_ws(object_json, pos);
        if (pos >= object_json.size() || object_json[pos] != ':') {
            break;
        }
        ++pos;
        skip_json_ws(object_json, pos);
        if (pos >= object_json.size() || object_json[pos] != '{') {
            const std::string skipped = json_value_for_key_backend(object_json.substr(pos), key);
            (void)skipped;
            break;
        }
        const std::size_t end = find_json_matching_delim(object_json, pos, '{', '}');
        if (end == std::string::npos) {
            break;
        }
        members[key] = object_json.substr(pos, end - pos + 1);
        pos = end + 1;
        skip_json_ws(object_json, pos);
        if (pos < object_json.size() && object_json[pos] == ',') {
            ++pos;
        }
    }
    return members;
}

std::vector<std::string> json_string_array_or_split_backend(const std::string& object_json, const std::string& key) {
    std::vector<std::string> values;
    std::string raw = json_value_for_key_backend(object_json, key);
    std::size_t pos = 0;
    skip_json_ws(raw, pos);
    if (pos >= raw.size()) {
        return values;
    }
    if (raw[pos] == '[') {
        ++pos;
        while (pos < raw.size()) {
            skip_json_ws(raw, pos);
            if (pos >= raw.size() || raw[pos] == ']') {
                break;
            }
            if (raw[pos] == '"') {
                std::string item = parse_json_string_literal_backend(raw, pos);
                if (!item.empty()) {
                    values.push_back(item);
                }
            } else {
                while (pos < raw.size() && raw[pos] != ',' && raw[pos] != ']') {
                    ++pos;
                }
            }
            skip_json_ws(raw, pos);
            if (pos < raw.size() && raw[pos] == ',') {
                ++pos;
            }
        }
        return values;
    }
    if (raw[pos] == '"') {
        std::string value = parse_json_string_literal_backend(raw, pos);
        std::istringstream stream(value);
        std::string item;
        while (stream >> item) {
            values.push_back(item);
        }
    }
    return values;
}

ScyllasBandPronunciationOverrideMap load_pronunciation_overrides(const std::filesystem::path& path) {
    ScyllasBandPronunciationOverrideMap out;
    if (!std::filesystem::is_regular_file(path)) {
        return out;
    }
    std::vector<uint8_t> bytes = read_binary_file(path);
    std::string json(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    std::string languages_json = json_value_for_key_backend(json, "languages");
    if (languages_json.empty()) {
        languages_json = json;
    }
    for (const auto& language_item : json_object_members_backend(languages_json)) {
        const std::string language = lowercase_ascii_backend(language_item.first);
        std::vector<ScyllasBandPronunciationOverrideSpec> specs;
        for (const auto& word_item : json_object_members_backend(language_item.second)) {
            ScyllasBandPronunciationOverrideSpec spec;
            spec.word = lowercase_ascii_backend(trim_backend(word_item.first));
            spec.replace = json_string_array_or_split_backend(word_item.second, "replace");
            spec.phones = json_string_array_or_split_backend(word_item.second, "phones");
            if (!spec.word.empty() && !spec.replace.empty() && !spec.phones.empty()) {
                specs.push_back(std::move(spec));
            }
        }
        if (!specs.empty()) {
            std::sort(specs.begin(), specs.end(), [](const auto& lhs, const auto& rhs) {
                return lhs.word < rhs.word;
            });
            out[language] = std::move(specs);
        }
    }
    return out;
}

std::set<std::string> words_for_pronunciation_overrides(const std::string& text) {
    std::set<std::string> words;
    std::string current;
    for (std::size_t index = 0; index < text.size(); ++index) {
        const unsigned char ch = static_cast<unsigned char>(text[index]);
        if (std::isalpha(ch)) {
            current.push_back(static_cast<char>(std::tolower(ch)));
            continue;
        }
        if (ch == '\'' && !current.empty() && index + 1 < text.size() &&
            std::isalpha(static_cast<unsigned char>(text[index + 1]))) {
            current.push_back('\'');
            continue;
        }
        if (!current.empty()) {
            words.insert(current);
            current.clear();
        }
    }
    if (!current.empty()) {
        words.insert(current);
    }
    return words;
}

std::pair<std::vector<std::string>, int> replace_phone_subsequence(
    const std::vector<std::string>& phones,
    const std::vector<std::string>& source,
    const std::vector<std::string>& target
) {
    if (phones.empty() || source.empty()) {
        return {phones, 0};
    }
    std::vector<std::string> output;
    int count = 0;
    std::size_t index = 0;
    while (index < phones.size()) {
        bool match = index + source.size() <= phones.size();
        for (std::size_t offset = 0; match && offset < source.size(); ++offset) {
            if (phones[index + offset] != source[offset]) {
                match = false;
            }
        }
        if (match) {
            output.insert(output.end(), target.begin(), target.end());
            index += source.size();
            ++count;
        } else {
            output.push_back(phones[index]);
            ++index;
        }
    }
    return {output, count};
}

std::vector<ScyllasBandPronunciationOverrideApplied> apply_pronunciation_overrides(
    std::vector<std::string>& phones,
    const std::string& text,
    const std::string& language,
    const ScyllasBandPronunciationOverrideMap& overrides
) {
    std::vector<ScyllasBandPronunciationOverrideApplied> applied;
    const auto override_it = overrides.find(lowercase_ascii_backend(language));
    if (override_it == overrides.end()) {
        return applied;
    }
    const std::set<std::string> words = words_for_pronunciation_overrides(text);
    if (words.empty()) {
        return applied;
    }
    for (const ScyllasBandPronunciationOverrideSpec& spec : override_it->second) {
        if (words.find(spec.word) == words.end()) {
            continue;
        }
        auto replacement = replace_phone_subsequence(phones, spec.replace, spec.phones);
        if (replacement.second <= 0) {
            continue;
        }
        phones = std::move(replacement.first);
        applied.push_back({spec.word, spec.replace, spec.phones, replacement.second});
    }
    return applied;
}

std::string pronunciation_overrides_json(const std::vector<ScyllasBandPronunciationOverrideApplied>& applied) {
    std::ostringstream out;
    out << "[";
    for (std::size_t index = 0; index < applied.size(); ++index) {
        if (index > 0) {
            out << ",";
        }
        out << "{"
            << "\"word\":\"" << json_escape_backend(applied[index].word) << "\","
            << "\"replace\":" << string_vector_json_backend(applied[index].replace) << ","
            << "\"phones\":" << string_vector_json_backend(applied[index].phones) << ","
            << "\"count\":" << applied[index].count
            << "}";
    }
    out << "]";
    return out.str();
}

std::string terminal_tail_repairs_json(const std::vector<ScyllasBandG2PTerminalTailRepair>& repairs) {
    std::ostringstream out;
    out << "[";
    for (std::size_t index = 0; index < repairs.size(); ++index) {
        if (index > 0) {
            out << ",";
        }
        out << "{"
            << "\"word\":\"" << json_escape_backend(repairs[index].word) << "\","
            << "\"phones\":" << string_vector_json_backend(repairs[index].phones) << ","
            << "\"removed\":" << string_vector_json_backend(repairs[index].removed)
            << "}";
    }
    out << "]";
    return out.str();
}

ScyllasBandG2PResult run_g2p_text(
    ScyllasBandLiteRtSession* session,
    const ScyllasBandBundleInfo& bundle,
    const ScyllasBandResolvedRequest& resolved_request,
    const ScyllasBandSynthesisRequest& request,
    const ScyllasBandPronunciationOverrideMap& pronunciation_overrides
) {
    if (request.text == nullptr || request.text[0] == '\0') {
        throw std::runtime_error("Raw text synthesis requires text or explicit_phones");
    }
    ScyllasBandG2PResult result;
    const std::string language = resolve_g2p_language(bundle, resolved_request.language);
    const int g2p_char_repeats = std::max(1, bundle.g2p_char_repeats);
    const int g2p_text_capacity = std::max(1, bundle.g2p_text_tokens - 2);
    const std::size_t safe_phrase_chars = static_cast<std::size_t>(
        std::max(1, g2p_text_capacity / g2p_char_repeats)
    );
    result.segments = split_g2p_segments(
        request.text,
        std::min<std::size_t>(140, safe_phrase_chars)
    );
    const bool has_silence = bundle.phone_to_id.find("<sil>") != bundle.phone_to_id.end();
    const bool has_pause_comma = bundle.phone_to_id.find("<pause_comma>") != bundle.phone_to_id.end();
    const bool explicit_punctuation_silence = bundle.punctuation_silence_target == "explicit_silence";
    const std::string before = resolved_request.boundary_before.empty() ? "paragraph_start" : resolved_request.boundary_before;
    if (has_silence && (before.empty() || before == "paragraph_start" || before == "sentence_start")) {
        result.phones.push_back("<sil>");
    }
    for (std::size_t index = 0; index < result.segments.size(); ++index) {
        std::vector<std::string> segment_phones = run_g2p_segment(session, bundle, result.segments[index], language);
        std::vector<ScyllasBandPronunciationOverrideApplied> applied = apply_pronunciation_overrides(
            segment_phones,
            result.segments[index],
            language,
            pronunciation_overrides
        );
        result.pronunciation_overrides.insert(result.pronunciation_overrides.end(), applied.begin(), applied.end());
        const std::string terminal_word = terminal_g2p_word(result.segments[index]);
        if (!terminal_word.empty()) {
            std::vector<std::string> isolated_phones = run_g2p_segment(
                session,
                bundle,
                terminal_word,
                language
            );
            apply_pronunciation_overrides(
                isolated_phones,
                terminal_word,
                language,
                pronunciation_overrides
            );
            std::vector<std::string> removed = trim_g2p_terminal_artifacts(
                segment_phones,
                isolated_phones
            );
            if (!removed.empty()) {
                result.terminal_tail_repairs.push_back({terminal_word, isolated_phones, removed});
            }
        }
        for (const std::string& phone : segment_phones) {
            if (is_silence_phone(phone)) {
                continue;
            }
            result.phones.push_back(phone);
        }
        const bool has_following_segment = index + 1 < result.segments.size();
        const std::string punctuation_phone = punctuation_phone_token_backend(result.segments[index]);
        if (!punctuation_phone.empty() && bundle.phone_to_id.find(punctuation_phone) != bundle.phone_to_id.end()) {
            result.phones.push_back(punctuation_phone);
            if (explicit_punctuation_silence && has_silence) {
                result.phones.push_back("<sil>");
            }
        } else if (has_following_segment && has_pause_comma) {
            result.phones.push_back("<pause_comma>");
            if (explicit_punctuation_silence && has_silence) {
                result.phones.push_back("<sil>");
            }
        }
    }
    const std::string after = resolved_request.boundary_after.empty() ? "sentence_end" : resolved_request.boundary_after;
    const std::string trailing_phone = trailing_boundary_pause_phone(
        after,
        has_pause_comma,
        ends_in_g2p_boundary_phone(result.phones)
    );
    if (!trailing_phone.empty()) {
        result.phones.push_back(trailing_phone);
    }
    if (result.phones.empty()) {
        throw std::runtime_error("G2P produced no usable Scylla's Band phone symbols");
    }
    result.prediction_text = join_phone_tokens(result.phones);
    std::ostringstream metadata;
    metadata << "{"
             << "\"phone_source\":\"g2p\","
             << "\"g2p_language\":\"" << json_escape_backend(language) << "\","
             << "\"g2p_input_text\":\"" << json_escape_backend(request.text) << "\","
             << "\"g2p_segments\":" << string_vector_json_backend(result.segments) << ","
             << "\"g2p_prediction_text\":\"" << json_escape_backend(result.prediction_text) << "\","
             << "\"phones\":" << string_vector_json_backend(result.phones, 64);
    if (!result.pronunciation_overrides.empty()) {
        metadata << ",\"pronunciation_overrides\":" << pronunciation_overrides_json(result.pronunciation_overrides);
    }
    if (!result.terminal_tail_repairs.empty()) {
        metadata << ",\"g2p_terminal_tail_repairs\":"
                 << terminal_tail_repairs_json(result.terminal_tail_repairs);
    }
    metadata << "}";
    result.metadata_json = metadata.str();
    return result;
}

std::string component_path(
    const ScyllasBandBundleInfo& bundle,
    const std::string& component_name
) {
    const auto artifact = bundle.component_artifacts.find(component_name);
    if (artifact == bundle.component_artifacts.end()) {
        throw std::runtime_error("Bundle is missing component artifact '" + component_name + "'");
    }
    return (std::filesystem::path(bundle.bundle_dir) / artifact->second).string();
}

struct ScyllasBandTargetBucketSelection {
    ScyllasBandTargetBucketInfo bucket;
    ScyllasBandBundleInfo bundle;
    std::string metadata_json = "null";
};

ScyllasBandTargetBucketInfo default_target_bucket(const ScyllasBandBundleInfo& bundle) {
    ScyllasBandTargetBucketInfo bucket;
    bucket.latent_frames = bundle.latent_frames;
    bucket.vector_component = "vector_estimator";
    bucket.vector_prefix_component = "vector_estimator_prefix";
    bucket.vector_tail_component = "vector_estimator_tail";
    bucket.vocoder_component = "vocoder";
    return bucket;
}

void remap_bucket_component(
    ScyllasBandBundleInfo& selected,
    const ScyllasBandBundleInfo& source,
    const std::string& from_component,
    const std::string& to_component
) {
    if (from_component.empty()) {
        selected.component_artifacts.erase(to_component);
        selected.component_inputs.erase(to_component);
        selected.component_outputs.erase(to_component);
        return;
    }
    const auto artifact = source.component_artifacts.find(from_component);
    if (artifact == source.component_artifacts.end()) {
        throw std::runtime_error("Target bucket component artifact is missing: " + from_component);
    }
    selected.component_artifacts[to_component] = artifact->second;
    const auto inputs = source.component_inputs.find(from_component);
    if (inputs != source.component_inputs.end()) {
        selected.component_inputs[to_component] = inputs->second;
    }
    const auto outputs = source.component_outputs.find(from_component);
    if (outputs != source.component_outputs.end()) {
        selected.component_outputs[to_component] = outputs->second;
    }
}

std::string target_bucket_metadata_json(
    const ScyllasBandBundleInfo& source,
    const ScyllasBandTargetBucketInfo& bucket,
    int64_t predicted_latent_frames,
    const std::string& policy
) {
    std::ostringstream metadata;
    metadata << "{"
             << "\"enabled\":" << (source.target_buckets.empty() ? "false" : "true") << ","
             << "\"policy\":\"" << json_escape_backend(policy) << "\","
             << "\"selected_latent_frames\":" << bucket.latent_frames << ","
             << "\"max_latent_frames\":" << source.latent_frames << ","
             << "\"predicted_latent_frames\":" << predicted_latent_frames << ","
             << "\"bucket_margin_frames\":" << source.target_bucket_margin_frames << ","
             << "\"fits\":" << (predicted_latent_frames <= bucket.latent_frames ? "true" : "false") << ","
             << "\"requires_replan\":" << (predicted_latent_frames > bucket.latent_frames ? "true" : "false") << ","
             << "\"vector_component\":\"" << json_escape_backend(bucket.vector_component) << "\","
             << "\"vector_prefix_component\":\"" << json_escape_backend(bucket.vector_prefix_component) << "\","
             << "\"vector_tail_component\":\"" << json_escape_backend(bucket.vector_tail_component) << "\","
             << "\"vocoder_component\":\"" << json_escape_backend(bucket.vocoder_component) << "\""
             << "}";
    return metadata.str();
}

ScyllasBandTargetBucketSelection select_target_bucket_bundle(
    const ScyllasBandBundleInfo& source,
    const ScyllasBandDurationExpansionMetadata& expansion,
    bool allow_overflow_for_preflight = false
) {
    ScyllasBandTargetBucketInfo selected_bucket = default_target_bucket(source);
    std::string policy = "single_fixed_shape";
    std::vector<int> bucket_frames;
    bucket_frames.reserve(source.target_buckets.size());
    for (const ScyllasBandTargetBucketInfo& candidate : source.target_buckets) {
        bucket_frames.push_back(candidate.latent_frames);
    }
    const ScyllasBandTargetBucketDecision decision = select_smallest_target_bucket(
        expansion.predicted_latent_frames,
        source.target_bucket_margin_frames,
        source.latent_frames,
        bucket_frames
    );
    if (!source.target_buckets.empty()) {
        policy = "smallest_fit";
        for (const ScyllasBandTargetBucketInfo& candidate : source.target_buckets) {
            if (candidate.latent_frames == decision.latent_frames) {
                selected_bucket = candidate;
                break;
            }
        }
    }
    if (selected_bucket.latent_frames <= 0) {
        throw std::runtime_error("Selected target bucket has invalid latent frame count");
    }
    if (!decision.fits && !allow_overflow_for_preflight) {
        throw std::runtime_error(
            "Predicted " + std::to_string(expansion.predicted_latent_frames) +
            " latent frames, but selected target bucket supports " +
            std::to_string(selected_bucket.latent_frames)
        );
    }
    ScyllasBandTargetBucketSelection selection;
    selection.bucket = selected_bucket;
    selection.bundle = source;
    selection.bundle.latent_frames = selected_bucket.latent_frames;
    remap_bucket_component(selection.bundle, source, selected_bucket.vector_component, "vector_estimator");
    remap_bucket_component(selection.bundle, source, selected_bucket.vocoder_component, "vocoder");
    if (source.component_artifacts.count(selected_bucket.vector_prefix_component) > 0 &&
        source.component_artifacts.count(selected_bucket.vector_tail_component) > 0) {
        remap_bucket_component(selection.bundle, source, selected_bucket.vector_prefix_component, "vector_estimator_prefix");
        remap_bucket_component(selection.bundle, source, selected_bucket.vector_tail_component, "vector_estimator_tail");
    } else {
        remap_bucket_component(selection.bundle, source, "", "vector_estimator_prefix");
        remap_bucket_component(selection.bundle, source, "", "vector_estimator_tail");
    }
    selection.metadata_json = target_bucket_metadata_json(
        source,
        selected_bucket,
        expansion.predicted_latent_frames,
        policy
    );
    return selection;
}

void append_tensor_view(
    ScyllasBandLiteRtInputStorage& storage,
    std::string name,
    int32_t data_type,
    const std::vector<int64_t>& shape,
    const void* data,
    uint64_t byte_length
) {
    storage.names.push_back(std::move(name));
    storage.views.push_back(ScyllasBandTensorView{
        storage.names.back().c_str(),
        data_type,
        shape.data(),
        static_cast<int32_t>(shape.size()),
        data,
        byte_length,
    });
}

std::string tensor_outputs_json(const ScyllasBandOwnedTensor* tensors, int32_t tensor_count) {
    std::ostringstream metadata;
    metadata << "[";
    for (int32_t index = 0; index < tensor_count; ++index) {
        if (index > 0) {
            metadata << ",";
        }
        const ScyllasBandOwnedTensor& tensor = tensors[index];
        metadata << "{"
                 << "\"name\":\"" << (tensor.name == nullptr ? "" : tensor.name) << "\","
                 << "\"dtype\":" << tensor.data_type << ","
                 << "\"rank\":" << tensor.rank << ","
                 << "\"shape\":[";
        for (int32_t dim = 0; dim < tensor.rank; ++dim) {
            if (dim > 0) {
                metadata << ",";
            }
            metadata << tensor.shape[dim];
        }
        metadata << "],\"bytes\":" << tensor.byte_length;
        if (tensor.data_type == SCYLLASBAND_TENSOR_FLOAT32 && tensor.data != nullptr) {
            const auto* values = static_cast<const float*>(tensor.data);
            const std::size_t value_count = tensor.byte_length / sizeof(float);
            std::size_t non_finite_count = 0;
            for (std::size_t value_index = 0; value_index < value_count; ++value_index) {
                if (!std::isfinite(values[value_index])) {
                    ++non_finite_count;
                }
            }
            metadata << ",\"non_finite_count\":" << non_finite_count;
            const std::size_t count = std::min<std::size_t>(8, value_count);
            metadata << ",\"preview\":[";
            for (std::size_t value_index = 0; value_index < count; ++value_index) {
                if (value_index > 0) {
                    metadata << ",";
                }
                if (std::isfinite(values[value_index])) {
                    metadata << values[value_index];
                } else {
                    metadata << "null";
                }
            }
            metadata << "]";
        }
        metadata << "}";
    }
    metadata << "]";
    return metadata.str();
}

std::vector<float> copy_first_float_tensor_values(
    const ScyllasBandOwnedTensor* tensors,
    int32_t tensor_count,
    const std::string& component_name,
    std::size_t expected_min_values = 0,
    std::size_t copy_count = 0
) {
    if (tensor_count <= 0 || tensors == nullptr) {
        throw std::runtime_error(component_name + " returned no tensors");
    }
    const ScyllasBandOwnedTensor& tensor = tensors[0];
    if (tensor.data_type != SCYLLASBAND_TENSOR_FLOAT32 || tensor.data == nullptr) {
        throw std::runtime_error(component_name + " output_0 must be float32");
    }
    const std::size_t available_values = tensor.byte_length / sizeof(float);
    if (available_values < expected_min_values) {
        throw std::runtime_error(component_name + " output_0 is shorter than expected");
    }
    const std::size_t count = copy_count == 0 ? available_values : std::min(copy_count, available_values);
    const auto* values = static_cast<const float*>(tensor.data);
    std::size_t non_finite_count = 0;
    for (std::size_t index = 0; index < count; ++index) {
        if (!std::isfinite(values[index])) {
            ++non_finite_count;
        }
    }
    if (non_finite_count > 0) {
        throw std::runtime_error(
            component_name + " output_0 contains " + std::to_string(non_finite_count) +
            " non-finite values"
        );
    }
    return std::vector<float>(values, values + count);
}

std::string int64_vector_json(const std::vector<int64_t>& values, std::size_t limit = 0) {
    std::ostringstream out;
    out << "[";
    const std::size_t count = limit == 0 ? values.size() : std::min(values.size(), limit);
    for (std::size_t index = 0; index < count; ++index) {
        if (index > 0) {
            out << ",";
        }
        out << values[index];
    }
    out << "]";
    return out.str();
}

std::string float_vector_json(const std::vector<float>& values, std::size_t limit = 0) {
    std::ostringstream out;
    out << "[";
    const std::size_t count = limit == 0 ? values.size() : std::min(values.size(), limit);
    for (std::size_t index = 0; index < count; ++index) {
        if (index > 0) {
            out << ",";
        }
        out << values[index];
    }
    out << "]";
    return out.str();
}

int64_t backend_pad_phone_id(const ScyllasBandBundleInfo& bundle) {
    const auto it = bundle.phone_to_id.find("<pad>");
    return it == bundle.phone_to_id.end() ? 0 : static_cast<int64_t>(it->second);
}

int64_t backend_phone_id_for_token(const ScyllasBandBundleInfo& bundle, const std::string& phone) {
    const auto it = bundle.phone_to_id.find(phone);
    if (it != bundle.phone_to_id.end()) {
        return static_cast<int64_t>(it->second);
    }
    throw std::runtime_error("Span context phone token '" + phone + "' is not in the bundle phone vocabulary");
}

void append_span_context_segment(
    const ScyllasBandBundleInfo& bundle,
    ScyllasBandSpanContextEncoding& encoding,
    const std::vector<std::string>& phones,
    std::size_t start,
    std::size_t count,
    int64_t segment_id
) {
    const std::size_t end = std::min(phones.size(), start + count);
    for (std::size_t index = start; index < end; ++index) {
        if (encoding.context_phone_count >= encoding.max_context_phones) {
            break;
        }
        encoding.phone_ids[static_cast<std::size_t>(encoding.context_phone_count)] =
            backend_phone_id_for_token(bundle, phones[index]);
        encoding.segment_ids[static_cast<std::size_t>(encoding.context_phone_count)] = segment_id;
        encoding.mask[static_cast<std::size_t>(encoding.context_phone_count)] = 1;
        ++encoding.context_phone_count;
    }
}

ScyllasBandSpanContextEncoding build_span_context_encoding(
    const ScyllasBandBundleInfo& bundle,
    const std::vector<std::string>& previous_phones,
    const std::vector<std::string>& target_phones,
    const std::vector<std::string>& next_phones
) {
    ScyllasBandSpanContextEncoding encoding;
    encoding.max_context_phones = std::max(0, bundle.span_context_max_phones);
    if (encoding.max_context_phones <= 0) {
        return encoding;
    }
    const std::size_t max_count = static_cast<std::size_t>(encoding.max_context_phones);
    const int64_t pad_id = backend_pad_phone_id(bundle);
    encoding.phone_ids.assign(max_count, pad_id);
    encoding.segment_ids.assign(max_count, 0);
    encoding.mask.assign(max_count, 0);

    const std::size_t target_take = std::min(target_phones.size(), max_count);
    std::size_t remaining = max_count - target_take;
    std::size_t previous_take = std::min(previous_phones.size(), remaining / 2);
    std::size_t next_take = std::min(next_phones.size(), remaining - previous_take);
    std::size_t extra = remaining - previous_take - next_take;
    if (extra > 0) {
        const std::size_t take = std::min(previous_phones.size() - previous_take, extra);
        previous_take += take;
        extra -= take;
    }
    if (extra > 0) {
        next_take += std::min(next_phones.size() - next_take, extra);
    }

    encoding.previous_phone_count = static_cast<int>(previous_take);
    encoding.target_phone_count = static_cast<int>(target_take);
    encoding.next_phone_count = static_cast<int>(next_take);

    const std::size_t previous_start = previous_phones.size() - previous_take;
    append_span_context_segment(bundle, encoding, previous_phones, previous_start, previous_take, 0);
    append_span_context_segment(bundle, encoding, target_phones, 0, target_take, 1);
    append_span_context_segment(bundle, encoding, next_phones, 0, next_take, 2);
    return encoding;
}

float rms_float_values(const std::vector<float>& values) {
    if (values.empty()) {
        return 0.0f;
    }
    double sum = 0.0;
    for (float value : values) {
        if (std::isfinite(value)) {
            sum += static_cast<double>(value) * static_cast<double>(value);
        }
    }
    return static_cast<float>(std::sqrt(sum / static_cast<double>(values.size())));
}

std::string span_context_metadata_json(
    const ScyllasBandSpanContextEncoding& encoding,
    const std::vector<float>& hidden,
    const std::string& mode,
    const std::string& outputs_json
) {
    std::ostringstream out;
    out << "{"
        << "\"mode\":\"" << json_escape_backend(mode) << "\","
        << "\"max_context_phones\":" << encoding.max_context_phones << ","
        << "\"context_phone_count\":" << encoding.context_phone_count << ","
        << "\"previous_phone_count\":" << encoding.previous_phone_count << ","
        << "\"target_phone_count\":" << encoding.target_phone_count << ","
        << "\"next_phone_count\":" << encoding.next_phone_count << ","
        << "\"hidden_size\":" << hidden.size() << ","
        << "\"hidden_rms\":" << rms_float_values(hidden) << ","
        << "\"outputs\":" << outputs_json
        << "}";
    return out.str();
}

ScyllasBandLiteRtInputStorage build_vector_context_encoder_inputs(
    const ScyllasBandBundleInfo& bundle,
    const ScyllasBandSpanContextEncoding& encoding
) {
    const auto inputs_it = bundle.component_inputs.find("vector_context_encoder");
    if (inputs_it == bundle.component_inputs.end()) {
        throw std::runtime_error("Bundle vector_context_encoder component does not declare inputs");
    }
    ScyllasBandLiteRtInputStorage storage;
    storage.names.reserve(inputs_it->second.size());
    storage.views.reserve(inputs_it->second.size());
    storage.span_context_shape = {1, static_cast<int64_t>(encoding.max_context_phones)};
    storage.span_context_phone_ids = encoding.phone_ids;
    storage.span_context_segment_ids = encoding.segment_ids;
    storage.span_context_mask = encoding.mask;
    for (std::size_t index = 0; index < inputs_it->second.size(); ++index) {
        const std::string raw_name = "args_" + std::to_string(index);
        const std::string& semantic = inputs_it->second[index];
        if (semantic == "span_context_phone_ids") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_INT64, storage.span_context_shape,
                               storage.span_context_phone_ids.data(), storage.span_context_phone_ids.size() * sizeof(int64_t));
        } else if (semantic == "span_context_segment_ids") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_INT64, storage.span_context_shape,
                               storage.span_context_segment_ids.data(), storage.span_context_segment_ids.size() * sizeof(int64_t));
        } else if (semantic == "span_context_mask") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_BOOL, storage.span_context_shape,
                               storage.span_context_mask.data(), storage.span_context_mask.size() * sizeof(uint8_t));
        } else {
            throw std::runtime_error("Unsupported vector_context_encoder input '" + semantic + "'");
        }
    }
    return storage;
}

ScyllasBandSpanContextResult zero_span_context_result(
    const ScyllasBandBundleInfo& bundle,
    const ScyllasBandSpanContextEncoding& encoding,
    const std::string& mode
) {
    ScyllasBandSpanContextResult result;
    result.hidden.assign(static_cast<std::size_t>(std::max(0, bundle.span_context_hidden_size)), 0.0f);
    result.metadata_json = span_context_metadata_json(encoding, result.hidden, mode, "null");
    return result;
}

ScyllasBandSpanContextResult run_span_context_encoder(
    ScyllasBandLiteRtSession* session,
    const ScyllasBandBundleInfo& bundle,
    const ScyllasBandSpanContextEncoding& encoding
) {
    if (session == nullptr) {
        return zero_span_context_result(bundle, encoding, "zero_no_encoder_session");
    }
    ScyllasBandLiteRtInputStorage inputs = build_vector_context_encoder_inputs(bundle, encoding);
    ScyllasBandOwnedTensor* outputs = nullptr;
    int32_t output_count = 0;
    const int run_status = scyllasband_litert_session_run(
        session,
        "serving_default",
        inputs.views.data(),
        static_cast<int32_t>(inputs.views.size()),
        &outputs,
        &output_count
    );
    if (run_status != 0) {
        const char* detail = last_error();
        throw std::runtime_error(std::string("vector_context_encoder failed: ") +
                                 (detail == nullptr ? "" : detail));
    }
    ScyllasBandSpanContextResult result;
    try {
        result.outputs_json = tensor_outputs_json(outputs, output_count);
        result.hidden = copy_first_float_tensor_values(
            outputs,
            output_count,
            "vector_context_encoder",
            static_cast<std::size_t>(std::max(0, bundle.span_context_hidden_size)),
            static_cast<std::size_t>(std::max(0, bundle.span_context_hidden_size))
        );
        result.metadata_json = span_context_metadata_json(encoding, result.hidden, "encoded", result.outputs_json);
    } catch (...) {
        scyllasband_tensors_destroy(outputs, output_count);
        throw;
    }
    scyllasband_tensors_destroy(outputs, output_count);
    return result;
}

ScyllasBandLiteRtInputStorage build_duration_predictor_inputs(
    const ScyllasBandBundleInfo& bundle,
    const ScyllasBandDurationFlowPreparedInputs& prepared,
    int64_t emotion_id,
    float emotion_condition_scale,
    float reference_condition_scale
) {
    const auto inputs_it = bundle.component_inputs.find("duration_predictor");
    if (inputs_it == bundle.component_inputs.end()) {
        throw std::runtime_error("Bundle duration_predictor component does not declare inputs");
    }
    const std::vector<std::string>& semantic_inputs = inputs_it->second;

    ScyllasBandLiteRtInputStorage storage;
    storage.names.reserve(semantic_inputs.size());
    storage.views.reserve(semantic_inputs.size());
    storage.phone_shape = {1, static_cast<int64_t>(prepared.phone_ids.size())};
    storage.scalar_shape = {1};
    storage.affect_shape = {1, static_cast<int64_t>(prepared.affect_values.size())};
    storage.affect_condition_mask_shape = {1, static_cast<int64_t>(prepared.affect_condition_mask_values.size())};
    storage.reference_style_shape = {1, static_cast<int64_t>(prepared.reference_style.size())};
    storage.reference_prosody_shape = {1, static_cast<int64_t>(prepared.reference_prosody.size())};
    storage.identity_reference_shape = {1, static_cast<int64_t>(prepared.identity_reference.size())};
    storage.prosody_baseline_shape = {1, static_cast<int64_t>(prepared.prosody_baseline.size())};
    storage.prosody_delta_shape = {1, static_cast<int64_t>(prepared.prosody_delta.size())};
    storage.prosody_feature_mask_shape = {1, static_cast<int64_t>(prepared.prosody_feature_mask.size())};
    storage.phone_ids = prepared.phone_ids;
    storage.phone_mask = prepared.phone_mask;
    storage.voice_id = {prepared.voice_id};
    storage.language_id = {prepared.language_id};
    storage.emotion_id = {emotion_id};
    storage.boundary_before_id = {prepared.boundary_before_id};
    storage.boundary_after_id = {prepared.boundary_after_id};
    storage.emotion_condition_mask = {emotion_condition_scale};
    storage.affect_values = prepared.affect_values;
    storage.affect_condition_mask = prepared.affect_condition_mask_values;
    for (float& value : storage.affect_condition_mask) {
        value *= emotion_condition_scale;
    }
    storage.identity_reference = prepared.identity_reference;
    storage.identity_reference_mask = {
        prepared.identity_reference_mask * reference_condition_scale
    };
    storage.prosody_baseline = prepared.prosody_baseline;
    storage.prosody_delta = prepared.prosody_delta;
    storage.prosody_feature_mask = prepared.prosody_feature_mask;
    storage.prosody_confidence = {prepared.prosody_confidence};
    storage.reference_style = prepared.reference_style;
    storage.reference_prosody = prepared.reference_prosody;
    storage.reference_mask = {prepared.reference_mask};
    storage.reference_condition_mask = {reference_condition_scale};

    for (std::size_t index = 0; index < semantic_inputs.size(); ++index) {
        const std::string raw_name = "args_" + std::to_string(index);
        const std::string& semantic = semantic_inputs[index];
        if (semantic == "phone_ids") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_INT64, storage.phone_shape,
                               storage.phone_ids.data(), storage.phone_ids.size() * sizeof(int64_t));
        } else if (semantic == "voice_id") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_INT64, storage.scalar_shape,
                               storage.voice_id.data(), sizeof(int64_t));
        } else if (semantic == "language_id") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_INT64, storage.scalar_shape,
                               storage.language_id.data(), sizeof(int64_t));
        } else if (semantic == "emotion_id") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_INT64, storage.scalar_shape,
                               storage.emotion_id.data(), sizeof(int64_t));
        } else if (semantic == "boundary_before_id") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_INT64, storage.scalar_shape,
                               storage.boundary_before_id.data(), sizeof(int64_t));
        } else if (semantic == "boundary_after_id") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_INT64, storage.scalar_shape,
                               storage.boundary_after_id.data(), sizeof(int64_t));
        } else if (semantic == "phone_mask") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_BOOL, storage.phone_shape,
                               storage.phone_mask.data(), storage.phone_mask.size() * sizeof(uint8_t));
        } else if (semantic == "emotion_condition_mask") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_FLOAT32, storage.scalar_shape,
                               storage.emotion_condition_mask.data(), sizeof(float));
        } else if (semantic == "affect_values") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_FLOAT32, storage.affect_shape,
                               storage.affect_values.data(), storage.affect_values.size() * sizeof(float));
        } else if (semantic == "affect_condition_mask") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_FLOAT32, storage.affect_condition_mask_shape,
                               storage.affect_condition_mask.data(), storage.affect_condition_mask.size() * sizeof(float));
        } else if (semantic == "identity_reference") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_FLOAT32, storage.identity_reference_shape,
                               storage.identity_reference.data(), storage.identity_reference.size() * sizeof(float));
        } else if (semantic == "identity_reference_mask") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_FLOAT32, storage.scalar_shape,
                               storage.identity_reference_mask.data(), sizeof(float));
        } else if (semantic == "prosody_baseline") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_FLOAT32, storage.prosody_baseline_shape,
                               storage.prosody_baseline.data(), storage.prosody_baseline.size() * sizeof(float));
        } else if (semantic == "prosody_delta") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_FLOAT32, storage.prosody_delta_shape,
                               storage.prosody_delta.data(), storage.prosody_delta.size() * sizeof(float));
        } else if (semantic == "prosody_feature_mask") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_FLOAT32, storage.prosody_feature_mask_shape,
                               storage.prosody_feature_mask.data(), storage.prosody_feature_mask.size() * sizeof(float));
        } else if (semantic == "prosody_confidence") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_FLOAT32, storage.scalar_shape,
                               storage.prosody_confidence.data(), sizeof(float));
        } else if (semantic == "reference_style") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_FLOAT32, storage.reference_style_shape,
                               storage.reference_style.data(), storage.reference_style.size() * sizeof(float));
        } else if (semantic == "reference_prosody") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_FLOAT32, storage.reference_prosody_shape,
                               storage.reference_prosody.data(), storage.reference_prosody.size() * sizeof(float));
        } else if (semantic == "reference_mask") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_FLOAT32, storage.scalar_shape,
                               storage.reference_mask.data(), sizeof(float));
        } else if (semantic == "reference_condition_mask") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_FLOAT32, storage.scalar_shape,
                               storage.reference_condition_mask.data(), sizeof(float));
        } else {
            throw std::runtime_error("Unsupported duration_predictor input '" + semantic + "'");
        }
    }
    return storage;
}

ScyllasBandDurationPrediction run_duration_predictor(
    ScyllasBandLiteRtSession* session,
    const ScyllasBandBundleInfo& bundle,
    const ScyllasBandDurationFlowPreparedInputs& prepared,
    int64_t emotion_id,
    float emotion_condition_scale,
    float reference_condition_scale
) {
    ScyllasBandLiteRtInputStorage inputs = build_duration_predictor_inputs(
        bundle,
        prepared,
        emotion_id,
        emotion_condition_scale,
        reference_condition_scale
    );
    ScyllasBandOwnedTensor* outputs = nullptr;
    int32_t output_count = 0;
    const int run_status = scyllasband_litert_session_run(
        session,
        "serving_default",
        inputs.views.data(),
        static_cast<int32_t>(inputs.views.size()),
        &outputs,
        &output_count
    );
    if (run_status != 0) {
        const char* detail = last_error();
        throw std::runtime_error(std::string("duration_predictor failed: ") +
                                 (detail == nullptr ? "" : detail));
    }
    ScyllasBandDurationPrediction prediction;
    try {
        prediction.outputs_json = tensor_outputs_json(outputs, output_count);
        prediction.values = copy_first_float_tensor_values(
            outputs,
            output_count,
            "duration_predictor",
            static_cast<std::size_t>(prepared.phone_count),
            static_cast<std::size_t>(prepared.phone_count)
        );
    } catch (...) {
        scyllasband_tensors_destroy(outputs, output_count);
        throw;
    }
    scyllasband_tensors_destroy(outputs, output_count);
    return prediction;
}

ScyllasBandDurationPrediction predict_duration_values(
    ScyllasBandLiteRtSession* session,
    const ScyllasBandBundleInfo& bundle,
    const ScyllasBandDurationFlowPreparedInputs& prepared,
    const ScyllasBandResolvedRequest& resolved_request
) {
    if (bundle.affect_enabled) {
        if (resolved_request.affect_guidance_scale == 1.0f) {
            return run_duration_predictor(
                session,
                bundle,
                prepared,
                resolved_request.emotion_index,
                1.0f,
                1.0f
            );
        }
        ScyllasBandDurationPrediction null_prediction = run_duration_predictor(
            session, bundle, prepared, resolved_request.emotion_index, 0.0f, 1.0f
        );
        ScyllasBandDurationPrediction conditioned_prediction = run_duration_predictor(
            session, bundle, prepared, resolved_request.emotion_index, 1.0f, 1.0f
        );
        if (null_prediction.values.size() != conditioned_prediction.values.size()) {
            throw std::runtime_error("Affect duration guidance branches returned mismatched shapes");
        }
        std::vector<float> blended(null_prediction.values.size(), 0.0f);
        for (std::size_t index = 0; index < blended.size(); ++index) {
            blended[index] = std::max(
                0.0f,
                null_prediction.values[index] + resolved_request.affect_guidance_scale *
                    (conditioned_prediction.values[index] - null_prediction.values[index])
            );
        }
        ScyllasBandDurationPrediction prediction;
        prediction.values = std::move(blended);
        std::ostringstream metadata;
        metadata << "{\"mode\":\"affect_cfg\",\"branches\":2,"
                 << "\"scale\":" << resolved_request.affect_guidance_scale << ","
                 << "\"reference_retained\":true,"
                 << "\"conditioned_outputs\":" << conditioned_prediction.outputs_json << "}";
        prediction.outputs_json = metadata.str();
        return prediction;
    }
    if (resolved_request.emotion_guidance.empty()) {
        return run_duration_predictor(
            session,
            bundle,
            prepared,
            resolved_request.emotion_index,
            resolved_request.emotion_embed_scale,
            1.0f
        );
    }

    const float null_reference_scale = resolved_request.guidance_null_reference ? 0.0f : 1.0f;
    ScyllasBandDurationPrediction null_prediction = run_duration_predictor(
        session,
        bundle,
        prepared,
        resolved_request.emotion_index,
        0.0f,
        null_reference_scale
    );
    std::vector<float> blended(null_prediction.values.size(), 0.0f);
    for (std::size_t index = 0; index < blended.size(); ++index) {
        blended[index] = resolved_request.emotion_guidance_null_weight * null_prediction.values[index];
    }
    std::string last_outputs = null_prediction.outputs_json;
    int branch_count = 1;
    for (const ScyllasBandEmotionGuidanceTerm& term : resolved_request.emotion_guidance) {
        ScyllasBandDurationPrediction term_prediction = run_duration_predictor(
            session,
            bundle,
            prepared,
            term.emotion_id,
            1.0f,
            1.0f
        );
        last_outputs = term_prediction.outputs_json;
        ++branch_count;
        for (std::size_t index = 0; index < blended.size(); ++index) {
            blended[index] += term.scale * term_prediction.values[index];
        }
    }
    for (float& value : blended) {
        value = std::max(0.0f, value);
    }

    ScyllasBandDurationPrediction prediction;
    prediction.values = std::move(blended);
    std::ostringstream metadata;
    metadata << "{"
             << "\"branches\":" << branch_count << ","
             << "\"null_weight\":" << resolved_request.emotion_guidance_null_weight << ","
             << "\"last_outputs\":" << last_outputs << ","
             << "\"blended_values\":" << float_vector_json(prediction.values)
             << "}";
    prediction.outputs_json = metadata.str();
    return prediction;
}

bool is_silence_phone(const std::string& phone) {
    return phone == "<sil>" || phone == "sil" || phone == "sp" || phone == "<sp>";
}


ScyllasBandDurationExpansionMetadata expand_duration_values(
    const ScyllasBandBundleInfo& bundle,
    const ScyllasBandDurationFlowPreparedInputs& prepared,
    const ScyllasBandSynthesisRequest& request,
    const std::string& language,
    const std::vector<float>& duration_values,
    bool allow_over_budget = false
) {
    if (duration_values.size() < static_cast<std::size_t>(prepared.phone_count)) {
        throw std::runtime_error("duration_predictor output is shorter than the active phone sequence");
    }
    const float speed = request.speed <= 0.0f ? 1.0f : request.speed;
    ScyllasBandDurationExpansionMetadata expanded;
    expanded.duration_scale = 1.0f / speed;
    expanded.duration_values.reserve(static_cast<std::size_t>(prepared.phone_count));
    expanded.predicted_durations.reserve(static_cast<std::size_t>(prepared.phone_count));
    const int64_t sentence_pause_floor = pause_ms_to_latent_frames(
        request.min_sentence_pause_ms,
        bundle.sample_rate,
        bundle.latent_hop_length
    );
    const int64_t clause_pause_floor = pause_ms_to_latent_frames(
        request.min_clause_pause_ms,
        bundle.sample_rate,
        bundle.latent_hop_length
    );
    auto calibrated_floor = [&](const std::string& phone) -> int64_t {
        const auto it = bundle.punctuation_pause_floor_table_ms.find(language + "|" + phone);
        if (it == bundle.punctuation_pause_floor_table_ms.end()) {
            return 0;
        }
        return pause_ms_to_latent_frames(
            it->second,
            bundle.sample_rate,
            bundle.latent_hop_length
        );
    };
    for (int index = 0; index < prepared.phone_count; ++index) {
        const float value = std::max(0.0f, duration_values[static_cast<std::size_t>(index)]);
        int64_t frame_count = static_cast<int64_t>(std::llround(value * expanded.duration_scale));
        const std::string& phone = prepared.phones[static_cast<std::size_t>(index)];
        if (scyllasband_detail::is_non_acoustic_modifier_phone(phone) ||
            (bundle.punctuation_silence_target == "explicit_silence" &&
             scyllasband_detail::is_zero_duration_punctuation_phone(phone))) {
            frame_count = 0;
        } else if (!phone.empty()) {
            frame_count = std::max<int64_t>(1, frame_count);
        }
        const int64_t punctuation_floor = punctuation_duration_floor_frames(
            phone,
            sentence_pause_floor,
            clause_pause_floor,
            calibrated_floor(phone)
        );
        if (punctuation_floor > 0) {
            const bool floor_following_silence = (
                index + 1 < prepared.phone_count &&
                is_silence_phone(prepared.phones[static_cast<std::size_t>(index + 1)])
            );
            if (!floor_following_silence && !is_terminal_pause_target(index, prepared.phone_count)) {
                frame_count = std::max<int64_t>(frame_count, punctuation_floor);
            }
        } else if (is_silence_phone(phone) && index > 0 &&
                   !is_terminal_pause_target(index, prepared.phone_count)) {
            const int64_t preceding_floor = punctuation_duration_floor_frames(
                prepared.phones[static_cast<std::size_t>(index - 1)],
                sentence_pause_floor,
                clause_pause_floor,
                calibrated_floor(prepared.phones[static_cast<std::size_t>(index - 1)])
            );
            if (preceding_floor > 0) {
                frame_count = std::max<int64_t>(frame_count, preceding_floor);
            }
        }
        expanded.duration_values.push_back(value);
        expanded.predicted_durations.push_back(frame_count);
        expanded.predicted_latent_frames += frame_count;
    }
    if (expanded.predicted_latent_frames <= 0) {
        throw std::runtime_error("Predicted zero latent frames; cannot synthesize");
    }
    if (!allow_over_budget && bundle.latent_frames > 0 && expanded.predicted_latent_frames > bundle.latent_frames) {
        throw std::runtime_error(
            "Predicted " + std::to_string(expanded.predicted_latent_frames) +
            " latent frames, but this bundle supports at most " + std::to_string(bundle.latent_frames)
        );
    }
    expanded.expanded_phone_ids.reserve(static_cast<std::size_t>(expanded.predicted_latent_frames));
    for (std::size_t index = 0; index < expanded.predicted_durations.size(); ++index) {
        const int64_t phone_id = prepared.active_phone_ids[index];
        for (int64_t frame = 0; frame < expanded.predicted_durations[index]; ++frame) {
            expanded.expanded_phone_ids.push_back(phone_id);
        }
    }
    return expanded;
}

std::string duration_expansion_json(const ScyllasBandDurationExpansionMetadata& expansion) {
    std::ostringstream metadata;
    metadata << "{"
             << "\"duration_scale\":" << expansion.duration_scale << ","
             << "\"duration_values\":" << float_vector_json(expansion.duration_values) << ","
             << "\"predicted_durations\":" << int64_vector_json(expansion.predicted_durations) << ","
             << "\"predicted_latent_frames\":" << expansion.predicted_latent_frames << ","
             << "\"expanded_phone_count\":" << expansion.expanded_phone_ids.size() << ","
             << "\"expanded_phone_ids_preview\":" << int64_vector_json(expansion.expanded_phone_ids, 32)
             << "}";
    return metadata.str();
}

std::vector<float> initial_latents(
    const ScyllasBandBundleInfo& bundle,
    const ScyllasBandDurationExpansionMetadata& expansion,
    const ScyllasBandSynthesisRequest& request
) {
    const std::size_t count = static_cast<std::size_t>(bundle.latent_dim * bundle.latent_frames);
    std::vector<float> latents(count, 0.0f);
    const float noise_scale = std::isfinite(request.temperature) ? request.temperature : 1.0f;
    std::mt19937_64 rng;
    if (request.has_seed) {
        rng.seed(request.seed);
    } else {
        std::random_device device;
        rng.seed((static_cast<uint64_t>(device()) << 32) ^ static_cast<uint64_t>(device()));
    }
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (int dim = 0; dim < bundle.latent_dim; ++dim) {
        for (int frame = 0; frame < bundle.latent_frames; ++frame) {
            const std::size_t index = static_cast<std::size_t>(dim * bundle.latent_frames + frame);
            if (frame < expansion.predicted_latent_frames) {
                latents[index] = normal(rng) * noise_scale;
            }
        }
    }
    return latents;
}

void apply_latent_mask(
    std::vector<float>& latents,
    const ScyllasBandBundleInfo& bundle,
    const ScyllasBandDurationExpansionMetadata& expansion
) {
    for (int dim = 0; dim < bundle.latent_dim; ++dim) {
        for (int frame = static_cast<int>(expansion.predicted_latent_frames); frame < bundle.latent_frames; ++frame) {
            latents[static_cast<std::size_t>(dim * bundle.latent_frames + frame)] = 0.0f;
        }
    }
}

ScyllasBandLiteRtInputStorage build_vector_estimator_inputs(
    const ScyllasBandBundleInfo& bundle,
    const ScyllasBandDurationFlowPreparedInputs& prepared,
    const ScyllasBandDurationExpansionMetadata& expansion,
    const std::vector<float>& latents,
    float time_value,
    int64_t emotion_id,
    float emotion_condition_scale,
    float reference_condition_scale,
    const std::string& component_name = "vector_estimator",
    const std::vector<float>* hidden = nullptr,
    const std::vector<int64_t>* hidden_shape = nullptr
) {
    const auto inputs_it = bundle.component_inputs.find(component_name);
    if (inputs_it == bundle.component_inputs.end()) {
        throw std::runtime_error("Bundle " + component_name + " component does not declare inputs");
    }
    if (expansion.predicted_latent_frames <= 0 || expansion.predicted_latent_frames > bundle.latent_frames) {
        throw std::runtime_error("Cannot build vector inputs for invalid predicted latent frame count");
    }
    const std::size_t expected_latents = static_cast<std::size_t>(bundle.latent_dim * bundle.latent_frames);
    if (latents.size() != expected_latents) {
        throw std::runtime_error("vector_estimator latents size does not match bundle fixed shape");
    }
    const std::vector<std::string>& semantic_inputs = inputs_it->second;

    ScyllasBandLiteRtInputStorage storage;
    storage.names.reserve(semantic_inputs.size());
    storage.views.reserve(semantic_inputs.size());
    storage.scalar_shape = {1};
    storage.affect_shape = {1, static_cast<int64_t>(prepared.affect_values.size())};
    storage.affect_condition_mask_shape = {1, static_cast<int64_t>(prepared.affect_condition_mask_values.size())};
    storage.reference_style_shape = {1, static_cast<int64_t>(prepared.reference_style.size())};
    storage.reference_prosody_shape = {1, static_cast<int64_t>(prepared.reference_prosody.size())};
    storage.identity_reference_shape = {1, static_cast<int64_t>(prepared.identity_reference.size())};
    storage.prosody_baseline_shape = {1, static_cast<int64_t>(prepared.prosody_baseline.size())};
    storage.prosody_delta_shape = {1, static_cast<int64_t>(prepared.prosody_delta.size())};
    storage.prosody_feature_mask_shape = {1, static_cast<int64_t>(prepared.prosody_feature_mask.size())};
    storage.latent_shape = {1, static_cast<int64_t>(bundle.latent_dim), static_cast<int64_t>(bundle.latent_frames)};
    storage.latent_frame_shape = {1, static_cast<int64_t>(bundle.latent_frames)};
    if (hidden != nullptr && hidden_shape != nullptr) {
        storage.hidden = *hidden;
        storage.hidden_shape = *hidden_shape;
    }
    const int prefix_frames = std::max(0, bundle.prefix_max_frames);
    storage.prefix_latents_shape = {1, static_cast<int64_t>(bundle.latent_dim), static_cast<int64_t>(prefix_frames)};
    storage.prefix_mask_shape = {1, static_cast<int64_t>(prefix_frames)};
    storage.latents = latents;
    storage.time = {time_value};
    storage.expanded_phone_ids.assign(static_cast<std::size_t>(bundle.latent_frames), 0);
    const std::size_t expanded_count = std::min(
        expansion.expanded_phone_ids.size(),
        static_cast<std::size_t>(bundle.latent_frames)
    );
    std::copy_n(expansion.expanded_phone_ids.begin(), expanded_count, storage.expanded_phone_ids.begin());
    storage.latent_mask.assign(static_cast<std::size_t>(bundle.latent_frames), 0);
    for (int64_t frame = 0; frame < expansion.predicted_latent_frames; ++frame) {
        storage.latent_mask[static_cast<std::size_t>(frame)] = 1;
    }
    storage.voice_id = {prepared.voice_id};
    storage.language_id = {prepared.language_id};
    storage.emotion_id = {emotion_id};
    storage.boundary_before_id = {prepared.boundary_before_id};
    storage.boundary_after_id = {prepared.boundary_after_id};
    storage.emotion_condition_mask = {emotion_condition_scale};
    storage.affect_values = prepared.affect_values;
    storage.affect_condition_mask = prepared.affect_condition_mask_values;
    for (float& value : storage.affect_condition_mask) {
        value *= emotion_condition_scale;
    }
    storage.reference_style = prepared.reference_style;
    storage.reference_prosody = prepared.reference_prosody;
    storage.reference_mask = {prepared.reference_mask};
    storage.identity_reference = prepared.identity_reference;
    storage.identity_reference_mask = {
        prepared.identity_reference_mask * reference_condition_scale
    };
    storage.prosody_baseline = prepared.prosody_baseline;
    storage.prosody_delta = prepared.prosody_delta;
    storage.prosody_feature_mask = prepared.prosody_feature_mask;
    storage.prosody_confidence = {prepared.prosody_confidence};
    storage.reference_condition_mask = {reference_condition_scale};
    storage.prefix_latents = prepared.prefix_latents;
    storage.prefix_mask = prepared.prefix_mask;
    const int span_hidden_size = std::max(0, bundle.span_context_hidden_size);
    storage.span_context_hidden_shape = {1, static_cast<int64_t>(span_hidden_size)};
    storage.span_context_hidden.assign(static_cast<std::size_t>(span_hidden_size), 0.0f);
    if (span_hidden_size > 0 && !prepared.span_context_hidden.empty()) {
        const std::size_t count = std::min(
            storage.span_context_hidden.size(),
            prepared.span_context_hidden.size()
        );
        std::copy_n(prepared.span_context_hidden.begin(), count, storage.span_context_hidden.begin());
    }

    for (std::size_t index = 0; index < semantic_inputs.size(); ++index) {
        const std::string raw_name = "args_" + std::to_string(index);
        const std::string& semantic = semantic_inputs[index];
        if (semantic == "hidden") {
            if (storage.hidden.empty() || storage.hidden_shape.empty()) {
                throw std::runtime_error(component_name + " requires hidden input from vector prefix");
            }
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_FLOAT32, storage.hidden_shape,
                               storage.hidden.data(), storage.hidden.size() * sizeof(float));
        } else if (semantic == "noise") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_FLOAT32, storage.latent_shape,
                               storage.latents.data(), storage.latents.size() * sizeof(float));
        } else if (semantic == "time") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_FLOAT32, storage.scalar_shape,
                               storage.time.data(), sizeof(float));
        } else if (semantic == "expanded_phone_ids") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_INT64, storage.latent_frame_shape,
                               storage.expanded_phone_ids.data(), storage.expanded_phone_ids.size() * sizeof(int64_t));
        } else if (semantic == "voice_id") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_INT64, storage.scalar_shape,
                               storage.voice_id.data(), sizeof(int64_t));
        } else if (semantic == "language_id") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_INT64, storage.scalar_shape,
                               storage.language_id.data(), sizeof(int64_t));
        } else if (semantic == "emotion_id") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_INT64, storage.scalar_shape,
                               storage.emotion_id.data(), sizeof(int64_t));
        } else if (semantic == "boundary_before_id") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_INT64, storage.scalar_shape,
                               storage.boundary_before_id.data(), sizeof(int64_t));
        } else if (semantic == "boundary_after_id") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_INT64, storage.scalar_shape,
                               storage.boundary_after_id.data(), sizeof(int64_t));
        } else if (semantic == "latent_mask") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_BOOL, storage.latent_frame_shape,
                               storage.latent_mask.data(), storage.latent_mask.size() * sizeof(uint8_t));
        } else if (semantic == "emotion_condition_mask") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_FLOAT32, storage.scalar_shape,
                               storage.emotion_condition_mask.data(), sizeof(float));
        } else if (semantic == "affect_values") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_FLOAT32, storage.affect_shape,
                               storage.affect_values.data(), storage.affect_values.size() * sizeof(float));
        } else if (semantic == "affect_condition_mask") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_FLOAT32, storage.affect_condition_mask_shape,
                               storage.affect_condition_mask.data(), storage.affect_condition_mask.size() * sizeof(float));
} else if (semantic == "identity_reference") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_FLOAT32, storage.identity_reference_shape,
                               storage.identity_reference.data(), storage.identity_reference.size() * sizeof(float));
        } else if (semantic == "identity_reference_mask") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_FLOAT32, storage.scalar_shape,
                               storage.identity_reference_mask.data(), sizeof(float));
        } else if (semantic == "prosody_baseline") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_FLOAT32, storage.prosody_baseline_shape,
                               storage.prosody_baseline.data(), storage.prosody_baseline.size() * sizeof(float));
        } else if (semantic == "prosody_delta") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_FLOAT32, storage.prosody_delta_shape,
                               storage.prosody_delta.data(), storage.prosody_delta.size() * sizeof(float));
        } else if (semantic == "prosody_feature_mask") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_FLOAT32, storage.prosody_feature_mask_shape,
                               storage.prosody_feature_mask.data(), storage.prosody_feature_mask.size() * sizeof(float));
        } else if (semantic == "prosody_confidence") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_FLOAT32, storage.scalar_shape,
                               storage.prosody_confidence.data(), sizeof(float));
        } else if (semantic == "reference_style") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_FLOAT32, storage.reference_style_shape,
                               storage.reference_style.data(), storage.reference_style.size() * sizeof(float));
        } else if (semantic == "reference_prosody") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_FLOAT32, storage.reference_prosody_shape,
                               storage.reference_prosody.data(), storage.reference_prosody.size() * sizeof(float));
        } else if (semantic == "reference_mask") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_FLOAT32, storage.scalar_shape,
                               storage.reference_mask.data(), sizeof(float));
        } else if (semantic == "reference_condition_mask") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_FLOAT32, storage.scalar_shape,
                               storage.reference_condition_mask.data(), sizeof(float));
        } else if (semantic == "prefix_latents") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_FLOAT32, storage.prefix_latents_shape,
                               storage.prefix_latents.data(), storage.prefix_latents.size() * sizeof(float));
        } else if (semantic == "prefix_mask") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_BOOL, storage.prefix_mask_shape,
                               storage.prefix_mask.data(), storage.prefix_mask.size() * sizeof(uint8_t));
        } else if (semantic == "span_context_hidden") {
            if (storage.span_context_hidden.empty() || storage.span_context_hidden_shape.empty()) {
                throw std::runtime_error(component_name + " requires span_context_hidden but bundle span hidden size is not declared");
            }
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_FLOAT32, storage.span_context_hidden_shape,
                               storage.span_context_hidden.data(), storage.span_context_hidden.size() * sizeof(float));
        } else {
            throw std::runtime_error("Unsupported " + component_name + " input '" + semantic + "'");
        }
    }
    return storage;
}

ScyllasBandVectorVelocityResult vector_velocity(
    ScyllasBandLiteRtSession* session,
    const ScyllasBandBundleInfo& bundle,
    const ScyllasBandDurationFlowPreparedInputs& prepared,
    const ScyllasBandDurationExpansionMetadata& expansion,
    const std::vector<float>& latents,
    float time_value,
    int64_t emotion_id,
    float emotion_condition_scale,
    float reference_condition_scale
) {
    ScyllasBandLiteRtInputStorage inputs = build_vector_estimator_inputs(
        bundle,
        prepared,
        expansion,
        latents,
        time_value,
        emotion_id,
        emotion_condition_scale,
        reference_condition_scale
    );
    ScyllasBandOwnedTensor* outputs = nullptr;
    int32_t output_count = 0;
    const int run_status = scyllasband_litert_session_run(
        session,
        "serving_default",
        inputs.views.data(),
        static_cast<int32_t>(inputs.views.size()),
        &outputs,
        &output_count
    );
    if (run_status != 0) {
        const char* detail = last_error();
        throw std::runtime_error(std::string("vector_estimator failed: ") +
                                 (detail == nullptr ? "" : detail));
    }
    ScyllasBandVectorVelocityResult result;
    try {
        result.outputs_json = tensor_outputs_json(outputs, output_count);
        result.values = copy_first_float_tensor_values(
            outputs,
            output_count,
            "vector_estimator",
            static_cast<std::size_t>(bundle.latent_dim * bundle.latent_frames),
            static_cast<std::size_t>(bundle.latent_dim * bundle.latent_frames)
        );
    } catch (...) {
        scyllasband_tensors_destroy(outputs, output_count);
        throw;
    }
    scyllasband_tensors_destroy(outputs, output_count);
    return result;
}

ScyllasBandVectorPrefixResult vector_prefix_hidden(
    ScyllasBandLiteRtSession* session,
    const ScyllasBandBundleInfo& bundle,
    const ScyllasBandDurationFlowPreparedInputs& prepared,
    const ScyllasBandDurationExpansionMetadata& expansion,
    const std::vector<float>& latents,
    float time_value,
    int64_t emotion_id,
    float emotion_condition_scale,
    float reference_condition_scale
) {
    ScyllasBandLiteRtInputStorage inputs = build_vector_estimator_inputs(
        bundle,
        prepared,
        expansion,
        latents,
        time_value,
        emotion_id,
        emotion_condition_scale,
        reference_condition_scale,
        "vector_estimator_prefix"
    );
    ScyllasBandOwnedTensor* outputs = nullptr;
    int32_t output_count = 0;
    const int run_status = scyllasband_litert_session_run(
        session,
        "serving_default",
        inputs.views.data(),
        static_cast<int32_t>(inputs.views.size()),
        &outputs,
        &output_count
    );
    if (run_status != 0) {
        const char* detail = last_error();
        throw std::runtime_error(std::string("vector_estimator_prefix failed: ") +
                                 (detail == nullptr ? "" : detail));
    }
    ScyllasBandVectorPrefixResult result;
    try {
        if (output_count <= 0 || outputs == nullptr) {
            throw std::runtime_error("vector_estimator_prefix returned no tensors");
        }
        const ScyllasBandOwnedTensor& tensor = outputs[0];
        if (tensor.shape == nullptr || tensor.rank <= 0) {
            throw std::runtime_error("vector_estimator_prefix output_0 has no shape");
        }
        result.shape.assign(tensor.shape, tensor.shape + tensor.rank);
        result.outputs_json = tensor_outputs_json(outputs, output_count);
        result.values = copy_first_float_tensor_values(outputs, output_count, "vector_estimator_prefix");
    } catch (...) {
        scyllasband_tensors_destroy(outputs, output_count);
        throw;
    }
    scyllasband_tensors_destroy(outputs, output_count);
    return result;
}

ScyllasBandVectorVelocityResult vector_velocity_split(
    ScyllasBandLiteRtSession* prefix_session,
    ScyllasBandLiteRtSession* tail_session,
    const ScyllasBandBundleInfo& bundle,
    const ScyllasBandDurationFlowPreparedInputs& prepared,
    const ScyllasBandDurationExpansionMetadata& expansion,
    const std::vector<float>& latents,
    float time_value,
    int64_t emotion_id,
    float emotion_condition_scale,
    float reference_condition_scale
) {
    ScyllasBandVectorPrefixResult prefix = vector_prefix_hidden(
        prefix_session,
        bundle,
        prepared,
        expansion,
        latents,
        time_value,
        emotion_id,
        emotion_condition_scale,
        reference_condition_scale
    );
    ScyllasBandLiteRtInputStorage inputs = build_vector_estimator_inputs(
        bundle,
        prepared,
        expansion,
        latents,
        time_value,
        emotion_id,
        emotion_condition_scale,
        reference_condition_scale,
        "vector_estimator_tail",
        &prefix.values,
        &prefix.shape
    );
    ScyllasBandOwnedTensor* outputs = nullptr;
    int32_t output_count = 0;
    const int run_status = scyllasband_litert_session_run(
        tail_session,
        "serving_default",
        inputs.views.data(),
        static_cast<int32_t>(inputs.views.size()),
        &outputs,
        &output_count
    );
    if (run_status != 0) {
        const char* detail = last_error();
        throw std::runtime_error(std::string("vector_estimator_tail failed: ") +
                                 (detail == nullptr ? "" : detail));
    }
    ScyllasBandVectorVelocityResult result;
    try {
        const std::string tail_outputs_json = tensor_outputs_json(outputs, output_count);
        result.outputs_json = std::string("{\"mode\":\"split_prefix_tail\",\"prefix\":") +
                              prefix.outputs_json + ",\"tail\":" + tail_outputs_json + "}";
        result.values = copy_first_float_tensor_values(
            outputs,
            output_count,
            "vector_estimator_tail",
            static_cast<std::size_t>(bundle.latent_dim * bundle.latent_frames),
            static_cast<std::size_t>(bundle.latent_dim * bundle.latent_frames)
        );
    } catch (...) {
        scyllasband_tensors_destroy(outputs, output_count);
        throw;
    }
    scyllasband_tensors_destroy(outputs, output_count);
    return result;
}

ScyllasBandVectorVelocityResult vector_velocity_for_runtime(
    ScyllasBandLiteRtSession* session,
    ScyllasBandLiteRtSession* prefix_session,
    ScyllasBandLiteRtSession* tail_session,
    bool split_vector,
    const ScyllasBandBundleInfo& bundle,
    const ScyllasBandDurationFlowPreparedInputs& prepared,
    const ScyllasBandDurationExpansionMetadata& expansion,
    const std::vector<float>& latents,
    float time_value,
    int64_t emotion_id,
    float emotion_condition_scale,
    float reference_condition_scale
) {
    if (split_vector) {
        return vector_velocity_split(
            prefix_session,
            tail_session,
            bundle,
            prepared,
            expansion,
            latents,
            time_value,
            emotion_id,
            emotion_condition_scale,
            reference_condition_scale
        );
    }
    return vector_velocity(
        session,
        bundle,
        prepared,
        expansion,
        latents,
        time_value,
        emotion_id,
        emotion_condition_scale,
        reference_condition_scale
    );
}

ScyllasBandVectorVelocityResult guided_vector_velocity(
    ScyllasBandLiteRtSession* session,
    ScyllasBandLiteRtSession* prefix_session,
    ScyllasBandLiteRtSession* tail_session,
    bool split_vector,
    const ScyllasBandBundleInfo& bundle,
    const ScyllasBandDurationFlowPreparedInputs& prepared,
    const ScyllasBandDurationExpansionMetadata& expansion,
    const ScyllasBandResolvedRequest& resolved_request,
    const std::vector<float>& latents,
    float time_value
) {
    if (bundle.affect_enabled) {
        if (resolved_request.affect_guidance_scale == 1.0f) {
            return vector_velocity_for_runtime(
                session, prefix_session, tail_session, split_vector,
                bundle, prepared, expansion, latents, time_value,
                resolved_request.emotion_index, 1.0f, 1.0f
            );
        }
        ScyllasBandVectorVelocityResult null_velocity = vector_velocity_for_runtime(
            session,
            prefix_session,
            tail_session,
            split_vector,
            bundle,
            prepared,
            expansion,
            latents,
            time_value,
            resolved_request.emotion_index,
            0.0f,
            1.0f
        );
        ScyllasBandVectorVelocityResult conditioned_velocity = vector_velocity_for_runtime(
            session,
            prefix_session,
            tail_session,
            split_vector,
            bundle,
            prepared,
            expansion,
            latents,
            time_value,
            resolved_request.emotion_index,
            1.0f,
            1.0f
        );
        if (null_velocity.values.size() != conditioned_velocity.values.size()) {
            throw std::runtime_error("Affect vector guidance branches returned mismatched shapes");
        }
        std::vector<float> blended(null_velocity.values.size(), 0.0f);
        for (std::size_t index = 0; index < blended.size(); ++index) {
            blended[index] = null_velocity.values[index] + resolved_request.affect_guidance_scale *
                (conditioned_velocity.values[index] - null_velocity.values[index]);
        }
        ScyllasBandVectorVelocityResult result;
        result.values = std::move(blended);
        std::ostringstream metadata;
        metadata << "{\"mode\":\"affect_cfg\",\"branches\":2,"
                 << "\"scale\":" << resolved_request.affect_guidance_scale << ","
                 << "\"reference_retained\":true,"
                 << "\"conditioned_outputs\":" << conditioned_velocity.outputs_json << "}";
        result.outputs_json = metadata.str();
        return result;
    }
    if (resolved_request.emotion_guidance.empty()) {
        return vector_velocity_for_runtime(
            session,
            prefix_session,
            tail_session,
            split_vector,
            bundle,
            prepared,
            expansion,
            latents,
            time_value,
            resolved_request.emotion_index,
            resolved_request.emotion_embed_scale,
            1.0f
        );
    }

    const float null_reference_scale = resolved_request.guidance_null_reference ? 0.0f : 1.0f;
    ScyllasBandVectorVelocityResult null_velocity = vector_velocity_for_runtime(
        session,
        prefix_session,
        tail_session,
        split_vector,
        bundle,
        prepared,
        expansion,
        latents,
        time_value,
        resolved_request.emotion_index,
        0.0f,
        null_reference_scale
    );
    std::vector<float> blended(null_velocity.values.size(), 0.0f);
    for (std::size_t index = 0; index < blended.size(); ++index) {
        blended[index] = resolved_request.emotion_guidance_null_weight * null_velocity.values[index];
    }
    std::string last_outputs = null_velocity.outputs_json;
    for (const ScyllasBandEmotionGuidanceTerm& term : resolved_request.emotion_guidance) {
        ScyllasBandVectorVelocityResult term_velocity = vector_velocity_for_runtime(
            session,
            prefix_session,
            tail_session,
            split_vector,
            bundle,
            prepared,
            expansion,
            latents,
            time_value,
            term.emotion_id,
            resolved_request.emotion_embed_scale,
            1.0f
        );
        last_outputs = term_velocity.outputs_json;
        for (std::size_t index = 0; index < blended.size(); ++index) {
            blended[index] += term.scale * term_velocity.values[index];
        }
    }
    ScyllasBandVectorVelocityResult result;
    result.values = std::move(blended);
    result.outputs_json = last_outputs;
    return result;
}

ScyllasBandLatentSamplingResult sample_latents(
    ScyllasBandLiteRtSession* session,
    ScyllasBandLiteRtSession* prefix_session,
    ScyllasBandLiteRtSession* tail_session,
    bool split_vector,
    const ScyllasBandBundleInfo& bundle,
    const ScyllasBandDurationFlowPreparedInputs& prepared,
    const ScyllasBandDurationExpansionMetadata& expansion,
    const ScyllasBandResolvedRequest& resolved_request,
    const ScyllasBandSynthesisRequest& request
) {
    const int steps = std::max<int>(1, request.steps);
    const float dt = 1.0f / static_cast<float>(steps);
    ScyllasBandLatentSamplingResult result;
    result.latents = initial_latents(bundle, expansion, request);
    const std::size_t latent_count = result.latents.size();

    for (int step = 0; step < steps; ++step) {
        if (request.sampler == SCYLLASBAND_SAMPLER_HEUN) {
            const float start_time = static_cast<float>(step) / static_cast<float>(steps);
            const float end_time = static_cast<float>(step + 1) / static_cast<float>(steps);
            ScyllasBandVectorVelocityResult start_velocity = guided_vector_velocity(
                session,
                prefix_session,
                tail_session,
                split_vector,
                bundle,
                prepared,
                expansion,
                resolved_request,
                result.latents,
                start_time
            );
            ++result.vector_calls;
            result.last_vector_outputs_json = start_velocity.outputs_json;
            std::vector<float> predicted = result.latents;
            for (std::size_t index = 0; index < latent_count; ++index) {
                predicted[index] += dt * start_velocity.values[index];
            }
            apply_latent_mask(predicted, bundle, expansion);
            ScyllasBandVectorVelocityResult end_velocity = guided_vector_velocity(
                session,
                prefix_session,
                tail_session,
                split_vector,
                bundle,
                prepared,
                expansion,
                resolved_request,
                predicted,
                end_time
            );
            ++result.vector_calls;
            result.last_vector_outputs_json = end_velocity.outputs_json;
            for (std::size_t index = 0; index < latent_count; ++index) {
                result.latents[index] += 0.5f * dt * (start_velocity.values[index] + end_velocity.values[index]);
            }
            apply_latent_mask(result.latents, bundle, expansion);
        } else {
            const float time_value = (static_cast<float>(step) + 0.5f) / static_cast<float>(steps);
            ScyllasBandVectorVelocityResult velocity = guided_vector_velocity(
                session,
                prefix_session,
                tail_session,
                split_vector,
                bundle,
                prepared,
                expansion,
                resolved_request,
                result.latents,
                time_value
            );
            ++result.vector_calls;
            result.last_vector_outputs_json = velocity.outputs_json;
            for (std::size_t index = 0; index < latent_count; ++index) {
                result.latents[index] += dt * velocity.values[index];
            }
            apply_latent_mask(result.latents, bundle, expansion);
        }
    }
    return result;
}

ScyllasBandLiteRtInputStorage build_vocoder_inputs(
    const ScyllasBandBundleInfo& bundle,
    const ScyllasBandDurationFlowPreparedInputs& prepared,
    const std::vector<float>& latents,
    int64_t emotion_id,
    int64_t predicted_latent_frames
) {
    const auto inputs_it = bundle.component_inputs.find("vocoder");
    if (inputs_it == bundle.component_inputs.end()) {
        throw std::runtime_error("Bundle vocoder component does not declare inputs");
    }
    const std::size_t expected_latents = static_cast<std::size_t>(bundle.latent_dim * bundle.latent_frames);
    if (latents.size() != expected_latents) {
        throw std::runtime_error("vocoder latents size does not match bundle fixed shape");
    }
    const std::vector<std::string>& semantic_inputs = inputs_it->second;

    ScyllasBandLiteRtInputStorage storage;
    storage.names.reserve(semantic_inputs.size());
    storage.views.reserve(semantic_inputs.size());
    storage.scalar_shape = {1};
    storage.reference_style_shape = {1, static_cast<int64_t>(prepared.reference_style.size())};
    storage.reference_prosody_shape = {1, static_cast<int64_t>(prepared.reference_prosody.size())};
    storage.latent_shape = {1, static_cast<int64_t>(bundle.latent_dim), static_cast<int64_t>(bundle.latent_frames)};
    storage.latent_frame_shape = {1, static_cast<int64_t>(bundle.latent_frames)};
    storage.latents = latents;
    storage.latent_mask.assign(static_cast<std::size_t>(bundle.latent_frames), 0);
    const int64_t valid_latent_frames = std::max<int64_t>(
        0,
        std::min<int64_t>(predicted_latent_frames, bundle.latent_frames)
    );
    for (int64_t frame = 0; frame < valid_latent_frames; ++frame) {
        storage.latent_mask[static_cast<std::size_t>(frame)] = 1;
    }
    storage.voice_id = {prepared.voice_id};
    storage.language_id = {prepared.language_id};
    storage.emotion_id = {emotion_id};
    storage.reference_style = prepared.reference_style;
    storage.reference_prosody = prepared.reference_prosody;
    storage.reference_mask = {prepared.reference_mask};
    storage.reference_condition_mask = {1.0f};

    for (std::size_t index = 0; index < semantic_inputs.size(); ++index) {
        const std::string raw_name = "args_" + std::to_string(index);
        const std::string& semantic = semantic_inputs[index];
        if (semantic == "latents") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_FLOAT32, storage.latent_shape,
                               storage.latents.data(), storage.latents.size() * sizeof(float));
        } else if (semantic == "latent_mask") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_BOOL, storage.latent_frame_shape,
                               storage.latent_mask.data(), storage.latent_mask.size() * sizeof(uint8_t));
        } else if (semantic == "voice_id") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_INT64, storage.scalar_shape,
                               storage.voice_id.data(), sizeof(int64_t));
        } else if (semantic == "language_id") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_INT64, storage.scalar_shape,
                               storage.language_id.data(), sizeof(int64_t));
        } else if (semantic == "emotion_id") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_INT64, storage.scalar_shape,
                               storage.emotion_id.data(), sizeof(int64_t));
        } else if (semantic == "reference_style") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_FLOAT32, storage.reference_style_shape,
                               storage.reference_style.data(), storage.reference_style.size() * sizeof(float));
        } else if (semantic == "reference_prosody") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_FLOAT32, storage.reference_prosody_shape,
                               storage.reference_prosody.data(), storage.reference_prosody.size() * sizeof(float));
        } else if (semantic == "reference_mask") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_FLOAT32, storage.scalar_shape,
                               storage.reference_mask.data(), sizeof(float));
        } else if (semantic == "reference_condition_mask") {
            append_tensor_view(storage, raw_name, SCYLLASBAND_TENSOR_FLOAT32, storage.scalar_shape,
                               storage.reference_condition_mask.data(), sizeof(float));
        } else {
            throw std::runtime_error("Unsupported vocoder input '" + semantic + "'");
        }
    }
    return storage;
}

ScyllasBandVocoderResult run_vocoder(
    ScyllasBandLiteRtSession* session,
    const ScyllasBandBundleInfo& bundle,
    const ScyllasBandDurationFlowPreparedInputs& prepared,
    const std::vector<float>& latents,
    int64_t emotion_id,
    const ScyllasBandDurationExpansionMetadata& expansion
) {
    ScyllasBandLiteRtInputStorage inputs = build_vocoder_inputs(
        bundle,
        prepared,
        latents,
        emotion_id,
        expansion.predicted_latent_frames
    );
    ScyllasBandOwnedTensor* outputs = nullptr;
    int32_t output_count = 0;
    const int run_status = scyllasband_litert_session_run(
        session,
        "serving_default",
        inputs.views.data(),
        static_cast<int32_t>(inputs.views.size()),
        &outputs,
        &output_count
    );
    if (run_status != 0) {
        const char* detail = last_error();
        throw std::runtime_error(std::string("vocoder failed: ") +
                                 (detail == nullptr ? "" : detail));
    }
    ScyllasBandVocoderResult result;
    try {
        result.outputs_json = tensor_outputs_json(outputs, output_count);
        result.audio = copy_first_float_tensor_values(outputs, output_count, "vocoder");
    } catch (...) {
        scyllasband_tensors_destroy(outputs, output_count);
        throw;
    }
    scyllasband_tensors_destroy(outputs, output_count);
    return result;
}

void copy_audio_to_result(
    ScyllasBandSynthesisResult* out_result,
    const ScyllasBandBundleInfo& bundle,
    const ScyllasBandDurationExpansionMetadata& expansion,
    const std::vector<float>& audio,
    const std::vector<float>& latents
) {
    const int64_t trim_samples = std::max<int64_t>(
        1,
        (expansion.predicted_latent_frames * 2 - 1) * static_cast<int64_t>(bundle.hop_length)
    );
    const std::size_t sample_count = std::min<std::size_t>(
        audio.size(),
        static_cast<std::size_t>(trim_samples)
    );
    if (sample_count > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("vocoder output is too large for ScyllasBandSynthesisResult");
    }
    auto* samples = static_cast<float*>(std::malloc(sample_count * sizeof(float)));
    if (samples == nullptr && sample_count > 0) {
        throw std::runtime_error("malloc failed for ScyllasBandSynthesisResult samples");
    }
    for (std::size_t index = 0; index < sample_count; ++index) {
        if (!std::isfinite(audio[index])) {
            std::free(samples);
            throw std::runtime_error("vocoder audio contains non-finite values");
        }
        samples[index] = std::max(-1.0f, std::min(1.0f, audio[index]));
    }
    float* latent_copy = nullptr;
    const int32_t latent_dim = static_cast<int32_t>(std::max(0, bundle.latent_dim));
    const int32_t latent_frames = static_cast<int32_t>(std::max<int64_t>(0, expansion.predicted_latent_frames));
    const std::size_t latent_count = static_cast<std::size_t>(latent_dim) * static_cast<std::size_t>(latent_frames);
    if (latent_count > 0) {
        const std::size_t fixed_frames = static_cast<std::size_t>(std::max(0, bundle.latent_frames));
        const std::size_t expected_fixed = static_cast<std::size_t>(latent_dim) * fixed_frames;
        if (latents.size() < expected_fixed || fixed_frames < static_cast<std::size_t>(latent_frames)) {
            std::free(samples);
            throw std::runtime_error("sampled latents do not match bundle fixed shape");
        }
        latent_copy = static_cast<float*>(std::malloc(latent_count * sizeof(float)));
        if (latent_copy == nullptr) {
            std::free(samples);
            throw std::runtime_error("malloc failed for ScyllasBandSynthesisResult latents");
        }
        for (int32_t dim = 0; dim < latent_dim; ++dim) {
            for (int32_t frame = 0; frame < latent_frames; ++frame) {
                latent_copy[static_cast<std::size_t>(dim) * latent_frames + frame] =
                    latents[static_cast<std::size_t>(dim) * fixed_frames + frame];
            }
        }
    }
    out_result->samples = samples;
    out_result->sample_count = static_cast<int32_t>(sample_count);
    out_result->sample_rate = bundle.sample_rate;
    out_result->latents = latent_copy;
    out_result->latent_dim = latent_dim;
    out_result->latent_frames = latent_frames;
}

std::string sampler_json(
    const ScyllasBandLatentSamplingResult& sampling,
    const ScyllasBandSynthesisRequest& request
) {
    std::ostringstream metadata;
    metadata << "{"
             << "\"steps\":" << std::max<int>(1, request.steps) << ","
             << "\"sampler\":\"" << (request.sampler == SCYLLASBAND_SAMPLER_HEUN ? "heun" : "euler") << "\","
             << "\"vector_calls\":" << sampling.vector_calls << ","
             << "\"seed\":" << request.seed << ","
             << "\"has_seed\":" << (request.has_seed ? "true" : "false") << ","
             << "\"noise_scale\":" << request.temperature
             << "}";
    return metadata.str();
}

class DurationFlowBackend final : public ScyllasBandBackendEngine {
public:
    DurationFlowBackend(
        std::string bundle_dir,
        ScyllasBandBundleInfo bundle_info,
        std::string backend_name,
        ScyllasBandLiteRtAccelerator litert_accelerator,
        int litert_max_threads
    )
        : bundle_dir_(std::move(bundle_dir)),
          bundle_info_(std::move(bundle_info)),
          backend_name_(std::move(backend_name)),
          litert_accelerator_(litert_accelerator),
          litert_max_threads_(litert_max_threads) {}

    ~DurationFlowBackend() override {
        if (g2p_session_ != nullptr) {
            scyllasband_litert_session_destroy(g2p_session_);
            g2p_session_ = nullptr;
        }
        if (vector_context_session_ != nullptr) {
            scyllasband_litert_session_destroy(vector_context_session_);
            vector_context_session_ = nullptr;
        }
        if (duration_session_ != nullptr) {
            scyllasband_litert_session_destroy(duration_session_);
            duration_session_ = nullptr;
        }
        destroy_session_map(vector_sessions_);
        destroy_session_map(vector_prefix_sessions_);
        destroy_session_map(vector_tail_sessions_);
        destroy_session_map(vocoder_sessions_);
    }

    ScyllasBandStatus synthesize(
        const ScyllasBandSynthesisRequest& request,
        ScyllasBandSynthesisResult* out_result
    ) override {
        if (out_result == nullptr) {
            set_error("scyllasband " + backend_name_ + " backend synthesize: output result is required");
            return SCYLLASBAND_STATUS_INVALID_ARGUMENT;
        }
        out_result->samples = nullptr;
        out_result->sample_count = 0;
        out_result->sample_rate = 0;

        const auto total_started_at = ScyllasBandSteadyClock::now();
        int64_t resolve_plan_ms = 0;
        int64_t ensure_g2p_session_ms = 0;
        int64_t g2p_ms = 0;
        int64_t prepare_inputs_ms = 0;
        int64_t reference_features_ms = 0;
        int64_t span_context_ms = 0;
        int64_t ensure_duration_session_ms = 0;
        int64_t ensure_vector_session_ms = 0;
        int64_t ensure_vocoder_session_ms = 0;
        int64_t duration_predict_ms = 0;
        int64_t duration_expand_ms = 0;
        int64_t vector_sample_ms = 0;
        int64_t vocoder_ms = 0;
        int64_t copy_audio_ms = 0;
        int vector_calls = 0;
        std::size_t sampled_latent_values = 0;

        ScyllasBandResolvedRequest resolved_request;
        ScyllasBandDurationFlowExecutionPlan plan;
        ScyllasBandDurationFlowPreparedInputs prepared_inputs;
        ScyllasBandSynthesisRequest effective_request = request;
        std::string generated_phone_text;
        std::string g2p_metadata = "null";
        try {
            auto stage_started_at = ScyllasBandSteadyClock::now();
            resolved_request = resolve_scyllasband_request_context(bundle_info_, request);
            plan = build_scyllasband_duration_flow_plan(bundle_info_, resolved_request);
            resolve_plan_ms = elapsed_ms_backend(stage_started_at, ScyllasBandSteadyClock::now());
            if (!resolved_request.has_explicit_phones) {
                stage_started_at = ScyllasBandSteadyClock::now();
                ensure_g2p_session();
                ensure_g2p_session_ms = elapsed_ms_backend(stage_started_at, ScyllasBandSteadyClock::now());
                stage_started_at = ScyllasBandSteadyClock::now();
                ScyllasBandG2PResult g2p = run_g2p_text(
                    g2p_session_,
                    bundle_info_,
                    resolved_request,
                    request,
                    pronunciation_overrides()
                );
                g2p_ms = elapsed_ms_backend(stage_started_at, ScyllasBandSteadyClock::now());
                generated_phone_text = join_phone_tokens(g2p.phones);
                effective_request.explicit_phones = generated_phone_text.c_str();
                g2p_metadata = g2p.metadata_json;
            }
            stage_started_at = ScyllasBandSteadyClock::now();
            prepared_inputs = prepare_scyllasband_duration_flow_inputs(bundle_info_, resolved_request, effective_request);
            if (!generated_phone_text.empty()) {
                prepared_inputs.phone_source = "g2p";
            }
            prepare_inputs_ms = elapsed_ms_backend(stage_started_at, ScyllasBandSteadyClock::now());
            stage_started_at = ScyllasBandSteadyClock::now();
            apply_reference_features(prepared_inputs, resolved_request);
            reference_features_ms = elapsed_ms_backend(stage_started_at, ScyllasBandSteadyClock::now());
            stage_started_at = ScyllasBandSteadyClock::now();
            apply_span_context_features(prepared_inputs, resolved_request, request);
            span_context_ms = elapsed_ms_backend(stage_started_at, ScyllasBandSteadyClock::now());
            stage_started_at = ScyllasBandSteadyClock::now();
            ensure_duration_session();
            ensure_duration_session_ms = elapsed_ms_backend(stage_started_at, ScyllasBandSteadyClock::now());
        } catch (const std::exception& exc) {
            set_error("scyllasband " + backend_name_ + " backend synthesize: " + exc.what());
            return SCYLLASBAND_STATUS_INVALID_ARGUMENT;
        }

        std::string duration_output_metadata = "null";
        std::string duration_expansion_metadata = "null";
        std::string vector_output_metadata = "null";
        std::string vector_sampler_metadata = "null";
        std::string vocoder_output_metadata = "null";
        std::string target_bucket_metadata = "null";
        ScyllasBandLiteRtAccelerator actual_vocoder_accelerator = vocoder_litert_accelerator();
        std::string selected_vocoder_fallback_error;
        bool split_vector_for_request = false;
        ScyllasBandStatus status = SCYLLASBAND_STATUS_OK;
        try {
            auto stage_started_at = ScyllasBandSteadyClock::now();
            ScyllasBandDurationPrediction duration_prediction = predict_duration_values(
                duration_session_,
                bundle_info_,
                prepared_inputs,
                resolved_request
            );
            duration_predict_ms = elapsed_ms_backend(stage_started_at, ScyllasBandSteadyClock::now());
            duration_output_metadata = duration_prediction.outputs_json;
            stage_started_at = ScyllasBandSteadyClock::now();
            ScyllasBandDurationExpansionMetadata duration_expansion = expand_duration_values(
                bundle_info_,
                prepared_inputs,
                request,
                resolved_request.language,
                duration_prediction.values
            );
            duration_expand_ms = elapsed_ms_backend(stage_started_at, ScyllasBandSteadyClock::now());
            duration_expansion_metadata = duration_expansion_json(duration_expansion);
            ScyllasBandTargetBucketSelection target_bucket = select_target_bucket_bundle(bundle_info_, duration_expansion);
            target_bucket_metadata = target_bucket.metadata_json;
            const ScyllasBandBundleInfo& runtime_bundle = target_bucket.bundle;
            activate_target_bucket(runtime_bundle);
            split_vector_for_request = use_split_vector_runtime(runtime_bundle, duration_expansion);
            vector_fallback_error_.clear();
            ScyllasBandLiteRtSession* vector_session = nullptr;
            ScyllasBandLiteRtSession* vector_prefix_session = nullptr;
            ScyllasBandLiteRtSession* vector_tail_session = nullptr;
            stage_started_at = ScyllasBandSteadyClock::now();
            ensure_vector_session(
                runtime_bundle,
                split_vector_for_request,
                &vector_session,
                &vector_prefix_session,
                &vector_tail_session
            );
            ensure_vector_session_ms = elapsed_ms_backend(stage_started_at, ScyllasBandSteadyClock::now());
            stage_started_at = ScyllasBandSteadyClock::now();
            ScyllasBandLatentSamplingResult sampling;
            try {
                sampling = sample_latents(
                    vector_session,
                    vector_prefix_session,
                    vector_tail_session,
                    split_vector_for_request,
                    runtime_bundle,
                    prepared_inputs,
                    duration_expansion,
                    resolved_request,
                    request
                );
                vector_sample_ms = elapsed_ms_backend(stage_started_at, ScyllasBandSteadyClock::now());
            } catch (const std::exception& exc) {
                if (!split_vector_for_request) {
                    throw;
                }
                vector_fallback_error_ = exc.what();
                split_vector_for_request = false;
                vector_sample_ms += elapsed_ms_backend(stage_started_at, ScyllasBandSteadyClock::now());
                stage_started_at = ScyllasBandSteadyClock::now();
                ensure_vector_session(
                    runtime_bundle,
                    split_vector_for_request,
                    &vector_session,
                    &vector_prefix_session,
                    &vector_tail_session
                );
                ensure_vector_session_ms += elapsed_ms_backend(stage_started_at, ScyllasBandSteadyClock::now());
                stage_started_at = ScyllasBandSteadyClock::now();
                sampling = sample_latents(
                    vector_session,
                    nullptr,
                    nullptr,
                    split_vector_for_request,
                    runtime_bundle,
                    prepared_inputs,
                    duration_expansion,
                    resolved_request,
                    request
                );
                vector_sample_ms += elapsed_ms_backend(stage_started_at, ScyllasBandSteadyClock::now());
                clear_error();
            }
            vector_calls = sampling.vector_calls;
            sampled_latent_values = sampling.latents.size();
            vector_output_metadata = sampling.last_vector_outputs_json;
            vector_sampler_metadata = sampler_json(sampling, request);
            stage_started_at = ScyllasBandSteadyClock::now();
            ScyllasBandLiteRtSession* vocoder_session = ensure_vocoder_session(
                runtime_bundle,
                &actual_vocoder_accelerator,
                &selected_vocoder_fallback_error
            );
            ensure_vocoder_session_ms = elapsed_ms_backend(stage_started_at, ScyllasBandSteadyClock::now());
            stage_started_at = ScyllasBandSteadyClock::now();
            ScyllasBandVocoderResult vocoder = run_vocoder(
                vocoder_session,
                runtime_bundle,
                prepared_inputs,
                sampling.latents,
                resolved_request.emotion_index,
                duration_expansion
            );
            vocoder_ms = elapsed_ms_backend(stage_started_at, ScyllasBandSteadyClock::now());
            vocoder_output_metadata = vocoder.outputs_json;
            stage_started_at = ScyllasBandSteadyClock::now();
            copy_audio_to_result(out_result, runtime_bundle, duration_expansion, vocoder.audio, sampling.latents);
            copy_audio_ms = elapsed_ms_backend(stage_started_at, ScyllasBandSteadyClock::now());
            clear_error();
        } catch (const std::exception& exc) {
            set_error("scyllasband " + backend_name_ + " backend synthesize: " + exc.what());
            status = SCYLLASBAND_STATUS_RUNTIME_ERROR;
        }

        const int64_t total_ms = elapsed_ms_backend(total_started_at, ScyllasBandSteadyClock::now());
        const std::string vector_fallback_error_json = vector_fallback_error_.empty()
            ? std::string("null")
            : std::string("\"") + json_escape_backend(vector_fallback_error_) + "\"";
        const std::string vocoder_fallback_error_json = selected_vocoder_fallback_error.empty()
            ? std::string("null")
            : std::string("\"") + json_escape_backend(selected_vocoder_fallback_error) + "\"";
        std::ostringstream metadata;
        metadata << "{\"status\":\"" << (status == SCYLLASBAND_STATUS_OK ? "ok" : "runtime_error") << "\","
                 << "\"stage\":\"" << backend_name_ << "_vocoder\","
                 << "\"backend\":\"" << backend_name_ << "\","
                 << "\"litert_accelerator\":\"" << litert_accelerator_name_backend(litert_accelerator_) << "\","
                 << "\"litert_g2p_accelerator\":\""
                 << litert_accelerator_name_backend(frontend_litert_accelerator()) << "\","
                 << "\"litert_g2p_accelerator_policy\":\""
                 << frontend_accelerator_policy() << "\","
                 << "\"litert_duration_accelerator\":\""
                 << litert_accelerator_name_backend(frontend_litert_accelerator()) << "\","
                 << "\"litert_duration_accelerator_policy\":\""
                 << frontend_accelerator_policy() << "\","
                 << "\"litert_vector_accelerator\":\""
                 << litert_accelerator_name_backend(vector_litert_accelerator(split_vector_for_request)) << "\","
                 << "\"litert_vector_tail_accelerator\":\""
                 << litert_accelerator_name_backend(vector_tail_litert_accelerator(split_vector_for_request)) << "\","
                 << "\"litert_vector_execution\":\""
                 << (split_vector_for_request ? "split_prefix_tail" : "single_graph") << "\","
                 << "\"litert_vector_accelerator_policy\":\""
                 << vector_litert_accelerator_policy(split_vector_for_request) << "\","
                 << "\"litert_vector_fallback_error\":"
                 << vector_fallback_error_json << ","
                 << "\"litert_min_gpu_split_vector_latent_frames\":"
                 << kMinGpuSplitVectorLatentFrames << ","
                 << "\"litert_vocoder_accelerator\":\""
                 << litert_accelerator_name_backend(actual_vocoder_accelerator) << "\","
                 << "\"litert_vocoder_requested_accelerator\":\""
                 << litert_accelerator_name_backend(vocoder_litert_accelerator()) << "\","
                 << "\"litert_vocoder_accelerator_policy\":\""
                 << vocoder_accelerator_policy(actual_vocoder_accelerator) << "\","
                 << "\"litert_vocoder_fallback_error\":"
                 << vocoder_fallback_error_json << ","
                 << "\"litert_max_threads\":" << litert_max_threads_ << ","
                 << "\"bundle_dir\":\"" << bundle_dir_ << "\","
                 << "\"bundle\":" << bundle_summary_json(bundle_info_) << ","
                 << "\"request\":" << request_context_json(resolved_request) << ","
                 << "\"execution_plan\":" << execution_plan_json(plan) << ","
                 << "\"g2p\":" << g2p_metadata << ","
                 << "\"prepared_inputs\":" << prepared_inputs_json(prepared_inputs) << ","
                 << "\"duration_outputs\":" << duration_output_metadata << ","
                 << "\"duration_expansion\":" << duration_expansion_metadata << ","
                 << "\"target_bucket\":" << target_bucket_metadata << ","
                 << "\"session_cache\":" << target_bucket_session_cache_json() << ","
                 << "\"vector_sampler\":" << vector_sampler_metadata << ","
                 << "\"vector_outputs\":" << vector_output_metadata << ","
                 << "\"vocoder_outputs\":" << vocoder_output_metadata << ","
                 << "\"timing_ms\":{"
                 << "\"total\":" << total_ms << ","
                 << "\"resolve_plan\":" << resolve_plan_ms << ","
                 << "\"ensure_g2p_session\":" << ensure_g2p_session_ms << ","
                 << "\"g2p\":" << g2p_ms << ","
                 << "\"prepare_inputs\":" << prepare_inputs_ms << ","
                 << "\"reference_features\":" << reference_features_ms << ","
                 << "\"span_context\":" << span_context_ms << ","
                 << "\"ensure_duration_session\":" << ensure_duration_session_ms << ","
                 << "\"ensure_vector_session\":" << ensure_vector_session_ms << ","
                 << "\"ensure_vocoder_session\":" << ensure_vocoder_session_ms << ","
                 << "\"duration_predict\":" << duration_predict_ms << ","
                 << "\"duration_expand\":" << duration_expand_ms << ","
                 << "\"vector_sample\":" << vector_sample_ms << ","
                 << "\"vocoder\":" << vocoder_ms << ","
                 << "\"copy_audio\":" << copy_audio_ms
                 << "},"
                 << "\"execution_stats\":{"
                 << "\"steps\":" << std::max<int>(1, request.steps) << ","
                 << "\"vector_calls\":" << vector_calls << ","
                 << "\"sampled_latent_values\":" << sampled_latent_values
                 << "},"
                 << "\"sample_count\":" << out_result->sample_count << ","
                 << "\"sample_rate\":" << out_result->sample_rate << "}";
        out_result->metadata_json = duplicate_c_string(metadata.str().c_str());
        if (status != SCYLLASBAND_STATUS_OK) {
            std::free(out_result->samples);
            out_result->samples = nullptr;
            out_result->sample_count = 0;
            out_result->sample_rate = 0;
        }
        return status;
    }

    ScyllasBandStatus estimate_latent_frames(
        const ScyllasBandSynthesisRequest& request,
        ScyllasBandDurationEstimate* out_estimate
    ) override {
        if (out_estimate == nullptr) {
            set_error("scyllasband " + backend_name_ + " backend estimate_latent_frames: output estimate is required");
            return SCYLLASBAND_STATUS_INVALID_ARGUMENT;
        }
        out_estimate->predicted_latent_frames = 0;
        out_estimate->fixed_latent_frames = bundle_info_.latent_frames;
        out_estimate->metadata_json.clear();

        const auto total_started_at = ScyllasBandSteadyClock::now();
        int64_t resolve_plan_ms = 0;
        int64_t ensure_g2p_session_ms = 0;
        int64_t g2p_ms = 0;
        int64_t prepare_inputs_ms = 0;
        int64_t reference_features_ms = 0;
        int64_t ensure_duration_session_ms = 0;
        int64_t duration_predict_ms = 0;
        int64_t duration_expand_ms = 0;
        std::string target_bucket_metadata = "null";

        ScyllasBandResolvedRequest resolved_request;
        ScyllasBandDurationFlowExecutionPlan plan;
        ScyllasBandDurationFlowPreparedInputs prepared_inputs;
        ScyllasBandSynthesisRequest effective_request = request;
        std::string generated_phone_text;
        std::string g2p_metadata = "null";
        try {
            auto stage_started_at = ScyllasBandSteadyClock::now();
            resolved_request = resolve_scyllasband_request_context(bundle_info_, request);
            plan = build_scyllasband_duration_flow_plan(bundle_info_, resolved_request);
            resolve_plan_ms = elapsed_ms_backend(stage_started_at, ScyllasBandSteadyClock::now());
            if (!resolved_request.has_explicit_phones) {
                stage_started_at = ScyllasBandSteadyClock::now();
                ensure_g2p_session();
                ensure_g2p_session_ms = elapsed_ms_backend(stage_started_at, ScyllasBandSteadyClock::now());
                stage_started_at = ScyllasBandSteadyClock::now();
                ScyllasBandG2PResult g2p = run_g2p_text(
                    g2p_session_,
                    bundle_info_,
                    resolved_request,
                    request,
                    pronunciation_overrides()
                );
                g2p_ms = elapsed_ms_backend(stage_started_at, ScyllasBandSteadyClock::now());
                generated_phone_text = join_phone_tokens(g2p.phones);
                effective_request.explicit_phones = generated_phone_text.c_str();
                g2p_metadata = g2p.metadata_json;
            }
            stage_started_at = ScyllasBandSteadyClock::now();
            prepared_inputs = prepare_scyllasband_duration_flow_inputs(bundle_info_, resolved_request, effective_request);
            if (!generated_phone_text.empty()) {
                prepared_inputs.phone_source = "g2p";
            }
            prepare_inputs_ms = elapsed_ms_backend(stage_started_at, ScyllasBandSteadyClock::now());
            stage_started_at = ScyllasBandSteadyClock::now();
            apply_reference_features(prepared_inputs, resolved_request);
            reference_features_ms = elapsed_ms_backend(stage_started_at, ScyllasBandSteadyClock::now());
            stage_started_at = ScyllasBandSteadyClock::now();
            ensure_duration_session();
            ensure_duration_session_ms = elapsed_ms_backend(stage_started_at, ScyllasBandSteadyClock::now());
        } catch (const std::exception& exc) {
            set_error("scyllasband " + backend_name_ + " backend estimate_latent_frames: " + exc.what());
            return SCYLLASBAND_STATUS_INVALID_ARGUMENT;
        }

        try {
            auto stage_started_at = ScyllasBandSteadyClock::now();
            ScyllasBandDurationPrediction duration_prediction = predict_duration_values(
                duration_session_,
                bundle_info_,
                prepared_inputs,
                resolved_request
            );
            duration_predict_ms = elapsed_ms_backend(stage_started_at, ScyllasBandSteadyClock::now());
            stage_started_at = ScyllasBandSteadyClock::now();
            ScyllasBandDurationExpansionMetadata duration_expansion = expand_duration_values(
                bundle_info_,
                prepared_inputs,
                request,
                resolved_request.language,
                duration_prediction.values,
                true
            );
            duration_expand_ms = elapsed_ms_backend(stage_started_at, ScyllasBandSteadyClock::now());
            ScyllasBandTargetBucketSelection target_bucket = select_target_bucket_bundle(
                bundle_info_,
                duration_expansion,
                true
            );
            target_bucket_metadata = target_bucket.metadata_json;
            out_estimate->predicted_latent_frames = static_cast<int>(std::min<int64_t>(
                duration_expansion.predicted_latent_frames,
                static_cast<int64_t>(std::numeric_limits<int>::max())
            ));
            out_estimate->fixed_latent_frames = target_bucket.bucket.latent_frames;
            const int64_t total_ms = elapsed_ms_backend(total_started_at, ScyllasBandSteadyClock::now());
            std::ostringstream metadata;
            metadata << "{\"status\":\"ok\","
                     << "\"stage\":\"" << backend_name_ << "_duration_preflight\","
                     << "\"backend\":\"" << backend_name_ << "\","
                     << "\"litert_accelerator\":\"" << litert_accelerator_name_backend(litert_accelerator_) << "\","
                     << "\"litert_max_threads\":" << litert_max_threads_ << ","
                     << "\"bundle_dir\":\"" << bundle_dir_ << "\","
                     << "\"bundle\":" << bundle_summary_json(bundle_info_) << ","
                     << "\"request\":" << request_context_json(resolved_request) << ","
                     << "\"execution_plan\":" << execution_plan_json(plan) << ","
                     << "\"g2p\":" << g2p_metadata << ","
                     << "\"prepared_inputs\":" << prepared_inputs_json(prepared_inputs) << ","
                     << "\"duration_outputs\":" << duration_prediction.outputs_json << ","
                     << "\"duration_expansion\":" << duration_expansion_json(duration_expansion) << ","
                     << "\"target_bucket\":" << target_bucket_metadata << ","
                     << "\"timing_ms\":{"
                     << "\"total\":" << total_ms << ","
                     << "\"resolve_plan\":" << resolve_plan_ms << ","
                     << "\"ensure_g2p_session\":" << ensure_g2p_session_ms << ","
                     << "\"g2p\":" << g2p_ms << ","
                     << "\"prepare_inputs\":" << prepare_inputs_ms << ","
                     << "\"reference_features\":" << reference_features_ms << ","
                     << "\"ensure_duration_session\":" << ensure_duration_session_ms << ","
                     << "\"duration_predict\":" << duration_predict_ms << ","
                     << "\"duration_expand\":" << duration_expand_ms
                     << "},"
                     << "\"predicted_latent_frames\":" << duration_expansion.predicted_latent_frames << ","
                     << "\"fixed_latent_frames\":" << target_bucket.bucket.latent_frames << "}";
            out_estimate->metadata_json = metadata.str();
            clear_error();
            return SCYLLASBAND_STATUS_OK;
        } catch (const std::exception& exc) {
            set_error("scyllasband " + backend_name_ + " backend estimate_latent_frames: " + exc.what());
            return SCYLLASBAND_STATUS_RUNTIME_ERROR;
        }
    }

    const char* name() const override {
        return backend_name_.c_str();
    }

    void set_target_bucket_cache_capacity(int capacity) override {
        if (backend_name_ == "coreai") {
            // All fixed frame buckets are functions in one shared-weight
            // .aimodel. Evicting by logical bucket would discard that one model.
            return;
        }
        evict_target_buckets(target_bucket_cache_policy_.set_capacity(capacity));
    }

private:
    struct TargetBucketSessionPaths {
        std::string vector;
        std::string vector_prefix;
        std::string vector_tail;
        std::string vocoder;
    };

    static void destroy_session_map(std::map<std::string, ScyllasBandLiteRtSession*>& sessions) {
        for (auto& item : sessions) {
            if (item.second != nullptr) {
                scyllasband_litert_session_destroy(item.second);
                item.second = nullptr;
            }
        }
        sessions.clear();
    }

    static std::vector<std::string> destroy_sessions_for_path(
        std::map<std::string, ScyllasBandLiteRtSession*>& sessions,
        const std::string& path
    ) {
        std::vector<std::string> removed_keys;
        if (path.empty()) {
            return removed_keys;
        }
        const std::string prefix = path + "|";
        auto item = sessions.begin();
        while (item != sessions.end()) {
            if (item->first.rfind(prefix, 0) != 0) {
                ++item;
                continue;
            }
            if (item->second != nullptr) {
                scyllasband_litert_session_destroy(item->second);
            }
            removed_keys.push_back(item->first);
            item = sessions.erase(item);
        }
        return removed_keys;
    }

    void activate_target_bucket(const ScyllasBandBundleInfo& bundle) {
        const std::string bucket_key = std::to_string(bundle.latent_frames);
        TargetBucketSessionPaths paths;
        paths.vector = component_path(bundle, "vector_estimator");
        paths.vocoder = component_path(bundle, "vocoder");
        if (bundle.component_artifacts.count("vector_estimator_prefix") > 0) {
            paths.vector_prefix = component_path(bundle, "vector_estimator_prefix");
        }
        if (bundle.component_artifacts.count("vector_estimator_tail") > 0) {
            paths.vector_tail = component_path(bundle, "vector_estimator_tail");
        }
        target_bucket_session_paths_[bucket_key] = std::move(paths);
        evict_target_buckets(target_bucket_cache_policy_.touch(bucket_key));
    }

    void evict_target_buckets(const std::vector<std::string>& bucket_keys) {
        for (const std::string& bucket_key : bucket_keys) {
            const auto paths_item = target_bucket_session_paths_.find(bucket_key);
            if (paths_item == target_bucket_session_paths_.end()) {
                continue;
            }
            const TargetBucketSessionPaths& paths = paths_item->second;
            destroy_sessions_for_path(vector_sessions_, paths.vector);
            destroy_sessions_for_path(vector_prefix_sessions_, paths.vector_prefix);
            destroy_sessions_for_path(vector_tail_sessions_, paths.vector_tail);
            const std::vector<std::string> vocoder_keys = destroy_sessions_for_path(
                vocoder_sessions_,
                paths.vocoder
            );
            for (const std::string& key : vocoder_keys) {
                vocoder_session_accelerators_.erase(key);
                vocoder_session_fallback_errors_.erase(key);
            }
            target_bucket_session_paths_.erase(paths_item);
        }
    }

    std::string target_bucket_session_cache_json() const {
        const auto& keys = target_bucket_cache_policy_.lru_keys();
        std::ostringstream out;
        out << "{"
            << "\"target_bucket_capacity\":" << target_bucket_cache_policy_.capacity() << ","
            << "\"target_bucket_count\":" << keys.size() << ","
            << "\"target_bucket_frames_lru\":[";
        for (std::size_t index = 0; index < keys.size(); ++index) {
            if (index > 0) {
                out << ",";
            }
            out << keys[index];
        }
        out << "]}";
        return out.str();
    }

    const ScyllasBandPronunciationOverrideMap& pronunciation_overrides() {
        if (pronunciation_overrides_ready_) {
            return pronunciation_overrides_;
        }
        const auto asset_it = bundle_info_.assets.find("g2p_pronunciation_overrides");
        if (asset_it != bundle_info_.assets.end() && !asset_it->second.empty()) {
            const std::filesystem::path path = std::filesystem::path(bundle_info_.bundle_dir) / asset_it->second;
            pronunciation_overrides_ = load_pronunciation_overrides(path);
        }
        pronunciation_overrides_ready_ = true;
        return pronunciation_overrides_;
    }

    const ReferenceNormalizationStats& reference_normalization_stats() {
        if (reference_stats_ready_) {
            return reference_stats_;
        }
        std::vector<std::vector<float>> style_rows;
        std::vector<std::vector<float>> prosody_rows;
        const auto pack_dir_it = bundle_info_.assets.find("voice_packs");
        if (pack_dir_it != bundle_info_.assets.end() && !pack_dir_it->second.empty()) {
            const std::filesystem::path pack_dir = std::filesystem::path(bundle_info_.bundle_dir) / pack_dir_it->second;
            if (std::filesystem::is_directory(pack_dir)) {
                std::vector<std::filesystem::path> paths;
                for (const auto& entry : std::filesystem::directory_iterator(pack_dir)) {
                    if (entry.path().extension() == ".npz") {
                        paths.push_back(entry.path());
                    }
                }
                std::sort(paths.begin(), paths.end());
                for (const auto& path : paths) {
                    const auto pack = load_uncompressed_npz(path);
                    for (const auto& item : pack) {
                        constexpr const char* prefix = "reference_mask__";
                        const std::string& key = item.first;
                        if (key.rfind(prefix, 0) != 0 || mask_value(pack, key) <= 0.0f) {
                            continue;
                        }
                        const std::string suffix = key.substr(std::strlen(prefix));
                        style_rows.push_back(array_or_zeros(pack, "style_embedding__" + suffix, bundle_info_.reference_style_dim));
                        prosody_rows.push_back(transform_prosody(array_or_zeros(pack, "prosody_stats__" + suffix, bundle_info_.reference_prosody_dim)));
                    }
                }
            }
        }
        auto style_stats = standardize_rows(style_rows, bundle_info_.reference_style_dim);
        auto prosody_stats = standardize_rows(prosody_rows, bundle_info_.reference_prosody_dim);
        reference_stats_.style_mean = std::move(style_stats.first);
        reference_stats_.style_std = std::move(style_stats.second);
        reference_stats_.prosody_mean = std::move(prosody_stats.first);
        reference_stats_.prosody_std = std::move(prosody_stats.second);
        reference_stats_ready_ = true;
        return reference_stats_;
    }


    const ScyllasBandNpyPack& reference_pack_v4(
        const std::filesystem::path& path
    ) {
        const std::string cache_key = path.string();
        const auto cached = reference_pack_v4_cache_.find(cache_key);
        if (cached != reference_pack_v4_cache_.end()) {
            return cached->second;
        }
        auto inserted = reference_pack_v4_cache_.emplace(
            cache_key, load_uncompressed_npz(path)
        );
        return inserted.first->second;
    }


    void apply_reference_features(
        ScyllasBandDurationFlowPreparedInputs& prepared,
        const ScyllasBandResolvedRequest& resolved_request
    ) {
        if (!bundle_info_.reference_packs_enabled) {
            return;
        }
        const std::filesystem::path path = reference_pack_path_for_voice(bundle_info_, resolved_request.voice_id);
        if (path.empty()) {
            return;
        }
        prepared.reference_pack_path = path.string();
        if (!std::filesystem::is_regular_file(path)) {
            return;
        }
        const auto pack = load_uncompressed_npz(path);

        if (bundle_info_.reference_pack_schema_version == 4) {
            if (
                bundle_info_.reference_affect_routing_version !=
                "native_exact_primary_relative_residual_v4"
            ) {
                throw std::runtime_error(
                    "unsupported schema-v4 affect-reference routing version '" +
                    bundle_info_.reference_affect_routing_version + "'"
                );
            }
            if (
                bundle_info_.reference_identity_dim != 512 ||
                bundle_info_.reference_baseline_dim != 32 ||
                bundle_info_.reference_delta_dim != 32 ||
                bundle_info_.reference_prosody_dim != 32
            ) {
                throw std::runtime_error(
                    "schema-v4 reference dimensions do not match the native contract"
                );
            }
            const auto routed = route_reference_pack_v4_native(
                reference_pack_v4(path),
                resolved_request.language,
                prepared.affect_values,
                prepared.affect_condition_mask_values
            );
            prepared.identity_reference = routed.identity;
            prepared.identity_reference_mask = 1.0f;
            prepared.prosody_baseline = routed.prosody_baseline;
            prepared.prosody_delta = routed.prosody_delta;
            prepared.prosody_feature_mask = routed.prosody_feature_mask;
            prepared.prosody_confidence = routed.prosody_confidence;
            prepared.reference_mask = 1.0f;
            prepared.native_reference_mask = 1.0f;
            prepared.fallback_reference_mask = 0.0f;
            prepared.reference_key =
                resolved_request.language + "|v4|" + routed.route_kind +
                "|baseline=" + std::to_string(routed.baseline_index) +
                "|prototype=" + (
                    routed.prototype_index < 0
                    ? std::string("none")
                    : std::to_string(routed.prototype_index)
                );
            return;
        }
        if (
            bundle_info_.reference_style_dim <= 0 ||
            bundle_info_.reference_prosody_dim <= 0
        ) {
            return;
        }
        const std::string suffix = select_reference_suffix(pack, resolved_request.language, resolved_request.emotion);
        if (suffix.empty()) {
            return;
        }
        const float reference_mask = mask_value(pack, "reference_mask__" + suffix);
        const float native_mask = mask_value(pack, "native_reference_mask__" + suffix);
        const float fallback_mask = mask_value(pack, "fallback_reference_mask__" + suffix);
        if (reference_mask <= 0.0f) {
            return;
        }
        const ReferenceNormalizationStats& stats = reference_normalization_stats();
        prepared.reference_style = normalize_style(
            array_or_zeros(pack, "style_embedding__" + suffix, bundle_info_.reference_style_dim),
            stats.style_mean,
            stats.style_std,
            bundle_info_.reference_style_dim
        );
        prepared.reference_prosody = normalize_prosody(
            array_or_zeros(pack, "prosody_stats__" + suffix, bundle_info_.reference_prosody_dim),
            stats.prosody_mean,
            stats.prosody_std,
            bundle_info_.reference_prosody_dim
        );
        prepared.reference_mask = effective_reference_mask(
            reference_mask,
            native_mask,
            fallback_mask,
            bundle_info_.reference_fallback_weight
        );
        prepared.native_reference_mask = native_mask;
        prepared.fallback_reference_mask = fallback_mask;
        prepared.reference_key = reference_key_from_suffix(suffix);
    }

    void ensure_g2p_session() {
        if (g2p_session_ != nullptr) {
            return;
        }
        const std::string g2p_path = component_path(bundle_info_, "g2p");
        g2p_session_ = scyllasband_litert_session_create(
            g2p_path.c_str(),
            frontend_litert_accelerator(),
            litert_max_threads_
        );
        if (g2p_session_ == nullptr) {
            const char* detail = last_error();
            throw std::runtime_error(std::string("failed to create g2p LiteRT session: ") +
                                     (detail == nullptr ? "" : detail));
        }
        if (!scyllasband_litert_session_has_signature(g2p_session_, "serving_default")) {
            throw std::runtime_error("g2p LiteRT graph is missing serving_default signature");
        }
    }

    void ensure_vector_context_session() {
        if (vector_context_session_ != nullptr) {
            return;
        }
        const std::string context_path = component_path(bundle_info_, "vector_context_encoder");
        vector_context_session_ = scyllasband_litert_session_create(
            context_path.c_str(),
            SCYLLASBAND_LITERT_ACCELERATOR_CPU,
            litert_max_threads_
        );
        if (vector_context_session_ == nullptr) {
            const char* detail = last_error();
            throw std::runtime_error(std::string("failed to create vector_context_encoder LiteRT session: ") +
                                     (detail == nullptr ? "" : detail));
        }
        if (!scyllasband_litert_session_has_signature(vector_context_session_, "serving_default")) {
            throw std::runtime_error("vector_context_encoder LiteRT graph is missing serving_default signature");
        }
    }

    std::vector<std::string> phones_for_span_context_text(
        const std::string& text,
        const ScyllasBandResolvedRequest& resolved_request,
        const ScyllasBandSynthesisRequest& request
    ) {
        if (trim_backend(text).empty()) {
            return {};
        }
        ensure_g2p_session();
        ScyllasBandResolvedRequest context_resolved = resolved_request;
        context_resolved.context_before.clear();
        context_resolved.context_after.clear();
        context_resolved.boundary_before = "chunk_continue";
        context_resolved.boundary_after = "chunk_continue";
        ScyllasBandSynthesisRequest context_request = request;
        const std::string boundary = "chunk_continue";
        context_request.text = text.c_str();
        context_request.explicit_phones = nullptr;
        context_request.context_before = nullptr;
        context_request.context_after = nullptr;
        context_request.boundary_before = boundary.c_str();
        context_request.boundary_after = boundary.c_str();
        context_request.prefix_latents = nullptr;
        context_request.prefix_latent_dim = 0;
        context_request.prefix_latent_frames = 0;
        return run_g2p_text(
            g2p_session_,
            bundle_info_,
            context_resolved,
            context_request,
            pronunciation_overrides()
        ).phones;
    }

    void apply_span_context_features(
        ScyllasBandDurationFlowPreparedInputs& prepared,
        const ScyllasBandResolvedRequest& resolved_request,
        const ScyllasBandSynthesisRequest& request
    ) {
        if (!bundle_info_.span_conditioning_enabled || bundle_info_.span_context_hidden_size <= 0) {
            prepared.span_context_hidden.clear();
            prepared.span_context_metadata = "null";
            return;
        }
        std::vector<std::string> previous_phones;
        std::vector<std::string> next_phones;
        previous_phones = phones_for_span_context_text(resolved_request.context_before, resolved_request, request);
        next_phones = phones_for_span_context_text(resolved_request.context_after, resolved_request, request);
        ScyllasBandSpanContextEncoding encoding = build_span_context_encoding(
            bundle_info_,
            previous_phones,
            prepared.phones,
            next_phones
        );
        const char* disabled = std::getenv("SCYLLASBAND_DISABLE_SPAN_CONTEXT");
        if (disabled != nullptr && disabled[0] != '\0' && std::string(disabled) != "0") {
            ScyllasBandSpanContextResult zero = zero_span_context_result(
                bundle_info_,
                encoding,
                "zero_disabled_by_env"
            );
            prepared.span_context_hidden = std::move(zero.hidden);
            prepared.span_context_metadata = zero.metadata_json;
            return;
        }
        ensure_vector_context_session();
        ScyllasBandSpanContextResult span_context = run_span_context_encoder(
            vector_context_session_,
            bundle_info_,
            encoding
        );
        prepared.span_context_hidden = std::move(span_context.hidden);
        prepared.span_context_metadata = span_context.metadata_json;
    }

    void ensure_duration_session() {
        if (duration_session_ != nullptr) {
            return;
        }
        const std::string duration_path = component_path(bundle_info_, "duration_predictor");
        duration_session_ = scyllasband_litert_session_create(
            duration_path.c_str(),
            frontend_litert_accelerator(),
            litert_max_threads_
        );
        if (duration_session_ == nullptr) {
            const char* detail = last_error();
            throw std::runtime_error(std::string("failed to create duration_predictor LiteRT session: ") +
                                     (detail == nullptr ? "" : detail));
        }
        if (!scyllasband_litert_session_has_signature(duration_session_, "serving_default")) {
            throw std::runtime_error("duration_predictor LiteRT graph is missing serving_default signature");
        }
    }

    std::string session_key(
        const ScyllasBandBundleInfo& bundle,
        const std::string& component_name,
        ScyllasBandLiteRtAccelerator accelerator
    ) const {
        return component_path(bundle, component_name) + "|" + litert_accelerator_name_backend(accelerator);
    }

    ScyllasBandLiteRtSession* ensure_component_session(
        std::map<std::string, ScyllasBandLiteRtSession*>& sessions,
        const ScyllasBandBundleInfo& bundle,
        const std::string& component_name,
        ScyllasBandLiteRtAccelerator accelerator,
        const std::string& label
    ) {
        const std::string key = session_key(bundle, component_name, accelerator);
        const auto existing = sessions.find(key);
        if (existing != sessions.end() && existing->second != nullptr) {
            return existing->second;
        }
        const std::string path = component_path(bundle, component_name);
        ScyllasBandLiteRtSession* session = scyllasband_litert_session_create(
            path.c_str(),
            accelerator,
            litert_max_threads_
        );
        if (session == nullptr) {
            const char* detail = last_error();
            throw std::runtime_error(std::string("failed to create ") + label + " LiteRT session: " +
                                     (detail == nullptr ? "" : detail));
        }
        if (!scyllasband_litert_session_has_signature(session, "serving_default")) {
            scyllasband_litert_session_destroy(session);
            throw std::runtime_error(label + " LiteRT graph is missing serving_default signature");
        }
        sessions[key] = session;
        return session;
    }

    void ensure_vector_session(
        const ScyllasBandBundleInfo& bundle,
        bool split_vector_for_request,
        ScyllasBandLiteRtSession** out_session,
        ScyllasBandLiteRtSession** out_prefix_session,
        ScyllasBandLiteRtSession** out_tail_session
    ) {
        if (out_session != nullptr) {
            *out_session = nullptr;
        }
        if (out_prefix_session != nullptr) {
            *out_prefix_session = nullptr;
        }
        if (out_tail_session != nullptr) {
            *out_tail_session = nullptr;
        }
        if (split_vector_for_request) {
            ScyllasBandLiteRtSession* prefix = ensure_component_session(
                vector_prefix_sessions_,
                bundle,
                "vector_estimator_prefix",
                vector_prefix_litert_accelerator(split_vector_for_request),
                "vector_estimator_prefix"
            );
            ScyllasBandLiteRtSession* tail = ensure_component_session(
                vector_tail_sessions_,
                bundle,
                "vector_estimator_tail",
                vector_tail_litert_accelerator(split_vector_for_request),
                "vector_estimator_tail"
            );
            if (out_prefix_session != nullptr) {
                *out_prefix_session = prefix;
            }
            if (out_tail_session != nullptr) {
                *out_tail_session = tail;
            }
            return;
        }
        ScyllasBandLiteRtSession* session = ensure_component_session(
            vector_sessions_,
            bundle,
            "vector_estimator",
            vector_full_litert_accelerator(split_vector_for_request),
            "vector_estimator"
        );
        if (out_session != nullptr) {
            *out_session = session;
        }
    }

    ScyllasBandLiteRtSession* ensure_vocoder_session(
        const ScyllasBandBundleInfo& bundle,
        ScyllasBandLiteRtAccelerator* out_accelerator,
        std::string* out_fallback_error
    ) {
        const ScyllasBandLiteRtAccelerator requested_accelerator = vocoder_litert_accelerator();
        const std::string key = session_key(bundle, "vocoder", requested_accelerator);
        const auto existing = vocoder_sessions_.find(key);
        if (existing != vocoder_sessions_.end() && existing->second != nullptr) {
            if (out_accelerator != nullptr) {
                *out_accelerator = vocoder_session_accelerators_[key];
            }
            if (out_fallback_error != nullptr) {
                *out_fallback_error = vocoder_session_fallback_errors_[key];
            }
            return existing->second;
        }

        const std::string vocoder_path = component_path(bundle, "vocoder");
        std::string fallback_error;
        ScyllasBandLiteRtAccelerator actual_accelerator = requested_accelerator;
        ScyllasBandLiteRtSession* session = scyllasband_litert_session_create(
            vocoder_path.c_str(),
            requested_accelerator,
            litert_max_threads_
        );
        if (session == nullptr &&
            requested_accelerator != SCYLLASBAND_LITERT_ACCELERATOR_CPU &&
            litert_vocoder_can_fallback_to_cpu_backend(litert_accelerator_)) {
            const char* detail = last_error();
            fallback_error = detail == nullptr ? "" : detail;
            clear_error();
            session = scyllasband_litert_session_create(
                vocoder_path.c_str(),
                SCYLLASBAND_LITERT_ACCELERATOR_CPU,
                litert_max_threads_
            );
            actual_accelerator = SCYLLASBAND_LITERT_ACCELERATOR_CPU;
            if (session == nullptr) {
                const char* fallback_detail = last_error();
                throw std::runtime_error(
                    std::string("failed to create vocoder LiteRT session: ") +
                    (fallback_error.empty() ? "requested accelerator failed" : fallback_error) +
                    "; CPU fallback failed: " +
                    (fallback_detail == nullptr ? "" : fallback_detail)
                );
            }
            clear_error();
        }
        if (session == nullptr) {
            const char* detail = last_error();
            throw std::runtime_error(std::string("failed to create vocoder LiteRT session: ") +
                                     (detail == nullptr ? "" : detail));
        }
        if (!scyllasband_litert_session_has_signature(session, "serving_default")) {
            scyllasband_litert_session_destroy(session);
            throw std::runtime_error("vocoder LiteRT graph is missing serving_default signature");
        }
        vocoder_sessions_[key] = session;
        vocoder_session_accelerators_[key] = actual_accelerator;
        vocoder_session_fallback_errors_[key] = fallback_error;
        if (out_accelerator != nullptr) {
            *out_accelerator = actual_accelerator;
        }
        if (out_fallback_error != nullptr) {
            *out_fallback_error = fallback_error;
        }
        return session;
    }

    ScyllasBandLiteRtAccelerator frontend_litert_accelerator() const {
        if (backend_name_ == "coreai") {
            return litert_accelerator_;
        }
        return litert_frontend_accelerator_backend(litert_accelerator_);
    }

    bool split_vector_available(const ScyllasBandBundleInfo& bundle) const {
        return bundle.component_artifacts.count("vector_estimator_prefix") > 0 &&
               bundle.component_artifacts.count("vector_estimator_tail") > 0;
    }

    bool use_split_vector_runtime(const ScyllasBandBundleInfo& bundle) const {
        return litert_accelerator_ == SCYLLASBAND_LITERT_ACCELERATOR_GPU && split_vector_available(bundle);
    }

    bool use_split_vector_runtime(
        const ScyllasBandBundleInfo& bundle,
        const ScyllasBandDurationExpansionMetadata& expansion
    ) const {
        return use_split_vector_runtime(bundle) &&
               expansion.predicted_latent_frames >= kMinGpuSplitVectorLatentFrames;
    }

    ScyllasBandLiteRtAccelerator vector_full_litert_accelerator(bool split_vector_for_request) const {
        if (backend_name_ == "coreai") {
            return litert_accelerator_;
        }
        if (litert_accelerator_ == SCYLLASBAND_LITERT_ACCELERATOR_GPU && !split_vector_for_request) {
            return SCYLLASBAND_LITERT_ACCELERATOR_CPU;
        }
        return litert_vector_accelerator_backend(litert_accelerator_);
    }

    ScyllasBandLiteRtAccelerator vector_prefix_litert_accelerator(bool split_vector_for_request) const {
        return split_vector_for_request
            ? SCYLLASBAND_LITERT_ACCELERATOR_GPU
            : vector_full_litert_accelerator(split_vector_for_request);
    }

    ScyllasBandLiteRtAccelerator vector_tail_litert_accelerator(bool split_vector_for_request) const {
        return split_vector_for_request
            ? SCYLLASBAND_LITERT_ACCELERATOR_CPU
            : vector_full_litert_accelerator(split_vector_for_request);
    }

    ScyllasBandLiteRtAccelerator vector_litert_accelerator(bool split_vector_for_request) const {
        return split_vector_for_request
            ? vector_prefix_litert_accelerator(split_vector_for_request)
            : vector_full_litert_accelerator(split_vector_for_request);
    }

    const char* vector_litert_accelerator_policy(bool split_vector_for_request) const {
        if (backend_name_ == "coreai") {
            return "coreai_preferred_compute_unit";
        }
        if (split_vector_for_request) {
            return "gpu_prefix_cpu_tail_split_vector";
        }
        if (!vector_fallback_error_.empty()) {
            return "cpu_fallback_after_vector_gpu_runtime_failure";
        }
        if (litert_accelerator_ == SCYLLASBAND_LITERT_ACCELERATOR_GPU && !split_vector_for_request) {
            return "cpu_fallback_for_short_or_unavailable_split_vector_request";
        }
        return litert_vector_accelerator_policy_backend(litert_accelerator_);
    }

    ScyllasBandLiteRtAccelerator vocoder_litert_accelerator() const {
        if (backend_name_ == "coreai") {
            return litert_accelerator_;
        }
        return litert_vocoder_accelerator_backend(litert_accelerator_);
    }

    const char* frontend_accelerator_policy() const {
        return backend_name_ == "coreai"
            ? "coreai_preferred_compute_unit"
            : litert_frontend_accelerator_policy_backend(litert_accelerator_);
    }

    const char* vocoder_accelerator_policy(
        ScyllasBandLiteRtAccelerator actual_accelerator
    ) const {
        if (backend_name_ == "coreai") {
            return "coreai_preferred_compute_unit";
        }
        return actual_accelerator == vocoder_litert_accelerator()
            ? litert_vocoder_accelerator_policy_backend(litert_accelerator_)
            : "cpu_fallback_after_vocoder_gpu_compile_failure";
    }

    std::string bundle_dir_;
    ScyllasBandBundleInfo bundle_info_;
    std::string backend_name_;
    ScyllasBandLiteRtAccelerator litert_accelerator_ = SCYLLASBAND_LITERT_ACCELERATOR_CPU;
    int litert_max_threads_ = 0;
    ReferenceNormalizationStats reference_stats_;
    std::map<std::string, ScyllasBandNpyPack> reference_pack_v4_cache_;
    bool reference_stats_ready_ = false;
    ScyllasBandPronunciationOverrideMap pronunciation_overrides_;
    bool pronunciation_overrides_ready_ = false;
    ScyllasBandLiteRtSession* g2p_session_ = nullptr;
    ScyllasBandLiteRtSession* vector_context_session_ = nullptr;
    ScyllasBandLiteRtSession* duration_session_ = nullptr;
    std::map<std::string, ScyllasBandLiteRtSession*> vector_sessions_;
    std::map<std::string, ScyllasBandLiteRtSession*> vector_prefix_sessions_;
    std::map<std::string, ScyllasBandLiteRtSession*> vector_tail_sessions_;
    std::map<std::string, ScyllasBandLiteRtSession*> vocoder_sessions_;
    std::map<std::string, ScyllasBandLiteRtAccelerator> vocoder_session_accelerators_;
    std::map<std::string, std::string> vocoder_session_fallback_errors_;
    ScyllasBandSessionCachePolicy target_bucket_cache_policy_;
    std::map<std::string, TargetBucketSessionPaths> target_bucket_session_paths_;
    std::string vector_fallback_error_;
    std::string vocoder_fallback_error_;
};

#endif  // graph runtime enabled

}  // namespace

std::unique_ptr<ScyllasBandBackendEngine> create_backend_engine(
    ScyllasBandBackend backend,
    std::string bundle_dir,
    bool validate_bundle,
    ScyllasBandLiteRtAccelerator litert_accelerator,
    int litert_max_threads
) {
    const ScyllasBandBackend selected = scyllasband_select_backend(backend);
    ScyllasBandBundleInfo bundle_info;
    bundle_info.bundle_dir = bundle_dir;
    bundle_info.selected_backend = scyllasband_backend_name(selected);
    const bool must_load_bundle = validate_bundle || selected == SCYLLASBAND_BACKEND_LITERT ||
        selected == SCYLLASBAND_BACKEND_ONNX || selected == SCYLLASBAND_BACKEND_COREAI;
    if (must_load_bundle) {
        try {
            bundle_info = load_scyllasband_bundle_info(bundle_dir, selected);
        } catch (const std::exception& exc) {
            set_error(std::string("scyllasband bundle validation failed: ") + exc.what());
            return nullptr;
        }
    }
#ifdef SCYLLASBAND_WITH_ONNXRUNTIME
    if (selected == SCYLLASBAND_BACKEND_ONNX) {
        return std::make_unique<DurationFlowBackend>(
            std::move(bundle_dir),
            std::move(bundle_info),
            scyllasband_backend_name(selected),
            litert_accelerator,
            litert_max_threads
        );
    }
#endif
#ifdef SCYLLASBAND_WITH_LITERT
    if (selected == SCYLLASBAND_BACKEND_LITERT) {
        return std::make_unique<DurationFlowBackend>(
            std::move(bundle_dir),
            std::move(bundle_info),
            scyllasband_backend_name(selected),
            litert_accelerator,
            litert_max_threads
        );
    }
#endif
#ifdef SCYLLASBAND_WITH_COREAI
    if (selected == SCYLLASBAND_BACKEND_COREAI) {
        return std::make_unique<DurationFlowBackend>(
            std::move(bundle_dir),
            std::move(bundle_info),
            scyllasband_backend_name(selected),
            litert_accelerator,
            litert_max_threads
        );
    }
#endif
    return std::make_unique<NotLinkedBackend>(
        scyllasband_backend_name(selected),
        std::move(bundle_dir),
        std::move(bundle_info)
    );
}

}  // namespace scyllasband_detail
