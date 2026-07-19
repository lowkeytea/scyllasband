#include "scyllasband.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string bundle;
    std::string output = "native_output.wav";
    std::string metadata;
    std::string text;
    std::string text_file;
    std::string voice = "gwen";
    std::string language = "en_us";
    std::string emotion = "neutral";
    std::string emotion_guidance;
    std::string affect;
    float affect_guidance_scale = 1.0f;
    int steps = 8;
    ScyllasBandSampler sampler = SCYLLASBAND_SAMPLER_HEUN;
    uint64_t seed = 0;
    bool has_seed = false;
    bool long_form = false;
    bool plan_only = false;
    bool stream = false;
    std::string events;
    std::string chunk_output_dir;
    int max_chunk_chars = 220;
    int min_chunk_chars = 48;
    int pause_ms = 0;
    int continuation_pause_ms = 0;
    float min_sentence_pause_ms = 320.0f;
    float min_clause_pause_ms = 160.0f;
    bool use_prefix_latents = true;
    bool preflight_chunks = false;
    bool progress = true;
    bool validate_bundle = true;
    ScyllasBandBackend backend = SCYLLASBAND_BACKEND_ONNX;
    ScyllasBandLiteRtAccelerator litert_accelerator = SCYLLASBAND_LITERT_ACCELERATOR_CPU;
};

void usage(std::ostream& out) {
    out << "Usage: scyllasband_native_speak --bundle DIR (--text TEXT | --file TEXT.txt) "
           "[--output FILE.wav] [options]\n\n"
           "Options:\n"
           "  --metadata FILE.json          Write result metadata JSON\n"
           "  --voice ID                    Voice id, default gwen\n"
           "  --language ID                 Language id, default en_us\n"
           "  --emotion ID                  Emotion id, default neutral\n"
           "  --emotion-guidance SPEC       Mixed emotion guidance, e.g. sad:0.7,excited:0.4\n"
           "  --affect SPEC                 Affect preset or comma-separated axis=value terms\n"
           "  --affect-guidance-scale N     Affect CFG scale, default 1.0\n"
           "  --steps N                     Flow steps, default 8\n"
           "  --sampler euler|heun          Flow sampler, default heun\n"
           "  --seed N                      Deterministic seed\n"
           "  --long-form                   Use native long-form chunking\n"
           "  --plan-only                   Write the native long-form plan JSON and exit\n"
           "  --stream                      Use native long-form streaming callbacks\n"
           "  --events FILE.jsonl           Write streaming callback events as JSONL\n"
           "  --chunk-output-dir DIR        Write per-chunk WAVs during --stream\n"
           "  --no-progress                 Disable streaming progress lines on stderr\n"
           "  --chunk-max-chars N           Long-form max chunk chars, default 220\n"
           "  --chunk-min-chars N           Long-form min chunk chars, default 48\n"
           "  --pause-ms N                  Long-form sentence pause, default 0\n"
           "  --continuation-pause-ms N     Long-form continuation pause, default 0\n"
           "  --min-sentence-pause-ms N     In-chunk sentence punctuation floor, default 320\n"
           "  --min-clause-pause-ms N       In-chunk clause punctuation floor, default 160\n"
           "  --no-prefix-latents           Disable long-form prefix carryover\n"
           "  --preflight-chunks            Run full duration preflight before synthesis\n"
           "  --backend auto|onnx|litert|coreml  Backend selection, default onnx; auto aliases onnx\n"
           "  --litert-accelerator MODE     LiteRT accelerator auto|cpu|gpu|npu, default cpu\n"
           "  --no-validate-bundle          Skip manifest file validation\n";
}

bool consume_value(int& index, int argc, char** argv, std::string* value) {
    if (index + 1 >= argc) {
        std::cerr << "Missing value for " << argv[index] << "\n";
        return false;
    }
    *value = argv[++index];
    return true;
}

bool parse_int(const std::string& value, int* out) {
    char* end = nullptr;
    long parsed = std::strtol(value.c_str(), &end, 10);
    if (end == value.c_str() || *end != '\0' ||
        parsed < std::numeric_limits<int>::min() ||
        parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    *out = static_cast<int>(parsed);
    return true;
}

bool parse_uint64(const std::string& value, uint64_t* out) {
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
    if (end == value.c_str() || *end != '\0') {
        return false;
    }
    *out = static_cast<uint64_t>(parsed);
    return true;
}

bool parse_float(const std::string& value, float* out) {
    char* end = nullptr;
    float parsed = std::strtof(value.c_str(), &end);
    if (end == value.c_str() || *end != '\0') {
        return false;
    }
    *out = parsed;
    return true;
}

bool parse_backend(const std::string& value, ScyllasBandBackend* out) {
    if (value == "auto") {
        *out = SCYLLASBAND_BACKEND_AUTO;
        return true;
    }
    if (value == "litert") {
        *out = SCYLLASBAND_BACKEND_LITERT;
        return true;
    }
    if (value == "coreml") {
        *out = SCYLLASBAND_BACKEND_COREML;
        return true;
    }
    if (value == "onnx") {
        *out = SCYLLASBAND_BACKEND_ONNX;
        return true;
    }
    return false;
}

bool parse_litert_accelerator(const std::string& value, ScyllasBandLiteRtAccelerator* out) {
    if (value == "auto") {
        *out = SCYLLASBAND_LITERT_ACCELERATOR_AUTO;
        return true;
    }
    if (value == "cpu") {
        *out = SCYLLASBAND_LITERT_ACCELERATOR_CPU;
        return true;
    }
    if (value == "gpu") {
        *out = SCYLLASBAND_LITERT_ACCELERATOR_GPU;
        return true;
    }
    if (value == "npu") {
        *out = SCYLLASBAND_LITERT_ACCELERATOR_NPU;
        return true;
    }
    return false;
}

bool parse_sampler(const std::string& value, ScyllasBandSampler* out) {
    if (value == "euler") {
        *out = SCYLLASBAND_SAMPLER_EULER;
        return true;
    }
    if (value == "heun") {
        *out = SCYLLASBAND_SAMPLER_HEUN;
        return true;
    }
    return false;
}

bool parse_args(int argc, char** argv, Options* options) {
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        std::string value;
        if (arg == "--help" || arg == "-h") {
            usage(std::cout);
            std::exit(0);
        } else if (arg == "--bundle") {
            if (!consume_value(index, argc, argv, &options->bundle)) return false;
        } else if (arg == "--output" || arg == "-o") {
            if (!consume_value(index, argc, argv, &options->output)) return false;
        } else if (arg == "--metadata") {
            if (!consume_value(index, argc, argv, &options->metadata)) return false;
        } else if (arg == "--text") {
            if (!consume_value(index, argc, argv, &options->text)) return false;
        } else if (arg == "--file" || arg == "-f") {
            if (!consume_value(index, argc, argv, &options->text_file)) return false;
        } else if (arg == "--voice") {
            if (!consume_value(index, argc, argv, &options->voice)) return false;
        } else if (arg == "--language") {
            if (!consume_value(index, argc, argv, &options->language)) return false;
        } else if (arg == "--emotion") {
            if (!consume_value(index, argc, argv, &options->emotion)) return false;
        } else if (arg == "--emotion-guidance") {
            if (!consume_value(index, argc, argv, &options->emotion_guidance)) return false;
        } else if (arg == "--affect") {
            if (!consume_value(index, argc, argv, &options->affect)) return false;
        } else if (arg == "--affect-guidance-scale") {
            if (!consume_value(index, argc, argv, &value) ||
                !parse_float(value, &options->affect_guidance_scale)) {
                std::cerr << "Invalid --affect-guidance-scale value\n";
                return false;
            }
        } else if (arg == "--steps") {
            if (!consume_value(index, argc, argv, &value) || !parse_int(value, &options->steps)) {
                std::cerr << "Invalid --steps value\n";
                return false;
            }
        } else if (arg == "--sampler") {
            if (!consume_value(index, argc, argv, &value) || !parse_sampler(value, &options->sampler)) {
                std::cerr << "Invalid --sampler value\n";
                return false;
            }
        } else if (arg == "--seed") {
            if (!consume_value(index, argc, argv, &value) || !parse_uint64(value, &options->seed)) {
                std::cerr << "Invalid --seed value\n";
                return false;
            }
            options->has_seed = true;
        } else if (arg == "--long-form") {
            options->long_form = true;
        } else if (arg == "--plan-only") {
            options->plan_only = true;
            options->long_form = true;
        } else if (arg == "--stream") {
            options->stream = true;
            options->long_form = true;
        } else if (arg == "--events") {
            if (!consume_value(index, argc, argv, &options->events)) return false;
        } else if (arg == "--chunk-output-dir") {
            if (!consume_value(index, argc, argv, &options->chunk_output_dir)) return false;
        } else if (arg == "--no-progress") {
            options->progress = false;
        } else if (arg == "--chunk-max-chars") {
            if (!consume_value(index, argc, argv, &value) || !parse_int(value, &options->max_chunk_chars)) {
                std::cerr << "Invalid --chunk-max-chars value\n";
                return false;
            }
        } else if (arg == "--chunk-min-chars") {
            if (!consume_value(index, argc, argv, &value) || !parse_int(value, &options->min_chunk_chars)) {
                std::cerr << "Invalid --chunk-min-chars value\n";
                return false;
            }
        } else if (arg == "--pause-ms") {
            if (!consume_value(index, argc, argv, &value) || !parse_int(value, &options->pause_ms)) {
                std::cerr << "Invalid --pause-ms value\n";
                return false;
            }
        } else if (arg == "--continuation-pause-ms") {
            if (!consume_value(index, argc, argv, &value) || !parse_int(value, &options->continuation_pause_ms)) {
                std::cerr << "Invalid --continuation-pause-ms value\n";
                return false;
            }
        } else if (arg == "--min-sentence-pause-ms") {
            if (!consume_value(index, argc, argv, &value) || !parse_float(value, &options->min_sentence_pause_ms)) {
                std::cerr << "Invalid --min-sentence-pause-ms value\n";
                return false;
            }
        } else if (arg == "--min-clause-pause-ms") {
            if (!consume_value(index, argc, argv, &value) || !parse_float(value, &options->min_clause_pause_ms)) {
                std::cerr << "Invalid --min-clause-pause-ms value\n";
                return false;
            }
        } else if (arg == "--no-prefix-latents") {
            options->use_prefix_latents = false;
        } else if (arg == "--preflight-chunks") {
            options->preflight_chunks = true;
        } else if (arg == "--backend") {
            if (!consume_value(index, argc, argv, &value) || !parse_backend(value, &options->backend)) {
                std::cerr << "Invalid --backend value\n";
                return false;
            }
        } else if (arg == "--litert-accelerator") {
            if (!consume_value(index, argc, argv, &value) || !parse_litert_accelerator(value, &options->litert_accelerator)) {
                std::cerr << "Invalid --litert-accelerator value\n";
                return false;
            }
        } else if (arg == "--no-validate-bundle") {
            options->validate_bundle = false;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return false;
        }
    }
    if (options->bundle.empty()) {
        std::cerr << "--bundle is required\n";
        return false;
    }
    if (options->text.empty() == options->text_file.empty()) {
        std::cerr << "Pass exactly one of --text or --file\n";
        return false;
    }
    if (options->steps <= 0) {
        std::cerr << "--steps must be positive\n";
        return false;
    }
    if (options->plan_only && options->stream) {
        std::cerr << "Pass only one of --plan-only or --stream\n";
        return false;
    }
    return true;
}

std::string read_text_file(const std::string& path) {
    std::ifstream input(path);
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

void write_u16(std::ofstream& out, uint16_t value) {
    out.put(static_cast<char>(value & 0xff));
    out.put(static_cast<char>((value >> 8) & 0xff));
}

void write_u32(std::ofstream& out, uint32_t value) {
    out.put(static_cast<char>(value & 0xff));
    out.put(static_cast<char>((value >> 8) & 0xff));
    out.put(static_cast<char>((value >> 16) & 0xff));
    out.put(static_cast<char>((value >> 24) & 0xff));
}

bool write_wav(const std::string& path, const float* samples, int32_t sample_count, int32_t sample_rate) {
    if (sample_count < 0 || sample_rate <= 0 || (sample_count > 0 && samples == nullptr)) {
        std::cerr << "Invalid synthesis result audio buffer\n";
        return false;
    }
    const uint32_t data_bytes = static_cast<uint32_t>(std::max<int32_t>(0, sample_count) * 2);
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::cerr << "Could not open output WAV: " << path << "\n";
        return false;
    }
    out.write("RIFF", 4);
    write_u32(out, 36U + data_bytes);
    out.write("WAVE", 4);
    out.write("fmt ", 4);
    write_u32(out, 16);
    write_u16(out, 1);
    write_u16(out, 1);
    write_u32(out, static_cast<uint32_t>(sample_rate));
    write_u32(out, static_cast<uint32_t>(sample_rate * 2));
    write_u16(out, 2);
    write_u16(out, 16);
    out.write("data", 4);
    write_u32(out, data_bytes);
    for (int32_t index = 0; index < sample_count; ++index) {
        const float clipped = std::max(-1.0f, std::min(1.0f, samples[index]));
        const int32_t pcm = static_cast<int32_t>(clipped * 32767.0f + (clipped >= 0.0f ? 0.5f : -0.5f));
        write_u16(out, static_cast<uint16_t>(static_cast<int16_t>(pcm)));
    }
    return true;
}

bool write_metadata(const std::string& path, const char* metadata) {
    if (path.empty()) {
        return true;
    }
    std::ofstream out(path);
    if (!out) {
        std::cerr << "Could not open metadata path: " << path << "\n";
        return false;
    }
    out << (metadata == nullptr ? "{}" : metadata) << "\n";
    return true;
}

ScyllasBandLongFormSynthesisRequest make_long_request(
    const ScyllasBandSynthesisRequest& request,
    const Options& options
) {
    ScyllasBandLongFormSynthesisRequest long_request{};
    long_request.request = request;
    long_request.max_chunk_chars = options.max_chunk_chars;
    long_request.min_chunk_chars = options.min_chunk_chars;
    long_request.pause_ms = options.pause_ms;
    long_request.continuation_pause_ms = options.continuation_pause_ms;
    long_request.use_prefix_latents = options.use_prefix_latents ? 1 : 0;
    long_request.disable_auto_split_overlong = 0;
    long_request.preflight_chunks = options.preflight_chunks ? 1 : 0;
    return long_request;
}

std::string json_escape_tool(const std::string& value) {
    std::ostringstream out;
    for (unsigned char c : value) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    constexpr char hex[] = "0123456789abcdef";
                    out << "\\u00" << hex[(c >> 4) & 0x0f] << hex[c & 0x0f];
                } else {
                    out << static_cast<char>(c);
                }
        }
    }
    return out.str();
}

const char* stream_event_name(ScyllasBandStreamingEventType type) {
    switch (type) {
        case SCYLLASBAND_STREAM_EVENT_PLAN_READY: return "plan_ready";
        case SCYLLASBAND_STREAM_EVENT_CHUNK_STARTED: return "chunk_started";
        case SCYLLASBAND_STREAM_EVENT_AUDIO_CHUNK: return "audio_chunk";
        case SCYLLASBAND_STREAM_EVENT_CHUNK_FINISHED: return "chunk_finished";
        case SCYLLASBAND_STREAM_EVENT_WARNING: return "warning";
        case SCYLLASBAND_STREAM_EVENT_DONE: return "done";
    }
    return "unknown";
}

bool write_stream_event_json(
    std::ostream& out,
    const ScyllasBandStreamingEvent* event,
    const std::string& chunk_wav,
    int64_t elapsed_ms,
    int64_t delta_ms,
    int64_t first_audio_ms
) {
    if (event == nullptr) {
        return false;
    }
    out << "{"
        << "\"type\":\"" << stream_event_name(event->type) << "\","
        << "\"chunk_index\":" << event->chunk_index << ","
        << "\"chunk_count\":" << event->chunk_count << ",";
    if (event->chunk_id == nullptr) {
        out << "\"chunk_id\":null,";
    } else {
        out << "\"chunk_id\":\"" << json_escape_tool(event->chunk_id) << "\",";
    }
    out << "\"sample_count\":" << event->sample_count << ","
        << "\"sample_rate\":" << event->sample_rate << ","
        << "\"latent_dim\":" << event->latent_dim << ","
        << "\"latent_frames\":" << event->latent_frames << ","
        << "\"elapsed_ms\":" << elapsed_ms << ","
        << "\"delta_ms\":" << delta_ms << ","
        << "\"first_audio_ms\":" << first_audio_ms;
    if (!chunk_wav.empty()) {
        out << ",\"chunk_wav\":\"" << json_escape_tool(chunk_wav) << "\"";
    }
    out << ",\"metadata\":" << (event->metadata_json == nullptr ? "null" : event->metadata_json)
        << "}\n";
    return static_cast<bool>(out);
}

struct StreamState {
    const Options* options = nullptr;
    std::ofstream events;
    std::vector<float> audio;
    int32_t sample_rate = 0;
    std::string final_metadata;
    std::string error;
    std::chrono::steady_clock::time_point started_at = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point previous_event_at = started_at;
    int64_t first_audio_ms = -1;
};

void write_progress_line(
    const ScyllasBandStreamingEvent* event,
    const std::string& chunk_wav,
    int64_t elapsed_ms,
    int64_t delta_ms,
    int64_t first_audio_ms
) {
    if (event == nullptr) {
        return;
    }
    std::cerr << "[stream] t=" << elapsed_ms << "ms"
              << " +" << delta_ms << "ms"
              << " type=" << stream_event_name(event->type);
    if (event->chunk_id != nullptr) {
        std::cerr << " chunk=" << event->chunk_id;
    }
    if (event->chunk_count >= 0) {
        std::cerr << " count=" << event->chunk_count;
    }
    if (event->type == SCYLLASBAND_STREAM_EVENT_AUDIO_CHUNK) {
        std::cerr << " samples=" << event->sample_count
                  << " first_audio_ms=" << first_audio_ms;
        if (!chunk_wav.empty()) {
            std::cerr << " wav=" << chunk_wav;
        }
    }
    std::cerr << "\n" << std::flush;
}

int stream_callback(const ScyllasBandStreamingEvent* event, void* user_data) {
    auto* state = static_cast<StreamState*>(user_data);
    if (event == nullptr || state == nullptr || state->options == nullptr) {
        return 1;
    }
    const auto now = std::chrono::steady_clock::now();
    const int64_t elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - state->started_at
    ).count();
    const int64_t delta_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - state->previous_event_at
    ).count();
    state->previous_event_at = now;
    if (event->type == SCYLLASBAND_STREAM_EVENT_AUDIO_CHUNK && state->first_audio_ms < 0) {
        state->first_audio_ms = elapsed_ms;
    }
    if (event->type == SCYLLASBAND_STREAM_EVENT_AUDIO_CHUNK) {
        if (event->sample_count < 0 || event->sample_rate <= 0 ||
            (event->sample_count > 0 && event->samples == nullptr)) {
            state->error = "Streaming audio chunk has an invalid sample buffer";
            return 1;
        }
        if (state->sample_rate == 0) {
            state->sample_rate = event->sample_rate;
        } else if (state->sample_rate != event->sample_rate) {
            state->error = "Streaming audio chunks changed sample rate";
            return 1;
        }
        if (event->sample_count > 0) {
            state->audio.insert(
                state->audio.end(),
                event->samples,
                event->samples + event->sample_count
            );
        }
    }
    std::string chunk_wav;
    if (event->type == SCYLLASBAND_STREAM_EVENT_AUDIO_CHUNK && !state->options->chunk_output_dir.empty()) {
        std::error_code error_code;
        std::filesystem::create_directories(state->options->chunk_output_dir, error_code);
        if (error_code) {
            state->error = "Could not create chunk output directory: " + state->options->chunk_output_dir;
            return 1;
        }
        const std::string chunk_id = event->chunk_id == nullptr ? "chunk" : event->chunk_id;
        std::filesystem::path path = state->options->chunk_output_dir;
        path /= chunk_id + ".wav";
        chunk_wav = path.string();
        if (!write_wav(chunk_wav, event->samples, event->sample_count, event->sample_rate)) {
            state->error = "Could not write chunk WAV: " + chunk_wav;
            return 1;
        }
    }
    if (state->events.is_open()) {
        if (!write_stream_event_json(state->events, event, chunk_wav, elapsed_ms, delta_ms, state->first_audio_ms)) {
            state->error = "Could not write streaming event JSONL";
            return 1;
        }
        state->events.flush();
    }
    if (state->options->progress) {
        write_progress_line(event, chunk_wav, elapsed_ms, delta_ms, state->first_audio_ms);
    }
    if (event->type == SCYLLASBAND_STREAM_EVENT_DONE && event->metadata_json != nullptr) {
        state->final_metadata = event->metadata_json;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_args(argc, argv, &options)) {
        usage(std::cerr);
        return 2;
    }

    std::string text = options.text;
    if (!options.text_file.empty()) {
        text = read_text_file(options.text_file);
        if (text.empty()) {
            std::cerr << "Text file is empty or unreadable: " << options.text_file << "\n";
            return 2;
        }
    }

    ScyllasBandRuntimeOptions runtime_options{};
    runtime_options.bundle_dir = options.bundle.c_str();
    runtime_options.backend = options.backend;
    runtime_options.validate_bundle = options.validate_bundle ? 1 : 0;
    runtime_options.litert_accelerator = options.litert_accelerator;
    runtime_options.litert_max_threads = 0;

    ScyllasBandRuntime* runtime = nullptr;
    ScyllasBandStatus status = scyllasband_runtime_create(&runtime_options, &runtime);
    if (status != SCYLLASBAND_STATUS_OK) {
        std::cerr << "scyllasband_runtime_create failed: " << scyllasband_status_message(status)
                  << ": " << (scyllasband_last_error() == nullptr ? "" : scyllasband_last_error()) << "\n";
        return 1;
    }

    ScyllasBandSynthesisRequest request{};
    request.text = text.c_str();
    request.voice_id = options.voice.c_str();
    request.language = options.language.c_str();
    request.emotion = options.emotion.c_str();
    request.emotion_guidance = options.emotion_guidance.empty() ? nullptr : options.emotion_guidance.c_str();
    request.affect = options.affect.empty() ? nullptr : options.affect.c_str();
    request.affect_guidance_scale = options.affect_guidance_scale;
    request.has_affect_guidance_scale = 1;
    request.guidance_null_reference = 1;
    request.emotion_embed_scale = 1.0f;
    request.steps = options.steps;
    request.sampler = options.sampler;
    request.seed = options.seed;
    request.has_seed = options.has_seed ? 1 : 0;
    request.speed = 1.0f;
    request.temperature = 1.0f;
    request.min_sentence_pause_ms = options.min_sentence_pause_ms;
    request.min_clause_pause_ms = options.min_clause_pause_ms;

    if (options.plan_only) {
        ScyllasBandLongFormSynthesisRequest long_request = make_long_request(request, options);
        ScyllasBandChunkPlanResult plan{};
        status = scyllasband_runtime_plan_long_form(runtime, &long_request, &plan);
        if (status != SCYLLASBAND_STATUS_OK) {
            std::cerr << "planning failed: " << scyllasband_status_message(status)
                      << ": " << (scyllasband_last_error() == nullptr ? "" : scyllasband_last_error()) << "\n";
            scyllasband_runtime_destroy(runtime);
            return 1;
        }
        const bool ok = options.metadata.empty()
            ? (std::cout << (plan.metadata_json == nullptr ? "{}" : plan.metadata_json) << "\n", true)
            : write_metadata(options.metadata, plan.metadata_json);
        scyllasband_chunk_plan_result_free(&plan);
        scyllasband_runtime_destroy(runtime);
        return ok ? 0 : 1;
    }

    if (options.stream) {
        ScyllasBandLongFormSynthesisRequest long_request = make_long_request(request, options);
        StreamState stream_state{};
        stream_state.options = &options;
        stream_state.started_at = std::chrono::steady_clock::now();
        stream_state.previous_event_at = stream_state.started_at;
        if (!options.events.empty()) {
            stream_state.events.open(options.events);
            if (!stream_state.events) {
                std::cerr << "Could not open events path: " << options.events << "\n";
                scyllasband_runtime_destroy(runtime);
                return 1;
            }
        }
        status = scyllasband_runtime_synthesize_long_form_stream(runtime, &long_request, stream_callback, &stream_state);
        if (status != SCYLLASBAND_STATUS_OK) {
            std::cerr << "streaming synthesis failed: " << scyllasband_status_message(status)
                      << ": " << (scyllasband_last_error() == nullptr ? "" : scyllasband_last_error()) << "\n";
            if (!stream_state.error.empty()) {
                std::cerr << stream_state.error << "\n";
            }
            scyllasband_runtime_destroy(runtime);
            return 1;
        }
        const int32_t stream_sample_count = static_cast<int32_t>(std::min<std::size_t>(
            stream_state.audio.size(),
            static_cast<std::size_t>(std::numeric_limits<int32_t>::max())
        ));
        const bool ok = write_wav(
            options.output,
            stream_state.audio.empty() ? nullptr : stream_state.audio.data(),
            stream_sample_count,
            stream_state.sample_rate
        ) && write_metadata(
            options.metadata,
            stream_state.final_metadata.empty() ? nullptr : stream_state.final_metadata.c_str()
        );
        std::cout << "{"
                  << "\"status\":\"" << (ok ? "ok" : "error") << "\","
                  << "\"mode\":\"stream\","
                  << "\"output\":\"" << json_escape_tool(options.output) << "\","
                  << "\"metadata\":\"" << json_escape_tool(options.metadata) << "\","
                  << "\"events\":\"" << json_escape_tool(options.events) << "\","
                  << "\"chunk_output_dir\":\"" << json_escape_tool(options.chunk_output_dir) << "\","
                  << "\"sample_rate\":" << stream_state.sample_rate << ","
                  << "\"sample_count\":" << stream_sample_count << ","
                  << "\"first_audio_ms\":" << stream_state.first_audio_ms
                  << "}\n";
        scyllasband_runtime_destroy(runtime);
        return ok ? 0 : 1;
    }

    ScyllasBandSynthesisResult result{};
    if (options.long_form) {
        ScyllasBandLongFormSynthesisRequest long_request = make_long_request(request, options);
        status = scyllasband_runtime_synthesize_long_form(runtime, &long_request, &result);
    } else {
        status = scyllasband_runtime_synthesize(runtime, &request, &result);
    }

    if (status != SCYLLASBAND_STATUS_OK) {
        std::cerr << "synthesis failed: " << scyllasband_status_message(status)
                  << ": " << (scyllasband_last_error() == nullptr ? "" : scyllasband_last_error()) << "\n";
        scyllasband_runtime_destroy(runtime);
        return 1;
    }

    const bool ok = write_wav(options.output, result.samples, result.sample_count, result.sample_rate) &&
                    write_metadata(options.metadata, result.metadata_json);
    std::cout << "{"
              << "\"status\":\"" << (ok ? "ok" : "error") << "\","
              << "\"output\":\"" << options.output << "\","
              << "\"metadata\":\"" << options.metadata << "\","
              << "\"sample_rate\":" << result.sample_rate << ","
              << "\"sample_count\":" << result.sample_count << ","
              << "\"latent_dim\":" << result.latent_dim << ","
              << "\"latent_frames\":" << result.latent_frames
              << "}\n";

    scyllasband_synthesis_result_free(&result);
    scyllasband_runtime_destroy(runtime);
    return ok ? 0 : 1;
}
