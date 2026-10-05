// scyllasband_speak: synthesize text with libscyllasband and write a 16-bit PCM WAV.
#include "scyllasband.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

const char* kUsage =
    "usage: scyllasband_speak --bundle DIR --voice ID (--text TEXT | --file PATH) [--output OUT.wav]\n"
    "  [--language CODE] [--delivery SPEC] [--speed 1.0] [--steps N] [--sampler heun|euler] [--seed N]\n"
    "  [--temperature 1.0] [--threads N] [--accelerator cpu|gpu|auto|ane] [--no-normalize] [--metadata OUT.json]\n"
    "  [--plan-only] [--stream] [--warmup] [--timings N]\n"
    "  --timings N: speak the text N times in this process and print load, warm-up and per-request first-audio and total\n"
    "  times as JSON (the first request includes opening its graphs; later ones show a warm runtime)\n";

double milliseconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

void write_wav(const std::string& path, const float* samples, int64_t count, int32_t rate) {
    std::ofstream out(path, std::ios::binary);
    auto u32 = [&](uint32_t v) { out.write(reinterpret_cast<const char*>(&v), 4); };
    auto u16 = [&](uint16_t v) { out.write(reinterpret_cast<const char*>(&v), 2); };
    const uint32_t bytes = static_cast<uint32_t>(count * 2);
    out.write("RIFF", 4);
    u32(36 + bytes);
    out.write("WAVEfmt ", 8);
    u32(16);
    u16(1);
    u16(1);
    u32(static_cast<uint32_t>(rate));
    u32(static_cast<uint32_t>(rate) * 2);
    u16(2);
    u16(16);
    out.write("data", 4);
    u32(bytes);
    for (int64_t i = 0; i < count; ++i) {
        float clipped = std::fmax(-1.0f, std::fmin(1.0f, samples[i]));
        u16(static_cast<uint16_t>(static_cast<int16_t>(std::lrint(clipped * 32767.0f))));
    }
}

int fail(const char* what) {
    std::fprintf(stderr, "scyllasband_speak: %s: %s\n", what, scyllasband_last_error());
    return 1;
}

}  // namespace

int main(int argc, char** argv) {
    std::string bundle, text, output, metadata_path, sampler, accelerator = "cpu";
    ScyllasBandRequest request;
    scyllasband_request_init(&request);
    std::string voice, language, delivery;
    bool plan_only = false, stream = false, warm = false;
    int threads = 0, timings = 0;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n%s", arg.c_str(), kUsage);
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--bundle") bundle = value();
        else if (arg == "--text") text = value();
        else if (arg == "--file") {
            std::ifstream in(value());
            std::stringstream buffer;
            buffer << in.rdbuf();
            text = buffer.str();
        } else if (arg == "--voice") voice = value();
        else if (arg == "--language") language = value();
        else if (arg == "--delivery") delivery = value();
        else if (arg == "--speed") request.speed = std::stof(value());
        else if (arg == "--steps") request.steps = std::stoi(value());
        else if (arg == "--sampler") sampler = value();
        else if (arg == "--seed") request.seed = std::stoull(value()), request.has_seed = 1;
        else if (arg == "--temperature") request.temperature = std::stof(value());
        else if (arg == "--threads") threads = std::stoi(value());
        else if (arg == "--accelerator") accelerator = value();
        else if (arg == "--no-normalize") request.normalize_text = 0;
        else if (arg == "--output") output = value();
        else if (arg == "--metadata") metadata_path = value();
        else if (arg == "--plan-only") plan_only = true;
        else if (arg == "--stream") stream = true;
        else if (arg == "--warmup") warm = true;
        else if (arg == "--timings") timings = std::stoi(value());
        else {
            std::fprintf(stderr, "unknown argument %s\n%s", arg.c_str(), kUsage);
            return 2;
        }
    }
    if (bundle.empty() || voice.empty() || text.empty()) {
        std::fprintf(stderr, "%s", kUsage);
        return 2;
    }
    request.text = text.c_str();
    request.voice_id = voice.c_str();
    request.language = language.empty() ? nullptr : language.c_str();
    request.delivery = delivery.empty() ? nullptr : delivery.c_str();
    request.sampler = sampler == "euler" ? SCYLLASBAND_SAMPLER_EULER : sampler == "heun" ? SCYLLASBAND_SAMPLER_HEUN : SCYLLASBAND_SAMPLER_DEFAULT;

    ScyllasBandRuntimeOptions options{};
    options.bundle_dir = bundle.c_str();
    options.threads = threads;
    options.accelerator = accelerator == "gpu"    ? SCYLLASBAND_ACCELERATOR_GPU
                          : accelerator == "auto" ? SCYLLASBAND_ACCELERATOR_AUTO
                          : accelerator == "ane"  ? SCYLLASBAND_ACCELERATOR_NEURAL_ENGINE
                                                  : SCYLLASBAND_ACCELERATOR_CPU;
    ScyllasBandRuntime* runtime = nullptr;
    const auto load_start = std::chrono::steady_clock::now();
    if (scyllasband_runtime_create(&options, &runtime) != SCYLLASBAND_OK) return fail("cannot load the bundle");
    const double load_ms = milliseconds_since(load_start);
    double warmup_ms = 0.0;
    if (warm) {
        const auto start = std::chrono::steady_clock::now();
        if (scyllasband_warmup(runtime, voice.c_str()) != SCYLLASBAND_OK) return fail("warm-up failed");
        warmup_ms = milliseconds_since(start);
    }

    int status = 0;
    if (timings > 0) {
        struct Timing {
            std::chrono::steady_clock::time_point start;
            double first_audio_ms = -1.0;
            int64_t samples = 0;
            int32_t rate = 24000;
        };
        std::printf("{\"load_ms\": %.1f, \"warmup_ms\": %.1f, \"requests\": [", load_ms, warmup_ms);
        for (int run = 0; run < timings && status == 0; ++run) {
            Timing timing{std::chrono::steady_clock::now()};
            auto callback = [](const ScyllasBandEvent* event, void* user) -> int32_t {
                auto* t = static_cast<Timing*>(user);
                if (event->type == SCYLLASBAND_EVENT_AUDIO) {
                    if (t->first_audio_ms < 0) t->first_audio_ms = milliseconds_since(t->start);
                    t->samples += event->sample_count;
                    t->rate = event->sample_rate;
                }
                return 0;
            };
            if (scyllasband_synthesize_stream(runtime, &request, callback, &timing) != SCYLLASBAND_OK) {
                status = fail("synthesis failed");
                break;
            }
            const double total_ms = milliseconds_since(timing.start);
            std::printf("%s{\"first_audio_ms\": %.1f, \"total_ms\": %.1f, \"audio_s\": %.3f}", run ? ", " : "", timing.first_audio_ms, total_ms,
                        static_cast<double>(timing.samples) / timing.rate);
        }
        std::printf("]}\n");
    } else if (plan_only) {
        char* plan = nullptr;
        if (scyllasband_plan_json(runtime, &request, &plan) != SCYLLASBAND_OK) status = fail("planning failed");
        else std::printf("%s\n", plan);
        scyllasband_string_free(plan);
    } else if (stream) {
        struct State {
            std::vector<float> samples;
            int32_t rate = 0;
        } state;
        auto callback = [](const ScyllasBandEvent* event, void* user) -> int32_t {
            auto* s = static_cast<State*>(user);
            std::fprintf(stderr, "[%d] %s %s\n", static_cast<int>(event->type), event->chunk_id ? event->chunk_id : "-", event->metadata_json);
            if (event->type == SCYLLASBAND_EVENT_AUDIO) {
                s->samples.insert(s->samples.end(), event->samples, event->samples + event->sample_count);
                s->rate = event->sample_rate;
            }
            return 0;
        };
        if (scyllasband_synthesize_stream(runtime, &request, callback, &state) != SCYLLASBAND_OK) status = fail("synthesis failed");
        else if (!output.empty()) write_wav(output, state.samples.data(), static_cast<int64_t>(state.samples.size()), state.rate);
    } else {
        ScyllasBandAudio audio{};
        if (scyllasband_synthesize(runtime, &request, &audio) != SCYLLASBAND_OK) {
            status = fail("synthesis failed");
        } else {
            if (!output.empty()) write_wav(output, audio.samples, audio.sample_count, audio.sample_rate);
            if (!metadata_path.empty()) std::ofstream(metadata_path) << audio.metadata_json;
            std::fprintf(stderr, "%lld samples at %d Hz\n", static_cast<long long>(audio.sample_count), audio.sample_rate);
        }
        scyllasband_audio_free(&audio);
    }
    scyllasband_runtime_destroy(runtime);
    return status;
}
