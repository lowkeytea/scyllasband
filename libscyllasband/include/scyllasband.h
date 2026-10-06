/*
 * Scylla's Band native runtime: text-to-speech for scyllasband_measured_delivery_v2 bundles.
 *
 * One graph backend is linked into each build: ONNX Runtime (bundle flavor "onnx"), LiteRT ("litert"), or on Apple
 * platforms Core ML and Core AI ("coreml", iOS 18 / macOS 15; "coreai", iOS 27 / macOS 27). scyllasband_backend() names it.
 *
 * Strings are UTF-8. Audio is mono float PCM in [-1, 1] at the bundle sample rate. A runtime serializes its
 * requests; create one runtime per bundle and reuse it. Functions that return a status record a thread-local
 * message for scyllasband_last_error() on failure.
 */
#ifndef SCYLLASBAND_H
#define SCYLLASBAND_H

#include <stdint.h>

#if defined(_WIN32)
#if defined(SCYLLASBAND_BUILDING)
#define SCYLLASBAND_API __declspec(dllexport)
#else
#define SCYLLASBAND_API __declspec(dllimport)
#endif
#else
#define SCYLLASBAND_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SCYLLASBAND_ABI_VERSION "2.0.0"

typedef struct ScyllasBandRuntime ScyllasBandRuntime;

typedef enum {
    SCYLLASBAND_OK = 0,
    SCYLLASBAND_ERROR_INVALID_ARGUMENT = 1,
    SCYLLASBAND_ERROR_RUNTIME = 2,
    SCYLLASBAND_ERROR_CANCELLED = 3
} ScyllasBandStatus;

typedef enum {
    SCYLLASBAND_ACCELERATOR_CPU = 0,
    SCYLLASBAND_ACCELERATOR_GPU = 1,  /* LiteRT GPU, with CPU for unsupported ops */
    SCYLLASBAND_ACCELERATOR_AUTO = 2,  /* LiteRT: GPU when the graphs compile for it, else CPU. Core ML / Core AI: the bundle's
                                          recommended compute unit per graph */
    SCYLLASBAND_ACCELERATOR_NEURAL_ENGINE = 3  /* Core ML / Core AI: the graphs the bundle lists as accurate on the Neural
                                                  Engine run there, the rest on the CPU (usable from background apps) */
} ScyllasBandAccelerator;

typedef enum {
    SCYLLASBAND_SAMPLER_DEFAULT = 0,  /* the bundle default (Heun) */
    SCYLLASBAND_SAMPLER_HEUN = 1,
    SCYLLASBAND_SAMPLER_EULER = 2
} ScyllasBandSampler;

typedef struct {
    const char* bundle_dir;               /* directory holding manifest.json */
    int32_t threads;                      /* CPU threads per graph; 0: up to 4 (LiteRT), ONNX Runtime's default (ONNX) */
    ScyllasBandAccelerator accelerator;   /* LiteRT and Apple builds; ONNX Runtime builds run on CPU */
} ScyllasBandRuntimeOptions;

typedef struct {
    const char* text;
    const char* voice_id;
    const char* language;     /* NULL or "": the voice's default language ("en" also selects it) */
    /* NULL, "" or "neutral": neutral delivery. "auto": unconditioned. Otherwise axis=value pairs, e.g.
       "energy=2.5,tension=2,valence=2,assertiveness=2,whisper=off": axes on 0-4 (2 = neutral) or "auto",
       whisper on/off/auto; omitted axes stay neutral. */
    const char* delivery;
    float speed;              /* duration scale; 1.0 = the model's own pace; must be positive */
    int32_t steps;            /* flow steps; 0 = bundle default (8) */
    ScyllasBandSampler sampler;
    uint64_t seed;            /* sentence i of the plan draws its noise from seed + i */
    int32_t has_seed;         /* 0: fresh random noise */
    float temperature;        /* noise scale; 1.0 = default */
    int32_t normalize_text;   /* nonzero: expand numbers, dates, symbols and abbreviations before G2P */
} ScyllasBandRequest;

typedef struct {
    float* samples;
    int64_t sample_count;
    int32_t sample_rate;
    char* metadata_json;
} ScyllasBandAudio;

typedef enum {
    SCYLLASBAND_EVENT_PLAN = 1,           /* metadata_json: the sentence plan */
    SCYLLASBAND_EVENT_CHUNK_STARTED = 2,  /* metadata_json: the target about to be spoken (whole plan sentences or part of one) */
    SCYLLASBAND_EVENT_AUDIO = 3,          /* samples of one target; they join the previous audio directly */
    SCYLLASBAND_EVENT_DONE = 4            /* metadata_json: timing summary */
} ScyllasBandEventType;

typedef struct {
    ScyllasBandEventType type;
    int32_t chunk_index;   /* plan index of the target's last sentence; -1 for plan and done events */
    int32_t chunk_count;   /* sentences in the plan */
    const char* chunk_id;  /* NULL for plan and done events */
    const char* metadata_json;
    const float* samples;  /* audio events only; valid during the callback */
    int64_t sample_count;
    int32_t sample_rate;
} ScyllasBandEvent;

/* Return 0 to continue, nonzero to stop (the call then returns SCYLLASBAND_ERROR_CANCELLED). */
typedef int32_t (*ScyllasBandEventCallback)(const ScyllasBandEvent* event, void* user_data);

SCYLLASBAND_API const char* scyllasband_version(void);
SCYLLASBAND_API const char* scyllasband_backend(void); /* "onnx", "litert" or "apple" (Core ML and Core AI bundles) */
SCYLLASBAND_API const char* scyllasband_last_error(void);

/* Fills the defaults: speed 1, steps 0, default sampler, no seed, temperature 1, normalize_text 1. */
SCYLLASBAND_API void scyllasband_request_init(ScyllasBandRequest* request);

SCYLLASBAND_API ScyllasBandStatus scyllasband_runtime_create(const ScyllasBandRuntimeOptions* options, ScyllasBandRuntime** out_runtime);
SCYLLASBAND_API void scyllasband_runtime_destroy(ScyllasBandRuntime* runtime);

/* {"backend", "accelerator", "sample_rate", "release_id", "default_voice", "voices": [{"id", "languages", "default_language"}]}.
   Owned by the runtime. */
SCYLLASBAND_API const char* scyllasband_voices_json(ScyllasBandRuntime* runtime);

/* Loads the graphs a short sentence needs so the first request starts quickly. voice_id may be NULL. */
SCYLLASBAND_API ScyllasBandStatus scyllasband_warmup(ScyllasBandRuntime* runtime, const char* voice_id);

/* The sentence plan for a request (normalized text split into sentences), without synthesizing.
   Free *out_json with scyllasband_string_free. */
SCYLLASBAND_API ScyllasBandStatus scyllasband_plan_json(ScyllasBandRuntime* runtime, const ScyllasBandRequest* request, char** out_json);

/* Speaks the whole request into one waveform. Free with scyllasband_audio_free. */
SCYLLASBAND_API ScyllasBandStatus scyllasband_synthesize(ScyllasBandRuntime* runtime, const ScyllasBandRequest* request, ScyllasBandAudio* out_audio);

/* Speaks the request target by target, calling `callback` from the calling thread as audio is ready. */
SCYLLASBAND_API ScyllasBandStatus scyllasband_synthesize_stream(ScyllasBandRuntime* runtime, const ScyllasBandRequest* request,
                                                                ScyllasBandEventCallback callback, void* user_data);

/* Stops, at their next graph call, the requests already issued on `runtime` (running or waiting for it); later
   requests are unaffected. Safe to call from any thread. */
SCYLLASBAND_API void scyllasband_cancel(ScyllasBandRuntime* runtime);

SCYLLASBAND_API void scyllasband_audio_free(ScyllasBandAudio* audio);
SCYLLASBAND_API void scyllasband_string_free(char* value);

#ifdef __cplusplus
}
#endif

#endif /* SCYLLASBAND_H */
