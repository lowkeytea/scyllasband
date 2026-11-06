/*
 * Scylla's Band native runtime: text-to-speech for scyllasband_measured_controls_v3 and _v4 bundles, and singing from a
 * melody score on v4 bundles (scyllasband_sing, scyllasband_sing_stream).
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

#define SCYLLASBAND_ABI_VERSION "3.1.0"   /* 3.1.0 adds singing; every 3.0.0 type and function is unchanged */

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
    SCYLLASBAND_ACCELERATOR_AUTO = 2,  /* LiteRT: the GPU for the graphs the bundle lists (controls.litert.gpu.graphs) when they
                                          compile for it, the CPU for the rest (every graph when it lists none). Core ML / Core AI:
                                          the bundle's recommended compute unit per graph */
    SCYLLASBAND_ACCELERATOR_NEURAL_ENGINE = 3  /* Core ML / Core AI: the graphs the bundle lists as accurate on the Neural
                                                  Engine run there, the rest on the CPU (usable from background apps) */
} ScyllasBandAccelerator;

typedef enum {
    SCYLLASBAND_SAMPLER_DEFAULT = 0,  /* the bundle default: for speech Euler in the released bundles, for singing the bundle's singing sampler */
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
    /* NULL, "" or "default": the voice's own delivery. Otherwise control=value pairs, e.g. "intensity=1,pace=-0.5,pitch=0":
       intensity (calm, negative, to intense, positive) and pace (slower to faster, words and pauses together) in the voice's
       own standard deviations, pitch as a shift in semitones; each clamped into the bundle's ranges (-2 to 2 for all three
       in the released model, see scyllasband_voices_json "delivery"); omitted controls take the voice's default. */
    const char* delivery;
    float speed;              /* a plain time stretch on top of pace; 1.0 = none; must be positive */
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

/* {"backend", "accelerator", "sample_rate", "release_id", "default_voice", "voices": [{"id", "languages", "default_language"}],
   "delivery": {"ranges": {control: [low, high]}, "defaults": {voice: {control: value}}, "units": {control: unit}},
   "graph_contract", "singing": {"supported", "styles", "decoders", "default_decoder", "sampler", "steps", "guidance",
   "voice_midi_center": {voice: MIDI pitch}, "register_band": [below, above] (semitones around the centre the register aims
   for), "max_phrase_seconds"}} ("singing" has supported = false on spoken-only bundles). Owned by the runtime. */
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

/* --- records (ABI 3.1.0): several voices, languages and deliveries in one request, as group-speak --------------------------

   records_json is a JSON array of {"text", "voice", "language"?, "delivery"?, "intensity"?, "pace"?, "pitch"?} objects: the
   delivery as a control=value string or a {"intensity", "pace", "pitch"} mapping, or the controls as fields (not both);
   omitted controls take the voice's default. Consecutive records with the same voice, language and delivery are spoken as one
   continuous chain; a change of any starts a new chain (with pause_ms of silence before it). Sentence i of the plan draws its
   noise from seed + i, across records. */
typedef struct {
    uint32_t struct_size;      /* sizeof(ScyllasBandRecordsRequest): set by scyllasband_records_request_init, checked by every call */
    const char* records_json;
    float speed;               /* a plain time stretch on top of pace; 1.0 = none */
    int32_t steps;             /* flow steps; 0 = bundle default */
    ScyllasBandSampler sampler;
    uint64_t seed;
    int32_t has_seed;          /* 0: fresh random noise */
    float temperature;         /* noise scale; 1.0 = default */
    int32_t normalize_text;    /* nonzero: expand numbers, dates, symbols and abbreviations before G2P */
    float pause_ms;            /* silence before each chain after the first */
} ScyllasBandRecordsRequest;

/* Fills the defaults and struct_size: speed 1, steps 0, default sampler, no seed, temperature 1, normalize_text 1, pause 0. */
SCYLLASBAND_API void scyllasband_records_request_init(ScyllasBandRecordsRequest* request);

/* The sentence plan for the records (as scyllasband_plan_json: {"source_kind", "chunks": [...], "chains"}), without
   synthesizing. Free *out_json with scyllasband_string_free. */
SCYLLASBAND_API ScyllasBandStatus scyllasband_plan_records_json(ScyllasBandRuntime* runtime, const ScyllasBandRecordsRequest* request,
                                                                char** out_json);

/* Speaks the records target by target, as scyllasband_synthesize_stream; the AUDIO event of a chain's first target starts
   with the pause. */
SCYLLASBAND_API ScyllasBandStatus scyllasband_synthesize_records_stream(ScyllasBandRuntime* runtime, const ScyllasBandRecordsRequest* request,
                                                                        ScyllasBandEventCallback callback, void* user_data);

/* --- singing (ABI 3.1.0; v4 bundles whose voices_json "singing.supported" is true) ---------------------------------------

   A score is JSON: {"text", "bpm", "style": "pop" | "ballad", "notes": [[label, note, beats], ...]}; note a pitch name with
   octave ("C4", "F#3", "Bb3") or "rest" (label "-"); a word over several notes is numbered ("baby~1", "baby~2", or its
   syllables "ba~1", "by~2"). The song is sung phrase by phrase (phrases split at rests of 0.25 s or more), moved by octaves
   into the voice's sung range. */
typedef struct {
    uint32_t struct_size;      /* sizeof(ScyllasBandSingRequest): set by scyllasband_sing_request_init, checked by every call */
    const char* score_json;
    const char* voice_id;
    const char* language;      /* NULL or "": the voice's default language */
    int32_t transpose;         /* semitones added to the whole score before the octave register */
    int32_t use_register;      /* nonzero (default): move the song by octaves into the voice's sung range */
    int32_t steps;             /* flow steps; 0: the bundle's singing setting (8) */
    ScyllasBandSampler sampler;/* DEFAULT: the bundle's singing sampler (v4-20261106: Euler) */
    uint64_t seed;             /* phrase i draws its noise from seed + i */
    int32_t has_seed;          /* 0: fresh random noise */
    float guidance;            /* classifier-free guidance on the melody and mode when has_guidance is set; 1 turns it off */
    int32_t has_guidance;      /* 0: the bundle's setting (1, off, for v4-20261106) */
    /* NULL or "": the bundle's singing decoder ("bigvgan" where this backend has it, else "vocos"); "bigvgan"; or "vocos", the
       speech decoder: much faster ("turbo"), with some harmonic artefacts on sustained notes in some voices. */
    const char* decoder;
} ScyllasBandSingRequest;

/* Fills the defaults and struct_size: no transpose, register on, the bundle's steps, sampler, guidance and decoder, no seed. */
SCYLLASBAND_API void scyllasband_sing_request_init(ScyllasBandSingRequest* request);

/* Checks a score without a runtime and returns it normalized after transposing it by `transpose` semitones: {"text", "bpm",
   "style", "mode", "seconds", "transpose", "notes": [{"index", "label", "word_index", "part", "note", "midi", "beats", "start",
   "end"}], "words": [{"word", "start", "end", "parts"}], "range": {"low", "high", "mean", "low_note", "high_note"}, "lines":
   [{"first", "last"}]} (lines: note index ranges that end at rests of 0.25 s or more). On SCYLLASBAND_ERROR_INVALID_ARGUMENT
   *out_json is {"error": message, "note_index": index of the note at fault, or null}. Thread-safe. Free *out_json with
   scyllasband_string_free. */
SCYLLASBAND_API ScyllasBandStatus scyllasband_score_parse(const char* score_json, int32_t transpose, char** out_json);

/* How the score sits in the voice's range after the transpose and the octave register: {"voice_center", "band": [low, high],
   "transpose", "register_shift", "mean_midi", "low_midi", "high_midi", "low_note", "high_note", "outside_semitones",
   "suggested_transpose"} (suggested_transpose: the request's transpose when the song fits, else the smallest transpose in
   [-6, 6] that fits or comes nearest). Runs no graph and does not wait for a request in progress. */
SCYLLASBAND_API ScyllasBandStatus scyllasband_sing_fit_json(ScyllasBandRuntime* runtime, const ScyllasBandSingRequest* request, char** out_json);

/* The phrase plan for a score, without singing: {"voice", "language", "mode", "mode_id", "register_shift", "transpose", "bpm",
   "style", "text", "seconds", "samples", "decoder", "steps", "sampler", "guidance", "fit", "phrases": [{"index", "start",
   "end", "text", "frames", "phones", "durations", "notes"}]}. Runs the bundle's G2P. */
SCYLLASBAND_API ScyllasBandStatus scyllasband_sing_plan_json(ScyllasBandRuntime* runtime, const ScyllasBandSingRequest* request, char** out_json);

/* Sings the whole score into one waveform (silence in the rests). Free with scyllasband_audio_free. metadata_json: the plan,
   the rendered phrases (bucket, seconds, seed) and the settings used. */
SCYLLASBAND_API ScyllasBandStatus scyllasband_sing(ScyllasBandRuntime* runtime, const ScyllasBandSingRequest* request, ScyllasBandAudio* out_audio);

/* Sings phrase by phrase, calling `callback` from the calling thread:
   - SCYLLASBAND_EVENT_PLAN: metadata is the plan (as scyllasband_sing_plan_json, with "samples", the song's length);
   - SCYLLASBAND_EVENT_CHUNK_STARTED per phrase (chunk_id "phrase_0001", ...);
   - SCYLLASBAND_EVENT_AUDIO once per phrase: the song audio that phrase settles, contiguous with the previous AUDIO event's
     (metadata "start_sample", "sample_count", "song_samples"); appended in order, the events are the whole song, sample for
     sample as scyllasband_sing returns it. An event may carry no samples;
   - SCYLLASBAND_EVENT_DONE: timing summary.
   Cancelling (or returning nonzero from the callback) stops at the next graph call; audio already delivered is final. */
SCYLLASBAND_API ScyllasBandStatus scyllasband_sing_stream(ScyllasBandRuntime* runtime, const ScyllasBandSingRequest* request,
                                                          ScyllasBandEventCallback callback, void* user_data);

/* The score as plain tones at the pitch it will be sung (after the transpose and, with a runtime, the voice's octave register),
   for an instant preview. `runtime` may be NULL (no register); sample_rate 0 takes the runtime's (24000 without one). Runs no
   graph and does not wait for a request in progress. metadata_json: {"transpose", "register_shift", "seconds"}. Free with
   scyllasband_audio_free. */
SCYLLASBAND_API ScyllasBandStatus scyllasband_sing_preview(ScyllasBandRuntime* runtime, const ScyllasBandSingRequest* request,
                                                           int32_t sample_rate, ScyllasBandAudio* out_audio);

/* Loads the graphs a short sung phrase needs with `decoder` (NULL or "": the bundle's singing decoder), so the first song
   starts quickly. voice_id may be NULL. */
SCYLLASBAND_API ScyllasBandStatus scyllasband_warmup_singing(ScyllasBandRuntime* runtime, const char* voice_id, const char* decoder);

SCYLLASBAND_API void scyllasband_audio_free(ScyllasBandAudio* audio);
SCYLLASBAND_API void scyllasband_string_free(char* value);

#ifdef __cplusplus
}
#endif

#endif /* SCYLLASBAND_H */
