#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ScyllasBandRuntime ScyllasBandRuntime;

typedef struct ScyllasBandLiteRtSession ScyllasBandLiteRtSession;

typedef enum {
    SCYLLASBAND_TENSOR_FLOAT32 = 1,
    SCYLLASBAND_TENSOR_FLOAT16 = 2,
    SCYLLASBAND_TENSOR_FLOAT64 = 3,
    SCYLLASBAND_TENSOR_INT64 = 4,
    SCYLLASBAND_TENSOR_INT32 = 5,
    SCYLLASBAND_TENSOR_INT16 = 6,
    SCYLLASBAND_TENSOR_INT8 = 7,
    SCYLLASBAND_TENSOR_UINT8 = 8,
    SCYLLASBAND_TENSOR_BOOL = 9
} ScyllasBandTensorType;

typedef enum {
    SCYLLASBAND_LITERT_ACCELERATOR_AUTO = 0,
    SCYLLASBAND_LITERT_ACCELERATOR_CPU = 1,
    SCYLLASBAND_LITERT_ACCELERATOR_GPU = 2,
    SCYLLASBAND_LITERT_ACCELERATOR_NPU = 3
} ScyllasBandLiteRtAccelerator;

typedef struct {
    const char* name;
    int32_t data_type;
    const int64_t* shape;
    int32_t rank;
    const void* data;
    uint64_t byte_length;
} ScyllasBandTensorView;

typedef struct {
    char* name;
    int32_t data_type;
    int64_t* shape;
    int32_t rank;
    void* data;
    uint64_t byte_length;
} ScyllasBandOwnedTensor;


typedef enum {
    SCYLLASBAND_STATUS_OK = 0,
    SCYLLASBAND_STATUS_INVALID_ARGUMENT = 1,
    SCYLLASBAND_STATUS_NOT_IMPLEMENTED = 2,
    SCYLLASBAND_STATUS_RUNTIME_ERROR = 3
} ScyllasBandStatus;

typedef enum {
    /* Resolves to the graph runtime linked into the current build. */
    SCYLLASBAND_BACKEND_AUTO = 0,
    SCYLLASBAND_BACKEND_LITERT = 1,
    SCYLLASBAND_BACKEND_COREML = 2,
    SCYLLASBAND_BACKEND_ONNX = 3,
    SCYLLASBAND_BACKEND_COREAI = 4
} ScyllasBandBackend;

typedef enum {
    SCYLLASBAND_SAMPLER_EULER = 0,
    SCYLLASBAND_SAMPLER_HEUN = 1
} ScyllasBandSampler;

typedef enum {
    SCYLLASBAND_DURATION_HIERARCHY_DEFAULT = 0,
    SCYLLASBAND_DURATION_HIERARCHY_P50 = 1,
    SCYLLASBAND_DURATION_HIERARCHY_SAMPLED = 2
} ScyllasBandDurationHierarchyMode;

typedef struct {
    const char* bundle_dir;
    ScyllasBandBackend backend;
    int32_t validate_bundle;
    ScyllasBandLiteRtAccelerator litert_accelerator;
    int32_t litert_max_threads;
} ScyllasBandRuntimeOptions;

typedef struct {
    const char* text;
    const char* explicit_phones;
    const char* voice_id;
    const char* language;
    const char* emotion;
    const char* emotion_guidance;
    int32_t guidance_null_reference;
    float emotion_embed_scale;
    const float* prefix_latents;
    int32_t prefix_latent_dim;
    int32_t prefix_latent_frames;
    const char* context_before;
    const char* context_after;
    int32_t chunk_index;
    int32_t chunk_count;
    const char* boundary_before;
    const char* boundary_after;
    float min_sentence_pause_ms;
    float min_clause_pause_ms;
    int32_t steps;
    ScyllasBandSampler sampler;
    uint64_t seed;
    int32_t has_seed;
    float speed;
    float temperature;
    const char* affect;
    float affect_guidance_scale;
    int32_t has_affect_guidance_scale;
    ScyllasBandDurationHierarchyMode duration_hierarchy_mode;
} ScyllasBandSynthesisRequest;

typedef struct {
    float* samples;
    int32_t sample_count;
    int32_t sample_rate;
    char* metadata_json;
    float* latents;
    int32_t latent_dim;
    int32_t latent_frames;
} ScyllasBandSynthesisResult;

typedef struct {
    int32_t predicted_latent_frames;
    int32_t fixed_latent_frames;
    char* metadata_json;
} ScyllasBandDurationEstimateResult;

typedef struct {
    ScyllasBandSynthesisRequest request;
    int32_t max_chunk_chars;
    int32_t min_chunk_chars;
    int32_t pause_ms;
    int32_t continuation_pause_ms;
    int32_t use_prefix_latents;
    int32_t disable_auto_split_overlong;
    int32_t preflight_chunks;
} ScyllasBandLongFormSynthesisRequest;

typedef struct {
    const char* text;
    int32_t max_chunk_chars;
    int32_t min_chunk_chars;
} ScyllasBandChunkPlanRequest;

typedef struct {
    char* metadata_json;
} ScyllasBandChunkPlanResult;

typedef enum {
    SCYLLASBAND_STREAM_EVENT_PLAN_READY = 1,
    SCYLLASBAND_STREAM_EVENT_CHUNK_STARTED = 2,
    SCYLLASBAND_STREAM_EVENT_AUDIO_CHUNK = 3,
    SCYLLASBAND_STREAM_EVENT_CHUNK_FINISHED = 4,
    SCYLLASBAND_STREAM_EVENT_WARNING = 5,
    SCYLLASBAND_STREAM_EVENT_DONE = 6
} ScyllasBandStreamingEventType;

typedef struct {
    ScyllasBandStreamingEventType type;
    int32_t chunk_index;
    int32_t chunk_count;
    const char* chunk_id;
    const char* metadata_json;
    const float* samples;
    int32_t sample_count;
    int32_t sample_rate;
    const float* latents;
    int32_t latent_dim;
    int32_t latent_frames;
} ScyllasBandStreamingEvent;

typedef int32_t (*ScyllasBandStreamingCallback)(
    const ScyllasBandStreamingEvent* event,
    void* user_data
);


const char* scyllasband_last_error(void);
void scyllasband_clear_error(void);
/** Internal graph adapters use this to populate the thread-local native error. */
void scyllasband_graph_session_set_error(const char* message);
uint64_t scyllasband_tensor_element_size(int32_t data_type);
void scyllasband_tensors_destroy(ScyllasBandOwnedTensor* tensors, int32_t tensor_count);

const char* scyllasband_version(void);

ScyllasBandStatus scyllasband_runtime_create(
    const ScyllasBandRuntimeOptions* options,
    ScyllasBandRuntime** out_runtime
);

void scyllasband_runtime_destroy(ScyllasBandRuntime* runtime);

/*
 * Bound the number of target-bucket vector/vocoder session pairs retained by
 * a runtime. Zero keeps every encountered bucket warm; positive values use
 * least-recently-used eviction. Existing cached buckets are trimmed
 * immediately when the capacity is lowered.
 */
ScyllasBandStatus scyllasband_runtime_set_target_bucket_cache_capacity(
    ScyllasBandRuntime* runtime,
    int32_t capacity
);

ScyllasBandStatus scyllasband_runtime_synthesize(
    ScyllasBandRuntime* runtime,
    const ScyllasBandSynthesisRequest* request,
    ScyllasBandSynthesisResult* out_result
);

ScyllasBandStatus scyllasband_runtime_synthesize_long_form(
    ScyllasBandRuntime* runtime,
    const ScyllasBandLongFormSynthesisRequest* request,
    ScyllasBandSynthesisResult* out_result
);

ScyllasBandStatus scyllasband_runtime_plan_long_form(
    ScyllasBandRuntime* runtime,
    const ScyllasBandLongFormSynthesisRequest* request,
    ScyllasBandChunkPlanResult* out_result
);

ScyllasBandStatus scyllasband_runtime_synthesize_long_form_stream(
    ScyllasBandRuntime* runtime,
    const ScyllasBandLongFormSynthesisRequest* request,
    ScyllasBandStreamingCallback callback,
    void* user_data
);

ScyllasBandStatus scyllasband_runtime_estimate_latent_frames(
    ScyllasBandRuntime* runtime,
    const ScyllasBandSynthesisRequest* request,
    ScyllasBandDurationEstimateResult* out_result
);

ScyllasBandStatus scyllasband_plan_long_form_chunks(
    const ScyllasBandChunkPlanRequest* request,
    ScyllasBandChunkPlanResult* out_result
);

void scyllasband_chunk_plan_result_free(ScyllasBandChunkPlanResult* result);

void scyllasband_duration_estimate_result_free(ScyllasBandDurationEstimateResult* result);

void scyllasband_synthesis_result_free(ScyllasBandSynthesisResult* result);

const char* scyllasband_status_message(ScyllasBandStatus status);


ScyllasBandLiteRtSession* scyllasband_litert_session_create(
    const char* model_path,
    int32_t accelerator,
    int32_t max_threads
);

void scyllasband_litert_session_destroy(ScyllasBandLiteRtSession* session);

int scyllasband_litert_session_has_signature(
    const ScyllasBandLiteRtSession* session,
    const char* signature_name
);

int scyllasband_litert_session_run(
    const ScyllasBandLiteRtSession* session,
    const char* signature_name,
    const ScyllasBandTensorView* inputs,
    int32_t input_count,
    ScyllasBandOwnedTensor** out_tensors,
    int32_t* out_tensor_count
);

int scyllasband_litert_session_run_resized(
    const ScyllasBandLiteRtSession* session,
    const char* signature_name,
    const ScyllasBandTensorView* inputs,
    int32_t input_count,
    ScyllasBandOwnedTensor** out_tensors,
    int32_t* out_tensor_count
);

#ifdef __cplusplus
}
#endif
