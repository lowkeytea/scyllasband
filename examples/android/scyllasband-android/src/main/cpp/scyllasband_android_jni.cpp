#include <jni.h>

#include "scyllasband.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

constexpr int kFastSynthesisSteps = 2;
constexpr int kFastMaxChunkChars = 180;
constexpr int kMinChunkChars = 48;

class UtfChars {
public:
    UtfChars(JNIEnv* environment, jstring value)
        : environment_(environment), value_(value), chars_(nullptr) {
        if (value_ != nullptr) {
            chars_ = environment_->GetStringUTFChars(value_, nullptr);
        }
    }

    ~UtfChars() {
        if (chars_ != nullptr) {
            environment_->ReleaseStringUTFChars(value_, chars_);
        }
    }

    const char* get() const { return chars_; }

private:
    JNIEnv* environment_;
    jstring value_;
    const char* chars_;
};

void throw_java(JNIEnv* environment, const std::string& message) {
    jclass error_class = environment->FindClass("java/lang/IllegalStateException");
    if (error_class != nullptr) {
        environment->ThrowNew(error_class, message.c_str());
    }
}

std::string last_error_or(const char* fallback) {
    const char* detail = scyllasband_last_error();
    if (detail == nullptr || detail[0] == '\0') {
        return fallback;
    }
    return detail;
}

ScyllasBandRuntime* runtime_from_handle(jlong handle) {
    return reinterpret_cast<ScyllasBandRuntime*>(static_cast<intptr_t>(handle));
}

void configure_long_form_request(
    ScyllasBandLongFormSynthesisRequest& long_form,
    const char* text,
    const char* voice_id,
    const char* language,
    const char* affect,
    jfloat emotion_cfg,
    jlong seed
) {
    ScyllasBandSynthesisRequest& request = long_form.request;
    request.text = text;
    request.voice_id = voice_id;
    request.language = language;
    request.guidance_null_reference = 1;
    request.emotion_embed_scale = 1.0f;
    request.min_sentence_pause_ms = 320.0f;
    request.min_clause_pause_ms = 160.0f;
    // Match the core Python --faster sampling profile. Two Heun steps retain
    // the higher quality sampler while halving vector-estimator evaluations.
    request.steps = kFastSynthesisSteps;
    request.sampler = SCYLLASBAND_SAMPLER_HEUN;
    request.seed = static_cast<uint64_t>(seed);
    request.has_seed = 1;
    request.speed = 1.0f;
    request.temperature = 1.0f;
    request.affect = affect;
    request.affect_guidance_scale = emotion_cfg;
    request.has_affect_guidance_scale = 1;

    long_form.max_chunk_chars = kFastMaxChunkChars;
    long_form.min_chunk_chars = kMinChunkChars;
    long_form.pause_ms = 0;
    long_form.continuation_pause_ms = 0;
    long_form.use_prefix_latents = 1;
    long_form.disable_auto_split_overlong = 0;
    // Android favors immediate rendering. The native retry path still splits
    // a chunk if an unusual input exceeds a model budget.
    long_form.preflight_chunks = 0;
}

struct JavaStreamContext {
    JNIEnv* environment = nullptr;
    jobject listener = nullptr;
    jmethodID chunk_started = nullptr;
    jmethodID audio_chunk = nullptr;
};

int32_t forward_stream_event(const ScyllasBandStreamingEvent* event, void* user_data) {
    auto* context = static_cast<JavaStreamContext*>(user_data);
    if (event == nullptr || context == nullptr || context->environment == nullptr) {
        return 1;
    }
    JNIEnv* environment = context->environment;
    if (event->type == SCYLLASBAND_STREAM_EVENT_CHUNK_STARTED) {
        jstring metadata = event->metadata_json == nullptr
            ? nullptr
            : environment->NewStringUTF(event->metadata_json);
        if (event->metadata_json != nullptr && metadata == nullptr) {
            return 1;
        }
        const jboolean keep_going = environment->CallBooleanMethod(
            context->listener,
            context->chunk_started,
            static_cast<jint>(event->chunk_index),
            static_cast<jint>(event->chunk_count),
            metadata
        );
        if (metadata != nullptr) {
            environment->DeleteLocalRef(metadata);
        }
        return environment->ExceptionCheck() || keep_going != JNI_TRUE ? 1 : 0;
    }
    if (event->type != SCYLLASBAND_STREAM_EVENT_AUDIO_CHUNK) {
        return 0;
    }
    if (event->sample_count < 0 ||
        (event->sample_count > 0 && event->samples == nullptr)) {
        return 1;
    }
    jfloatArray samples = environment->NewFloatArray(event->sample_count);
    if (samples == nullptr) {
        return 1;
    }
    if (event->sample_count > 0) {
        environment->SetFloatArrayRegion(samples, 0, event->sample_count, event->samples);
    }
    if (environment->ExceptionCheck()) {
        environment->DeleteLocalRef(samples);
        return 1;
    }
    const jboolean keep_going = environment->CallBooleanMethod(
        context->listener,
        context->audio_chunk,
        samples,
        static_cast<jint>(event->chunk_index),
        static_cast<jint>(event->chunk_count)
    );
    environment->DeleteLocalRef(samples);
    return environment->ExceptionCheck() || keep_going != JNI_TRUE ? 1 : 0;
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_org_scyllasband_android_ScyllasBandNative_create(
    JNIEnv* environment,
    jobject,
    jstring bundle_path,
    jint thread_count,
    jint target_bucket_cache_capacity
) {
    UtfChars path(environment, bundle_path);
    if (path.get() == nullptr) {
        throw_java(environment, "Scylla's Band bundle path is required");
        return 0;
    }
    if (target_bucket_cache_capacity < 0) {
        throw_java(environment, "Target-bucket cache capacity must be non-negative");
        return 0;
    }
    ScyllasBandRuntimeOptions options{};
    options.bundle_dir = path.get();
    options.backend = SCYLLASBAND_BACKEND_ONNX;
    options.validate_bundle = 1;
    options.litert_accelerator = SCYLLASBAND_LITERT_ACCELERATOR_CPU;
    options.litert_max_threads = std::max(1, static_cast<int>(thread_count));
    ScyllasBandRuntime* runtime = nullptr;
    const ScyllasBandStatus status = scyllasband_runtime_create(&options, &runtime);
    if (status != SCYLLASBAND_STATUS_OK || runtime == nullptr) {
        throw_java(environment, last_error_or("Unable to create the Scylla's Band ONNX runtime"));
        return 0;
    }
    const ScyllasBandStatus cache_status =
        scyllasband_runtime_set_target_bucket_cache_capacity(
            runtime,
            static_cast<int>(target_bucket_cache_capacity)
        );
    if (cache_status != SCYLLASBAND_STATUS_OK) {
        const std::string message = last_error_or(
            "Unable to configure the Scylla's Band mobile session cache"
        );
        scyllasband_runtime_destroy(runtime);
        throw_java(environment, message);
        return 0;
    }
    return static_cast<jlong>(reinterpret_cast<intptr_t>(runtime));
}

extern "C" JNIEXPORT void JNICALL
Java_org_scyllasband_android_ScyllasBandNative_warmup(
    JNIEnv* environment,
    jobject,
    jlong handle,
    jstring voice_id,
    jstring language
) {
    ScyllasBandRuntime* runtime = runtime_from_handle(handle);
    if (runtime == nullptr) {
        throw_java(environment, "Scylla's Band runtime is closed");
        return;
    }
    UtfChars voice_chars(environment, voice_id);
    UtfChars language_chars(environment, language);
    if (voice_chars.get() == nullptr || language_chars.get() == nullptr) {
        throw_java(environment, "Voice and language are required for warmup");
        return;
    }

    ScyllasBandLongFormSynthesisRequest long_form{};
    configure_long_form_request(
        long_form,
        "Warmup.",
        voice_chars.get(),
        language_chars.get(),
        nullptr,
        1.0f,
        0
    );
    // A discarded one-step Euler render initializes the frontend, vector, and
    // vocoder sessions without doing more startup work than necessary.
    long_form.request.steps = 1;
    long_form.request.sampler = SCYLLASBAND_SAMPLER_EULER;
    long_form.request.has_affect_guidance_scale = 0;
    long_form.request.boundary_before = "paragraph_start";
    long_form.request.boundary_after = "paragraph_end";

    ScyllasBandSynthesisResult result{};
    const ScyllasBandStatus status = scyllasband_runtime_synthesize(
        runtime,
        &long_form.request,
        &result
    );
    scyllasband_synthesis_result_free(&result);
    if (status != SCYLLASBAND_STATUS_OK) {
        throw_java(environment, last_error_or("Scylla's Band warmup failed"));
    }
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_org_scyllasband_android_ScyllasBandNative_synthesizeSegment(
    JNIEnv* environment,
    jobject,
    jlong handle,
    jstring text,
    jstring voice_id,
    jstring language,
    jstring affect,
    jfloat emotion_cfg,
    jlong seed
) {
    ScyllasBandRuntime* runtime = runtime_from_handle(handle);
    if (runtime == nullptr) {
        throw_java(environment, "Scylla's Band runtime is closed");
        return nullptr;
    }
    UtfChars text_chars(environment, text);
    UtfChars voice_chars(environment, voice_id);
    UtfChars language_chars(environment, language);
    UtfChars affect_chars(environment, affect);
    if (text_chars.get() == nullptr || voice_chars.get() == nullptr || language_chars.get() == nullptr) {
        throw_java(environment, "Text, voice, and language are required");
        return nullptr;
    }

    ScyllasBandLongFormSynthesisRequest long_form{};
    configure_long_form_request(
        long_form,
        text_chars.get(),
        voice_chars.get(),
        language_chars.get(),
        affect_chars.get(),
        emotion_cfg,
        seed
    );

    ScyllasBandSynthesisResult result{};
    const ScyllasBandStatus status = scyllasband_runtime_synthesize_long_form(
        runtime, &long_form, &result
    );
    if (status != SCYLLASBAND_STATUS_OK) {
        scyllasband_synthesis_result_free(&result);
        throw_java(environment, last_error_or("Scylla's Band synthesis failed"));
        return nullptr;
    }
    if (result.sample_count < 0 ||
        (result.sample_count > 0 && result.samples == nullptr)) {
        scyllasband_synthesis_result_free(&result);
        throw_java(environment, "Scylla's Band returned an invalid audio buffer");
        return nullptr;
    }

    jfloatArray output = environment->NewFloatArray(result.sample_count);
    if (output != nullptr && result.sample_count > 0) {
        environment->SetFloatArrayRegion(output, 0, result.sample_count, result.samples);
    }
    scyllasband_synthesis_result_free(&result);
    return output;
}

extern "C" JNIEXPORT void JNICALL
Java_org_scyllasband_android_ScyllasBandNative_synthesizeSegmentStreaming(
    JNIEnv* environment,
    jobject,
    jlong handle,
    jstring text,
    jstring voice_id,
    jstring language,
    jstring affect,
    jfloat emotion_cfg,
    jlong seed,
    jobject listener
) {
    ScyllasBandRuntime* runtime = runtime_from_handle(handle);
    if (runtime == nullptr) {
        throw_java(environment, "Scylla's Band runtime is closed");
        return;
    }
    if (listener == nullptr) {
        throw_java(environment, "Scylla's Band streaming listener is required");
        return;
    }
    UtfChars text_chars(environment, text);
    UtfChars voice_chars(environment, voice_id);
    UtfChars language_chars(environment, language);
    UtfChars affect_chars(environment, affect);
    if (text_chars.get() == nullptr || voice_chars.get() == nullptr || language_chars.get() == nullptr) {
        throw_java(environment, "Text, voice, and language are required");
        return;
    }

    jclass listener_class = environment->GetObjectClass(listener);
    if (listener_class == nullptr) {
        return;
    }
    JavaStreamContext context;
    context.environment = environment;
    context.listener = listener;
    context.chunk_started = environment->GetMethodID(
        listener_class,
        "onChunkStarted",
        "(IILjava/lang/String;)Z"
    );
    context.audio_chunk = environment->GetMethodID(listener_class, "onAudioChunk", "([FII)Z");
    environment->DeleteLocalRef(listener_class);
    if (context.chunk_started == nullptr || context.audio_chunk == nullptr) {
        if (!environment->ExceptionCheck()) {
            throw_java(environment, "Scylla's Band streaming listener has an incompatible shape");
        }
        return;
    }

    ScyllasBandLongFormSynthesisRequest long_form{};
    configure_long_form_request(
        long_form,
        text_chars.get(),
        voice_chars.get(),
        language_chars.get(),
        affect_chars.get(),
        emotion_cfg,
        seed
    );
    const ScyllasBandStatus status = scyllasband_runtime_synthesize_long_form_stream(
        runtime,
        &long_form,
        forward_stream_event,
        &context
    );
    if (status != SCYLLASBAND_STATUS_OK && !environment->ExceptionCheck()) {
        throw_java(environment, last_error_or("Scylla's Band streaming synthesis failed"));
    }
}

extern "C" JNIEXPORT void JNICALL
Java_org_scyllasband_android_ScyllasBandNative_destroy(
    JNIEnv*,
    jobject,
    jlong handle
) {
    ScyllasBandRuntime* runtime = runtime_from_handle(handle);
    if (runtime != nullptr) {
        scyllasband_runtime_destroy(runtime);
    }
}
