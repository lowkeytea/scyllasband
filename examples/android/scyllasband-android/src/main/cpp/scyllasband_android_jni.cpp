#include <jni.h>

#include "scyllasband.h"

#include <cstdint>
#include <string>

namespace {

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

    UtfChars(const UtfChars&) = delete;
    UtfChars& operator=(const UtfChars&) = delete;

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

struct JavaStreamContext {
    JNIEnv* environment = nullptr;
    jobject listener = nullptr;
    jmethodID chunk_started = nullptr;
    jmethodID audio_chunk = nullptr;
};

// Returns nonzero to stop synthesis. A listener that returns false, or throws, stops it.
int32_t forward_event(const ScyllasBandEvent* event, void* user_data) {
    auto* context = static_cast<JavaStreamContext*>(user_data);
    if (event == nullptr || context == nullptr || context->environment == nullptr) {
        return 1;
    }
    JNIEnv* environment = context->environment;
    if (event->type == SCYLLASBAND_EVENT_CHUNK_STARTED) {
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
    if (event->type != SCYLLASBAND_EVENT_AUDIO) {
        return 0;
    }
    if (event->sample_count < 0 || (event->sample_count > 0 && event->samples == nullptr)) {
        return 1;
    }
    jfloatArray samples = environment->NewFloatArray(static_cast<jsize>(event->sample_count));
    if (samples == nullptr) {
        return 1;
    }
    if (event->sample_count > 0) {
        environment->SetFloatArrayRegion(
            samples, 0, static_cast<jsize>(event->sample_count), event->samples
        );
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

struct JavaSingContext {
    JNIEnv* environment = nullptr;
    jobject listener = nullptr;
    jmethodID phrase_started = nullptr;
    jmethodID phrase_audio = nullptr;
};

// Singing events: a phrase about to be sung, and its audio with its metadata (offset_samples places it on the song's
// timeline). Returns nonzero to stop; a listener that returns false, or throws, stops singing.
int32_t forward_sing_event(const ScyllasBandEvent* event, void* user_data) {
    auto* context = static_cast<JavaSingContext*>(user_data);
    if (event == nullptr || context == nullptr || context->environment == nullptr) {
        return 1;
    }
    if (event->type != SCYLLASBAND_EVENT_CHUNK_STARTED && event->type != SCYLLASBAND_EVENT_AUDIO) {
        return 0;
    }
    JNIEnv* environment = context->environment;
    jstring metadata = event->metadata_json == nullptr ? nullptr : environment->NewStringUTF(event->metadata_json);
    if (event->metadata_json != nullptr && metadata == nullptr) {
        return 1;
    }
    jboolean keep_going = JNI_FALSE;
    if (event->type == SCYLLASBAND_EVENT_CHUNK_STARTED) {
        keep_going = environment->CallBooleanMethod(
            context->listener, context->phrase_started, static_cast<jint>(event->chunk_index),
            static_cast<jint>(event->chunk_count), metadata
        );
    } else {
        if (event->sample_count < 0 || (event->sample_count > 0 && event->samples == nullptr)) {
            if (metadata != nullptr) environment->DeleteLocalRef(metadata);
            return 1;
        }
        jfloatArray samples = environment->NewFloatArray(static_cast<jsize>(event->sample_count));
        if (samples == nullptr) {
            if (metadata != nullptr) environment->DeleteLocalRef(metadata);
            return 1;
        }
        if (event->sample_count > 0) {
            environment->SetFloatArrayRegion(samples, 0, static_cast<jsize>(event->sample_count), event->samples);
        }
        if (!environment->ExceptionCheck()) {
            keep_going = environment->CallBooleanMethod(
                context->listener, context->phrase_audio, samples, static_cast<jint>(event->chunk_index),
                static_cast<jint>(event->chunk_count), metadata
            );
        }
        environment->DeleteLocalRef(samples);
    }
    if (metadata != nullptr) {
        environment->DeleteLocalRef(metadata);
    }
    return environment->ExceptionCheck() || keep_going != JNI_TRUE ? 1 : 0;
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_org_scyllasband_android_ScyllasBandNative_create(
    JNIEnv* environment,
    jobject,
    jstring bundle_path,
    jint thread_count,
    jint accelerator
) {
    UtfChars path(environment, bundle_path);
    if (path.get() == nullptr) {
        throw_java(environment, "Scylla's Band bundle path is required");
        return 0;
    }
    if (thread_count < 0 || accelerator < SCYLLASBAND_ACCELERATOR_CPU ||
        accelerator > SCYLLASBAND_ACCELERATOR_AUTO) {
        throw_java(environment, "Invalid Scylla's Band runtime options");
        return 0;
    }
    ScyllasBandRuntimeOptions options{};
    options.bundle_dir = path.get();
    options.threads = thread_count;
    options.accelerator = static_cast<ScyllasBandAccelerator>(accelerator);
    ScyllasBandRuntime* runtime = nullptr;
    const ScyllasBandStatus status = scyllasband_runtime_create(&options, &runtime);
    if (status != SCYLLASBAND_OK || runtime == nullptr) {
        throw_java(environment, last_error_or("Unable to create the Scylla's Band runtime"));
        return 0;
    }
    return static_cast<jlong>(reinterpret_cast<intptr_t>(runtime));
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_scyllasband_android_ScyllasBandNative_voicesJson(
    JNIEnv* environment,
    jobject,
    jlong handle
) {
    ScyllasBandRuntime* runtime = runtime_from_handle(handle);
    if (runtime == nullptr) {
        throw_java(environment, "Scylla's Band runtime is closed");
        return nullptr;
    }
    const char* json = scyllasband_voices_json(runtime);
    if (json == nullptr) {
        throw_java(environment, last_error_or("Scylla's Band returned no voice list"));
        return nullptr;
    }
    return environment->NewStringUTF(json);
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_scyllasband_android_ScyllasBandNative_backend(JNIEnv* environment, jobject) {
    const char* name = scyllasband_backend();
    return environment->NewStringUTF(name == nullptr ? "" : name);
}

extern "C" JNIEXPORT void JNICALL
Java_org_scyllasband_android_ScyllasBandNative_warmup(
    JNIEnv* environment,
    jobject,
    jlong handle,
    jstring voice_id
) {
    ScyllasBandRuntime* runtime = runtime_from_handle(handle);
    if (runtime == nullptr) {
        throw_java(environment, "Scylla's Band runtime is closed");
        return;
    }
    UtfChars voice_chars(environment, voice_id);
    if (scyllasband_warmup(runtime, voice_chars.get()) != SCYLLASBAND_OK) {
        throw_java(environment, last_error_or("Scylla's Band warmup failed"));
    }
}

// Returns true when the whole request was spoken, false when the listener or cancel() stopped it.
extern "C" JNIEXPORT jboolean JNICALL
Java_org_scyllasband_android_ScyllasBandNative_synthesizeStreaming(
    JNIEnv* environment,
    jobject,
    jlong handle,
    jstring text,
    jstring voice_id,
    jstring language,
    jstring delivery,
    jint steps,
    jlong seed,
    jboolean has_seed,
    jobject listener
) {
    ScyllasBandRuntime* runtime = runtime_from_handle(handle);
    if (runtime == nullptr) {
        throw_java(environment, "Scylla's Band runtime is closed");
        return JNI_FALSE;
    }
    if (listener == nullptr) {
        throw_java(environment, "Scylla's Band streaming listener is required");
        return JNI_FALSE;
    }
    UtfChars text_chars(environment, text);
    UtfChars voice_chars(environment, voice_id);
    UtfChars language_chars(environment, language);
    UtfChars delivery_chars(environment, delivery);
    if (text_chars.get() == nullptr || voice_chars.get() == nullptr) {
        throw_java(environment, "Text and voice are required");
        return JNI_FALSE;
    }

    jclass listener_class = environment->GetObjectClass(listener);
    if (listener_class == nullptr) {
        return JNI_FALSE;
    }
    JavaStreamContext context;
    context.environment = environment;
    context.listener = listener;
    context.chunk_started = environment->GetMethodID(
        listener_class, "onChunkStarted", "(IILjava/lang/String;)Z"
    );
    context.audio_chunk = environment->GetMethodID(listener_class, "onAudioChunk", "([FII)Z");
    environment->DeleteLocalRef(listener_class);
    if (context.chunk_started == nullptr || context.audio_chunk == nullptr) {
        if (!environment->ExceptionCheck()) {
            throw_java(environment, "Scylla's Band streaming listener has an incompatible shape");
        }
        return JNI_FALSE;
    }

    ScyllasBandRequest request;
    scyllasband_request_init(&request);
    request.text = text_chars.get();
    request.voice_id = voice_chars.get();
    request.language = language_chars.get();
    request.delivery = delivery_chars.get();
    request.steps = steps;
    request.seed = static_cast<uint64_t>(seed);
    request.has_seed = has_seed == JNI_TRUE ? 1 : 0;

    const ScyllasBandStatus status =
        scyllasband_synthesize_stream(runtime, &request, forward_event, &context);
    if (status == SCYLLASBAND_OK) {
        return JNI_TRUE;
    }
    if (status != SCYLLASBAND_ERROR_CANCELLED && !environment->ExceptionCheck()) {
        throw_java(environment, last_error_or("Scylla's Band streaming synthesis failed"));
    }
    return JNI_FALSE;
}

namespace {

// decoder: null or "" for the bundle's singing decoder, "bigvgan" or "vocos". guidance <= 0: the bundle's setting.
ScyllasBandSingRequest sing_request(const UtfChars& score, const UtfChars& voice, const UtfChars& language, const UtfChars& decoder,
                                    jint steps, jfloat guidance, jint transpose, jboolean use_register, jlong seed, jboolean has_seed) {
    ScyllasBandSingRequest request;
    scyllasband_sing_request_init(&request);
    request.score_json = score.get();
    request.voice_id = voice.get();
    request.language = language.get();
    request.decoder = decoder.get();
    request.steps = steps;
    request.guidance = guidance;
    request.transpose = transpose;
    request.use_register = use_register == JNI_TRUE ? 1 : 0;
    request.seed = static_cast<uint64_t>(seed);
    request.has_seed = has_seed == JNI_TRUE ? 1 : 0;
    return request;
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_org_scyllasband_android_ScyllasBandNative_singPlanJson(
    JNIEnv* environment,
    jobject,
    jlong handle,
    jstring score_json,
    jstring voice_id,
    jstring language,
    jint transpose,
    jboolean use_register
) {
    ScyllasBandRuntime* runtime = runtime_from_handle(handle);
    if (runtime == nullptr) {
        throw_java(environment, "Scylla's Band runtime is closed");
        return nullptr;
    }
    UtfChars score(environment, score_json), voice(environment, voice_id), language_chars(environment, language), no_decoder(environment, nullptr);
    if (score.get() == nullptr || voice.get() == nullptr) {
        throw_java(environment, "A score and a voice are required");
        return nullptr;
    }
    const ScyllasBandSingRequest request =
        sing_request(score, voice, language_chars, no_decoder, 0, 0.0f, transpose, use_register, 0, JNI_FALSE);
    char* plan = nullptr;
    if (scyllasband_sing_plan_json(runtime, &request, &plan) != SCYLLASBAND_OK) {
        throw_java(environment, last_error_or("Scylla's Band could not plan the score"));
        return nullptr;
    }
    jstring out = environment->NewStringUTF(plan);
    scyllasband_string_free(plan);
    return out;
}

// Returns true when the whole score was sung, false when the listener or cancel() stopped it.
extern "C" JNIEXPORT jboolean JNICALL
Java_org_scyllasband_android_ScyllasBandNative_singStreaming(
    JNIEnv* environment,
    jobject,
    jlong handle,
    jstring score_json,
    jstring voice_id,
    jstring language,
    jstring decoder,
    jint steps,
    jfloat guidance,
    jint transpose,
    jboolean use_register,
    jlong seed,
    jboolean has_seed,
    jobject listener
) {
    ScyllasBandRuntime* runtime = runtime_from_handle(handle);
    if (runtime == nullptr) {
        throw_java(environment, "Scylla's Band runtime is closed");
        return JNI_FALSE;
    }
    if (listener == nullptr) {
        throw_java(environment, "Scylla's Band singing listener is required");
        return JNI_FALSE;
    }
    UtfChars score(environment, score_json), voice(environment, voice_id), language_chars(environment, language), decoder_chars(environment, decoder);
    if (score.get() == nullptr || voice.get() == nullptr) {
        throw_java(environment, "A score and a voice are required");
        return JNI_FALSE;
    }
    jclass listener_class = environment->GetObjectClass(listener);
    if (listener_class == nullptr) {
        return JNI_FALSE;
    }
    JavaSingContext context;
    context.environment = environment;
    context.listener = listener;
    context.phrase_started = environment->GetMethodID(listener_class, "onPhraseStarted", "(IILjava/lang/String;)Z");
    context.phrase_audio = environment->GetMethodID(listener_class, "onPhraseAudio", "([FIILjava/lang/String;)Z");
    environment->DeleteLocalRef(listener_class);
    if (context.phrase_started == nullptr || context.phrase_audio == nullptr) {
        if (!environment->ExceptionCheck()) {
            throw_java(environment, "Scylla's Band singing listener has an incompatible shape");
        }
        return JNI_FALSE;
    }
    const ScyllasBandSingRequest request =
        sing_request(score, voice, language_chars, decoder_chars, steps, guidance, transpose, use_register, seed, has_seed);
    const ScyllasBandStatus status = scyllasband_sing_stream(runtime, &request, forward_sing_event, &context);
    if (status == SCYLLASBAND_OK) {
        return JNI_TRUE;
    }
    if (status != SCYLLASBAND_ERROR_CANCELLED && !environment->ExceptionCheck()) {
        throw_java(environment, last_error_or("Scylla's Band singing failed"));
    }
    return JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_org_scyllasband_android_ScyllasBandNative_cancel(JNIEnv*, jobject, jlong handle) {
    ScyllasBandRuntime* runtime = runtime_from_handle(handle);
    if (runtime != nullptr) {
        scyllasband_cancel(runtime);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_org_scyllasband_android_ScyllasBandNative_destroy(JNIEnv*, jobject, jlong handle) {
    ScyllasBandRuntime* runtime = runtime_from_handle(handle);
    if (runtime != nullptr) {
        scyllasband_runtime_destroy(runtime);
    }
}
