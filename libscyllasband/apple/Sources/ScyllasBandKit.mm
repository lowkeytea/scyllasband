#import "include/ScyllasBandKit.h"

#include "scyllasband.h"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>

NSErrorDomain const ScyllasBandErrorDomain = @"org.scyllasband.ScyllasBandKit";

namespace {

constexpr int kFastSynthesisSteps = 5;
constexpr int kFastMaxChunkCharacters = 180;
constexpr int kMinimumChunkCharacters = 48;

NSError *make_error(SBScyllasBandErrorCode code, NSString *message) {
    return [NSError errorWithDomain:ScyllasBandErrorDomain
                               code:code
                           userInfo:@{NSLocalizedDescriptionKey: message}];
}

NSString *last_error_or(NSString *fallback) {
    const char *detail = scyllasband_last_error();
    if (detail == nullptr || detail[0] == '\0') {
        return fallback;
    }
    NSString *message = [NSString stringWithUTF8String:detail];
    return message ?: fallback;
}

BOOL fail(NSError **error, SBScyllasBandErrorCode code, NSString *message) {
    if (error != nullptr) {
        *error = make_error(code, message);
    }
    return NO;
}

NSArray<NSString *> *string_array(id values) {
    if (![values isKindOfClass:NSArray.class]) {
        return @[];
    }
    NSArray *array = (NSArray *)values;
    NSMutableArray<NSString *> *output = [NSMutableArray arrayWithCapacity:array.count];
    for (id value in array) {
        if ([value isKindOfClass:NSString.class]) {
            [output addObject:value];
        }
    }
    return output.copy;
}

NSString *chunk_text_from_metadata(const char *metadata) {
    if (metadata == nullptr || metadata[0] == '\0') {
        return nil;
    }
    NSData *data = [[NSData alloc] initWithBytes:metadata length:std::strlen(metadata)];
    NSDictionary *json = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
    id text = [json isKindOfClass:NSDictionary.class] ? json[@"text"] : nil;
    return [text isKindOfClass:NSString.class] ? text : nil;
}

void configure_long_form_request(
    ScyllasBandLongFormSynthesisRequest& long_form,
    NSString *text,
    SBScyllasBandSegmentSettings *settings,
    std::string& affect,
    uint64_t seed
) {
    ScyllasBandSynthesisRequest& request = long_form.request;
    request.text = text.UTF8String;
    request.voice_id = settings.voiceIdentifier.UTF8String;
    request.language = settings.language.UTF8String;
    request.guidance_null_reference = 1;
    request.emotion_embed_scale = 1.0f;
    request.min_sentence_pause_ms = 320.0f;
    request.min_clause_pause_ms = 160.0f;
    request.steps = kFastSynthesisSteps;
    request.sampler = SCYLLASBAND_SAMPLER_HEUN;
    request.seed = seed;
    request.has_seed = 1;
    request.speed = 1.0f;
    request.temperature = 1.0f;
    if (settings.emotion.length > 0) {
        affect = std::string(settings.emotion.UTF8String) + "=" +
            std::to_string(std::clamp(settings.emotionStrength, 0.0f, 1.0f));
        request.affect = affect.c_str();
    }
    request.affect_guidance_scale = settings.emotionCFG;
    request.has_affect_guidance_scale = 1;

    long_form.max_chunk_chars = kFastMaxChunkCharacters;
    long_form.min_chunk_chars = kMinimumChunkCharacters;
    long_form.pause_ms = 0;
    long_form.continuation_pause_ms = 0;
    long_form.use_prefix_latents = 1;
    long_form.disable_auto_split_overlong = 0;
    long_form.preflight_chunks = 0;
}

}  // namespace


@interface SBScyllasBandVoice ()
- (instancetype)initWithIdentifier:(NSString *)identifier
                        displayName:(NSString *)displayName
                          languages:(NSArray<NSString *> *)languages
                    defaultLanguage:(NSString *)defaultLanguage;
@end

@implementation SBScyllasBandVoice

- (instancetype)initWithIdentifier:(NSString *)identifier
                        displayName:(NSString *)displayName
                          languages:(NSArray<NSString *> *)languages
                    defaultLanguage:(NSString *)defaultLanguage {
    self = [super init];
    if (self) {
        _identifier = identifier.copy;
        _displayName = displayName.copy;
        _languages = languages.copy;
        _defaultLanguage = defaultLanguage.copy;
    }
    return self;
}

@end


@interface SBScyllasBandBundleInfo ()
- (instancetype)initWithModelName:(NSString *)modelName
                        sampleRate:(NSInteger)sampleRate
                            voices:(NSArray<SBScyllasBandVoice *> *)voices
                        affectAxes:(NSArray<NSString *> *)affectAxes;
@end

@implementation SBScyllasBandBundleInfo

- (instancetype)initWithModelName:(NSString *)modelName
                        sampleRate:(NSInteger)sampleRate
                            voices:(NSArray<SBScyllasBandVoice *> *)voices
                        affectAxes:(NSArray<NSString *> *)affectAxes {
    self = [super init];
    if (self) {
        _modelName = modelName.copy;
        _sampleRate = sampleRate;
        _voices = voices.copy;
        _affectAxes = affectAxes.copy;
    }
    return self;
}

@end


@implementation SBScyllasBandSegmentSettings

- (instancetype)initWithVoiceIdentifier:(NSString *)voiceIdentifier
                               language:(NSString *)language
                                emotion:(NSString *)emotion
                         emotionStrength:(float)emotionStrength
                             emotionCFG:(float)emotionCFG {
    self = [super init];
    if (self) {
        _voiceIdentifier = voiceIdentifier.copy;
        _language = language.copy;
        _emotion = emotion.copy;
        _emotionStrength = emotionStrength;
        _emotionCFG = emotionCFG;
    }
    return self;
}

- (id)copyWithZone:(NSZone *)zone {
    return [[SBScyllasBandSegmentSettings allocWithZone:zone]
        initWithVoiceIdentifier:self.voiceIdentifier
                       language:self.language
                        emotion:self.emotion
                 emotionStrength:self.emotionStrength
                     emotionCFG:self.emotionCFG];
}

@end


@interface SBScyllasBandAudioChunk ()
- (instancetype)initWithSamples:(const float *)samples
                     sampleCount:(NSInteger)sampleCount
                      sampleRate:(NSInteger)sampleRate
                      chunkIndex:(NSInteger)chunkIndex
                      chunkCount:(NSInteger)chunkCount;
@end

@implementation SBScyllasBandAudioChunk

- (instancetype)initWithSamples:(const float *)samples
                     sampleCount:(NSInteger)sampleCount
                      sampleRate:(NSInteger)sampleRate
                      chunkIndex:(NSInteger)chunkIndex
                      chunkCount:(NSInteger)chunkCount {
    self = [super init];
    if (self) {
        _pcmFloat32Data = [[NSData alloc] initWithBytes:samples
                                                length:sampleCount * sizeof(float)];
        _sampleCount = sampleCount;
        _sampleRate = sampleRate;
        _chunkIndex = chunkIndex;
        _chunkCount = chunkCount;
    }
    return self;
}

@end


struct SBStreamingContext {
    std::atomic_bool *cancelRequested;
    __unsafe_unretained SBScyllasBandChunkStartedBlock chunkStarted;
    __unsafe_unretained SBScyllasBandAudioChunkBlock audioChunk;
};

static int32_t forward_stream_event(const ScyllasBandStreamingEvent *event, void *userData) {
    auto *context = static_cast<SBStreamingContext *>(userData);
    if (event == nullptr || context == nullptr ||
        context->cancelRequested->load(std::memory_order_relaxed)) {
        return 1;
    }
    @autoreleasepool {
        if (event->type == SCYLLASBAND_STREAM_EVENT_CHUNK_STARTED) {
            if (context->chunkStarted == nil) {
                return 0;
            }
            const BOOL keepGoing = context->chunkStarted(
                event->chunk_index,
                event->chunk_count,
                chunk_text_from_metadata(event->metadata_json)
            );
            return keepGoing ? 0 : 1;
        }
        if (event->type != SCYLLASBAND_STREAM_EVENT_AUDIO_CHUNK) {
            return 0;
        }
        if (event->sample_count < 0 ||
            (event->sample_count > 0 && event->samples == nullptr)) {
            return 1;
        }
        SBScyllasBandAudioChunk *chunk = [[SBScyllasBandAudioChunk alloc]
            initWithSamples:event->samples
                sampleCount:event->sample_count
                 sampleRate:event->sample_rate
                 chunkIndex:event->chunk_index
                 chunkCount:event->chunk_count];
        return context->audioChunk(chunk) ? 0 : 1;
    }
}


@implementation SBScyllasBand {
    ScyllasBandRuntime *_runtime;
    std::atomic_bool _cancelRequested;
}

+ (NSInteger)defaultMobileTargetBucketCacheCapacity {
    return 1;
}

- (nullable instancetype)initWithBundleURL:(NSURL *)bundleURL
                               threadCount:(NSInteger)threadCount
                 targetBucketCacheCapacity:(NSInteger)targetBucketCacheCapacity
                                     error:(NSError **)error {
    self = [super init];
    if (!self) {
        return nil;
    }
    if (!bundleURL.isFileURL || bundleURL.path.length == 0) {
        fail(error, SBScyllasBandErrorInvalidArgument, @"The model bundle must be a file URL.");
        return nil;
    }
    if (threadCount < 1 || targetBucketCacheCapacity < 0) {
        fail(error, SBScyllasBandErrorInvalidArgument,
             @"Thread count must be positive and target-bucket cache capacity must be non-negative.");
        return nil;
    }

    NSURL *manifestURL = [bundleURL URLByAppendingPathComponent:@"manifest.json"];
    NSData *manifestData = [NSData dataWithContentsOfURL:manifestURL options:0 error:error];
    if (manifestData == nil) {
        return nil;
    }
    NSDictionary *manifest = [NSJSONSerialization JSONObjectWithData:manifestData options:0 error:error];
    if (![manifest isKindOfClass:NSDictionary.class]) {
        fail(error, SBScyllasBandErrorInvalidBundle, @"The Scylla's Band manifest is not a JSON object.");
        return nil;
    }
    NSDictionary *audio = [manifest[@"audio"] isKindOfClass:NSDictionary.class]
        ? manifest[@"audio"] : nil;
    NSDictionary *controls = [manifest[@"controls"] isKindOfClass:NSDictionary.class]
        ? manifest[@"controls"] : nil;
    NSDictionary *affectControls = [controls[@"affect"] isKindOfClass:NSDictionary.class]
        ? controls[@"affect"] : nil;
    NSString *modelName = manifest[@"model_name"];
    NSNumber *sampleRate = audio[@"sample_rate"];
    NSArray *voiceValues = manifest[@"voices"];
    NSArray *axisValues = affectControls[@"axes"];
    if (![modelName isKindOfClass:NSString.class] || ![sampleRate isKindOfClass:NSNumber.class] ||
        ![voiceValues isKindOfClass:NSArray.class] || ![axisValues isKindOfClass:NSArray.class]) {
        fail(error, SBScyllasBandErrorInvalidBundle,
             @"The Scylla's Band manifest is missing model, audio, voice, or affect metadata.");
        return nil;
    }

    NSMutableArray<SBScyllasBandVoice *> *voices = [NSMutableArray array];
    for (id value in voiceValues) {
        if (![value isKindOfClass:NSDictionary.class]) {
            continue;
        }
        NSString *identifier = value[@"id"];
        NSArray<NSString *> *languages = string_array(value[@"languages"] ?: @[]);
        if (![identifier isKindOfClass:NSString.class] || languages.count == 0) {
            continue;
        }
        NSString *displayName = [value[@"name"] isKindOfClass:NSString.class]
            ? value[@"name"] : identifier;
        NSString *defaultLanguage = [value[@"default_language"] isKindOfClass:NSString.class]
            ? value[@"default_language"] : languages.firstObject;
        [voices addObject:[[SBScyllasBandVoice alloc]
            initWithIdentifier:identifier
                   displayName:displayName.capitalizedString
                     languages:languages
               defaultLanguage:defaultLanguage]];
    }
    if (voices.count == 0) {
        fail(error, SBScyllasBandErrorInvalidBundle, @"The manifest does not declare any usable voices.");
        return nil;
    }

    _bundleURL = bundleURL.copy;
    _bundleInfo = [[SBScyllasBandBundleInfo alloc]
        initWithModelName:modelName
               sampleRate:sampleRate.integerValue
                   voices:voices
               affectAxes:string_array(axisValues)];
    _cancelRequested.store(false, std::memory_order_relaxed);

    ScyllasBandRuntimeOptions options{};
    options.bundle_dir = bundleURL.fileSystemRepresentation;
    options.backend = SCYLLASBAND_BACKEND_ONNX;
    options.validate_bundle = 1;
    options.litert_accelerator = SCYLLASBAND_LITERT_ACCELERATOR_CPU;
    options.litert_max_threads = static_cast<int32_t>(std::min<NSInteger>(threadCount, INT32_MAX));
    const ScyllasBandStatus createStatus = scyllasband_runtime_create(&options, &_runtime);
    if (createStatus != SCYLLASBAND_STATUS_OK || _runtime == nullptr) {
        fail(error, SBScyllasBandErrorRuntime,
             last_error_or(@"Unable to create the Scylla's Band ONNX runtime."));
        return nil;
    }
    const ScyllasBandStatus cacheStatus = scyllasband_runtime_set_target_bucket_cache_capacity(
        _runtime,
        static_cast<int32_t>(std::min<NSInteger>(targetBucketCacheCapacity, INT32_MAX))
    );
    if (cacheStatus != SCYLLASBAND_STATUS_OK) {
        NSString *message = last_error_or(@"Unable to configure the mobile session cache.");
        scyllasband_runtime_destroy(_runtime);
        _runtime = nullptr;
        fail(error, SBScyllasBandErrorRuntime, message);
        return nil;
    }
    return self;
}

- (void)dealloc {
    _cancelRequested.store(true, std::memory_order_relaxed);
    if (_runtime != nullptr) {
        scyllasband_runtime_destroy(_runtime);
        _runtime = nullptr;
    }
}

- (BOOL)isCancellationRequested {
    return _cancelRequested.load(std::memory_order_relaxed);
}

- (void)requestCancellation {
    _cancelRequested.store(true, std::memory_order_relaxed);
}

- (void)resetCancellation {
    _cancelRequested.store(false, std::memory_order_relaxed);
}

- (BOOL)warmUpWithError:(NSError **)error {
    if (_runtime == nullptr) {
        return fail(error, SBScyllasBandErrorRuntime, @"The Scylla's Band runtime is closed.");
    }
    SBScyllasBandVoice *voice = self.bundleInfo.voices.firstObject;
    SBScyllasBandSegmentSettings *settings = [[SBScyllasBandSegmentSettings alloc]
        initWithVoiceIdentifier:voice.identifier
                       language:voice.defaultLanguage
                        emotion:nil
                 emotionStrength:0
                     emotionCFG:1];
    ScyllasBandLongFormSynthesisRequest longForm{};
    std::string affect;
    configure_long_form_request(longForm, @"Warmup.", settings, affect, 0);
    longForm.request.steps = 1;
    longForm.request.sampler = SCYLLASBAND_SAMPLER_EULER;
    longForm.request.has_affect_guidance_scale = 0;
    longForm.request.boundary_before = "paragraph_start";
    longForm.request.boundary_after = "paragraph_end";

    ScyllasBandSynthesisResult result{};
    const ScyllasBandStatus status = scyllasband_runtime_synthesize(_runtime, &longForm.request, &result);
    scyllasband_synthesis_result_free(&result);
    if (status != SCYLLASBAND_STATUS_OK) {
        return fail(error, SBScyllasBandErrorRuntime,
                    last_error_or(@"Scylla's Band warmup failed."));
    }
    return YES;
}

- (BOOL)synthesizeText:(NSString *)text
              settings:(SBScyllasBandSegmentSettings *)settings
                  seed:(uint64_t)seed
          chunkStarted:(SBScyllasBandChunkStartedBlock)chunkStarted
             audioChunk:(SBScyllasBandAudioChunkBlock)audioChunk
                  error:(NSError **)error {
    if (_runtime == nullptr) {
        return fail(error, SBScyllasBandErrorRuntime, @"The Scylla's Band runtime is closed.");
    }
    if ([text stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet].length == 0) {
        return fail(error, SBScyllasBandErrorInvalidArgument, @"Synthesis text must not be empty.");
    }
    SBScyllasBandVoice *voice = nil;
    for (SBScyllasBandVoice *candidate in self.bundleInfo.voices) {
        if ([candidate.identifier isEqualToString:settings.voiceIdentifier]) {
            voice = candidate;
            break;
        }
    }
    if (voice == nil || ![voice.languages containsObject:settings.language]) {
        return fail(error, SBScyllasBandErrorInvalidArgument,
                    @"The requested voice or language is not declared by the bundle.");
    }
    if (settings.emotion != nil && ![self.bundleInfo.affectAxes containsObject:settings.emotion]) {
        return fail(error, SBScyllasBandErrorInvalidArgument,
                    @"The requested emotion is not declared by the bundle.");
    }
    if (!std::isfinite(settings.emotionStrength) || settings.emotionStrength < 0 ||
        settings.emotionStrength > 1 || !std::isfinite(settings.emotionCFG) ||
        settings.emotionCFG < 0) {
        return fail(error, SBScyllasBandErrorInvalidArgument,
                    @"Emotion strength must be within 0...1 and CFG must be non-negative.");
    }

    ScyllasBandLongFormSynthesisRequest longForm{};
    std::string affect;
    configure_long_form_request(longForm, text, settings, affect, seed);
    SBScyllasBandChunkStartedBlock startedCopy = [chunkStarted copy];
    SBScyllasBandAudioChunkBlock audioCopy = [audioChunk copy];
    SBStreamingContext context{&_cancelRequested, startedCopy, audioCopy};
    const ScyllasBandStatus status = scyllasband_runtime_synthesize_long_form_stream(
        _runtime,
        &longForm,
        forward_stream_event,
        &context
    );
    if (status != SCYLLASBAND_STATUS_OK) {
        if (_cancelRequested.load(std::memory_order_relaxed)) {
            return fail(error, SBScyllasBandErrorCancelled, @"Synthesis was cancelled.");
        }
        return fail(error, SBScyllasBandErrorRuntime,
                    last_error_or(@"Scylla's Band streaming synthesis failed."));
    }
    return YES;
}

@end
