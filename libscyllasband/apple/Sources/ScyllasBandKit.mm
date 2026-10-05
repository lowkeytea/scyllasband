#import "include/ScyllasBandKit.h"

#include "scyllasband.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

NSErrorDomain const ScyllasBandErrorDomain = @"org.scyllasband.ScyllasBandKit";

namespace {

BOOL fail(NSError **error, SBScyllasBandErrorCode code, NSString *message) {
    if (error != nullptr) {
        *error = [NSError errorWithDomain:ScyllasBandErrorDomain code:code userInfo:@{NSLocalizedDescriptionKey: message}];
    }
    return NO;
}

NSString *last_error_or(NSString *fallback) {
    const char *detail = scyllasband_last_error();
    NSString *message = detail && detail[0] ? [NSString stringWithUTF8String:detail] : nil;
    return message.length ? message : fallback;
}

SBScyllasBandErrorCode error_code(ScyllasBandStatus status) {
    switch (status) {
        case SCYLLASBAND_ERROR_INVALID_ARGUMENT: return SBScyllasBandErrorInvalidArgument;
        case SCYLLASBAND_ERROR_CANCELLED: return SBScyllasBandErrorCancelled;
        default: return SBScyllasBandErrorRuntime;
    }
}

id json_object(const char *text) {
    if (text == nullptr || text[0] == '\0') return nil;
    NSData *data = [NSData dataWithBytes:text length:std::strlen(text)];
    return [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
}

NSString *format_value(float value) {
    // Shortest form: 2 -> "2", 2.5 -> "2.5".
    return [NSString stringWithFormat:@"%g", std::round(value * 1000.0f) / 1000.0f];
}

}  // namespace

@interface SBScyllasBandVoice ()
- (instancetype)initWithIdentifier:(NSString *)identifier languages:(NSArray<NSString *> *)languages defaultLanguage:(NSString *)defaultLanguage;
@end

@implementation SBScyllasBandVoice
- (instancetype)initWithIdentifier:(NSString *)identifier languages:(NSArray<NSString *> *)languages defaultLanguage:(NSString *)defaultLanguage {
    if ((self = [super init])) {
        _identifier = identifier.copy;
        _languages = languages.copy;
        _defaultLanguage = defaultLanguage.copy;
    }
    return self;
}
@end

@interface SBScyllasBandBundleInfo ()
- (nullable instancetype)initWithVoicesJSON:(NSDictionary *)json;
@end

@implementation SBScyllasBandBundleInfo
- (nullable instancetype)initWithVoicesJSON:(NSDictionary *)json {
    if (![json isKindOfClass:NSDictionary.class]) return nil;
    if ((self = [super init])) {
        _backend = [json[@"backend"] isKindOfClass:NSString.class] ? json[@"backend"] : @"";
        _releaseIdentifier = [json[@"release_id"] isKindOfClass:NSString.class] ? json[@"release_id"] : @"";
        _sampleRate = [json[@"sample_rate"] integerValue];
        NSMutableArray<SBScyllasBandVoice *> *voices = [NSMutableArray array];
        for (NSDictionary *voice in [json[@"voices"] isKindOfClass:NSArray.class] ? json[@"voices"] : @[]) {
            if (![voice isKindOfClass:NSDictionary.class] || ![voice[@"id"] isKindOfClass:NSString.class]) continue;
            NSArray *languages = [voice[@"languages"] isKindOfClass:NSArray.class] ? voice[@"languages"] : @[];
            NSString *fallback = languages.firstObject ?: @"en_us";
            [voices addObject:[[SBScyllasBandVoice alloc] initWithIdentifier:voice[@"id"]
                                                                   languages:languages
                                                             defaultLanguage:[voice[@"default_language"] isKindOfClass:NSString.class]
                                                                                 ? voice[@"default_language"] : fallback]];
        }
        _voices = voices.copy;
        _defaultVoice = [json[@"default_voice"] isKindOfClass:NSString.class] ? json[@"default_voice"] : (voices.firstObject.identifier ?: @"");
        if (_sampleRate <= 0 || _voices.count == 0) return nil;
    }
    return self;
}

- (nullable SBScyllasBandVoice *)voiceWithIdentifier:(NSString *)identifier {
    for (SBScyllasBandVoice *voice in self.voices) {
        if ([voice.identifier isEqualToString:identifier]) return voice;
    }
    return nil;
}
@end

@implementation SBScyllasBandDelivery

+ (SBScyllasBandDelivery *)neutral {
    return [[SBScyllasBandDelivery alloc] initWithEnergy:2 tension:2 valence:2 assertiveness:2 whisper:NO];
}

- (instancetype)initWithEnergy:(float)energy tension:(float)tension valence:(float)valence assertiveness:(float)assertiveness whisper:(BOOL)whisper {
    if ((self = [super init])) {
        auto clamp = [](float value) { return std::isfinite(value) ? std::clamp(value, 0.0f, 4.0f) : 2.0f; };
        _energy = clamp(energy);
        _tension = clamp(tension);
        _valence = clamp(valence);
        _assertiveness = clamp(assertiveness);
        _whisper = whisper;
    }
    return self;
}

- (NSString *)specification {
    return [NSString stringWithFormat:@"energy=%@,tension=%@,valence=%@,assertiveness=%@,whisper=%@", format_value(self.energy),
                                      format_value(self.tension), format_value(self.valence), format_value(self.assertiveness),
                                      self.whisper ? @"on" : @"off"];
}

- (id)copyWithZone:(NSZone *)zone {
    return self;   // immutable
}

- (BOOL)isEqual:(id)other {
    if (![other isKindOfClass:SBScyllasBandDelivery.class]) return NO;
    return [self.specification isEqualToString:[other specification]];
}

- (NSUInteger)hash {
    return self.specification.hash;
}

@end

@implementation SBScyllasBandRequest

- (instancetype)initWithText:(NSString *)text voiceIdentifier:(NSString *)voiceIdentifier {
    if ((self = [super init])) {
        _text = text.copy;
        _voiceIdentifier = voiceIdentifier.copy;
        _delivery = SBScyllasBandDelivery.neutral;
        _speed = 1.0f;
        _normalizeText = YES;
    }
    return self;
}

- (id)copyWithZone:(NSZone *)zone {
    SBScyllasBandRequest *copy = [[SBScyllasBandRequest alloc] initWithText:self.text voiceIdentifier:self.voiceIdentifier];
    copy.language = self.language;
    copy.delivery = self.delivery;
    copy.seed = self.seed;
    copy.steps = self.steps;
    copy.speed = self.speed;
    copy.normalizeText = self.normalizeText;
    return copy;
}

@end

@interface SBScyllasBandAudioChunk ()
- (instancetype)initWithSamples:(const float *)samples count:(NSInteger)count sampleRate:(NSInteger)sampleRate
                          index:(NSInteger)index chunks:(NSInteger)chunks;
@end

@implementation SBScyllasBandAudioChunk
- (instancetype)initWithSamples:(const float *)samples count:(NSInteger)count sampleRate:(NSInteger)sampleRate
                          index:(NSInteger)index chunks:(NSInteger)chunks {
    if ((self = [super init])) {
        _pcmFloat32Data = [NSData dataWithBytes:samples length:(NSUInteger)count * sizeof(float)];
        _sampleCount = count;
        _sampleRate = sampleRate;
        _chunkIndex = index;
        _chunkCount = chunks;
    }
    return self;
}
@end

namespace {

struct StreamContext {
    __unsafe_unretained SBScyllasBandChunkStartedBlock chunkStarted;
    __unsafe_unretained SBScyllasBandAudioChunkBlock audioChunk;
};

int32_t forward_event(const ScyllasBandEvent *event, void *user_data) {
    auto *context = static_cast<StreamContext *>(user_data);
    @autoreleasepool {
        if (event->type == SCYLLASBAND_EVENT_CHUNK_STARTED && context->chunkStarted) {
            NSDictionary *metadata = json_object(event->metadata_json);
            NSString *text = [metadata isKindOfClass:NSDictionary.class] && [metadata[@"text"] isKindOfClass:NSString.class] ? metadata[@"text"] : nil;
            return context->chunkStarted(event->chunk_index, event->chunk_count, text) ? 0 : 1;
        }
        if (event->type == SCYLLASBAND_EVENT_AUDIO) {
            if (event->sample_count < 0 || (event->sample_count > 0 && event->samples == nullptr)) return 1;
            SBScyllasBandAudioChunk *chunk = [[SBScyllasBandAudioChunk alloc] initWithSamples:event->samples
                                                                                        count:(NSInteger)event->sample_count
                                                                                   sampleRate:event->sample_rate
                                                                                        index:event->chunk_index
                                                                                       chunks:event->chunk_count];
            return context->audioChunk(chunk) ? 0 : 1;
        }
    }
    return 0;
}

}  // namespace

@implementation SBScyllasBand {
    ScyllasBandRuntime *_runtime;
}

+ (nullable NSURL *)preferredBundleURLInDirectory:(NSURL *)directory {
    NSMutableArray<NSString *> *order = [NSMutableArray array];
#if !TARGET_OS_SIMULATOR   // Core AI runs on devices only
    if (@available(iOS 27.0, macOS 27.0, *)) [order addObject:@"coreai"];
#endif
    if (@available(iOS 18.0, macOS 15.0, *)) [order addObject:@"coreml"];
    for (NSString *flavor in order) {
        NSURL *bundle = [directory URLByAppendingPathComponent:flavor isDirectory:YES];
        if ([NSFileManager.defaultManager fileExistsAtPath:[bundle URLByAppendingPathComponent:@"manifest.json"].path]) return bundle;
    }
    return nil;
}

- (nullable instancetype)initWithBundleURL:(NSURL *)bundleURL computeUnit:(SBScyllasBandComputeUnit)computeUnit error:(NSError **)error {
    if (!(self = [super init])) return nil;
    if (!bundleURL.isFileURL || bundleURL.path.length == 0) {
        fail(error, SBScyllasBandErrorInvalidArgument, @"The model bundle must be a file URL.");
        return nil;
    }
    ScyllasBandRuntimeOptions options{};
    options.bundle_dir = bundleURL.path.fileSystemRepresentation;
    options.threads = 0;
    options.accelerator = static_cast<ScyllasBandAccelerator>(computeUnit);
    const ScyllasBandStatus status = scyllasband_runtime_create(&options, &_runtime);
    if (status != SCYLLASBAND_OK) {
        fail(error, status == SCYLLASBAND_ERROR_INVALID_ARGUMENT ? SBScyllasBandErrorInvalidBundle : SBScyllasBandErrorRuntime,
             last_error_or(@"The Scylla's Band runtime could not load the bundle."));
        return nil;
    }
    _bundleInfo = [[SBScyllasBandBundleInfo alloc] initWithVoicesJSON:json_object(scyllasband_voices_json(_runtime))];
    if (_bundleInfo == nil) {
        fail(error, SBScyllasBandErrorInvalidBundle, @"The bundle declares no voices.");
        return nil;
    }
    _bundleURL = bundleURL.copy;
    return self;
}

- (void)dealloc {
    scyllasband_runtime_destroy(_runtime);
}

- (BOOL)warmUpWithVoice:(nullable NSString *)voiceIdentifier error:(NSError **)error {
    const ScyllasBandStatus status = scyllasband_warmup(_runtime, voiceIdentifier.UTF8String);
    return status == SCYLLASBAND_OK ? YES : fail(error, error_code(status), last_error_or(@"Warm-up failed."));
}

- (BOOL)synthesizeRequest:(SBScyllasBandRequest *)request
             chunkStarted:(nullable SBScyllasBandChunkStartedBlock)chunkStarted
               audioChunk:(SBScyllasBandAudioChunkBlock)audioChunk
                    error:(NSError **)error {
    if (request.text.length == 0 || request.voiceIdentifier.length == 0) {
        return fail(error, SBScyllasBandErrorInvalidArgument, @"Text and voice are required.");
    }
    if (!(request.speed > 0) || request.steps < 0) {
        return fail(error, SBScyllasBandErrorInvalidArgument, @"Speed must be positive and steps zero or positive.");
    }
    __attribute__((objc_precise_lifetime)) NSString *delivery = request.delivery.specification;   // its C string is used below
    ScyllasBandRequest native;
    scyllasband_request_init(&native);
    native.text = request.text.UTF8String;
    native.voice_id = request.voiceIdentifier.UTF8String;
    native.language = request.language.UTF8String;
    native.delivery = delivery.UTF8String;
    native.speed = request.speed;
    native.steps = static_cast<int32_t>(request.steps);
    native.normalize_text = request.normalizeText ? 1 : 0;
    if (request.seed != nil) {
        native.seed = request.seed.unsignedLongLongValue;
        native.has_seed = 1;
    }
    StreamContext context{chunkStarted, audioChunk};
    const ScyllasBandStatus status = scyllasband_synthesize_stream(_runtime, &native, forward_event, &context);
    if (status == SCYLLASBAND_OK) return YES;
    return fail(error, error_code(status), status == SCYLLASBAND_ERROR_CANCELLED ? @"Synthesis was cancelled." : last_error_or(@"Synthesis failed."));
}

- (void)requestCancellation {
    scyllasband_cancel(_runtime);
}

@end
