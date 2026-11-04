#import "include/ScyllasBandKit.h"

#include "scyllasband.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

NSErrorDomain const ScyllasBandErrorDomain = @"org.scyllasband.ScyllasBandKit";
SBScyllasBandDeliveryControl const SBScyllasBandDeliveryControlIntensity = @"intensity";
SBScyllasBandDeliveryControl const SBScyllasBandDeliveryControlPace = @"pace";
SBScyllasBandDeliveryControl const SBScyllasBandDeliveryControlPitch = @"pitch";

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
    // Shortest form: 1 -> "1", -0.5 -> "-0.5"; -0.0001 -> "0", not "-0".
    return [NSString stringWithFormat:@"%g", std::round(value * 1000.0f) / 1000.0f + 0.0f];
}

// The released model's range for every control; the C runtime clamps into it, then into the bundle's ranges.
constexpr float kDeliveryMinimum = -2.0f;
constexpr float kDeliveryMaximum = 2.0f;

NSArray<SBScyllasBandDeliveryControl> *delivery_controls() {
    return @[SBScyllasBandDeliveryControlIntensity, SBScyllasBandDeliveryControlPace, SBScyllasBandDeliveryControlPitch];
}

// nil for nil, NaN and infinities (the voice's default); finite values clamped into -2 to 2.
NSNumber *_Nullable delivery_value(NSNumber *_Nullable value) {
    if (![value isKindOfClass:NSNumber.class]) return nil;
    const float number = value.floatValue;
    if (!std::isfinite(number)) return nil;
    return @(std::clamp(number, kDeliveryMinimum, kDeliveryMaximum));
}

NSNumber *_Nullable json_number(id value) {
    return [value isKindOfClass:NSNumber.class] && std::isfinite([value doubleValue]) ? value : nil;
}

}  // namespace

@implementation SBScyllasBandDelivery

+ (SBScyllasBandDelivery *)voiceDefault {
    return [[SBScyllasBandDelivery alloc] initWithIntensity:nil pace:nil pitch:nil];
}

+ (SBScyllasBandDelivery *)neutral {
    return [[SBScyllasBandDelivery alloc] initWithIntensity:@0 pace:@0 pitch:@0];
}

- (instancetype)initWithIntensity:(nullable NSNumber *)intensity pace:(nullable NSNumber *)pace pitch:(nullable NSNumber *)pitch {
    if ((self = [super init])) {
        _intensity = delivery_value(intensity);
        _pace = delivery_value(pace);
        _pitch = delivery_value(pitch);
    }
    return self;
}

- (NSString *)specification {
    NSMutableArray<NSString *> *terms = [NSMutableArray arrayWithCapacity:3];
    auto append = [terms](SBScyllasBandDeliveryControl control, NSNumber *value) {
        if (value != nil) [terms addObject:[NSString stringWithFormat:@"%@=%@", control, format_value(value.floatValue)]];
    };
    append(SBScyllasBandDeliveryControlIntensity, self.intensity);
    append(SBScyllasBandDeliveryControlPace, self.pace);
    append(SBScyllasBandDeliveryControlPitch, self.pitch);
    return [terms componentsJoinedByString:@","];
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

- (NSString *)description {
    NSString *specification = self.specification;
    return [NSString stringWithFormat:@"<SBScyllasBandDelivery %@>", specification.length ? specification : @"voice default"];
}

@end

@interface SBScyllasBandDeliveryRange ()
- (instancetype)initWithMinimum:(float)minimum maximum:(float)maximum unit:(NSString *)unit;
@end

@implementation SBScyllasBandDeliveryRange
- (instancetype)initWithMinimum:(float)minimum maximum:(float)maximum unit:(NSString *)unit {
    if ((self = [super init])) {
        _minimum = minimum;
        _maximum = maximum;
        _unit = unit.copy;
    }
    return self;
}

- (NSString *)description {
    return [NSString stringWithFormat:@"<SBScyllasBandDeliveryRange %@ to %@ %@>", format_value(self.minimum), format_value(self.maximum), self.unit];
}
@end

@interface SBScyllasBandVoice ()
- (instancetype)initWithIdentifier:(NSString *)identifier
                         languages:(NSArray<NSString *> *)languages
                   defaultLanguage:(NSString *)defaultLanguage
                  deliveryDefaults:(SBScyllasBandDelivery *)deliveryDefaults;
@end

@implementation SBScyllasBandVoice
- (instancetype)initWithIdentifier:(NSString *)identifier
                         languages:(NSArray<NSString *> *)languages
                   defaultLanguage:(NSString *)defaultLanguage
                  deliveryDefaults:(SBScyllasBandDelivery *)deliveryDefaults {
    if ((self = [super init])) {
        _identifier = identifier.copy;
        _languages = languages.copy;
        _defaultLanguage = defaultLanguage.copy;
        _deliveryDefaults = deliveryDefaults;
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
        // "delivery": {"ranges": {control: [low, high]}, "defaults": {voice: {control: value}}, "units": {control: unit}}
        NSDictionary *delivery = [json[@"delivery"] isKindOfClass:NSDictionary.class] ? json[@"delivery"] : @{};
        NSDictionary *ranges = [delivery[@"ranges"] isKindOfClass:NSDictionary.class] ? delivery[@"ranges"] : @{};
        NSDictionary *units = [delivery[@"units"] isKindOfClass:NSDictionary.class] ? delivery[@"units"] : @{};
        NSDictionary *defaults = [delivery[@"defaults"] isKindOfClass:NSDictionary.class] ? delivery[@"defaults"] : @{};
        NSMutableDictionary<SBScyllasBandDeliveryControl, SBScyllasBandDeliveryRange *> *deliveryRanges = [NSMutableDictionary dictionary];
        for (SBScyllasBandDeliveryControl control in delivery_controls()) {
            NSArray *range = [ranges[control] isKindOfClass:NSArray.class] ? ranges[control] : nil;
            NSNumber *low = range.count == 2 ? json_number(range[0]) : nil;
            NSNumber *high = range.count == 2 ? json_number(range[1]) : nil;
            const BOOL valid = low != nil && high != nil && low.floatValue <= high.floatValue;
            NSString *unit = [units[control] isKindOfClass:NSString.class]
                ? units[control] : ([control isEqualToString:SBScyllasBandDeliveryControlPitch] ? @"semitones" : @"sd");
            deliveryRanges[control] = [[SBScyllasBandDeliveryRange alloc] initWithMinimum:valid ? low.floatValue : kDeliveryMinimum
                                                                                  maximum:valid ? high.floatValue : kDeliveryMaximum
                                                                                     unit:unit];
        }
        _deliveryRanges = deliveryRanges.copy;
        NSMutableArray<SBScyllasBandVoice *> *voices = [NSMutableArray array];
        for (NSDictionary *voice in [json[@"voices"] isKindOfClass:NSArray.class] ? json[@"voices"] : @[]) {
            if (![voice isKindOfClass:NSDictionary.class] || ![voice[@"id"] isKindOfClass:NSString.class]) continue;
            NSArray *languages = [voice[@"languages"] isKindOfClass:NSArray.class] ? voice[@"languages"] : @[];
            NSString *fallback = languages.firstObject ?: @"en_us";
            NSDictionary *values = [defaults[voice[@"id"]] isKindOfClass:NSDictionary.class] ? defaults[voice[@"id"]] : @{};
            SBScyllasBandDelivery *voiceDefaults =
                [[SBScyllasBandDelivery alloc] initWithIntensity:json_number(values[SBScyllasBandDeliveryControlIntensity]) ?: @0
                                                            pace:json_number(values[SBScyllasBandDeliveryControlPace]) ?: @0
                                                           pitch:json_number(values[SBScyllasBandDeliveryControlPitch]) ?: @0];
            [voices addObject:[[SBScyllasBandVoice alloc] initWithIdentifier:voice[@"id"]
                                                                   languages:languages
                                                             defaultLanguage:[voice[@"default_language"] isKindOfClass:NSString.class]
                                                                                 ? voice[@"default_language"] : fallback
                                                            deliveryDefaults:voiceDefaults]];
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

@implementation SBScyllasBandRequest

- (instancetype)initWithText:(NSString *)text voiceIdentifier:(NSString *)voiceIdentifier {
    if ((self = [super init])) {
        _text = text.copy;
        _voiceIdentifier = voiceIdentifier.copy;
        _delivery = SBScyllasBandDelivery.voiceDefault;
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
    if (@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)) [order addObject:@"coreai"];
#endif
    if (@available(iOS 18.0, macOS 15.0, visionOS 2.0, *)) [order addObject:@"coreml"];
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
    // No control set passes NULL: every control takes the voice's default. The C string is used below.
    __attribute__((objc_precise_lifetime)) NSString *delivery = request.delivery.specification;
    if (delivery.length == 0) delivery = nil;
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
