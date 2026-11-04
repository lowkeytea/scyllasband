#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

FOUNDATION_EXPORT NSErrorDomain const ScyllasBandErrorDomain;

typedef NS_ERROR_ENUM(ScyllasBandErrorDomain, SBScyllasBandErrorCode) {
    SBScyllasBandErrorInvalidArgument = 1,
    SBScyllasBandErrorInvalidBundle = 2,
    SBScyllasBandErrorRuntime = 3,
    SBScyllasBandErrorCancelled = 4,
};

/** Where the graphs run. Automatic uses the bundle's recommendation for each graph. */
typedef NS_ENUM(NSInteger, SBScyllasBandComputeUnit) {
    SBScyllasBandComputeUnitCPU = 0,
    SBScyllasBandComputeUnitGPU = 1,
    SBScyllasBandComputeUnitAutomatic = 2,
    SBScyllasBandComputeUnitNeuralEngine = 3,
};

/** The delivery controls: the keys of `SBScyllasBandBundleInfo.deliveryRanges` and of the runtime's delivery string. */
typedef NSString *SBScyllasBandDeliveryControl NS_TYPED_ENUM;
/** Calm (negative) to intense (positive), in the voice's own standard deviations. */
FOUNDATION_EXPORT SBScyllasBandDeliveryControl const SBScyllasBandDeliveryControlIntensity;
/** Slower (negative) to faster (positive), words and pauses together, in the voice's own standard deviations. */
FOUNDATION_EXPORT SBScyllasBandDeliveryControl const SBScyllasBandDeliveryControlPace;
/** A shift of the voice's pitch in semitones. */
FOUNDATION_EXPORT SBScyllasBandDeliveryControl const SBScyllasBandDeliveryControlPitch;

/**
 How a voice delivers a line: intensity, pace and pitch, each optional. A control left nil takes the voice's default
 (`SBScyllasBandVoice.deliveryDefaults`; 0 for every voice in the released model), and 0 is the voice's own delivery.

 - `intensity`: calm (negative) to intense (positive), in the voice's own standard deviations. The emotion itself comes
   from the text.
 - `pace`: slower (negative) to faster (positive), words and pauses together, in the voice's own standard deviations.
 - `pitch`: a shift of the voice's pitch in semitones.

 The released model supports -2 to 2 for each; the runtime clamps every request into the bundle's ranges
 (`SBScyllasBandBundleInfo.deliveryRanges`). The v2 axes (energy, tension, valence, assertiveness, whisper) are gone.

 From Swift: `SBScyllasBandDelivery(intensity: 1, pace: -0.5, pitch: nil)`; read a control as `delivery.pace?.floatValue`.
 */
@interface SBScyllasBandDelivery : NSObject <NSCopying>
/** nil: the voice's default. */
@property(nonatomic, copy, readonly, nullable) NSNumber *intensity;
/** nil: the voice's default. */
@property(nonatomic, copy, readonly, nullable) NSNumber *pace;
/** nil: the voice's default. */
@property(nonatomic, copy, readonly, nullable) NSNumber *pitch;
/** No control set: every control takes the voice's default. The delivery string is empty. A request's initial delivery. */
@property(class, nonatomic, readonly) SBScyllasBandDelivery *voiceDefault;
/** Every control 0, the voice's own delivery, whatever the bundle's defaults: "intensity=0,pace=0,pitch=0". */
@property(class, nonatomic, readonly) SBScyllasBandDelivery *neutral;
/**
 nil leaves a control at the voice's default. Finite values outside -2 to 2 are clamped into it (the runtime then clamps
 into the bundle's ranges); NaN and infinities leave the control at the voice's default.
 */
- (instancetype)initWithIntensity:(nullable NSNumber *)intensity
                             pace:(nullable NSNumber *)pace
                            pitch:(nullable NSNumber *)pitch NS_DESIGNATED_INITIALIZER;
/** The runtime's delivery string for the controls that are set, e.g. "intensity=1,pace=-0.5"; "" when none is. */
@property(nonatomic, copy, readonly) NSString *specification;
- (instancetype)init NS_UNAVAILABLE;
@end

/** The range the runtime clamps one delivery control into. */
@interface SBScyllasBandDeliveryRange : NSObject
@property(nonatomic, readonly) float minimum;
@property(nonatomic, readonly) float maximum;
/** "sd" (the voice's own standard deviations: intensity and pace) or "semitones" (pitch). */
@property(nonatomic, copy, readonly) NSString *unit;
- (instancetype)init NS_UNAVAILABLE;
@end

@interface SBScyllasBandVoice : NSObject
@property(nonatomic, copy, readonly) NSString *identifier;
@property(nonatomic, copy, readonly) NSArray<NSString *> *languages;
@property(nonatomic, copy, readonly) NSString *defaultLanguage;
/** What a request that leaves a control out uses for this voice, with every control set (0 = the voice's own). */
@property(nonatomic, copy, readonly) SBScyllasBandDelivery *deliveryDefaults;
- (instancetype)init NS_UNAVAILABLE;
@end

@interface SBScyllasBandBundleInfo : NSObject
/** "coreml" or "coreai". */
@property(nonatomic, copy, readonly) NSString *backend;
@property(nonatomic, copy, readonly) NSString *releaseIdentifier;
@property(nonatomic, readonly) NSInteger sampleRate;
@property(nonatomic, copy, readonly) NSArray<SBScyllasBandVoice *> *voices;
@property(nonatomic, copy, readonly) NSString *defaultVoice;
/**
 The range the runtime clamps each delivery control into (intensity and pace in standard deviations, pitch in
 semitones; -2 to 2 for each in the released model). Holds all three controls.
 */
@property(nonatomic, copy, readonly) NSDictionary<SBScyllasBandDeliveryControl, SBScyllasBandDeliveryRange *> *deliveryRanges;
- (nullable SBScyllasBandVoice *)voiceWithIdentifier:(NSString *)identifier;
- (instancetype)init NS_UNAVAILABLE;
@end

@interface SBScyllasBandRequest : NSObject <NSCopying>
@property(nonatomic, copy) NSString *text;
@property(nonatomic, copy) NSString *voiceIdentifier;
/** nil: the voice's default language. */
@property(nonatomic, copy, nullable) NSString *language;
/** Initially `SBScyllasBandDelivery.voiceDefault`. */
@property(nonatomic, copy) SBScyllasBandDelivery *delivery;
/** nil: fresh noise for every request. Sentence i of the plan uses seed + i. */
@property(nonatomic, copy, nullable) NSNumber *seed;
/** Flow steps; 0 uses the bundle default (8). */
@property(nonatomic) NSInteger steps;
/** A plain time stretch on top of the delivery's pace; 1 is none. Must be positive. */
@property(nonatomic) float speed;
@property(nonatomic) BOOL normalizeText;
- (instancetype)initWithText:(NSString *)text voiceIdentifier:(NSString *)voiceIdentifier NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;
@end

/** A copy of one sentence's mono Float32 PCM; it stays valid after the callback returns. */
@interface SBScyllasBandAudioChunk : NSObject
@property(nonatomic, copy, readonly) NSData *pcmFloat32Data;
@property(nonatomic, readonly) NSInteger sampleCount;
@property(nonatomic, readonly) NSInteger sampleRate;
@property(nonatomic, readonly) NSInteger chunkIndex;
@property(nonatomic, readonly) NSInteger chunkCount;
- (instancetype)init NS_UNAVAILABLE;
@end

/** Return NO to stop the request. `text` is the sentence about to be spoken. */
typedef BOOL (^SBScyllasBandChunkStartedBlock)(NSInteger chunkIndex, NSInteger chunkCount, NSString *_Nullable text);
typedef BOOL (^SBScyllasBandAudioChunkBlock)(SBScyllasBandAudioChunk *chunk);

/**
 One libscyllasband runtime over a Core ML (iOS 18+) or Core AI (iOS 27+) bundle.

 Create it, warm it and synthesize from one serial worker queue, never the main thread: the first load of a Core AI
 bundle specializes its graphs for the device, which the system caches across launches until the next OS update.
 `requestCancellation` is safe from any thread.
 */
@interface SBScyllasBand : NSObject

@property(nonatomic, copy, readonly) NSURL *bundleURL;
@property(nonatomic, strong, readonly) SBScyllasBandBundleInfo *bundleInfo;

/**
 The bundle this device should run from a directory holding `coreai/` and/or `coreml/` bundle folders (each with
 its manifest.json): Core AI on iOS 27 and later (devices only), Core ML otherwise. nil when neither usable bundle is
 present. In the Simulator, Core ML runs on the CPU whatever compute unit is requested.
 */
+ (nullable NSURL *)preferredBundleURLInDirectory:(NSURL *)directory;

- (nullable instancetype)initWithBundleURL:(NSURL *)bundleURL
                               computeUnit:(SBScyllasBandComputeUnit)computeUnit
                                     error:(NSError **)error NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;

/** Loads the graphs a short sentence needs, so the first request starts quickly. */
- (BOOL)warmUpWithVoice:(nullable NSString *)voiceIdentifier error:(NSError **)error;

/**
 Speaks the request sentence by sentence; each sentence's audio joins the previous one directly. Callbacks run on
 the calling thread. Returns NO with SBScyllasBandErrorCancelled when a callback returned NO or
 `requestCancellation` was called.
 */
- (BOOL)synthesizeRequest:(SBScyllasBandRequest *)request
             chunkStarted:(nullable SBScyllasBandChunkStartedBlock)chunkStarted
               audioChunk:(SBScyllasBandAudioChunkBlock)audioChunk
                    error:(NSError **)error;

/** Stops requests already issued (running or waiting) at their next graph call; later requests are unaffected. */
- (void)requestCancellation;

@end

NS_ASSUME_NONNULL_END
