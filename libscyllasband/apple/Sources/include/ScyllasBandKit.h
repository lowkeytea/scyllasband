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

@interface SBScyllasBandVoice : NSObject
@property(nonatomic, copy, readonly) NSString *identifier;
@property(nonatomic, copy, readonly) NSArray<NSString *> *languages;
@property(nonatomic, copy, readonly) NSString *defaultLanguage;
- (instancetype)init NS_UNAVAILABLE;
@end

@interface SBScyllasBandBundleInfo : NSObject
/** "coreml" or "coreai". */
@property(nonatomic, copy, readonly) NSString *backend;
@property(nonatomic, copy, readonly) NSString *releaseIdentifier;
@property(nonatomic, readonly) NSInteger sampleRate;
@property(nonatomic, copy, readonly) NSArray<SBScyllasBandVoice *> *voices;
@property(nonatomic, copy, readonly) NSString *defaultVoice;
- (nullable SBScyllasBandVoice *)voiceWithIdentifier:(NSString *)identifier;
- (instancetype)init NS_UNAVAILABLE;
@end

/**
 Delivery controls on the 0–4 scale with 2 neutral; the runtime clamps them into the supported ranges (energy 1.4–2.3,
 tension 1.0–2.4, valence 1.4–2.2, assertiveness 1.8–2.4). Combinations give familiar deliveries, e.g. joyful energy 2.3
 tension 2 valence 2.2 assertiveness 2.2; angry energy 2.3 tension 2.4 valence 1.4 assertiveness 2.4. `whisper` is no longer
 supported and is ignored.
 */
@interface SBScyllasBandDelivery : NSObject <NSCopying>
@property(nonatomic, readonly) float energy;
@property(nonatomic, readonly) float tension;
@property(nonatomic, readonly) float valence;
@property(nonatomic, readonly) float assertiveness;
@property(nonatomic, readonly) BOOL whisper;
@property(class, nonatomic, readonly) SBScyllasBandDelivery *neutral;
/** Values outside 0–4 are clamped. */
- (instancetype)initWithEnergy:(float)energy
                       tension:(float)tension
                       valence:(float)valence
                 assertiveness:(float)assertiveness
                       whisper:(BOOL)whisper NS_DESIGNATED_INITIALIZER;
/** The runtime's delivery string, e.g. "energy=2.3,tension=2,valence=2,assertiveness=2". */
@property(nonatomic, copy, readonly) NSString *specification;
- (instancetype)init NS_UNAVAILABLE;
@end

@interface SBScyllasBandRequest : NSObject <NSCopying>
@property(nonatomic, copy) NSString *text;
@property(nonatomic, copy) NSString *voiceIdentifier;
/** nil: the voice's default language. */
@property(nonatomic, copy, nullable) NSString *language;
@property(nonatomic, copy) SBScyllasBandDelivery *delivery;
/** nil: fresh noise for every request. Sentence i of the plan uses seed + i. */
@property(nonatomic, copy, nullable) NSNumber *seed;
/** Flow steps; 0 uses the bundle default (8). */
@property(nonatomic) NSInteger steps;
/** Duration scale; 1 is the model's own pace. */
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
