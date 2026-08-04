#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

FOUNDATION_EXPORT NSErrorDomain const ScyllasBandErrorDomain;

typedef NS_ERROR_ENUM(ScyllasBandErrorDomain, SBScyllasBandErrorCode) {
    SBScyllasBandErrorInvalidArgument = 1,
    SBScyllasBandErrorInvalidBundle = 2,
    SBScyllasBandErrorRuntime = 3,
    SBScyllasBandErrorCancelled = 4,
};

@interface SBScyllasBandVoice : NSObject

@property(nonatomic, copy, readonly) NSString *identifier;
@property(nonatomic, copy, readonly) NSString *displayName;
@property(nonatomic, copy, readonly) NSArray<NSString *> *languages;
@property(nonatomic, copy, readonly) NSString *defaultLanguage;

- (instancetype)init NS_UNAVAILABLE;

@end

@interface SBScyllasBandBundleInfo : NSObject

@property(nonatomic, copy, readonly) NSString *modelName;
@property(nonatomic, readonly) NSInteger sampleRate;
@property(nonatomic, copy, readonly) NSArray<SBScyllasBandVoice *> *voices;
@property(nonatomic, copy, readonly) NSArray<NSString *> *affectAxes;

- (instancetype)init NS_UNAVAILABLE;

@end

@interface SBScyllasBandSegmentSettings : NSObject <NSCopying>

@property(nonatomic, copy, readonly) NSString *voiceIdentifier;
@property(nonatomic, copy, readonly) NSString *language;
@property(nonatomic, copy, readonly, nullable) NSString *emotion;
@property(nonatomic, readonly) float emotionStrength;
@property(nonatomic, readonly) float emotionCFG;

- (instancetype)initWithVoiceIdentifier:(NSString *)voiceIdentifier
                               language:(NSString *)language
                                emotion:(nullable NSString *)emotion
                         emotionStrength:(float)emotionStrength
                             emotionCFG:(float)emotionCFG NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;

@end

/**
 A copied block of mono, native-endian Float32 PCM. The bytes remain valid
 after the native streaming callback returns.
 */
@interface SBScyllasBandAudioChunk : NSObject

@property(nonatomic, copy, readonly) NSData *pcmFloat32Data;
@property(nonatomic, readonly) NSInteger sampleCount;
@property(nonatomic, readonly) NSInteger sampleRate;
@property(nonatomic, readonly) NSInteger chunkIndex;
@property(nonatomic, readonly) NSInteger chunkCount;

- (instancetype)init NS_UNAVAILABLE;

@end


typedef BOOL (^SBScyllasBandChunkStartedBlock)(
    NSInteger chunkIndex,
    NSInteger chunkCount,
    NSString * _Nullable chunkText
);

typedef BOOL (^SBScyllasBandAudioChunkBlock)(SBScyllasBandAudioChunk *chunk);

/**
 A long-lived wrapper around one libscyllasband runtime.

 Create and use this object from one serial worker queue. It keeps the shared
 G2P/duration sessions warm and applies libscyllasband's target-bucket LRU.
 `requestCancellation` is thread-safe and may be called from the main thread.
 */
@interface SBScyllasBand : NSObject

@property(nonatomic, copy, readonly) NSURL *bundleURL;
@property(nonatomic, strong, readonly) SBScyllasBandBundleInfo *bundleInfo;
@property(nonatomic, readonly, getter=isCancellationRequested) BOOL cancellationRequested;

/** The memory-first profile used by the sample applications. */
@property(class, nonatomic, readonly) NSInteger defaultMobileTargetBucketCacheCapacity;

- (nullable instancetype)initWithBundleURL:(NSURL *)bundleURL
                               threadCount:(NSInteger)threadCount
                 targetBucketCacheCapacity:(NSInteger)targetBucketCacheCapacity
                                     error:(NSError **)error NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;

/** Performs one discarded, one-step render so first playback is predictable. */
- (BOOL)warmUpWithError:(NSError **)error;

/**
 Synchronously renders one logical speaker segment using native long-form
 planning. Call this off the main thread. Return NO from either callback, or
 call `requestCancellation`, to stop at the next streaming event.
 */
- (BOOL)synthesizeText:(NSString *)text
              settings:(SBScyllasBandSegmentSettings *)settings
                  seed:(uint64_t)seed
          chunkStarted:(nullable SBScyllasBandChunkStartedBlock)chunkStarted
             audioChunk:(SBScyllasBandAudioChunkBlock)audioChunk
                  error:(NSError **)error;

- (void)requestCancellation;

/** Clear a previous cancellation before enqueueing a new synthesis run. */
- (void)resetCancellation;

@end

NS_ASSUME_NONNULL_END
