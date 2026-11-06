#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

FOUNDATION_EXPORT NSErrorDomain const ScyllasBandErrorDomain;
/** In the userInfo of a score error: the index (NSNumber) of the note at fault in the score's "notes". */
FOUNDATION_EXPORT NSErrorUserInfoKey const SBScyllasBandScoreNoteIndexKey;

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

/** The waveform decoder a sung score renders through. */
typedef NS_ENUM(NSInteger, SBScyllasBandSingingDecoder) {
    /** The bundle's singing decoder: BigVGAN where the bundle carries it, else Vocos. */
    SBScyllasBandSingingDecoderDefault = 0,
    /** BigVGAN v2, fine-tuned on the voices: the singing decoder, cleanest on sustained notes, and the slowest. */
    SBScyllasBandSingingDecoderBigVGAN = 1,
    /** Vocos, the speech decoder ("turbo"): much faster, with some harmonic artefacts on sustained notes in some voices. */
    SBScyllasBandSingingDecoderVocos = 2,
};

/** What a bundle can sing (v4 bundles, graph contract scyllasband_measured_controls_v4); `supported` is NO otherwise. */
@interface SBScyllasBandSingingInfo : NSObject
@property(nonatomic, readonly, getter=isSupported) BOOL supported;
/** Score styles the bundle sings: "pop" (also "sung") and "ballad". */
@property(nonatomic, copy, readonly) NSArray<NSString *> *styles;
/** Decoders with graphs in this bundle: "bigvgan", "vocos". */
@property(nonatomic, copy, readonly) NSArray<NSString *> *decoders;
/** What SBScyllasBandSingingDecoderDefault uses. */
@property(nonatomic, copy, readonly) NSString *defaultDecoder;
@property(nonatomic, readonly) BOOL hasBigVGAN;
/** The bundle's singing flow steps, sampler and guidance (v4-20261106: 4, "euler", 1). */
@property(nonatomic, readonly) NSInteger steps;
@property(nonatomic, copy, readonly) NSString *sampler;
@property(nonatomic, readonly) float guidance;
/** The MIDI pitch each voice sings around; a score is moved by octaves next to it. */
@property(nonatomic, copy, readonly) NSDictionary<NSString *, NSNumber *> *voiceMidiCenters;
/** The band around a voice's centre, in semitones, that the octave register aims a song's mean pitch for (-3, +5). */
@property(nonatomic, readonly) float registerBelow;
@property(nonatomic, readonly) float registerAbove;
/** The longest phrase the bundle sings in one piece; put rests (0.25 s or more) between lines. */
@property(nonatomic, readonly) float maxPhraseSeconds;
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
/** "scyllasband_measured_controls_v3" (speech) or "_v4" (speech and singing). */
@property(nonatomic, copy, readonly) NSString *graphContract;
@property(nonatomic, strong, readonly) SBScyllasBandSingingInfo *singing;
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
/** Flow steps; 0 uses the bundle default (4 Euler steps in the released bundles). */
@property(nonatomic) NSInteger steps;
/** A plain time stretch on top of the delivery's pace; 1 is none. Must be positive. */
@property(nonatomic) float speed;
@property(nonatomic) BOOL normalizeText;
- (instancetype)initWithText:(NSString *)text voiceIdentifier:(NSString *)voiceIdentifier NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;
@end

/**
 A melody score to sing. `scoreJSON` is {"text", "bpm", "style": "pop" | "ballad", "notes": [[label, note, beats], ...]}:
 note a pitch name with octave ("C4", "F#3", "Bb3") or "rest" (label "-"), beats in quarter notes at bpm, and a word
 over several notes numbered ("baby~1", "baby~2"). Phrases split at rests of 0.25 s or more.
 */
@interface SBScyllasBandSingRequest : NSObject <NSCopying>
@property(nonatomic, copy) NSString *scoreJSON;
@property(nonatomic, copy) NSString *voiceIdentifier;
/** nil: the voice's default language. */
@property(nonatomic, copy, nullable) NSString *language;
/** nil: fresh noise for every request. Phrase i uses seed + i. */
@property(nonatomic, copy, nullable) NSNumber *seed;
/** Flow steps; 0 uses the bundle's singing setting (v4-20261106: 4). */
@property(nonatomic) NSInteger steps;
/** Classifier-free guidance on the melody and mode; nil uses the bundle's setting (1 for v4-20261106); 1 turns it off. */
@property(nonatomic, copy, nullable) NSNumber *guidance;
/** Initially SBScyllasBandSingingDecoderDefault (BigVGAN where the bundle carries it). Vocos is the fast ("turbo") option. */
@property(nonatomic) SBScyllasBandSingingDecoder decoder;
/** Semitones added to the whole score, before the octave register, to put a song in the voice's key. */
@property(nonatomic) NSInteger transpose;
/** YES (initially): move the song by octaves into the voice's sung range. */
@property(nonatomic) BOOL usesRegister;
- (instancetype)initWithScoreJSON:(NSString *)scoreJSON voiceIdentifier:(NSString *)voiceIdentifier NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;
@end

/** A copy of mono Float32 PCM (one spoken sentence, a whole sung song, or a tone preview); valid after the callback returns. */
@interface SBScyllasBandAudioChunk : NSObject
@property(nonatomic, copy, readonly) NSData *pcmFloat32Data;
@property(nonatomic, readonly) NSInteger sampleCount;
@property(nonatomic, readonly) NSInteger sampleRate;
@property(nonatomic, readonly) NSInteger chunkIndex;
@property(nonatomic, readonly) NSInteger chunkCount;
- (instancetype)init NS_UNAVAILABLE;
@end

/**
 The song audio one sung phrase settles, as mono Float32 PCM: samples [startSample, startSample + sampleCount) of the song,
 final and contiguous with the previous phrase's, so a player appends them in order (rests are already silence in them).
 The song is `songSampleCount` samples long. `sampleCount` may be 0.
 */
@interface SBScyllasBandSungPhrase : NSObject
@property(nonatomic, copy, readonly) NSData *pcmFloat32Data;
@property(nonatomic, readonly) NSInteger sampleCount;
@property(nonatomic, readonly) NSInteger sampleRate;
@property(nonatomic, readonly) NSInteger phraseIndex;
@property(nonatomic, readonly) NSInteger phraseCount;
@property(nonatomic, readonly) NSInteger startSample;
@property(nonatomic, readonly) NSInteger songSampleCount;
@property(nonatomic, copy, readonly) NSString *text;
- (instancetype)init NS_UNAVAILABLE;
@end

/** Return NO to stop the request. `text` is the sentence about to be spoken. */
typedef BOOL (^SBScyllasBandChunkStartedBlock)(NSInteger chunkIndex, NSInteger chunkCount, NSString *_Nullable text);
typedef BOOL (^SBScyllasBandAudioChunkBlock)(SBScyllasBandAudioChunk *chunk);
/** Return NO to stop the request. `text` is the phrase about to be sung. */
typedef BOOL (^SBScyllasBandPhraseStartedBlock)(NSInteger phraseIndex, NSInteger phraseCount, NSString *_Nullable text);
typedef BOOL (^SBScyllasBandSungPhraseBlock)(SBScyllasBandSungPhrase *phrase);

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

/**
 The phrase plan of a score without singing it: {"seconds", "samples", "register_shift", "phrases": [{"text", "start",
 "end", "frames", ...}], ...}. nil with an error for an invalid score or a bundle that cannot sing.
 */
- (nullable NSDictionary *)singPlanForRequest:(SBScyllasBandSingRequest *)request error:(NSError **)error NS_SWIFT_NAME(singPlan(for:));

/**
 Sings the score phrase by phrase (bundleInfo.singing.supported must be YES). Callbacks run on the calling thread. Returns
 NO with SBScyllasBandErrorCancelled when a callback returned NO or `requestCancellation` was called.
 */
- (BOOL)singRequest:(SBScyllasBandSingRequest *)request
      phraseStarted:(nullable SBScyllasBandPhraseStartedBlock)phraseStarted
        phraseAudio:(SBScyllasBandSungPhraseBlock)phraseAudio
              error:(NSError **)error NS_SWIFT_NAME(sing(_:phraseStarted:phraseAudio:));

/** Sings the whole score into one waveform (phrases on the song's timeline, silence in the rests). */
- (nullable SBScyllasBandAudioChunk *)singRequest:(SBScyllasBandSingRequest *)request error:(NSError **)error NS_SWIFT_NAME(sing(_:));

/**
 Checks a score without a runtime and returns it normalized after `transpose` semitones: {"text", "bpm", "style", "mode",
 "seconds", "notes": [{"index", "label", "word_index", "part", "note", "midi", "beats", "start", "end"}], "words", "range",
 "lines": [{"first", "last"}]}. An invalid score fails with SBScyllasBandErrorInvalidArgument, and its userInfo names the
 note at fault (SBScyllasBandScoreNoteIndexKey) when there is one.
 */
+ (nullable NSDictionary *)parseScoreJSON:(NSString *)scoreJSON transpose:(NSInteger)transpose error:(NSError **)error
    NS_SWIFT_NAME(parseScore(json:transpose:));

/**
 How the request's score sits in its voice's range: {"voice_center", "band", "transpose", "register_shift", "mean_midi",
 "low_note", "high_note", "outside_semitones", "suggested_transpose"}. Runs no graph and does not wait for a song in
 progress, so it can follow every edit.
 */
- (nullable NSDictionary *)singFitForRequest:(SBScyllasBandSingRequest *)request error:(NSError **)error NS_SWIFT_NAME(singFit(for:));

/** The request's score as plain tones at the pitch it will be sung (transpose and register applied); instant, no graph. */
- (nullable SBScyllasBandAudioChunk *)previewForRequest:(SBScyllasBandSingRequest *)request error:(NSError **)error NS_SWIFT_NAME(preview(for:));

/** Loads the graphs a short sung phrase needs with `decoder`, so the first song starts quickly. */
- (BOOL)warmUpSingingWithVoice:(nullable NSString *)voiceIdentifier decoder:(SBScyllasBandSingingDecoder)decoder error:(NSError **)error
    NS_SWIFT_NAME(warmUpSinging(voice:decoder:));

/** Stops requests already issued (running or waiting) at their next graph call; later requests are unaffected. */
- (void)requestCancellation;

@end

NS_ASSUME_NONNULL_END
