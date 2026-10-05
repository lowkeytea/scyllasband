/*
 * Internal entry points for the parity harness (tests/parity_harness.py). Not part of the public ABI.
 * Every char** / float** output is malloc'd; release it with scyllasband_test_free.
 */
#ifndef SCYLLASBAND_TEST_API_H
#define SCYLLASBAND_TEST_API_H

#include "scyllasband.h"

#ifdef __cplusplus
extern "C" {
#endif

SCYLLASBAND_API void scyllasband_test_free(void* pointer);

/* Text frontend without a runtime. */
SCYLLASBAND_API int32_t scyllasband_test_normalize(const char* text, const char* language, char** out_text);
SCYLLASBAND_API int32_t scyllasband_test_split_sentences(const char* text, char** out_json);
SCYLLASBAND_API int32_t scyllasband_test_punctuated_segments(const char* text, int32_t max_chars, char** out_json);
SCYLLASBAND_API int32_t scyllasband_test_unicode(const char* text, const char* operation /* "lower" | "nfc" */, char** out_text);

/* {"phones", "word_starts", "word_boundary_candidates", "punctuation", "g2p_segments"} */
SCYLLASBAND_API int32_t scyllasband_test_phonemize(ScyllasBandRuntime* runtime, const char* text, const char* language, char** out_json);

/*
 * One sentence from explicit inputs. request_json: {"phones", "word_starts", "before_ids", "after_ids", "voice", "language",
 * "delivery", "steps", "sampler", "speed", "durations" (optional: skip the duration graph)}.
 * noise: [latent_dim, noise_frames] row-major; noise_frames must equal the sentence's latent frames. NULL noise only
 * predicts durations. prefix: [latent_dim, prefix_frames] previously spoken latents (flow prefix and decode context).
 * out_json: {"durations", "latent_frames", "bucket", ...}; out_latents [latent_dim, frames]; out_audio samples.
 */
SCYLLASBAND_API int32_t scyllasband_test_synthesize_sentence(ScyllasBandRuntime* runtime, const char* request_json, const float* noise,
                                                             int64_t noise_frames, const float* prefix, int64_t prefix_frames, char** out_json,
                                                             float** out_latents, int64_t* out_latent_count, float** out_audio,
                                                             int64_t* out_sample_count);

/* Replaces the flow noise of every later synthesis on `runtime`: (chunk index, seed, has seed, latent dim, frames,
   out [latent_dim, frames], user data). NULL restores the built-in generator. */
typedef void (*ScyllasBandTestNoise)(int32_t chunk_index, uint64_t seed, int32_t has_seed, int32_t latent_dim, int64_t frames, float* out,
                                     void* user_data);
SCYLLASBAND_API void scyllasband_test_set_noise(ScyllasBandRuntime* runtime, ScyllasBandTestNoise noise, void* user_data);

#ifdef __cplusplus
}
#endif

#endif /* SCYLLASBAND_TEST_API_H */
