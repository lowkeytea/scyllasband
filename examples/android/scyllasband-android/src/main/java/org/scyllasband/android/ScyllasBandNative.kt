package org.scyllasband.android

internal object ScyllasBandNative {
    init {
        System.loadLibrary("scyllasband_android_jni")
    }

    external fun create(bundlePath: String, threadCount: Int, accelerator: Int): Long

    external fun voicesJson(handle: Long): String

    external fun backend(): String

    external fun warmup(handle: Long, voiceId: String?)

    /** Returns false when the listener or [cancel] stopped the request before it finished. */
    external fun synthesizeStreaming(
        handle: Long,
        text: String,
        voiceId: String,
        language: String?,
        delivery: String?,
        steps: Int,
        seed: Long,
        hasSeed: Boolean,
        listener: ScyllasBandNativeChunkListener,
    ): Boolean

    /** The phrase plan of a score as JSON, without singing it. */
    external fun singPlanJson(
        handle: Long,
        scoreJson: String,
        voiceId: String,
        language: String?,
        transpose: Int,
        useRegister: Boolean,
    ): String

    /**
     * Sings a score phrase by phrase. [decoder]: null for the bundle's singing decoder, "bigvgan" or "vocos";
     * [guidance] <= 0 uses the bundle's setting. Returns false when the listener or [cancel] stopped it.
     */
    external fun singStreaming(
        handle: Long,
        scoreJson: String,
        voiceId: String,
        language: String?,
        decoder: String?,
        steps: Int,
        guidance: Float,
        transpose: Int,
        useRegister: Boolean,
        seed: Long,
        hasSeed: Boolean,
        listener: ScyllasBandNativeSingListener,
    ): Boolean

    /**
     * Checks a score without a runtime: the normalized score JSON, or {"error": message, "note_index": n | null} when the
     * runtime refuses it.
     */
    external fun scoreParseJson(scoreJson: String, transpose: Int): String

    /** How a score sits in its voice's range (no graph runs). */
    external fun singFitJson(
        handle: Long,
        scoreJson: String,
        voiceId: String,
        language: String?,
        transpose: Int,
        useRegister: Boolean,
    ): String

    /** The score as tones at the pitch it will be sung, at the bundle's sample rate (no graph runs). */
    external fun singPreview(handle: Long, scoreJson: String, voiceId: String, transpose: Int, useRegister: Boolean): FloatArray

    external fun warmupSinging(handle: Long, voiceId: String?, decoder: String?)

    external fun cancel(handle: Long)

    external fun destroy(handle: Long)
}

internal interface ScyllasBandNativeSingListener {
    fun onPhraseStarted(phraseIndex: Int, phraseCount: Int, metadataJson: String?): Boolean

    /** [samples] continue the song where the previous phrase's ended; [metadataJson] carries start_sample and song_samples. */
    fun onPhraseAudio(samples: FloatArray, phraseIndex: Int, phraseCount: Int, metadataJson: String?): Boolean
}

internal interface ScyllasBandNativeChunkListener {
    fun onChunkStarted(chunkIndex: Int, chunkCount: Int, metadataJson: String?): Boolean

    fun onAudioChunk(samples: FloatArray, chunkIndex: Int, chunkCount: Int): Boolean
}
