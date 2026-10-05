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

    external fun cancel(handle: Long)

    external fun destroy(handle: Long)
}

internal interface ScyllasBandNativeChunkListener {
    fun onChunkStarted(chunkIndex: Int, chunkCount: Int, metadataJson: String?): Boolean

    fun onAudioChunk(samples: FloatArray, chunkIndex: Int, chunkCount: Int): Boolean
}
