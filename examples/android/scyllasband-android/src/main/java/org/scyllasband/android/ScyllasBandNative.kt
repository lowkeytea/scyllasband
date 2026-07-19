package org.scyllasband.android

internal object ScyllasBandNative {
    init {
        System.loadLibrary("scyllasband_android_jni")
    }

    external fun create(
        bundlePath: String,
        threadCount: Int,
        targetBucketCacheCapacity: Int,
    ): Long

    external fun warmup(handle: Long, voiceId: String, language: String)

    external fun synthesizeSegment(
        handle: Long,
        text: String,
        voiceId: String,
        language: String,
        affect: String?,
        emotionCfg: Float,
        seed: Long,
    ): FloatArray

    external fun synthesizeSegmentStreaming(
        handle: Long,
        text: String,
        voiceId: String,
        language: String,
        affect: String?,
        emotionCfg: Float,
        seed: Long,
        listener: ScyllasBandNativeChunkListener,
    )

    external fun destroy(handle: Long)
}

internal interface ScyllasBandNativeChunkListener {
    fun onChunkStarted(chunkIndex: Int, chunkCount: Int, metadataJson: String?): Boolean

    fun onAudioChunk(samples: FloatArray, chunkIndex: Int, chunkCount: Int): Boolean
}
