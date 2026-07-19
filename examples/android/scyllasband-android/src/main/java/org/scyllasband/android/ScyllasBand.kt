package org.scyllasband.android

import android.content.Context
import org.json.JSONObject
import java.io.Closeable
import java.io.File

class ScyllasBand private constructor(
    private val appContext: Context,
) : Closeable {
    private var nativeHandle = 0L
    private var installedBundle: File? = null

    var bundleInfo: ScyllasBandBundleInfo? = null
        private set

    /**
     * Installs and initializes one persistent native ONNX runtime.
     *
     * [targetBucketCacheCapacity] bounds heavyweight vector/vocoder session
     * pairs with an LRU policy implemented by libscyllasband. The default of
     * one is the mobile memory profile; zero retains every encountered bucket
     * for maximum steady-state throughput on high-memory devices.
     */
    @Synchronized
    fun initialize(
        threadCount: Int = defaultThreadCount(),
        targetBucketCacheCapacity: Int = DEFAULT_MOBILE_TARGET_BUCKET_CACHE_CAPACITY,
    ): ScyllasBandBundleInfo {
        bundleInfo?.let { return it }
        require(targetBucketCacheCapacity >= 0) {
            "targetBucketCacheCapacity must be non-negative"
        }
        val bundle = ScyllasBandAssetInstaller(appContext).install()
        val info = parseBundleInfo(bundle)
        val handle = ScyllasBandNative.create(
            bundle.absolutePath,
            threadCount.coerceAtLeast(1),
            targetBucketCacheCapacity,
        )
        check(handle != 0L) { "Scylla's Band native runtime returned an invalid handle" }
        try {
            val warmupVoice = info.voices.first()
            ScyllasBandNative.warmup(handle, warmupVoice.id, warmupVoice.defaultLanguage)
        } catch (error: Throwable) {
            ScyllasBandNative.destroy(handle)
            throw error
        }
        installedBundle = bundle
        nativeHandle = handle
        bundleInfo = info
        return info
    }

    @Synchronized
    fun synthesizeSegment(
        text: String,
        settings: ScyllasBandSegmentSettings,
        seed: Long,
    ): FloatArray {
        validateSegment(text, settings)
        return ScyllasBandNative.synthesizeSegment(
            nativeHandle,
            text,
            settings.voiceId,
            settings.language,
            settings.affectSpec(),
            settings.emotionCfg,
            seed,
        )
    }

    @Synchronized
    fun synthesizeSegmentStreaming(
        text: String,
        settings: ScyllasBandSegmentSettings,
        seed: Long,
        onChunkStarted: (chunkIndex: Int, chunkCount: Int, chunkText: String?) -> Boolean,
        onAudioChunk: (samples: FloatArray, chunkIndex: Int, chunkCount: Int) -> Boolean,
    ) {
        validateSegment(text, settings)
        ScyllasBandNative.synthesizeSegmentStreaming(
            nativeHandle,
            text,
            settings.voiceId,
            settings.language,
            settings.affectSpec(),
            settings.emotionCfg,
            seed,
            object : ScyllasBandNativeChunkListener {
                override fun onChunkStarted(
                    chunkIndex: Int,
                    chunkCount: Int,
                    metadataJson: String?,
                ): Boolean = onChunkStarted(
                    chunkIndex,
                    chunkCount,
                    metadataJson?.let(::chunkTextFromMetadata),
                )

                override fun onAudioChunk(
                    samples: FloatArray,
                    chunkIndex: Int,
                    chunkCount: Int,
                ): Boolean = onAudioChunk(samples, chunkIndex, chunkCount)
            },
        )
    }

    @Synchronized
    override fun close() {
        if (nativeHandle != 0L) {
            ScyllasBandNative.destroy(nativeHandle)
            nativeHandle = 0L
        }
        bundleInfo = null
        installedBundle = null
    }

    private fun validateSegment(text: String, settings: ScyllasBandSegmentSettings) {
        require(text.isNotBlank()) { "Segment text must not be blank" }
        val info = checkNotNull(bundleInfo) { "Call initialize() before synthesis" }
        val voice = info.voices.firstOrNull { it.id == settings.voiceId }
            ?: error("Unknown Scylla's Band voice: ${settings.voiceId}")
        require(settings.language in voice.languages) {
            "${settings.language} is not available for ${settings.voiceId}"
        }
        require(settings.emotion == null || settings.emotion in info.affectAxes) {
            "Unknown Scylla's Band affect axis: ${settings.emotion}"
        }
    }

    private fun chunkTextFromMetadata(metadataJson: String): String? =
        runCatching {
            JSONObject(metadataJson).optString("text").trim().takeIf { it.isNotEmpty() }
        }.getOrNull()

    private fun parseBundleInfo(bundle: File): ScyllasBandBundleInfo {
        val manifest = JSONObject(File(bundle, "manifest.json").readText(Charsets.UTF_8))
        val voicesJson = manifest.getJSONArray("voices")
        val voices = buildList {
            for (index in 0 until voicesJson.length()) {
                val value = voicesJson.getJSONObject(index)
                val languagesJson = value.getJSONArray("languages")
                val languages = buildList {
                    for (languageIndex in 0 until languagesJson.length()) {
                        add(languagesJson.getString(languageIndex))
                    }
                }
                val id = value.getString("id")
                add(
                    ScyllasBandVoice(
                        id = id,
                        displayName = value.optString("name", id).replaceFirstChar { it.uppercase() },
                        languages = languages,
                        defaultLanguage = value.optString("default_language", languages.first()),
                    ),
                )
            }
        }
        val axesJson = manifest.getJSONObject("controls")
            .getJSONObject("affect")
            .getJSONArray("axes")
        val axes = buildList {
            for (index in 0 until axesJson.length()) add(axesJson.getString(index))
        }
        return ScyllasBandBundleInfo(
            modelName = manifest.getString("model_name"),
            sampleRate = manifest.getJSONObject("audio").getInt("sample_rate"),
            voices = voices,
            affectAxes = axes,
        )
    }

    companion object {
        const val DEFAULT_MOBILE_TARGET_BUCKET_CACHE_CAPACITY = 1

        fun create(context: Context): ScyllasBand = ScyllasBand(context.applicationContext)

        private fun defaultThreadCount(): Int =
            Runtime.getRuntime().availableProcessors().coerceIn(1, 6)
    }
}
