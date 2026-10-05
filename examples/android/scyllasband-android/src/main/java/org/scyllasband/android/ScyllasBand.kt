package org.scyllasband.android

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.io.Closeable

/**
 * One persistent Scylla's Band runtime. [initialize] installs the bundled model into the app's
 * no-backup storage on first use, creates the native runtime and warms it up; [synthesizeStreaming]
 * then speaks text sentence by sentence; [cancel] stops a request from any thread.
 */
class ScyllasBand private constructor(
    private val appContext: Context,
) : Closeable {
    @Volatile
    private var nativeHandle = 0L
    private val handleLock = Any()

    var bundleInfo: ScyllasBandBundleInfo? = null
        private set

    /**
     * Installs the bundle, creates the runtime and warms it up. Blocking: call it off the main thread.
     *
     * [threadCount] is the CPU thread count per graph; 0 lets the runtime choose. The default [accelerator]
     * is the CPU; GPU and AUTO use LiteRT's GPU accelerator; AUTO falls back to the CPU if the GPU runtime
     * fails to compile or run its warmup.
     */
    @Synchronized
    fun initialize(
        threadCount: Int = 0,
        accelerator: ScyllasBandAccelerator = ScyllasBandAccelerator.CPU,
    ): ScyllasBandBundleInfo {
        bundleInfo?.let { return it }
        require(threadCount >= 0) { "threadCount must not be negative" }
        val bundle = ScyllasBandAssetInstaller(appContext).install()
        return try {
            open(bundle.absolutePath, threadCount, accelerator)
        } catch (error: Exception) {
            // AUTO promises a working runtime: a GPU that compiles but fails its first invoke (the warmup)
            // is replaced by the CPU.
            if (accelerator != ScyllasBandAccelerator.AUTO) throw error
            Log.w(TAG, "GPU runtime failed, falling back to the CPU: ${error.message}")
            open(bundle.absolutePath, threadCount, ScyllasBandAccelerator.CPU)
        }
    }

    private fun open(
        bundlePath: String,
        threadCount: Int,
        accelerator: ScyllasBandAccelerator,
    ): ScyllasBandBundleInfo {
        val handle = ScyllasBandNative.create(bundlePath, threadCount, accelerator.code)
        check(handle != 0L) { "Scylla's Band native runtime returned an invalid handle" }
        try {
            val info = parseBundleInfo(ScyllasBandNative.voicesJson(handle))
            ScyllasBandNative.warmup(handle, info.voices.first().id)
            nativeHandle = handle
            bundleInfo = info
            return info
        } catch (error: Throwable) {
            ScyllasBandNative.destroy(handle)
            throw error
        }
    }

    /**
     * Speaks [text] one sentence at a time. [onChunkStarted] reports the sentence about to be spoken and
     * [onAudioChunk] receives its mono float PCM at [ScyllasBandBundleInfo.sampleRate]; consecutive chunks
     * join directly. A callback returns false to stop. [steps] 0 uses the bundle default (8 Heun steps).
     * Returns false when the request was stopped by a callback or [cancel].
     */
    @Synchronized
    fun synthesizeStreaming(
        text: String,
        settings: ScyllasBandSegmentSettings,
        seed: Long? = null,
        steps: Int = 0,
        onChunkStarted: (chunkIndex: Int, chunkCount: Int, chunkText: String?) -> Boolean,
        onAudioChunk: (samples: FloatArray, chunkIndex: Int, chunkCount: Int) -> Boolean,
    ): Boolean {
        require(text.isNotBlank()) { "Segment text must not be blank" }
        require(steps >= 0) { "steps must not be negative" }
        val info = checkNotNull(bundleInfo) { "Call initialize() before synthesis" }
        val voice = info.voices.firstOrNull { it.id == settings.voiceId }
            ?: error("Unknown Scylla's Band voice: ${settings.voiceId}")
        require(settings.language in voice.languages) {
            "${settings.language} is not available for ${settings.voiceId}"
        }
        return ScyllasBandNative.synthesizeStreaming(
            nativeHandle,
            text,
            settings.voiceId,
            settings.language,
            settings.delivery.spec(),
            steps,
            seed ?: 0L,
            seed != null,
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

    /** Stops requests already issued, at their next graph call. Safe to call from any thread. */
    fun cancel() {
        // Under handleLock so close() cannot destroy the runtime while this call is using it.
        synchronized(handleLock) {
            val handle = nativeHandle
            if (handle != 0L) ScyllasBandNative.cancel(handle)
        }
    }

    @Synchronized
    override fun close() {
        val handle = synchronized(handleLock) { nativeHandle.also { nativeHandle = 0L } }
        if (handle != 0L) ScyllasBandNative.destroy(handle)
        bundleInfo = null
    }

    private fun chunkTextFromMetadata(metadataJson: String): String? =
        runCatching {
            JSONObject(metadataJson).optString("text").trim().takeIf { it.isNotEmpty() }
        }.getOrNull()

    private fun parseBundleInfo(voicesJson: String): ScyllasBandBundleInfo {
        val root = JSONObject(voicesJson)
        val voicesArray = root.getJSONArray("voices")
        val voices = buildList {
            for (index in 0 until voicesArray.length()) {
                val value = voicesArray.getJSONObject(index)
                val languagesArray = value.getJSONArray("languages")
                val languages = buildList {
                    for (languageIndex in 0 until languagesArray.length()) {
                        add(languagesArray.getString(languageIndex))
                    }
                }
                val id = value.getString("id")
                add(
                    ScyllasBandVoice(
                        id = id,
                        displayName = id.replaceFirstChar { it.uppercase() },
                        languages = languages,
                        defaultLanguage = value.optString("default_language", languages.first()),
                    ),
                )
            }
        }
        check(voices.isNotEmpty()) { "The Scylla's Band bundle lists no voices" }
        return ScyllasBandBundleInfo(
            backend = root.optString("backend", ScyllasBandNative.backend()),
            accelerator = root.optString("accelerator", "cpu"),
            sampleRate = root.getInt("sample_rate"),
            releaseId = root.optString("release_id"),
            voices = voices,
        )
    }

    companion object {
        private const val TAG = "ScyllasBand"

        fun create(context: Context): ScyllasBand = ScyllasBand(context.applicationContext)
    }
}
