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

    /** The phrase plan of [scoreJson] (seconds, samples, register_shift, phrases with text and timing), without singing it. */
    @Synchronized
    fun singPlan(
        scoreJson: String,
        voiceId: String,
        language: String? = null,
        options: ScyllasBandSingOptions = ScyllasBandSingOptions(),
    ): JSONObject {
        requireSinging(scoreJson, voiceId, language)
        return JSONObject(
            ScyllasBandNative.singPlanJson(nativeHandle, scoreJson, voiceId, language, options.transpose, options.useRegister),
        )
    }

    /**
     * Sings [scoreJson] ({"text", "bpm", "style", "notes": [[label, note, beats], ...]}) phrase by phrase.
     * [onPhraseStarted] reports the phrase about to be sung; [onPhraseAudio] receives its PCM and its place on the song's
     * timeline. A callback returns false to stop. Returns false when the request was stopped by a callback or [cancel].
     * [ScyllasBandSingOptions.decoder] VOCOS is the fast ("turbo") option. Blocking: call it off the main thread.
     */
    @Synchronized
    fun singStreaming(
        scoreJson: String,
        voiceId: String,
        language: String? = null,
        options: ScyllasBandSingOptions = ScyllasBandSingOptions(),
        onPhraseStarted: (phraseIndex: Int, phraseCount: Int, phraseText: String?) -> Boolean = { _, _, _ -> true },
        onPhraseAudio: (phrase: ScyllasBandSungPhrase) -> Boolean,
    ): Boolean {
        val info = requireSinging(scoreJson, voiceId, language)
        options.decoder.id?.let { decoder ->
            require(decoder in info.singing.decoders) { "This bundle has no $decoder decoder (it has ${info.singing.decoders})" }
        }
        return ScyllasBandNative.singStreaming(
            nativeHandle,
            scoreJson,
            voiceId,
            language,
            options.decoder.id,
            options.steps,
            options.guidance ?: 0f,
            options.transpose,
            options.useRegister,
            options.seed ?: 0L,
            options.seed != null,
            object : ScyllasBandNativeSingListener {
                override fun onPhraseStarted(phraseIndex: Int, phraseCount: Int, metadataJson: String?): Boolean =
                    onPhraseStarted(phraseIndex, phraseCount, metadataJson?.let(::chunkTextFromMetadata))

                override fun onPhraseAudio(samples: FloatArray, phraseIndex: Int, phraseCount: Int, metadataJson: String?): Boolean {
                    val metadata = metadataJson?.let { runCatching { JSONObject(it) }.getOrNull() }
                    return onPhraseAudio(
                        ScyllasBandSungPhrase(
                            samples = samples,
                            phraseIndex = phraseIndex,
                            phraseCount = phraseCount,
                            startSample = metadata?.optInt("start_sample", 0) ?: 0,
                            songSamples = metadata?.optInt("song_samples", 0) ?: 0,
                            text = metadata?.optString("text").orEmpty(),
                        ),
                    )
                }
            },
        )
    }

    /** Sings the whole score into one waveform at [ScyllasBandBundleInfo.sampleRate]; null when stopped by [cancel]. */
    fun sing(
        scoreJson: String,
        voiceId: String,
        language: String? = null,
        options: ScyllasBandSingOptions = ScyllasBandSingOptions(),
    ): FloatArray? {
        val phrases = mutableListOf<ScyllasBandSungPhrase>()
        val finished = singStreaming(scoreJson, voiceId, language, options) { phrase -> phrases += phrase; true }
        if (!finished) return null
        return joinSong(phrases)
    }

    /** How [scoreJson] sits in the voice's range after the transpose and the octave register; runs no graph. */
    fun singFit(
        scoreJson: String,
        voiceId: String,
        language: String? = null,
        options: ScyllasBandSingOptions = ScyllasBandSingOptions(),
    ): JSONObject {
        requireSinging(scoreJson, voiceId, language)
        val handle = synchronized(handleLock) { nativeHandle }
        return JSONObject(ScyllasBandNative.singFitJson(handle, scoreJson, voiceId, language, options.transpose, options.useRegister))
    }

    /** [scoreJson] as plain tones at the pitch it will be sung, at [ScyllasBandBundleInfo.sampleRate]; runs no graph. */
    fun preview(scoreJson: String, voiceId: String, options: ScyllasBandSingOptions = ScyllasBandSingOptions()): FloatArray {
        requireSinging(scoreJson, voiceId, null)
        val handle = synchronized(handleLock) { nativeHandle }
        return ScyllasBandNative.singPreview(handle, scoreJson, voiceId, options.transpose, options.useRegister)
    }

    /** Loads the graphs a short sung phrase needs with [decoder], so the first song starts quickly. */
    @Synchronized
    fun warmUpSinging(voiceId: String? = null, decoder: ScyllasBandSingingDecoder = ScyllasBandSingingDecoder.DEFAULT) {
        checkNotNull(bundleInfo) { "Call initialize() before singing" }
        ScyllasBandNative.warmupSinging(nativeHandle, voiceId, decoder.id)
    }

    private fun requireSinging(scoreJson: String, voiceId: String, language: String?): ScyllasBandBundleInfo {
        require(scoreJson.isNotBlank()) { "The score must not be blank" }
        val info = checkNotNull(bundleInfo) { "Call initialize() before singing" }
        check(info.singing.supported) { "This bundle cannot sing: it needs a v4 bundle (graph contract scyllasband_measured_controls_v4)" }
        val voice = info.voices.firstOrNull { it.id == voiceId } ?: error("Unknown Scylla's Band voice: $voiceId")
        require(language == null || language in voice.languages) { "$language is not available for $voiceId" }
        return info
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
        val delivery = root.optJSONObject("delivery")
        val defaults = delivery?.optJSONObject("defaults")
        val ranges = delivery?.optJSONObject("ranges")
        return ScyllasBandBundleInfo(
            backend = root.optString("backend", ScyllasBandNative.backend()),
            accelerator = root.optString("accelerator", "cpu"),
            sampleRate = root.getInt("sample_rate"),
            releaseId = root.optString("release_id"),
            voices = voices.map { voice ->
                val values = defaults?.optJSONObject(voice.id) ?: return@map voice
                voice.copy(
                    deliveryDefaults = ScyllasBandDelivery(
                        intensity = values.optDouble("intensity", 0.0).toFloat(),
                        pace = values.optDouble("pace", 0.0).toFloat(),
                        pitch = values.optDouble("pitch", 0.0).toFloat(),
                    ),
                )
            },
            deliveryRanges = ScyllasBandDelivery.DEFAULT_RANGES.mapValues { (name, fallback) ->
                val range = ranges?.optJSONArray(name)
                if (range == null || range.length() != 2) fallback else range.getDouble(0).toFloat()..range.getDouble(1).toFloat()
            },
            graphContract = root.optString("graph_contract"),
            singing = parseSingingInfo(root.optJSONObject("singing")),
        )
    }

    private fun parseSingingInfo(singing: JSONObject?): ScyllasBandSingingInfo {
        if (singing == null) return ScyllasBandSingingInfo()
        fun strings(name: String): List<String> {
            val array = singing.optJSONArray(name) ?: return emptyList()
            return (0 until array.length()).map { array.getString(it) }
        }
        val centers = singing.optJSONObject("voice_midi_center")
        return ScyllasBandSingingInfo(
            supported = singing.optBoolean("supported", false),
            styles = strings("styles"),
            decoders = strings("decoders"),
            defaultDecoder = singing.optString("default_decoder", "vocos"),
            steps = singing.optInt("steps", 0),
            sampler = singing.optString("sampler"),
            guidance = singing.optDouble("guidance", 1.0).toFloat(),
            voiceMidiCenters = centers?.keys()?.asSequence()?.associateWith { centers.getDouble(it).toFloat() } ?: emptyMap(),
            registerBelow = singing.optJSONArray("register_band")?.optDouble(0, -3.0)?.toFloat() ?: -3f,
            registerAbove = singing.optJSONArray("register_band")?.optDouble(1, 5.0)?.toFloat() ?: 5f,
            maxPhraseSeconds = singing.optDouble("max_phrase_seconds", 10.0).toFloat(),
        )
    }

    companion object {
        private const val TAG = "ScyllasBand"

        fun create(context: Context): ScyllasBand = ScyllasBand(context.applicationContext)

        /**
         * Checks [scoreJson] without a runtime and returns it normalized after [transpose] semitones: notes with their MIDI,
         * word, part and times, words, range and the lines an editor shows. Throws [ScyllasBandScoreException] (naming the
         * note at fault when there is one) for a score the runtime refuses.
         */
        fun parseScore(scoreJson: String, transpose: Int = 0): JSONObject {
            val result = JSONObject(ScyllasBandNative.scoreParseJson(scoreJson, transpose))
            if (result.has("error")) {
                val index = if (result.isNull("note_index")) null else result.optInt("note_index")
                throw ScyllasBandScoreException(result.getString("error"), index)
            }
            return result
        }
    }
}
