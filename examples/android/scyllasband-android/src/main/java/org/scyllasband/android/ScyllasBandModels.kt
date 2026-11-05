package org.scyllasband.android

import java.util.Locale

data class ScyllasBandVoice(
    val id: String,
    val displayName: String,
    val languages: List<String>,
    val defaultLanguage: String,
    /** What a request that leaves a control out uses for this voice (0 = the voice's own average). */
    val deliveryDefaults: ScyllasBandDelivery = ScyllasBandDelivery(intensity = 0f, pace = 0f, pitch = 0f),
)

data class ScyllasBandBundleInfo(
    val backend: String,
    val accelerator: String,
    val sampleRate: Int,
    val releaseId: String,
    val voices: List<ScyllasBandVoice>,
    /** The range the runtime clamps each control into: intensity and pace in standard deviations, pitch in semitones. */
    val deliveryRanges: Map<String, ClosedFloatingPointRange<Float>> = ScyllasBandDelivery.DEFAULT_RANGES,
    /** "scyllasband_measured_controls_v3" (speech) or "_v4" (speech and singing). */
    val graphContract: String = "",
    val singing: ScyllasBandSingingInfo = ScyllasBandSingingInfo(),
)

/** The waveform decoder a sung score renders through. */
enum class ScyllasBandSingingDecoder(internal val id: String?) {
    /** The bundle's singing decoder: BigVGAN where the bundle carries it, else Vocos. */
    DEFAULT(null),

    /** BigVGAN v2, fine-tuned on the voices: the singing decoder, cleanest on sustained notes, and the slowest. */
    BIGVGAN("bigvgan"),

    /** Vocos, the speech decoder ("turbo"): much faster, with some harmonic artefacts on sustained notes in some voices. */
    VOCOS("vocos"),
}

/** What a bundle can sing (v4 bundles); [supported] is false otherwise. */
data class ScyllasBandSingingInfo(
    val supported: Boolean = false,
    /** Score styles: "pop" (also "sung") and "ballad". */
    val styles: List<String> = emptyList(),
    /** Decoders with graphs in this bundle: "bigvgan", "vocos". */
    val decoders: List<String> = emptyList(),
    /** What [ScyllasBandSingingDecoder.DEFAULT] uses. */
    val defaultDecoder: String = "vocos",
    val steps: Int = 0,
    val sampler: String = "",
    val guidance: Float = 1f,
    /** The MIDI pitch each voice sings around; a score is moved by octaves next to it. */
    val voiceMidiCenters: Map<String, Float> = emptyMap(),
) {
    val hasBigVgan: Boolean get() = "bigvgan" in decoders
}

/**
 * How to sing a score. [steps] 0 uses the bundle's singing steps (8); [guidance] null uses the bundle's (2), 1 turns it off;
 * [transpose] adds semitones to the whole score before [useRegister] moves it by octaves into the voice's sung range; phrase i
 * draws its noise from [seed] + i.
 */
data class ScyllasBandSingOptions(
    val decoder: ScyllasBandSingingDecoder = ScyllasBandSingingDecoder.DEFAULT,
    val steps: Int = 0,
    val guidance: Float? = null,
    val transpose: Int = 0,
    val useRegister: Boolean = true,
    val seed: Long? = null,
) {
    init {
        require(steps >= 0) { "steps must not be negative" }
        require(guidance == null || (guidance.isFinite() && guidance > 0f)) { "guidance must be positive and finite" }
    }
}

/**
 * One sung phrase: mono float PCM that starts [offsetSamples] into the song (the gap after the previous phrase is silence; a
 * phrase can overlap the previous one's release, in which case the samples add). The song is [songSamples] long.
 */
class ScyllasBandSungPhrase(
    val samples: FloatArray,
    val phraseIndex: Int,
    val phraseCount: Int,
    val offsetSamples: Int,
    val songSamples: Int,
    val text: String,
)

/** Places sung phrases on the song's timeline, adding where they overlap, clipped to [-1, 1]. */
fun assembleSong(phrases: List<ScyllasBandSungPhrase>, songSamples: Int): FloatArray {
    val song = FloatArray(songSamples)
    for (phrase in phrases) {
        val count = minOf(phrase.samples.size, songSamples - phrase.offsetSamples)
        for (i in 0 until maxOf(0, count)) song[phrase.offsetSamples + i] += phrase.samples[i]
    }
    for (i in song.indices) song[i] = song[i].coerceIn(-1f, 1f)
    return song
}

/** Where the LiteRT graphs run. ONNX Runtime builds always run on the CPU. */
enum class ScyllasBandAccelerator(internal val code: Int) {
    CPU(0),

    /** LiteRT GPU, with the CPU for unsupported ops. */
    GPU(1),

    /** The GPU when the graphs compile for it, else the CPU. */
    AUTO(2),
}

/**
 * How a voice delivers a line. Each control is optional; null takes the voice's default.
 *
 * - [intensity]: calm (negative) to intense (positive), in the voice's own standard deviations. The emotion itself comes from
 *   the text.
 * - [pace]: slower (negative) to faster (positive), words and pauses together, in the voice's own standard deviations.
 * - [pitch]: a shift of the voice's pitch in semitones.
 *
 * The released model supports -2 to 2 for each ([ScyllasBandBundleInfo.deliveryRanges]). [spec] is the ABI delivery string.
 */
data class ScyllasBandDelivery(
    val intensity: Float? = null,
    val pace: Float? = null,
    val pitch: Float? = null,
) {
    init {
        require(listOfNotNull(intensity, pace, pitch).all { it.isFinite() && it in MIN..MAX }) {
            "Delivery controls must be finite and within [$MIN, $MAX]"
        }
    }

    /** "intensity=1,pace=-0.5,pitch=0" for the controls that are set; "" (every voice default) when none is. */
    fun spec(): String = listOf("intensity" to intensity, "pace" to pace, "pitch" to pitch)
        .filter { it.second != null }
        .joinToString(",") { (name, value) -> "$name=${format(value!!)}" }

    companion object {
        const val MIN = -2f
        const val MAX = 2f
        val DEFAULT_RANGES: Map<String, ClosedFloatingPointRange<Float>> =
            mapOf("intensity" to MIN..MAX, "pace" to MIN..MAX, "pitch" to MIN..MAX)

        private fun format(value: Float): String =
            String.format(Locale.US, "%.2f", value).trimEnd('0').trimEnd('.').let { if (it == "-0") "0" else it }
    }
}

data class ScyllasBandSegmentSettings(
    val voiceId: String,
    val language: String,
    val delivery: ScyllasBandDelivery = ScyllasBandDelivery(),
) {
    init {
        require(voiceId.isNotBlank()) { "voiceId must not be blank" }
        require(language.isNotBlank()) { "language must not be blank" }
    }
}
