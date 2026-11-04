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
)

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
