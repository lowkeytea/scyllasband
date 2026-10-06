package org.scyllasband.android

import java.util.Locale

data class ScyllasBandVoice(
    val id: String,
    val displayName: String,
    val languages: List<String>,
    val defaultLanguage: String,
)

data class ScyllasBandBundleInfo(
    val backend: String,
    val accelerator: String,
    val sampleRate: Int,
    val releaseId: String,
    val voices: List<ScyllasBandVoice>,
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
 * Four continuous delivery axes on 0-4 (2 is neutral). [spec] is the ABI delivery string. [whisper] is no longer supported by
 * the runtime and is ignored.
 */
data class ScyllasBandDelivery(
    val energy: Float = NEUTRAL,
    val tension: Float = NEUTRAL,
    val valence: Float = NEUTRAL,
    val assertiveness: Float = NEUTRAL,
    val whisper: Boolean = false,
) {
    init {
        require(listOf(energy, tension, valence, assertiveness).all { it.isFinite() && it in MIN..MAX }) {
            "Delivery axes must be finite and within [$MIN, $MAX]"
        }
    }

    fun spec(): String = String.format(
        Locale.US,
        "energy=%s,tension=%s,valence=%s,assertiveness=%s",
        format(energy),
        format(tension),
        format(valence),
        format(assertiveness),
    )

    companion object {
        const val MIN = 0f
        const val MAX = 4f
        const val NEUTRAL = 2f

        private fun format(value: Float): String =
            String.format(Locale.US, "%.2f", value).trimEnd('0').trimEnd('.')
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
