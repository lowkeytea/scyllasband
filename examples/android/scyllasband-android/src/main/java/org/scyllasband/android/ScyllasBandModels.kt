package org.scyllasband.android

data class ScyllasBandVoice(
    val id: String,
    val displayName: String,
    val languages: List<String>,
    val defaultLanguage: String,
)

data class ScyllasBandBundleInfo(
    val modelName: String,
    val sampleRate: Int,
    val voices: List<ScyllasBandVoice>,
    val affectAxes: List<String>,
)

data class ScyllasBandSegmentSettings(
    val voiceId: String,
    val language: String,
    val emotion: String? = null,
    val emotionStrength: Float = 0f,
    val emotionCfg: Float = 1f,
) {
    init {
        require(voiceId.isNotBlank()) { "voiceId must not be blank" }
        require(language.isNotBlank()) { "language must not be blank" }
        require(emotionStrength in 0f..1f) { "emotionStrength must be within [0, 1]" }
        require(emotionCfg.isFinite() && emotionCfg >= 0f) { "emotionCfg must be finite and non-negative" }
    }

    fun affectSpec(): String? {
        val axis = emotion?.trim()?.lowercase()?.takeIf { it.isNotEmpty() }
        return axis?.let { "$it=${emotionStrength.coerceIn(0f, 1f)}" }
    }
}
