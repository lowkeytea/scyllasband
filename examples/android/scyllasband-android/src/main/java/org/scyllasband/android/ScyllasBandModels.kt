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
    val deliveryEnabled: Boolean = false,
)

data class ScyllasBandDelivery(
    val energy: Float = 2f,
    val tension: Float = 2f,
    val valence: Float = 2f,
    val assertiveness: Float = 2f,
    val whisper: Boolean = false,
) {
    init {
        require(listOf(energy, tension, valence, assertiveness).all { it.isFinite() && it in 0f..4f }) {
            "Measured delivery values must be finite and within [0, 4]"
        }
    }
    fun spec(): String = "delivery:energy=$energy,tension=$tension,valence=$valence,assertiveness=$assertiveness,whisper=${if (whisper) "on" else "off"}"
}

data class ScyllasBandSegmentSettings(
    val voiceId: String,
    val language: String,
    val emotion: String? = null,
    val emotionStrength: Float = 0f,
    val emotionCfg: Float = 1f,
    val delivery: ScyllasBandDelivery? = null,
) {
    init {
        require(voiceId.isNotBlank()) { "voiceId must not be blank" }
        require(language.isNotBlank()) { "language must not be blank" }
        require(emotionStrength in 0f..1f) { "emotionStrength must be within [0, 1]" }
        require(emotionCfg.isFinite() && emotionCfg >= 0f) { "emotionCfg must be finite and non-negative" }
    }

    fun affectSpec(): String? {
        delivery?.let {
            require(emotion == null && emotionCfg == 1f) { "Measured delivery cannot be mixed with emotion or CFG" }
            return it.spec()
        }
        val axis = emotion?.trim()?.lowercase()?.takeIf { it.isNotEmpty() }
        return axis?.let { "$it=${emotionStrength.coerceIn(0f, 1f)}" }
    }
}
