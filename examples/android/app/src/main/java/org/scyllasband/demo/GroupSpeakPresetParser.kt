package org.scyllasband.demo

import org.scyllasband.android.ScyllasBandBundleInfo
import org.scyllasband.android.ScyllasBandDelivery
import org.scyllasband.android.ScyllasBandSegmentSettings

/**
 * Converts `[voice:language:delivery]` tagged text into plain text plus speaker points.
 *
 * - `[voice:language:intensity=1,pace=-0.5]` sets voice, language and delivery. An empty voice or language
 *   part keeps the active one. A delivery part replaces the active delivery; unspecified controls take the voice's own.
 * - `[es]` (a language) keeps the active voice and delivery; `[ink]` (anything else) switches voice only.
 * - Every line starts again from the defaults.
 */
object GroupSpeakPresetParser {
    private val tagPattern = Regex("\\[([-A-Za-z0-9_.,:=]+)]")
    private val languageTags = setOf("en", "en_us", "en_gb", "es", "it", "fr", "de", "vi")

    fun parse(
        source: String,
        defaults: ScyllasBandSegmentSettings,
        bundleInfo: ScyllasBandBundleInfo,
    ): SpeakerDocumentSnapshot {
        val output = StringBuilder()
        val points = mutableListOf<SpeakerPoint>()
        val lines = source.lines()
        lines.forEachIndexed { lineIndex, line ->
            var active = defaults
            var cursor = 0
            tagPattern.findAll(line).forEach { match ->
                output.append(line, cursor, match.range.first)
                active = parseTag(match.groupValues[1], active, bundleInfo)
                points += SpeakerPoint(output.length, active)
                cursor = match.range.last + 1
            }
            output.append(line, cursor, line.length)
            if (lineIndex < lines.lastIndex) output.append('\n')
        }
        return SpeakerDocumentSnapshot(output.toString().trimEnd(), points)
    }

    /**
     * Parses `intensity=1,pace=-0.5,pitch=0` into a delivery; omitted controls take the voice's own. Values outside -2 to 2
     * are clamped, as the runtime would.
     */
    fun parseDelivery(spec: String): ScyllasBandDelivery {
        val clean = spec.trim().lowercase()
        if (clean.isEmpty() || clean == "default" || clean == "neutral") return ScyllasBandDelivery()
        var delivery = ScyllasBandDelivery()
        val seen = mutableSetOf<String>()
        for (term in clean.split(',')) {
            val pair = term.split('=', limit = 2)
            require(pair.size == 2 && pair[0].isNotBlank()) { "Invalid delivery term '$term'" }
            val control = pair[0].trim()
            val raw = pair[1].trim()
            require(seen.add(control)) { "Delivery control '$control' is repeated" }
            require(control !in V2_AXES) {
                "This model takes intensity, pace and pitch ('$control' belonged to the v2 models)"
            }
            if (raw == "default") continue
            val value = raw.toFloatOrNull()
            require(value != null && value.isFinite()) { "Delivery '$control' must be a number" }
            val range = DELIVERY_UI_RANGES[control] ?: error("Unknown delivery control '$control'; use intensity, pace or pitch")
            delivery = when (control) {
                "intensity" -> delivery.copy(intensity = value.coerceIn(range))
                "pace" -> delivery.copy(pace = value.coerceIn(range))
                else -> delivery.copy(pitch = value.coerceIn(range))
            }
        }
        return delivery
    }

    private val V2_AXES = setOf("energy", "tension", "valence", "assertiveness", "whisper", "emotion")

    private fun parseTag(
        label: String,
        current: ScyllasBandSegmentSettings,
        bundleInfo: ScyllasBandBundleInfo,
    ): ScyllasBandSegmentSettings {
        var voiceId = current.voiceId
        var language = current.language
        var delivery = current.delivery
        var explicitLanguage = false
        val clean = label.trim().lowercase()

        if (':' in clean) {
            val parts = clean.split(':', limit = 3)
            if (parts[0].isNotBlank()) voiceId = parts[0]
            if (parts.getOrElse(1) { "" }.isNotBlank()) {
                language = parts[1]
                explicitLanguage = true
            }
            parts.getOrNull(2)?.takeIf { it.isNotBlank() }?.let { delivery = parseDelivery(it) }
        } else if (clean in languageTags) {
            language = clean
            explicitLanguage = true
        } else if (clean.isNotBlank()) {
            voiceId = clean
        }

        val voice = bundleInfo.voices.firstOrNull { it.id == voiceId }
            ?: error("Preset refers to unknown voice '$voiceId'")
        language = when {
            language == "en" -> voice.defaultLanguage
            language in voice.languages -> language
            explicitLanguage -> error("Preset language '$language' is not available for ${voice.id}")
            else -> voice.defaultLanguage
        }
        return ScyllasBandSegmentSettings(voice.id, language, delivery)
    }
}
