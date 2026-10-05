package org.scyllasband.demo

import org.scyllasband.android.ScyllasBandBundleInfo
import org.scyllasband.android.ScyllasBandDelivery
import org.scyllasband.android.ScyllasBandSegmentSettings

/**
 * Converts `[voice:language:delivery]` tagged text into plain text plus speaker points.
 *
 * - `[voice:language:energy=2.3,valence=2.5]` sets voice, language and delivery. An empty voice or language
 *   part keeps the active one. A delivery part replaces the active delivery; unspecified axes are neutral.
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

    /** Parses `energy=2.5,tension=2,whisper=on` into a delivery; omitted axes are neutral. */
    fun parseDelivery(spec: String): ScyllasBandDelivery {
        val clean = spec.trim().lowercase()
        if (clean.isEmpty() || clean == "neutral") return ScyllasBandDelivery()
        var delivery = ScyllasBandDelivery()
        val seen = mutableSetOf<String>()
        for (term in clean.split(',')) {
            val pair = term.split('=', limit = 2)
            require(pair.size == 2 && pair[0].isNotBlank()) { "Invalid delivery term '$term'" }
            val axis = pair[0].trim()
            val raw = pair[1].trim()
            require(seen.add(axis)) { "Delivery axis '$axis' is repeated" }
            require(raw != "auto") { "'auto' delivery is not supported by this editor ('$axis')" }
            if (axis == "whisper") {
                require(raw == "on" || raw == "off") { "whisper must be on or off" }
                delivery = delivery.copy(whisper = raw == "on")
                continue
            }
            val value = raw.toFloatOrNull()
            require(value != null && value.isFinite() && value in ScyllasBandDelivery.MIN..ScyllasBandDelivery.MAX) {
                "Delivery '$axis' must be within [0, 4]"
            }
            delivery = when (axis) {
                "energy" -> delivery.copy(energy = value)
                "tension" -> delivery.copy(tension = value)
                "valence" -> delivery.copy(valence = value)
                "assertiveness" -> delivery.copy(assertiveness = value)
                else -> error("Unknown delivery axis '$axis'")
            }
        }
        return delivery
    }

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
            parts.getOrNull(2)?.takeIf { it.isNotBlank() }?.let {
                // Tagged documents may exceed the app's 0.5-3.1 range; clamp (sparse training data beyond about 3).
                delivery = parseDelivery(it).coerceToUiRange()
            }
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
