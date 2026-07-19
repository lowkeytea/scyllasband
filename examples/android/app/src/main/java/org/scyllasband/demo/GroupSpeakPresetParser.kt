package org.scyllasband.demo

import org.scyllasband.android.ScyllasBandBundleInfo
import org.scyllasband.android.ScyllasBandSegmentSettings

object GroupSpeakPresetParser {
    private val tagPattern = Regex("\\[([-A-Za-z0-9_.,:=]+)]")
    private val languageTags = setOf("en", "en_us", "en_gb", "es", "it")

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

    private fun parseTag(
        label: String,
        current: ScyllasBandSegmentSettings,
        bundleInfo: ScyllasBandBundleInfo,
    ): ScyllasBandSegmentSettings {
        var voiceId = current.voiceId
        var language = current.language
        var emotion = current.emotion
        var strength = current.emotionStrength
        val clean = label.trim().lowercase()

        if (':' in clean) {
            val parts = clean.split(':', limit = 3)
            if (parts[0].isNotBlank()) voiceId = parts[0]
            if (parts.getOrElse(1) { "" }.isNotBlank()) language = parts[1]
            parts.getOrNull(2)?.takeIf { it.isNotBlank() }?.let { affect ->
                val terms = affect.split(',')
                require(terms.size == 1) {
                    "This Android editor supports one emotion per point; preset tag [$label] mixes several axes"
                }
                val (axis, value) = parseAffectTerm(terms.single(), bundleInfo)
                emotion = axis
                strength = value
            }
        } else if (clean in languageTags) {
            language = clean
        } else if (clean in bundleInfo.affectAxes) {
            emotion = clean
            strength = 1f
        } else if (clean.isNotBlank()) {
            voiceId = clean
        }

        val voice = bundleInfo.voices.firstOrNull { it.id == voiceId }
            ?: error("Preset refers to unknown voice '$voiceId'")
        language = when (language) {
            "en" -> voice.defaultLanguage
            in voice.languages -> language
            else -> error("Preset language '$language' is not available for ${voice.id}")
        }
        return current.copy(
            voiceId = voice.id,
            language = language,
            emotion = emotion,
            emotionStrength = strength,
        )
    }

    private fun parseAffectTerm(
        term: String,
        bundleInfo: ScyllasBandBundleInfo,
    ): Pair<String, Float> {
        val parts = term.split('=', limit = 2)
        require(parts.size == 2) { "Invalid preset emotion '$term'" }
        val axis = parts[0].trim()
        require(axis in bundleInfo.affectAxes) { "Unknown preset emotion '$axis'" }
        val value = parts[1].trim().toFloatOrNull()
        require(value != null && value.isFinite() && value in 0f..1f) {
            "Preset emotion strength for '$axis' must be within [0, 1]"
        }
        return axis to value
    }
}
