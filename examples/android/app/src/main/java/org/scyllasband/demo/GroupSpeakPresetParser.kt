package org.scyllasband.demo

import org.scyllasband.android.ScyllasBandDelivery
import org.scyllasband.android.ScyllasBandBundleInfo
import org.scyllasband.android.ScyllasBandSegmentSettings

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

    private fun parseTag(
        label: String,
        current: ScyllasBandSegmentSettings,
        bundleInfo: ScyllasBandBundleInfo,
    ): ScyllasBandSegmentSettings {
        var voiceId = current.voiceId
        var language = current.language
        var delivery = current.delivery
        var emotion = current.emotion
        var strength = current.emotionStrength
        val clean = label.trim().lowercase()

        if (':' in clean) {
            val parts = clean.split(':', limit = 3)
            if (parts[0].isNotBlank()) voiceId = parts[0]
            if (parts.getOrElse(1) { "" }.isNotBlank()) language = parts[1]
            parts.getOrNull(2)?.takeIf { it.isNotBlank() }?.let { affect ->
                if (bundleInfo.deliveryEnabled) {
                    var d = ScyllasBandDelivery()
                    val seen = mutableSetOf<String>()
                    for (term in affect.split(',')) {
                        val pair = term.split('=', limit = 2)
                        require(pair.size == 2 && seen.add(pair[0])) { "Invalid delivery tag" }
                        val axis = pair[0]; val raw = pair[1]
                        if (axis == "whisper") {
                            require(raw == "on" || raw == "off") { "Whisper must be on or off" }
                            d = d.copy(whisper = raw == "on")
                        } else {
                            val v = raw.toFloatOrNull() ?: error("Invalid delivery value")
                            d = when (axis) {
                                "energy" -> d.copy(energy = v)
                                "tension" -> d.copy(tension = v)
                                "valence" -> d.copy(valence = v)
                                "assertiveness" -> d.copy(assertiveness = v)
                                else -> error("Unknown measured delivery axis '$axis'")
                            }
                        }
                    }
                    delivery = d; emotion = null; strength = 0f
                    return@let
                }
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
            delivery = delivery,
            emotionCfg = if (bundleInfo.deliveryEnabled) 1f else current.emotionCfg,
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
