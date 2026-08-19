package org.scyllasband.demo

import org.scyllasband.android.ScyllasBandBundleInfo
import org.scyllasband.android.ScyllasBandSegmentSettings
import org.scyllasband.android.ScyllasBandVoice
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class DefaultWalkthroughTest {
    private val voices = ALL_VOICES.map { id ->
        val english = if (id in BRITISH_VOICES) "en_gb" else "en_us"
        ScyllasBandVoice(id, id.replaceFirstChar { it.uppercase() }, listOf(english) + NON_ENGLISH_LANGUAGES, english)
    }
    private val info = ScyllasBandBundleInfo(
        modelName = "test",
        sampleRate = 24_000,
        voices = voices,
        affectAxes = listOf("calm", "joy", "anger", "sadness", "sarcasm", "whisper"),
    )
    private val defaults = ScyllasBandSegmentSettings("gwen", "en_us")

    @Test
    fun launchesWithTheRequestedSpeakersAndLanguages() {
        val snapshot = DefaultWalkthrough.snapshot(defaults, info)
        val segments = SpeakerDocument.segments(snapshot, defaults)

        assertEquals(
            listOf(
                "gwen", "gwen", "gwen", "ink", "scylla", "tuesday", "felix",
                "stone", "scylla", "rex", "max", "ariadne", "orpheus",
            ),
            segments.map { it.settings.voiceId },
        )
        assertEquals("de", segments.single { it.settings.voiceId == "felix" }.settings.language)
        assertEquals("fr", segments.single { it.settings.voiceId == "stone" }.settings.language)
        assertEquals("es", segments.single { it.settings.voiceId == "rex" }.settings.language)
        assertEquals("it", segments.single { it.settings.voiceId == "max" }.settings.language)
        assertEquals(
            listOf("en_us", "vi"),
            segments.filter { it.settings.voiceId == "scylla" }.map { it.settings.language },
        )
        assertEquals(ALL_VOICES.toSet(), segments.map { it.settings.voiceId }.toSet())
    }

    @Test
    fun convertsTagsToEditablePointsWithVariedEmotionControls() {
        val snapshot = DefaultWalkthrough.snapshot(defaults, info)

        assertEquals(13, snapshot.points.size)
        assertFalse(snapshot.text.contains('['))
        assertEquals(
            info.affectAxes.toSet(),
            snapshot.points.mapNotNull { it.settings.emotion }.toSet(),
        )
        assertTrue(snapshot.points.map { it.settings.emotionStrength }.distinct().size >= 4)
        assertTrue(snapshot.points.map { it.settings.emotionCfg }.distinct().size >= 4)
    }

    @Test
    fun dialogueTouchesEverySupportedLanguage() {
        val text = DefaultWalkthrough.snapshot(defaults, info).text.lowercase()

        assertTrue("flüstere" in text)
        assertTrue("le français" in text)
        assertTrue("tiếng việt" in text)
        assertTrue("el español" in text)
        assertTrue("l'italiano" in text)
        assertTrue("cookies" in text)
    }

    private companion object {
        val ALL_VOICES = listOf(
            "ariadne", "felix", "gwen", "ink", "max",
            "orpheus", "rex", "scylla", "stone", "tuesday",
        )
        val BRITISH_VOICES = setOf("ink", "orpheus", "tuesday")
        val NON_ENGLISH_LANGUAGES = listOf("es", "it", "fr", "de", "vi")
    }
}
