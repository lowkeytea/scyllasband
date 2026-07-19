package org.scyllasband.demo

import org.scyllasband.android.ScyllasBandBundleInfo
import org.scyllasband.android.ScyllasBandSegmentSettings
import org.scyllasband.android.ScyllasBandVoice
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class DefaultWalkthroughTest {
    private val voices = listOf("scylla", "max", "tuesday", "ink", "rex").map { id ->
        ScyllasBandVoice(id, id.replaceFirstChar { it.uppercase() }, LANGUAGES, "en_us")
    }
    private val info = ScyllasBandBundleInfo(
        modelName = "test",
        sampleRate = 24_000,
        voices = voices,
        affectAxes = listOf("calm", "joy", "anger", "sadness", "sarcasm", "questioning"),
    )
    private val defaults = ScyllasBandSegmentSettings("scylla", "en_us")

    @Test
    fun launchesWithTheRequestedSpeakersAndLanguages() {
        val snapshot = DefaultWalkthrough.snapshot(defaults, info)
        val segments = SpeakerDocument.segments(snapshot, defaults)

        assertEquals(
            listOf("scylla", "scylla", "scylla", "max", "tuesday", "rex", "ink", "max", "scylla"),
            segments.map { it.settings.voiceId },
        )
        assertEquals("es", segments.single { it.settings.voiceId == "rex" }.settings.language)
        assertEquals("it", segments.single { it.settings.voiceId == "ink" }.settings.language)
        assertEquals(
            setOf("scylla", "max", "tuesday", "ink", "rex"),
            segments.map { it.settings.voiceId }.toSet(),
        )
    }

    @Test
    fun convertsTagsToEditablePointsWithVariedEmotionControls() {
        val snapshot = DefaultWalkthrough.snapshot(defaults, info)

        assertEquals(9, snapshot.points.size)
        assertFalse(snapshot.text.contains('['))
        assertTrue(snapshot.points.mapNotNull { it.settings.emotion }.toSet().size >= 4)
        assertTrue(snapshot.points.map { it.settings.emotionStrength }.distinct().size >= 4)
        assertTrue(snapshot.points.map { it.settings.emotionCfg }.distinct().size >= 4)
    }

    @Test
    fun dialogueExplainsTagsAndContainsJokesInAllThreeLanguages() {
        val text = DefaultWalkthrough.snapshot(defaults, info).text.lowercase()

        assertTrue("full tag" in text)
        assertTrue("emotion" in text)
        assertTrue("c f g" in text)
        assertTrue("final joke" in text)
        assertTrue("un chiste" in text)
        assertTrue("una battuta" in text)
    }

    private companion object {
        val LANGUAGES = listOf("en_us", "en_gb", "es", "it")
    }
}
