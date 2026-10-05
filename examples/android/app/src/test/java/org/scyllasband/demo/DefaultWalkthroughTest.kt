package org.scyllasband.demo

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.scyllasband.android.ScyllasBandBundleInfo
import org.scyllasband.android.ScyllasBandDelivery
import org.scyllasband.android.ScyllasBandSegmentSettings
import org.scyllasband.android.ScyllasBandVoice

/** Parses the real data/walkthrough_demo.txt that the app bundles as its first-launch document. */
class DefaultWalkthroughTest {
    private val voices = ALL_VOICES.map { id ->
        val english = if (id in BRITISH_VOICES) "en_gb" else "en_us"
        ScyllasBandVoice(id, id.replaceFirstChar { it.uppercase() }, listOf(english) + NON_ENGLISH_LANGUAGES, english)
    }
    private val info = ScyllasBandBundleInfo(
        backend = "litert",
        accelerator = "cpu",
        sampleRate = 24_000,
        releaseId = "test",
        voices = voices,
    )
    private val defaults = ScyllasBandSegmentSettings("scylla", "en_us")
    private val source = File("../../../data/${DefaultWalkthrough.ASSET_NAME}").readText(Charsets.UTF_8)

    private fun snapshot() = DefaultWalkthrough.snapshot(source, defaults, info)

    @Test
    fun usesEveryVoiceAndLanguageOfTheDemo() {
        val segments = SpeakerDocument.segments(snapshot(), defaults)

        assertEquals(ALL_VOICES.toSet(), segments.map { it.settings.voiceId }.toSet())
        assertEquals("de", segments.single { it.settings.voiceId == "felix" }.settings.language)
        assertEquals("fr", segments.single { it.settings.voiceId == "stone" }.settings.language)
        assertEquals("es", segments.single { it.settings.voiceId == "rex" }.settings.language)
        assertEquals("it", segments.single { it.settings.voiceId == "max" }.settings.language)
        assertTrue(segments.any { it.settings.voiceId == "scylla" && it.settings.language == "vi" })
    }

    @Test
    fun convertsEachTaggedLineToAnEditablePointWithDelivery() {
        val snapshot = snapshot()

        assertEquals(source.lines().count { it.isNotBlank() }, snapshot.points.size)
        assertFalse(snapshot.text.contains('['))
        assertTrue(snapshot.points.first().settings.delivery.whisper)
        assertEquals(
            ScyllasBandDelivery(energy = 2.7f, tension = 3.1f, valence = 1.3f, assertiveness = 2.5f),
            snapshot.points.map { it.settings.delivery }.first { it.tension == 3.1f },
        )
        snapshot.points.forEach {
            val d = it.settings.delivery
            listOf(d.energy, d.tension, d.valence, d.assertiveness).forEach { v ->
                assertTrue(v in DELIVERY_UI_MIN..DELIVERY_UI_MAX)
            }
        }
        assertTrue(snapshot.points.map { it.settings.delivery }.distinct().size >= 8)
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
