package org.scyllasband.demo

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import org.scyllasband.android.ScyllasBandBundleInfo
import org.scyllasband.android.ScyllasBandDelivery
import org.scyllasband.android.ScyllasBandSegmentSettings
import org.scyllasband.android.ScyllasBandVoice

class GroupSpeakPresetParserTest {
    private val info = ScyllasBandBundleInfo(
        backend = "litert",
        accelerator = "cpu",
        sampleRate = 24_000,
        releaseId = "test",
        voices = listOf(
            ScyllasBandVoice("ariadne", "Ariadne", listOf("en_us", "es", "it"), "en_us"),
            ScyllasBandVoice("ink", "Ink", listOf("en_gb", "es", "it"), "en_gb"),
        ),
    )
    private val defaults = ScyllasBandSegmentSettings("ariadne", "en_us")

    @Test
    fun parsesVoiceAndInlineLanguageTagsIntoPoints() {
        val snapshot = GroupSpeakPresetParser.parse(
            "[ariadne:en] Hello, [es] hola.\n[ink:en_gb] Quite right.",
            defaults,
            info,
        )

        assertEquals(" Hello,  hola.\n Quite right.", snapshot.text)
        assertEquals(3, snapshot.points.size)
        assertEquals(SpeakerPoint(0, defaults), snapshot.points[0])
        assertEquals("es", snapshot.points[1].settings.language)
        assertEquals("ink", snapshot.points[2].settings.voiceId)
        assertEquals("en_gb", snapshot.points[2].settings.language)
    }

    @Test
    fun parsesDeliveryAxesAndWhisper() {
        val settings = GroupSpeakPresetParser.parse(
            "[ink:en_gb:energy=2.5,tension=1.2,whisper=on] Lovely.",
            defaults,
            info,
        ).points.single().settings

        assertEquals(ScyllasBandDelivery(energy = 2.5f, tension = 1.2f, whisper = true), settings.delivery)
    }

    @Test
    fun clampsTaggedAxesToTheAppRange() {
        val settings = GroupSpeakPresetParser.parse(
            "[ink:en_gb:energy=3.6,tension=0.1,valence=4,assertiveness=3] Loud.",
            defaults,
            info,
        ).points.single().settings

        assertEquals(ScyllasBandDelivery(energy = 3.1f, tension = 0.5f, valence = 3.1f, assertiveness = 3f), settings.delivery)
    }

    @Test
    fun languageTagKeepsActiveVoiceAndDelivery() {
        val points = GroupSpeakPresetParser.parse(
            "[ariadne:en_us:energy=2.3,valence=2.5] Good to see you. [es] Me alegra verte.",
            defaults,
            info,
        ).points

        val expected = ScyllasBandDelivery(energy = 2.3f, valence = 2.5f)
        assertEquals(expected, points[0].settings.delivery)
        assertEquals(ScyllasBandSegmentSettings("ariadne", "es", expected), points[1].settings)
    }

    @Test
    fun newDeliveryTagResetsUnspecifiedAxesToNeutral() {
        val points = GroupSpeakPresetParser.parse(
            "[ariadne:en_us:energy=3.5,whisper=on] One. [ariadne:en_us:valence=3] Two.",
            defaults,
            info,
        ).points

        assertEquals(ScyllasBandDelivery(valence = 3f), points[1].settings.delivery)
    }

    @Test
    fun voiceOnlyTagKeepsDeliveryAndFallsBackToTheVoiceDefaultLanguage() {
        val points = GroupSpeakPresetParser.parse(
            "[ariadne:en_us:energy=3] Hi. [ink] Hello.",
            defaults,
            info,
        ).points

        assertEquals("ink", points[1].settings.voiceId)
        assertEquals("en_gb", points[1].settings.language)
        assertEquals(ScyllasBandDelivery(energy = 3f), points[1].settings.delivery)
    }

    @Test
    fun everyLineStartsFromTheDefaults() {
        val points = GroupSpeakPresetParser.parse(
            "[ink:en_gb:whisper=on] Psst.\n[es] Hola.",
            defaults,
            info,
        ).points

        assertEquals(SpeakerPoint(0, ScyllasBandSegmentSettings("ariadne", "es")), points[1].copy(offset = 0))
    }

    @Test
    fun rejectsBadTags() {
        listOf(
            "[nobody:en_us] x",
            "[ink:fr] x",
            "[ink:en_gb:energy=5] x",
            "[ink:en_gb:mood=2] x",
            "[ink:en_gb:whisper=maybe] x",
            "[ink:en_gb:energy=1,energy=2] x",
            "[ink:en_gb:energy=auto] x",
        ).forEach { source ->
            assertThrows(source, RuntimeException::class.java) { GroupSpeakPresetParser.parse(source, defaults, info) }
        }
    }

    @Test
    fun parseDeliveryAcceptsNeutralAndBlank() {
        assertEquals(ScyllasBandDelivery(), GroupSpeakPresetParser.parseDelivery("neutral"))
        assertEquals(ScyllasBandDelivery(), GroupSpeakPresetParser.parseDelivery(" "))
    }

    @Test
    fun leavesPlainTextWithoutPoints() {
        val snapshot = GroupSpeakPresetParser.parse("A plain example.\n", defaults, info)

        assertEquals("A plain example.", snapshot.text)
        assertTrue(snapshot.points.isEmpty())
    }
}
