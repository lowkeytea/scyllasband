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
    fun parsesDeliveryControls() {
        val settings = GroupSpeakPresetParser.parse(
            "[ink:en_gb:intensity=1,pace=-0.5] Lovely.",
            defaults,
            info,
        ).points.single().settings

        assertEquals(ScyllasBandDelivery(intensity = 1f, pace = -0.5f), settings.delivery)
    }

    @Test
    fun clampsTaggedControlsToTheSupportedRanges() {
        val settings = GroupSpeakPresetParser.parse(
            "[ink:en_gb:intensity=3,pace=-2.5,pitch=4] Loud.",
            defaults,
            info,
        ).points.single().settings

        assertEquals(ScyllasBandDelivery(intensity = 2f, pace = -2f, pitch = 2f), settings.delivery)
    }

    @Test
    fun languageTagKeepsActiveVoiceAndDelivery() {
        val points = GroupSpeakPresetParser.parse(
            "[ariadne:en_us:intensity=1,pitch=1] Good to see you. [es] Me alegra verte.",
            defaults,
            info,
        ).points

        val expected = ScyllasBandDelivery(intensity = 1f, pitch = 1f)
        assertEquals(expected, points[0].settings.delivery)
        assertEquals(ScyllasBandSegmentSettings("ariadne", "es", expected), points[1].settings)
    }

    @Test
    fun newDeliveryTagResetsUnspecifiedControlsToTheVoicesOwn() {
        val points = GroupSpeakPresetParser.parse(
            "[ariadne:en_us:intensity=1] One. [ariadne:en_us:pace=-1] Two.",
            defaults,
            info,
        ).points

        assertEquals(ScyllasBandDelivery(pace = -1f), points[1].settings.delivery)
    }

    @Test
    fun voiceOnlyTagKeepsDeliveryAndFallsBackToTheVoiceDefaultLanguage() {
        val points = GroupSpeakPresetParser.parse(
            "[ariadne:en_us:intensity=0.5] Hi. [ink] Hello.",
            defaults,
            info,
        ).points

        assertEquals("ink", points[1].settings.voiceId)
        assertEquals("en_gb", points[1].settings.language)
        assertEquals(ScyllasBandDelivery(intensity = 0.5f), points[1].settings.delivery)
    }

    @Test
    fun everyLineStartsFromTheDefaults() {
        val points = GroupSpeakPresetParser.parse(
            "[ink:en_gb:intensity=-1] Psst.\n[es] Hola.",
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
            "[ink:en_gb:energy=1] x",
            "[ink:en_gb:mood=2] x",
            "[ink:en_gb:whisper=on] x",
            "[ink:en_gb:pace=1,pace=2] x",
            "[ink:en_gb:pace=fast] x",
        ).forEach { source ->
            assertThrows(source, RuntimeException::class.java) { GroupSpeakPresetParser.parse(source, defaults, info) }
        }
    }

    @Test
    fun parseDeliveryAcceptsDefaultAndBlank() {
        assertEquals(ScyllasBandDelivery(), GroupSpeakPresetParser.parseDelivery("default"))
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
