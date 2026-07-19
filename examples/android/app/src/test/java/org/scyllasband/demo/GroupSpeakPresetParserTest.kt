package org.scyllasband.demo

import org.scyllasband.android.ScyllasBandBundleInfo
import org.scyllasband.android.ScyllasBandSegmentSettings
import org.scyllasband.android.ScyllasBandVoice
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class GroupSpeakPresetParserTest {
    private val info = ScyllasBandBundleInfo(
        modelName = "test",
        sampleRate = 24_000,
        voices = listOf(
            ScyllasBandVoice("ariadne", "Ariadne", listOf("en_us", "en_gb", "es", "it"), "en_us"),
            ScyllasBandVoice("ink", "Ink", listOf("en_gb", "en_us", "es", "it"), "en_gb"),
        ),
        affectAxes = listOf("calm", "joy", "anger"),
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
    fun parsesSingleEmotionTag() {
        val snapshot = GroupSpeakPresetParser.parse(
            "[ink:en_gb:joy=0.75] Lovely.",
            defaults,
            info,
        )

        assertEquals("joy", snapshot.points.single().settings.emotion)
        assertEquals(0.75f, snapshot.points.single().settings.emotionStrength)
    }

    @Test
    fun leavesPlainTextWithoutPoints() {
        val snapshot = GroupSpeakPresetParser.parse("A plain example.\n", defaults, info)

        assertEquals("A plain example.", snapshot.text)
        assertTrue(snapshot.points.isEmpty())
    }
}
