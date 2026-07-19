package org.scyllasband.demo

import org.scyllasband.android.ScyllasBandSegmentSettings
import org.junit.Assert.assertEquals
import org.junit.Test

class SpeakerDocumentTest {
    private val default = ScyllasBandSegmentSettings("scylla", "en_us")
    private val spanishJoy = ScyllasBandSegmentSettings("ariadne", "es", "joy", 0.75f, 1.25f)
    private val italianAnger = ScyllasBandSegmentSettings("ink", "it", "anger", 0.5f, 1.5f)

    @Test
    fun pointAppliesUntilTheNextPoint() {
        val text = "Before. Hola mundo. Dopo."
        val snapshot = SpeakerDocumentSnapshot(
            text,
            listOf(
                SpeakerPoint(text.indexOf("Hola"), spanishJoy),
                SpeakerPoint(text.indexOf("Dopo"), italianAnger),
            ),
        )

        assertEquals(
            listOf(
                SpeechSegment("Before.", default),
                SpeechSegment("Hola mundo.", spanishJoy),
                SpeechSegment("Dopo.", italianAnger),
            ),
            SpeakerDocument.segments(snapshot, default),
        )
    }

    @Test
    fun documentWithoutPointsUsesDefaults() {
        assertEquals(
            listOf(SpeechSegment("One voice only.", default)),
            SpeakerDocument.segments(
                SpeakerDocumentSnapshot("  One voice only.  ", emptyList()),
                default,
            ),
        )
    }

    @Test
    fun laterPointAtSameOffsetWinsWithoutCreatingEmptyAudio() {
        val snapshot = SpeakerDocumentSnapshot(
            "Hello",
            listOf(SpeakerPoint(0, spanishJoy), SpeakerPoint(0, italianAnger)),
        )
        assertEquals(
            listOf(SpeechSegment("Hello", italianAnger)),
            SpeakerDocument.segments(snapshot, default),
        )
    }
}
