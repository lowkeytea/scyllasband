package org.scyllasband.demo

import org.scyllasband.android.ScyllasBandDelivery
import org.scyllasband.android.ScyllasBandSegmentSettings
import org.junit.Assert.assertEquals
import org.junit.Test

class SpeakerDocumentTest {
    private val default = ScyllasBandSegmentSettings("scylla", "en_us")
    private val spanishJoyful = ScyllasBandSegmentSettings("ariadne", "es", DeliveryPreset.JOYFUL.delivery)
    private val italianSoft = ScyllasBandSegmentSettings("ink", "it", ScyllasBandDelivery(energy = 1.6f, tension = 1.2f))

    @Test
    fun pointAppliesUntilTheNextPoint() {
        val text = "Before. Hola mundo. Dopo."
        val snapshot = SpeakerDocumentSnapshot(
            text,
            listOf(
                SpeakerPoint(text.indexOf("Hola"), spanishJoyful),
                SpeakerPoint(text.indexOf("Dopo"), italianSoft),
            ),
        )

        assertEquals(
            listOf(
                SpeechSegment("Before.", default),
                SpeechSegment("Hola mundo.", spanishJoyful),
                SpeechSegment("Dopo.", italianSoft),
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
            listOf(SpeakerPoint(0, spanishJoyful), SpeakerPoint(0, italianSoft)),
        )
        assertEquals(
            listOf(SpeechSegment("Hello", italianSoft)),
            SpeakerDocument.segments(snapshot, default),
        )
    }
}
