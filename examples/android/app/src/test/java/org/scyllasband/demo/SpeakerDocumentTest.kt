package org.scyllasband.demo

import org.scyllasband.android.ScyllasBandDelivery
import org.scyllasband.android.ScyllasBandSegmentSettings
import org.junit.Assert.assertEquals
import org.junit.Test

class SpeakerDocumentTest {
    private val default = ScyllasBandSegmentSettings("scylla", "en_us")
    private val spanishJoyful = ScyllasBandSegmentSettings("ariadne", "es", DeliveryPreset.JOYFUL.delivery)
    private val italianWhisper = ScyllasBandSegmentSettings("ink", "it", ScyllasBandDelivery(energy = 3.4f, whisper = true))

    @Test
    fun pointAppliesUntilTheNextPoint() {
        val text = "Before. Hola mundo. Dopo."
        val snapshot = SpeakerDocumentSnapshot(
            text,
            listOf(
                SpeakerPoint(text.indexOf("Hola"), spanishJoyful),
                SpeakerPoint(text.indexOf("Dopo"), italianWhisper),
            ),
        )

        assertEquals(
            listOf(
                SpeechSegment("Before.", default),
                SpeechSegment("Hola mundo.", spanishJoyful),
                SpeechSegment("Dopo.", italianWhisper),
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
            listOf(SpeakerPoint(0, spanishJoyful), SpeakerPoint(0, italianWhisper)),
        )
        assertEquals(
            listOf(SpeechSegment("Hello", italianWhisper)),
            SpeakerDocument.segments(snapshot, default),
        )
    }
}
