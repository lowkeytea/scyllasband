package org.scyllasband.demo

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.scyllasband.android.ScyllasBandDelivery

class DeliveryPresetsTest {
    @Test
    fun presetsMatchTheDocumentedCombinations() {
        assertEquals(ScyllasBandDelivery(), DeliveryPreset.NEUTRAL.delivery)
        assertEquals(
            "energy=1.8,tension=1,valence=2.5,assertiveness=2.1,whisper=off",
            DeliveryPreset.CALM.delivery.spec(),
        )
        assertEquals(
            "energy=2.6,tension=2.4,valence=2.1,assertiveness=2.8,whisper=off",
            DeliveryPreset.ASSERTIVE.delivery.spec(),
        )
        assertEquals(
            "energy=2.4,tension=2,valence=2.8,assertiveness=2.2,whisper=off",
            DeliveryPreset.JOYFUL.delivery.spec(),
        )
        assertEquals(
            "energy=2.7,tension=3.1,valence=1.3,assertiveness=2.5,whisper=off",
            DeliveryPreset.ANGRY.delivery.spec(),
        )
        assertEquals(
            "energy=1.7,tension=1.7,valence=1.8,assertiveness=2,whisper=off",
            DeliveryPreset.SAD.delivery.spec(),
        )
        assertEquals(ScyllasBandDelivery(whisper = true), DeliveryPreset.WHISPER.delivery)
    }

    @Test
    fun everyPresetStaysWithinTheUiRange() {
        DeliveryPreset.entries.forEach { preset ->
            val d = preset.delivery
            listOf(d.energy, d.tension, d.valence, d.assertiveness).forEach {
                assertTrue("${preset.name} $it", it in DELIVERY_UI_MIN..DELIVERY_UI_MAX)
            }
        }
    }

    @Test
    fun coerceToUiRangeClampsBothEnds() {
        assertEquals(
            ScyllasBandDelivery(energy = 3.1f, tension = 0.5f, valence = 2.5f, assertiveness = 3.1f, whisper = true),
            ScyllasBandDelivery(energy = 3.6f, tension = 0.1f, valence = 2.5f, assertiveness = 4f, whisper = true)
                .coerceToUiRange(),
        )
    }

    @Test
    fun summaryListsOnlyNonNeutralAxes() {
        assertEquals("neutral", ScyllasBandDelivery().summary())
        assertEquals("E 2.4 V 2.8 A 2.2", DeliveryPreset.JOYFUL.delivery.summary())
        assertEquals("whisper", DeliveryPreset.WHISPER.delivery.summary())
    }
}
