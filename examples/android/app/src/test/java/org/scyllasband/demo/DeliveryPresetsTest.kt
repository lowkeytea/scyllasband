package org.scyllasband.demo

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.scyllasband.android.ScyllasBandDelivery

class DeliveryPresetsTest {
    @Test
    fun presetsMatchTheDocumentedCombinations() {
        assertEquals(ScyllasBandDelivery(), DeliveryPreset.NEUTRAL.delivery)
        assertEquals("energy=1.8,tension=1,valence=2.2,assertiveness=2.1", DeliveryPreset.CALM.delivery.spec())
        assertEquals("energy=2.3,tension=2.4,valence=2.1,assertiveness=2.4", DeliveryPreset.ASSERTIVE.delivery.spec())
        assertEquals("energy=2.3,tension=2,valence=2.2,assertiveness=2.2", DeliveryPreset.JOYFUL.delivery.spec())
        assertEquals("energy=2.3,tension=2.4,valence=1.4,assertiveness=2.4", DeliveryPreset.ANGRY.delivery.spec())
        assertEquals("energy=1.7,tension=1.5,valence=1.4,assertiveness=2", DeliveryPreset.SAD.delivery.spec())
    }

    @Test
    fun everyPresetStaysWithinTheSupportedRanges() {
        DeliveryPreset.entries.forEach { preset ->
            assertEquals(preset.name, preset.delivery, preset.delivery.coerceToUiRange())
        }
    }

    @Test
    fun neutralIsInsideEveryRange() {
        DELIVERY_UI_RANGES.forEach { (axis, range) -> assertTrue(axis, ScyllasBandDelivery.NEUTRAL in range) }
    }

    @Test
    fun coerceToUiRangeClampsEachAxisToItsRange() {
        assertEquals(
            ScyllasBandDelivery(energy = 2.3f, tension = 1f, valence = 2.2f, assertiveness = 1.8f),
            ScyllasBandDelivery(energy = 3.6f, tension = 0.1f, valence = 2.5f, assertiveness = 0f).coerceToUiRange(),
        )
    }

    @Test
    fun summaryListsOnlyNonNeutralAxes() {
        assertEquals("neutral", ScyllasBandDelivery().summary())
        assertEquals("E 2.3 V 2.2 A 2.2", DeliveryPreset.JOYFUL.delivery.summary())
    }
}
