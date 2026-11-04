package org.scyllasband.demo

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.scyllasband.android.ScyllasBandDelivery

class DeliveryPresetsTest {
    @Test
    fun theResetPresetIsTheVoicesOwnDelivery() {
        assertEquals(ScyllasBandDelivery(intensity = 0f, pace = 0f, pitch = 0f), DeliveryPreset.DEFAULT.delivery)
        assertEquals("intensity=0,pace=0,pitch=0", DeliveryPreset.DEFAULT.delivery.spec())
    }

    @Test
    fun everyPresetStaysWithinTheSupportedRanges() {
        DeliveryPreset.entries.forEach { preset ->
            assertEquals(preset.name, preset.delivery, preset.delivery.coerceToUiRange())
        }
    }

    @Test
    fun theVoicesOwnDeliveryIsInsideEveryRange() {
        DELIVERY_UI_RANGES.forEach { (control, range) -> assertTrue(control, 0f in range) }
    }

    @Test
    fun coerceToUiRangeKeepsValuesAtTheEdgesAndUnsetControls() {
        val edges = ScyllasBandDelivery(intensity = 2f, pace = -2f)
        assertEquals(edges, edges.coerceToUiRange())
        assertEquals(null, edges.coerceToUiRange().pitch)
    }

    @Test
    fun summaryListsOnlyControlsAwayFromTheVoicesOwn() {
        assertEquals("default", ScyllasBandDelivery().summary())
        assertEquals("default", DeliveryPreset.DEFAULT.delivery.summary())
        assertEquals("intensity 1 pace -0.5", ScyllasBandDelivery(intensity = 1f, pace = -0.5f, pitch = 0f).summary())
    }
}
