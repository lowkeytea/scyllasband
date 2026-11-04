package org.scyllasband.android

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class ScyllasBandDeliveryTest {
    @Test
    fun unsetControlsTakeTheVoiceDefaults() {
        assertEquals("", ScyllasBandDelivery().spec())
        assertEquals("pace=-0.5", ScyllasBandDelivery(pace = -0.5f).spec())
    }

    @Test
    fun specListsTheSetControlsAndKeepsFractions() {
        assertEquals(
            "intensity=1.25,pace=-1,pitch=0",
            ScyllasBandDelivery(intensity = 1.25f, pace = -1f, pitch = -0f).spec(),
        )
    }

    @Test
    fun rejectsControlsOutsideTheSupportedRange() {
        assertThrows(IllegalArgumentException::class.java) { ScyllasBandDelivery(intensity = 2.1f) }
        assertThrows(IllegalArgumentException::class.java) { ScyllasBandDelivery(pace = -2.5f) }
        assertThrows(IllegalArgumentException::class.java) { ScyllasBandDelivery(pitch = Float.NaN) }
    }
}
