package org.scyllasband.android

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class ScyllasBandDeliveryTest {
    @Test
    fun neutralSpecIsAllTwosWithWhisperOff() {
        assertEquals(
            "energy=2,tension=2,valence=2,assertiveness=2,whisper=off",
            ScyllasBandDelivery().spec(),
        )
    }

    @Test
    fun specHasNoLegacyPrefixAndKeepsFractions() {
        val delivery = ScyllasBandDelivery(energy = 3.3f, tension = 0.8f, valence = 3.5f, assertiveness = 4f, whisper = true)

        assertEquals(
            "energy=3.3,tension=0.8,valence=3.5,assertiveness=4,whisper=on",
            delivery.spec(),
        )
    }

    @Test
    fun rejectsAxesOutsideZeroToFour() {
        assertThrows(IllegalArgumentException::class.java) { ScyllasBandDelivery(energy = 4.1f) }
        assertThrows(IllegalArgumentException::class.java) { ScyllasBandDelivery(valence = -0.1f) }
        assertThrows(IllegalArgumentException::class.java) { ScyllasBandDelivery(tension = Float.NaN) }
    }
}
