package org.scyllasband.android

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Test

class ScyllasBandSingingTest {
    @Test
    fun decodersMapToTheAbiNames() {
        assertNull(ScyllasBandSingingDecoder.DEFAULT.id)
        assertEquals("bigvgan", ScyllasBandSingingDecoder.BIGVGAN.id)
        assertEquals("vocos", ScyllasBandSingingDecoder.VOCOS.id)
    }

    @Test
    fun optionsRejectNegativeStepsAndNonPositiveGuidance() {
        assertThrows(IllegalArgumentException::class.java) { ScyllasBandSingOptions(steps = -1) }
        assertThrows(IllegalArgumentException::class.java) { ScyllasBandSingOptions(guidance = 0f) }
        assertThrows(IllegalArgumentException::class.java) { ScyllasBandSingOptions(guidance = Float.NaN) }
        assertEquals(ScyllasBandSingingDecoder.VOCOS, ScyllasBandSingOptions(decoder = ScyllasBandSingingDecoder.VOCOS).decoder)
    }

    @Test
    fun phrasesLandOnTheTimelineAndOverlapsAdd() {
        val first = ScyllasBandSungPhrase(floatArrayOf(0.5f, 0.5f, 0.5f), 0, 2, 1, 8, "a")
        val second = ScyllasBandSungPhrase(floatArrayOf(0.75f, 0.25f, 0.25f, 0.25f), 1, 2, 3, 8, "b")
        assertArrayEquals(floatArrayOf(0f, 0.5f, 0.5f, 1f, 0.25f, 0.25f, 0.25f, 0f), assembleSong(listOf(first, second), 8), 0f)
        val tooLong = ScyllasBandSungPhrase(floatArrayOf(0.1f, 0.1f, 0.1f), 0, 1, 6, 8, "c")
        assertArrayEquals(floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0.1f, 0.1f), assembleSong(listOf(tooLong), 8), 0f)
    }
}
