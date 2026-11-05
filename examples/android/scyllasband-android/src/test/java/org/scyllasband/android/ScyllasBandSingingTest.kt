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
    fun phrasesAreContiguousSoTheSongIsTheirSamplesInOrder() {
        val first = ScyllasBandSungPhrase(floatArrayOf(0f, 0.5f, 0.5f), 0, 3, 0, 7, "a")
        val empty = ScyllasBandSungPhrase(floatArrayOf(), 1, 3, 3, 7, "b")
        val last = ScyllasBandSungPhrase(floatArrayOf(1f, 0.25f, 0.25f, 0f), 2, 3, 3, 7, "c")
        assertArrayEquals(floatArrayOf(0f, 0.5f, 0.5f, 1f, 0.25f, 0.25f, 0f), joinSong(listOf(first, empty, last)), 0f)
    }

    @Test
    fun scoreErrorsCarryTheNoteAtFault() {
        assertEquals(3, ScyllasBandScoreException("Unknown note 'H4'", 3).noteIndex)
        assertNull(ScyllasBandScoreException("Score needs a numeric bpm", null).noteIndex)
    }
}
