package org.scyllasband.demo.sing

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder

class SongFilesTest {
    @Test
    fun wavHeaderAndSamples() {
        val bytes = SongFiles.wav(floatArrayOf(0f, 1f, -1f, 2f, 0.5f), 24000)
        assertEquals(44 + 10, bytes.size)
        assertEquals("RIFF", String(bytes, 0, 4, Charsets.US_ASCII))
        assertEquals("WAVEfmt ", String(bytes, 8, 8, Charsets.US_ASCII))
        assertEquals("data", String(bytes, 36, 4, Charsets.US_ASCII))
        val b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        assertEquals(36 + 10, b.getInt(4))
        assertEquals(16, b.getInt(16))
        assertEquals(1, b.getShort(20).toInt())
        assertEquals(1, b.getShort(22).toInt())
        assertEquals(24000, b.getInt(24))
        assertEquals(48000, b.getInt(28))
        assertEquals(2, b.getShort(32).toInt())
        assertEquals(16, b.getShort(34).toInt())
        assertEquals(10, b.getInt(40))
        assertEquals(listOf(0, 32767, -32767, 32767, 16384), (0 until 5).map { b.getShort(44 + it * 2).toInt() })
    }

    @Test
    fun emptyWav() {
        assertEquals(44, SongFiles.wav(FloatArray(0), 16000).size)
    }

    @Test
    fun fileNames() {
        assertEquals("My song 2", SongFiles.fileName("  My song! 2?/ "))
        assertEquals("Song", SongFiles.fileName("!!!"))
        assertEquals("Song", SongFiles.fileName(""))
        assertEquals("Café", SongFiles.fileName("Café"))
    }

    @Test
    fun exampleTitles() {
        assertEquals("Pop song · Gwen", ExampleScore("song_pop_gwen.json").title)
        assertEquals("Ballad phrase 4", ExampleScore("phrase_ballad_03.json").title)
        assertEquals("Some Other Thing", ExampleScore("some_other_thing.json").title)
    }

    @Test
    fun exampleVoiceAndKind() {
        assertEquals("gwen", ExampleScore("song_pop_gwen.json").voice)
        assertNull(ExampleScore("phrase_pop_00.json").voice)
        assertTrue(ExampleScore("song_pop_gwen.json").isSong)
        assertFalse(ExampleScore("phrase_pop_00.json").isSong)
    }

    @Test
    fun songsSortFirstThenByStem() {
        val sorted = ExampleScore.sorted(listOf("phrase_pop_01.json", "song_pop_poppy.json", "phrase_ballad_00.json", "song_ballad_felix.json"))
        assertEquals(
            listOf("song_ballad_felix.json", "song_pop_poppy.json", "phrase_ballad_00.json", "phrase_pop_01.json"),
            sorted.map { it.assetName },
        )
    }
}
