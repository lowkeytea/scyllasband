package org.scyllasband.demo

import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

class AudioChunkBufferTest {
    @Test
    fun preservesChunkOrderUntilInputIsDrained() {
        val buffer = AudioChunkBuffer(capacity = 2)
        val first = floatArrayOf(1f, 2f)
        val second = floatArrayOf(3f)

        assertTrue(buffer.enqueue(first) { false })
        assertTrue(buffer.enqueue(second) { false })
        buffer.finishInput()

        assertSame(first, buffer.poll()?.samples)
        assertFalse(buffer.isDrained())
        assertSame(second, buffer.poll()?.samples)
        assertTrue(buffer.isDrained())
    }

    @Test
    fun cancellationRejectsAudioWithoutRetainingIt() {
        val buffer = AudioChunkBuffer(capacity = 1)

        assertFalse(buffer.enqueue(floatArrayOf(1f)) { true })
        buffer.finishInput()

        assertNull(buffer.poll())
        assertTrue(buffer.isDrained())
    }

    @Test
    fun closeDropsQueuedAudioAndRejectsNewChunks() {
        val buffer = AudioChunkBuffer(capacity = 1)
        assertTrue(buffer.enqueue(floatArrayOf(1f)) { false })

        buffer.close()

        assertFalse(buffer.enqueue(floatArrayOf(2f)) { false })
        assertNull(buffer.poll())
    }

    @Test
    fun retainsPlaybackNotificationWithItsAudioChunk() {
        val buffer = AudioChunkBuffer(capacity = 1)
        var playbackStarted = false

        assertTrue(
            buffer.enqueue(
                floatArrayOf(1f),
                onPlaybackStarted = { playbackStarted = true },
                shouldStop = { false },
            ),
        )

        val chunk = buffer.poll()
        assertFalse(playbackStarted)
        chunk?.onPlaybackStarted?.invoke()
        assertTrue(playbackStarted)
    }
}
