package org.scyllasband.demo

import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Bounded handoff between native synthesis and real-time playback.
 *
 * Native streaming callbacks should return as soon as audio has been retained
 * by this queue. If synthesis gets far ahead, bounded backpressure prevents a
 * long document from retaining unbounded waveform memory.
 */
internal class AudioChunkBuffer(
    capacity: Int = DEFAULT_CAPACITY,
) {
    private val chunks = ArrayBlockingQueue<BufferedAudioChunk>(capacity)
    private val inputFinished = AtomicBoolean(false)
    private val closed = AtomicBoolean(false)

    init {
        require(capacity > 0) { "Audio chunk capacity must be positive" }
    }

    fun enqueue(
        samples: FloatArray,
        onPlaybackStarted: () -> Unit = {},
        shouldStop: () -> Boolean,
    ): Boolean {
        if (samples.isEmpty()) return !closed.get() && !shouldStop()
        if (closed.get() || shouldStop()) return false
        check(!inputFinished.get()) { "Audio input is already finished" }
        val chunk = BufferedAudioChunk(samples, onPlaybackStarted)
        while (!closed.get() && !shouldStop()) {
            try {
                if (chunks.offer(chunk, OFFER_POLL_MS, TimeUnit.MILLISECONDS)) return true
            } catch (_: InterruptedException) {
                Thread.currentThread().interrupt()
                return false
            }
        }
        return false
    }

    fun poll(): BufferedAudioChunk? = chunks.poll(CONSUMER_POLL_MS, TimeUnit.MILLISECONDS)

    fun finishInput() {
        inputFinished.set(true)
    }

    fun isDrained(): Boolean = inputFinished.get() && chunks.isEmpty()

    fun close() {
        closed.set(true)
        chunks.clear()
    }

    companion object {
        // Two chunks cover normal producer/consumer jitter without retaining a
        // long queue of generated waveforms or letting transcript updates race
        // far ahead of audible playback.
        private const val DEFAULT_CAPACITY = 2
        private const val OFFER_POLL_MS = 25L
        private const val CONSUMER_POLL_MS = 25L
    }
}

internal data class BufferedAudioChunk(
    val samples: FloatArray,
    val onPlaybackStarted: () -> Unit,
)
