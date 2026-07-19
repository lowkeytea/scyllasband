package org.scyllasband.demo

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.SystemClock
import java.io.Closeable
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.max

class FloatAudioPlayer(
    private val sampleRate: Int,
) : Closeable {
    private val track: AudioTrack
    private val chunks = AudioChunkBuffer()
    private val playbackCues = PlaybackCueTimeline()
    private val closed = AtomicBoolean(false)
    private val playbackComplete = CountDownLatch(1)
    private val playbackCuesComplete = CountDownLatch(1)
    private val playbackFailure = AtomicReference<Throwable?>()
    private val playbackThread: Thread
    private val playbackCueThread: Thread
    private var writtenFrames = 0L
    @Volatile
    private var started = false

    init {
        val minimum = AudioTrack.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_FLOAT,
        )
        check(minimum != AudioTrack.ERROR && minimum != AudioTrack.ERROR_BAD_VALUE) {
            "The device does not support 24 kHz float audio playback"
        }
        track = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build(),
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                    .setSampleRate(sampleRate)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build(),
            )
            .setTransferMode(AudioTrack.MODE_STREAM)
            .setBufferSizeInBytes(max(minimum, sampleRate * FLOAT_BYTES * TRACK_BUFFER_SECONDS))
            .build()
        playbackThread = Thread(::playbackLoop, "scyllasband-audio-playback")
        playbackCueThread = Thread(::playbackCueLoop, "scyllasband-audio-cues")
        playbackThread.isDaemon = true
        playbackCueThread.isDaemon = true
        playbackThread.start()
        playbackCueThread.start()
    }

    /** Retain a native callback's audio and let synthesis continue immediately. */
    fun enqueue(
        samples: FloatArray,
        onPlaybackStarted: () -> Unit,
        shouldStop: () -> Boolean,
    ): Boolean {
        throwIfPlaybackFailed()
        return chunks.enqueue(samples, onPlaybackStarted, shouldStop)
    }

    private fun playbackLoop() {
        try {
            while (!closed.get()) {
                val chunk = chunks.poll()
                if (chunk != null) {
                    writeToTrack(chunk)
                } else if (chunks.isDrained()) {
                    break
                }
            }
            if (!closed.get()) waitForPresentation()
        } catch (error: Throwable) {
            if (!closed.get()) playbackFailure.compareAndSet(null, error)
            chunks.close()
        } finally {
            playbackComplete.countDown()
        }
    }

    private fun writeToTrack(chunk: BufferedAudioChunk) {
        val samples = chunk.samples
        val cueFrame = writtenFrames
        var offset = 0
        var cueScheduled = false
        while (offset < samples.size && !closed.get()) {
            val written = track.write(
                samples,
                offset,
                minOf(WRITE_BLOCK_FRAMES, samples.size - offset),
                AudioTrack.WRITE_BLOCKING,
            )
            check(written >= 0) { "AudioTrack write failed: $written" }
            if (written == 0) continue
            offset += written
            writtenFrames += written
            if (!cueScheduled) {
                // Register only after the first samples have entered AudioTrack.
                // This avoids announcing a cue during an output underrun.
                playbackCues.add(cueFrame, chunk.onPlaybackStarted)
                cueScheduled = true
            }
            if (!started) {
                // Prime one write block before starting the clock. This avoids
                // a startup underrun without delaying native synthesis.
                track.play()
                started = true
            }
        }
    }

    private fun playbackCueLoop() {
        try {
            while (!closed.get()) {
                val dispatched = if (started) {
                    playbackCues.dispatchThrough(playedFrames())
                } else {
                    0
                }
                if (playbackComplete.count == 0L && playbackCues.isEmpty()) break
                if (dispatched == 0) Thread.sleep(CUE_POLL_MS)
            }
        } catch (_: InterruptedException) {
            Thread.currentThread().interrupt()
        } finally {
            playbackCuesComplete.countDown()
        }
    }

    /** Finish producer input and wait until queued audio has been presented. */
    fun finish(shouldStop: () -> Boolean) {
        chunks.finishInput()
        while (!closed.get() && !shouldStop()) {
            if (playbackComplete.await(FINISH_POLL_MS, TimeUnit.MILLISECONDS)) break
        }
        if (!closed.get() && !shouldStop()) {
            playbackCuesComplete.await(CUE_FINISH_TIMEOUT_MS, TimeUnit.MILLISECONDS)
            throwIfPlaybackFailed()
        }
    }

    private fun waitForPresentation() {
        if (!started || writtenFrames == 0L || closed.get()) return
        val remainingAtStart = (writtenFrames - playedFrames()).coerceAtLeast(0L)
        val timeoutMs = (remainingAtStart * 1_000L / sampleRate) + 2_000L
        val deadline = SystemClock.elapsedRealtime() + timeoutMs
        while (
            !closed.get() &&
            playedFrames() < writtenFrames &&
            SystemClock.elapsedRealtime() < deadline
        ) {
            SystemClock.sleep(10L)
        }
    }

    private fun playedFrames(): Long =
        track.playbackHeadPosition.toLong() and 0xffff_ffffL

    private fun throwIfPlaybackFailed() {
        playbackFailure.get()?.let { error ->
            throw IllegalStateException("Scylla's Band audio playback failed", error)
        }
    }

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        chunks.close()
        playbackCues.clear()
        runCatching { track.pause() }
        runCatching { track.flush() }
        runCatching { track.stop() }
        playbackThread.interrupt()
        playbackCueThread.interrupt()
        if (Thread.currentThread() !== playbackThread) {
            runCatching { playbackThread.join(CLOSE_JOIN_MS) }
        }
        if (Thread.currentThread() !== playbackCueThread) {
            runCatching { playbackCueThread.join(CLOSE_JOIN_MS) }
        }
        runCatching { track.release() }
    }

    companion object {
        private const val FLOAT_BYTES = 4
        private const val TRACK_BUFFER_SECONDS = 2
        private const val WRITE_BLOCK_FRAMES = 8192
        private const val FINISH_POLL_MS = 25L
        private const val CUE_POLL_MS = 5L
        private const val CUE_FINISH_TIMEOUT_MS = 250L
        private const val CLOSE_JOIN_MS = 1_000L
    }
}
