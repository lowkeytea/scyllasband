package org.scyllasband.demo

import java.util.concurrent.ConcurrentLinkedQueue

/** Thread-safe, ordered callbacks keyed to absolute AudioTrack frame positions. */
internal class PlaybackCueTimeline {
    private val cues = ConcurrentLinkedQueue<PlaybackCue>()

    fun add(frame: Long, action: () -> Unit) {
        require(frame >= 0L) { "Playback cue frame must not be negative" }
        cues.add(PlaybackCue(frame, action))
    }

    fun dispatchThrough(playedFrame: Long): Int {
        var dispatched = 0
        while (true) {
            val next = cues.peek() ?: break
            if (next.frame > playedFrame) break
            val ready = cues.poll() ?: continue
            ready.action()
            dispatched += 1
        }
        return dispatched
    }

    fun isEmpty(): Boolean = cues.isEmpty()

    fun clear() {
        cues.clear()
    }
}

private data class PlaybackCue(
    val frame: Long,
    val action: () -> Unit,
)
