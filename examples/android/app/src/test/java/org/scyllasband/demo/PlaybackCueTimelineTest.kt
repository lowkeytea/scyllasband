package org.scyllasband.demo

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class PlaybackCueTimelineTest {
    @Test
    fun dispatchesCuesOnlyWhenTheirFirstFrameIsAudible() {
        val timeline = PlaybackCueTimeline()
        val dispatched = mutableListOf<String>()
        timeline.add(0L) { dispatched += "first" }
        timeline.add(120L) { dispatched += "second" }
        timeline.add(120L) { dispatched += "third" }

        assertEquals(1, timeline.dispatchThrough(0L))
        assertEquals(listOf("first"), dispatched)
        assertFalse(timeline.isEmpty())

        assertEquals(0, timeline.dispatchThrough(119L))
        assertEquals(2, timeline.dispatchThrough(120L))
        assertEquals(listOf("first", "second", "third"), dispatched)
        assertTrue(timeline.isEmpty())
    }

    @Test
    fun clearDropsPendingCues() {
        val timeline = PlaybackCueTimeline()
        var called = false
        timeline.add(10L) { called = true }

        timeline.clear()

        assertEquals(0, timeline.dispatchThrough(10L))
        assertFalse(called)
    }
}
