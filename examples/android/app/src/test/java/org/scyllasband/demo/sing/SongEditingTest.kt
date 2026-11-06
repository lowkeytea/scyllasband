package org.scyllasband.demo.sing

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class SongEditingTest {
    private val a = SongNote(syllable = "a", midi = 60, beats = 1.0)
    private val b = SongNote(syllable = "b", midi = 62, beats = 0.5)
    private val c = SongNote(syllable = "c", midi = 64, beats = 0.5, continuesWord = true)
    private val line = SongLine(notes = listOf(a, b, c))
    private val song = SongDocument(title = "t", bpm = 104.0, style = SongStyle.POP, voice = "v", lines = listOf(line))

    private fun notes(s: SongDocument) = s.lines[0].notes

    @Test
    fun noteLookup() {
        assertEquals(b, SongEditing.note(song, line.id, b.id))
        assertNull(SongEditing.note(song, line.id, "x"))
        assertNull(SongEditing.note(song, "x", b.id))
    }

    @Test
    fun unknownIdsLeaveSongUnchanged() {
        assertEquals(song, SongEditing.setPitch(song, line.id, "x", 70))
        assertEquals(song, SongEditing.toggleRest(song, "x", a.id))
        assertEquals(song, SongEditing.split(song, line.id, "x"))
        assertEquals(song, SongEditing.updateLyrics(song, "x", "hi"))
    }

    @Test
    fun setPitchClamps() {
        assertEquals(96, notes(SongEditing.setPitch(song, line.id, a.id, 200))[0].midi)
        assertEquals(24, notes(SongEditing.setPitch(song, line.id, a.id, 3))[0].midi)
        assertEquals(70, notes(SongEditing.setPitch(song, line.id, a.id, 70))[0].midi)
    }

    @Test
    fun setBeatsAndSyllable() {
        assertEquals(2.0, notes(SongEditing.setBeats(song, line.id, b.id, 2.0))[1].beats, 0.0)
        assertEquals("ho-me", notes(SongEditing.setSyllable(song, line.id, b.id, " ho -\tme "))[1].syllable)
        assertEquals("home", notes(SongEditing.setSyllable(song, line.id, b.id, "ho me"))[1].syllable)
    }

    @Test
    fun toggleRestBothWays() {
        val rested = SongEditing.toggleRest(song, line.id, b.id)
        assertTrue(notes(rested)[1].isRest)
        assertEquals(0.5, notes(rested)[1].beats, 0.0)
        assertEquals(b.id, notes(rested)[1].id)
        assertFalse(notes(rested)[2].continuesWord)   // the next note starts its own word
        val back = SongEditing.toggleRest(rested, line.id, b.id)
        assertEquals(60, notes(back)[1].midi)           // previous sung pitch
        assertEquals("la", notes(back)[1].syllable)
        val firstRest = SongEditing.toggleRest(SongEditing.toggleRest(song, line.id, a.id), line.id, a.id)
        assertEquals(60, notes(firstRest)[0].midi)      // no previous note: C4
    }

    @Test
    fun splitHalvesAndHoldsSyllable() {
        val out = notes(SongEditing.split(song, line.id, a.id))
        assertEquals(4, out.size)
        assertEquals(0.5, out[0].beats, 0.0)
        assertEquals(0.5, out[1].beats, 0.0)
        assertEquals("", out[1].syllable)
        assertEquals(60, out[1].midi)
        assertTrue(out[1].continuesWord)
        // Floor of a quarter beat.
        val tiny = SongEditing.setBeats(song, line.id, a.id, 0.3)
        assertEquals(0.25, notes(SongEditing.split(tiny, line.id, a.id))[0].beats, 0.0)
    }

    @Test
    fun splitRestStaysRest() {
        val rested = SongEditing.toggleRest(song, line.id, a.id)
        val out = notes(SongEditing.split(rested, line.id, a.id))
        assertTrue(out[0].isRest && out[1].isRest)
        assertFalse(out[1].continuesWord)
    }

    @Test
    fun mergeWithNext() {
        val out = notes(SongEditing.mergeWithNext(song, line.id, b.id))
        assertEquals(2, out.size)
        assertEquals(1.0, out[1].beats, 0.0)
        assertEquals("bc", out[1].syllable)   // c continues the word
        assertEquals(62, out[1].midi)
        val plain = notes(SongEditing.mergeWithNext(song, line.id, a.id))
        assertEquals("a", plain[0].syllable)  // b starts a new word
        assertEquals(1.5, plain[0].beats, 0.0)
        assertEquals(song, SongEditing.mergeWithNext(song, line.id, c.id))   // last note: nothing to merge
    }

    @Test
    fun insertRestAfter() {
        val out = notes(SongEditing.insertRestAfter(song, line.id, b.id))
        assertEquals(4, out.size)
        assertTrue(out[2].isRest)
        assertEquals(0.5, out[2].beats, 0.0)
        assertFalse(out[3].continuesWord)
    }

    @Test
    fun deleteKeepsAtLeastOneNote() {
        val out = notes(SongEditing.delete(song, line.id, b.id))
        assertEquals(listOf(a.id, c.id), out.map { it.id })
        assertFalse(out[1].continuesWord)   // b was not a continuation, so c now starts a word
        val one = SongDocument(title = "t", bpm = 100.0, style = SongStyle.POP, voice = "v", lines = listOf(SongLine(notes = listOf(a))))
        assertEquals(one, SongEditing.delete(one, one.lines[0].id, a.id))
        val keepsContinuation = notes(SongEditing.delete(song, line.id, c.id))
        assertEquals(2, keepsContinuation.size)
    }

    @Test
    fun deletingContinuationKeepsNextContinuation() {
        val d = SongNote(syllable = "d", midi = 65, beats = 0.5, continuesWord = true)
        val l = SongLine(notes = listOf(a, c, d))
        val s = song.copy(lines = listOf(l))
        assertTrue(notes(SongEditing.delete(s, l.id, c.id))[1].continuesWord)
    }

    @Test
    fun addAndRemoveLine() {
        val out = SongEditing.addLine(song)
        assertEquals(2, out.lines.size)
        val added = out.lines[1]
        assertEquals(1, added.notes.size)
        assertEquals("la", added.notes[0].syllable)
        assertEquals(64, added.notes[0].midi)   // last sung pitch
        assertEquals(1.0, added.notes[0].beats, 0.0)   // pop default 0.5 x 2
        assertEquals(1.0, added.restAfter, 0.0)
        val removed = SongEditing.removeLine(out, line.id)
        assertEquals(listOf(added.id), removed.lines.map { it.id })
        assertEquals(removed, SongEditing.removeLine(removed, added.id))   // keeps one line
    }

    @Test
    fun setRestAfterAndUpdateLyrics() {
        assertEquals(2.5, SongEditing.setRestAfter(song, line.id, 2.5).lines[0].restAfter, 0.0)
        val out = SongEditing.updateLyrics(song, line.id, "x y")
        assertEquals(listOf("x", "y"), notes(out).map { it.syllable })
    }
}
