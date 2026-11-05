package org.scyllasband.demo.sing

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class SongDocumentTest {
    private fun fixtureDir(): File = File(checkNotNull(javaClass.getResource("/melodies")).toURI())

    private fun fixtureNames(): List<String> =
        fixtureDir().list().orEmpty().filter { it.endsWith(".json") && !it.endsWith(".parsed.json") }.map { it.removeSuffix(".json") }.sorted()

    private fun read(name: String) = JSONObject(File(fixtureDir(), name).readText())

    private fun sung(vararg syllables: String, beats: Double = 0.5, midi: Int = 60): SongLine {
        val notes = syllables.mapIndexed { i, s -> SongNote(syllable = s, midi = midi + i, beats = beats) }
        return SongLine(notes = notes)
    }

    @Test
    fun twentyFixturesExist() {
        assertEquals(20, fixtureNames().size)
    }

    @Test
    fun everyFixtureRoundTripsThroughSummaryAndScore() {
        for (name in fixtureNames()) {
            val original = read("$name.json")
            val parsed = read("$name.parsed.json")
            val song = checkNotNull(SongDocument.fromSummary(parsed, name, "gwen")) { name }
            val score = song.score()
            val out = JSONObject(score.json)
            val outRows = out.getJSONArray("notes")
            val inRows = original.getJSONArray("notes")
            assertEquals("$name row count", inRows.length(), outRows.length())
            assertEquals(outRows.length(), score.rows.size)
            for (i in 0 until inRows.length()) {
                val a = inRows.getJSONArray(i)
                val b = outRows.getJSONArray(i)
                assertEquals("$name row $i kind", a.getString(1) == "rest", b.getString(1) == "rest")
                if (a.getString(1) == "rest") {
                    assertEquals("$name row $i rest", a.getDouble(2), b.getDouble(2), 1e-9)
                    assertEquals("-", b.getString(0))
                } else {
                    assertEquals("$name row $i pitch", a.getString(1), b.getString(1))
                    assertEquals("$name row $i beats", a.getDouble(2), b.getDouble(2), 1e-9)
                    assertEquals("$name row $i label", a.getString(0), b.getString(0))
                }
            }
            assertEquals("$name bpm", original.getDouble("bpm"), out.getDouble("bpm"), 1e-9)
            assertEquals("$name style", original.getString("style"), out.getString("style"))
            val words = { s: String -> s.trim().split(Regex("\\s+")) }
            assertEquals("$name text", words(parsed.getString("text")), words(out.getString("text")))
            // Only line-separating rests have no note behind them.
            assertTrue(score.rows.count { it == null } == song.lines.size - 1)
        }
    }

    @Test
    fun noteNames() {
        assertEquals("C4", SongDocument.noteName(60))
        assertEquals("C#4", SongDocument.noteName(61))
        assertEquals("B0", SongDocument.noteName(23))
        assertEquals("C-1", SongDocument.noteName(0))
    }

    @Test
    fun minimumLineRest() {
        val song = SongDocument.blank("gwen")
        assertEquals(0.5, song.copy(bpm = 104.0).minimumLineRest, 1e-9)
        assertEquals(1.0, song.copy(bpm = 200.0).minimumLineRest, 1e-9)
    }

    @Test
    fun scoreLabelsSyllablesAndHeldWords() {
        val line = SongLine(
            notes = listOf(
                SongNote(syllable = "ba", midi = 64, beats = 0.5),
                SongNote(syllable = "by", midi = 62, beats = 0.5, continuesWord = true),
                SongNote(syllable = "home", midi = 60, beats = 0.5),
                SongNote(syllable = "", midi = 62, beats = 0.5, continuesWord = true),
                SongNote.rest(1.0),
                SongNote(syllable = "", midi = 60, beats = 0.5),
            ),
        )
        val song = SongDocument(title = "t", bpm = 100.0, style = SongStyle.POP, voice = "v", lines = listOf(line))
        val rows = JSONObject(song.scoreJson()).getJSONArray("notes")
        val labels = (0 until rows.length()).map { rows.getJSONArray(it).getString(0) }
        assertEquals(listOf("ba~1", "by~2", "home~1", "home~2", "-", "la"), labels)
        assertEquals("baby home la", JSONObject(song.scoreJson()).getString("text"))
        assertTrue(JSONObject(song.scoreJson()).get("bpm") is Number)
    }

    @Test
    fun lineSeparatingRestIsAtLeastMinimum() {
        val song = SongDocument(
            title = "t", bpm = 200.0, style = SongStyle.POP, voice = "v",
            lines = listOf(sung("a").copy(restAfter = 0.5), sung("b")),
        )
        val rows = song.score()
        assertEquals(listOf(song.lines[0].notes[0].id, null, song.lines[1].notes[0].id), rows.rows)
        assertEquals(1.0, JSONObject(rows.json).getJSONArray("notes").getJSONArray(1).getDouble(2), 1e-9)
    }

    @Test
    fun applyingLyricsMoreSyllablesThanNotes() {
        val line = sung("a", "b")
        val out = SongDocument.applyingLyrics("one two three", line, SongStyle.BALLAD)
        assertEquals(listOf("one", "two", "three"), out.notes.map { it.syllable })
        assertEquals(listOf(60, 61, 61), out.notes.map { it.midi })
        assertEquals(1.0, out.notes[2].beats, 0.0)
        assertEquals(line.notes[0].id, out.notes[0].id)
    }

    @Test
    fun applyingLyricsFewerSyllablesThanNotes() {
        val out = SongDocument.applyingLyrics("one", sung("a", "b", "c"), SongStyle.POP)
        assertEquals(1, out.notes.size)
        assertEquals("one", out.notes[0].syllable)
    }

    @Test
    fun applyingLyricsSyllablesAndHolds() {
        val line = sung("x", "x", "x", "x", "x")
        val out = SongDocument.applyingLyrics("ba-by _ go", line, SongStyle.POP)
        assertEquals(listOf("ba", "by", "", "go"), out.notes.map { it.syllable })
        assertEquals(listOf(false, true, true, false), out.notes.map { it.continuesWord })
        assertEquals("ba-by_ go", out.lyrics)
    }

    @Test
    fun applyingLyricsKeepsRestsAndClearsLeadingContinuation() {
        val line = SongLine(notes = listOf(SongNote.rest(1.0), sung("a").notes[0], SongNote.rest(0.5), sung("b").notes[0]))
        val out = SongDocument.applyingLyrics("_ hi", line, SongStyle.POP)
        assertEquals(listOf(true, false, true, false), out.notes.map { it.isRest })
        assertEquals(1.0, out.notes[0].beats, 0.0)
        assertFalse(out.notes[1].continuesWord)
        assertEquals("_ hi", out.lyrics)
    }

    @Test
    fun lyricsNotationRoundTrips() {
        for (text in listOf("ba-by", "give me a ring", "ho_ me", "a-b-c d", "_ x")) {
            val line = SongDocument.applyingLyrics(text, sung("z"), SongStyle.POP)
            assertEquals(text, line.lyrics)
            assertEquals(line.lyrics, SongDocument.applyingLyrics(line.lyrics, line, SongStyle.POP).lyrics)
        }
    }

    @Test
    fun fixtureLyricsSurviveRetyping() {
        val song = checkNotNull(SongDocument.fromSummary(read("phrase_pop_00.parsed.json"), "t", "gwen"))
        val line = song.lines[1]
        assertEquals("ba-by make me feel so", line.lyrics)
        assertEquals(line.notes.map { it.syllable }, SongDocument.applyingLyrics(line.lyrics, line, SongStyle.POP).notes.map { it.syllable })
    }

    @Test
    fun fromSummaryRejectsBrokenInput() {
        assertNull(SongDocument.fromSummary(JSONObject(), "t", "v"))
        assertNull(SongDocument.fromSummary(JSONObject().put("bpm", 100).put("notes", JSONArray()), "t", "v"))
        assertNotNull(SongDocument.fromSummary(read("phrase_pop_01.parsed.json"), "t", "v"))
    }

    @Test
    fun jsonRoundTrip() {
        val base = checkNotNull(SongDocument.fromSummary(read("song_ballad_felix.parsed.json"), "Felix", "felix"))
        val song = base.copy(decoder = SongDecoder.TURBO, transpose = -3)
        val back = SongDocument.fromJson(JSONObject(song.toJson().toString()))
        assertEquals(song, back)
        assertEquals(SongDecoder.TURBO, back.decoder)
    }

    @Test
    fun enumsFallBackToDefaults() {
        assertEquals(SongStyle.POP, SongStyle.fromId(null))
        assertEquals(SongStyle.BALLAD, SongStyle.fromId("ballad"))
        assertEquals(SongDecoder.QUALITY, SongDecoder.fromId("nope"))
        assertEquals(0.5, SongStyle.POP.defaultBeats, 0.0)
        assertEquals(1.0, SongStyle.BALLAD.defaultBeats, 0.0)
    }
}
