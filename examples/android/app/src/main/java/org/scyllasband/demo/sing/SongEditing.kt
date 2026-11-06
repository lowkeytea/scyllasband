package org.scyllasband.demo.sing

/** The song's edits as pure functions: each returns the updated song, or the song itself when an id is not found. */
object SongEditing {
    fun note(song: SongDocument, lineId: String, noteId: String): SongNote? =
        song.lines.firstOrNull { it.id == lineId }?.notes?.firstOrNull { it.id == noteId }

    fun updateLyrics(song: SongDocument, lineId: String, text: String): SongDocument =
        editLine(song, lineId) { line, style -> SongDocument.applyingLyrics(text, line, style) }

    /** Clamped to 24..96. */
    fun setPitch(song: SongDocument, lineId: String, noteId: String, midi: Int): SongDocument =
        editNote(song, lineId, noteId) { it.copy(midi = midi.coerceIn(24, 96)) }

    fun setBeats(song: SongDocument, lineId: String, noteId: String, beats: Double): SongDocument =
        editNote(song, lineId, noteId) { it.copy(beats = beats) }

    fun setSyllable(song: SongDocument, lineId: String, noteId: String, text: String): SongDocument =
        editNote(song, lineId, noteId) { it.copy(syllable = text.filter { c -> !c.isWhitespace() }) }

    /** A sung note becomes a rest of the same length, and a rest a note at the previous pitch. */
    fun toggleRest(song: SongDocument, lineId: String, noteId: String): SongDocument =
        editLine(song, lineId) { line, _ ->
            val index = line.notes.indexOfFirst { it.id == noteId }
            if (index < 0) return@editLine line
            val notes = line.notes.toMutableList()
            val note = notes[index]
            if (note.isRest) {
                val pitch = notes.subList(0, index).lastOrNull { !it.isRest }?.midi ?: 60
                notes[index] = SongNote(id = note.id, syllable = "la", midi = pitch, beats = note.beats)
            } else {
                notes[index] = SongNote(id = note.id, syllable = "", midi = null, beats = note.beats)
                if (index + 1 < notes.size) notes[index + 1] = notes[index + 1].copy(continuesWord = false)
            }
            line.copy(notes = notes)
        }

    /** Splits the note in two halves; the second holds the same syllable, so it can take a second pitch (a melisma). */
    fun split(song: SongDocument, lineId: String, noteId: String): SongDocument =
        editLine(song, lineId) { line, _ ->
            val index = line.notes.indexOfFirst { it.id == noteId }
            if (index < 0) return@editLine line
            val first = line.notes[index].let { it.copy(beats = maxOf(0.25, it.beats / 2)) }
            val second = SongNote(syllable = "", midi = first.midi, beats = first.beats, continuesWord = !first.isRest)
            val notes = line.notes.toMutableList()
            notes[index] = first
            notes.add(index + 1, second)
            line.copy(notes = notes)
        }

    /** Joins the note with the next one: their lengths add up and the first note's pitch and syllable stay. */
    fun mergeWithNext(song: SongDocument, lineId: String, noteId: String): SongDocument =
        editLine(song, lineId) { line, _ ->
            val index = line.notes.indexOfFirst { it.id == noteId }
            if (index < 0 || index + 1 >= line.notes.size) return@editLine line
            val notes = line.notes.toMutableList()
            val next = notes.removeAt(index + 1)
            var merged = notes[index].copy(beats = notes[index].beats + next.beats)
            if (next.syllable.isNotEmpty() && next.continuesWord && !merged.isRest) {
                merged = merged.copy(syllable = merged.syllable + next.syllable)
            }
            notes[index] = merged
            line.copy(notes = notes)
        }

    fun insertRestAfter(song: SongDocument, lineId: String, noteId: String): SongDocument =
        editLine(song, lineId) { line, style ->
            val index = line.notes.indexOfFirst { it.id == noteId }
            if (index < 0) return@editLine line
            val notes = line.notes.toMutableList()
            notes.add(index + 1, SongNote.rest(style.defaultBeats))
            if (index + 2 < notes.size) notes[index + 2] = notes[index + 2].copy(continuesWord = false)
            line.copy(notes = notes)
        }

    /** Keeps at least one note per line. */
    fun delete(song: SongDocument, lineId: String, noteId: String): SongDocument =
        editLine(song, lineId) { line, _ ->
            val index = line.notes.indexOfFirst { it.id == noteId }
            if (line.notes.size <= 1 || index < 0) return@editLine line
            val notes = line.notes.toMutableList()
            val removed = notes.removeAt(index)
            if (index < notes.size && !removed.continuesWord) notes[index] = notes[index].copy(continuesWord = false)
            line.copy(notes = notes)
        }

    /** Appends a line of one "la" at the last sung pitch. */
    fun addLine(song: SongDocument): SongDocument {
        val pitch = song.lines.lastOrNull()?.notes?.lastOrNull { !it.isRest }?.midi ?: 60
        val line = SongLine(
            notes = listOf(SongNote(syllable = "la", midi = pitch, beats = song.style.defaultBeats * 2)),
            restAfter = maxOf(1.0, song.minimumLineRest),
        )
        return song.copy(lines = song.lines + line)
    }

    /** Keeps at least one line. */
    fun removeLine(song: SongDocument, lineId: String): SongDocument {
        if (song.lines.size <= 1) return song
        return song.copy(lines = song.lines.filter { it.id != lineId })
    }

    fun setRestAfter(song: SongDocument, lineId: String, beats: Double): SongDocument =
        editLine(song, lineId) { line, _ -> line.copy(restAfter = beats) }

    private fun editLine(song: SongDocument, lineId: String, change: (SongLine, SongStyle) -> SongLine): SongDocument {
        val index = song.lines.indexOfFirst { it.id == lineId }
        if (index < 0) return song
        val lines = song.lines.toMutableList()
        lines[index] = change(lines[index], song.style)
        return song.copy(lines = lines)
    }

    private fun editNote(song: SongDocument, lineId: String, noteId: String, change: (SongNote) -> SongNote): SongDocument =
        editLine(song, lineId) { line, _ ->
            val index = line.notes.indexOfFirst { it.id == noteId }
            if (index < 0) return@editLine line
            line.copy(notes = line.notes.toMutableList().also { it[index] = change(it[index]) })
        }
}
