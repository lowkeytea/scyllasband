package org.scyllasband.demo.sing

import org.json.JSONArray
import org.json.JSONObject
import java.util.UUID
import kotlin.math.ceil
import kotlin.math.floor

fun newSongId(): String = UUID.randomUUID().toString()

/** One note of a song line: the syllable sung on it, its pitch (null for a rest) and its length in beats (quarter notes). */
data class SongNote(
    val id: String = newSongId(),
    /** The text sung on this note: a word, one syllable of it, or "" when the note holds the previous syllable. */
    val syllable: String,
    /** MIDI pitch (60 = C4); null for a rest. */
    val midi: Int?,
    val beats: Double,
    /** True when this note continues the previous note's word (its next syllable, or the same syllable held). */
    val continuesWord: Boolean = false,
) {
    val isRest: Boolean get() = midi == null

    companion object {
        fun rest(beats: Double): SongNote = SongNote(syllable = "", midi = null, beats = beats)
    }
}

/** A line of the song: a run of notes sung as one phrase. Lines are separated by [restAfter] beats of rest. */
data class SongLine(
    val id: String = newSongId(),
    val notes: List<SongNote>,
    /** The rest after the line (beats). It must last at least 0.25 s for the runtime to start a new phrase there. */
    val restAfter: Double = 1.0,
) {
    /** The line's lyrics in the editor's notation: words separated by spaces, syllables by "-", a held note by "_". */
    val lyrics: String
        get() {
            val words = mutableListOf<String>()
            for (note in notes) {
                if (note.isRest) continue
                if (note.continuesWord && words.isNotEmpty()) {
                    words[words.size - 1] += if (note.syllable.isEmpty()) "_" else "-" + note.syllable
                } else {
                    words.add(note.syllable.ifEmpty { "_" })
                }
            }
            return words.joinToString(" ")
        }
}

enum class SongStyle(val id: String, val title: String) {
    POP("pop", "Pop"),
    BALLAD("ballad", "Ballad");

    /** The length a new note gets. */
    val defaultBeats: Double get() = if (this == POP) 0.5 else 1.0

    companion object {
        fun fromId(id: String?): SongStyle = entries.firstOrNull { it.id == id } ?: POP
    }
}

/**
 * Quality is BigVGAN, the singing decoder; Turbo is Vocos, the speech decoder: much faster, with some harmonic artefacts on
 * sustained notes in some voices.
 */
enum class SongDecoder(val id: String, val title: String) {
    QUALITY("quality", "Quality"),
    TURBO("turbo", "Turbo");

    companion object {
        fun fromId(id: String?): SongDecoder = entries.firstOrNull { it.id == id } ?: QUALITY
    }
}

/** The score JSON and, for each of its rows, the id of the note behind it (null for the rests between lines). */
data class ScoreRows(val json: String, val rows: List<String?>)

data class SongDocument(
    val id: String = newSongId(),
    val title: String,
    val bpm: Double,
    val style: SongStyle,
    val voice: String,
    val decoder: SongDecoder = SongDecoder.QUALITY,
    val transpose: Int = 0,
    val lines: List<SongLine>,
) {
    val secondsPerBeat: Double get() = 60.0 / bpm

    /** The shortest rest, in beats, that still separates two lines (0.25 s, rounded up to a sixteenth). */
    val minimumLineRest: Double get() = ceil(0.25 / secondsPerBeat * 4) / 4

    val noteCount: Int get() = lines.sumOf { it.notes.size }

    // The runtime's score

    /** The score the runtime sings: {"text", "bpm", "style", "notes": [[label, note, beats], ...]}. */
    fun scoreJson(): String = score().json

    /**
     * A word over several notes is written as its syllables ("ba~1", "by~2") when each note has its own, else as the
     * whole word on every part ("home~1", "home~2", a syllable held over two pitches). Lines are joined by their rests.
     */
    fun score(): ScoreRows {
        val rows = JSONArray()
        val rowNotes = mutableListOf<String?>()
        val words = mutableListOf<String>()
        for ((lineIndex, line) in lines.withIndex()) {
            var index = 0
            while (index < line.notes.size) {
                val note = line.notes[index]
                if (note.isRest) {
                    rows.put(JSONArray().put("-").put("rest").put(note.beats))
                    rowNotes.add(note.id)
                    index += 1
                    continue
                }
                val group = mutableListOf(note)
                var next = index + 1
                while (next < line.notes.size && !line.notes[next].isRest && line.notes[next].continuesWord) {
                    group.add(line.notes[next])
                    next += 1
                }
                val syllables = group.map { clean(it.syllable) }
                val word = syllables.joinToString("")
                val spelled = syllables.none { it.isEmpty() } && syllables.toSet().size > 1
                words.add(word.ifEmpty { "la" })
                for ((part, member) in group.withIndex()) {
                    val label = if (group.size == 1) {
                        word.ifEmpty { "la" }
                    } else {
                        "${if (spelled) syllables[part] else word.ifEmpty { "la" }}~${part + 1}"
                    }
                    rows.put(JSONArray().put(label).put(noteName(member.midi ?: 60)).put(member.beats))
                    rowNotes.add(member.id)
                }
                index = next
            }
            if (lineIndex < lines.size - 1) {
                rows.put(JSONArray().put("-").put("rest").put(maxOf(line.restAfter, minimumLineRest)))
                rowNotes.add(null)
            }
        }
        val score = JSONObject()
            .put("text", words.joinToString(" "))
            .put("bpm", bpm)
            .put("style", style.id)
            .put("notes", rows)
        return ScoreRows(score.toString(), rowNotes)
    }

    // Saving

    /** The whole document (ids, decoder and transpose included) for saving. */
    fun toJson(): JSONObject {
        val linesJson = JSONArray()
        for (line in lines) {
            val notesJson = JSONArray()
            for (note in line.notes) {
                val noteJson = JSONObject()
                    .put("id", note.id)
                    .put("syllable", note.syllable)
                    .put("beats", note.beats)
                    .put("continuesWord", note.continuesWord)
                if (note.midi != null) noteJson.put("midi", note.midi)
                notesJson.put(noteJson)
            }
            linesJson.put(JSONObject().put("id", line.id).put("notes", notesJson).put("restAfter", line.restAfter))
        }
        return JSONObject()
            .put("id", id)
            .put("title", title)
            .put("bpm", bpm)
            .put("style", style.id)
            .put("voice", voice)
            .put("decoder", decoder.id)
            .put("transpose", transpose)
            .put("lines", linesJson)
    }

    companion object {
        fun blank(voice: String): SongDocument = SongDocument(
            title = "New song", bpm = 104.0, style = SongStyle.POP, voice = voice,
            lines = listOf(SongLine(notes = listOf(SongNote(syllable = "la", midi = 60, beats = 1.0)))),
        )

        /** The inverse of [toJson]; missing optional fields take their defaults. */
        fun fromJson(json: JSONObject): SongDocument {
            val linesJson = json.getJSONArray("lines")
            val lines = (0 until linesJson.length()).map { l ->
                val lineJson = linesJson.getJSONObject(l)
                val notesJson = lineJson.getJSONArray("notes")
                val notes = (0 until notesJson.length()).map { n ->
                    val noteJson = notesJson.getJSONObject(n)
                    SongNote(
                        id = noteJson.optString("id", "").ifEmpty { newSongId() },
                        syllable = noteJson.optString("syllable", ""),
                        midi = if (noteJson.isNull("midi")) null else noteJson.getInt("midi"),
                        beats = noteJson.getDouble("beats"),
                        continuesWord = noteJson.optBoolean("continuesWord", false),
                    )
                }
                SongLine(
                    id = lineJson.optString("id", "").ifEmpty { newSongId() },
                    notes = notes,
                    restAfter = lineJson.optDouble("restAfter", 1.0),
                )
            }
            return SongDocument(
                id = json.optString("id", "").ifEmpty { newSongId() },
                title = json.getString("title"),
                bpm = json.getDouble("bpm"),
                style = SongStyle.fromId(json.optString("style", "pop")),
                voice = json.getString("voice"),
                decoder = SongDecoder.fromId(json.optString("decoder", "quality")),
                transpose = json.optInt("transpose", 0),
                lines = lines,
            )
        }

        /**
         * A song from the runtime's normalized score (`ScyllasBand.parseScore`): its lines become lines, each closing
         * rest becomes the line's `restAfter`, and words over several notes become syllable notes.
         */
        fun fromSummary(summary: JSONObject, title: String, voice: String): SongDocument? {
            val notes = summary.optJSONArray("notes") ?: return null
            val bpm = (summary.opt("bpm") as? Number)?.toDouble() ?: return null
            if (notes.length() == 0) return null
            val styleId = summary.opt("style") as? String ?: "pop"
            val style = SongStyle.entries.firstOrNull { it.id == styleId }
                ?: if (summary.opt("mode") == "ballad") SongStyle.BALLAD else SongStyle.POP
            val linesJson = summary.optJSONArray("lines")
            val ranges: List<IntRange> = if (linesJson == null) {
                listOf(0 until notes.length())
            } else {
                (0 until linesJson.length()).mapNotNull { i ->
                    val line = linesJson.optJSONObject(i) ?: return@mapNotNull null
                    val first = (line.opt("first") as? Number)?.toInt() ?: return@mapNotNull null
                    val last = (line.opt("last") as? Number)?.toInt() ?: return@mapNotNull null
                    if (first < 0 || first > last || last >= notes.length()) null else first..last
                }
            }
            fun noteAt(i: Int) = notes.getJSONObject(i)
            fun intOf(note: JSONObject, key: String) = (note.opt(key) as? Number)?.toInt()

            // The syllable each note carries: a word's labels name its syllables ("ba~1", "by~2"), or the whole word on
            // every part ("home~1", "home~2"; the first note carries the word, the rest hold it).
            val syllables = Array(notes.length()) { "" }
            var index = 0
            while (index < notes.length()) {
                val word = intOf(noteAt(index), "word_index")
                if (word == null) {
                    index += 1
                    continue
                }
                val members = mutableListOf(index)
                var next = index + 1
                while (next < notes.length() && intOf(noteAt(next), "word_index") == word) {
                    members.add(next)
                    next += 1
                }
                val bases = members.map { baseOf(noteAt(it).optString("label", "")) }
                if (bases.toSet().size == 1) {
                    syllables[members[0]] = bases[0]
                } else {
                    for ((member, part) in members.zip(bases)) syllables[member] = part
                }
                index = next
            }
            val lines = mutableListOf<SongLine>()
            for ((lineIndex, range) in ranges.withIndex()) {
                val lineNotes = mutableListOf<SongNote>()
                for (position in range) {
                    val note = noteAt(position)
                    val beats = (note.opt("beats") as? Number)?.toDouble() ?: 1.0
                    val midi = intOf(note, "midi")
                    if (midi != null) {
                        val part = intOf(note, "part") ?: 1
                        lineNotes.add(SongNote(syllable = syllables[position], midi = midi, beats = beats, continuesWord = part > 1))
                    } else {
                        lineNotes.add(SongNote.rest(beats))
                    }
                }
                // The rests that close a line (every line but the last) become its restAfter.
                var closing = 0.0
                while (lineIndex < ranges.size - 1 && lineNotes.isNotEmpty() && lineNotes.last().isRest) {
                    closing += lineNotes.removeAt(lineNotes.size - 1).beats
                }
                lines.add(SongLine(notes = lineNotes, restAfter = if (closing > 0) closing else 1.0))
            }
            if (lines.isEmpty()) return null
            return SongDocument(title = title, bpm = bpm, style = style, voice = voice, lines = lines)
        }

        // Lyrics

        /**
         * Rewrites a line's sung notes from typed lyrics (words by spaces, syllables by "-", "_" holds the previous syllable
         * over another note), keeping the existing notes' pitches and lengths in order and the rests where they are.
         */
        fun applyingLyrics(lyrics: String, line: SongLine, style: SongStyle): SongLine {
            val syllables = mutableListOf<Pair<String, Boolean>>()   // text, continues the previous word
            for (word in lyrics.split(Regex("\\s+")).filter { it.isNotEmpty() }) {
                var first = true
                val pending = StringBuilder()
                fun flush() {
                    if (pending.isEmpty()) return
                    syllables.add(pending.toString() to !first)
                    first = false
                    pending.clear()
                }
                for (character in word) {
                    when (character) {
                        '-' -> flush()
                        '_' -> {
                            flush()
                            syllables.add("" to (!first || syllables.isNotEmpty()))
                            first = false
                        }
                        else -> pending.append(character)
                    }
                }
                flush()
            }
            val result = mutableListOf<SongNote>()
            var used = 0
            var lastPitch = line.notes.firstOrNull { !it.isRest }?.midi ?: 60
            for (note in line.notes) {
                if (note.isRest) {
                    result.add(note)
                    continue
                }
                if (used >= syllables.size) continue   // fewer syllables: the extra notes go
                val updated = note.copy(syllable = syllables[used].first, continuesWord = syllables[used].second)
                lastPitch = updated.midi ?: lastPitch
                result.add(updated)
                used += 1
            }
            while (used < syllables.size) {   // more syllables: new notes continue the line's last pitch
                result.add(
                    SongNote(syllable = syllables[used].first, midi = lastPitch, beats = style.defaultBeats,
                        continuesWord = syllables[used].second),
                )
                used += 1
            }
            val firstSung = result.indexOfFirst { !it.isRest }
            if (firstSung >= 0) result[firstSung] = result[firstSung].copy(continuesWord = false)
            return line.copy(notes = result)
        }

        // Helpers

        fun noteName(midi: Int): String {
            val names = listOf("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
            val octave = floor(midi / 12.0).toInt() - 1
            return names[Math.floorMod(midi, 12)] + octave
        }

        /** A syllable as a score label allows it: no spaces, no "~", never the rest label "-". */
        private fun clean(text: String): String {
            val kept = text.filter { !it.isWhitespace() && it != '~' }
            return if (kept == "-") "" else kept
        }

        private fun baseOf(label: String): String = label.substringBefore('~')
    }
}
