package org.scyllasband.demo.sing

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.floor

/** An example score embedded with the app (assets scyllasband/melodies, the singing release's examples). */
data class ExampleScore(val assetName: String) {
    private val stem: String
        get() = assetName.substringAfterLast('/').let { if (it.contains('.')) it.substringBeforeLast('.') else it }

    /** song_pop_gwen -> "Pop song · Gwen"; phrase_ballad_03 -> "Ballad phrase 4". */
    val title: String
        get() {
            val parts = stem.split("_").filter { it.isNotEmpty() }
            if (parts.size == 3 && parts[0] == "song") return "${capitalized(parts[1])} song · ${capitalized(parts[2])}"
            val number = parts.getOrNull(2)?.toIntOrNull()
            if (parts.size == 3 && parts[0] == "phrase" && number != null) return "${capitalized(parts[1])} phrase ${number + 1}"
            return capitalized(stem.replace("_", " "))
        }

    /** The voice a whole-song example was written for (song_pop_gwen -> gwen). */
    val voice: String?
        get() {
            val parts = stem.split("_").filter { it.isNotEmpty() }
            return if (parts.size == 3 && parts[0] == "song") parts[2] else null
        }

    val isSong: Boolean get() = stem.startsWith("song_")

    companion object {
        /** The .json names as examples: songs first, then by stem. */
        fun sorted(names: List<String>): List<ExampleScore> =
            names.filter { it.endsWith(".json") }
                .map(::ExampleScore)
                .sortedWith(compareBy<ExampleScore> { if (it.isSong) 0 else 1 }.thenBy { it.stem })

        /** Each word's first letter upper case, the rest lower case (Foundation's capitalized). */
        private fun capitalized(text: String): String =
            text.split(" ").joinToString(" ") { word ->
                word.lowercase().replaceFirstChar { it.titlecase() }
            }
    }
}

object SongFiles {
    /** Mono Float samples as a 16-bit PCM WAV file. */
    fun wav(samples: FloatArray, sampleRate: Int): ByteArray {
        val count = samples.size
        val buffer = ByteBuffer.allocate(44 + count * 2).order(ByteOrder.LITTLE_ENDIAN)
        buffer.put("RIFF".toByteArray(Charsets.US_ASCII))
        buffer.putInt(36 + count * 2)
        buffer.put("WAVEfmt ".toByteArray(Charsets.US_ASCII))
        buffer.putInt(16)
        buffer.putShort(1)                    // PCM
        buffer.putShort(1)                    // mono
        buffer.putInt(sampleRate)
        buffer.putInt(sampleRate * 2)         // bytes per second
        buffer.putShort(2)                    // block align
        buffer.putShort(16)                   // bits per sample
        buffer.put("data".toByteArray(Charsets.US_ASCII))
        buffer.putInt(count * 2)
        for (value in samples) {
            val scaled = (if (value.isNaN()) 0f else value.coerceIn(-1f, 1f)) * 32767.0
            // Round half away from zero, as Swift's rounded().
            val rounded = if (scaled < 0) -floor(abs(scaled) + 0.5) else floor(scaled + 0.5)
            buffer.putShort(rounded.toInt().toShort())
        }
        return buffer.array()
    }

    /** The title as a file name: letters, digits and spaces only, trimmed; "Song" when nothing is left. */
    fun fileName(title: String): String {
        val kept = StringBuilder()
        title.codePoints().forEach { if (isAllowed(it)) kept.appendCodePoint(it) }
        val cleaned = kept.toString().trim { isSpaceOrTab(it.code) }
        return cleaned.ifEmpty { "Song" }
    }

    private fun isSpaceOrTab(codePoint: Int): Boolean =
        codePoint == '\t'.code || Character.getType(codePoint) == Character.SPACE_SEPARATOR.toInt()

    private fun isAllowed(codePoint: Int): Boolean {
        if (isSpaceOrTab(codePoint) || Character.isLetterOrDigit(codePoint)) return true
        return when (Character.getType(codePoint).toByte()) {
            Character.NON_SPACING_MARK, Character.COMBINING_SPACING_MARK, Character.ENCLOSING_MARK,
            Character.LETTER_NUMBER, Character.OTHER_NUMBER -> true
            else -> false
        }
    }
}
