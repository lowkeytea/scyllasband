package org.scyllasband.demo.sing

import android.content.Context
import android.content.Intent
import android.net.Uri
import androidx.core.content.FileProvider
import org.json.JSONObject
import java.io.File

/** Songs saved in the app's files (one JSON file per song) and the files a song is shared as. */
class SongStore(private val context: Context) {
    private val songsDirectory: File
        get() = File(context.filesDir, "songs").apply { mkdirs() }

    private val exportDirectory: File
        get() = File(context.cacheDir, SHARED_DIRECTORY).apply { mkdirs() }

    /** Saved songs, newest first. */
    fun saved(): List<SongDocument> =
        songsDirectory.listFiles { file -> file.extension == "json" }.orEmpty()
            .sortedByDescending { it.lastModified() }
            .mapNotNull { file -> runCatching { SongDocument.fromJson(JSONObject(file.readText())) }.getOrNull() }

    fun save(song: SongDocument) {
        val target = File(songsDirectory, "${song.id}.json")
        val temporary = File(songsDirectory, "${song.id}.json.saving")
        temporary.writeText(song.toJson().toString(2))
        check(temporary.renameTo(target)) { "Unable to write ${target.name}" }
    }

    fun delete(song: SongDocument) {
        File(songsDirectory, "${song.id}.json").delete()
    }

    /**
     * A share intent for the song's score (the runtime's format, which `python -m scyllasband sing --score` also reads) and,
     * when there is one, its last take as a 16-bit WAV.
     */
    fun shareIntent(song: SongDocument, take: FloatArray?, sampleRate: Int): Intent {
        val name = SongFiles.fileName(song.title)
        val uris = arrayListOf(uriFor(File(exportDirectory, "$name.json").apply { writeText(song.scoreJson()) }))
        if (take != null) {
            uris += uriFor(File(exportDirectory, "$name.wav").apply { writeBytes(SongFiles.wav(take, sampleRate)) })
        }
        val send = if (uris.size == 1) {
            Intent(Intent.ACTION_SEND).putExtra(Intent.EXTRA_STREAM, uris[0]).setType("application/json")
        } else {
            Intent(Intent.ACTION_SEND_MULTIPLE).putParcelableArrayListExtra(Intent.EXTRA_STREAM, uris).setType("*/*")
        }
        return send.putExtra(Intent.EXTRA_SUBJECT, song.title).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }

    private fun uriFor(file: File): Uri = FileProvider.getUriForFile(context, "${context.packageName}.files", file)

    companion object {
        /** The cache subdirectory the app's FileProvider shares (res/xml/shared_files.xml). */
        const val SHARED_DIRECTORY = "shared"
    }
}
