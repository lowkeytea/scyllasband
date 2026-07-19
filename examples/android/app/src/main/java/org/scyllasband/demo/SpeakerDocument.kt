package org.scyllasband.demo

import org.scyllasband.android.ScyllasBandSegmentSettings

data class SpeakerPoint(
    val offset: Int,
    val settings: ScyllasBandSegmentSettings,
)

data class SpeakerDocumentSnapshot(
    val text: String,
    val points: List<SpeakerPoint>,
)

data class SpeechSegment(
    val text: String,
    val settings: ScyllasBandSegmentSettings,
)

object SpeakerDocument {
    fun segments(
        snapshot: SpeakerDocumentSnapshot,
        defaultSettings: ScyllasBandSegmentSettings,
    ): List<SpeechSegment> {
        val output = mutableListOf<SpeechSegment>()
        var cursor = 0
        var activeSettings = defaultSettings
        snapshot.points
            .map { it.copy(offset = it.offset.coerceIn(0, snapshot.text.length)) }
            .sortedBy { it.offset }
            .forEach { point ->
                appendSegment(output, snapshot.text.substring(cursor, point.offset), activeSettings)
                cursor = point.offset
                activeSettings = point.settings
            }
        appendSegment(output, snapshot.text.substring(cursor), activeSettings)
        return output
    }

    private fun appendSegment(
        output: MutableList<SpeechSegment>,
        text: String,
        settings: ScyllasBandSegmentSettings,
    ) {
        val speakable = text.trim()
        if (speakable.isNotEmpty()) {
            output += SpeechSegment(speakable, settings)
        }
    }
}
