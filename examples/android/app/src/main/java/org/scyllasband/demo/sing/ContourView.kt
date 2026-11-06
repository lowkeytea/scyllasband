package org.scyllasband.demo.sing

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import com.google.android.material.color.MaterialColors
import kotlin.math.max

/** A sketch of a line's melody: each sung note as a bar at its pitch, as long as it lasts; rests leave gaps. */
class ContourView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {
    private var notes: List<SongNote> = emptyList()
    private var selected: String? = null
    private var problem: String? = null

    private val accent = MaterialColors.getColor(this, com.google.android.material.R.attr.colorPrimary, Color.BLUE)
    private val error = MaterialColors.getColor(this, com.google.android.material.R.attr.colorError, Color.RED)
    private val muted = MaterialColors.getColor(this, com.google.android.material.R.attr.colorOnSurface, Color.GRAY)
    private val bar = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        strokeCap = Paint.Cap.ROUND
        strokeWidth = resources.displayMetrics.density * 5
    }

    fun setNotes(notes: List<SongNote>, selected: String?, problem: String?) {
        this.notes = notes
        this.selected = selected
        this.problem = problem
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val pitches = notes.mapNotNull { it.midi }
        if (pitches.isEmpty()) return
        val total = max(notes.sumOf { it.beats }, 0.25)
        val low = pitches.min()
        val span = max(pitches.max() - low, 4)
        val inset = bar.strokeWidth
        val usableHeight = height - 2 * inset
        var start = 0.0
        for (note in notes) {
            val midi = note.midi
            if (midi != null) {
                val x0 = (start / total * width).toFloat() + inset
                val x1 = ((start + note.beats) / total * width).toFloat() - inset
                val y = (inset + usableHeight * (1 - (midi - low).toFloat() / span)).toFloat()
                bar.color = when (note.id) {
                    problem -> error
                    selected -> accent
                    else -> Color.argb(110, Color.red(muted), Color.green(muted), Color.blue(muted))
                }
                canvas.drawLine(x0, y, max(x0, x1), y, bar)
            }
            start += note.beats
        }
    }
}
