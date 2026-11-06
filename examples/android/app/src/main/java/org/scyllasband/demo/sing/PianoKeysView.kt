package org.scyllasband.demo.sing

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import com.google.android.material.color.MaterialColors

/** One octave of a piano keyboard; tapping a key reports its MIDI pitch through [onPick]. */
class PianoKeysView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {
    /** The octave shown (4: C4 to B4). */
    var octave: Int = 4
        set(value) {
            field = value
            contentDescription = "Piano keys, octave $value"
            invalidate()
        }

    /** The pitch drawn as selected, if it is in this octave. */
    var selected: Int? = null
        set(value) {
            field = value
            invalidate()
        }

    var onPick: (Int) -> Unit = {}

    private val accent = MaterialColors.getColor(this, com.google.android.material.R.attr.colorPrimary, Color.BLUE)
    private val fill = Paint(Paint.ANTI_ALIAS_FLAG)
    private val outline = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = Color.argb(150, 128, 128, 128)
        strokeWidth = resources.displayMetrics.density * 0.75f
    }
    private val label = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER
        textSize = 11f * resources.displayMetrics.scaledDensity
    }
    private val key = RectF()

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val width = width / 7f
        WHITES.forEachIndexed { index, semitone ->
            val midi = midiOf(semitone)
            key.set(index * width, 0f, (index + 1) * width, height.toFloat())
            fill.color = if (midi == selected) accent else Color.WHITE
            canvas.drawRect(key, fill)
            canvas.drawRect(key, outline)
            label.color = if (midi == selected) Color.WHITE else Color.rgb(77, 77, 77)
            canvas.drawText(SongDocument.noteName(midi), key.centerX(), height - label.textSize * 0.6f, label)
        }
        BLACKS.forEach { (semitone, afterWhite) ->
            blackKey(afterWhite, width)
            fill.color = if (midiOf(semitone) == selected) accent else Color.BLACK
            canvas.drawRect(key, fill)
            canvas.drawRect(key, outline)
        }
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (event.action == MotionEvent.ACTION_DOWN) return true
        if (event.action != MotionEvent.ACTION_UP) return super.onTouchEvent(event)
        val width = width / 7f
        val black = BLACKS.firstOrNull { (_, afterWhite) ->
            blackKey(afterWhite, width)
            key.contains(event.x, event.y)
        }
        val semitone = black?.first ?: WHITES[(event.x / width).toInt().coerceIn(0, 6)]
        performClick()
        onPick(midiOf(semitone))
        return true
    }

    override fun performClick(): Boolean = super.performClick()

    private fun midiOf(semitone: Int): Int = 12 * (octave + 1) + semitone

    private fun blackKey(afterWhite: Int, width: Float) {
        val left = width * (afterWhite + 1) - width * 0.31f
        key.set(left, 0f, left + width * 0.62f, height * 0.6f)
    }

    private companion object {
        val WHITES = listOf(0, 2, 4, 5, 7, 9, 11)
        /** (semitone, the white key it sits after). */
        val BLACKS = listOf(1 to 0, 3 to 1, 6 to 3, 8 to 4, 10 to 5)
    }
}
