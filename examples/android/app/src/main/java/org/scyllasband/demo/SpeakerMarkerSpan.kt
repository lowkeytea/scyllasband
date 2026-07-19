package org.scyllasband.demo

import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.text.style.ReplacementSpan
import org.scyllasband.android.ScyllasBandSegmentSettings
import java.util.Locale

class SpeakerMarkerSpan(
    var settings: ScyllasBandSegmentSettings,
    private val density: Float,
) : ReplacementSpan() {
    private val horizontalPadding = 9f * density
    private val verticalPadding = 3f * density
    private val radius = 10f * density

    val label: String
        get() {
            val affect = settings.emotion?.let {
                "${it.lowercase()} ${(settings.emotionStrength * 100).toInt()}%"
            } ?: "neutral"
            return "${settings.voiceId} · ${settings.language} · $affect · ${formatCfg(settings.emotionCfg)}×"
        }

    override fun getSize(
        paint: Paint,
        text: CharSequence?,
        start: Int,
        end: Int,
        fm: Paint.FontMetricsInt?,
    ): Int {
        fm?.let {
            val source = paint.fontMetricsInt
            val padding = verticalPadding.toInt()
            it.ascent = source.ascent - padding
            it.descent = source.descent + padding
            it.top = it.ascent
            it.bottom = it.descent
        }
        return (paint.measureText(label) + horizontalPadding * 2).toInt()
    }

    override fun draw(
        canvas: Canvas,
        text: CharSequence?,
        start: Int,
        end: Int,
        x: Float,
        top: Int,
        y: Int,
        bottom: Int,
        paint: Paint,
    ) {
        val previousColor = paint.color
        val previousStyle = paint.style
        val width = paint.measureText(label) + horizontalPadding * 2
        val metrics = paint.fontMetrics
        val textTop = y + metrics.ascent - verticalPadding
        val textBottom = y + metrics.descent + verticalPadding
        paint.color = 0xffd7e7ff.toInt()
        paint.style = Paint.Style.FILL
        canvas.drawRoundRect(RectF(x, textTop, x + width, textBottom), radius, radius, paint)
        paint.color = 0xff153b67.toInt()
        canvas.drawText(label, x + horizontalPadding, y.toFloat(), paint)
        paint.color = previousColor
        paint.style = previousStyle
    }

    private fun formatCfg(value: Float): String =
        String.format(Locale.US, if (value % 1f == 0f) "%.0f" else "%.2f", value)
}
