package org.scyllasband.demo

import android.content.Context
import android.text.Spannable
import android.util.AttributeSet
import android.view.GestureDetector
import android.view.MotionEvent
import androidx.appcompat.widget.AppCompatEditText

class MarkerEditText @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = android.R.attr.editTextStyle,
) : AppCompatEditText(context, attrs, defStyleAttr) {
    var onAddMarkerRequested: ((Int) -> Unit)? = null
    var onEditMarkerRequested: ((SpeakerMarkerSpan) -> Unit)? = null
    private var pointGestureHandled = false

    private val gestures = GestureDetector(
        context,
        object : GestureDetector.SimpleOnGestureListener() {
            override fun onDown(event: MotionEvent): Boolean = true

            override fun onSingleTapConfirmed(event: MotionEvent): Boolean {
                markerAt(event)?.let {
                    onEditMarkerRequested?.invoke(it)
                    performClick()
                    return true
                }
                return false
            }

            override fun onDoubleTap(event: MotionEvent): Boolean {
                pointGestureHandled = true
                val offset = getOffsetForPosition(event.x, event.y).coerceAtLeast(0)
                markerAtOffset(offset)?.let {
                    onEditMarkerRequested?.invoke(it)
                    return true
                }
                onAddMarkerRequested?.invoke(offset)
                return true
            }

            override fun onLongPress(event: MotionEvent) {
                pointGestureHandled = true
                val offset = getOffsetForPosition(event.x, event.y).coerceAtLeast(0)
                markerAtOffset(offset)?.let {
                    onEditMarkerRequested?.invoke(it)
                    return
                }
                onAddMarkerRequested?.invoke(offset)
            }
        },
    )

    override fun onTouchEvent(event: MotionEvent): Boolean {
        pointGestureHandled = false
        gestures.onTouchEvent(event)
        return if (pointGestureHandled) true else super.onTouchEvent(event)
    }

    override fun performClick(): Boolean {
        super.performClick()
        return true
    }

    private fun markerAt(event: MotionEvent): SpeakerMarkerSpan? =
        markerAtOffset(getOffsetForPosition(event.x, event.y).coerceAtLeast(0))

    private fun markerAtOffset(offset: Int): SpeakerMarkerSpan? {
        val value = text as? Spannable ?: return null
        val bounded = offset.coerceIn(0, value.length)
        value.getSpans(bounded, bounded, SpeakerMarkerSpan::class.java).firstOrNull()?.let {
            return it
        }
        if (bounded > 0) {
            return value.getSpans(
                bounded - 1,
                bounded,
                SpeakerMarkerSpan::class.java,
            ).firstOrNull()
        }
        return null
    }
}
