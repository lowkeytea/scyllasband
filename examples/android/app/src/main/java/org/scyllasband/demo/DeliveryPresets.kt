package org.scyllasband.demo

import java.util.Locale
import org.scyllasband.android.ScyllasBandDelivery

/**
 * The sliders and imported tags keep each control inside the range the runtime supports (-2 to 2): intensity and pace in
 * the voice's own standard deviations, pitch in semitones. 0 is the voice's own delivery.
 */
val DELIVERY_UI_RANGES = ScyllasBandDelivery.DEFAULT_RANGES

fun ScyllasBandDelivery.coerceToUiRange(): ScyllasBandDelivery = copy(
    intensity = intensity?.coerceIn(DELIVERY_UI_RANGES.getValue("intensity")),
    pace = pace?.coerceIn(DELIVERY_UI_RANGES.getValue("pace")),
    pitch = pitch?.coerceIn(DELIVERY_UI_RANGES.getValue("pitch")),
)

/** The point dialog's quick reset: every control back to the voice's own delivery. */
enum class DeliveryPreset(val titleResource: Int, val delivery: ScyllasBandDelivery) {
    DEFAULT(R.string.preset_default, ScyllasBandDelivery(intensity = 0f, pace = 0f, pitch = 0f)),
}

/** A short marker label for a delivery: "default", or the controls that differ from the voice's own. */
fun ScyllasBandDelivery.summary(): String {
    val parts = buildList {
        intensity?.takeIf { it != 0f }?.let { add("intensity ${formatAxis(it)}") }
        pace?.takeIf { it != 0f }?.let { add("pace ${formatAxis(it)}") }
        pitch?.takeIf { it != 0f }?.let { add("pitch ${formatAxis(it)}") }
    }
    return if (parts.isEmpty()) "default" else parts.joinToString(" ")
}

fun formatAxis(value: Float): String =
    String.format(Locale.US, "%.2f", value).trimEnd('0').trimEnd('.').let { if (it == "-0") "0" else it }
