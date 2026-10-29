package org.scyllasband.demo

import java.util.Locale
import org.scyllasband.android.ScyllasBandDelivery

/**
 * The sliders and imported tags keep each axis inside the range the runtime supports (the voices render most reliably
 * there; the runtime clamps any other request into it): energy 1.4-2.3, tension 1.0-2.4, valence 1.4-2.2 and
 * assertiveness 1.8-2.4, with 2 neutral in each. Every preset lies within them.
 */
val DELIVERY_UI_RANGES = mapOf(
    "energy" to 1.4f..2.3f,
    "tension" to 1.0f..2.4f,
    "valence" to 1.4f..2.2f,
    "assertiveness" to 1.8f..2.4f,
)

fun ScyllasBandDelivery.coerceToUiRange(): ScyllasBandDelivery = copy(
    energy = energy.coerceIn(DELIVERY_UI_RANGES.getValue("energy")),
    tension = tension.coerceIn(DELIVERY_UI_RANGES.getValue("tension")),
    valence = valence.coerceIn(DELIVERY_UI_RANGES.getValue("valence")),
    assertiveness = assertiveness.coerceIn(DELIVERY_UI_RANGES.getValue("assertiveness")),
)

/** Quick delivery presets, the same combinations as the main README. */
enum class DeliveryPreset(val titleResource: Int, val delivery: ScyllasBandDelivery) {
    NEUTRAL(R.string.preset_neutral, ScyllasBandDelivery()),
    CALM(R.string.preset_calm, ScyllasBandDelivery(energy = 1.8f, tension = 1f, valence = 2.2f, assertiveness = 2.1f)),
    ASSERTIVE(R.string.preset_assertive, ScyllasBandDelivery(energy = 2.3f, tension = 2.4f, valence = 2.1f, assertiveness = 2.4f)),
    JOYFUL(R.string.preset_joyful, ScyllasBandDelivery(energy = 2.3f, tension = 2f, valence = 2.2f, assertiveness = 2.2f)),
    ANGRY(R.string.preset_angry, ScyllasBandDelivery(energy = 2.3f, tension = 2.4f, valence = 1.4f, assertiveness = 2.4f)),
    SAD(R.string.preset_sad, ScyllasBandDelivery(energy = 1.7f, tension = 1.5f, valence = 1.4f, assertiveness = 2f)),
}

/** A short marker label for a delivery: "neutral", or the axes that differ from neutral. */
fun ScyllasBandDelivery.summary(): String {
    val parts = buildList {
        if (energy != ScyllasBandDelivery.NEUTRAL) add("E ${formatAxis(energy)}")
        if (tension != ScyllasBandDelivery.NEUTRAL) add("T ${formatAxis(tension)}")
        if (valence != ScyllasBandDelivery.NEUTRAL) add("V ${formatAxis(valence)}")
        if (assertiveness != ScyllasBandDelivery.NEUTRAL) add("A ${formatAxis(assertiveness)}")
    }
    return if (parts.isEmpty()) "neutral" else parts.joinToString(" ")
}

fun formatAxis(value: Float): String =
    String.format(Locale.US, "%.2f", value).trimEnd('0').trimEnd('.')
