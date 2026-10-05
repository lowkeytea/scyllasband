package org.scyllasband.demo

import java.util.Locale
import org.scyllasband.android.ScyllasBandDelivery

/**
 * The app keeps the four axes within 0.5-3.1 even though the model accepts 0-4: training data thins out past about 3
 * (tension reaches 3.1) and pushing further distorts the voice. The range holds every preset.
 */
const val DELIVERY_UI_MIN = 0.5f
const val DELIVERY_UI_MAX = 3.1f

fun ScyllasBandDelivery.coerceToUiRange(): ScyllasBandDelivery = copy(
    energy = energy.coerceIn(DELIVERY_UI_MIN, DELIVERY_UI_MAX),
    tension = tension.coerceIn(DELIVERY_UI_MIN, DELIVERY_UI_MAX),
    valence = valence.coerceIn(DELIVERY_UI_MIN, DELIVERY_UI_MAX),
    assertiveness = assertiveness.coerceIn(DELIVERY_UI_MIN, DELIVERY_UI_MAX),
)

/** Quick delivery presets: each is just a combination of the four axes and whisper. */
enum class DeliveryPreset(val titleResource: Int, val delivery: ScyllasBandDelivery) {
    NEUTRAL(R.string.preset_neutral, ScyllasBandDelivery()),
    CALM(R.string.preset_calm, ScyllasBandDelivery(energy = 1.8f, tension = 1f, valence = 2.5f, assertiveness = 2.1f)),
    ASSERTIVE(R.string.preset_assertive, ScyllasBandDelivery(energy = 2.6f, tension = 2.4f, valence = 2.1f, assertiveness = 2.8f)),
    JOYFUL(R.string.preset_joyful, ScyllasBandDelivery(energy = 2.4f, tension = 2f, valence = 2.8f, assertiveness = 2.2f)),
    ANGRY(
        R.string.preset_angry,
        ScyllasBandDelivery(energy = 2.7f, tension = 3.1f, valence = 1.3f, assertiveness = 2.5f),
    ),
    SAD(R.string.preset_sad, ScyllasBandDelivery(energy = 1.7f, tension = 1.7f, valence = 1.8f, assertiveness = 2f)),
    WHISPER(R.string.preset_whisper, ScyllasBandDelivery(whisper = true)),
}

/** A short marker label for a delivery: "neutral", or the axes that differ from neutral. */
fun ScyllasBandDelivery.summary(): String {
    val parts = buildList {
        if (energy != ScyllasBandDelivery.NEUTRAL) add("E ${formatAxis(energy)}")
        if (tension != ScyllasBandDelivery.NEUTRAL) add("T ${formatAxis(tension)}")
        if (valence != ScyllasBandDelivery.NEUTRAL) add("V ${formatAxis(valence)}")
        if (assertiveness != ScyllasBandDelivery.NEUTRAL) add("A ${formatAxis(assertiveness)}")
        if (whisper) add("whisper")
    }
    return if (parts.isEmpty()) "neutral" else parts.joinToString(" ")
}

fun formatAxis(value: Float): String =
    String.format(Locale.US, "%.2f", value).trimEnd('0').trimEnd('.')
