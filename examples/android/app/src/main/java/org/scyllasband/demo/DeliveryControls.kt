package org.scyllasband.demo

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.SeekBar
import android.widget.TextView
import org.scyllasband.android.ScyllasBandDelivery

/** The four delivery sliders (0.1 steps, each over its supported range) and the quick presets of the point dialog. */
class DeliveryControls(root: View, inflater: LayoutInflater) {
    private class AxisSlider(val bar: SeekBar, val value: TextView, val range: ClosedFloatingPointRange<Float>)

    private val sliders: Map<String, AxisSlider>

    var delivery: ScyllasBandDelivery
        get() = ScyllasBandDelivery(
            energy = axis("energy"),
            tension = axis("tension"),
            valence = axis("valence"),
            assertiveness = axis("assertiveness"),
        )
        set(value) {
            setAxis("energy", value.energy)
            setAxis("tension", value.tension)
            setAxis("valence", value.valence)
            setAxis("assertiveness", value.assertiveness)
        }

    init {
        val panel = root.findViewById<LinearLayout>(R.id.sliderPanel)
        sliders = AXES.associate { (axis, labelResource) ->
            val row = inflater.inflate(R.layout.row_delivery_slider, panel, false)
            row.findViewById<TextView>(R.id.sliderLabel).setText(labelResource)
            val range = DELIVERY_UI_RANGES.getValue(axis)
            val slider = AxisSlider(row.findViewById(R.id.slider), row.findViewById(R.id.sliderValue), range)
            slider.bar.max = Math.round((range.endInclusive - range.start) * STEPS_PER_UNIT)
            slider.bar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                    slider.value.text = formatAxis(axisValue(slider, progress))
                }

                override fun onStartTrackingTouch(seekBar: SeekBar?) = Unit
                override fun onStopTrackingTouch(seekBar: SeekBar?) = Unit
            })
            panel.addView(row)
            axis to slider
        }
        val presetRow = root.findViewById<ViewGroup>(R.id.presetRow)
        DeliveryPreset.entries.forEach { preset ->
            val button = inflater.inflate(R.layout.button_preset, presetRow, false) as Button
            button.setText(preset.titleResource)
            button.setOnClickListener { delivery = preset.delivery }
            presetRow.addView(button)
        }
    }

    private fun axis(name: String): Float = sliders.getValue(name).let { axisValue(it, it.bar.progress) }

    private fun setAxis(name: String, value: Float) {
        val slider = sliders.getValue(name)
        slider.bar.progress = Math.round((value.coerceIn(slider.range) - slider.range.start) * STEPS_PER_UNIT)
        slider.value.text = formatAxis(axisValue(slider, slider.bar.progress))
    }

    private companion object {
        const val STEPS_PER_UNIT = 10f

        // Rounded to the slider's 0.1 grid, so 1.4 + 0.9 reads back as 2.3.
        fun axisValue(slider: AxisSlider, progress: Int): Float =
            Math.round((slider.range.start + progress / STEPS_PER_UNIT) * STEPS_PER_UNIT) / STEPS_PER_UNIT
        val AXES = listOf(
            "energy" to R.string.axis_energy,
            "tension" to R.string.axis_tension,
            "valence" to R.string.axis_valence,
            "assertiveness" to R.string.axis_assertiveness,
        )
    }
}
