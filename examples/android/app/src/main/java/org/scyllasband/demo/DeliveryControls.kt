package org.scyllasband.demo

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.SeekBar
import android.widget.TextView
import com.google.android.material.switchmaterial.SwitchMaterial
import org.scyllasband.android.ScyllasBandDelivery

/** The four 0-4 delivery sliders (0.1 steps), the whisper switch and the quick presets of the point dialog. */
class DeliveryControls(root: View, inflater: LayoutInflater) {
    private class AxisSlider(val bar: SeekBar, val value: TextView)

    private val sliders: Map<String, AxisSlider>
    private val whisperSwitch: SwitchMaterial = root.findViewById(R.id.whisperSwitch)

    var delivery: ScyllasBandDelivery
        get() = ScyllasBandDelivery(
            energy = axis("energy"),
            tension = axis("tension"),
            valence = axis("valence"),
            assertiveness = axis("assertiveness"),
            whisper = whisperSwitch.isChecked,
        )
        set(value) {
            setAxis("energy", value.energy)
            setAxis("tension", value.tension)
            setAxis("valence", value.valence)
            setAxis("assertiveness", value.assertiveness)
            whisperSwitch.isChecked = value.whisper
        }

    init {
        val panel = root.findViewById<LinearLayout>(R.id.sliderPanel)
        sliders = AXES.associate { (axis, labelResource) ->
            val row = inflater.inflate(R.layout.row_delivery_slider, panel, false)
            row.findViewById<TextView>(R.id.sliderLabel).setText(labelResource)
            val slider = AxisSlider(row.findViewById(R.id.slider), row.findViewById(R.id.sliderValue))
            slider.bar.max = MAX_PROGRESS
            slider.bar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                    slider.value.text = formatAxis(axisValue(progress))
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

    private fun axis(name: String): Float = axisValue(sliders.getValue(name).bar.progress)

    private fun setAxis(name: String, value: Float) {
        val slider = sliders.getValue(name)
        slider.bar.progress = Math.round((value.coerceIn(DELIVERY_UI_MIN, DELIVERY_UI_MAX) - DELIVERY_UI_MIN) * STEPS_PER_UNIT)
        slider.value.text = formatAxis(axisValue(slider.bar.progress))
    }

    private companion object {
        const val STEPS_PER_UNIT = 10f
        val MAX_PROGRESS = Math.round((DELIVERY_UI_MAX - DELIVERY_UI_MIN) * STEPS_PER_UNIT)

        fun axisValue(progress: Int): Float = DELIVERY_UI_MIN + progress / STEPS_PER_UNIT
        val AXES = listOf(
            "energy" to R.string.axis_energy,
            "tension" to R.string.axis_tension,
            "valence" to R.string.axis_valence,
            "assertiveness" to R.string.axis_assertiveness,
        )
    }
}
