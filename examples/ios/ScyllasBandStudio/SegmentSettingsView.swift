import SwiftUI
import ScyllasBandKit

struct SegmentSettingsView: View {
    @Binding var settings: SegmentSettings
    let bundleInfo: SBScyllasBandBundleInfo
    @Environment(\.dismiss) private var dismiss

    private var selectedVoice: SBScyllasBandVoice {
        bundleInfo.voice(withIdentifier: settings.voiceIdentifier) ?? bundleInfo.voices[0]
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("Speaker") {
                    Picker("Voice", selection: $settings.voiceIdentifier) {
                        ForEach(bundleInfo.voices, id: \.identifier) { voice in
                            Text(voice.friendlyName).tag(voice.identifier)
                        }
                    }
                    .onChange(of: settings.voiceIdentifier) {
                        if !selectedVoice.languages.contains(settings.language) {
                            settings.language = selectedVoice.defaultLanguage
                        }
                    }

                    Picker("Language", selection: $settings.language) {
                        ForEach(selectedVoice.languages, id: \.self) { language in
                            Text(languageLabel(language)).tag(language)
                        }
                    }
                }

                Section {
                    ForEach(DeliveryControl.allCases) { control in
                        DeliverySlider(
                            control: control,
                            range: control.range(in: bundleInfo),
                            isVoiceDefault: deliveryValue(control) == control.defaultValue(for: selectedVoice),
                            value: deliveryBinding(control)
                        )
                    }
                    ForEach(DeliveryPreset.allCases) { preset in
                        presetButton(preset)
                    }
                } header: {
                    Text("Delivery")
                } footer: {
                    Text("0 is the voice’s own delivery. Intensity runs from calm (left) to intense (right) and pace from slower to faster, in the voice’s own range; pitch shifts the voice in semitones. The emotion comes from the text.")
                }
            }
            .navigationTitle("Speaker settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") {
                        settings.validate(using: bundleInfo)
                        dismiss()
                    }
                }
            }
        }
    }

    /// The control's value, or the voice's default when the segment leaves it unset.
    private func deliveryValue(_ control: DeliveryControl) -> Float {
        settings.delivery[keyPath: control.keyPath] ?? control.defaultValue(for: selectedVoice)
    }

    /// Moving a slider sets its control, on the slider's 0.1 grid (so -2 + 2.3 reads back as 0.3).
    private func deliveryBinding(_ control: DeliveryControl) -> Binding<Float> {
        Binding(
            get: { deliveryValue(control) },
            set: { settings.delivery[keyPath: control.keyPath] = ($0 * 10).rounded() / 10 }
        )
    }

    private func presetButton(_ preset: DeliveryPreset) -> some View {
        let applied = settings.delivery.resolved(for: selectedVoice) == preset.delivery.resolved(for: selectedVoice)
        return Button {
            settings.delivery = preset.delivery
        } label: {
            Label(preset.title, systemImage: preset.systemImage)
        }
        .disabled(applied)
        .accessibilityHint("Sets intensity, pace and pitch back to the voice’s own")
    }
}

private struct DeliverySlider: View {
    let control: DeliveryControl
    let range: ClosedRange<Float>
    let isVoiceDefault: Bool
    @Binding var value: Float

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(control.title)
                Spacer()
                Text(value, format: .number.precision(.fractionLength(1...2)).sign(strategy: .always(includingZero: false)))
                    .monospacedDigit()
                    .foregroundStyle(isVoiceDefault ? Color(uiColor: .secondaryLabel) : .accentColor)
            }
            Slider(value: $value, in: range, step: 0.1) {
                Text(control.title)
            }
            .accessibilityValue("\(DeliverySettings.format(value)) \(control.unit)")
            HStack {
                Text(control.ends.low)
                Spacer()
                Text(control.ends.high)
            }
            .font(.caption2)
            .foregroundStyle(Color(uiColor: .secondaryLabel))
            .accessibilityHidden(true)
        }
        .accessibilityElement(children: .contain)
        .accessibilityHint("Lower: \(control.ends.low.lowercased()). Higher: \(control.ends.high.lowercased()).")
    }
}
