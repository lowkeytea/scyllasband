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

                Section("Presets") {
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 92), spacing: 8)], spacing: 8) {
                        ForEach(DeliveryPreset.allCases) { preset in
                            presetButton(preset)
                        }
                    }
                    .padding(.vertical, 4)
                }

                Section {
                    ForEach(DeliveryAxis.allCases) { axis in
                        DeliverySlider(axis: axis, value: $settings.delivery[dynamicMember: axis.keyPath])
                    }
                    Toggle("Whisper", isOn: $settings.delivery.whisper)
                } header: {
                    Text("Delivery")
                } footer: {
                    Text("2 is neutral. The model accepts 0 to 4, but this app keeps each control between 0.5 and 3.1, because past about 3 the training data thins out and the voice distorts. How far a control moves the delivery varies with voice, language, and text, so start near 2 and listen as you move away from it.")
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

    @ViewBuilder
    private func presetButton(_ preset: DeliveryPreset) -> some View {
        let button = Button {
            settings.delivery = preset.delivery
        } label: {
            Text(preset.title)
                .frame(maxWidth: .infinity)
        }
        .accessibilityHint("Applies the \(preset.title.lowercased()) delivery")
        if settings.delivery == preset.delivery {
            button.buttonStyle(.borderedProminent)
                .accessibilityAddTraits(.isSelected)
        } else {
            button.buttonStyle(.bordered)
        }
    }
}

private struct DeliverySlider: View {
    let axis: DeliveryAxis
    @Binding var value: Float

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(axis.title)
                Spacer()
                Text(value, format: .number.precision(.fractionLength(1)))
                    .monospacedDigit()
                    .foregroundStyle(value == DeliverySettings.neutralValue ? Color(uiColor: .secondaryLabel) : .accentColor)
            }
            Slider(value: $value, in: DeliverySettings.valueRange, step: 0.1) {
                Text(axis.title)
            }
            HStack {
                Text(axis.range.low)
                Spacer()
                Text(axis.range.high)
            }
            .font(.caption2)
            .foregroundStyle(Color(uiColor: .secondaryLabel))
            .accessibilityHidden(true)
        }
        .accessibilityElement(children: .contain)
        .accessibilityHint("Lower: \(axis.range.low). Higher: \(axis.range.high).")
    }
}
