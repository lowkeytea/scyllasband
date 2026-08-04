import SwiftUI
import ScyllasBandKit

struct SegmentSettingsView: View {
    @Binding var settings: SegmentSettings
    let bundleInfo: SBScyllasBandBundleInfo
    @Environment(\.dismiss) private var dismiss

    private var selectedVoice: SBScyllasBandVoice {
        bundleInfo.voices.first { $0.identifier == settings.voiceIdentifier } ?? bundleInfo.voices[0]
    }

    private var emotionSelection: Binding<String> {
        Binding(
            get: { settings.emotion ?? "" },
            set: { settings.emotion = $0.isEmpty ? nil : $0 }
        )
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
                    .onChange(of: settings.voiceIdentifier) { _ in
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
                    Picker("Emotion", selection: emotionSelection) {
                        Text("Neutral").tag("")
                        ForEach(bundleInfo.affectAxes, id: \.self) { emotion in
                            Text(emotion.capitalized).tag(emotion)
                        }
                    }
                    if settings.emotion != nil {
                        VStack(alignment: .leading) {
                            Text("Strength · \(Int(settings.emotionStrength * 100))%")
                            Slider(value: $settings.emotionStrength, in: 0...1)
                        }
                    }
                    TextField("Emotion CFG", value: $settings.emotionCFG, format: .number)
                        .keyboardType(.decimalPad)
                } header: {
                    Text("Delivery")
                } footer: {
                    Text("CFG 0 selects the learned null-affect branch, 1 applies the requested emotion directly, and values above 1 amplify it. High values can sound unstable.")
                }
            }
            .navigationTitle("Speaker settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") {
                        settings.emotionStrength = min(max(settings.emotionStrength, 0), 1)
                        if !settings.emotionCFG.isFinite || settings.emotionCFG < 0 {
                            settings.emotionCFG = 1
                        }
                        dismiss()
                    }
                }
            }
        }
    }
}
