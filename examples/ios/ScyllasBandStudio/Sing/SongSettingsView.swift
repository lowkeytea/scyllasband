import SwiftUI
import ScyllasBandKit

/// The song's settings: title, voice, style, tempo, the decoder it renders through, and its key (transpose, with how the
/// melody sits in the voice's range).
struct SongSettingsView: View {
    @ObservedObject var model: SongViewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section("Song") {
                    TextField("Title", text: binding(\.title))
                    Picker("Voice", selection: binding(\.voice)) {
                        ForEach(model.voices, id: \.identifier) { voice in
                            Text(voice.identifier.capitalized).tag(voice.identifier)
                        }
                    }
                    Picker("Style", selection: binding(\.style)) {
                        ForEach(SongStyle.allCases) { style in
                            Text(style.title).tag(style)
                        }
                    }
                    .pickerStyle(.segmented)
                    Stepper("Tempo \(Int(model.song.bpm)) bpm", value: binding(\.bpm), in: 40...200, step: 2)
                }
                Section {
                    Picker("Sound", selection: binding(\.decoder)) {
                        ForEach(SongDecoder.allCases) { decoder in
                            Text(decoder.title).tag(decoder)
                        }
                    }
                    .pickerStyle(.segmented)
                    .disabled(!model.hasBigVGAN)
                } header: {
                    Text("Sound")
                } footer: {
                    Text(model.hasBigVGAN
                         ? "Quality renders with BigVGAN, the singing decoder: cleanest on held notes, and the slowest. Turbo renders with Vocos, the speech decoder: much faster, with some harmonic artefacts on sustained notes in some voices."
                         : "This bundle has no BigVGAN decoder here; songs render with Vocos.")
                }
                Section {
                    Stepper("Transpose \(model.song.transpose > 0 ? "+" : "")\(model.song.transpose)", value: binding(\.transpose), in: -12...12)
                    if !model.fitSummary.isEmpty {
                        Text(model.fitSummary)
                            .font(.footnote)
                            .foregroundStyle(Color(uiColor: .secondaryLabel))
                    }
                    if let suggested = model.suggestedTranspose {
                        Button("Use transpose \(suggested > 0 ? "+" : "")\(suggested)") { model.applySuggestedTranspose() }
                    }
                } header: {
                    Text("Key")
                } footer: {
                    Text("The song is moved by octaves next to the voice’s range on its own; transpose moves it by semitones first, to put it in the voice’s key. A voice sings most accurately within a few semitones of its centre.")
                }
            }
            .navigationTitle("Song")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    /// A binding to one of the song's settings that rechecks the song after each change.
    private func binding<Value>(_ path: WritableKeyPath<SongDocument, Value>) -> Binding<Value> {
        Binding(get: { model.song[keyPath: path] }, set: { value in
            model.song[keyPath: path] = value
            model.songChanged()
        })
    }
}
