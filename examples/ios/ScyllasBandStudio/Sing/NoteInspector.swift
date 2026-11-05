import SwiftUI

/// Edits the selected note: its syllable, pitch (a one-octave keyboard with octave steps) and length, and turns it into a
/// rest, splits, joins, inserts a rest or deletes it.
struct NoteInspector: View {
    @ObservedObject var model: SongViewModel
    @Environment(\.dismiss) private var dismiss
    @State private var octave = 4

    private static let lengths: [Double] = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4]

    var body: some View {
        NavigationStack {
            Group {
                if let note = model.note(model.selection) {
                    Form {
                        if !note.isRest {
                            Section {
                                TextField("Syllable", text: Binding(get: { note.syllable }, set: { model.setSyllable($0) }))
                                    .textInputAutocapitalization(.never)
                                    .autocorrectionDisabled()
                            } header: {
                                Text("Syllable")
                            } footer: {
                                Text(note.continuesWord ? "Continues the previous note’s word. Empty holds its syllable over this pitch." : "Starts a word.")
                            }
                            Section("Pitch · \(note.midi.map(SongDocument.noteName) ?? "")") {
                                PianoKeys(octave: octave, selected: note.midi) { model.setPitch($0) }
                                    .frame(height: 120)
                                Stepper("Octave \(octave)", value: $octave, in: 1...6)
                            }
                        }
                        Section("Length · \(LineCard.beatsTitle(note.beats))") {
                            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 6), count: 4), spacing: 6) {
                                ForEach(Self.lengths, id: \.self) { beats in
                                    Button {
                                        model.setBeats(beats)
                                    } label: {
                                        Text(NoteChip.lengthSymbol(beats))
                                            .font(.callout.monospaced())
                                            .frame(maxWidth: .infinity, minHeight: 36)
                                    }
                                    .buttonStyle(.bordered)
                                    .tint(note.beats == beats ? .accentColor : .secondary)
                                    .accessibilityLabel(LineCard.beatsTitle(beats))
                                }
                            }
                        }
                        Section {
                            Button(note.isRest ? "Make it a note" : "Make it a rest", systemImage: note.isRest ? "music.note" : "pause") {
                                model.toggleRest()
                            }
                            if !note.isRest {
                                Button("Split: hold the syllable over two pitches", systemImage: "square.split.2x1") { model.splitSelected() }
                            }
                            Button("Join with the next note", systemImage: "arrow.right.to.line") { model.mergeWithNext() }
                            Button("Insert a rest after", systemImage: "plus.square.dashed") { model.insertRestAfter() }
                            Button("Delete the note", systemImage: "trash", role: .destructive) {
                                model.deleteSelected()
                                dismiss()
                            }
                        }
                    }
                    .onAppear {
                        if let midi = note.midi { octave = midi / 12 - 1 }
                    }
                } else {
                    Text("Select a note.")
                        .foregroundStyle(Color(uiColor: .secondaryLabel))
                }
            }
            .navigationTitle("Note")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

/// One octave of a piano keyboard; tapping a key picks its MIDI pitch.
struct PianoKeys: View {
    let octave: Int
    let selected: Int?
    let pick: (Int) -> Void

    private static let whites = [0, 2, 4, 5, 7, 9, 11]
    private static let blacks: [(semitone: Int, afterWhite: Int)] = [(1, 0), (3, 1), (6, 3), (8, 4), (10, 5)]

    var body: some View {
        GeometryReader { geometry in
            let width = geometry.size.width / 7
            ZStack(alignment: .topLeading) {
                HStack(spacing: 0) {
                    ForEach(Self.whites, id: \.self) { semitone in
                        key(semitone, black: false)
                            .frame(width: width, height: geometry.size.height)
                    }
                }
                ForEach(Self.blacks, id: \.semitone) { black in
                    key(black.semitone, black: true)
                        .frame(width: width * 0.62, height: geometry.size.height * 0.6)
                        .offset(x: width * CGFloat(black.afterWhite + 1) - width * 0.31)
                }
            }
        }
    }

    private func key(_ semitone: Int, black: Bool) -> some View {
        let midi = 12 * (octave + 1) + semitone
        let isSelected = midi == selected
        return Button {
            pick(midi)
        } label: {
            ZStack(alignment: .bottom) {
                // A piano's colours in light and dark mode alike.
                Rectangle()
                    .fill(isSelected ? Color.accentColor : (black ? Color.black : Color.white))
                Rectangle()
                    .stroke(Color.gray.opacity(0.6), lineWidth: 0.5)
                if !black {
                    Text(SongDocument.noteName(midi))
                        .font(.caption2)
                        .foregroundStyle(isSelected ? Color.white : Color(white: 0.3))
                        .padding(.bottom, 4)
                }
            }
        }
        .buttonStyle(.plain)
        .accessibilityLabel(SongDocument.noteName(midi))
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }
}
