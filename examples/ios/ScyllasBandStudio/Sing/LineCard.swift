import SwiftUI

/// One line of the song: its lyrics, a sketch of the melody, and the notes as chips (tap one to edit it).
struct LineCard: View {
    @ObservedObject var model: SongViewModel
    let index: Int
    let line: SongLine
    let isLast: Bool
    @State private var lyrics = ""
    @FocusState private var editingLyrics: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Line \(index + 1)")
                    .font(.headline)
                Spacer()
                Menu {
                    if !isLast {
                        Picker("Rest after the line", selection: Binding(get: { line.restAfter },
                                                                          set: { model.setRestAfter(line: line.id, beats: $0) })) {
                            ForEach(restChoices, id: \.self) { beats in
                                Text(Self.beatsTitle(beats)).tag(beats)
                            }
                        }
                    }
                    Button("Delete line", systemImage: "trash", role: .destructive) { model.removeLine(line.id) }
                        .disabled(model.song.lines.count < 2)
                } label: {
                    Image(systemName: "ellipsis.circle")
                }
                .accessibilityLabel("Line \(index + 1) options")
                .disabled(model.isSinging)
            }
            HStack {
                TextField("Lyrics, e.g. give me a ring ba-by", text: $lyrics)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .focused($editingLyrics)
                    .onSubmit(apply)
                    .padding(8)
                    .background(Color(uiColor: .tertiarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 10))
                if lyrics != line.lyrics {
                    Button("Apply", action: apply)
                        .buttonStyle(.bordered)
                }
            }
            ContourView(notes: line.notes, selected: model.selection?.note, problem: model.problemNote)
                .frame(height: 56)
                .accessibilityHidden(true)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(alignment: .top, spacing: 6) {
                    ForEach(line.notes) { note in
                        NoteChip(note: note, isSelected: model.selection?.note == note.id, isProblem: model.problemNote == note.id)
                            .onTapGesture { model.selection = NoteSelection(line: line.id, note: note.id) }
                            .accessibilityAddTraits(.isButton)
                    }
                }
                .padding(.vertical, 2)
            }
            if !isLast {
                Text("Then a rest of \(Self.beatsTitle(max(line.restAfter, model.song.minimumLineRest)))")
                    .font(.caption)
                    .foregroundStyle(Color(uiColor: .secondaryLabel))
            }
        }
        .padding()
        .background(Color(uiColor: .secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 14))
        .overlay {
            RoundedRectangle(cornerRadius: 14)
                .stroke(Color(uiColor: .separator), lineWidth: 0.5)
        }
        .disabled(model.isSinging)
        .onAppear { lyrics = line.lyrics }
        .onChange(of: line.lyrics) { _, updated in
            if !editingLyrics { lyrics = updated }
        }
    }

    private var restChoices: [Double] {
        let minimum = model.song.minimumLineRest
        return Array(Set([minimum, 0.5, 1, 1.5, 2, 3, 4].filter { $0 >= minimum })).sorted()
    }

    private func apply() {
        model.updateLyrics(line: line.id, to: lyrics)
        editingLyrics = false
    }

    static func beatsTitle(_ beats: Double) -> String {
        let value = beats.formatted(.number.precision(.fractionLength(0...2)))
        return beats == 1 ? "1 beat" : "\(value) beats"
    }
}

/// A note as a chip: its syllable, pitch name and length; wider for longer notes.
struct NoteChip: View {
    let note: SongNote
    let isSelected: Bool
    let isProblem: Bool

    var body: some View {
        VStack(spacing: 2) {
            Text(note.isRest ? "rest" : (note.syllable.isEmpty ? "–" : note.syllable))
                .font(.subheadline.weight(note.isRest ? .regular : .semibold))
                .lineLimit(1)
            Text(note.midi.map(SongDocument.noteName) ?? " ")
                .font(.caption2.monospaced())
            Text(NoteChip.lengthSymbol(note.beats))
                .font(.caption2)
                .foregroundStyle(Color(uiColor: .secondaryLabel))
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .frame(minWidth: max(44, 36 * note.beats), minHeight: 54)
        .foregroundStyle(note.isRest ? Color(uiColor: .secondaryLabel) : Color(uiColor: .label))
        .background(note.isRest ? Color(uiColor: .tertiarySystemGroupedBackground) : Color.accentColor.opacity(0.12),
                    in: RoundedRectangle(cornerRadius: 9))
        .overlay {
            RoundedRectangle(cornerRadius: 9)
                .stroke(isProblem ? Color(uiColor: .systemRed) : (isSelected ? Color.accentColor : Color(uiColor: .separator)),
                        lineWidth: isSelected || isProblem ? 2 : 0.5)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(spokenLabel)
    }

    private var spokenLabel: String {
        let length = LineCard.beatsTitle(note.beats)
        guard let midi = note.midi else { return "Rest, \(length)" }
        let syllable = note.syllable.isEmpty ? "held" : note.syllable
        return "\(syllable), \(SongDocument.noteName(midi)), \(length)"
    }

    /// 1/16, 1/8, 1/8., 1/4, 1/4., 1/2, 1/2., 1 (beats are quarter notes).
    static func lengthSymbol(_ beats: Double) -> String {
        switch beats {
        case 0.25: return "1/16"
        case 0.5: return "1/8"
        case 0.75: return "1/8."
        case 1: return "1/4"
        case 1.5: return "1/4."
        case 2: return "1/2"
        case 3: return "1/2."
        case 4: return "1"
        default: return "\(beats.formatted(.number.precision(.fractionLength(0...2)))) b"
        }
    }
}

/// The line's melody: each note a bar at its pitch, as long as the note, joined by thin lines.
struct ContourView: View {
    let notes: [SongNote]
    let selected: UUID?
    let problem: UUID?

    var body: some View {
        Canvas { context, size in
            let total = max(notes.reduce(0) { $0 + $1.beats }, 0.25)
            let pitches = notes.compactMap(\.midi)
            guard let low = pitches.min(), let high = pitches.max() else { return }
            let span = Double(max(high - low, 4))
            let middle = Double(low + high) / 2
            func y(_ midi: Int) -> CGFloat {
                let fraction = (Double(midi) - (middle - span / 2)) / span
                return size.height - 6 - CGFloat(fraction) * (size.height - 12)
            }
            var position = 0.0
            var previous: CGPoint?
            for note in notes {
                let x0 = CGFloat(position / total) * size.width
                position += note.beats
                let x1 = CGFloat(position / total) * size.width
                guard let midi = note.midi else {
                    previous = nil
                    continue
                }
                let level = y(midi)
                if let previous {
                    var joint = Path()
                    joint.move(to: previous)
                    joint.addLine(to: CGPoint(x: x0, y: level))
                    context.stroke(joint, with: .color(.secondary.opacity(0.5)), lineWidth: 1)
                }
                var bar = Path()
                bar.move(to: CGPoint(x: x0 + 1.5, y: level))
                bar.addLine(to: CGPoint(x: max(x0 + 3, x1 - 1.5), y: level))
                let color: Color = note.id == problem ? .red : (note.id == selected ? .accentColor : .primary.opacity(0.75))
                context.stroke(bar, with: .color(color), style: StrokeStyle(lineWidth: note.id == selected ? 5 : 3.5, lineCap: .round))
                previous = CGPoint(x: x1 - 1.5, y: level)
            }
        }
    }
}
