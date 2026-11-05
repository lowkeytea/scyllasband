import SwiftUI
import ScyllasBandKit

/// The Sing tab: build a melody line by line (lyrics first, then each note's pitch and length), hear it as tones, and sing it.
struct SingView: View {
    @ObservedObject var model: SongViewModel
    @State private var showingSettings = false
    @State private var showingShare = false

    var body: some View {
        NavigationStack {
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 14) {
                    Text("Type a line of lyrics (ba-by splits a word into syllables, _ holds one over another note), then tap a note to set its pitch and length. Rests of a quarter second or more separate the lines the voice sings one at a time.")
                        .font(.subheadline)
                        .foregroundStyle(Color(uiColor: .secondaryLabel))
                    HStack {
                        examplesMenu
                        songsMenu
                        Spacer()
                        Label(model.runtimeSummary, systemImage: "cpu")
                            .font(.caption.weight(.medium))
                            .foregroundStyle(Color(uiColor: .secondaryLabel))
                    }
                    if model.bundleInfo != nil, !model.canSing {
                        notice("This bundle speaks only. Download the current release (python -m scyllasband download) and rebuild to sing.")
                    }
                    songCard
                    ForEach(Array(model.song.lines.enumerated()), id: \.element.id) { index, line in
                        LineCard(model: model, index: index, line: line, isLast: index == model.song.lines.count - 1)
                    }
                    Button(action: model.addLine) {
                        Label("Add a line", systemImage: "plus.circle")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .disabled(model.isSinging)
                    actions
                    statusLine
                        .padding(.bottom, 16)
                }
                .padding()
            }
            .background(Color(uiColor: .systemGroupedBackground).ignoresSafeArea())
            .navigationTitle("Sing")
            .toolbar {
                ToolbarItemGroup(placement: .topBarTrailing) {
                    Button {
                        model.save()
                    } label: {
                        Image(systemName: "square.and.arrow.down")
                    }
                    .accessibilityLabel("Save song")
                    Button {
                        model.prepareShare()
                        showingShare = true
                    } label: {
                        Image(systemName: "square.and.arrow.up")
                    }
                    .accessibilityLabel("Share the score or the last take")
                }
            }
            .sheet(isPresented: $showingSettings) {
                SongSettingsView(model: model)
            }
            .sheet(isPresented: Binding(get: { model.selection != nil }, set: { if !$0 { model.selection = nil } })) {
                NoteInspector(model: model)
                    .presentationDetents([.medium, .large])
            }
            .sheet(isPresented: $showingShare) {
                ShareSheet(model: model)
                    .presentationDetents([.medium])
            }
        }
    }

    private var examplesMenu: some View {
        Menu {
            ForEach(model.examples) { example in
                Button(example.title) { model.load(example) }
            }
        } label: {
            Label("Examples", systemImage: "music.note.list")
        }
        .disabled(model.examples.isEmpty || model.isSinging)
    }

    private var songsMenu: some View {
        Menu {
            Button("New song", systemImage: "plus") { model.newSong() }
            if !model.savedSongs.isEmpty {
                Section("Saved") {
                    ForEach(model.savedSongs) { saved in
                        Button(saved.title) { model.open(saved) }
                    }
                }
            }
        } label: {
            Label("Songs", systemImage: "folder")
        }
        .disabled(model.isSinging)
    }

    private var songCard: some View {
        Button {
            showingSettings = true
        } label: {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text(model.song.title)
                        .font(.headline)
                        .foregroundStyle(Color(uiColor: .label))
                    Spacer()
                    Image(systemName: "slider.horizontal.3")
                        .foregroundStyle(.tint)
                        .accessibilityHidden(true)
                }
                Text(songDetails)
                    .font(.caption)
                    .foregroundStyle(Color(uiColor: .secondaryLabel))
                if !model.fitSummary.isEmpty {
                    Label(model.fitSummary, systemImage: model.suggestedTranspose == nil ? "checkmark.circle" : "arrow.up.arrow.down.circle")
                        .font(.caption)
                        .foregroundStyle(model.suggestedTranspose == nil ? Color(uiColor: .secondaryLabel) : Color.orange)
                }
            }
            .padding()
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color(uiColor: .secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 14))
            .overlay {
                RoundedRectangle(cornerRadius: 14)
                    .stroke(Color(uiColor: .separator), lineWidth: 0.5)
            }
        }
        .buttonStyle(.plain)
        .accessibilityHint("Opens the song settings: title, voice, style, tempo, sound and key.")
    }

    private var songDetails: String {
        let transpose = model.song.transpose == 0 ? "" : " · transpose \(model.song.transpose > 0 ? "+" : "")\(model.song.transpose)"
        return "\(model.song.voice.capitalized) · \(model.song.style.title) · \(Int(model.song.bpm)) bpm · \(model.song.decoder.title)\(transpose)"
    }

    private var actions: some View {
        VStack(spacing: 10) {
            HStack(spacing: 10) {
                Button(action: model.preview) {
                    Label(model.isPreviewing ? "Stop" : "Preview", systemImage: model.isPreviewing ? "stop.fill" : "pianokeys")
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 4)
                }
                .buttonStyle(.bordered)
                .disabled(model.isSinging || !model.canSing || model.problem != nil)
                Button(action: model.singOrStop) {
                    Label(model.isSinging ? "Stop" : "Sing", systemImage: model.isSinging ? "stop.fill" : "music.mic")
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 4)
                }
                .buttonStyle(.borderedProminent)
                .tint(model.isSinging ? Color(uiColor: .systemRed) : .accentColor)
                .disabled(model.isPreviewing || !model.canSing || (!model.isSinging && model.problem != nil))
            }
            Button("Another take", systemImage: "dice", action: model.anotherTake)
                .font(.footnote)
                .disabled(model.isBusy || !model.canSing)
        }
    }

    private var statusLine: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let problem = model.problem {
                Label(problem, systemImage: "exclamationmark.triangle")
                    .font(.footnote)
                    .foregroundStyle(Color(uiColor: .systemRed))
            }
            HStack(alignment: .top, spacing: 10) {
                if model.isSinging || model.controller.isPreparing {
                    ProgressView()
                        .tint(.accentColor)
                }
                VStack(alignment: .leading, spacing: 2) {
                    if let progress = model.progress {
                        Text(progress)
                            .font(.footnote.weight(.medium))
                    }
                    Text(model.controller.failure ?? model.status)
                        .font(.footnote)
                        .foregroundStyle(Color(uiColor: .secondaryLabel))
                }
                Spacer(minLength: 0)
            }
            .accessibilityElement(children: .combine)
        }
    }

    private func notice(_ text: String) -> some View {
        Label(text, systemImage: "info.circle")
            .font(.footnote)
            .padding()
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color(uiColor: .secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 14))
    }
}

/// The files a song is shared as: its score (the runtime's JSON) and, after singing, the take (WAV).
private struct ShareSheet: View {
    @ObservedObject var model: SongViewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                if let score = model.scoreURL {
                    ShareLink(item: score) {
                        Label("Score (JSON)", systemImage: "doc.text")
                    }
                }
                if let take = model.takeURL {
                    ShareLink(item: take) {
                        Label("Last take (WAV)", systemImage: "waveform")
                    }
                } else {
                    Text("Sing the song to share its audio.")
                        .foregroundStyle(Color(uiColor: .secondaryLabel))
                }
            }
            .navigationTitle("Share")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}
