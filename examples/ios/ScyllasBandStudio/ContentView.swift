import SwiftUI
import ScyllasBandKit

struct ContentView: View {
    @ObservedObject var model: StudioViewModel
    @State private var showingRuntimeSettings = false

    var body: some View {
        NavigationStack {
            Group {
                if model.isPlaying {
                    playbackScreen
                } else {
                    editorScreen
                }
            }
            .background(Color(uiColor: .systemGroupedBackground).ignoresSafeArea())
            .navigationTitle("Scylla’s Band Studio")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        showingRuntimeSettings = true
                    } label: {
                        Image(systemName: "gearshape")
                    }
                    .accessibilityLabel("Runtime settings")
                }
            }
            .sheet(isPresented: $showingRuntimeSettings) {
                RuntimeSettingsView(model: model)
            }
        }
    }

    private var editorScreen: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 16) {
                VStack(alignment: .leading, spacing: 10) {
                    Text("Build a script from speaker segments. Each segment has its own voice, language, and delivery; the runtime speaks it sentence by sentence.")
                        .font(.subheadline)
                        .foregroundStyle(Color(uiColor: .secondaryLabel))
                    HStack {
                        presetMenu
                        Spacer()
                        runtimeBadge
                    }
                }

                if model.isPreparing {
                    preparingCard
                }

                ForEach($model.segments) { $segment in
                    SegmentCard(
                        segment: $segment,
                        bundleInfo: model.bundleInfo,
                        isFirst: segment.id == model.segments.first?.id,
                        canDelete: model.segments.count > 1,
                        onDelete: { model.removeSegment(id: segment.id) }
                    )
                }

                Button(action: model.addSegment) {
                    Label("Add speaker segment", systemImage: "plus.circle")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .disabled(model.bundleInfo == nil)

                playbackButton
                statusLine
                    .padding(.bottom, 16)
            }
            .padding()
        }
        .id("script-editor")
    }

    private var playbackScreen: some View {
        VStack(spacing: 0) {
            ScrollView {
                nowPlaying
                    .padding()
            }
            .id("active-playback")

            VStack(spacing: 12) {
                playbackButton
                statusLine
            }
            .padding()
            .background(Color(uiColor: .secondarySystemGroupedBackground))
            .overlay(alignment: .top) {
                Divider()
            }
        }
    }

    private var runtimeBadge: some View {
        Button {
            showingRuntimeSettings = true
        } label: {
            Label(model.runtimeSummary, systemImage: "cpu")
                .font(.caption.weight(.medium))
        }
        .buttonStyle(.bordered)
        .controlSize(.small)
        .accessibilityLabel("Runtime: \(model.runtimeSummary). Opens runtime settings.")
    }

    private var preparingCard: some View {
        HStack(alignment: .top, spacing: 12) {
            ProgressView()
                .tint(.accentColor)
            VStack(alignment: .leading, spacing: 4) {
                Text("Preparing voices…")
                    .font(.headline)
                Text(model.preparingDetail)
                    .font(.footnote)
                    .foregroundStyle(Color(uiColor: .secondaryLabel))
            }
            Spacer(minLength: 0)
        }
        .padding()
        .background(Color(uiColor: .secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 14))
        .overlay {
            RoundedRectangle(cornerRadius: 14)
                .stroke(Color.accentColor.opacity(0.55), lineWidth: 1)
        }
        .accessibilityElement(children: .combine)
    }

    private var playbackButton: some View {
        Button(action: model.playOrStop) {
            Label(model.isPlaying ? "Stop" : "Play script", systemImage: model.isPlaying ? "stop.fill" : "play.fill")
                .frame(maxWidth: .infinity)
                .padding(.vertical, 4)
        }
        .buttonStyle(.borderedProminent)
        .tint(model.isPlaying ? Color(uiColor: .systemRed) : .accentColor)
        .disabled(!model.isPlaying && !model.canPlay)
    }

    private var statusLine: some View {
        HStack(alignment: .top, spacing: 10) {
            if model.isPreparing || model.isPlaying {
                ProgressView()
                    .tint(.accentColor)
            }
            Text(model.status)
                .font(.footnote)
                .foregroundStyle(Color(uiColor: .secondaryLabel))
                .accessibilityLabel("Status: \(model.status)")
            Spacer(minLength: 0)
        }
    }

    private var presetMenu: some View {
        Menu {
            ForEach(ExampleDocument.allCases) { example in
                Button(example.title) { model.loadExample(example) }
            }
        } label: {
            Label("Load example", systemImage: "doc.text")
        }
        .disabled(model.bundleInfo == nil || model.isPlaying)
    }

    private var nowPlaying: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(spacing: 10) {
                Image(systemName: "waveform")
                    .font(.title2.weight(.semibold))
                    .foregroundStyle(.tint)
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 2) {
                    Text("NOW PLAYING")
                        .font(.caption.weight(.bold))
                        .foregroundStyle(Color(uiColor: .secondaryLabel))
                    if let progress = model.nowPlayingProgress {
                        Text(progress)
                            .font(.caption)
                            .foregroundStyle(Color(uiColor: .secondaryLabel))
                    }
                }
                Spacer(minLength: 0)
                Text(model.runtimeSummary)
                    .font(.caption)
                    .foregroundStyle(Color(uiColor: .secondaryLabel))
            }

            if let context = model.nowPlayingContext {
                Text(context)
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(Color(uiColor: .label))
            }

            Divider()

            Text(model.nowPlayingText ?? "Preparing the first sentence…")
                .font(.title3.weight(.medium))
                .foregroundStyle(Color(uiColor: .label))
                .frame(maxWidth: .infinity, minHeight: 220, alignment: .topLeading)
                .contentTransition(.opacity)
        }
        .padding(20)
        .background(
            Color(uiColor: .secondarySystemGroupedBackground),
            in: RoundedRectangle(cornerRadius: 16)
        )
        .overlay {
            RoundedRectangle(cornerRadius: 16)
                .stroke(Color.accentColor.opacity(0.55), lineWidth: 1.5)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(
            "Now playing. \(model.nowPlayingContext ?? ""). \(model.nowPlayingText ?? "Preparing audio")"
        )
    }

}

private struct SegmentCard: View {
    @Binding var segment: ScriptSegment
    let bundleInfo: SBScyllasBandBundleInfo?
    let isFirst: Bool
    let canDelete: Bool
    let onDelete: () -> Void
    @State private var showingSettings = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(segment.settings.voiceIdentifier.capitalized)
                        .font(.headline)
                    Text(summary)
                        .font(.caption)
                        .foregroundStyle(Color(uiColor: .secondaryLabel))
                }
                Spacer()
                Button("Settings") { showingSettings = true }
                    .buttonStyle(.bordered)
                if canDelete {
                    Button(role: .destructive, action: onDelete) {
                        Image(systemName: "trash")
                    }
                    .accessibilityLabel("Delete segment")
                }
            }
            TextEditor(text: $segment.text)
                .frame(minHeight: 120)
                .padding(8)
                .scrollContentBackground(.hidden)
                .foregroundStyle(Color(uiColor: .label))
                .background(Color(uiColor: .tertiarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 10))
                .overlay {
                    RoundedRectangle(cornerRadius: 10)
                        .stroke(Color(uiColor: .separator), lineWidth: 0.5)
                }
        }
        .padding()
        .background(Color(uiColor: .secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 14))
        .overlay {
            RoundedRectangle(cornerRadius: 14)
                .stroke(Color(uiColor: .separator), lineWidth: 0.5)
        }
        .disabled(bundleInfo == nil)
        .sheet(isPresented: $showingSettings) {
            if let bundleInfo {
                SegmentSettingsView(settings: $segment.settings, bundleInfo: bundleInfo)
            }
        }
        #if DEBUG
        .onAppear {
            if isFirst, bundleInfo != nil, LaunchOptions.consumeShowSegmentSettings() {
                showingSettings = true
            }
        }
        #endif
    }

    private var summary: String {
        "\(languageLabel(segment.settings.language)) · \(segment.settings.delivery.summary)"
    }
}
