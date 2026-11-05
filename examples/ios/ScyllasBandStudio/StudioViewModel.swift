import Combine
import Foundation
import os
@preconcurrency import ScyllasBandKit

private let log = Logger(subsystem: "org.scyllasband.studio", category: "runtime")

@MainActor
final class StudioViewModel: ObservableObject {
    @Published var segments: [ScriptSegment] = []
    @Published var status = "Preparing voices…"
    @Published var isPlaying = false
    @Published var nowPlayingText: String?
    @Published var nowPlayingContext: String?
    @Published var nowPlayingProgress: String?

    let controller: RuntimeController
    private var audioPlayer: StreamingAudioPlayer?
    private var playbackCancellation: CancellationFlag?
    private var didAutoplay = false
    private var changes: AnyCancellable?

    init(controller: RuntimeController) {
        self.controller = controller
        // The views watch this model; pass the runtime's changes (ready, compute unit, activity) on to them.
        changes = controller.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }
        controller.onReady { [weak self] _, message in self?.runtimeReady(message) }
        controller.onWillReload { [weak self] in self?.stopPlayback() }
    }

    var bundleInfo: SBScyllasBandBundleInfo? { controller.bundleInfo }
    var isPreparing: Bool { controller.isPreparing }
    var preparingDetail: String { controller.preparingDetail }
    var backend: String? { controller.backend }
    var computeUnit: ComputeUnitOption { controller.computeUnit }
    var activeComputeUnit: ComputeUnitOption? { controller.activeComputeUnit }
    var runtimeSummary: String { controller.summary }
    /// The status line: the runtime's failure while it has one, else this tab's status.
    var statusText: String { controller.failure ?? status }

    var canPlay: Bool {
        controller.runtime != nil && !controller.isPreparing && (controller.activity == nil || isPlaying)
            && segments.contains { !$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
    }

    func initialize() {
        controller.initialize()
    }

    func setComputeUnit(_ option: ComputeUnitOption) {
        controller.setComputeUnit(option)
    }

    private func runtimeReady(_ ready: String) {
        guard let info = controller.bundleInfo else { return }
        if segments.isEmpty {
            loadExample(.walkthrough, statusPrefix: ready)
        } else {
            for index in segments.indices {
                segments[index].settings.validate(using: info)
            }
            status = ready + "."
        }
        #if DEBUG
        if LaunchOptions.autoplay, !didAutoplay {
            didAutoplay = true
            startPlayback()
        }
        #endif
    }

    func loadExample(_ example: ExampleDocument, statusPrefix: String? = nil) {
        guard let info = bundleInfo else { return }
        let voice = info.voice(withIdentifier: info.defaultVoice) ?? info.voices[0]
        let defaults = SegmentSettings(voiceIdentifier: voice.identifier, language: voice.defaultLanguage)
        var message: String
        do {
            guard let url = example.url else { throw CocoaError(.fileNoSuchFile) }
            let source = try String(contentsOf: url, encoding: .utf8)
            segments = try PresetParser.parse(source, defaults: defaults, bundleInfo: info)
            message = "Loaded “\(example.title)” with \(segments.count) speaker segments."
        } catch {
            message = "Could not load example: \(error.localizedDescription)"
        }
        status = [statusPrefix, message].compactMap { $0 }.joined(separator: ". ")
    }

    func addSegment() {
        guard let info = bundleInfo else { return }
        let voice = info.voice(withIdentifier: info.defaultVoice) ?? info.voices[0]
        let settings = segments.last?.settings
            ?? SegmentSettings(voiceIdentifier: voice.identifier, language: voice.defaultLanguage)
        segments.append(ScriptSegment(text: "", settings: settings))
    }

    func removeSegment(id: UUID) {
        segments.removeAll { $0.id == id }
    }

    func playOrStop() {
        if isPlaying {
            stopPlayback()
        } else {
            startPlayback()
        }
    }

    func stopPlayback() {
        guard isPlaying else { return }
        playbackCancellation?.set()
        controller.runtime?.requestCancellation()
        audioPlayer?.stop()
        status = "Stopping after the current model call…"
    }

    private func startPlayback() {
        guard let runtime = controller.runtime, let info = bundleInfo else { return }
        let playable = segments.compactMap { segment -> ScriptSegment? in
            let text = segment.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { return nil }
            var copy = segment
            copy.text = text
            copy.settings.validate(using: info)
            return copy
        }
        guard !playable.isEmpty else {
            status = "Add some text before playing."
            return
        }

        guard controller.begin("speaking") else {
            status = "The runtime is busy singing; stop it first."
            return
        }
        let cancellation = CancellationFlag()
        playbackCancellation = cancellation
        isPlaying = true
        nowPlayingText = playable[0].text
        nowPlayingContext = context(for: playable[0].settings)
        nowPlayingProgress = "Segment 1 of \(playable.count) · Preparing audio"
        status = "Starting playback…"
        let requested = Date()
        log.notice("Playback requested: \(playable.count) segments")
        controller.worker.async { [weak self] in
            var firstAudio: TimeInterval?
            var playerError: Error?
            do {
                let player = try StreamingAudioPlayer(sampleRate: info.sampleRate)
                DispatchQueue.main.async { self?.audioPlayer = player }
                for (segmentIndex, segment) in playable.enumerated() {
                    if cancellation.isSet { break }
                    var currentChunkText = segment.text
                    try runtime.synthesizeRequest(
                        segment.settings.request(text: segment.text, seed: 31_415 + UInt64(segmentIndex)),
                        chunkStarted: { chunkIndex, chunkCount, chunkText in
                            if let chunkText, !chunkText.isEmpty { currentChunkText = chunkText }
                            DispatchQueue.main.async {
                                self?.status = "Rendering segment \(segmentIndex + 1)/\(playable.count) · sentence \(chunkIndex + 1)/\(chunkCount)…"
                            }
                            return !cancellation.isSet
                        },
                        audioChunk: { chunk in
                            if firstAudio == nil {
                                let seconds = Date().timeIntervalSince(requested)
                                firstAudio = seconds
                                log.notice("First audio after \(seconds, format: .fixed(precision: 2)) s (\(chunk.sampleCount) samples)")
                            }
                            let audibleText = currentChunkText
                            do {
                                return try player.enqueue(
                                    chunk,
                                    shouldStop: { cancellation.isSet },
                                    onPlaybackStarted: {
                                        self?.nowPlayingText = audibleText
                                        self?.nowPlayingContext = self?.context(for: segment.settings)
                                        self?.nowPlayingProgress = "Segment \(segmentIndex + 1) of \(playable.count) · Sentence \(chunk.chunkIndex + 1) of \(chunk.chunkCount)"
                                        self?.status = "Playing segment \(segmentIndex + 1)/\(playable.count) · sentence \(chunk.chunkIndex + 1)/\(chunk.chunkCount)…"
                                    }
                                )
                            } catch {
                                playerError = error
                                return false
                            }
                        }
                    )
                }
                player.finish { cancellation.isSet }
                player.stop()
                let cancelled = cancellation.isSet
                log.notice("Playback \(cancelled ? "stopped" : "finished", privacy: .public) after \(Date().timeIntervalSince(requested), format: .fixed(precision: 1)) s")
                DispatchQueue.main.async {
                    self?.finishPlayback(cancelled: cancelled, firstAudio: firstAudio, error: nil)
                }
            } catch {
                let cancelled = cancellation.isSet && playerError == nil
                let failure = playerError ?? error
                if !cancelled {
                    log.error("Playback failed: \(failure.localizedDescription, privacy: .public)")
                }
                DispatchQueue.main.async {
                    self?.audioPlayer?.stop()
                    self?.finishPlayback(cancelled: cancelled, firstAudio: firstAudio, error: cancelled ? nil : failure)
                }
            }
        }
    }

    private func finishPlayback(cancelled: Bool, firstAudio: TimeInterval?, error: Error?) {
        controller.end("speaking")
        audioPlayer = nil
        playbackCancellation = nil
        isPlaying = false
        nowPlayingText = nil
        nowPlayingContext = nil
        nowPlayingProgress = nil
        if let error {
            status = "Playback failed: \(error.localizedDescription)"
        } else {
            status = cancelled ? "Playback stopped." : "Playback finished."
            if let firstAudio {
                status += " First audio after \(firstAudio.formatted(.number.precision(.fractionLength(1)))) s."
            }
        }
    }

    private func context(for settings: SegmentSettings) -> String {
        "\(settings.voiceIdentifier.capitalized) · \(languageLabel(settings.language)) · \(settings.delivery.summary)"
    }
}

#if DEBUG
/// Debug launch arguments: `-autoplay` plays the first-launch document as soon as the voices are ready;
/// `-showSegmentSettings` opens the first segment's settings.
@MainActor
enum LaunchOptions {
    static let autoplay = ProcessInfo.processInfo.arguments.contains("-autoplay")
    private static var showSegmentSettings = ProcessInfo.processInfo.arguments.contains("-showSegmentSettings")

    /// True once, for the first caller, when `-showSegmentSettings` was passed.
    static func consumeShowSegmentSettings() -> Bool {
        defer { showSegmentSettings = false }
        return showSegmentSettings
    }

    // The Sing tab: `-singTab` opens it, `-singExample <file stem>` loads an example (e.g. song_pop_gwen), `-singTurbo` picks
    // Turbo, `-selectFirstNote` opens the note inspector, `-autosing` sings once the voices are ready.
    static let singTab = ProcessInfo.processInfo.arguments.contains("-singTab")
    static let singTurbo = ProcessInfo.processInfo.arguments.contains("-singTurbo")
    static let selectFirstNote = ProcessInfo.processInfo.arguments.contains("-selectFirstNote")
    static let autosing = ProcessInfo.processInfo.arguments.contains("-autosing")
    /// With -autosing: print the result (`SBSING <status>`) and quit once the song has played, for automated checks.
    static let exitAfterSing = ProcessInfo.processInfo.arguments.contains("-exitAfterSing")
    static var singExample: String? {
        let arguments = ProcessInfo.processInfo.arguments
        guard let index = arguments.firstIndex(of: "-singExample"), index + 1 < arguments.count else { return nil }
        return arguments[index + 1]
    }
}
#endif
