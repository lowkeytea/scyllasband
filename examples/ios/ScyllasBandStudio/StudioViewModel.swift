import Foundation
import os
@preconcurrency import ScyllasBandKit

private let log = Logger(subsystem: "org.scyllasband.studio", category: "runtime")

/// Set from the main thread by Stop, polled by the synthesis worker.
private final class CancellationFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false

    var isSet: Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelled
    }

    func set() {
        lock.lock()
        cancelled = true
        lock.unlock()
    }
}

@MainActor
final class StudioViewModel: ObservableObject {
    @Published var bundleInfo: SBScyllasBandBundleInfo?
    @Published var segments: [ScriptSegment] = []
    @Published var status = "Preparing voices…"
    @Published var isPreparing = true
    @Published var preparingDetail = ""
    @Published var isPlaying = false
    @Published var nowPlayingText: String?
    @Published var nowPlayingContext: String?
    @Published var nowPlayingProgress: String?
    /// "Core ML" or "Core AI" once a runtime is ready.
    @Published private(set) var backend: String?
    /// The compute unit chosen in the runtime settings.
    @Published private(set) var computeUnit: ComputeUnitOption
    /// The compute unit the current runtime was created with.
    @Published private(set) var activeComputeUnit: ComputeUnitOption?

    private static let computeUnitKey = "computeUnit"
    private let worker = DispatchQueue(label: "org.scyllasband.studio.runtime", qos: .userInitiated)
    private var runtime: SBScyllasBand?
    private var runtimeGeneration = 0
    private var audioPlayer: StreamingAudioPlayer?
    private var playbackCancellation: CancellationFlag?
    private var didAutoplay = false

    init() {
        computeUnit = ComputeUnitOption(rawValue: UserDefaults.standard.integer(forKey: Self.computeUnitKey)) ?? .automatic
    }

    var canPlay: Bool {
        runtime != nil && !isPreparing && segments.contains { !$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
    }

    /// e.g. "Core ML · Automatic".
    var runtimeSummary: String {
        "\(backend ?? "Preparing") · \((activeComputeUnit ?? effectiveComputeUnit).title)"
    }

    private var effectiveComputeUnit: ComputeUnitOption {
        ComputeUnitOption.isSelectable ? computeUnit : .cpu
    }

    func initialize() {
        guard runtimeGeneration == 0 else { return }
        loadRuntime()
    }

    func setComputeUnit(_ option: ComputeUnitOption) {
        guard option != computeUnit else { return }
        computeUnit = option
        UserDefaults.standard.set(option.rawValue, forKey: Self.computeUnitKey)
        stopPlayback()
        loadRuntime()
    }

    /// The embedded bundles to try, best first: Core AI on iOS 27 devices, then Core ML. The Simulator has no Core AI.
    private static func bundleCandidates() -> [URL] {
        guard let directory = Bundle.main.resourceURL?.appendingPathComponent("scyllasband", isDirectory: true) else {
            return []
        }
        var candidates: [URL] = []
        #if !targetEnvironment(simulator)
        if let preferred = SBScyllasBand.preferredBundleURL(inDirectory: directory) {
            candidates.append(preferred)
        }
        #endif
        let coreML = directory.appendingPathComponent("coreml", isDirectory: true)
        if FileManager.default.fileExists(atPath: coreML.appendingPathComponent("manifest.json").path),
           !candidates.contains(where: { $0.lastPathComponent == "coreml" }) {
            candidates.append(coreML)
        }
        return candidates
    }

    /// Creates and warms a runtime on the worker queue; it runs after any playback still finishing there.
    private func loadRuntime() {
        runtimeGeneration += 1
        let generation = runtimeGeneration
        runtime = nil
        backend = nil
        activeComputeUnit = nil
        isPreparing = true
        let candidates = Self.bundleCandidates()
        guard let first = candidates.first else {
            isPreparing = false
            status = "Model bundle missing. Download a Core ML or Core AI bundle and rebuild (see README)."
            return
        }
        status = "Preparing voices…"
        preparingDetail = first.lastPathComponent == "coreai"
            ? "Core AI optimizes the voices for this device the first time they load after installing the app or updating iOS. This can take a minute; later launches are quick."
            : "Loading the Core ML voices. The first launch after installing takes longest."
        let unit = effectiveComputeUnit
        worker.async { [weak self] in
            var failures: [String] = []
            for bundleURL in candidates {
                let flavor = backendLabel(bundleURL.lastPathComponent)
                let started = Date()
                do {
                    log.notice("Creating the \(flavor, privacy: .public) runtime on \(unit.title, privacy: .public)")
                    let runtime = try SBScyllasBand(bundleURL: bundleURL, computeUnit: unit.native)
                    try runtime.warmUp(withVoice: runtime.bundleInfo.defaultVoice)
                    let seconds = Date().timeIntervalSince(started)
                    log.notice("\(flavor, privacy: .public) runtime created and warmed up in \(seconds, format: .fixed(precision: 2)) s")
                    let fallbackNote = failures.first
                    DispatchQueue.main.async {
                        self?.runtimeReady(runtime, unit: unit, generation: generation, seconds: seconds, fallbackNote: fallbackNote)
                    }
                    return
                } catch {
                    log.error("\(flavor, privacy: .public) runtime failed: \(error.localizedDescription, privacy: .public)")
                    failures.append("\(flavor) failed: \(error.localizedDescription)")
                }
            }
            DispatchQueue.main.async {
                guard let self, generation == self.runtimeGeneration else { return }
                self.isPreparing = false
                self.status = "Model initialization failed. " + failures.joined(separator: " ")
            }
        }
    }

    private func runtimeReady(
        _ runtime: SBScyllasBand,
        unit: ComputeUnitOption,
        generation: Int,
        seconds: TimeInterval,
        fallbackNote: String?
    ) {
        guard generation == runtimeGeneration else { return }
        let info = runtime.bundleInfo
        let label = backendLabel(info.backend)
        self.runtime = runtime
        bundleInfo = info
        backend = label
        activeComputeUnit = unit
        isPreparing = false
        var ready = "\(label) ready on \(unit.title) in \(seconds.formatted(.number.precision(.fractionLength(1)))) s"
        if let fallbackNote {
            ready += " (\(fallbackNote))"
        }
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
        runtime?.requestCancellation()
        audioPlayer?.stop()
        status = "Stopping after the current model call…"
    }

    private func startPlayback() {
        guard let runtime, let info = bundleInfo else { return }
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

        let cancellation = CancellationFlag()
        playbackCancellation = cancellation
        isPlaying = true
        nowPlayingText = playable[0].text
        nowPlayingContext = context(for: playable[0].settings)
        nowPlayingProgress = "Segment 1 of \(playable.count) · Preparing audio"
        status = "Starting playback…"
        let requested = Date()
        log.notice("Playback requested: \(playable.count) segments")
        worker.async { [weak self] in
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
}
#endif
