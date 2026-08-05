import Foundation
@preconcurrency import ScyllasBandKit

@MainActor
final class StudioViewModel: ObservableObject {
    @Published var bundleInfo: SBScyllasBandBundleInfo?
    @Published var segments: [ScriptSegment] = []
    @Published var status = "Preparing the bundled speech model…"
    @Published var isInitializing = true
    @Published var isPlaying = false
    @Published var nowPlayingText: String?
    @Published var nowPlayingContext: String?
    @Published var nowPlayingProgress: String?

    private let worker = DispatchQueue(label: "org.scyllasband.studio.runtime", qos: .userInitiated)
    private var runtime: SBScyllasBand?
    private var audioPlayer: StreamingAudioPlayer?

    var canPlay: Bool {
        bundleInfo != nil && !isInitializing && segments.contains { !$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
    }

    /// Core AI on iOS 27+ when the Core AI bundle was embedded; otherwise the
    /// embedded ONNX bundle (int8 preferred). Mirrors the download defaults.
    private static func embeddedBundleURL() -> URL? {
        var candidates: [String] = []
        if #available(iOS 27.0, *) {
            candidates.append("coreai")
        }
        candidates.append(contentsOf: ["onnx-int8", "onnx"])
        for name in candidates {
            if let url = Bundle.main.url(
                forResource: name,
                withExtension: nil,
                subdirectory: "scyllasband"
            ) {
                return url
            }
        }
        return nil
    }

    func initialize() {
        guard runtime == nil else { return }
        guard let bundleURL = Self.embeddedBundleURL() else {
            isInitializing = false
            status = "Model bundle missing. Run the asset preparation script or set SCYLLASBAND_IOS_BUNDLE_DIR."
            return
        }
        let threadCount = min(max(ProcessInfo.processInfo.activeProcessorCount, 1), 6)
        worker.async { [weak self] in
            do {
                let runtime = try SBScyllasBand(
                    bundleURL: bundleURL,
                    threadCount: threadCount,
                    targetBucketCacheCapacity: SBScyllasBand.defaultMobileTargetBucketCacheCapacity
                )
                try runtime.warmUp()
                DispatchQueue.main.async {
                    guard let self else { return }
                    self.runtime = runtime
                    self.bundleInfo = runtime.bundleInfo
                    self.isInitializing = false
                    self.status = "\(runtime.bundleInfo.modelName) ready · \(runtime.bundleInfo.voices.count) voices · \(runtime.bundleInfo.affectAxes.count) affect controls"
                    self.loadPreset(.walkthrough)
                }
            } catch {
                DispatchQueue.main.async {
                    self?.isInitializing = false
                    self?.status = "Model initialization failed: \(error.localizedDescription)"
                }
            }
        }
    }

    func loadPreset(_ preset: ExamplePreset) {
        guard let info = bundleInfo, let firstVoice = info.voices.first else { return }
        let defaults = SegmentSettings(
            voiceIdentifier: firstVoice.identifier,
            language: firstVoice.defaultLanguage
        )
        do {
            if preset == .walkthrough {
                segments = try DefaultWalkthrough.segments(defaults: defaults, bundleInfo: info)
            } else if let name = preset.resourceName,
                      let url = Bundle.main.url(forResource: name, withExtension: "txt", subdirectory: "scyllasband/examples") {
                let source = try String(contentsOf: url, encoding: .utf8)
                segments = try PresetParser.parse(source, defaults: defaults, bundleInfo: info)
            } else {
                throw CocoaError(.fileNoSuchFile)
            }
            status = "Loaded “\(preset.title)” with \(segments.count) speaker segments."
        } catch {
            status = "Could not load example: \(error.localizedDescription)"
        }
    }

    func addSegment() {
        guard let info = bundleInfo, let firstVoice = info.voices.first else { return }
        let settings = segments.last?.settings ?? SegmentSettings(
            voiceIdentifier: firstVoice.identifier,
            language: firstVoice.defaultLanguage
        )
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
        runtime?.requestCancellation()
        audioPlayer?.stop()
        status = "Stopping after the current synthesis call…"
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

        runtime.resetCancellation()
        isPlaying = true
        nowPlayingText = playable[0].text
        nowPlayingContext = context(for: playable[0].settings)
        nowPlayingProgress = "Segment 1 of \(playable.count) · Preparing audio"
        status = "Starting playback…"
        worker.async { [weak self] in
            do {
                let player = try StreamingAudioPlayer(sampleRate: info.sampleRate)
                DispatchQueue.main.async { self?.audioPlayer = player }
                for (segmentIndex, segment) in playable.enumerated() {
                    if runtime.isCancellationRequested { break }
                    var currentChunkText = segment.text
                    try runtime.synthesizeText(
                        segment.text,
                        settings: segment.settings.nativeSettings(),
                        seed: 31_415 + UInt64(segmentIndex),
                        chunkStarted: { chunkIndex, chunkCount, chunkText in
                            if let chunkText, !chunkText.isEmpty { currentChunkText = chunkText }
                            DispatchQueue.main.async {
                                self?.status = "Rendering segment \(segmentIndex + 1)/\(playable.count) · chunk \(chunkIndex + 1)/\(chunkCount)…"
                            }
                            return !runtime.isCancellationRequested
                        },
                        audioChunk: { chunk in
                            do {
                                let audibleText = currentChunkText
                                return try player.enqueue(
                                    chunk,
                                    shouldStop: { runtime.isCancellationRequested },
                                    onPlaybackStarted: {
                                        self?.nowPlayingText = audibleText
                                        self?.nowPlayingContext = self?.context(for: segment.settings)
                                        self?.nowPlayingProgress = "Segment \(segmentIndex + 1) of \(playable.count) · Chunk \(chunk.chunkIndex + 1) of \(chunk.chunkCount)"
                                        self?.status = "Playing segment \(segmentIndex + 1)/\(playable.count) · chunk \(chunk.chunkIndex + 1)/\(chunk.chunkCount)…"
                                    }
                                )
                            } catch {
                                return false
                            }
                        }
                    )
                }
                player.finish { runtime.isCancellationRequested }
                let cancelled = runtime.isCancellationRequested
                player.stop()
                DispatchQueue.main.async {
                    self?.finishPlayback(cancelled: cancelled, error: nil)
                }
            } catch {
                let cancelled = runtime.isCancellationRequested
                DispatchQueue.main.async {
                    self?.audioPlayer?.stop()
                    self?.finishPlayback(cancelled: cancelled, error: cancelled ? nil : error)
                }
            }
        }
    }

    private func finishPlayback(cancelled: Bool, error: Error?) {
        audioPlayer = nil
        isPlaying = false
        nowPlayingText = nil
        nowPlayingContext = nil
        nowPlayingProgress = nil
        if let error {
            status = "Playback failed: \(error.localizedDescription)"
        } else {
            status = cancelled ? "Playback stopped." : "Playback finished."
        }
    }

    private func context(for settings: SegmentSettings) -> String {
        let emotion = settings.emotion.map { " · \($0) \(Int(settings.emotionStrength * 100))%" } ?? " · neutral"
        return "\(settings.voiceIdentifier.capitalized) · \(languageLabel(settings.language))\(emotion)"
    }
}
