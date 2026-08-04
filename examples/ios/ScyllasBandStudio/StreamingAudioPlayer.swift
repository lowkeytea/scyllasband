import AVFoundation
import Foundation
import ScyllasBandKit

enum StreamingAudioPlayerError: LocalizedError {
    case invalidFormat
    case invalidChunk

    var errorDescription: String? {
        switch self {
        case .invalidFormat: return "Unable to create a mono Float32 audio format."
        case .invalidChunk: return "Scylla's Band returned an invalid audio chunk."
        }
    }
}

/// A one-buffer queue keeps transcript updates aligned while allowing the
/// native runtime to render the next chunk during current-chunk playback.
final class StreamingAudioPlayer {
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let format: AVAudioFormat
    private let condition = NSCondition()
    private var queuedBufferCount = 0
    private var stopped = false

    init(sampleRate: Int) throws {
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(sampleRate),
            channels: 1,
            interleaved: false
        ) else {
            throw StreamingAudioPlayerError.invalidFormat
        }
        self.format = format
        let audioSession = AVAudioSession.sharedInstance()
        try audioSession.setCategory(.playback, mode: .spokenAudio)
        try audioSession.setActive(true)
        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: format)
        engine.prepare()
        try engine.start()
        player.play()
    }

    func enqueue(
        _ chunk: SBScyllasBandAudioChunk,
        shouldStop: () -> Bool,
        onPlaybackStarted: @escaping @MainActor () -> Void
    ) throws -> Bool {
        condition.lock()
        while queuedBufferCount >= 1 && !stopped && !shouldStop() {
            _ = condition.wait(until: Date(timeIntervalSinceNow: 0.05))
        }
        guard !stopped, !shouldStop() else {
            condition.unlock()
            return false
        }
        queuedBufferCount += 1
        condition.unlock()

        guard chunk.sampleCount >= 0,
              chunk.pcmFloat32Data.count == chunk.sampleCount * MemoryLayout<Float>.size,
              let buffer = AVAudioPCMBuffer(
                pcmFormat: format,
                frameCapacity: AVAudioFrameCount(chunk.sampleCount)
              ),
              let samples = buffer.floatChannelData?[0] else {
            completeBuffer()
            throw StreamingAudioPlayerError.invalidChunk
        }
        chunk.pcmFloat32Data.copyBytes(
            to: UnsafeMutableRawBufferPointer(
                start: samples,
                count: chunk.pcmFloat32Data.count
            )
        )
        buffer.frameLength = AVAudioFrameCount(chunk.sampleCount)

        Task { @MainActor in onPlaybackStarted() }
        player.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
            self?.completeBuffer()
        }
        return true
    }

    func finish(shouldStop: () -> Bool) {
        condition.lock()
        while queuedBufferCount > 0 && !stopped && !shouldStop() {
            _ = condition.wait(until: Date(timeIntervalSinceNow: 0.05))
        }
        condition.unlock()
    }

    func stop() {
        condition.lock()
        guard !stopped else {
            condition.unlock()
            return
        }
        stopped = true
        condition.broadcast()
        condition.unlock()
        player.stop()
        engine.stop()
    }

    private func completeBuffer() {
        condition.lock()
        queuedBufferCount = max(0, queuedBufferCount - 1)
        condition.broadcast()
        condition.unlock()
    }

    deinit {
        stop()
    }
}
