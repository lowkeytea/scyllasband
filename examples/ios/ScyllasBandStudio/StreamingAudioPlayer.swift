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

/// Keep one buffer playing and one queued. Scheduling the next buffer only
/// after `.dataPlayedBack` drains `AVAudioPlayerNode`; on a real device that
/// can create gaps or cause a later buffer to be missed entirely.
final class StreamingAudioPlayer {
    private struct ScheduledBuffer {
        let id: UInt64
        var didAnnouncePlayback = false
        let onPlaybackStarted: @MainActor () -> Void
    }

    private let maximumScheduledBufferCount = 2
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let format: AVAudioFormat
    private let condition = NSCondition()
    private var scheduledBuffers: [ScheduledBuffer] = []
    private var nextBufferID: UInt64 = 0
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
    }

    func enqueue(
        _ chunk: SBScyllasBandAudioChunk,
        shouldStop: () -> Bool,
        onPlaybackStarted: @escaping @MainActor () -> Void
    ) throws -> Bool {
        guard chunk.sampleCount >= 0,
              chunk.pcmFloat32Data.count == chunk.sampleCount * MemoryLayout<Float>.size,
              let buffer = AVAudioPCMBuffer(
                pcmFormat: format,
                frameCapacity: AVAudioFrameCount(chunk.sampleCount)
              ),
              let samples = buffer.floatChannelData?[0] else {
            throw StreamingAudioPlayerError.invalidChunk
        }
        chunk.pcmFloat32Data.copyBytes(
            to: UnsafeMutableRawBufferPointer(
                start: samples,
                count: chunk.pcmFloat32Data.count
            )
        )
        buffer.frameLength = AVAudioFrameCount(chunk.sampleCount)

        condition.lock()
        while scheduledBuffers.count >= maximumScheduledBufferCount && !stopped && !shouldStop() {
            _ = condition.wait(until: Date(timeIntervalSinceNow: 0.05))
        }
        guard !stopped, !shouldStop() else {
            condition.unlock()
            return false
        }
        let bufferID = nextBufferID
        nextBufferID &+= 1
        let shouldAnnounceImmediately = scheduledBuffers.isEmpty
        scheduledBuffers.append(
            ScheduledBuffer(
                id: bufferID,
                didAnnouncePlayback: shouldAnnounceImmediately,
                onPlaybackStarted: onPlaybackStarted
            )
        )
        condition.unlock()

        player.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
            self?.completeBuffer(id: bufferID)
        }
        if !player.isPlaying {
            player.play()
        }
        if shouldAnnounceImmediately {
            Task { @MainActor in onPlaybackStarted() }
        }
        return true
    }

    func finish(shouldStop: () -> Bool) {
        condition.lock()
        while !scheduledBuffers.isEmpty && !stopped && !shouldStop() {
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
        scheduledBuffers.removeAll()
        condition.broadcast()
        condition.unlock()
        player.stop()
        engine.stop()
    }

    private func completeBuffer(id: UInt64) {
        var nextPlaybackCallback: (@MainActor () -> Void)?
        condition.lock()
        if let index = scheduledBuffers.firstIndex(where: { $0.id == id }) {
            let completedFrontBuffer = index == 0
            scheduledBuffers.remove(at: index)
            if completedFrontBuffer,
               !scheduledBuffers.isEmpty,
               !scheduledBuffers[0].didAnnouncePlayback {
                scheduledBuffers[0].didAnnouncePlayback = true
                nextPlaybackCallback = scheduledBuffers[0].onPlaybackStarted
            }
        }
        condition.broadcast()
        condition.unlock()
        if let nextPlaybackCallback {
            Task { @MainActor in nextPlaybackCallback() }
        }
    }

    deinit {
        stop()
    }
}
