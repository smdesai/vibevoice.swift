import AVFoundation
import Foundation
import MLX

public class RealtimeAudioPlayer: AudioStreamerDelegate {
    private let engine: AVAudioEngine
    private let playerNode: AVAudioPlayerNode
    private let format: AVAudioFormat

    public let sampleRate: Double = Double(AudioConstants.sampleRate)

    public private(set) var isPlaying = false

    public var onPlaybackComplete: (() -> Void)?

    private var scheduledBufferCount = 0
    private var completedBufferCount = 0
    private let bufferLock = NSLock()

    private var generationComplete = false

    public init() throws {
        engine = AVAudioEngine()
        playerNode = AVAudioPlayerNode()

        guard
            let audioFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: sampleRate,
                channels: 1,
                interleaved: false
            )
        else {
            throw RealtimeAudioPlayerError.failedToCreateFormat
        }
        format = audioFormat

        engine.attach(playerNode)
        engine.connect(playerNode, to: engine.mainMixerNode, format: format)
    }

    public func start() throws {
        guard !isPlaying else { return }

        try engine.start()
        playerNode.play()
        isPlaying = true
        generationComplete = false
        scheduledBufferCount = 0
        completedBufferCount = 0
    }

    public func stop() {
        guard isPlaying else { return }

        playerNode.stop()
        engine.stop()
        isPlaying = false
    }

    public func scheduleAudio(samples: [Float]) {
        guard isPlaying, !samples.isEmpty else { return }

        guard
            let buffer = AVAudioPCMBuffer(
                pcmFormat: format,
                frameCapacity: AVAudioFrameCount(samples.count)
            )
        else { return }

        buffer.frameLength = AVAudioFrameCount(samples.count)

        if let channelData = buffer.floatChannelData?[0] {
            samples.withUnsafeBufferPointer { sourceBuffer in
                if let baseAddress = sourceBuffer.baseAddress {
                    channelData.update(from: baseAddress, count: samples.count)
                }
            }
        }

        bufferLock.lock()
        scheduledBufferCount += 1
        bufferLock.unlock()

        playerNode.scheduleBuffer(buffer) { [weak self] in
            self?.bufferCompleted()
        }
    }

    public func scheduleAudio(chunk: MLXArray) {
        let flattened = chunk.reshaped([-1])
        eval(flattened)
        let samples = flattened.asArray(Float.self)
        scheduleAudio(samples: samples)
    }

    private func bufferCompleted() {
        bufferLock.lock()
        completedBufferCount += 1
        let allComplete = generationComplete && completedBufferCount >= scheduledBufferCount
        bufferLock.unlock()

        if allComplete {
            DispatchQueue.main.async { [weak self] in
                self?.onPlaybackComplete?()
            }
        }
    }

    public func audioStreamer(
        _ streamer: AudioStreamer, didGenerateChunk chunk: MLXArray, atIndex sampleIndex: Int
    ) {
        scheduleAudio(chunk: chunk)
    }

    public func audioStreamerDidFinish(_ streamer: AudioStreamer) {
        bufferLock.lock()
        generationComplete = true
        let allComplete = completedBufferCount >= scheduledBufferCount
        bufferLock.unlock()

        if allComplete {
            DispatchQueue.main.async { [weak self] in
                self?.onPlaybackComplete?()
            }
        }
    }

    public func audioStreamer(_ streamer: AudioStreamer, didEncounterError error: Error) {
        stop()
    }
}

public enum RealtimeAudioPlayerError: Error, LocalizedError {
    case failedToCreateFormat

    public var errorDescription: String? {
        switch self {
        case .failedToCreateFormat:
            return "Failed to create audio format"
        }
    }
}

public class SimpleAudioPlayer {
    private var engine: AVAudioEngine?
    private var playerNode: AVAudioPlayerNode?
    private let sampleRate: Double = Double(AudioConstants.sampleRate)

    public init() {}

    public func playSync(samples: [Float]) throws {
        let engine = AVAudioEngine()
        let playerNode = AVAudioPlayerNode()

        guard
            let format = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: sampleRate,
                channels: 1,
                interleaved: false
            )
        else {
            throw RealtimeAudioPlayerError.failedToCreateFormat
        }

        engine.attach(playerNode)
        engine.connect(playerNode, to: engine.mainMixerNode, format: format)

        guard
            let buffer = AVAudioPCMBuffer(
                pcmFormat: format,
                frameCapacity: AVAudioFrameCount(samples.count)
            )
        else { return }

        buffer.frameLength = AVAudioFrameCount(samples.count)

        if let channelData = buffer.floatChannelData?[0] {
            samples.withUnsafeBufferPointer { sourceBuffer in
                if let baseAddress = sourceBuffer.baseAddress {
                    channelData.update(from: baseAddress, count: samples.count)
                }
            }
        }

        let semaphore = DispatchSemaphore(value: 0)

        try engine.start()
        playerNode.play()

        playerNode.scheduleBuffer(buffer) {
            semaphore.signal()
        }

        semaphore.wait()

        Thread.sleep(forTimeInterval: 0.1)

        playerNode.stop()
        engine.stop()
    }

    public func playSync(audio: MLXArray) throws {
        let flattened = audio.reshaped([-1])
        eval(flattened)
        let samples = flattened.asArray(Float.self)
        try playSync(samples: samples)
    }
}
