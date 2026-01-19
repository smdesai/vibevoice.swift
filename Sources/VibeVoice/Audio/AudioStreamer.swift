import Foundation
import MLX

public protocol AudioStreamerDelegate: AnyObject {
    func audioStreamer(
        _ streamer: AudioStreamer, didGenerateChunk chunk: MLXArray, atIndex sampleIndex: Int)

    func audioStreamerDidFinish(_ streamer: AudioStreamer)

    func audioStreamer(_ streamer: AudioStreamer, didEncounterError error: Error)
}

extension AudioStreamerDelegate {
    public func audioStreamer(_ streamer: AudioStreamer, didEncounterError error: Error) {}
}

public class AudioStreamer {
    public weak var delegate: AudioStreamerDelegate?

    public var onChunk: ((MLXArray, Int) -> Void)?
    public var onFinish: (() -> Void)?
    public var onError: ((Error) -> Void)?

    public let sampleRate: Int = AudioConstants.sampleRate

    private var isCancelled = false

    private(set) public var chunkCount: Int = 0

    private(set) public var totalSamples: Int = 0

    public init() {}

    internal func emit(chunk: MLXArray, index: Int) {
        guard !isCancelled else { return }

        chunkCount += 1
        totalSamples += chunk.dim(-1)

        delegate?.audioStreamer(self, didGenerateChunk: chunk, atIndex: index)
        onChunk?(chunk, index)
    }

    internal func finish() {
        guard !isCancelled else { return }

        delegate?.audioStreamerDidFinish(self)
        onFinish?()
    }

    internal func error(_ error: Error) {
        delegate?.audioStreamer(self, didEncounterError: error)
        onError?(error)
    }

    public func cancel() {
        isCancelled = true
    }

    public var cancelled: Bool {
        isCancelled
    }

    public func reset() {
        isCancelled = false
        chunkCount = 0
        totalSamples = 0
    }

    public var duration: Double {
        Double(totalSamples) / Double(sampleRate)
    }
}

public class AudioChunkCollector: AudioStreamerDelegate {
    public private(set) var chunks: [MLXArray] = []
    public private(set) var isFinished = false

    public init() {}

    public func audioStreamer(
        _ streamer: AudioStreamer, didGenerateChunk chunk: MLXArray, atIndex sampleIndex: Int
    ) {
        chunks.append(chunk)
    }

    public func audioStreamerDidFinish(_ streamer: AudioStreamer) {
        isFinished = true
    }

    public func concatenatedAudio() -> MLXArray? {
        guard !chunks.isEmpty else { return nil }
        return concatenated(chunks, axis: -1)
    }

    public func reset() {
        chunks.removeAll()
        isFinished = false
    }
}
