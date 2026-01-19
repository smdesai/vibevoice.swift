import Foundation
import MLX

public struct WAVWriter {

    public static func save(
        _ audio: MLXArray,
        to path: String,
        sampleRate: Int = AudioConstants.sampleRate,
        targetRMS: Float = 0.063
    ) throws {
        let samples = audio.reshaped([-1]).asArray(Float.self)

        let url = URL(fileURLWithPath: path)
        try write(samples: samples, sampleRate: sampleRate, to: url, targetRMS: targetRMS)
    }

    private static func write(
        samples: [Float],
        sampleRate: Int,
        to url: URL,
        targetRMS: Float
    ) throws {
        guard !samples.isEmpty else {
            throw VibeVoiceError.audioProcessingError("No samples to write")
        }

        let rms = sqrt(samples.map { $0 * $0 }.reduce(0, +) / Float(samples.count))

        var normalizedSamples: [Float]
        if rms > 0 {
            let scale = targetRMS / rms
            normalizedSamples = samples.map { $0 * scale }
        } else {
            normalizedSamples = samples
        }

        normalizedSamples = normalizedSamples.map { max(-1.0, min(1.0, $0)) }

        let int16Samples = normalizedSamples.map { Int16($0 * 32767) }

        var data = Data()

        data.append(contentsOf: "RIFF".utf8)
        let fileSize = UInt32(36 + int16Samples.count * 2)
        data.append(contentsOf: withUnsafeBytes(of: fileSize.littleEndian) { Array($0) })
        data.append(contentsOf: "WAVE".utf8)

        data.append(contentsOf: "fmt ".utf8)
        let fmtSize: UInt32 = 16
        data.append(contentsOf: withUnsafeBytes(of: fmtSize.littleEndian) { Array($0) })
        let audioFormat: UInt16 = 1
        data.append(contentsOf: withUnsafeBytes(of: audioFormat.littleEndian) { Array($0) })
        let numChannels: UInt16 = 1
        data.append(contentsOf: withUnsafeBytes(of: numChannels.littleEndian) { Array($0) })
        let sampleRateU32 = UInt32(sampleRate)
        data.append(contentsOf: withUnsafeBytes(of: sampleRateU32.littleEndian) { Array($0) })
        let byteRate = UInt32(sampleRate * 2)
        data.append(contentsOf: withUnsafeBytes(of: byteRate.littleEndian) { Array($0) })
        let blockAlign: UInt16 = 2
        data.append(contentsOf: withUnsafeBytes(of: blockAlign.littleEndian) { Array($0) })
        let bitsPerSample: UInt16 = 16
        data.append(contentsOf: withUnsafeBytes(of: bitsPerSample.littleEndian) { Array($0) })

        data.append(contentsOf: "data".utf8)
        let dataSize = UInt32(int16Samples.count * 2)
        data.append(contentsOf: withUnsafeBytes(of: dataSize.littleEndian) { Array($0) })

        for sample in int16Samples {
            data.append(contentsOf: withUnsafeBytes(of: sample.littleEndian) { Array($0) })
        }

        try data.write(to: url)
    }

    public static func duration(
        samples sampleCount: Int, sampleRate: Int = AudioConstants.sampleRate
    ) -> Double {
        return Double(sampleCount) / Double(sampleRate)
    }

    public static func formatDuration(_ seconds: Double) -> String {
        if seconds < 60 {
            return String(format: "%.1fs", seconds)
        } else {
            let minutes = Int(seconds / 60)
            let remainingSeconds = seconds - Double(minutes * 60)
            return String(format: "%dm %.1fs", minutes, remainingSeconds)
        }
    }
}
