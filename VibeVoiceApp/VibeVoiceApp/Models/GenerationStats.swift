import Foundation

/// Statistics from TTS generation for display in UI
struct GenerationStats: Equatable {
    /// Total duration of generated audio in seconds
    let audioDuration: Double

    /// Total time taken to generate all audio in seconds
    let generationTime: Double

    /// Time to first audio chunk in milliseconds
    let timeToFirstChunk: Double

    /// Real-time factor (generationTime / audioDuration)
    /// < 1.0 means faster than realtime
    var rtf: Double {
        guard audioDuration > 0 else { return 0 }
        return generationTime / audioDuration
    }

    /// Whether generation was faster than realtime playback
    var isFasterThanRealtime: Bool {
        rtf < 1.0
    }

    // Formatted display strings

    var formattedAudioDuration: String {
        String(format: "%.2fs", audioDuration)
    }

    var formattedGenerationTime: String {
        String(format: "%.2fs", generationTime)
    }

    var formattedTimeToFirstChunk: String {
        String(format: "%.0fms", timeToFirstChunk)
    }

    var formattedRTF: String {
        String(format: "%.2fx", rtf)
    }
}
