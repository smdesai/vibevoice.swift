import Foundation
import MLX
import Tokenizers
import VibeVoice

actor TTSService {
    private var inference: VibeVoiceStreamInference?
    private var tokenizer: Tokenizer?
    private var currentVoice: Voice?
    private var audioPlayer: RealtimeAudioPlayer?

    var isModelLoaded: Bool {
        inference != nil && tokenizer != nil
    }

    var isVoiceLoaded: Bool {
        currentVoice != nil
    }

    func loadModel(progressHandler: ((Double, String) -> Void)? = nil) async throws {
        // Set MLX GPU memory cache limit to 200MB
        MLX.GPU.set(cacheLimit: 200 * 1024 * 1024)

        let modelId = "smdesai/VibeVoice-Realtime-0.5B-8bit"

        progressHandler?(0.0, "Checking for cached model...")

        // Check if model is already cached
        let modelURL: URL
        if let cachedURL = ModelResolution.findCachedModel(modelId: modelId, requireWeights: true) {
            progressHandler?(0.5, "Loading model from cache...")
            modelURL = cachedURL
        } else {
            progressHandler?(0.05, "Downloading model...")
            modelURL = try await ModelResolution.downloadModel(
                modelId: modelId,
                revision: "main",
                requireWeights: true,
                progressHandler: { progress in
                    progressHandler?(
                        0.05 + progress.fractionCompleted * 0.65, "Downloading model...")
                }
            )
            progressHandler?(0.7, "Loading model...")
        }

        inference = try VibeVoice.load(from: modelURL)

        // Check if tokenizer is cached
        let tokenizerModelId = "Qwen/Qwen2.5-0.5B"
        let tokenizerURL: URL
        if let cachedTokenizer = ModelResolution.findCachedModel(
            modelId: tokenizerModelId, requireWeights: false)
        {
            progressHandler?(0.85, "Loading tokenizer from cache...")
            tokenizerURL = cachedTokenizer
        } else {
            progressHandler?(0.85, "Downloading tokenizer...")
            tokenizerURL = try await ModelResolution.resolveTokenizer(modelSpec: nil)
        }
        tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerURL)

        progressHandler?(0.95, "Loading default voice...")

        // Auto-load Carter voice as default
        if let carterVoice = Voice.allVoices.first(where: { $0.id == "en-Carter" }) {
            let voicePath = VoiceDownloadService.shared.voicePath(for: carterVoice)

            if !FileManager.default.fileExists(atPath: voicePath.path) {
                _ = try await VoiceDownloadService.shared.downloadVoice(carterVoice)
            }

            try inference?.loadVoiceCache(from: voicePath.path)
            currentVoice = carterVoice
        }

        progressHandler?(1.0, "Ready")
    }

    func getCurrentVoice() -> Voice? {
        currentVoice
    }

    func loadVoice(_ voice: Voice) async throws {
        guard let inference = inference else {
            throw TTSError.modelNotLoaded
        }

        let voicePath = VoiceDownloadService.shared.voicePath(for: voice)

        if !FileManager.default.fileExists(atPath: voicePath.path) {
            _ = try await VoiceDownloadService.shared.downloadVoice(voice)
        }

        try inference.loadVoiceCache(from: voicePath.path)
        currentVoice = voice
    }

    func generateSpeech(
        text: String,
        onChunk: @escaping (MLXArray, Int) -> Void,
        onComplete: @escaping () -> Void,
        onError: @escaping (Error) -> Void
    ) async throws {
        guard let inference = inference else {
            throw TTSError.modelNotLoaded
        }

        guard let tokenizer = tokenizer else {
            throw TTSError.tokenizerNotLoaded
        }

        guard currentVoice != nil else {
            throw TTSError.voiceNotLoaded
        }

        let processedText = text.trimmingCharacters(in: .whitespacesAndNewlines) + "\n"
        let tokens = tokenizer.encode(text: processedText, addSpecialTokens: false)
        let tokenIds = MLXArray(tokens.map { Int32($0) }).reshaped([1, tokens.count])

        let streamer = AudioStreamer()
        streamer.onChunk = { chunk, index in
            onChunk(chunk, index)
        }
        streamer.onFinish = {
            onComplete()
        }
        streamer.onError = { error in
            onError(error)
        }

        try inference.generateStreamingWithVoiceCache(
            tokenIds: tokenIds,
            maxSpeechTokens: 500,
            audioStreamer: streamer
        )
    }

    func generateAndPlay(
        text: String,
        onStart: @escaping () -> Void,
        onComplete: @escaping () -> Void,
        onError: @escaping (Error) -> Void
    ) async throws {
        guard let inference = inference else {
            throw TTSError.modelNotLoaded
        }

        guard let tokenizer = tokenizer else {
            throw TTSError.tokenizerNotLoaded
        }

        guard currentVoice != nil else {
            throw TTSError.voiceNotLoaded
        }

        // Stop any previous playback before starting new generation
        stopPlayback()

        try AudioSessionManager.shared.configure()

        let player = try RealtimeAudioPlayer()
        audioPlayer = player

        player.onPlaybackComplete = {
            Task { @MainActor in
                onComplete()
            }
        }

        let processedText = text.trimmingCharacters(in: .whitespacesAndNewlines) + "\n"
        let tokens = tokenizer.encode(text: processedText, addSpecialTokens: false)
        let tokenIds = MLXArray(tokens.map { Int32($0) }).reshaped([1, tokens.count])

        let streamer = AudioStreamer()
        streamer.delegate = player

        try player.start()
        onStart()

        try inference.generateStreamingWithVoiceCache(
            tokenIds: tokenIds,
            maxSpeechTokens: 500,
            audioStreamer: streamer
        )
    }

    func stopPlayback() {
        audioPlayer?.stop()
        audioPlayer = nil
    }

    func reset() {
        stopPlayback()
        currentVoice = nil
    }
}

enum TTSError: LocalizedError {
    case modelNotLoaded
    case tokenizerNotLoaded
    case voiceNotLoaded
    case generationFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "Model is not loaded. Please wait for the model to download."
        case .tokenizerNotLoaded:
            return "Tokenizer is not loaded."
        case .voiceNotLoaded:
            return "No voice selected. Please select a voice first."
        case .generationFailed(let message):
            return "Speech generation failed: \(message)"
        }
    }
}
