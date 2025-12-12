import ArgumentParser
import AVFoundation
import Foundation
import MLX
import MLXNN
import Tokenizers
import VibeVoice

@main
struct VibeVoiceCLI: AsyncParsableCommand {
    static let version = "0.1.0"
    static let configuration = CommandConfiguration(
        commandName: "vibevoice",
        abstract: "VibeVoice Text-to-Speech CLI (Realtime 0.5B)",
        version: version,
        subcommands: [
            Generate.self,
            Speak.self,
            Stream.self,
            Quantize.self
        ],
        defaultSubcommand: Generate.self
    )
}

struct Generate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Generate speech from text"
    )

    @Argument(help: "Text to synthesize")
    var text: String

    @Option(name: .shortAndLong, help: "Output audio file path")
    var output: String = "output.wav"

    @Option(name: .long, help: "Number of diffusion inference steps")
    var steps: Int = 20

    @Option(name: .long, help: "Model ID or path to model directory")
    var model: String = VibeVoiceRepository.id

    @Option(name: .long, help: "Tokenizer model ID or path")
    var tokenizer: String = Qwen2TokenizerRepository.id

    @Option(name: .long, help: "Path to voice cache safetensors file (required)")
    var voiceCache: String

    @Option(name: .long, help: "Maximum speech tokens to generate")
    var maxTokens: Int = 100

    @Option(name: .long, help: "CFG scale for generation")
    var cfgScale: Float = 1.3

    func run() async throws {
        print("VibeVoice Text-to-Speech")
        print("=" .repeated(40))
        print()
        print("Text: \"\(text)\"")
        print("Output: \(output)")
        print("Steps: \(steps)")
        print("CFG Scale: \(cfgScale)")
        print()

        let modelDir = try await ModelResolution.resolve(modelSpec: model)
        print("Loading model from: \(modelDir.path)")

        let tokenizerDir = try await ModelResolution.resolveTokenizer(modelSpec: tokenizer)
        let textTokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDir)

        let processedText = text.trimmingCharacters(in: .whitespacesAndNewlines) + "\n"
        let inputTokens = textTokenizer.encode(text: processedText, addSpecialTokens: false)
        print("Tokens: \(inputTokens.count)")

        let vibeVoiceModel = try loadVibeVoiceStreamModel(from: modelDir)

        let inference = VibeVoiceStreamInference(
            model: vibeVoiceModel,
            numInferenceSteps: steps,
            cfgScale: cfgScale
        )

        guard FileManager.default.fileExists(atPath: voiceCache) else {
            print("Error: Voice cache file not found: \(voiceCache)")
            return
        }
        try inference.loadVoiceCache(from: voiceCache)

        print("Generating...")
        let tokenIds = MLXArray(inputTokens.map { Int32($0) }).reshaped([1, inputTokens.count])

        let startTime = Date()
        let audio = try inference.generateWithVoiceCache(tokenIds: tokenIds, maxSpeechTokens: maxTokens)
        eval(audio)
        let elapsed = Date().timeIntervalSince(startTime)

        let samples = audio.dim(2)
        if samples > 0 {
            let duration = Float(samples) / Float(AudioConstants.sampleRate)
            print("Duration: \(String(format: "%.2f", duration))s")
            print("Time: \(String(format: "%.2f", elapsed))s")

            try WAVWriter.save(audio, to: output)
            print("Saved: \(output)")
        } else {
            print("No audio generated")
        }
    }
}

struct Speak: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Generate and play speech in realtime (streaming)"
    )

    @Argument(help: "Text to synthesize")
    var text: String

    @Option(name: .shortAndLong, help: "Output audio file path (optional, saves after playback)")
    var output: String?

    @Option(name: .long, help: "Number of diffusion inference steps")
    var steps: Int = 20

    @Option(name: .long, help: "Model ID or path to model directory")
    var model: String = VibeVoiceRepository.id

    @Option(name: .long, help: "Tokenizer model ID or path")
    var tokenizer: String = Qwen2TokenizerRepository.id

    @Option(name: .long, help: "Path to voice cache safetensors file")
    var voiceCache: String = "voice_cache.safetensors"

    @Option(name: .long, help: "Maximum speech tokens to generate")
    var maxTokens: Int = 200

    @Option(name: .long, help: "CFG scale for generation")
    var cfgScale: Float = 1.3

    func run() async throws {
        print("VibeVoice Realtime TTS")
        print("=" .repeated(40))
        print()
        print("Text: \"\(text)\"")
        print("Steps: \(steps)")
        print("CFG Scale: \(cfgScale)")
        print()

        guard FileManager.default.fileExists(atPath: voiceCache) else {
            print("Error: Voice cache not found at \(voiceCache)")
            print("Please provide a voice cache file with --voice-cache")
            return
        }

        let modelDir = try await ModelResolution.resolve(modelSpec: model)
        print("Loading model from: \(modelDir.path)")

        let tokenizerDir = try await ModelResolution.resolveTokenizer(modelSpec: tokenizer)
        let textTokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDir)

        let processedText = text.trimmingCharacters(in: .whitespacesAndNewlines) + "\n"
        let inputTokens = textTokenizer.encode(text: processedText, addSpecialTokens: false)
        print("Tokens: \(inputTokens.count)")

        let vibeVoiceModel = try loadVibeVoiceStreamModel(from: modelDir)

        let inference = VibeVoiceStreamInference(
            model: vibeVoiceModel,
            numInferenceSteps: steps,
            cfgScale: cfgScale
        )

        try inference.loadVoiceCache(from: voiceCache)
        print("Voice cache loaded")

        let player = try RealtimeAudioPlayer()

        let streamer = AudioStreamer()
        streamer.delegate = player

        var allChunks: [MLXArray] = []
        var chunkTimestamps: [Date] = []

        print()
        print("Generating and playing audio in realtime...")
        let startTime = Date()

        try player.start()

        streamer.onChunk = { chunk, index in
            chunkTimestamps.append(Date())
            allChunks.append(chunk)
            if index == 0 {
                print("First audio chunk generated!")
            }
        }

        let tokenIds = MLXArray(inputTokens.map { Int32($0) }).reshaped([1, inputTokens.count])

        let playbackComplete = DispatchSemaphore(value: 0)
        player.onPlaybackComplete = {
            playbackComplete.signal()
        }

        try inference.generateStreamingWithVoiceCache(
            tokenIds: tokenIds,
            maxSpeechTokens: maxTokens,
            audioStreamer: streamer
        )

        let generationTime = Date().timeIntervalSince(startTime)

        print("Waiting for playback to complete...")
        await withCheckedContinuation { continuation in
            DispatchQueue.global().async {
                _ = playbackComplete.wait(timeout: .now() + 60)
                continuation.resume()
            }
        }

        player.stop()

        let totalTime = Date().timeIntervalSince(startTime)

        print()
        print("=" .repeated(40))
        print("Statistics:")
        print("  Chunks generated: \(streamer.chunkCount)")
        print("  Total samples: \(streamer.totalSamples)")
        print("  Audio duration: \(String(format: "%.2f", streamer.duration))s")
        print("  Generation time: \(String(format: "%.2f", generationTime))s")
        print("  Total time: \(String(format: "%.2f", totalTime))s")

        if let firstChunkTime = chunkTimestamps.first {
            let firstLatency = firstChunkTime.timeIntervalSince(startTime)
            print("  First chunk latency: \(String(format: "%.0f", firstLatency * 1000))ms")
        }

        if chunkTimestamps.count > 1 {
            var interChunkLatencies: [Double] = []
            for i in 1..<chunkTimestamps.count {
                let latency = chunkTimestamps[i].timeIntervalSince(chunkTimestamps[i - 1])
                interChunkLatencies.append(latency)
            }
            let avgLatency = interChunkLatencies.reduce(0, +) / Double(interChunkLatencies.count)
            let minLatency = interChunkLatencies.min() ?? 0
            let maxLatency = interChunkLatencies.max() ?? 0
            print("  Avg chunk latency: \(String(format: "%.0f", avgLatency * 1000))ms (min: \(String(format: "%.0f", minLatency * 1000))ms, max: \(String(format: "%.0f", maxLatency * 1000))ms)")
        }

        if streamer.duration > 0 {
            let rtf = generationTime / streamer.duration
            print("  RTF (Real Time Factor): \(String(format: "%.2f", rtf))x")

            let audioPerChunk = 3200.0 / Double(AudioConstants.sampleRate)
            if chunkTimestamps.count > 1 {
                var interChunkLatencies: [Double] = []
                for i in 1..<chunkTimestamps.count {
                    interChunkLatencies.append(chunkTimestamps[i].timeIntervalSince(chunkTimestamps[i - 1]))
                }
                let avgLatency = interChunkLatencies.reduce(0, +) / Double(interChunkLatencies.count)
                if avgLatency < audioPerChunk {
                    print("  Status: Faster than realtime (generating ahead of playback)")
                } else {
                    print("  Status: Slower than realtime (playback may stutter)")
                }
            }
        }

        if let outputPath = output, !allChunks.isEmpty {
            let audio = concatenated(allChunks, axis: -1)
            eval(audio)
            try WAVWriter.save(audio, to: outputPath)
            print("  Saved to: \(outputPath)")
        }
    }
}

struct Stream: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Stream text from stdin and generate speech in realtime"
    )

    @Option(name: .long, help: "Number of diffusion inference steps")
    var steps: Int = 20

    @Option(name: .long, help: "Model ID or path to model directory")
    var model: String = VibeVoiceRepository.id

    @Option(name: .long, help: "Tokenizer model ID or path")
    var tokenizer: String = Qwen2TokenizerRepository.id

    @Option(name: .long, help: "Path to voice cache safetensors file (required)")
    var voiceCache: String

    @Option(name: .long, help: "Maximum speech tokens to generate")
    var maxTokens: Int = 1000

    @Option(name: .long, help: "CFG scale for generation")
    var cfgScale: Float = 1.3

    @Option(name: .shortAndLong, help: "Output audio file path (optional)")
    var output: String?

    func run() async throws {
        let isPiped = isatty(FileHandle.standardInput.fileDescriptor) == 0

        if !isPiped {
            print("VibeVoice Streaming Mode")
            print("=" .repeated(40))
            print("Reading text from stdin")
            print("Pipe from LLM: mlx_lm.generate -p 'prompt' | vibevoice stream --voice-cache ...")
            print()
        }

        guard FileManager.default.fileExists(atPath: voiceCache) else {
            fputs("Error: Voice cache not found at \(voiceCache)\n", stderr)
            return
        }

        let modelDir = try await ModelResolution.resolve(modelSpec: model)
        fputs("Loading model from: \(modelDir.path)\n", stderr)

        let tokenizerDir = try await ModelResolution.resolveTokenizer(modelSpec: tokenizer)
        let textTokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDir)

        let vibeVoiceModel = try loadVibeVoiceStreamModel(from: modelDir)

        let inference = VibeVoiceStreamInference(
            model: vibeVoiceModel,
            numInferenceSteps: steps,
            cfgScale: cfgScale
        )

        try inference.loadVoiceCache(from: voiceCache)
        fputs("Voice cache loaded\n", stderr)

        let player = try RealtimeAudioPlayer()

        let streamer = AudioStreamer()
        streamer.delegate = player

        var allChunks: [MLXArray] = []

        streamer.onChunk = { chunk, index in
            allChunks.append(chunk)
            if index == 0 {
                fputs("First audio chunk generated!\n", stderr)
            }
        }

        try player.start()

        let session = try inference.createStreamingSession(audioStreamer: streamer)

        fputs("Ready for input...\n", stderr)

        var textBuffer = ""
        let stdin = FileHandle.standardInput

        while true {
            let data = stdin.availableData
            if data.isEmpty { break }

            guard let chunk = String(data: data, encoding: .utf8) else { continue }
            textBuffer += chunk

            let tokens = textTokenizer.encode(text: textBuffer, addSpecialTokens: false)
            if tokens.count >= 5 {
                let int32Tokens = tokens.map { Int32($0) }
                fputs("  Processing \(tokens.count) tokens...\n", stderr)
                try session.addTokens(int32Tokens)
                textBuffer = ""
            }
        }

        if !textBuffer.isEmpty {
            let tokens = textTokenizer.encode(text: textBuffer, addSpecialTokens: false)
            if !tokens.isEmpty {
                let int32Tokens = tokens.map { Int32($0) }
                fputs("  Processing final \(tokens.count) tokens...\n", stderr)
                try session.addTokens(int32Tokens)
            }
        }

        fputs("Flushing remaining audio...\n", stderr)
        try session.flush(maxSpeechTokens: maxTokens)

        fputs("Waiting for playback to complete...\n", stderr)
        let playbackComplete = DispatchSemaphore(value: 0)
        player.onPlaybackComplete = {
            playbackComplete.signal()
        }

        await withCheckedContinuation { continuation in
            DispatchQueue.global().async {
                _ = playbackComplete.wait(timeout: .now() + 60)
                continuation.resume()
            }
        }

        player.stop()

        fputs("\n", stderr)
        fputs("=" .repeated(40) + "\n", stderr)
        fputs("Statistics:\n", stderr)
        fputs("  Chunks generated: \(streamer.chunkCount)\n", stderr)
        fputs("  Total samples: \(streamer.totalSamples)\n", stderr)
        fputs("  Audio duration: \(String(format: "%.2f", streamer.duration))s\n", stderr)

        if let outputPath = output, !allChunks.isEmpty {
            let audio = concatenated(allChunks, axis: -1)
            eval(audio)
            try WAVWriter.save(audio, to: outputPath)
            fputs("  Saved to: \(outputPath)\n", stderr)
        }
    }
}

struct Quantize: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Quantize model weights to 8-bit or 4-bit"
    )

    @Option(name: .shortAndLong, help: "Input model ID or directory path")
    var input: String = VibeVoiceRepository.id

    @Option(name: .shortAndLong, help: "Output directory for quantized model")
    var output: String

    @Option(name: .long, help: "Number of bits for quantization (4 or 8)")
    var bits: Int = 8

    @Option(name: .long, help: "Group size for quantization (32, 64, or 128)")
    var groupSize: Int = 32

    @Option(name: .long, help: "Quantization mode (affine or mxfp4)")
    var mode: String = "affine"

    @Flag(name: .long, help: "Enable verbose output")
    var verbose: Bool = false

    func run() async throws {
        print("VibeVoice Model Quantization")
        print("=" .repeated(40))
        print()
        print("Input: \(input)")
        print("Output: \(output)")
        print("Bits: \(bits)")
        print("Group Size: \(groupSize)")
        print("Mode: \(mode)")
        print()

        let inputURL = try await ModelResolution.resolve(modelSpec: input)
        let outputURL = URL(fileURLWithPath: output)

        print("Resolved model path: \(inputURL.path)")

        let quantMode: VibeVoiceQuantizationMode
        switch mode.lowercased() {
        case "affine":
            quantMode = .affine
        case "mxfp4":
            quantMode = .mxfp4
        default:
            print("Error: Invalid mode '\(mode)'. Use 'affine' or 'mxfp4'")
            return
        }

        let spec = VibeVoiceQuantizationSpec(
            groupSize: groupSize,
            bits: bits,
            mode: quantMode
        )

        print("Quantizing...")
        let startTime = Date()

        do {
            try VibeVoiceQuantizer.quantizeAndSave(
                from: inputURL,
                to: outputURL,
                spec: spec,
                verbose: verbose
            )

            let elapsed = Date().timeIntervalSince(startTime)
            print()
            print("Quantization complete!")
            print("Time: \(String(format: "%.2f", elapsed))s")
            print("Output: \(output)")
        } catch {
            print("Error: \(error.localizedDescription)")
        }
    }
}

extension String {
    func repeated(_ count: Int) -> String {
        String(repeating: self, count: count)
    }
}
