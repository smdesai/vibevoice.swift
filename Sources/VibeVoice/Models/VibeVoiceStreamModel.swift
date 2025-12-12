import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

public class SpeechConnector: Module {
    @ModuleInfo(key: "fc1") public var fc1: Linear
    public let norm: RMSNorm
    @ModuleInfo(key: "fc2") public var fc2: Linear

    public init(inputDim: Int, outputDim: Int) {
        _fc1.wrappedValue = Linear(inputDim, outputDim, bias: true)
        self.norm = RMSNorm(dimensions: outputDim, eps: 1e-6)
        _fc2.wrappedValue = Linear(outputDim, outputDim, bias: true)
        super.init()
    }

    public func callAsFunction(_ features: MLXArray) -> MLXArray {
        var x = fc1(features)
        x = norm(x)
        x = fc2(x)
        return x
    }
}

public struct VibeVoiceConfiguration: Codable {
    public var decoderConfig: Qwen2Configuration
    public var acousticTokenizerConfig: AcousticTokenizerConfiguration
    public var diffusionHeadConfig: DiffusionHeadConfiguration
    public var ttsBackboneNumHiddenLayers: Int
    public var acousticVaeDim: Int

    enum CodingKeys: String, CodingKey {
        case decoderConfig = "decoder_config"
        case acousticTokenizerConfig = "acoustic_tokenizer_config"
        case diffusionHeadConfig = "diffusion_head_config"
        case ttsBackboneNumHiddenLayers = "tts_backbone_num_hidden_layers"
        case acousticVaeDim = "acoustic_vae_dim"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        decoderConfig = try container.decode(Qwen2Configuration.self, forKey: .decoderConfig)
        acousticTokenizerConfig = try container.decode(AcousticTokenizerConfiguration.self, forKey: .acousticTokenizerConfig)
        diffusionHeadConfig = try container.decode(DiffusionHeadConfiguration.self, forKey: .diffusionHeadConfig)
        acousticVaeDim = try container.decodeIfPresent(Int.self, forKey: .acousticVaeDim) ?? 64

        if let ttsLayers = try container.decodeIfPresent(Int.self, forKey: .ttsBackboneNumHiddenLayers) {
            ttsBackboneNumHiddenLayers = ttsLayers
        } else {
            let totalLayers = decoderConfig.hiddenLayers
            ttsBackboneNumHiddenLayers = max(totalLayers - 4, totalLayers * 3 / 4)
        }
    }

    public init(
        decoderConfig: Qwen2Configuration = Qwen2Configuration(),
        acousticTokenizerConfig: AcousticTokenizerConfiguration = AcousticTokenizerConfiguration(),
        diffusionHeadConfig: DiffusionHeadConfiguration = DiffusionHeadConfiguration(),
        ttsBackboneNumHiddenLayers: Int = 20,
        acousticVaeDim: Int = 64
    ) {
        self.decoderConfig = decoderConfig
        self.acousticTokenizerConfig = acousticTokenizerConfig
        self.diffusionHeadConfig = diffusionHeadConfig
        self.ttsBackboneNumHiddenLayers = ttsBackboneNumHiddenLayers
        self.acousticVaeDim = acousticVaeDim
    }
}

public class VibeVoiceStreamModel: Module {
    public let config: VibeVoiceConfiguration

    @ModuleInfo(key: "language_model") public var languageModel: Qwen2Model

    @ModuleInfo(key: "tts_language_model") public var ttsLanguageModel: Qwen2Model

    @ModuleInfo(key: "tts_input_types") public var ttsInputTypes: Embedding

    @ModuleInfo(key: "acoustic_tokenizer") public var acousticTokenizer: VibeVoiceAcousticTokenizer

    @ModuleInfo(key: "acoustic_connector") public var acousticConnector: SpeechConnector

    @ModuleInfo(key: "prediction_head") public var predictionHead: VibeVoiceDiffusionHead

    @ModuleInfo(key: "tts_eos_classifier") public var eosClassifier: EOSClassifier

    public var speechScalingFactor: MLXArray = MLXArray(1.0)
    public var speechBiasFactor: MLXArray = MLXArray(0.0)

    public var noiseScheduler: DPMSolverMultistepScheduler

    public init(_ config: VibeVoiceConfiguration) throws {
        self.config = config

        var lmConfig = config.decoderConfig
        let lmBackboneNumHiddenLayers = config.decoderConfig.hiddenLayers - config.ttsBackboneNumHiddenLayers
        lmConfig.hiddenLayers = lmBackboneNumHiddenLayers

        _languageModel.wrappedValue = Qwen2Model(lmConfig)

        var ttsLmConfig = config.decoderConfig
        ttsLmConfig.hiddenLayers = config.ttsBackboneNumHiddenLayers

        _ttsLanguageModel.wrappedValue = Qwen2Model(ttsLmConfig)

        _ttsInputTypes.wrappedValue = Embedding(embeddingCount: 2, dimensions: config.decoderConfig.hiddenSize)

        _acousticTokenizer.wrappedValue = VibeVoiceAcousticTokenizer(config.acousticTokenizerConfig)

        _acousticConnector.wrappedValue = SpeechConnector(
            inputDim: config.acousticVaeDim,
            outputDim: config.decoderConfig.hiddenSize
        )

        _predictionHead.wrappedValue = VibeVoiceDiffusionHead(config.diffusionHeadConfig)

        _eosClassifier.wrappedValue = EOSClassifier(hiddenSize: config.decoderConfig.hiddenSize)

        self.noiseScheduler = try DPMSolverMultistepScheduler(
            numTrainTimesteps: config.diffusionHeadConfig.ddpmNumSteps,
            betaSchedule: config.diffusionHeadConfig.ddpmBetaSchedule,
            predictionType: config.diffusionHeadConfig.predictionType
        )

        super.init()
    }

    public func getInputEmbeddings(_ inputIds: MLXArray) -> MLXArray {
        languageModel.embedTokens(inputIds)
    }

}
public class VibeVoiceStreamInference {
    public let model: VibeVoiceStreamModel
    public let numInferenceSteps: Int
    public let cfgScale: Float

    internal var lmCache: [KVCacheSimple] = []
    internal var ttsLmCache: [KVCacheSimple] = []
    internal var negLmCache: [KVCacheSimple] = []
    internal var negTtsLmCache: [KVCacheSimple] = []

    internal var lmLastHidden: MLXArray?
    internal var ttsLmLastHidden: MLXArray?
    internal var negTtsLmLastHidden: MLXArray?

    private var cachedTimesteps: [Int32] = []

    public init(model: VibeVoiceStreamModel, numInferenceSteps: Int = 20, cfgScale: Float = 3.0) {
        self.model = model
        self.numInferenceSteps = numInferenceSteps
        self.cfgScale = cfgScale

        model.noiseScheduler.setTimesteps(numInferenceSteps: numInferenceSteps)
        self.cachedTimesteps = model.noiseScheduler.timesteps.asArray(Int32.self)
    }

    public func loadVoiceCache(from path: String) throws {
        let url = URL(fileURLWithPath: path)

        let tensors = try MLX.loadArrays(url: url)

        let lmLayers = model.config.decoderConfig.hiddenLayers - model.config.ttsBackboneNumHiddenLayers
        let ttsLmLayers = model.config.ttsBackboneNumHiddenLayers

        lmCache = (0..<lmLayers).map { _ in KVCacheSimple() }
        ttsLmCache = (0..<ttsLmLayers).map { _ in KVCacheSimple() }
        negLmCache = (0..<lmLayers).map { _ in KVCacheSimple() }
        negTtsLmCache = (0..<ttsLmLayers).map { _ in KVCacheSimple() }

        lmLastHidden = tensors["lm_hidden"]
        for i in 0..<lmLayers {
            guard let key = tensors["lm_key_\(i)"], let value = tensors["lm_value_\(i)"] else {
                throw VibeVoiceError.weightsMissing(key: "lm_key_\(i) or lm_value_\(i)")
            }
            lmCache[i].initialize(keys: key, values: value)
        }

        ttsLmLastHidden = tensors["tts_lm_hidden"]
        for i in 0..<ttsLmLayers {
            guard let key = tensors["tts_lm_key_\(i)"], let value = tensors["tts_lm_value_\(i)"] else {
                throw VibeVoiceError.weightsMissing(key: "tts_lm_key_\(i) or tts_lm_value_\(i)")
            }
            ttsLmCache[i].initialize(keys: key, values: value)
        }

        for i in 0..<lmLayers {
            guard let key = tensors["neg_lm_key_\(i)"], let value = tensors["neg_lm_value_\(i)"] else {
                throw VibeVoiceError.weightsMissing(key: "neg_lm_key_\(i) or neg_lm_value_\(i)")
            }
            negLmCache[i].initialize(keys: key, values: value)
        }

        negTtsLmLastHidden = tensors["neg_tts_lm_hidden"]
        for i in 0..<ttsLmLayers {
            guard let key = tensors["neg_tts_lm_key_\(i)"], let value = tensors["neg_tts_lm_value_\(i)"] else {
                throw VibeVoiceError.weightsMissing(key: "neg_tts_lm_key_\(i) or neg_tts_lm_value_\(i)")
            }
            negTtsLmCache[i].initialize(keys: key, values: value)
        }
    }

    public func resetCaches() {
        let lmLayers = model.config.decoderConfig.hiddenLayers - model.config.ttsBackboneNumHiddenLayers
        lmCache = (0..<lmLayers).map { _ in KVCacheSimple() }
        ttsLmCache = (0..<model.config.ttsBackboneNumHiddenLayers).map { _ in KVCacheSimple() }
        negLmCache = (0..<lmLayers).map { _ in KVCacheSimple() }
        negTtsLmCache = (0..<model.config.ttsBackboneNumHiddenLayers).map { _ in KVCacheSimple() }

        lmLastHidden = nil
        ttsLmLastHidden = nil
        negTtsLmLastHidden = nil
    }

    internal func forwardLM(inputIds: MLXArray, cache: inout [KVCacheSimple]) -> MLXArray {
        let embeddings = model.languageModel.embedTokens(inputIds)
        return model.languageModel.forwardWithEmbeddings(embeddings, cache: cache, applyFinalNorm: false)
    }

    internal func forwardTTSLM(
        inputIds: MLXArray,
        lmHiddenState: MLXArray,
        ttsTextMask: MLXArray,
        cache: inout [KVCacheSimple]
    ) -> MLXArray {
        var inputsEmbeds = model.languageModel.embedTokens(inputIds)

        let startIdx = inputsEmbeds.dim(1) - lmHiddenState.dim(1)
        if startIdx > 0 {
            let prefix = inputsEmbeds[0..., 0..<startIdx, 0...]
            inputsEmbeds = concatenated([prefix, lmHiddenState], axis: 1)
        } else {
            inputsEmbeds = lmHiddenState
        }

        let ttsTypeEmbed = model.ttsInputTypes(ttsTextMask.asType(.int32))
        inputsEmbeds = inputsEmbeds + ttsTypeEmbed

        return model.ttsLanguageModel.forwardWithEmbeddings(inputsEmbeds, cache: cache)
    }

    internal func forwardTTSLMWithAcoustic(
        acousticEmbed: MLXArray,
        cache: inout [KVCacheSimple]
    ) -> MLXArray {
        let batchSize = acousticEmbed.dim(0)
        let speechTypeMask = MLXArray.zeros([batchSize, 1], dtype: .int32)
        let ttsTypeEmbed = model.ttsInputTypes(speechTypeMask.asType(.int32))
        let inputsEmbeds = acousticEmbed + ttsTypeEmbed

        return model.ttsLanguageModel.forwardWithEmbeddings(inputsEmbeds, cache: cache)
    }

    private func generateSpeechTokensCore(
        maxSpeechTokens: Int,
        acousticCache: StreamingConvCache?,
        collectLatentsOnly: Bool
    ) throws -> (scaledLatents: [MLXArray], audioChunks: [MLXArray], tokenCount: Int, eosDetected: Bool) {
        var scaledLatentChunks: [MLXArray] = []
        var audioChunks: [MLXArray] = []
        var tokenCount = 0
        var eosDetected = false

        while tokenCount < maxSpeechTokens {
            guard let ttsHidden = ttsLmLastHidden else {
                throw VibeVoiceError.modelNotInitialized(component: "TTS LM hidden state")
            }
            guard let negTtsHidden = negTtsLmLastHidden else {
                throw VibeVoiceError.modelNotInitialized(component: "Negative TTS LM hidden state")
            }

            let lastIdx = ttsHidden.dim(1) - 1
            let condition = ttsHidden[0..., lastIdx...(lastIdx), 0...].squeezed(axis: 1)

            let negLastIdx = negTtsHidden.dim(1) - 1
            let negCondition = negTtsHidden[0..., negLastIdx...(negLastIdx), 0...].squeezed(axis: 1)

            let speechLatent2D = try sampleSpeechLatent(
                condition: condition,
                negCondition: negCondition
            )

            let speechLatent = expandedDimensions(speechLatent2D, axis: 1)

            let scaledLatent = speechLatent / model.speechScalingFactor - model.speechBiasFactor

            if collectLatentsOnly {
                scaledLatentChunks.append(scaledLatent)
            } else if let cache = acousticCache {
                let audioChunk = model.acousticTokenizer.decode(scaledLatent, cache: cache, useCache: true)
                audioChunks.append(audioChunk)
            }

            let acousticEmbed = model.acousticConnector(speechLatent)

            ttsLmLastHidden = forwardTTSLMWithAcoustic(
                acousticEmbed: acousticEmbed,
                cache: &ttsLmCache
            )

            negTtsLmLastHidden = forwardTTSLMWithAcoustic(
                acousticEmbed: acousticEmbed,
                cache: &negTtsLmCache
            )

            tokenCount += 1

            if let ttsHidden = ttsLmLastHidden, let negTtsHidden = negTtsLmLastHidden {
                eval(ttsHidden, negTtsHidden)
            }

            if try checkEndOfSpeech() {
                eosDetected = true
                break
            }
        }

        return (scaledLatentChunks, audioChunks, tokenCount, eosDetected)
    }

    internal func checkEndOfSpeech() throws -> Bool {
        guard let ttsHidden = ttsLmLastHidden else {
            throw VibeVoiceError.modelNotInitialized(component: "TTS LM")
        }
        let lastIdx = ttsHidden.dim(1) - 1
        let eosHidden = ttsHidden[0..., lastIdx...(lastIdx), 0...].squeezed(axis: 1)

        let eosLogits = model.eosClassifier(eosHidden)
        let eosProb = sigmoid(eosLogits)
        eval(eosProb)
        let prob = eosProb[0, 0].item(Float.self)

        return prob > 0.5
    }

    public func generateWithVoiceCache(tokenIds ttsTextIds: MLXArray, maxSpeechTokens: Int = 500) throws -> MLXArray {
        let collector = AudioChunkCollector()
        let streamer = AudioStreamer()
        streamer.delegate = collector

        try generateStreamingWithVoiceCache(
            tokenIds: ttsTextIds,
            maxSpeechTokens: maxSpeechTokens,
            audioStreamer: streamer
        )

        return collector.concatenatedAudio() ?? MLXArray.zeros([ttsTextIds.dim(0), 1, 0])
    }

    public func generateStreamingWithVoiceCache(
        tokenIds ttsTextIds: MLXArray,
        maxSpeechTokens: Int = 500,
        audioStreamer: AudioStreamer
    ) throws {
        defer { audioStreamer.finish() }

        guard ttsLmLastHidden != nil, negTtsLmLastHidden != nil else {
            throw VibeVoiceError.voiceCacheNotLoaded
        }

        let batchSize = ttsTextIds.dim(0)
        let totalTextTokens = ttsTextIds.dim(1)

        let acousticCache = StreamingConvCache()

        var textWindowIndex = 0
        var totalGeneratedSpeech = 0
        var finished = false

        while !finished && !audioStreamer.cancelled {
            let windowStart = textWindowIndex * TTSConstants.textWindowSize
            let windowEnd = min((textWindowIndex + 1) * TTSConstants.textWindowSize, totalTextTokens)

            if windowStart < totalTextTokens {
                let curTextIds = ttsTextIds[0..., windowStart..<windowEnd]
                let curWindowSize = windowEnd - windowStart

                if curWindowSize > 0 {
                    lmLastHidden = forwardLM(inputIds: curTextIds, cache: &lmCache)

                    guard let lmHidden = lmLastHidden else {
                        throw VibeVoiceError.modelNotInitialized(component: "LM hidden state")
                    }

                    let textMask = MLXArray.ones([batchSize, curWindowSize], dtype: .int32)
                    ttsLmLastHidden = forwardTTSLM(
                        inputIds: curTextIds,
                        lmHiddenState: lmHidden,
                        ttsTextMask: textMask,
                        cache: &ttsLmCache
                    )
                }

                textWindowIndex += 1
            }

            for _ in 0..<TTSConstants.speechWindowSize {
                if totalGeneratedSpeech >= maxSpeechTokens || audioStreamer.cancelled {
                    finished = true
                    break
                }

                guard let ttsHidden = ttsLmLastHidden else {
                    throw VibeVoiceError.modelNotInitialized(component: "TTS LM hidden state")
                }
                guard let negTtsHidden = negTtsLmLastHidden else {
                    throw VibeVoiceError.modelNotInitialized(component: "Negative TTS LM hidden state")
                }

                let lastIdx = ttsHidden.dim(1) - 1
                let condition = ttsHidden[0..., lastIdx...(lastIdx), 0...].squeezed(axis: 1)

                let negLastIdx = negTtsHidden.dim(1) - 1
                let negCondition = negTtsHidden[0..., negLastIdx...(negLastIdx), 0...].squeezed(axis: 1)

                let speechLatent2D = try sampleSpeechLatent(
                    condition: condition,
                    negCondition: negCondition
                )

                let speechLatent = expandedDimensions(speechLatent2D, axis: 1)

                let scaledLatent = speechLatent / model.speechScalingFactor - model.speechBiasFactor

                let audioChunk = model.acousticTokenizer.decode(scaledLatent, cache: acousticCache, useCache: true)

                eval(audioChunk)
                audioStreamer.emit(chunk: audioChunk, index: totalGeneratedSpeech)

                let acousticEmbed = model.acousticConnector(speechLatent)

                ttsLmLastHidden = forwardTTSLMWithAcoustic(
                    acousticEmbed: acousticEmbed,
                    cache: &ttsLmCache
                )

                negTtsLmLastHidden = forwardTTSLMWithAcoustic(
                    acousticEmbed: acousticEmbed,
                    cache: &negTtsLmCache
                )

                totalGeneratedSpeech += 1

                if try checkEndOfSpeech() {
                    finished = true
                    break
                }
            }
        }
    }

    public func generate(tokenIds ttsTextIds: MLXArray, maxSpeechTokens: Int = 500) throws -> MLXArray {
        resetCaches()

        let batchSize = ttsTextIds.dim(0)
        let totalTextTokens = ttsTextIds.dim(1)

        let negTokenIds = MLXArray([Int32(TokenConstants.negativeTextId)]).reshaped([1, 1])
        let negLmHidden = forwardLM(inputIds: negTokenIds, cache: &negLmCache)

        let negTextMask = MLXArray.ones([batchSize, 1], dtype: .int32)
        negTtsLmLastHidden = forwardTTSLM(
            inputIds: negTokenIds,
            lmHiddenState: negLmHidden,
            ttsTextMask: negTextMask,
            cache: &negTtsLmCache
        )

        let acousticCache = StreamingConvCache()

        var audioChunks: [MLXArray] = []
        var textWindowIndex = 0
        var totalGeneratedSpeech = 0
        var finished = false

        while !finished {
            let windowStart = textWindowIndex * TTSConstants.textWindowSize
            let windowEnd = min((textWindowIndex + 1) * TTSConstants.textWindowSize, totalTextTokens)

            if windowStart >= totalTextTokens {
                break
            }

            let curTextIds = ttsTextIds[0..., windowStart..<windowEnd]
            let curWindowSize = windowEnd - windowStart

            if curWindowSize > 0 {
                lmLastHidden = forwardLM(inputIds: curTextIds, cache: &lmCache)

                guard let lmHidden = lmLastHidden else {
                    throw VibeVoiceError.modelNotInitialized(component: "LM hidden state")
                }

                let textMask = MLXArray.ones([batchSize, curWindowSize], dtype: .int32)
                ttsLmLastHidden = forwardTTSLM(
                    inputIds: curTextIds,
                    lmHiddenState: lmHidden,
                    ttsTextMask: textMask,
                    cache: &ttsLmCache
                )

                if let ttsHidden = ttsLmLastHidden {
                    eval(ttsHidden)
                }
            }

            textWindowIndex += 1

            for _ in 0..<TTSConstants.speechWindowSize {
                if totalGeneratedSpeech >= maxSpeechTokens {
                    finished = true
                    break
                }

                let (_, chunks, count, eosDetected) = try generateSpeechTokensCore(
                    maxSpeechTokens: 1,
                    acousticCache: acousticCache,
                    collectLatentsOnly: false
                )

                audioChunks.append(contentsOf: chunks)
                totalGeneratedSpeech += count

                if eosDetected {
                    finished = true
                    break
                }
            }
        }

        if audioChunks.isEmpty {
            return MLXArray.zeros([batchSize, 1, 0])
        }

        let audio = concatenated(audioChunks, axis: -1)
        eval(audio)
        return audio
    }

    internal func sampleSpeechLatent(condition: MLXArray, negCondition: MLXArray) throws -> MLXArray {
        let batchSize = condition.dim(0)
        let latentDim = model.config.diffusionHeadConfig.latentSize

        model.noiseScheduler.reset()

        let combinedCond = concatenated([condition, negCondition], axis: 0)

        var speech = MLXRandom.normal([batchSize, latentDim], dtype: condition.dtype)
        var prevX0: MLXArray? = nil

        for stepIdx in 0..<numInferenceSteps {
            let tVal = Float(cachedTimesteps[stepIdx])
            let timesteps = MLXArray([tVal, tVal])

            let combined = concatenated([speech, speech], axis: 0)

            let eps = model.predictionHead(
                noisyImages: combined,
                timesteps: timesteps,
                condition: combinedCond
            )

            let condEps = eps[0..<batchSize]
            let uncondEps = eps[batchSize...]
            let guidedEps = uncondEps + cfgScale * (condEps - uncondEps)

            let fullEps = concatenated([guidedEps, guidedEps], axis: 0)

            let (newSpeech, x0Pred) = try model.noiseScheduler.stepGPU(
                modelOutput: fullEps,
                stepIdx: stepIdx,
                sample: concatenated([speech, speech], axis: 0),
                prevX0: prevX0
            )

            speech = newSpeech[0..<batchSize]
            prevX0 = x0Pred[0..<batchSize]
        }

        return speech
    }

    public func generateSpeech(
        textEmbeddings: MLXArray,
        ttsInputTypes: MLXArray,
        numFrames: Int
    ) throws -> MLXArray {
        let batchSize = textEmbeddings.dim(0)

        let ttsTypeEmbed = model.ttsInputTypes(ttsInputTypes)
        let conditionedEmbeddings = textEmbeddings + ttsTypeEmbed

        let lmHidden = model.languageModel.forwardWithEmbeddings(conditionedEmbeddings)

        let ttsHidden = model.ttsLanguageModel.forwardWithEmbeddings(lmHidden)

        let latents = try generateLatentsDiffusion(
            condition: ttsHidden,
            numFrames: numFrames,
            batchSize: batchSize
        )

        let audio = model.acousticTokenizer.decode(latents)

        return audio
    }

    public func generateLatentsDiffusion(
        condition: MLXArray,
        negCondition: MLXArray? = nil,
        numFrames: Int,
        batchSize: Int
    ) throws -> MLXArray {
        let latentDim = model.config.diffusionHeadConfig.latentSize

        var latents = MLXRandom.normal([batchSize, numFrames, latentDim], dtype: condition.dtype)

        model.noiseScheduler.reset()

        var prevX0: MLXArray? = nil

        for stepIdx in 0..<numInferenceSteps {
            let tVal = Float(cachedTimesteps[stepIdx])
            let timesteps = MLXArray.ones([batchSize]) * tVal

            let modelOutput: MLXArray
            if let negCond = negCondition, cfgScale > 1.0 {
                let combinedCond = concatenated([condition, negCond], axis: 0)
                let combinedLatents = concatenated([latents, latents], axis: 0)
                let combinedTimesteps = concatenated([timesteps, timesteps], axis: 0)

                let combinedOutput = model.predictionHead(
                    noisyImages: combinedLatents,
                    timesteps: combinedTimesteps,
                    condition: combinedCond
                )

                let condOutput = combinedOutput[0..<batchSize]
                let uncondOutput = combinedOutput[batchSize...]
                modelOutput = uncondOutput + cfgScale * (condOutput - uncondOutput)
            } else {
                modelOutput = model.predictionHead(
                    noisyImages: latents,
                    timesteps: timesteps,
                    condition: condition
                )
            }

            let (newLatents, x0Pred) = try model.noiseScheduler.stepGPU(
                modelOutput: modelOutput,
                stepIdx: stepIdx,
                sample: latents,
                prevX0: prevX0
            )

            latents = newLatents
            prevX0 = x0Pred
        }

        eval(latents)
        return latents
    }

    public func scaleLatentsForDecoding(_ latents: MLXArray) -> MLXArray {
        return latents / model.speechScalingFactor - model.speechBiasFactor
    }

    public func createStreamingSession(audioStreamer: AudioStreamer) throws -> StreamingTextSession {
        guard ttsLmLastHidden != nil, negTtsLmLastHidden != nil else {
            throw VibeVoiceError.voiceCacheNotLoaded
        }
        return StreamingTextSession(inference: self, audioStreamer: audioStreamer)
    }
}

public class StreamingTextSession {
    private let inference: VibeVoiceStreamInference
    private let audioStreamer: AudioStreamer
    private let acousticCache: StreamingConvCache

    private var pendingTokens: [Int32] = []
    private var totalGeneratedSpeech = 0
    private var isFinished = false

    public init(inference: VibeVoiceStreamInference, audioStreamer: AudioStreamer) {
        self.inference = inference
        self.audioStreamer = audioStreamer
        self.acousticCache = StreamingConvCache()
    }

    public func addTokens(_ tokens: [Int32]) throws {
        guard !isFinished else { return }
        pendingTokens.append(contentsOf: tokens)

        while pendingTokens.count >= TTSConstants.textWindowSize && !isFinished {
            try processOneTextWindow()
        }
    }

    public func flush(maxSpeechTokens: Int = 500) throws {
        guard !isFinished else { return }

        if !pendingTokens.isEmpty {
            try processRemainingTokens()
        }

        while !isFinished && totalGeneratedSpeech < maxSpeechTokens && !audioStreamer.cancelled {
            if try generateOneSpeechToken() {
                isFinished = true
                break
            }
        }

        audioStreamer.finish()
        isFinished = true
    }

    private func processOneTextWindow() throws {
        let windowTokens = Array(pendingTokens.prefix(TTSConstants.textWindowSize))
        pendingTokens.removeFirst(min(TTSConstants.textWindowSize, pendingTokens.count))

        let tokenIds = MLXArray(windowTokens).reshaped([1, windowTokens.count])
        try processTextTokens(tokenIds)

        for _ in 0..<TTSConstants.speechWindowSize {
            if audioStreamer.cancelled {
                isFinished = true
                break
            }
            if try generateOneSpeechToken() {
                isFinished = true
                break
            }
        }
    }

    private func processRemainingTokens() throws {
        guard !pendingTokens.isEmpty else { return }

        let remainingTokenCount = pendingTokens.count
        let tokenIds = MLXArray(pendingTokens).reshaped([1, pendingTokens.count])
        pendingTokens.removeAll()
        try processTextTokens(tokenIds)

        let speechTokensToGenerate = max(
            1,
            (remainingTokenCount * TTSConstants.speechWindowSize) / TTSConstants.textWindowSize
        )
        for _ in 0..<speechTokensToGenerate {
            if audioStreamer.cancelled {
                isFinished = true
                break
            }
            if try generateOneSpeechToken() {
                isFinished = true
                break
            }
        }
    }

    private func processTextTokens(_ tokenIds: MLXArray) throws {
        let curWindowSize = tokenIds.dim(1)
        guard curWindowSize > 0 else { return }

        inference.lmLastHidden = inference.forwardLM(inputIds: tokenIds, cache: &inference.lmCache)

        guard let lmHidden = inference.lmLastHidden else {
            throw VibeVoiceError.modelNotInitialized(component: "LM hidden state")
        }

        let textMask = MLXArray.ones([1, curWindowSize], dtype: .int32)
        inference.ttsLmLastHidden = inference.forwardTTSLM(
            inputIds: tokenIds,
            lmHiddenState: lmHidden,
            ttsTextMask: textMask,
            cache: &inference.ttsLmCache
        )

        if let ttsHidden = inference.ttsLmLastHidden {
            eval(ttsHidden)
        }
    }

    private func generateOneSpeechToken() throws -> Bool {
        guard let ttsHidden = inference.ttsLmLastHidden else {
            throw VibeVoiceError.modelNotInitialized(component: "TTS LM hidden state")
        }
        guard let negTtsHidden = inference.negTtsLmLastHidden else {
            throw VibeVoiceError.modelNotInitialized(component: "Negative TTS LM hidden state")
        }

        let lastIdx = ttsHidden.dim(1) - 1
        let condition = ttsHidden[0..., lastIdx...(lastIdx), 0...].squeezed(axis: 1)

        let negLastIdx = negTtsHidden.dim(1) - 1
        let negCondition = negTtsHidden[0..., negLastIdx...(negLastIdx), 0...].squeezed(axis: 1)

        let speechLatent2D = try inference.sampleSpeechLatent(
            condition: condition,
            negCondition: negCondition
        )

        let speechLatent = expandedDimensions(speechLatent2D, axis: 1)

        let scaledLatent = speechLatent / inference.model.speechScalingFactor - inference.model.speechBiasFactor

        let audioChunk = inference.model.acousticTokenizer.decode(scaledLatent, cache: acousticCache, useCache: true)

        eval(audioChunk)
        audioStreamer.emit(chunk: audioChunk, index: totalGeneratedSpeech)

        let acousticEmbed = inference.model.acousticConnector(speechLatent)

        inference.ttsLmLastHidden = inference.forwardTTSLMWithAcoustic(
            acousticEmbed: acousticEmbed,
            cache: &inference.ttsLmCache
        )

        inference.negTtsLmLastHidden = inference.forwardTTSLMWithAcoustic(
            acousticEmbed: acousticEmbed,
            cache: &inference.negTtsLmCache
        )

        totalGeneratedSpeech += 1

        return try inference.checkEndOfSpeech()
    }
}
