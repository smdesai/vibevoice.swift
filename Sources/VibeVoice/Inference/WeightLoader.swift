import Foundation
import MLX
import MLXNN

public enum WeightLoadingError: Error {
    case fileNotFound(String)
    case configNotFound(String)
    case weightKeyMissing(String)
    case invalidWeightShape(key: String, expected: [Int], got: [Int])
}

public func loadVibeVoiceConfiguration(from directory: URL) throws -> VibeVoiceConfiguration {
    let configURL = directory.appendingPathComponent("config.json")
    let data = try Data(contentsOf: configURL)
    let decoder = JSONDecoder()
    return try decoder.decode(VibeVoiceConfiguration.self, from: data)
}

func loadWeights(from url: URL) throws -> [String: MLXArray] {
    try MLX.loadArrays(url: url)
}

func materializeWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    eval(Array(weights.values))
    return weights
}

func loadWeightsFromDirectory(_ directory: URL) throws -> [String: MLXArray] {
    let fileManager = FileManager.default
    let contents = try fileManager.contentsOfDirectory(
        at: directory, includingPropertiesForKeys: nil)

    var allWeights: [String: MLXArray] = [:]

    for file in contents {
        if file.pathExtension == "safetensors" {
            let weights = try loadWeights(from: file)
            for (key, value) in weights {
                allWeights[key] = value
            }
        }
    }

    return allWeights
}

private func mapCommonWeightKeys(_ key: String) -> String {
    var newKey = key
    newKey = newKey.replacingOccurrences(
        of: "adaLN_modulation.1.", with: "adaLN_modulation.linear.")
    newKey = newKey.replacingOccurrences(of: "t_embedder.mlp.0.", with: "t_embedder.mlp.linear1.")
    newKey = newKey.replacingOccurrences(of: "t_embedder.mlp.2.", with: "t_embedder.mlp.linear2.")
    return newKey
}

private let decoderStageOffsets: [Int] = [0, 8, 11, 14, 17, 20, 23]

// Static regex to avoid recompilation on every weight key
private let stagesRegex: NSRegularExpression? = try? NSRegularExpression(
    pattern: #"\.stages\.(\d+)\.(\d+)\."#)

private func flattenDecoderStagesKey(_ key: String) -> String {
    guard let regex = stagesRegex,
        let match = regex.firstMatch(in: key, range: NSRange(key.startIndex..., in: key)),
        let stageRange = Range(match.range(at: 1), in: key),
        let blockRange = Range(match.range(at: 2), in: key),
        let stageIdx = Int(key[stageRange]),
        let blockIdx = Int(key[blockRange]),
        stageIdx < decoderStageOffsets.count
    else {
        return key
    }

    let flatIdx = decoderStageOffsets[stageIdx] + blockIdx
    let matchRange = Range(match.range, in: key)!
    return key.replacingCharacters(in: matchRange, with: ".stages.\(flatIdx).")
}

private func mapDecoderWeightKeys(_ key: String) -> String {
    var newKey = key
    newKey = newKey.replacingOccurrences(of: ".upsample_layers.", with: ".upsampleLayers.")
    if let range = newKey.range(of: #"\.upsampleLayers\.(\d+)\.0\."#, options: .regularExpression) {
        let match = newKey[range]
        if let indexMatch = match.range(of: #"\d+"#, options: .regularExpression) {
            let index = String(match[indexMatch])
            newKey = newKey.replacingOccurrences(
                of: ".upsampleLayers.\(index).0.", with: ".upsampleLayers.\(index).")
        }
    }
    newKey = newKey.replacingOccurrences(of: ".mixer.conv.conv.conv.", with: ".mixer.conv.")
    newKey = newKey.replacingOccurrences(of: ".conv.conv.weight", with: ".conv.weight")
    newKey = newKey.replacingOccurrences(of: ".conv.conv.bias", with: ".conv.bias")
    newKey = newKey.replacingOccurrences(of: ".convtr.convtr.weight", with: ".convtr.weight")
    newKey = newKey.replacingOccurrences(of: ".convtr.convtr.bias", with: ".convtr.bias")
    newKey = newKey.replacingOccurrences(of: ".ffn_gamma", with: ".ffnGamma")
    newKey = newKey.replacingOccurrences(of: ".ffn_norm.", with: ".ffnNorm.")
    newKey = flattenDecoderStagesKey(newKey)
    return newKey
}

private func mapEncoderWeightKeys(_ key: String) -> String {
    var newKey = key
    newKey = newKey.replacingOccurrences(of: ".downsample_layers.", with: ".downsample_layers.")
    if let range = newKey.range(
        of: #"\.downsample_layers\.(\d+)\.0\."#, options: .regularExpression)
    {
        let match = newKey[range]
        if let indexMatch = match.range(of: #"\d+"#, options: .regularExpression) {
            let index = String(match[indexMatch])
            newKey = newKey.replacingOccurrences(
                of: ".downsample_layers.\(index).0.", with: ".downsample_layers.\(index).")
        }
    }
    newKey = newKey.replacingOccurrences(of: ".mixer.conv.conv.conv.", with: ".mixer.conv.")
    newKey = newKey.replacingOccurrences(of: ".conv.conv.weight", with: ".conv.weight")
    newKey = newKey.replacingOccurrences(of: ".conv.conv.bias", with: ".conv.bias")
    newKey = newKey.replacingOccurrences(of: ".ffn_gamma", with: ".ffnGamma")
    newKey = newKey.replacingOccurrences(of: ".ffn_norm.", with: ".ffnNorm.")
    return newKey
}

func mapVibeVoiceWeightKeys(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var mapped: [String: MLXArray] = [:]

    let prefixMappings: [(from: String, to: String)] = [
        ("model.language_model.", "language_model."),
        ("model.tts_language_model.", "tts_language_model."),
        ("model.prediction_head.", "prediction_head."),
        ("model.acoustic_tokenizer.", "acoustic_tokenizer."),
        ("model.acoustic_connector.", "acoustic_connector."),
        ("model.tts_input_types.", "tts_input_types."),
        ("tts_eos_classifier.", "tts_eos_classifier."),
    ]

    for (key, value) in weights {
        var newKey = key

        for (from, to) in prefixMappings {
            if key.hasPrefix(from) {
                newKey = to + key.dropFirst(from.count)
                break
            }
        }

        if key == "model.speech_scaling_factor" || key == "model.speech_bias_factor" {
            continue
        }

        newKey = mapCommonWeightKeys(newKey)

        if newKey.contains("acoustic_tokenizer.decoder.") {
            newKey = mapDecoderWeightKeys(newKey)
        }
        if newKey.contains("acoustic_tokenizer.encoder.") {
            newKey = mapEncoderWeightKeys(newKey)
        }

        mapped[newKey] = value
    }

    return mapped
}

func transposeConv1dWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var transposed: [String: MLXArray] = [:]

    for (key, value) in weights {
        if key.contains("acoustic_tokenizer.decoder.") && key.hasSuffix(".weight")
            && value.ndim == 3
        {

            if key.contains(".convtr.") {
                let transposedWeight = value.transposed(1, 2, 0)
                transposed[key] = transposedWeight
            } else {
                let transposedWeight = value.transposed(0, 2, 1)
                transposed[key] = transposedWeight
            }
        } else if key.contains("acoustic_tokenizer.encoder.") && key.hasSuffix(".weight")
            && value.ndim == 3
        {
            let transposedWeight = value.transposed(0, 2, 1)
            transposed[key] = transposedWeight
        } else {
            transposed[key] = value
        }
    }

    return transposed
}

private func loadStagesWeightsForDecoder(_ decoder: TokenizerDecoder, weights: [String: MLXArray]) {
    for (flatIdx, block) in decoder.blocks.enumerated() {
        let prefix = "acoustic_tokenizer.decoder.stages.\(flatIdx)"

        if let normWeight = weights["\(prefix).norm.weight"] {
            let params = ModuleParameters.unflattened(["weight": normWeight])
            try? block.norm.update(parameters: params, verify: .none)
        }

        if let ffnNormWeight = weights["\(prefix).ffnNorm.weight"] {
            let params = ModuleParameters.unflattened(["weight": ffnNormWeight])
            try? block.ffnNorm.update(parameters: params, verify: .none)
        }

        var mixerWeights: [String: MLXArray] = [:]
        if let w = weights["\(prefix).mixer.conv.weight"] { mixerWeights["weight"] = w }
        if let b = weights["\(prefix).mixer.conv.bias"] { mixerWeights["bias"] = b }
        if !mixerWeights.isEmpty {
            let params = ModuleParameters.unflattened(mixerWeights)
            try? block.mixer.conv.update(parameters: params, verify: .none)
        }

        var ffnLinear1Weights: [String: MLXArray] = [:]
        if let w = weights["\(prefix).ffn.linear1.weight"] { ffnLinear1Weights["weight"] = w }
        if let b = weights["\(prefix).ffn.linear1.bias"] { ffnLinear1Weights["bias"] = b }
        if let s = weights["\(prefix).ffn.linear1.scales"] { ffnLinear1Weights["scales"] = s }
        if let bs = weights["\(prefix).ffn.linear1.biases"] { ffnLinear1Weights["biases"] = bs }
        if !ffnLinear1Weights.isEmpty {
            let params = ModuleParameters.unflattened(ffnLinear1Weights)
            do {
                try block.ffn.linear1.update(parameters: params, verify: .none)
            } catch {
                print("[DEBUG] Error loading linear1 weights for \(prefix): \(error)")
            }
        }

        var ffnLinear2Weights: [String: MLXArray] = [:]
        if let w = weights["\(prefix).ffn.linear2.weight"] { ffnLinear2Weights["weight"] = w }
        if let b = weights["\(prefix).ffn.linear2.bias"] { ffnLinear2Weights["bias"] = b }
        if let s = weights["\(prefix).ffn.linear2.scales"] { ffnLinear2Weights["scales"] = s }
        if let bs = weights["\(prefix).ffn.linear2.biases"] { ffnLinear2Weights["biases"] = bs }
        if !ffnLinear2Weights.isEmpty {
            let params = ModuleParameters.unflattened(ffnLinear2Weights)
            do {
                try block.ffn.linear2.update(parameters: params, verify: .none)
            } catch {
                print("[DEBUG] Error loading linear2 weights for \(prefix): \(error)")
            }
        }

        if let gamma = weights["\(prefix).gamma"] {
            block.gamma = gamma
        }
        if let ffnGamma = weights["\(prefix).ffnGamma"] {
            block.ffnGamma = ffnGamma
        }

    }
}

private func loadHeadWeightsForDecoder(_ decoder: TokenizerDecoder, weights: [String: MLXArray]) {
    let prefix = "acoustic_tokenizer.decoder.head"
    var headWeights: [String: MLXArray] = [:]
    if let w = weights["\(prefix).conv.weight"] { headWeights["weight"] = w }
    if let b = weights["\(prefix).conv.bias"] { headWeights["bias"] = b }
    if !headWeights.isEmpty {
        let params = ModuleParameters.unflattened(headWeights)
        try? decoder.head.conv.update(parameters: params, verify: .none)
    }
}

private func loadUpsampleLayersWeightsForDecoder(
    _ decoder: TokenizerDecoder, weights: [String: MLXArray]
) {
    for (idx, layer) in decoder.upsampleLayers.enumerated() {
        if let sconv = layer as? SConv1d {
            let weightKey = "acoustic_tokenizer.decoder.upsampleLayers.\(idx).conv.weight"
            let biasKey = "acoustic_tokenizer.decoder.upsampleLayers.\(idx).conv.bias"

            var convWeights: [String: MLXArray] = [:]
            if let weight = weights[weightKey] {
                convWeights["weight"] = weight
            }
            if let bias = weights[biasKey] {
                convWeights["bias"] = bias
            }
            if !convWeights.isEmpty {
                let params = ModuleParameters.unflattened(convWeights)
                sconv.conv.update(parameters: params)
            }
        } else if let sconvtr = layer as? SConvTranspose1d {
            let weightKey = "acoustic_tokenizer.decoder.upsampleLayers.\(idx).convtr.weight"
            let biasKey = "acoustic_tokenizer.decoder.upsampleLayers.\(idx).convtr.bias"

            var convWeights: [String: MLXArray] = [:]
            if let weight = weights[weightKey] {
                convWeights["weight"] = weight
            }
            if let bias = weights[biasKey] {
                convWeights["bias"] = bias
            }
            if !convWeights.isEmpty {
                let params = ModuleParameters.unflattened(convWeights)
                sconvtr.convtr.update(parameters: params)
            }
        }
    }

    var weightsToEval: [MLXArray] = []
    weightsToEval.reserveCapacity(decoder.upsampleLayers.count * 2)
    for layer in decoder.upsampleLayers {
        if let sconv = layer as? SConv1d {
            weightsToEval.append(sconv.conv.weight)
            if let b = sconv.conv.bias { weightsToEval.append(b) }
        } else if let sconvtr = layer as? SConvTranspose1d {
            weightsToEval.append(sconvtr.convtr.weight)
            if let b = sconvtr.convtr.bias { weightsToEval.append(b) }
        }
    }
    eval(weightsToEval)
}

public func loadVibeVoiceStreamModel(from directory: URL) throws -> VibeVoiceStreamModel {
    let config = try loadVibeVoiceConfiguration(from: directory)
    let model = try VibeVoiceStreamModel(config)

    let isQuantized = VibeVoiceQuantizer.hasQuantization(at: directory)
    var quantManifest: VibeVoiceQuantizationManifest?
    if isQuantized {
        let manifestURL = directory.appendingPathComponent("quantization.json")
        quantManifest = try VibeVoiceQuantizationManifest.load(from: manifestURL)
    }

    var weights = try loadWeightsFromDirectory(directory)
    // Skip early materializeWeights - final eval(allParams) will materialize everything

    let scalingFactor = weights["model.speech_scaling_factor"]
    let biasFactor = weights["model.speech_bias_factor"]

    // Remove rotary embedding keys in-place instead of filtering entire dictionary
    let keysToRemove = weights.keys.filter { $0.contains("rotary_emb.inv_freq") }
    for key in keysToRemove {
        weights.removeValue(forKey: key)
    }

    var mappedWeights = mapVibeVoiceWeightKeys(weights)
    mappedWeights = transposeConv1dWeights(mappedWeights)

    let availableKeys = Set(mappedWeights.keys)

    if let sf = scalingFactor {
        model.speechScalingFactor = sf
    }
    if let bf = biasFactor {
        model.speechBiasFactor = bf
    }

    if let manifest = quantManifest {
        VibeVoiceQuantizer.applyQuantization(
            to: model,
            manifest: manifest,
            availableKeys: availableKeys
        )
    }

    let parameters = ModuleParameters.unflattened(mappedWeights)
    try model.update(parameters: parameters, verify: .none)

    loadUpsampleLayersWeightsForDecoder(model.acousticTokenizer.decoder, weights: mappedWeights)
    loadHeadWeightsForDecoder(model.acousticTokenizer.decoder, weights: mappedWeights)

    let allParams = model.parameters().flattened().map { $0.1 }
    eval(allParams)

    eval([
        model.noiseScheduler.betas,
        model.noiseScheduler.alphas,
        model.noiseScheduler.alphasCumprod,
        model.noiseScheduler.alphaT,
        model.noiseScheduler.sigmaT,
        model.noiseScheduler.lambdaT,
        model.noiseScheduler.sigmas,
    ])

    return model
}

extension VibeVoiceStreamModel {
    public static func fromPretrained(modelId: String) throws -> VibeVoiceStreamModel {
        guard
            let cachedURL = ModelResolution.findCachedModel(modelId: modelId, requireWeights: true)
        else {
            throw ModelResolutionError.cacheNotFound(modelId)
        }
        return try loadVibeVoiceStreamModel(from: cachedURL)
    }
}
