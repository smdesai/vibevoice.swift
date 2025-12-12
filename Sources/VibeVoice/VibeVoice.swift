import Foundation
import MLX
import MLXNN

public struct VibeVoice {
    public static let version = "0.1.0"

    public init() {}

    public static func load(from modelPath: URL) throws -> VibeVoiceStreamInference {
        let model = try loadVibeVoiceStreamModel(from: modelPath)
        return VibeVoiceStreamInference(model: model)
    }

    public static func fromPretrained(_ modelId: String) throws -> VibeVoiceStreamInference {
        let model = try VibeVoiceStreamModel.fromPretrained(modelId: modelId)
        return VibeVoiceStreamInference(model: model)
    }
}
