import Foundation
import Hub

public enum VibeVoiceRepository {
    public static let id = "microsoft/VibeVoice-Realtime-0.5B"
    public static let revision = "main"
}

public enum Qwen2TokenizerRepository {
    public static let id = "Qwen/Qwen2.5-0.5B"
    public static let revision = "main"
}

public enum ModelResolutionError: Error, LocalizedError {
    case modelNotFound(String)
    case cacheNotFound(String)
    case downloadFailed(String, Error)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let spec):
            return "Model not found: \(spec)"
        case .cacheNotFound(let modelId):
            return "Model '\(modelId)' not found in cache."
        case .downloadFailed(let modelId, let error):
            return "Failed to download model '\(modelId)': \(error.localizedDescription)"
        }
    }
}

public struct ModelResolution {

    public static func isHuggingFaceModelId(_ modelSpec: String) -> Bool {
        if modelSpec.hasPrefix("/") || modelSpec.hasPrefix("./") || modelSpec.hasPrefix("../") {
            return false
        }

        let url = URL(fileURLWithPath: modelSpec).standardizedFileURL
        if FileManager.default.fileExists(atPath: url.path) {
            return false
        }

        let baseSpec = String(modelSpec.split(separator: ":")[0])
        let parts = baseSpec.split(separator: "/")

        guard parts.count == 2 else {
            return false
        }

        let org = String(parts[0])
        let repo = String(parts[1])

        guard !org.isEmpty && !repo.isEmpty else {
            return false
        }

        let pathIndicators = [
            "models", "model", "weights", "data", "datasets", "checkpoints", "output", "tmp",
            "temp", "cache",
        ]
        if pathIndicators.contains(org.lowercased()) {
            return false
        }

        if org.filter({ $0 == "." }).count > 1 {
            return false
        }

        return true
    }

    public static func resolve(
        modelSpec: String,
        requireWeights: Bool = true
    ) async throws -> URL {
        let localURL = URL(fileURLWithPath: modelSpec).standardizedFileURL
        if FileManager.default.fileExists(atPath: localURL.path) {
            return localURL
        }

        if !isHuggingFaceModelId(modelSpec) {
            throw ModelResolutionError.modelNotFound(modelSpec)
        }

        let parts = modelSpec.split(separator: ":", maxSplits: 1)
        let modelId = String(parts[0])
        let revision = parts.count > 1 ? String(parts[1]) : "main"

        if let cachedURL = findCachedModel(modelId: modelId, requireWeights: requireWeights) {
            return cachedURL
        }

        return try await downloadModel(
            modelId: modelId,
            revision: revision,
            requireWeights: requireWeights
        )
    }

    public static func resolveOrDefault(
        modelSpec: String?,
        defaultModelId: String = VibeVoiceRepository.id,
        requireWeights: Bool = true
    ) async throws -> URL {
        if let spec = modelSpec {
            return try await resolve(
                modelSpec: spec,
                requireWeights: requireWeights
            )
        }

        if let cachedURL = findCachedModel(modelId: defaultModelId, requireWeights: requireWeights)
        {
            return cachedURL
        }

        return try await downloadModel(
            modelId: defaultModelId,
            revision: "main",
            requireWeights: requireWeights
        )
    }

    public static func resolveTokenizer(
        modelSpec: String?,
        defaultModelId: String = Qwen2TokenizerRepository.id
    ) async throws -> URL {
        if let spec = modelSpec {
            return try await resolve(
                modelSpec: spec,
                requireWeights: false
            )
        }

        if let cachedURL = findCachedModel(modelId: defaultModelId, requireWeights: false) {
            return cachedURL
        }

        return try await downloadModel(
            modelId: defaultModelId,
            revision: "main",
            requireWeights: false
        )
    }

    public static func getHuggingFaceCacheDirectory() -> URL {
        let env = ProcessInfo.processInfo.environment

        if let hubCache = env["HF_HUB_CACHE"], !hubCache.isEmpty {
            return URL(fileURLWithPath: hubCache)
        }

        if let hfHome = env["HF_HOME"], !hfHome.isEmpty {
            return URL(fileURLWithPath: hfHome).appendingPathComponent("hub")
        }

        #if os(iOS) || os(tvOS) || os(watchOS) || os(visionOS)
            // On iOS, use the app's Caches directory
            let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)
                .first!
            return cacheDir.appendingPathComponent("huggingface/hub")
        #else
            let homeDir = FileManager.default.homeDirectoryForCurrentUser
            return homeDir.appendingPathComponent(".cache/huggingface/hub")
        #endif
    }

    public static func findCachedModel(modelId: String, requireWeights: Bool = true) -> URL? {
        let fm = FileManager.default
        let cacheDir = getHuggingFaceCacheDirectory()

        let modelCachePath =
            cacheDir
            .appendingPathComponent("models--\(modelId.replacingOccurrences(of: "/", with: "--"))")
            .appendingPathComponent("snapshots")

        guard fm.fileExists(atPath: modelCachePath.path) else {
            return nil
        }

        guard
            let snapshots = try? fm.contentsOfDirectory(
                at: modelCachePath,
                includingPropertiesForKeys: [.contentModificationDateKey],
                options: [.skipsHiddenFiles]
            )
        else {
            return nil
        }

        let sortedSnapshots = snapshots.sorted { a, b in
            let aDate =
                (try? a.resourceValues(forKeys: [.contentModificationDateKey])
                    .contentModificationDate) ?? .distantPast
            let bDate =
                (try? b.resourceValues(forKeys: [.contentModificationDateKey])
                    .contentModificationDate) ?? .distantPast
            return aDate > bDate
        }

        for snapshot in sortedSnapshots {
            let configFile = snapshot.appendingPathComponent("config.json")

            guard fm.fileExists(atPath: configFile.path) else {
                continue
            }

            if requireWeights {
                let contents =
                    (try? fm.contentsOfDirectory(at: snapshot, includingPropertiesForKeys: nil))
                    ?? []
                let hasSafetensors = contents.contains { $0.pathExtension == "safetensors" }

                if !hasSafetensors {
                    continue
                }
            }

            return snapshot
        }

        return nil
    }

    public static func downloadModel(
        modelId: String,
        revision: String = "main",
        requireWeights: Bool = true,
        progressHandler: ((Progress) -> Void)? = nil
    ) async throws -> URL {
        let repo = Hub.Repo(id: modelId)

        var patterns: [String] = [
            "config.json", "tokenizer.json", "tokenizer_config.json", "quantization.json",
        ]
        if requireWeights {
            patterns.append("*.safetensors")
        }

        do {
            let hubApi = HubApi()
            let modelURL = try await hubApi.snapshot(
                from: repo,
                matching: patterns,
                progressHandler: progressHandler ?? { progress in
                    let percent = Int(progress.fractionCompleted * 100)
                    print("\rDownloading \(modelId): \(percent)%", terminator: "")
                    fflush(stdout)
                    if progress.isFinished {
                        print()
                    }
                }
            )
            return modelURL
        } catch {
            throw ModelResolutionError.downloadFailed(modelId, error)
        }
    }
}
