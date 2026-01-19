import Foundation

class VoiceDownloadService: ObservableObject {
    static let shared = VoiceDownloadService()

    @Published var downloadProgress: [String: Double] = [:]
    @Published var downloadedVoices: Set<String> = []

    private let fileManager = FileManager.default
    private var downloadTasks: [String: URLSessionDownloadTask] = [:]

    private init() {
        loadDownloadedVoices()
    }

    var voiceCacheDirectory: URL {
        let documentsURL = fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let cacheDir = documentsURL.appendingPathComponent("VoiceCaches", isDirectory: true)

        if !fileManager.fileExists(atPath: cacheDir.path) {
            try? fileManager.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        }

        return cacheDir
    }

    func voicePath(for voice: Voice) -> URL {
        voiceCacheDirectory.appendingPathComponent(voice.cacheFileName)
    }

    func isVoiceDownloaded(_ voice: Voice) -> Bool {
        let path = voicePath(for: voice)
        return fileManager.fileExists(atPath: path.path)
    }

    func loadDownloadedVoices() {
        var downloaded = Set<String>()
        for voice in Voice.allVoices {
            if isVoiceDownloaded(voice) {
                downloaded.insert(voice.id)
            }
        }
        DispatchQueue.main.async {
            self.downloadedVoices = downloaded
        }
    }

    func downloadVoice(_ voice: Voice) async throws -> URL {
        let destinationURL = voicePath(for: voice)

        if fileManager.fileExists(atPath: destinationURL.path) {
            return destinationURL
        }

        await MainActor.run {
            downloadProgress[voice.id] = 0.0
        }

        let (tempURL, _) = try await URLSession.shared.download(
            from: voice.huggingFaceURL,
            delegate: ProgressDelegate { progress in
                Task { @MainActor in
                    self.downloadProgress[voice.id] = progress
                }
            })

        try fileManager.moveItem(at: tempURL, to: destinationURL)

        await MainActor.run {
            self.downloadProgress.removeValue(forKey: voice.id)
            self.downloadedVoices.insert(voice.id)
        }

        return destinationURL
    }

    func cancelDownload(for voice: Voice) {
        downloadTasks[voice.id]?.cancel()
        downloadTasks.removeValue(forKey: voice.id)
        DispatchQueue.main.async {
            self.downloadProgress.removeValue(forKey: voice.id)
        }
    }

    func deleteVoice(_ voice: Voice) throws {
        let path = voicePath(for: voice)
        if fileManager.fileExists(atPath: path.path) {
            try fileManager.removeItem(at: path)
            DispatchQueue.main.async {
                self.downloadedVoices.remove(voice.id)
            }
        }
    }
}

private class ProgressDelegate: NSObject, URLSessionTaskDelegate {
    private let progressHandler: (Double) -> Void

    init(progressHandler: @escaping (Double) -> Void) {
        self.progressHandler = progressHandler
    }

    func urlSession(_ session: URLSession, didCreateTask task: URLSessionTask) {
        task.progress.addObserver(
            self, forKeyPath: "fractionCompleted", options: .new, context: nil)
    }

    override func observeValue(
        forKeyPath keyPath: String?, of object: Any?, change: [NSKeyValueChangeKey: Any]?,
        context: UnsafeMutableRawPointer?
    ) {
        if keyPath == "fractionCompleted", let progress = object as? Progress {
            progressHandler(progress.fractionCompleted)
        }
    }
}
