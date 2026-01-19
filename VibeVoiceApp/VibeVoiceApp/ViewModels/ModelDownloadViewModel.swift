import Foundation
import SwiftUI

@MainActor
class ModelDownloadViewModel: ObservableObject {
    enum State: Equatable {
        case checking
        case notDownloaded
        case downloading(progress: Double)
        case loading(progress: Double)
        case completed
        case error(String)

        static func == (lhs: State, rhs: State) -> Bool {
            switch (lhs, rhs) {
            case (.checking, .checking),
                (.notDownloaded, .notDownloaded),
                (.completed, .completed):
                return true
            case (.downloading(let p1), .downloading(let p2)):
                return p1 == p2
            case (.loading(let p1), .loading(let p2)):
                return p1 == p2
            case (.error(let e1), .error(let e2)):
                return e1 == e2
            default:
                return false
            }
        }
    }

    @Published var state: State = .checking
    @Published var statusMessage: String = "Checking model..."

    private let ttsService: TTSService

    init(ttsService: TTSService) {
        self.ttsService = ttsService
    }

    func checkAndDownloadModel() async {
        state = .checking
        statusMessage = "Checking for model..."

        do {
            try await ttsService.loadModel { [weak self] progress, message in
                Task { @MainActor in
                    let isDownloading = message.lowercased().contains("download")
                    if isDownloading {
                        self?.state = .downloading(progress: progress)
                    } else {
                        self?.state = .loading(progress: progress)
                    }
                    self?.statusMessage = message
                }
            }
            state = .completed
            statusMessage = "Model loaded successfully"
        } catch {
            state = .error(error.localizedDescription)
            statusMessage = "Error: \(error.localizedDescription)"
        }
    }

    func retry() {
        Task {
            await checkAndDownloadModel()
        }
    }
}
