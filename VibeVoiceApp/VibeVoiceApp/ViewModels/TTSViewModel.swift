import Combine
import Foundation
import SwiftUI

@MainActor
class TTSViewModel: ObservableObject {
    enum State: Equatable {
        case idle
        case loadingVoice
        case ready
        case generating
        case playing
        case error(String)

        static func == (lhs: State, rhs: State) -> Bool {
            switch (lhs, rhs) {
            case (.idle, .idle),
                (.loadingVoice, .loadingVoice),
                (.ready, .ready),
                (.generating, .generating),
                (.playing, .playing):
                return true
            case (.error(let e1), .error(let e2)):
                return e1 == e2
            default:
                return false
            }
        }
    }

    @Published var state: State = .idle
    @Published var text: String =
        "Hello! This is VibeVoice, a real-time text to speech model running locally on your device."
    @Published var selectedVoice: Voice? {
        didSet {
            if let voice = selectedVoice, voice != oldValue, isModelReady {
                Task {
                    await loadVoice(voice)
                }
            }
        }
    }
    @Published var isGenerating: Bool = false
    @Published var waveformAmplitude: CGFloat = 0

    let ttsService: TTSService
    @Published var voiceDownloadService = VoiceDownloadService.shared

    private var cancellables = Set<AnyCancellable>()
    private var waveformTimer: Timer?
    private var isModelReady = false

    init(ttsService: TTSService) {
        self.ttsService = ttsService
        // Set default selection to Carter (will be synced when model is ready)
        selectedVoice = Voice.allVoices.first(where: { $0.id == "en-Carter" })
    }

    /// Call this after model loading is complete to sync state
    func onModelReady() {
        isModelReady = true
        // Voice is already loaded by TTSService, just update state
        Task {
            if let loadedVoice = await ttsService.getCurrentVoice() {
                selectedVoice = loadedVoice
            }
            state = .ready
        }
    }

    func loadVoice(_ voice: Voice) async {
        state = .loadingVoice

        do {
            try await ttsService.loadVoice(voice)
            state = .ready
        } catch {
            state = .error("Failed to load voice: \(error.localizedDescription)")
        }
    }

    func generate() {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            state = .error("Please enter some text to speak")
            return
        }

        guard selectedVoice != nil else {
            state = .error("Please select a voice first")
            return
        }

        state = .generating
        isGenerating = true

        Task {
            do {
                try await ttsService.generateAndPlay(
                    text: text,
                    onStart: { [weak self] in
                        Task { @MainActor in
                            self?.state = .playing
                            self?.startWaveformAnimation()
                        }
                    },
                    onComplete: { [weak self] in
                        Task { @MainActor in
                            self?.state = .ready
                            self?.isGenerating = false
                            self?.stopWaveformAnimation()
                        }
                    },
                    onError: { [weak self] error in
                        Task { @MainActor in
                            self?.state = .error(error.localizedDescription)
                            self?.isGenerating = false
                            self?.stopWaveformAnimation()
                        }
                    }
                )
            } catch {
                state = .error(error.localizedDescription)
                isGenerating = false
                stopWaveformAnimation()
            }
        }
    }

    func stop() {
        Task {
            await ttsService.stopPlayback()
            state = .ready
            isGenerating = false
            stopWaveformAnimation()
        }
    }

    private func startWaveformAnimation() {
        waveformTimer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) {
            [weak self] _ in
            Task { @MainActor in
                self?.waveformAmplitude = CGFloat.random(in: 0.3 ... 1.0)
            }
        }
    }

    private func stopWaveformAnimation() {
        waveformTimer?.invalidate()
        waveformTimer = nil
        waveformAmplitude = 0
    }

    func clearError() {
        if case .error = state {
            state = selectedVoice != nil ? .ready : .idle
        }
    }
}
