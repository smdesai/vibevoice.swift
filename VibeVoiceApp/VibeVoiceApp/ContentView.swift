import SwiftUI

struct ContentView: View {
    @StateObject private var modelDownloadViewModel: ModelDownloadViewModel
    @StateObject private var ttsViewModel: TTSViewModel

    private let ttsService: TTSService

    init() {
        let service = TTSService()
        self.ttsService = service
        _modelDownloadViewModel = StateObject(
            wrappedValue: ModelDownloadViewModel(ttsService: service))
        _ttsViewModel = StateObject(wrappedValue: TTSViewModel(ttsService: service))
    }

    var body: some View {
        Group {
            if modelDownloadViewModel.state == .completed {
                MainTTSView(viewModel: ttsViewModel)
                    .transition(.opacity)
            } else {
                ModelDownloadView(viewModel: modelDownloadViewModel)
                    .transition(.opacity)
            }
        }
        .animation(.easeInOut(duration: 0.3), value: modelDownloadViewModel.state == .completed)
        .onChange(of: modelDownloadViewModel.state) { oldValue, newValue in
            if newValue == .completed {
                ttsViewModel.onModelReady()
            }
        }
    }
}

#Preview {
    ContentView()
}
