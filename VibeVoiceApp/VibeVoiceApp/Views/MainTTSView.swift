import SwiftUI

struct MainTTSView: View {
    @StateObject var viewModel: TTSViewModel

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 24) {
                    VoiceSelectorView(
                        selectedVoice: $viewModel.selectedVoice,
                        downloadService: viewModel.voiceDownloadService,
                        isEnabled: !viewModel.isGenerating
                    )

                    TextInputView(
                        text: $viewModel.text,
                        isEnabled: !viewModel.isGenerating
                    )

                    PlaybackControlsView(
                        state: viewModel.state,
                        waveformAmplitude: viewModel.waveformAmplitude,
                        onGenerate: viewModel.generate,
                        onStop: viewModel.stop
                    )

                    if let stats = viewModel.generationStats {
                        GenerationStatsView(stats: stats)
                            .transition(.opacity.combined(with: .move(edge: .top)))
                    }

                    if case .error(let message) = viewModel.state {
                        errorBanner(message: message)
                    }
                }
                .padding()
            }
            .scrollDismissesKeyboard(.interactively)
            .navigationTitle("VibeVoice")
            .navigationBarTitleDisplayMode(.large)
        }
        .onTapGesture {
            UIApplication.shared.sendAction(
                #selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
        }
    }

    private func errorBanner(message: String) -> some View {
        HStack {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundColor(.orange)

            Text(message)
                .font(.subheadline)
                .foregroundColor(.primary)

            Spacer()

            Button(action: viewModel.clearError) {
                Image(systemName: "xmark.circle.fill")
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.orange.opacity(0.1))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.orange.opacity(0.3), lineWidth: 1)
        )
    }
}

#Preview {
    let ttsService = TTSService()
    MainTTSView(viewModel: TTSViewModel(ttsService: ttsService))
}
