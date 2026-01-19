import SwiftUI

struct ModelDownloadView: View {
    @ObservedObject var viewModel: ModelDownloadViewModel

    var body: some View {
        VStack(spacing: 32) {
            Spacer()

            appIcon

            VStack(spacing: 16) {
                Text("VibeVoice")
                    .font(.largeTitle)
                    .fontWeight(.bold)

                Text("Real-time Text to Speech")
                    .font(.title3)
                    .foregroundColor(.secondary)
            }

            progressSection

            Spacer()

            footerText
        }
        .padding(32)
        .task {
            await viewModel.checkAndDownloadModel()
        }
    }

    private var appIcon: some View {
        ZStack {
            Circle()
                .fill(
                    LinearGradient(
                        colors: [.blue, .purple],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .frame(width: 120, height: 120)

            Image(systemName: "waveform")
                .font(.system(size: 50, weight: .medium))
                .foregroundColor(.white)
        }
    }

    @ViewBuilder
    private var progressSection: some View {
        switch viewModel.state {
        case .checking:
            VStack(spacing: 16) {
                ProgressView()
                    .scaleEffect(1.2)
                Text("Checking model...")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }

        case .notDownloaded:
            VStack(spacing: 16) {
                Image(systemName: "arrow.down.circle")
                    .font(.system(size: 40))
                    .foregroundColor(.blue)

                Text("Model needs to be downloaded")
                    .font(.subheadline)
                    .foregroundColor(.secondary)

                Button("Download Model") {
                    viewModel.retry()
                }
                .buttonStyle(.borderedProminent)
            }

        case .downloading(let progress):
            VStack(spacing: 20) {
                ZStack {
                    Circle()
                        .stroke(Color(.systemGray5), lineWidth: 8)
                        .frame(width: 100, height: 100)

                    Circle()
                        .trim(from: 0, to: progress)
                        .stroke(
                            LinearGradient(
                                colors: [.blue, .purple],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            ),
                            style: StrokeStyle(lineWidth: 8, lineCap: .round)
                        )
                        .frame(width: 100, height: 100)
                        .rotationEffect(.degrees(-90))
                        .animation(.easeInOut(duration: 0.3), value: progress)

                    Image(systemName: "arrow.down.circle")
                        .font(.system(size: 30))
                        .foregroundColor(.blue)
                }

                Text(viewModel.statusMessage)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
            }

        case .loading(let progress):
            VStack(spacing: 20) {
                ZStack {
                    Circle()
                        .stroke(Color(.systemGray5), lineWidth: 8)
                        .frame(width: 100, height: 100)

                    Circle()
                        .trim(from: 0, to: progress)
                        .stroke(
                            LinearGradient(
                                colors: [.green, .teal],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            ),
                            style: StrokeStyle(lineWidth: 8, lineCap: .round)
                        )
                        .frame(width: 100, height: 100)
                        .rotationEffect(.degrees(-90))
                        .animation(.easeInOut(duration: 0.3), value: progress)

                    Image(systemName: "memorychip")
                        .font(.system(size: 30))
                        .foregroundColor(.green)
                }

                Text(viewModel.statusMessage)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
            }

        case .completed:
            VStack(spacing: 16) {
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 50))
                    .foregroundColor(.green)

                Text("Model Ready")
                    .font(.headline)
            }

        case .error(let message):
            VStack(spacing: 16) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 40))
                    .foregroundColor(.orange)

                Text("Download Failed")
                    .font(.headline)

                Text(message)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)

                Button("Retry") {
                    viewModel.retry()
                }
                .buttonStyle(.borderedProminent)
            }
        }
    }

    private var footerText: some View {
        VStack(spacing: 4) {
            Text("Powered by MLX Swift")
                .font(.caption)
                .foregroundColor(.secondary)

            Text("Model: VibeVoice-Realtime-0.5B")
                .font(.caption2)
                .foregroundColor(.secondary.opacity(0.7))
        }
    }
}

#Preview {
    let ttsService = TTSService()
    ModelDownloadView(viewModel: ModelDownloadViewModel(ttsService: ttsService))
}
