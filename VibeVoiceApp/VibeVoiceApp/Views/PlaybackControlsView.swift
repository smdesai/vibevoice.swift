import SwiftUI

struct PlaybackControlsView: View {
    let state: TTSViewModel.State
    let waveformAmplitude: CGFloat
    let onGenerate: () -> Void
    let onStop: () -> Void

    var body: some View {
        VStack(spacing: 20) {
            if case .playing = state {
                AnimatedWaveform(isPlaying: true, barCount: 9)
                    .frame(height: 40)
            } else if case .generating = state {
                HStack(spacing: 8) {
                    ProgressView()
                        .scaleEffect(0.8)
                    Text("Generating...")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                .frame(height: 40)
            } else {
                Color.clear
                    .frame(height: 40)
            }

            mainButton
        }
    }

    @ViewBuilder
    private var mainButton: some View {
        switch state {
        case .idle, .ready:
            generateButton

        case .loadingVoice:
            loadingVoiceButton

        case .generating, .playing:
            stopButton

        case .error:
            retryButton
        }
    }

    private var generateButton: some View {
        Button(action: onGenerate) {
            Label("Generate Speech", systemImage: "play.fill")
                .font(.headline)
                .foregroundColor(.white)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 16)
                .background(
                    LinearGradient(
                        colors: [.blue, .purple],
                        startPoint: .leading,
                        endPoint: .trailing
                    )
                )
                .cornerRadius(16)
        }
        .buttonStyle(.plain)
    }

    private var loadingVoiceButton: some View {
        HStack {
            ProgressView()
                .scaleEffect(0.8)
            Text("Loading Voice...")
                .font(.headline)
        }
        .foregroundColor(.white)
        .frame(maxWidth: .infinity)
        .padding(.vertical, 16)
        .background(Color.gray)
        .cornerRadius(16)
    }

    private var stopButton: some View {
        Button(action: onStop) {
            Label("Stop", systemImage: "stop.fill")
                .font(.headline)
                .foregroundColor(.white)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 16)
                .background(Color.red)
                .cornerRadius(16)
        }
        .buttonStyle(.plain)
    }

    private var retryButton: some View {
        Button(action: onGenerate) {
            Label("Try Again", systemImage: "arrow.clockwise")
                .font(.headline)
                .foregroundColor(.white)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 16)
                .background(Color.orange)
                .cornerRadius(16)
        }
        .buttonStyle(.plain)
    }
}

#Preview {
    VStack(spacing: 40) {
        PlaybackControlsView(
            state: .ready,
            waveformAmplitude: 0,
            onGenerate: {},
            onStop: {}
        )

        PlaybackControlsView(
            state: .playing,
            waveformAmplitude: 0.7,
            onGenerate: {},
            onStop: {}
        )

        PlaybackControlsView(
            state: .error("Test error"),
            waveformAmplitude: 0,
            onGenerate: {},
            onStop: {}
        )
    }
    .padding()
}
