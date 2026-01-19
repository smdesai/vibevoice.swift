import SwiftUI

struct WaveformIndicator: View {
    let amplitude: CGFloat
    let barCount: Int

    init(amplitude: CGFloat, barCount: Int = 5) {
        self.amplitude = amplitude
        self.barCount = barCount
    }

    var body: some View {
        HStack(spacing: 4) {
            ForEach(0 ..< barCount, id: \.self) { index in
                WaveformBar(
                    amplitude: amplitude,
                    delay: Double(index) * 0.1
                )
            }
        }
    }
}

private struct WaveformBar: View {
    let amplitude: CGFloat
    let delay: Double

    @State private var currentHeight: CGFloat = 0.2

    var body: some View {
        RoundedRectangle(cornerRadius: 2)
            .fill(
                LinearGradient(
                    colors: [.blue, .purple],
                    startPoint: .bottom,
                    endPoint: .top
                )
            )
            .frame(width: 4, height: max(8, 40 * currentHeight))
            .animation(
                .easeInOut(duration: 0.15)
                    .delay(delay),
                value: currentHeight
            )
            .onChange(of: amplitude) { _, newValue in
                let variation = CGFloat.random(in: -0.2 ... 0.2)
                currentHeight = max(0.2, min(1.0, newValue + variation))
            }
    }
}

struct AnimatedWaveform: View {
    @State private var isAnimating = false

    let isPlaying: Bool
    let barCount: Int

    init(isPlaying: Bool, barCount: Int = 7) {
        self.isPlaying = isPlaying
        self.barCount = barCount
    }

    var body: some View {
        HStack(spacing: 3) {
            ForEach(0 ..< barCount, id: \.self) { index in
                AnimatedBar(
                    isAnimating: isPlaying,
                    delay: Double(index) * 0.08
                )
            }
        }
        .frame(height: 30)
    }
}

private struct AnimatedBar: View {
    let isAnimating: Bool
    let delay: Double

    @State private var height: CGFloat = 0.3

    var body: some View {
        RoundedRectangle(cornerRadius: 2)
            .fill(
                LinearGradient(
                    colors: [.blue.opacity(0.8), .purple.opacity(0.8)],
                    startPoint: .bottom,
                    endPoint: .top
                )
            )
            .frame(width: 4, height: max(6, 30 * height))
            .onAppear {
                if isAnimating {
                    startAnimation()
                }
            }
            .onChange(of: isAnimating) { _, newValue in
                if newValue {
                    startAnimation()
                } else {
                    withAnimation(.easeOut(duration: 0.3)) {
                        height = 0.3
                    }
                }
            }
    }

    private func startAnimation() {
        guard isAnimating else { return }

        withAnimation(
            .easeInOut(duration: Double.random(in: 0.2 ... 0.4))
                .delay(delay)
                .repeatForever(autoreverses: true)
        ) {
            height = CGFloat.random(in: 0.4 ... 1.0)
        }
    }
}

#Preview {
    VStack(spacing: 40) {
        WaveformIndicator(amplitude: 0.7)

        AnimatedWaveform(isPlaying: true)

        AnimatedWaveform(isPlaying: false)
    }
    .padding()
}
