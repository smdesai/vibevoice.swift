import SwiftUI

struct GenerationStatsView: View {
    let stats: GenerationStats

    var body: some View {
        VStack(spacing: 12) {
            HStack {
                Text("Generation Stats")
                    .font(.subheadline.weight(.semibold))
                    .foregroundColor(.secondary)
                Spacer()
            }

            HStack(spacing: 16) {
                statItem(
                    label: "Audio",
                    value: stats.formattedAudioDuration,
                    icon: "waveform"
                )

                Divider()
                    .frame(height: 32)

                statItem(
                    label: "Gen Time",
                    value: stats.formattedGenerationTime,
                    icon: "cpu"
                )

                Divider()
                    .frame(height: 32)

                statItem(
                    label: "TTFA",
                    value: stats.formattedTimeToFirstChunk,
                    icon: "bolt"
                )

                Divider()
                    .frame(height: 32)

                statItem(
                    label: "RTF",
                    value: stats.formattedRTF,
                    icon: stats.isFasterThanRealtime ? "hare" : "tortoise",
                    valueColor: stats.isFasterThanRealtime ? .green : .orange
                )
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(.systemGray6))
        )
    }

    private func statItem(
        label: String,
        value: String,
        icon: String,
        valueColor: Color = .primary
    ) -> some View {
        VStack(spacing: 4) {
            Image(systemName: icon)
                .font(.caption)
                .foregroundColor(.secondary)

            Text(value)
                .font(.system(.subheadline, design: .monospaced).weight(.medium))
                .foregroundColor(valueColor)

            Text(label)
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
    }
}

#Preview {
    VStack(spacing: 20) {
        GenerationStatsView(
            stats: GenerationStats(
                audioDuration: 3.45,
                generationTime: 2.12,
                timeToFirstChunk: 245
            )
        )

        GenerationStatsView(
            stats: GenerationStats(
                audioDuration: 2.0,
                generationTime: 2.5,
                timeToFirstChunk: 180
            )
        )
    }
    .padding()
}
