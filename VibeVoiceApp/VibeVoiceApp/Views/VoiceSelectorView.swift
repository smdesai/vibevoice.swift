import SwiftUI

struct VoiceSelectorView: View {
    @Binding var selectedVoice: Voice?
    @ObservedObject var downloadService: VoiceDownloadService
    let isEnabled: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Select Voice")
                .font(.headline)
                .foregroundColor(.primary)

            ScrollView(.horizontal, showsIndicators: false) {
                LazyHStack(spacing: 20) {
                    ForEach(VoiceLanguage.allCases) { language in
                        VoiceLanguageSection(
                            language: language,
                            voices: Voice.groupedByLanguage[language] ?? [],
                            selectedVoice: $selectedVoice,
                            downloadService: downloadService,
                            isEnabled: isEnabled
                        )
                    }
                }
                .padding(.horizontal)
            }
        }
    }
}

private struct VoiceLanguageSection: View {
    let language: VoiceLanguage
    let voices: [Voice]
    @Binding var selectedVoice: Voice?
    @ObservedObject var downloadService: VoiceDownloadService
    let isEnabled: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 4) {
                Text(language.flag)
                    .font(.title3)
                Text(language.displayName)
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .foregroundColor(.secondary)
            }

            HStack(spacing: 8) {
                ForEach(voices) { voice in
                    VoiceChip(
                        voice: voice,
                        isSelected: selectedVoice?.id == voice.id,
                        isDownloaded: downloadService.downloadedVoices.contains(voice.id),
                        downloadProgress: downloadService.downloadProgress[voice.id],
                        isEnabled: isEnabled
                    ) {
                        selectedVoice = voice
                    }
                }
            }
        }
    }
}

private struct VoiceChip: View {
    let voice: Voice
    let isSelected: Bool
    let isDownloaded: Bool
    let downloadProgress: Double?
    let isEnabled: Bool
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 6) {
                Image(systemName: voice.gender == .male ? "person.fill" : "person.fill")
                    .font(.caption)
                    .foregroundColor(voice.gender == .male ? .blue : .pink)

                Text(voice.displayName)
                    .font(.subheadline)
                    .fontWeight(isSelected ? .semibold : .regular)

                if let progress = downloadProgress {
                    ProgressView(value: progress)
                        .progressViewStyle(.circular)
                        .scaleEffect(0.6)
                        .frame(width: 16, height: 16)
                } else if !isDownloaded {
                    Image(systemName: "arrow.down.circle")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(
                RoundedRectangle(cornerRadius: 20)
                    .fill(isSelected ? Color.blue.opacity(0.15) : Color(.systemGray6))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 20)
                    .stroke(isSelected ? Color.blue : Color.clear, lineWidth: 2)
            )
        }
        .buttonStyle(.plain)
        .disabled(!isEnabled)
        .opacity(isEnabled ? 1.0 : 0.6)
    }
}

#Preview {
    VoiceSelectorView(
        selectedVoice: .constant(Voice.allVoices.first),
        downloadService: VoiceDownloadService.shared,
        isEnabled: true
    )
    .padding()
}
