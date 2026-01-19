import SwiftUI

struct TextInputView: View {
    @Binding var text: String
    let isEnabled: Bool
    let maxCharacters: Int

    init(text: Binding<String>, isEnabled: Bool = true, maxCharacters: Int = 500) {
        self._text = text
        self.isEnabled = isEnabled
        self.maxCharacters = maxCharacters
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ZStack(alignment: .topLeading) {
                if text.isEmpty {
                    Text("Enter text to speak...")
                        .foregroundColor(.secondary)
                        .padding(.horizontal, 4)
                        .padding(.vertical, 8)
                }

                TextEditor(text: $text)
                    .scrollContentBackground(.hidden)
                    .background(Color.clear)
                    .frame(minHeight: 120, maxHeight: 200)
                    .disabled(!isEnabled)
            }
            .padding(12)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color(.systemGray6))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(Color(.systemGray4), lineWidth: 1)
            )

            HStack {
                Text("\(text.count)/\(maxCharacters)")
                    .font(.caption)
                    .foregroundColor(text.count > maxCharacters ? .red : .secondary)

                Spacer()

                if !text.isEmpty {
                    Button(action: { text = "" }) {
                        Label("Clear", systemImage: "xmark.circle.fill")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .buttonStyle(.plain)
                }
            }
        }
        .onChange(of: text) { _, newValue in
            if newValue.count > maxCharacters {
                text = String(newValue.prefix(maxCharacters))
            }
        }
    }
}

#Preview {
    TextInputView(text: .constant("Hello, this is a test of the text input view."))
        .padding()
}
