import Foundation

enum VoiceLanguage: String, CaseIterable, Identifiable {
    case english = "en"
    case indianEnglish = "in"
    case german = "de"
    case french = "fr"
    case italian = "it"
    case japanese = "jp"
    case korean = "kr"
    case dutch = "nl"
    case polish = "pl"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .english: return "English"
        case .indianEnglish: return "Indian English"
        case .german: return "German"
        case .french: return "French"
        case .italian: return "Italian"
        case .japanese: return "Japanese"
        case .korean: return "Korean"
        case .dutch: return "Dutch"
        case .polish: return "Polish"
        }
    }

    var flag: String {
        switch self {
        case .english: return "🇺🇸"
        case .indianEnglish: return "🇮🇳"
        case .german: return "🇩🇪"
        case .french: return "🇫🇷"
        case .italian: return "🇮🇹"
        case .japanese: return "🇯🇵"
        case .korean: return "🇰🇷"
        case .dutch: return "🇳🇱"
        case .polish: return "🇵🇱"
        }
    }
}
