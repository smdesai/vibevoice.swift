import Foundation

struct Voice: Identifiable, Hashable {
    let id: String
    let displayName: String
    let language: VoiceLanguage
    let gender: Gender
    let cacheFileName: String

    enum Gender: String {
        case male = "man"
        case female = "woman"

        var icon: String {
            switch self {
            case .male: return "person.fill"
            case .female: return "person.fill"
            }
        }
    }

    var huggingFaceURL: URL {
        URL(
            string:
                "https://huggingface.co/smdesai/VibeVoice-Realtime-0.5B-8bit/resolve/main/voices/\(cacheFileName)"
        )!
    }

    static let allVoices: [Voice] = [
        // English voices
        Voice(
            id: "en-Carter", displayName: "Carter", language: .english, gender: .male,
            cacheFileName: "en-Carter_man.safetensors"),
        Voice(
            id: "en-Davis", displayName: "Davis", language: .english, gender: .male,
            cacheFileName: "en-Davis_man.safetensors"),
        Voice(
            id: "en-Emma", displayName: "Emma", language: .english, gender: .female,
            cacheFileName: "en-Emma_woman.safetensors"),
        Voice(
            id: "en-Frank", displayName: "Frank", language: .english, gender: .male,
            cacheFileName: "en-Frank_man.safetensors"),
        Voice(
            id: "en-Grace", displayName: "Grace", language: .english, gender: .female,
            cacheFileName: "en-Grace_woman.safetensors"),
        Voice(
            id: "en-Mike", displayName: "Mike", language: .english, gender: .male,
            cacheFileName: "en-Mike_man.safetensors"),

        // Indian English
        Voice(
            id: "in-Samuel", displayName: "Samuel", language: .indianEnglish, gender: .male,
            cacheFileName: "in-Samuel_man.safetensors"),

        // German voices
        Voice(
            id: "de-Hans", displayName: "Hans", language: .german, gender: .male,
            cacheFileName: "de-Spk0_man.safetensors"),
        Voice(
            id: "de-Greta", displayName: "Greta", language: .german, gender: .female,
            cacheFileName: "de-Spk1_woman.safetensors"),

        // French voices
        Voice(
            id: "fr-Pierre", displayName: "Pierre", language: .french, gender: .male,
            cacheFileName: "fr-Spk0_man.safetensors"),
        Voice(
            id: "fr-Marie", displayName: "Marie", language: .french, gender: .female,
            cacheFileName: "fr-Spk1_woman.safetensors"),

        // Italian voices
        Voice(
            id: "it-Sofia", displayName: "Sofia", language: .italian, gender: .female,
            cacheFileName: "it-Spk0_woman.safetensors"),
        Voice(
            id: "it-Marco", displayName: "Marco", language: .italian, gender: .male,
            cacheFileName: "it-Spk1_man.safetensors"),

        // Japanese voices
        Voice(
            id: "jp-Takeshi", displayName: "Takeshi", language: .japanese, gender: .male,
            cacheFileName: "jp-Spk0_man.safetensors"),
        Voice(
            id: "jp-Yuki", displayName: "Yuki", language: .japanese, gender: .female,
            cacheFileName: "jp-Spk1_woman.safetensors"),

        // Korean voices
        Voice(
            id: "kr-Jisoo", displayName: "Jisoo", language: .korean, gender: .female,
            cacheFileName: "kr-Spk0_woman.safetensors"),
        Voice(
            id: "kr-Minho", displayName: "Minho", language: .korean, gender: .male,
            cacheFileName: "kr-Spk1_man.safetensors"),

        // Dutch voices
        Voice(
            id: "nl-Willem", displayName: "Willem", language: .dutch, gender: .male,
            cacheFileName: "nl-Spk0_man.safetensors"),
        Voice(
            id: "nl-Anna", displayName: "Anna", language: .dutch, gender: .female,
            cacheFileName: "nl-Spk1_woman.safetensors"),

        // Polish voices
        Voice(
            id: "pl-Piotr", displayName: "Piotr", language: .polish, gender: .male,
            cacheFileName: "pl-Spk0_man.safetensors"),
        Voice(
            id: "pl-Kasia", displayName: "Kasia", language: .polish, gender: .female,
            cacheFileName: "pl-Spk1_woman.safetensors"),
    ]

    static var groupedByLanguage: [VoiceLanguage: [Voice]] {
        Dictionary(grouping: allVoices, by: { $0.language })
    }

    static func voice(for id: String) -> Voice? {
        allVoices.first { $0.id == id }
    }
}
