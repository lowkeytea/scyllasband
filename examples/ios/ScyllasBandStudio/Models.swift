import Foundation
import ScyllasBandKit

struct SegmentSettings: Codable, Equatable {
    var voiceIdentifier: String
    var language: String
    var emotion: String?
    var emotionStrength: Float
    var emotionCFG: Float

    init(
        voiceIdentifier: String,
        language: String,
        emotion: String? = nil,
        emotionStrength: Float = 0,
        emotionCFG: Float = 1
    ) {
        self.voiceIdentifier = voiceIdentifier
        self.language = language
        self.emotion = emotion
        self.emotionStrength = emotionStrength
        self.emotionCFG = emotionCFG
    }

    func nativeSettings() -> SBScyllasBandSegmentSettings {
        SBScyllasBandSegmentSettings(
            voiceIdentifier: voiceIdentifier,
            language: language,
            emotion: emotion,
            emotionStrength: emotionStrength,
            emotionCFG: emotionCFG
        )
    }

    mutating func validate(using info: SBScyllasBandBundleInfo) {
        let voice = info.voices.first { $0.identifier == voiceIdentifier } ?? info.voices[0]
        voiceIdentifier = voice.identifier
        if !voice.languages.contains(language) {
            language = voice.defaultLanguage
        }
        if let emotion, !info.affectAxes.contains(emotion) {
            self.emotion = nil
        }
        emotionStrength = min(max(emotionStrength, 0), 1)
        if !emotionCFG.isFinite || emotionCFG < 0 {
            emotionCFG = 1
        }
    }
}

struct ScriptSegment: Identifiable, Codable, Equatable {
    var id = UUID()
    var text: String
    var settings: SegmentSettings
}

enum ExamplePreset: String, CaseIterable, Identifiable {
    case walkthrough
    case emotional
    case groupSpeak
    case longForm

    var id: String { rawValue }

    var title: String {
        switch self {
        case .walkthrough: return "Studio walkthrough"
        case .emotional: return "Emotional monologue"
        case .groupSpeak: return "Multilingual group conversation"
        case .longForm: return "Long-form squirrel story"
        }
    }

    var resourceName: String? {
        switch self {
        case .walkthrough: return nil
        case .emotional: return "emotional_text"
        case .groupSpeak: return "groupSpeak"
        case .longForm: return "test_document"
        }
    }
}

extension SBScyllasBandVoice {
    var friendlyName: String {
        displayName.isEmpty ? identifier.capitalized : displayName
    }
}

func languageLabel(_ language: String) -> String {
    switch language {
    case "en_us": return "English (US)"
    case "en_gb": return "English (UK)"
    case "es": return "Spanish"
    case "it": return "Italian"
    default: return language
    }
}
