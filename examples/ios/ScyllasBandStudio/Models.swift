import Foundation
import ScyllasBandKit

/// The four delivery axes, each 0–4 with 2 neutral.
enum DeliveryAxis: String, CaseIterable, Identifiable {
    case energy
    case tension
    case valence
    case assertiveness

    var id: String { rawValue }
    var title: String { rawValue.capitalized }

    var keyPath: WritableKeyPath<DeliverySettings, Float> {
        switch self {
        case .energy: return \.energy
        case .tension: return \.tension
        case .valence: return \.valence
        case .assertiveness: return \.assertiveness
        }
    }

    /// The range the runtime supports for the control, where the voices render most reliably; the runtime clamps any
    /// other request into it. 2 (neutral) lies inside each.
    var supportedRange: ClosedRange<Float> {
        switch self {
        case .energy: return 1.4...2.3
        case .tension: return 1.0...2.4
        case .valence: return 1.4...2.2
        case .assertiveness: return 1.8...2.4
        }
    }

    /// What lowering and raising the control does.
    var range: (low: String, high: String) {
        switch self {
        case .energy: return ("Darker, more pauses", "Brighter, projected")
        case .tension: return ("Looser, steadier", "Tighter, wider pitch")
        case .valence: return ("Flatter, slower", "Livelier, quicker")
        case .assertiveness: return ("Hesitant", "Fast, clipped")
        }
    }
}

struct DeliverySettings: Codable, Equatable {
    static let neutralValue: Float = 2
    /// What delivery tags may contain: the full 0–4 scale, clamped to each control's `supportedRange` when imported.
    static let modelRange: ClosedRange<Float> = 0...4
    static let neutral = DeliverySettings()

    var energy: Float = 2
    var tension: Float = 2
    var valence: Float = 2
    var assertiveness: Float = 2

    var isNeutral: Bool { self == .neutral }

    var native: SBScyllasBandDelivery {
        SBScyllasBandDelivery(
            energy: energy,
            tension: tension,
            valence: valence,
            assertiveness: assertiveness,
            whisper: false
        )
    }

    /// "neutral", or the controls that differ from neutral, e.g. "energy 2.3 · valence 2.2".
    var summary: String {
        let parts = DeliveryAxis.allCases.compactMap { axis -> String? in
            let value = self[keyPath: axis.keyPath]
            return value == Self.neutralValue ? nil : "\(axis.rawValue) \(Self.format(value))"
        }
        return parts.isEmpty ? "neutral" : parts.joined(separator: " · ")
    }

    /// Clamps each control to its `supportedRange` and rounds to the slider step (0.1).
    mutating func normalize() {
        for axis in DeliveryAxis.allCases {
            let value = self[keyPath: axis.keyPath]
            let range = axis.supportedRange
            self[keyPath: axis.keyPath] = value.isFinite
                ? (min(max(value, range.lowerBound), range.upperBound) * 10).rounded() / 10
                : Self.neutralValue
        }
    }

    static func format(_ value: Float) -> String {
        value.formatted(.number.precision(.fractionLength(0...1)))
    }
}

/// Familiar deliveries made from the four controls, the same combinations as the main README; each lies within the
/// supported ranges.
enum DeliveryPreset: String, CaseIterable, Identifiable {
    case neutral
    case calm
    case assertive
    case joyful
    case angry
    case sad

    var id: String { rawValue }
    var title: String { rawValue.capitalized }

    var delivery: DeliverySettings {
        switch self {
        case .neutral:
            return .neutral
        case .calm:
            return DeliverySettings(energy: 1.8, tension: 1, valence: 2.2, assertiveness: 2.1)
        case .assertive:
            return DeliverySettings(energy: 2.3, tension: 2.4, valence: 2.1, assertiveness: 2.4)
        case .joyful:
            return DeliverySettings(energy: 2.3, tension: 2, valence: 2.2, assertiveness: 2.2)
        case .angry:
            return DeliverySettings(energy: 2.3, tension: 2.4, valence: 1.4, assertiveness: 2.4)
        case .sad:
            return DeliverySettings(energy: 1.7, tension: 1.5, valence: 1.4, assertiveness: 2)
        }
    }
}

struct SegmentSettings: Codable, Equatable {
    var voiceIdentifier: String
    var language: String
    var delivery = DeliverySettings.neutral

    func request(text: String, seed: UInt64) -> SBScyllasBandRequest {
        let request = SBScyllasBandRequest(text: text, voiceIdentifier: voiceIdentifier)
        request.language = language
        request.delivery = delivery.native
        request.seed = NSNumber(value: seed)
        return request
    }

    mutating func validate(using info: SBScyllasBandBundleInfo) {
        let voice = info.voice(withIdentifier: voiceIdentifier)
            ?? info.voice(withIdentifier: info.defaultVoice)
            ?? info.voices[0]
        voiceIdentifier = voice.identifier
        if !voice.languages.contains(language) {
            language = voice.defaultLanguage
        }
        delivery.normalize()
    }
}

struct ScriptSegment: Identifiable, Codable, Equatable {
    var id = UUID()
    var text: String
    var settings: SegmentSettings
}

/// The repository's data/ documents, embedded by Scripts/prepare_assets.sh.
enum ExampleDocument: String, CaseIterable, Identifiable {
    case walkthrough = "walkthrough_demo"
    case narration = "test_document"
    case energetic = "emotional_text"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .walkthrough: return "Multilingual walkthrough (tagged dialogue)"
        case .narration: return "Long-form narration"
        case .energetic: return "Energetic prose"
        }
    }

    var url: URL? {
        Bundle.main.url(forResource: rawValue, withExtension: "txt", subdirectory: "scyllasband/examples")
    }
}

/// Where the graphs run. Automatic follows the bundle's recommendation for each graph.
enum ComputeUnitOption: Int, CaseIterable, Identifiable {
    case automatic
    case cpu
    case gpu
    case neuralEngine

    var id: Int { rawValue }

    /// False in the Simulator, whose Core ML runs on the CPU only (its GPU path fails and it has no Neural Engine).
    static var isSelectable: Bool {
        #if targetEnvironment(simulator)
        return false
        #else
        return true
        #endif
    }

    var title: String {
        switch self {
        case .automatic: return "Automatic"
        case .cpu: return "CPU"
        case .gpu: return "GPU"
        case .neuralEngine: return "Neural Engine"
        }
    }

    var native: SBScyllasBandComputeUnit {
        switch self {
        case .automatic: return .automatic
        case .cpu: return .CPU
        case .gpu: return .GPU
        case .neuralEngine: return .neuralEngine
        }
    }
}

func backendLabel(_ backend: String) -> String {
    switch backend {
    case "coreai": return "Core AI"
    case "coreml": return "Core ML"
    default: return backend
    }
}

extension SBScyllasBandVoice {
    var friendlyName: String { identifier.capitalized }
}

func languageLabel(_ language: String) -> String {
    switch language {
    case "en_us": return "English (US)"
    case "en_gb": return "English (UK)"
    default:
        let name = Locale.current.localizedString(forLanguageCode: language)
        return name.map { $0.prefix(1).uppercased() + $0.dropFirst() } ?? language
    }
}
