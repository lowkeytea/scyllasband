import Foundation
import ScyllasBandKit

/// The three delivery controls. 0 is the voice's own delivery; the released model supports -2 to 2 for each.
enum DeliveryControl: String, CaseIterable, Identifiable {
    /// Calm (negative) to intense (positive), in the voice's own standard deviations. The emotion comes from the text.
    case intensity
    /// Slower (negative) to faster (positive), words and pauses together, in the voice's own standard deviations.
    case pace
    /// A shift of the voice's pitch in semitones.
    case pitch

    /// The released model's range for every control; ScyllasBandKit clamps any value into it.
    static let modelRange: ClosedRange<Float> = -2...2

    var id: String { rawValue }
    var title: String { rawValue.capitalized }

    var keyPath: WritableKeyPath<DeliverySettings, Float?> {
        switch self {
        case .intensity: return \.intensity
        case .pace: return \.pace
        case .pitch: return \.pitch
        }
    }

    var native: SBScyllasBandDeliveryControl {
        switch self {
        case .intensity: return .intensity
        case .pace: return .pace
        case .pitch: return .pitch
        }
    }

    /// What lowering and raising the control does.
    var ends: (low: String, high: String) {
        switch self {
        case .intensity: return ("Calm", "Intense")
        case .pace: return ("Slower", "Faster")
        case .pitch: return ("Lower", "Higher")
        }
    }

    var unit: String {
        switch self {
        case .intensity, .pace: return "standard deviations"
        case .pitch: return "semitones"
        }
    }

    /// The range the runtime clamps the control into, as the bundle reports it (-2 to 2 in the released model).
    func range(in info: SBScyllasBandBundleInfo?) -> ClosedRange<Float> {
        guard let range = info?.deliveryRanges[native] else { return Self.modelRange }
        let low = max(range.minimum, Self.modelRange.lowerBound), high = min(range.maximum, Self.modelRange.upperBound)
        return low <= high ? low...high : Self.modelRange
    }

    func value(in delivery: SBScyllasBandDelivery) -> Float? {
        switch self {
        case .intensity: return delivery.intensity?.floatValue
        case .pace: return delivery.pace?.floatValue
        case .pitch: return delivery.pitch?.floatValue
        }
    }

    /// What the voice uses when the control is left unset (0 for every voice in the released model).
    func defaultValue(for voice: SBScyllasBandVoice?) -> Float {
        voice.flatMap { value(in: $0.deliveryDefaults) } ?? 0
    }
}

/// A segment's delivery. Each control is optional: nil takes the voice's default.
struct DeliverySettings: Codable, Equatable {
    /// Every control unset: the voice's defaults.
    static let voiceDefault = DeliverySettings()

    var intensity: Float?
    var pace: Float?
    var pitch: Float?

    var native: SBScyllasBandDelivery {
        SBScyllasBandDelivery(
            intensity: intensity.map { NSNumber(value: $0) },
            pace: pace.map { NSNumber(value: $0) },
            pitch: pitch.map { NSNumber(value: $0) }
        )
    }

    /// "voice default", or the controls set away from 0, e.g. "intensity -1 · pace -0.5".
    var summary: String {
        let parts = DeliveryControl.allCases.compactMap { control -> String? in
            guard let value = self[keyPath: control.keyPath], value != 0 else { return nil }
            return "\(control.rawValue) \(Self.format(value))"
        }
        return parts.isEmpty ? "voice default" : parts.joined(separator: " · ")
    }

    /// Every control set, unset ones from the voice's defaults.
    func resolved(for voice: SBScyllasBandVoice?) -> DeliverySettings {
        var resolved = self
        for control in DeliveryControl.allCases where resolved[keyPath: control.keyPath] == nil {
            resolved[keyPath: control.keyPath] = control.defaultValue(for: voice)
        }
        return resolved
    }

    /// Clamps each set control into the bundle's range, as the runtime would; values that aren't finite become unset.
    mutating func normalize(using info: SBScyllasBandBundleInfo?) {
        for control in DeliveryControl.allCases {
            guard let value = self[keyPath: control.keyPath] else { continue }
            let range = control.range(in: info)
            self[keyPath: control.keyPath] = value.isFinite ? min(max(value, range.lowerBound), range.upperBound) : nil
        }
    }

    static func format(_ value: Float) -> String {
        value.formatted(.number.precision(.fractionLength(0...2)))
    }
}

/// The segment settings' quick reset, as in the Android sample: every control back to the voice's own delivery.
enum DeliveryPreset: String, CaseIterable, Identifiable {
    case voiceDefault

    var id: String { rawValue }

    var title: String {
        switch self {
        case .voiceDefault: return "Voice default"
        }
    }

    var systemImage: String {
        switch self {
        case .voiceDefault: return "arrow.counterclockwise"
        }
    }

    var delivery: DeliverySettings {
        switch self {
        case .voiceDefault: return .voiceDefault
        }
    }
}

struct SegmentSettings: Codable, Equatable {
    var voiceIdentifier: String
    var language: String
    var delivery = DeliverySettings.voiceDefault

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
        delivery.normalize(using: info)
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
