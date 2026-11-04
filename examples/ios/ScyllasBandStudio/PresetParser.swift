import Foundation
import ScyllasBandKit

enum PresetParserError: LocalizedError {
    case invalidTag(String, String)
    case unknownVoice(String)
    case unsupportedLanguage(String, String)

    var errorDescription: String? {
        switch self {
        case .invalidTag(let tag, let reason): return "Invalid speaker tag [\(tag)]: \(reason)."
        case .unknownVoice(let voice): return "The document refers to unknown voice “\(voice)”."
        case .unsupportedLanguage(let language, let voice):
            return "Language “\(language)” is not available for \(voice)."
        }
    }
}

/// Converts `[voice:language:delivery]` tagged text (data/walkthrough_demo.txt) into speaker segments.
///
/// - `[ariadne:en_us:intensity=1,pace=-0.5]` sets voice, language and delivery. An empty voice or language part
///   keeps the active one (a language the voice lacks becomes its default); a delivery part replaces the active
///   delivery, and controls it leaves out take the voice's own.
/// - A language tag such as `[es]` keeps the active voice and delivery; `en` selects the voice's own English.
/// - Any other single word, such as `[ink]`, switches voice and uses that voice's default language.
/// - Every line starts again from the defaults. Consecutive text with the same settings forms one segment.
enum PresetParser {
    private static let pattern = try! NSRegularExpression(pattern: #"\[([-A-Za-z0-9_.,:=]+)\]"#)

    static func parse(
        _ source: String,
        defaults: SegmentSettings,
        bundleInfo: SBScyllasBandBundleInfo
    ) throws -> [ScriptSegment] {
        let languages = Set(bundleInfo.voices.flatMap(\.languages)).union(["en"])
        var segments: [ScriptSegment] = []
        var separator = ""
        for line in source.components(separatedBy: .newlines) {
            if line.trimmingCharacters(in: .whitespaces).isEmpty {
                if !segments.isEmpty { separator = "\n\n" }
                continue
            }
            if separator.isEmpty { separator = "\n" }

            let nsLine = line as NSString
            var active = defaults
            var cursor = 0
            var spans: [(text: String, settings: SegmentSettings)] = []
            for match in pattern.matches(in: line, range: NSRange(location: 0, length: nsLine.length)) {
                spans.append((nsLine.substring(with: NSRange(location: cursor, length: match.range.location - cursor)), active))
                active = try parseTag(
                    nsLine.substring(with: match.range(at: 1)),
                    current: active,
                    languages: languages,
                    bundleInfo: bundleInfo
                )
                cursor = NSMaxRange(match.range)
            }
            spans.append((nsLine.substring(from: cursor), active))

            for span in spans {
                let text = span.text.trimmingCharacters(in: .whitespaces)
                guard !text.isEmpty else { continue }
                if let last = segments.indices.last, segments[last].settings == span.settings {
                    segments[last].text += separator + text
                } else {
                    segments.append(ScriptSegment(text: text, settings: span.settings))
                }
                separator = " "
            }
            separator = ""
        }
        return segments
    }

    /// The v2 delivery axes, which this model rejects.
    private static let v2Axes: Set<String> = ["energy", "tension", "valence", "assertiveness", "whisper", "emotion"]

    /// `intensity=1,pace=-0.5,pitch=0`: omitted controls take the voice's own; "default" (or "neutral") alone, or as a
    /// value, leaves controls at the voice's default. Values outside the bundle's range (-2 to 2) are clamped, as the
    /// runtime would. The v2 axes (energy, tension, valence, assertiveness, whisper) are rejected.
    static func parseDelivery(_ specification: String, tag: String, bundleInfo: SBScyllasBandBundleInfo?) throws -> DeliverySettings {
        var delivery = DeliverySettings.voiceDefault
        let clean = specification.trimmingCharacters(in: .whitespaces).lowercased()
        guard !clean.isEmpty, clean != "default", clean != "neutral" else { return delivery }
        var seen = Set<String>()
        for term in clean.split(separator: ",", omittingEmptySubsequences: false) {
            let pair = term.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
                .map { $0.trimmingCharacters(in: .whitespaces) }
            guard pair.count == 2, !pair[0].isEmpty else {
                throw PresetParserError.invalidTag(tag, "invalid delivery term “\(term)”")
            }
            let (name, raw) = (pair[0], pair[1])
            guard seen.insert(name).inserted else {
                throw PresetParserError.invalidTag(tag, "delivery control “\(name)” is repeated")
            }
            guard !v2Axes.contains(name) else {
                throw PresetParserError.invalidTag(tag, "this model takes intensity, pace and pitch (“\(name)” belonged to the v2 models)")
            }
            guard let control = DeliveryControl(rawValue: name) else {
                throw PresetParserError.invalidTag(tag, "unknown delivery control “\(name)”; use intensity, pace or pitch")
            }
            if raw == "default" { continue }
            guard let value = Float(raw), value.isFinite else {
                throw PresetParserError.invalidTag(tag, "delivery “\(name)” must be a number")
            }
            let range = control.range(in: bundleInfo)
            delivery[keyPath: control.keyPath] = min(max(value, range.lowerBound), range.upperBound)
        }
        return delivery
    }

    private static func parseTag(
        _ label: String,
        current: SegmentSettings,
        languages: Set<String>,
        bundleInfo: SBScyllasBandBundleInfo
    ) throws -> SegmentSettings {
        let clean = label.trimmingCharacters(in: .whitespaces).lowercased()
        var next = current
        var explicitLanguage: String?
        if clean.contains(":") {
            let parts = clean.split(separator: ":", maxSplits: 2, omittingEmptySubsequences: false).map(String.init)
            if !parts[0].isEmpty { next.voiceIdentifier = parts[0] }
            if parts.count > 1, !parts[1].isEmpty { explicitLanguage = parts[1] }
            if parts.count > 2, !parts[2].isEmpty {
                next.delivery = try parseDelivery(parts[2], tag: label, bundleInfo: bundleInfo)
            }
        } else if languages.contains(clean) {
            explicitLanguage = clean
        } else if !clean.isEmpty {
            next.voiceIdentifier = clean
            next.language = ""   // a voice tag alone selects that voice's default language
        }

        guard let voice = bundleInfo.voice(withIdentifier: next.voiceIdentifier) else {
            throw PresetParserError.unknownVoice(next.voiceIdentifier)
        }
        if let language = explicitLanguage, language != "en" {
            guard voice.languages.contains(language) else {
                throw PresetParserError.unsupportedLanguage(language, voice.friendlyName)
            }
            next.language = language
        } else if explicitLanguage == "en" || !voice.languages.contains(next.language) {
            next.language = voice.defaultLanguage
        }
        return next
    }
}
