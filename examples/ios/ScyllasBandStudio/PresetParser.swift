import Foundation
import ScyllasBandKit

enum PresetParserError: LocalizedError {
    case invalidTag(String)
    case unknownVoice(String)
    case unsupportedLanguage(String, String)
    case unknownEmotion(String)

    var errorDescription: String? {
        switch self {
        case .invalidTag(let tag): return "Invalid speaker tag [\(tag)]."
        case .unknownVoice(let voice): return "The preset refers to unknown voice “\(voice)”."
        case .unsupportedLanguage(let language, let voice):
            return "Language “\(language)” is not available for \(voice)."
        case .unknownEmotion(let emotion): return "Unknown preset emotion “\(emotion)”."
        }
    }
}

enum PresetParser {
    private static let pattern = try! NSRegularExpression(pattern: #"\[([-A-Za-z0-9_.,:=]+)\]"#)
    private static let languageTags: Set<String> = ["en", "en_us", "en_gb", "es", "it"]

    static func parse(
        _ source: String,
        defaults: SegmentSettings,
        bundleInfo: SBScyllasBandBundleInfo
    ) throws -> [ScriptSegment] {
        let nsSource = source as NSString
        let matches = pattern.matches(in: source, range: NSRange(location: 0, length: nsSource.length))
        guard !matches.isEmpty else {
            let text = source.trimmingCharacters(in: .whitespacesAndNewlines)
            return text.isEmpty ? [] : [ScriptSegment(text: text, settings: defaults)]
        }

        var output: [ScriptSegment] = []
        var cursor = 0
        var activeSettings = defaults
        for match in matches {
            let preceding = nsSource.substring(with: NSRange(location: cursor, length: match.range.location - cursor))
            append(preceding, settings: activeSettings, to: &output)

            let label = nsSource.substring(with: match.range(at: 1))
            activeSettings = try parseTag(label, current: activeSettings, bundleInfo: bundleInfo)
            cursor = NSMaxRange(match.range)
        }
        append(nsSource.substring(from: cursor), settings: activeSettings, to: &output)
        return output
    }

    private static func append(
        _ text: String,
        settings: SegmentSettings,
        to segments: inout [ScriptSegment]
    ) {
        let speakable = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !speakable.isEmpty else { return }
        segments.append(ScriptSegment(text: speakable, settings: settings))
    }

    private static func parseTag(
        _ label: String,
        current: SegmentSettings,
        bundleInfo: SBScyllasBandBundleInfo
    ) throws -> SegmentSettings {
        let clean = label.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        var next = current
        if clean.contains(":") {
            let parts = clean.split(separator: ":", maxSplits: 2, omittingEmptySubsequences: false)
            if !parts[0].isEmpty { next.voiceIdentifier = String(parts[0]) }
            if parts.count > 1, !parts[1].isEmpty { next.language = String(parts[1]) }
            if parts.count > 2, !parts[2].isEmpty {
                let terms = parts[2].split(separator: ",")
                guard terms.count == 1 else { throw PresetParserError.invalidTag(label) }
                let affect = terms[0].split(separator: "=", maxSplits: 1)
                guard affect.count == 2, let strength = Float(affect[1]), strength.isFinite,
                      (0...1).contains(strength) else {
                    throw PresetParserError.invalidTag(label)
                }
                next.emotion = String(affect[0])
                next.emotionStrength = strength
            }
        } else if languageTags.contains(clean) {
            next.language = clean
        } else if bundleInfo.affectAxes.contains(clean) {
            next.emotion = clean
            next.emotionStrength = 1
        } else if !clean.isEmpty {
            next.voiceIdentifier = clean
        }

        guard let voice = bundleInfo.voices.first(where: { $0.identifier == next.voiceIdentifier }) else {
            throw PresetParserError.unknownVoice(next.voiceIdentifier)
        }
        if next.language == "en" {
            next.language = voice.defaultLanguage
        }
        guard voice.languages.contains(next.language) else {
            throw PresetParserError.unsupportedLanguage(next.language, voice.identifier)
        }
        if let emotion = next.emotion, !bundleInfo.affectAxes.contains(emotion) {
            throw PresetParserError.unknownEmotion(emotion)
        }
        return next
    }
}

enum DefaultWalkthrough {
    static let taggedSource = """
    [scylla:en_us:sadness=0.6]Hey! Hey you, over there!

    [scylla:en_us:anger=0.6]Yo, I can see you reading the reddit post!

    [scylla:en_us:calm=0.6]Sorry, I'm getting ahead of myself. Hello guys, gals, this is a demo of a brand new voice model!

    [orpheus:en_gb:sarcasm=0.4]Really, a new voice model in this day and age? Why should anyone even care?

    [tuesday:en_gb:sarcasm=0.7]I think, sir... sorry to butt in... but it's because this demo has a bit more expression in it. And now it's running on Apple. Finally. Why was Android first? Apparently someone took alphabetical order far too seriously.

    [rex:es:joy=0.7]¡Por fin llegamos a Apple! Pero lo importante es esto: puedo hablar en español con alegría, carácter y sin mandar ni una palabra a la nube.

    [ink:it:sarcasm=0.6]Prima Android e poi Apple... un ordine davvero creativo. Ma sentite l'italiano: espressione, ritmo, tutto sul dispositivo. Possiamo perdonarglielo.

    [felix:en_us:calm=0.35]So hey, I'm glad you listened in. Scylla's Band already has a version 2 in the works with improved voices and new languages.

    [ariadne:en_us:questioning=0.8]I heard there were free cookies! Oh... sorry, I just wanted an excuse to say something. Pass the cupcake?

    [gwen:en_us:calm=0.8]Sorry, I think by cookies that means the author is taking suggestions while version 2 is still in the works. Thank you!
    """

    static let emotionCFG: [Float] = [3, 2.4, 3, 1.2, 2.5, 1.1, 1.2, 0.9, 1.15, 1.15]

    static func segments(
        defaults: SegmentSettings,
        bundleInfo: SBScyllasBandBundleInfo
    ) throws -> [ScriptSegment] {
        var segments = try PresetParser.parse(taggedSource, defaults: defaults, bundleInfo: bundleInfo)
        for index in segments.indices where index < emotionCFG.count {
            segments[index].settings.emotionCFG = emotionCFG[index]
        }
        return segments
    }
}
