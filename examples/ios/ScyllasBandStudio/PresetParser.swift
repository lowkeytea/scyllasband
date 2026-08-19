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
    /// Non-English lines, translated:
    /// - Felix (de, whispered): "Felix here. I'm not whispering because it's a secret. I whisper because it's what I do best."
    /// - Stone (fr): "My name is Stone. Drama isn't my thing. French is."
    /// - Scylla (vi): "See? Still me, just in Vietnamese."
    /// - Rex (es): "And Spanish is still here! Ten voices, a pile of languages, and not a single word leaves your phone."
    /// - Max (it): "And then there's Italian: every sentence a little drama. Please, no applause."
    static let taggedSource = """
    [gwen:en_us:whisper=0.8]Psst. Hey. You, with the headphones. Come closer... the others don't know you're here yet.

    [gwen:en_us:joy=0.7]Nothing? Fine, plan B! Big, warm, friendly voice! Everyone trusts a happy voice, right? ...Right?

    [gwen:en_us:anger=0.7]Oh, come on! I whispered, I sparkled, and you're still just poking at the screen!

    [ink:en_gb:sarcasm=0.6]Gwen. Deep breath. What is it you're actually trying to tell these nice people?

    [scylla:en_us:calm=0.6]What she's trying to say is: we got an upgrade. Version two, with new languages and new tricks. And since the whole thing is named after me, I'll do the honors.

    [tuesday:en_gb:sarcasm=0.5]Introductions, then, before she does all seven languages herself. And don't worry: everything you're hearing happens right on your iPhone. The cloud wasn't invited.

    [felix:de:whisper=0.7]Felix hier. Ich flüstere nicht, weil es geheim ist. Ich flüstere, weil ich es am besten kann.

    [stone:fr:calm=0.5]Je m'appelle Stone. Le drame, ce n'est pas mon genre. Le français, si.

    [scylla:vi:joy=0.5]Thấy chưa? Vẫn là tôi, chỉ là bằng tiếng Việt.

    [rex:es:joy=0.8]¡Y el español sigue aquí! Diez voces, un montón de idiomas, y ni una sola palabra sale de tu teléfono.

    [max:it:sarcasm=0.5]E poi c'è l'italiano: ogni frase un piccolo dramma. Prego, niente applausi.

    [ariadne:en_us:calm=0.6]Everyone is very impressive. But I have been here for two whole versions now, and still nobody has brought the little cookies.

    [orpheus:en_gb:sadness=0.5]And that's the demo. She waits for her cookies, I wait for a bigger part. Play it again. Perhaps we both get lucky.
    """

    static let emotionCFG: [Float] = [1.3, 2.4, 2.8, 1.2, 1.15, 2.0, 1.25, 1.1, 1.2, 1.15, 1.2, 1.15, 1.4]

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
