import Foundation

/// An example score embedded with the app (scyllasband/melodies, the singing release's examples).
struct ExampleScore: Identifiable, Hashable {
    let url: URL
    var id: String { url.lastPathComponent }
    private var stem: String { url.deletingPathExtension().lastPathComponent }

    /// song_pop_gwen -> "Pop song · Gwen"; phrase_ballad_03 -> "Ballad phrase 4".
    var title: String {
        let parts = stem.split(separator: "_").map(String.init)
        if parts.count == 3, parts[0] == "song" { return "\(parts[1].capitalized) song · \(parts[2].capitalized)" }
        if parts.count == 3, parts[0] == "phrase", let number = Int(parts[2]) { return "\(parts[1].capitalized) phrase \(number + 1)" }
        return stem.replacingOccurrences(of: "_", with: " ").capitalized
    }

    /// The voice a whole-song example was written for (song_pop_gwen -> gwen).
    var voice: String? {
        let parts = stem.split(separator: "_")
        return parts.count == 3 && parts[0] == "song" ? String(parts[2]) : nil
    }

    var isSong: Bool { stem.hasPrefix("song_") }

    static func all() -> [ExampleScore] {
        let urls = Bundle.main.urls(forResourcesWithExtension: "json", subdirectory: "scyllasband/melodies") ?? []
        return urls.map(ExampleScore.init).sorted { ($0.isSong ? 0 : 1, $0.stem) < ($1.isSong ? 0 : 1, $1.stem) }
    }
}

/// Songs saved in the app's Documents/Songs (one JSON file per song) and the files a take is shared as.
enum SongStore {
    private static var songsDirectory: URL {
        let directory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("Songs", isDirectory: true)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private static var exportDirectory: URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("SungExports", isDirectory: true)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    static func saved() -> [SongDocument] {
        let urls = (try? FileManager.default.contentsOfDirectory(at: songsDirectory, includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
        let dated = urls.filter { $0.pathExtension == "json" }.compactMap { url -> (Date, SongDocument)? in
            guard let data = try? Data(contentsOf: url), let song = try? JSONDecoder().decode(SongDocument.self, from: data) else { return nil }
            let date = (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            return (date, song)
        }
        return dated.sorted { $0.0 > $1.0 }.map(\.1)
    }

    static func save(_ song: SongDocument) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(song).write(to: songsDirectory.appendingPathComponent("\(song.id.uuidString).json"), options: .atomic)
    }

    static func delete(_ song: SongDocument) {
        try? FileManager.default.removeItem(at: songsDirectory.appendingPathComponent("\(song.id.uuidString).json"))
    }

    /// The song's score as a file to share (the runtime's format, which `python -m scyllasband sing --score` also reads).
    static func exportScore(_ song: SongDocument) throws -> URL {
        let url = exportDirectory.appendingPathComponent("\(fileName(song.title)).json")
        try Data(song.scoreJSON().utf8).write(to: url, options: .atomic)
        return url
    }

    /// Mono Float32 samples as a 16-bit PCM WAV file to share.
    static func exportTake(_ samples: Data, sampleRate: Int, title: String) throws -> URL {
        let url = exportDirectory.appendingPathComponent("\(fileName(title)).wav")
        try wav(samples, sampleRate: sampleRate).write(to: url, options: .atomic)
        return url
    }

    static func wav(_ samples: Data, sampleRate: Int) -> Data {
        let count = samples.count / MemoryLayout<Float>.size
        var out = Data(capacity: 44 + count * 2)
        func append<T: FixedWidthInteger>(_ value: T) { withUnsafeBytes(of: value.littleEndian) { out.append(contentsOf: $0) } }
        out.append(contentsOf: Array("RIFF".utf8))
        append(UInt32(36 + count * 2))
        out.append(contentsOf: Array("WAVEfmt ".utf8))
        append(UInt32(16))
        append(UInt16(1))                       // PCM
        append(UInt16(1))                       // mono
        append(UInt32(sampleRate))
        append(UInt32(sampleRate * 2))          // bytes per second
        append(UInt16(2))                       // block align
        append(UInt16(16))                      // bits per sample
        out.append(contentsOf: Array("data".utf8))
        append(UInt32(count * 2))
        samples.withUnsafeBytes { raw in
            for value in raw.bindMemory(to: Float.self) {
                append(Int16((max(-1, min(1, value)) * 32767).rounded()))
            }
        }
        return out
    }

    private static func fileName(_ title: String) -> String {
        let cleaned = title.components(separatedBy: CharacterSet.alphanumerics.union(.whitespaces).inverted).joined()
            .trimmingCharacters(in: .whitespaces)
        return cleaned.isEmpty ? "Song" : cleaned
    }
}
