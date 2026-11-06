import Foundation

/// One note of a song line: the syllable sung on it, its pitch (nil for a rest) and its length in beats (quarter notes).
struct SongNote: Identifiable, Codable, Equatable {
    var id = UUID()
    /// The text sung on this note: a word, one syllable of it, or "" when the note holds the previous syllable.
    var syllable: String
    /// MIDI pitch (60 = C4); nil for a rest.
    var midi: Int?
    var beats: Double
    /// True when this note continues the previous note's word (its next syllable, or the same syllable held).
    var continuesWord = false

    var isRest: Bool { midi == nil }

    static func rest(beats: Double) -> SongNote { SongNote(syllable: "", midi: nil, beats: beats) }
}

/// A line of the song: a run of notes that is sung as one phrase. Lines are separated by `restAfter` beats of rest.
struct SongLine: Identifiable, Codable, Equatable {
    var id = UUID()
    var notes: [SongNote]
    /// The rest after the line (beats). It must last at least 0.25 s for the runtime to start a new phrase there.
    var restAfter: Double = 1

    /// The line's lyrics in the editor's notation: words separated by spaces, syllables by "-", a held note by "_".
    var lyrics: String {
        var words: [String] = []
        for note in notes where !note.isRest {
            if note.continuesWord, !words.isEmpty {
                words[words.count - 1] += note.syllable.isEmpty ? "_" : "-" + note.syllable
            } else {
                words.append(note.syllable.isEmpty ? "_" : note.syllable)
            }
        }
        return words.joined(separator: " ")
    }
}

enum SongStyle: String, Codable, CaseIterable, Identifiable {
    case pop, ballad
    var id: String { rawValue }
    var title: String { self == .pop ? "Pop" : "Ballad" }
    /// The length a new note gets.
    var defaultBeats: Double { self == .pop ? 0.5 : 1 }
}

/// Quality is BigVGAN, the singing decoder; Turbo is Vocos, the speech decoder: much faster, with some harmonic artefacts on
/// sustained notes in some voices.
enum SongDecoder: String, Codable, CaseIterable, Identifiable {
    case quality, turbo
    var id: String { rawValue }
    var title: String { self == .quality ? "Quality" : "Turbo" }
}

struct SongDocument: Codable, Equatable, Identifiable {
    var id = UUID()
    var title: String
    var bpm: Double
    var style: SongStyle
    var voice: String
    var decoder: SongDecoder = .quality
    var transpose = 0
    var lines: [SongLine]

    static func blank(voice: String) -> SongDocument {
        SongDocument(title: "New song", bpm: 104, style: .pop, voice: voice,
                     lines: [SongLine(notes: [SongNote(syllable: "la", midi: 60, beats: 1)])])
    }

    var secondsPerBeat: Double { 60 / bpm }
    /// The shortest rest, in beats, that still separates two lines (0.25 s, rounded up to a sixteenth).
    var minimumLineRest: Double { (0.25 / secondsPerBeat * 4).rounded(.up) / 4 }

    var noteCount: Int { lines.reduce(0) { $0 + $1.notes.count } }

    // MARK: - The runtime's score

    /// The score the runtime sings: {"text", "bpm", "style", "notes": [[label, note, beats], ...]}. A word over several notes
    /// is written as its syllables ("ba~1", "by~2") when each note has its own, else as the whole word on every part
    /// ("home~1", "home~2", a syllable held over two pitches). Lines are joined by their rests.
    func scoreJSON() -> String { score().json }

    /// The score and, for each of its rows ("notes"), the note it came from (nil for the rests between lines), so an error
    /// the runtime reports for a row can point at a note.
    func score() -> (json: String, rows: [UUID?]) {
        var rows: [[Any]] = []
        var rowNotes: [UUID?] = []
        var words: [String] = []
        for (lineIndex, line) in lines.enumerated() {
            var index = 0
            while index < line.notes.count {
                let note = line.notes[index]
                if note.isRest {
                    rows.append(["-", "rest", note.beats])
                    rowNotes.append(note.id)
                    index += 1
                    continue
                }
                var group = [note]
                var next = index + 1
                while next < line.notes.count, !line.notes[next].isRest, line.notes[next].continuesWord {
                    group.append(line.notes[next])
                    next += 1
                }
                let syllables = group.map { Self.clean($0.syllable) }
                let word = syllables.joined()
                let spelled = !syllables.contains(where: \.isEmpty) && Set(syllables).count > 1
                words.append(word.isEmpty ? "la" : word)
                for (part, member) in group.enumerated() {
                    let label: String
                    if group.count == 1 {
                        label = word.isEmpty ? "la" : word
                    } else {
                        label = "\(spelled ? syllables[part] : (word.isEmpty ? "la" : word))~\(part + 1)"
                    }
                    rows.append([label, Self.noteName(member.midi ?? 60), member.beats])
                    rowNotes.append(member.id)
                }
                index = next
            }
            if lineIndex < lines.count - 1 {
                rows.append(["-", "rest", max(line.restAfter, minimumLineRest)])
                rowNotes.append(nil)
            }
        }
        let score: [String: Any] = ["text": words.joined(separator: " "), "bpm": bpm, "style": style.rawValue, "notes": rows]
        let data = (try? JSONSerialization.data(withJSONObject: score, options: [.sortedKeys])) ?? Data("{}".utf8)
        return (String(decoding: data, as: UTF8.self), rowNotes)
    }

    /// A song from the runtime's normalized score (`SBScyllasBand.parseScoreJSON`): its lines become lines, each closing
    /// rest becomes the line's `restAfter`, and words over several notes become syllable notes.
    static func from(summary: [String: Any], title: String, voice: String) -> SongDocument? {
        guard let notes = summary["notes"] as? [[String: Any]], let bpm = (summary["bpm"] as? NSNumber)?.doubleValue else { return nil }
        let style = SongStyle(rawValue: summary["style"] as? String ?? "pop") ?? (summary["mode"] as? String == "ballad" ? .ballad : .pop)
        let ranges = (summary["lines"] as? [[String: Any]])?.compactMap { line -> ClosedRange<Int>? in
            guard let first = (line["first"] as? NSNumber)?.intValue, let last = (line["last"] as? NSNumber)?.intValue, first <= last else { return nil }
            return first...last
        } ?? [0...(notes.count - 1)]
        // The syllable each note carries: a word's labels name its syllables ("ba~1", "by~2"), or the whole word on every
        // part ("home~1", "home~2"; the first note carries the word, the rest hold it).
        var syllables = [String](repeating: "", count: notes.count)
        var index = 0
        while index < notes.count {
            guard let word = (notes[index]["word_index"] as? NSNumber)?.intValue else {
                index += 1
                continue
            }
            var members = [index]
            var next = index + 1
            while next < notes.count, (notes[next]["word_index"] as? NSNumber)?.intValue == word {
                members.append(next)
                next += 1
            }
            let bases = members.map { base(of: notes[$0]["label"] as? String ?? "") }
            if Set(bases).count == 1 {
                syllables[members[0]] = bases[0]
            } else {
                for (member, part) in zip(members, bases) { syllables[member] = part }
            }
            index = next
        }
        var lines: [SongLine] = []
        for (lineIndex, range) in ranges.enumerated() {
            var lineNotes: [SongNote] = []
            for position in range {
                let note = notes[position]
                let beats = (note["beats"] as? NSNumber)?.doubleValue ?? 1
                if let midi = (note["midi"] as? NSNumber)?.intValue {
                    let part = (note["part"] as? NSNumber)?.intValue ?? 1
                    lineNotes.append(SongNote(syllable: syllables[position], midi: midi, beats: beats, continuesWord: part > 1))
                } else {
                    lineNotes.append(.rest(beats: beats))
                }
            }
            // The rests that close a line (every line but the last) become its restAfter.
            var closing = 0.0
            while lineIndex < ranges.count - 1, let last = lineNotes.last, last.isRest {
                closing += last.beats
                lineNotes.removeLast()
            }
            lines.append(SongLine(notes: lineNotes, restAfter: closing > 0 ? closing : 1))
        }
        guard !lines.isEmpty else { return nil }
        return SongDocument(title: title, bpm: bpm, style: style, voice: voice, lines: lines)
    }

    // MARK: - Lyrics

    /// Rewrites a line's sung notes from typed lyrics (words by spaces, syllables by "-", "_" holds the previous syllable over
    /// another note), keeping the existing notes' pitches and lengths in order and the rests where they are.
    static func applying(lyrics: String, to line: SongLine, style: SongStyle) -> SongLine {
        var syllables: [(text: String, continues: Bool)] = []
        for word in lyrics.split(whereSeparator: \.isWhitespace) {
            var first = true
            var pending = ""
            func flush() {
                guard !pending.isEmpty else { return }
                syllables.append((pending, !first))
                first = false
                pending = ""
            }
            for character in word {
                if character == "-" {
                    flush()
                } else if character == "_" {
                    flush()
                    syllables.append(("", !first || !syllables.isEmpty))
                    first = false
                } else {
                    pending.append(character)
                }
            }
            flush()
        }
        let sung = line.notes.filter { !$0.isRest }
        var result: [SongNote] = []
        var used = 0
        var lastPitch = sung.first?.midi ?? 60
        for note in line.notes {
            if note.isRest {
                result.append(note)
                continue
            }
            guard used < syllables.count else { continue }   // fewer syllables: the extra notes go
            var updated = note
            updated.syllable = syllables[used].text
            updated.continuesWord = syllables[used].continues
            lastPitch = updated.midi ?? lastPitch
            result.append(updated)
            used += 1
        }
        while used < syllables.count {   // more syllables: new notes continue the line's last pitch
            result.append(SongNote(syllable: syllables[used].text, midi: lastPitch, beats: style.defaultBeats,
                                   continuesWord: syllables[used].continues))
            used += 1
        }
        if let first = result.firstIndex(where: { !$0.isRest }) { result[first].continuesWord = false }
        var updatedLine = line
        updatedLine.notes = result
        return updatedLine
    }

    // MARK: - Helpers

    static func noteName(_ midi: Int) -> String {
        let names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        let octave = Int((Double(midi) / 12).rounded(.down)) - 1
        return names[((midi % 12) + 12) % 12] + String(octave)
    }

    /// A syllable as a score label allows it: no spaces, no "~", never the rest label "-".
    private static func clean(_ text: String) -> String {
        let kept = text.filter { !$0.isWhitespace && $0 != "~" }
        return kept == "-" ? "" : kept
    }

    private static func base(of label: String) -> String {
        String(label.split(separator: "~", maxSplits: 1, omittingEmptySubsequences: false).first ?? "")
    }
}
