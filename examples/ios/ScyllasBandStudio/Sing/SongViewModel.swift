import Combine
import Foundation
import os
@preconcurrency import ScyllasBandKit

private let log = Logger(subsystem: "org.scyllasband.studio", category: "sing")

/// The note being edited.
struct NoteSelection: Equatable {
    var line: UUID
    var note: UUID
}

/// The Sing tab: a song being written (lines of note chips), its fit to the voice, a tone preview, and sung takes.
///
/// Every check of the song goes through the runtime, so the app never re-implements the score rules: the score is
/// validated with `SBScyllasBand.parseScoreJSON`, its range read with `singFitForRequest` and previewed with
/// `previewForRequest` (no graph, so they answer at once even while a take renders), and sung with `singRequest`.
@MainActor
final class SongViewModel: ObservableObject {
    @Published var song: SongDocument
    @Published var selection: NoteSelection?
    @Published private(set) var fitSummary = ""
    @Published private(set) var suggestedTranspose: Int?
    /// The runtime's objection to the current score, and the note it names when it names one.
    @Published private(set) var problem: String?
    @Published private(set) var problemNote: UUID?
    @Published private(set) var isSinging = false
    @Published private(set) var isPreviewing = false
    @Published private(set) var progress: String?
    @Published var status = "Write a line of lyrics, then tap a note to set its pitch and length."
    @Published private(set) var savedSongs: [SongDocument] = []
    @Published private(set) var takeURL: URL?
    @Published private(set) var scoreURL: URL?

    let controller: RuntimeController
    let examples = ExampleScore.all()
    private var changes: AnyCancellable?
    private let checks = DispatchQueue(label: "org.scyllasband.studio.song-checks", qos: .userInitiated)
    private var checkGeneration = 0
    private var cancellation: CancellationFlag?
    private var player: StreamingAudioPlayer?
    /// The last take and what it was sung from, so Sing replays it until the song changes.
    private var take: (key: String, samples: Data, sampleRate: Int)?
    private var takeSeed: UInt64 = 31_415

    init(controller: RuntimeController) {
        self.controller = controller
        song = .blank(voice: "gwen")
        changes = controller.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }
        controller.onReady { [weak self] _, _ in self?.runtimeReady() }
        controller.onWillReload { [weak self] in self?.stop() }
        savedSongs = SongStore.saved()
    }

    var bundleInfo: SBScyllasBandBundleInfo? { controller.bundleInfo }
    var canSing: Bool { bundleInfo?.singing.isSupported == true }
    var isBusy: Bool { isSinging || isPreviewing }
    var runtimeSummary: String { controller.summary }
    var voices: [SBScyllasBandVoice] { bundleInfo?.voices ?? [] }
    var hasBigVGAN: Bool { bundleInfo?.singing.hasBigVGAN == true }

    private func runtimeReady() {
        guard let info = bundleInfo else { return }
        if info.voice(withIdentifier: song.voice) == nil { song.voice = info.defaultVoice }
        if !info.singing.isSupported {
            status = "This bundle speaks only. Download a singing bundle (v4) to sing."
        } else if let first = examples.first(where: { $0.voice != nil }), song.title == "New song" {
            load(first)
        }
        songChanged()
        #if DEBUG
        applyLaunchOptions()
        #endif
    }

    #if DEBUG
    private var didApplyLaunchOptions = false

    private func applyLaunchOptions() {
        guard !didApplyLaunchOptions else { return }
        didApplyLaunchOptions = true
        if let name = LaunchOptions.singExample, let example = examples.first(where: { $0.url.deletingPathExtension().lastPathComponent == name }) {
            load(example)
        }
        if LaunchOptions.singTurbo { song.decoder = .turbo }
        if LaunchOptions.selectFirstNote, let line = song.lines.first, let note = line.notes.first(where: { !$0.isRest }) {
            selection = NoteSelection(line: line.id, note: note.id)
        }
        if LaunchOptions.autosing {
            DispatchQueue.main.asyncAfter(deadline: .now() + 1) { self.singOrStop() }
        }
    }
    #endif

    // MARK: - Editing

    func note(_ selection: NoteSelection?) -> SongNote? {
        guard let selection, let line = song.lines.first(where: { $0.id == selection.line }) else { return nil }
        return line.notes.first { $0.id == selection.note }
    }

    func updateLyrics(line id: UUID, to text: String) {
        edit(line: id) { line, style in line = SongDocument.applying(lyrics: text, to: line, style: style) }
    }

    func setPitch(_ midi: Int) {
        editSelected { note in note.midi = max(24, min(96, midi)) }
    }

    func setBeats(_ beats: Double) {
        editSelected { note in note.beats = beats }
    }

    func setSyllable(_ text: String) {
        editSelected { note in note.syllable = text.filter { !$0.isWhitespace } }
    }

    /// A sung note becomes a rest of the same length, and a rest a note at the previous pitch.
    func toggleRest() {
        guard let selection else { return }
        edit(line: selection.line) { line, _ in
            guard let index = line.notes.firstIndex(where: { $0.id == selection.note }) else { return }
            if line.notes[index].isRest {
                let pitch = line.notes[..<index].last(where: { !$0.isRest })?.midi ?? 60
                line.notes[index] = SongNote(id: line.notes[index].id, syllable: "la", midi: pitch, beats: line.notes[index].beats)
            } else {
                line.notes[index] = SongNote(id: line.notes[index].id, syllable: "", midi: nil, beats: line.notes[index].beats)
                if index + 1 < line.notes.count { line.notes[index + 1].continuesWord = false }
            }
        }
    }

    /// Splits the note in two halves; the second holds the same syllable, so it can take a second pitch (a melisma).
    func splitSelected() {
        guard let selection else { return }
        edit(line: selection.line) { line, _ in
            guard let index = line.notes.firstIndex(where: { $0.id == selection.note }) else { return }
            var first = line.notes[index]
            first.beats = max(0.25, first.beats / 2)
            var second = SongNote(syllable: "", midi: first.midi, beats: first.beats, continuesWord: !first.isRest)
            if first.isRest { second.syllable = "" }
            line.notes[index] = first
            line.notes.insert(second, at: index + 1)
        }
    }

    /// Joins the note with the next one: their lengths add up and the first note's pitch and syllable stay.
    func mergeWithNext() {
        guard let selection else { return }
        edit(line: selection.line) { line, _ in
            guard let index = line.notes.firstIndex(where: { $0.id == selection.note }), index + 1 < line.notes.count else { return }
            let next = line.notes.remove(at: index + 1)
            line.notes[index].beats += next.beats
            if !next.syllable.isEmpty, next.continuesWord, !line.notes[index].isRest {
                line.notes[index].syllable += next.syllable
            }
        }
    }

    func insertRestAfter() {
        guard let selection else { return }
        edit(line: selection.line) { line, style in
            guard let index = line.notes.firstIndex(where: { $0.id == selection.note }) else { return }
            line.notes.insert(.rest(beats: style.defaultBeats), at: index + 1)
            if index + 2 < line.notes.count { line.notes[index + 2].continuesWord = false }
        }
    }

    func deleteSelected() {
        guard let target = selection else { return }
        edit(line: target.line) { line, _ in
            guard line.notes.count > 1, let index = line.notes.firstIndex(where: { $0.id == target.note }) else { return }
            let removed = line.notes.remove(at: index)
            if index < line.notes.count, !removed.continuesWord { line.notes[index].continuesWord = false }
        }
        selection = nil
    }

    func addLine() {
        let pitch = song.lines.last?.notes.last(where: { !$0.isRest })?.midi ?? 60
        song.lines.append(SongLine(notes: [SongNote(syllable: "la", midi: pitch, beats: song.style.defaultBeats * 2)],
                                   restAfter: max(1, song.minimumLineRest)))
        songChanged()
    }

    func removeLine(_ id: UUID) {
        guard song.lines.count > 1 else { return }
        song.lines.removeAll { $0.id == id }
        if selection?.line == id { selection = nil }
        songChanged()
    }

    func setRestAfter(line id: UUID, beats: Double) {
        edit(line: id) { line, _ in line.restAfter = beats }
    }

    /// The suggested transpose from the last fit.
    func applySuggestedTranspose() {
        guard let suggestedTranspose else { return }
        song.transpose = suggestedTranspose
        songChanged()
    }

    private func editSelected(_ change: (inout SongNote) -> Void) {
        guard let selection else { return }
        edit(line: selection.line) { line, _ in
            guard let index = line.notes.firstIndex(where: { $0.id == selection.note }) else { return }
            change(&line.notes[index])
        }
    }

    private func edit(line id: UUID, _ change: (inout SongLine, SongStyle) -> Void) {
        guard let index = song.lines.firstIndex(where: { $0.id == id }) else { return }
        change(&song.lines[index], song.style)
        songChanged()
    }

    // MARK: - Checks through the runtime

    /// Called after every change: the score is checked and its fit measured off the main thread, newest change wins.
    func songChanged() {
        checkGeneration += 1
        let generation = checkGeneration
        let (json, rows) = song.score()
        let request = makeRequest(json: json)
        guard let runtime = controller.runtime, canSing else { return }
        checks.asyncAfter(deadline: .now() + 0.15) { [weak self] in
            var problem: String?
            var problemRow: Int?
            var fit: [AnyHashable: Any]?
            do {
                _ = try SBScyllasBand.parseScore(json: json, transpose: 0)
                fit = try runtime.singFit(for: request)
            } catch let error as NSError {
                problem = error.localizedDescription
                problemRow = (error.userInfo[SBScyllasBandScoreNoteIndexKey] as? NSNumber)?.intValue
            }
            DispatchQueue.main.async {
                guard let self, generation == self.checkGeneration else { return }
                self.problem = problem
                self.problemNote = problemRow.flatMap { $0 < rows.count ? rows[$0] : nil }
                self.updateFit(fit)
            }
        }
    }

    private func updateFit(_ fit: [AnyHashable: Any]?) {
        guard let fit else {
            fitSummary = ""
            suggestedTranspose = nil
            return
        }
        let low = fit["low_note"] as? String ?? "", high = fit["high_note"] as? String ?? ""
        let outside = (fit["outside_semitones"] as? NSNumber)?.doubleValue ?? 0
        let mean = (fit["mean_midi"] as? NSNumber)?.doubleValue ?? 0
        let center = (fit["voice_center"] as? NSNumber)?.doubleValue
        let suggested = (fit["suggested_transpose"] as? NSNumber)?.intValue ?? song.transpose
        let voice = song.voice.capitalized
        var text = "Sung \(low)–\(high)"
        if let center, outside > 0 {
            let direction = mean > center ? "high" : "low"
            text += " · \(outside.formatted(.number.precision(.fractionLength(0...1)))) semitones \(direction) for \(voice)"
            suggestedTranspose = suggested != song.transpose ? suggested : nil
        } else {
            text += " · in \(voice)’s range"
            suggestedTranspose = nil
        }
        fitSummary = text
    }

    private func makeRequest(json: String? = nil, seed: UInt64? = nil) -> SBScyllasBandSingRequest {
        let request = SBScyllasBandSingRequest(scoreJSON: json ?? song.scoreJSON(), voiceIdentifier: song.voice)
        request.transpose = song.transpose
        request.decoder = song.decoder == .turbo || !hasBigVGAN ? .vocos : .bigVGAN
        if let seed { request.seed = NSNumber(value: seed) }
        return request
    }

    // MARK: - Preview and singing

    /// Plays the melody as tones at the pitch it will be sung; no model runs, so it starts at once.
    func preview() {
        if isPreviewing {
            stop()
            return
        }
        guard let runtime = controller.runtime, problem == nil, controller.begin("previewing") else { return }
        let request = makeRequest()
        let flag = CancellationFlag()
        cancellation = flag
        isPreviewing = true
        status = "Previewing the melody…"
        checks.async { [weak self] in
            var failure: Error?
            do {
                let tones = try runtime.preview(for: request)
                let player = try StreamingAudioPlayer(sampleRate: tones.sampleRate, mode: .default)
                DispatchQueue.main.async { self?.player = player }
                _ = try player.enqueue(tones, shouldStop: { flag.isSet }, onPlaybackStarted: {})
                player.finish { flag.isSet }
                player.stop()
            } catch {
                failure = error
            }
            DispatchQueue.main.async {
                guard let self else { return }
                self.player = nil
                self.isPreviewing = false
                self.controller.end("previewing")
                self.status = failure.map { "Preview failed: \($0.localizedDescription)" } ?? "Preview finished."
            }
        }
    }

    /// Sings the song. With Turbo, or when the first phrase renders faster than real time, it plays while the rest renders;
    /// otherwise it renders the whole song, then plays. The take is kept until the song changes.
    func singOrStop() {
        if isSinging || isPreviewing {
            stop()
            return
        }
        guard let runtime = controller.runtime, let info = bundleInfo, canSing else { return }
        if let problem {
            status = problem
            return
        }
        let request = makeRequest(seed: takeSeed)
        let key = "\(request.scoreJSON)|\(song.voice)|\(song.transpose)|\(song.decoder.rawValue)|\(takeSeed)"
        guard controller.begin("singing") else {
            status = "The runtime is busy speaking; stop it first."
            return
        }
        let flag = CancellationFlag()
        cancellation = flag
        isSinging = true
        takeURL = nil
        if let take, take.key == key {
            replay(take.samples, sampleRate: take.sampleRate, flag: flag)
            return
        }
        let turbo = request.decoder == .vocos
        let sampleRate = info.sampleRate
        progress = "Preparing…"
        status = turbo ? "Singing with Turbo (Vocos)…" : "Singing with Quality (BigVGAN)…"
        let started = Date()
        controller.worker.async { [weak self] in
            var samples = Data()
            var buffered: [Data] = []
            var streaming = turbo
            var decided = turbo
            var failure: Error?
            var player: StreamingAudioPlayer?
            do {
                player = try StreamingAudioPlayer(sampleRate: sampleRate, mode: .default)
                let livePlayer = player
                DispatchQueue.main.async { self?.player = livePlayer }
                try runtime.sing(request, phraseStarted: { index, count, text in
                    DispatchQueue.main.async {
                        self?.progress = "Phrase \(index + 1) of \(count)" + (text.map { ": \($0)" } ?? "")
                    }
                    return !flag.isSet
                }, phraseAudio: { phrase in
                    samples.append(phrase.pcmFloat32Data)
                    if !decided {
                        // Stream from now on only if rendering keeps ahead of playback.
                        let rendered = Date().timeIntervalSince(started)
                        let audio = Double(samples.count / MemoryLayout<Float>.size) / Double(sampleRate)
                        streaming = rendered < audio * 0.9
                        decided = true
                    }
                    if streaming, let player {
                        for pending in buffered + [phrase.pcmFloat32Data] {
                            guard (try? player.enqueue(pcmFloat32Data: pending, sampleCount: pending.count / MemoryLayout<Float>.size,
                                                       shouldStop: { flag.isSet }, onPlaybackStarted: {})) == true else { return false }
                        }
                        buffered.removeAll()
                    } else {
                        buffered.append(phrase.pcmFloat32Data)
                    }
                    return !flag.isSet
                })
                let renderSeconds = Date().timeIntervalSince(started)
                let audioSeconds = Double(samples.count / MemoryLayout<Float>.size) / Double(sampleRate)
                log.notice("Sang \(audioSeconds, format: .fixed(precision: 1)) s in \(renderSeconds, format: .fixed(precision: 1)) s")
                DispatchQueue.main.async {
                    self?.progress = "Playing"
                    self?.take = (key, samples, sampleRate)
                }
                if let player, !flag.isSet {
                    for pending in buffered {
                        guard (try? player.enqueue(pcmFloat32Data: pending, sampleCount: pending.count / MemoryLayout<Float>.size,
                                                   shouldStop: { flag.isSet }, onPlaybackStarted: {})) == true else { break }
                    }
                    player.finish { flag.isSet }
                }
                player?.stop()
                DispatchQueue.main.async {
                    self?.status = "Sang \(audioSeconds.formatted(.number.precision(.fractionLength(1)))) s of audio in \(renderSeconds.formatted(.number.precision(.fractionLength(1)))) s."
                }
            } catch {
                player?.stop()
                failure = error
            }
            DispatchQueue.main.async {
                self?.finishSinging(cancelled: flag.isSet, failure: failure)
            }
        }
    }

    /// A new seed: the next Sing renders a different take of the same song.
    func anotherTake() {
        takeSeed &+= 1
        take = nil
        takeURL = nil
        status = "The next Sing renders a new take."
    }

    func stop() {
        cancellation?.set()
        if isSinging { controller.runtime?.requestCancellation() }
        player?.stop()
    }

    private func replay(_ samples: Data, sampleRate: Int, flag: CancellationFlag) {
        status = "Playing the last take…"
        progress = "Playing"
        checks.async { [weak self] in
            var failure: Error?
            do {
                let player = try StreamingAudioPlayer(sampleRate: sampleRate, mode: .default)
                DispatchQueue.main.async { self?.player = player }
                _ = try player.enqueue(pcmFloat32Data: samples, sampleCount: samples.count / MemoryLayout<Float>.size,
                                       shouldStop: { flag.isSet }, onPlaybackStarted: {})
                player.finish { flag.isSet }
                player.stop()
            } catch {
                failure = error
            }
            DispatchQueue.main.async { self?.finishSinging(cancelled: flag.isSet, failure: failure) }
        }
    }

    private func finishSinging(cancelled: Bool, failure: Error?) {
        player = nil
        cancellation = nil
        isSinging = false
        progress = nil
        controller.end("singing")
        if let failure, !cancelled {
            status = "Singing failed: \(failure.localizedDescription)"
        } else if cancelled {
            status = "Stopped."
        }
        #if DEBUG
        if LaunchOptions.exitAfterSing {
            print("SBSING \(status)")
            fflush(stdout)
            exit(0)
        }
        #endif
    }

    // MARK: - Songs

    func load(_ example: ExampleScore) {
        guard let data = try? Data(contentsOf: example.url), let json = String(data: data, encoding: .utf8) else { return }
        do {
            let summary = try SBScyllasBand.parseScore(json: json, transpose: 0)
            let voice = example.voice.flatMap { bundleInfo?.voice(withIdentifier: $0)?.identifier } ?? song.voice
            if let loaded = SongDocument.from(summary: summary as? [String: Any] ?? [:], title: example.title, voice: voice) {
                stop()
                song = loaded
                selection = nil
                take = nil
                status = "Loaded “\(example.title)”: \(loaded.lines.count) lines, \(loaded.noteCount) notes."
                songChanged()
            }
        } catch {
            status = "Could not load “\(example.title)”: \(error.localizedDescription)"
        }
    }

    func newSong() {
        stop()
        song = .blank(voice: bundleInfo?.defaultVoice ?? song.voice)
        selection = nil
        take = nil
        status = "A new song: type lyrics into the line, then set each note."
        songChanged()
    }

    func open(_ saved: SongDocument) {
        stop()
        song = saved
        selection = nil
        take = nil
        status = "Opened “\(saved.title)”."
        songChanged()
    }

    func save() {
        do {
            try SongStore.save(song)
            savedSongs = SongStore.saved()
            status = "Saved “\(song.title)”."
        } catch {
            status = "Could not save: \(error.localizedDescription)"
        }
    }

    func delete(_ saved: SongDocument) {
        SongStore.delete(saved)
        savedSongs = SongStore.saved()
    }

    /// Writes the score and (when there is one) the last take as files to share.
    func prepareShare() {
        scoreURL = try? SongStore.exportScore(song)
        if let take {
            takeURL = try? SongStore.exportTake(take.samples, sampleRate: take.sampleRate, title: song.title)
        } else {
            takeURL = nil
        }
    }
}
