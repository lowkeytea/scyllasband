import Foundation
import os
@preconcurrency import ScyllasBandKit

private let log = Logger(subsystem: "org.scyllasband.studio", category: "runtime")

/// Set from the main thread by Stop, polled by the work running on the runtime's worker.
final class CancellationFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false

    var isSet: Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelled
    }

    func set() {
        lock.lock()
        cancelled = true
        lock.unlock()
    }
}

/// The one runtime the Speak and Sing tabs share: the bundle chosen for this device, its compute unit, and the serial
/// worker every runtime call runs on (requests queue there one after another; the runtime is never called from the main
/// thread). Only one tab plays at a time: `activity` names what is running.
@MainActor
final class RuntimeController: ObservableObject {
    @Published private(set) var runtime: SBScyllasBand?
    @Published private(set) var bundleInfo: SBScyllasBandBundleInfo?
    /// "Core ML" or "Core AI" once a runtime is ready.
    @Published private(set) var backend: String?
    /// The compute unit chosen in the runtime settings.
    @Published private(set) var computeUnit: ComputeUnitOption
    /// The compute unit the current runtime was created with.
    @Published private(set) var activeComputeUnit: ComputeUnitOption?
    @Published private(set) var isPreparing = true
    @Published private(set) var preparingDetail = ""
    /// Why the runtime could not be created, when it could not.
    @Published private(set) var failure: String?
    /// What is using the runtime right now ("speaking", "singing"); nil when idle.
    @Published private(set) var activity: String?

    let worker = DispatchQueue(label: "org.scyllasband.studio.runtime", qos: .userInitiated)
    private static let computeUnitKey = "computeUnit"
    private var generation = 0
    private var readyHandlers: [(SBScyllasBand, String) -> Void] = []
    private var reloadHandlers: [() -> Void] = []

    init() {
        computeUnit = ComputeUnitOption(rawValue: UserDefaults.standard.integer(forKey: Self.computeUnitKey)) ?? .automatic
    }

    /// e.g. "Core ML · Automatic".
    var summary: String {
        "\(backend ?? "Preparing") · \((activeComputeUnit ?? effectiveComputeUnit).title)"
    }

    private var effectiveComputeUnit: ComputeUnitOption {
        ComputeUnitOption.isSelectable ? computeUnit : .cpu
    }

    /// `handler(runtime, readyMessage)` runs on the main thread each time a runtime becomes ready.
    func onReady(_ handler: @escaping (SBScyllasBand, String) -> Void) {
        readyHandlers.append(handler)
    }

    /// `handler` runs before the runtime is replaced (a new compute unit): stop anything using it.
    func onWillReload(_ handler: @escaping () -> Void) {
        reloadHandlers.append(handler)
    }

    func initialize() {
        guard generation == 0 else { return }
        load()
    }

    func setComputeUnit(_ option: ComputeUnitOption) {
        guard option != computeUnit else { return }
        computeUnit = option
        UserDefaults.standard.set(option.rawValue, forKey: Self.computeUnitKey)
        reloadHandlers.forEach { $0() }
        load()
    }

    /// Claims the runtime for `name`; false when something else is using it.
    func begin(_ name: String) -> Bool {
        guard activity == nil, runtime != nil else { return false }
        activity = name
        return true
    }

    func end(_ name: String) {
        if activity == name { activity = nil }
    }

    /// The embedded bundles to try, best first: Core AI on iOS 27 devices, then Core ML. The Simulator has no Core AI.
    private static func bundleCandidates() -> [URL] {
        guard let directory = Bundle.main.resourceURL?.appendingPathComponent("scyllasband", isDirectory: true) else {
            return []
        }
        var candidates: [URL] = []
        #if !targetEnvironment(simulator)
        if let preferred = SBScyllasBand.preferredBundleURL(inDirectory: directory) {
            candidates.append(preferred)
        }
        #endif
        let coreML = directory.appendingPathComponent("coreml", isDirectory: true)
        if FileManager.default.fileExists(atPath: coreML.appendingPathComponent("manifest.json").path),
           !candidates.contains(where: { $0.lastPathComponent == "coreml" }) {
            candidates.append(coreML)
        }
        return candidates
    }

    /// Creates and warms a runtime on the worker; it runs after any work still finishing there.
    private func load() {
        generation += 1
        let generation = generation
        runtime = nil
        backend = nil
        activeComputeUnit = nil
        failure = nil
        isPreparing = true
        let candidates = Self.bundleCandidates()
        guard let first = candidates.first else {
            isPreparing = false
            failure = "Model bundle missing. Download a Core ML or Core AI bundle and rebuild (see README)."
            return
        }
        preparingDetail = first.lastPathComponent == "coreai"
            ? "Core AI optimizes the voices for this device the first time they load after installing the app or updating iOS. This can take a minute; later launches are quick."
            : "Loading the Core ML voices. The first launch after installing takes longest."
        let unit = effectiveComputeUnit
        worker.async { [weak self] in
            var failures: [String] = []
            for bundleURL in candidates {
                let flavor = backendLabel(bundleURL.lastPathComponent)
                let started = Date()
                do {
                    log.notice("Creating the \(flavor, privacy: .public) runtime on \(unit.title, privacy: .public)")
                    let runtime = try SBScyllasBand(bundleURL: bundleURL, computeUnit: unit.native)
                    try runtime.warmUp(withVoice: runtime.bundleInfo.defaultVoice)
                    let seconds = Date().timeIntervalSince(started)
                    log.notice("\(flavor, privacy: .public) runtime created and warmed up in \(seconds, format: .fixed(precision: 2)) s")
                    let fallbackNote = failures.first
                    DispatchQueue.main.async {
                        self?.ready(runtime, unit: unit, generation: generation, seconds: seconds, fallbackNote: fallbackNote)
                    }
                    return
                } catch {
                    log.error("\(flavor, privacy: .public) runtime failed: \(error.localizedDescription, privacy: .public)")
                    failures.append("\(flavor) failed: \(error.localizedDescription)")
                }
            }
            DispatchQueue.main.async {
                guard let self, generation == self.generation else { return }
                self.isPreparing = false
                self.failure = "Model initialization failed. " + failures.joined(separator: " ")
            }
        }
    }

    private func ready(_ runtime: SBScyllasBand, unit: ComputeUnitOption, generation: Int, seconds: TimeInterval, fallbackNote: String?) {
        guard generation == self.generation else { return }
        let info = runtime.bundleInfo
        let label = backendLabel(info.backend)
        self.runtime = runtime
        bundleInfo = info
        backend = label
        activeComputeUnit = unit
        isPreparing = false
        var message = "\(label) ready on \(unit.title) in \(seconds.formatted(.number.precision(.fractionLength(1)))) s"
        if let fallbackNote {
            message += " (\(fallbackNote))"
        }
        readyHandlers.forEach { $0(runtime, message) }
    }
}
