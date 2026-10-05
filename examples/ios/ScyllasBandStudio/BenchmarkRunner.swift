import Darwin
import Foundation
import ScyllasBandKit

/// On-device timing, started with launch arguments (used by the repository's benchmarks, not by the app's UI):
///
///     -benchmark <bundle> <unit> [requests]
///
/// `<bundle>` is `coreai` or `coreml` (embedded bundles) or a folder name under Documents/benchmark/; `<unit>` is
/// auto, cpu, gpu or ane. Set SCYLLASBAND_COMPUTE_UNITS (e.g. "gpu,vector_estimator=ane") to try another per-graph
/// placement with `auto`. One configuration per launch keeps load and memory numbers independent. Results are printed
/// as one `SBBENCH {...}` JSON line.
enum BenchmarkRunner {
    static let passage = "The last train leaves at midnight, so pack light. Bring a warm coat, and don't forget the map. "
        + "If we miss it, we walk, and nobody wants to walk that far in the rain."

    static var isRequested: Bool { ProcessInfo.processInfo.arguments.contains("-benchmark") }

    /// Runs the requested configuration off the main thread, prints the result and exits the app.
    static func start() {
        let arguments = ProcessInfo.processInfo.arguments
        guard let index = arguments.firstIndex(of: "-benchmark"), arguments.count > index + 2 else {
            print("SBBENCH {\"error\": \"usage: -benchmark <bundle> <unit> [requests]\"}")
            exit(2)
        }
        let bundleName = arguments[index + 1], unitName = arguments[index + 2]
        let requests = arguments.count > index + 3 ? Int(arguments[index + 3]) ?? 4 : 4
        DispatchQueue.global(qos: .userInitiated).async {
            let result = run(bundleName: bundleName, unitName: unitName, requests: requests)
            let data = (try? JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])) ?? Data()
            print("SBBENCH " + (String(data: data, encoding: .utf8) ?? "{}"))
            fflush(stdout)
            exit(0)
        }
    }

    private static func run(bundleName: String, unitName: String, requests: Int) -> [String: Any] {
        var row: [String: Any] = ["bundle": bundleName, "unit": unitName,
                                  "units_override": ProcessInfo.processInfo.environment["SCYLLASBAND_COMPUTE_UNITS"] ?? ""]
        let units: [String: SBScyllasBandComputeUnit] = ["auto": .automatic, "cpu": .CPU, "gpu": .GPU, "ane": .neuralEngine]
        guard let unit = units[unitName], let url = bundleURL(bundleName) else {
            row["error"] = "unknown bundle or unit"
            return row
        }
        row["size_mb"] = round(Double(directorySize(url)) / 1e5) / 10
        let baseline = footprintMB()
        let loadStart = Date()
        let runtime: SBScyllasBand
        do {
            runtime = try SBScyllasBand(bundleURL: url, computeUnit: unit)
        } catch {
            row["error"] = "\(error)"
            return row
        }
        row["backend"] = runtime.bundleInfo.backend
        row["load_ms"] = Date().timeIntervalSince(loadStart) * 1000
        var timings: [[String: Double]] = []
        for _ in 0..<requests {
            let request = SBScyllasBandRequest(text: passage, voiceIdentifier: "scylla")
            request.seed = 7
            let start = Date()
            var first: Double?
            var samples = 0
            do {
                try runtime.synthesizeRequest(request, chunkStarted: nil) { chunk in
                    if first == nil { first = Date().timeIntervalSince(start) * 1000 }
                    samples += chunk.sampleCount
                    return true
                }
            } catch {
                row["error"] = "\(error)"
                break
            }
            timings.append(["first_audio_ms": first ?? -1, "total_ms": Date().timeIntervalSince(start) * 1000,
                            "audio_s": Double(samples) / Double(runtime.bundleInfo.sampleRate)])
        }
        row["requests"] = timings
        if let cold = timings.first {
            row["launch_to_first_audio_ms"] = (row["load_ms"] as? Double ?? 0) + (cold["first_audio_ms"] ?? 0)
        }
        let warm = timings.dropFirst()
        if !warm.isEmpty {
            row["warm_first_audio_ms"] = median(warm.map { $0["first_audio_ms"] ?? 0 })
            row["warm_rtf"] = median(warm.map { ($0["total_ms"] ?? 0) / 1000 / max($0["audio_s"] ?? 1, 0.001) })
        }
        row["footprint_mb"] = footprintMB() - baseline
        row["peak_footprint_mb"] = peakFootprintMB()
        return row
    }

    private static func bundleURL(_ name: String) -> URL? {
        let fileManager = FileManager.default
        if let resources = Bundle.main.resourceURL?.appendingPathComponent("scyllasband/\(name)"),
           fileManager.fileExists(atPath: resources.appendingPathComponent("manifest.json").path) {
            return resources
        }
        let documents = fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("benchmark/\(name)")
        return fileManager.fileExists(atPath: documents.appendingPathComponent("manifest.json").path) ? documents : nil
    }

    private static func directorySize(_ url: URL) -> Int {
        let keys: [URLResourceKey] = [.fileSizeKey, .isRegularFileKey]
        guard let walker = FileManager.default.enumerator(at: url, includingPropertiesForKeys: keys) else { return 0 }
        var total = 0
        for case let file as URL in walker {
            let values = try? file.resourceValues(forKeys: Set(keys))
            if values?.isRegularFile == true { total += values?.fileSize ?? 0 }
        }
        return total
    }

    private static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        return sorted.isEmpty ? 0 : sorted[sorted.count / 2]
    }

    private static func vmInfo() -> task_vm_info_data_t? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let status = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        return status == KERN_SUCCESS ? info : nil
    }

    private static func footprintMB() -> Double { Double(vmInfo()?.phys_footprint ?? 0) / 1e6 }
    private static func peakFootprintMB() -> Double { Double(vmInfo()?.ledger_phys_footprint_peak ?? 0) / 1e6 }
}
