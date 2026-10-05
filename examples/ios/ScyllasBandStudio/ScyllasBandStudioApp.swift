import SwiftUI

@main
struct ScyllasBandStudioApp: App {
    @StateObject private var model = StudioViewModel()

    var body: some Scene {
        WindowGroup {
            if BenchmarkRunner.isRequested {
                // On-device timing for the repository's benchmarks: no studio runtime is created.
                Text("Running benchmark…").task { BenchmarkRunner.start() }
            } else {
                ContentView(model: model)
                    .task { model.initialize() }
            }
        }
    }
}
