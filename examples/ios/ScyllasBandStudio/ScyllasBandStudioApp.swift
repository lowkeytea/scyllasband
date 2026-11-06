import SwiftUI

@main
struct ScyllasBandStudioApp: App {
    @StateObject private var studio: StudioViewModel
    @StateObject private var song: SongViewModel
    @State private var tab = Self.initialTab

    private enum Tab: Hashable { case speak, sing }

    private static var initialTab: Tab {
        #if DEBUG
        if LaunchOptions.singTab { return .sing }
        #endif
        return .speak
    }

    init() {
        // One runtime for both tabs: speech and singing run through the same bundle and worker.
        let controller = RuntimeController()
        _studio = StateObject(wrappedValue: StudioViewModel(controller: controller))
        _song = StateObject(wrappedValue: SongViewModel(controller: controller))
    }

    var body: some Scene {
        WindowGroup {
            if BenchmarkRunner.isRequested {
                // On-device timing for the repository's benchmarks: no studio runtime is created.
                Text("Running benchmark…").task { BenchmarkRunner.start() }
            } else {
                TabView(selection: $tab) {
                    ContentView(model: studio)
                        .tabItem { Label("Speak", systemImage: "text.bubble") }
                        .tag(Tab.speak)
                    SingView(model: song)
                        .tabItem { Label("Sing", systemImage: "music.note") }
                        .tag(Tab.sing)
                }
                .task { studio.initialize() }
            }
        }
    }
}
