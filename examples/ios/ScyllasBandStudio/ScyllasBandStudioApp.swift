import SwiftUI

@main
struct ScyllasBandStudioApp: App {
    @StateObject private var model = StudioViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView(model: model)
                .task { model.initialize() }
        }
    }
}
