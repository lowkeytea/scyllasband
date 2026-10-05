import SwiftUI
import ScyllasBandKit

struct RuntimeSettingsView: View {
    @ObservedObject var model: StudioViewModel
    @Environment(\.dismiss) private var dismiss

    private var computeUnitSelection: Binding<ComputeUnitOption> {
        Binding(
            get: { ComputeUnitOption.isSelectable ? model.computeUnit : .cpu },
            set: { model.setComputeUnit($0) }
        )
    }

    private var footer: String {
        var text = "Automatic follows the bundle’s recommendation for each model (the GPU). Neural Engine runs the models that are accurate there on the Neural Engine (Core AI’s speech generator) and the rest on the CPU. Changing the compute unit reloads the voices. iOS doesn’t allow GPU work from background apps, so choose Neural Engine (Core AI) or CPU for an app that keeps speaking in the background."
        if !ComputeUnitOption.isSelectable {
            text += " The Simulator always runs Core ML on the CPU."
        }
        return text
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("Runtime") {
                    LabeledContent("Backend") {
                        if model.isPreparing {
                            HStack(spacing: 8) {
                                ProgressView()
                                Text("Preparing voices…")
                            }
                        } else {
                            Text(model.backend ?? "Unavailable")
                        }
                    }
                    LabeledContent("Compute unit", value: model.activeComputeUnit?.title ?? "–")
                    if let info = model.bundleInfo {
                        if !info.releaseIdentifier.isEmpty {
                            LabeledContent("Model release", value: info.releaseIdentifier)
                        }
                        LabeledContent("Voices", value: "\(info.voices.count)")
                        LabeledContent("Sample rate", value: "\(info.sampleRate) Hz")
                    }
                }

                Section {
                    Picker("Compute unit", selection: computeUnitSelection) {
                        ForEach(ComputeUnitOption.allCases) { option in
                            Text(option.title).tag(option)
                        }
                    }
                    .pickerStyle(.inline)
                    .labelsHidden()
                    .disabled(!ComputeUnitOption.isSelectable)
                } header: {
                    Text("Run the voices on")
                } footer: {
                    Text(footer)
                }
            }
            .navigationTitle("Runtime")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}
