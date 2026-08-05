#if canImport(CoreAI)
import CoreAI
#endif
import Darwin
import Foundation

private let tensorFloat32: Int32 = 1
private let tensorInt64: Int32 = 4
private let tensorInt32: Int32 = 5
private let tensorBool: Int32 = 9
private let acceleratorCPU: Int32 = 1
private let acceleratorNPU: Int32 = 3

private struct TensorViewLayout {
    var name: UnsafePointer<CChar>?
    var dataType: Int32
    var shape: UnsafePointer<Int64>?
    var rank: Int32
    var data: UnsafeRawPointer?
    var byteLength: UInt64
}

private struct OwnedTensorLayout {
    var name: UnsafeMutablePointer<CChar>?
    var dataType: Int32
    var shape: UnsafeMutablePointer<Int64>?
    var rank: Int32
    var data: UnsafeMutableRawPointer?
    var byteLength: UInt64
}

@_silgen_name("scyllasband_graph_session_set_error")
private func setNativeGraphError(_ message: UnsafePointer<CChar>?)

@_silgen_name("scyllasband_tensors_destroy")
private func destroyNativeTensors(_ tensors: UnsafeMutableRawPointer?, _ count: Int32)

// Core AI 27 is Swift-only. These C entry points preserve libscyllasband's
// existing graph-session boundary so all duration-flow orchestration remains
// in the shared C++ runtime.
//
// CoreAI ships only in the iOS/macOS 27 device SDKs; when it is absent
// (simulator, older SDKs) the #else branch at the bottom of this file exports
// stub sessions so the dispatch layer still links and reports a clear error.

#if canImport(CoreAI)

@available(iOS 27.0, macOS 27.0, *)
private final class CoreAISession: @unchecked Sendable {
    let model: AIModel
    let modelPath: String
    let functions: [String: InferenceFunction]

    init(modelPath: String, accelerator: Int32) throws {
        self.modelPath = modelPath
        var mutableOptions: SpecializationOptions
        switch accelerator {
        case acceleratorCPU:
            mutableOptions = .cpuOnly
        case acceleratorNPU:
            mutableOptions = SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
        default:
            mutableOptions = SpecializationOptions(preferredComputeUnitKind: .gpu)
        }
        mutableOptions.expectFrequentReshapes = false
        let options = mutableOptions
        let loadedModel = try waitForAsync {
            try await AIModel(
                contentsOf: URL(fileURLWithPath: modelPath),
                options: options
            )
        }
        var loadedFunctions: [String: InferenceFunction] = [:]
        loadedFunctions.reserveCapacity(loadedModel.functionNames.count)
        for name in loadedModel.functionNames {
            guard let function = try loadedModel.loadFunction(named: name) else {
                throw CoreAISessionError.missingFunction(name)
            }
            loadedFunctions[name] = function
        }
        model = loadedModel
        functions = loadedFunctions
    }

    func run(_ inputViews: UnsafeBufferPointer<TensorViewLayout>) throws -> [OwnedCoreAIOutput] {
        let functionName = try selectFunction(inputViews)
        guard let function = functions[functionName] else {
            throw CoreAISessionError.missingFunction(functionName)
        }
        let descriptor = function.descriptor
        guard descriptor.inputCount == inputViews.count else {
            throw CoreAISessionError.inputCount(
                function: functionName,
                expected: descriptor.inputCount,
                actual: inputViews.count
            )
        }
        var inputs: [String: NDArray] = [:]
        inputs.reserveCapacity(inputViews.count)
        for (index, name) in descriptor.inputNames.enumerated() {
            guard case .ndArray(let expected) = descriptor.inputDescriptor(of: name) else {
                throw CoreAISessionError.unsupportedInput(name)
            }
            inputs[name] = try makeArray(from: inputViews[index], expected: expected, name: name)
        }

        let runInputs = inputs
        return try waitForAsync {
            var values = try await function.run(inputs: runInputs)
            var outputs: [OwnedCoreAIOutput] = []
            outputs.reserveCapacity(descriptor.outputCount)
            for name in descriptor.outputNames {
                guard let value = values.remove(name), let array = value.ndArray else {
                    throw CoreAISessionError.missingOutput(name)
                }
                outputs.append(try OwnedCoreAIOutput(name: name, array: array))
            }
            return outputs
        }
    }

    private func selectFunction(
        _ inputs: UnsafeBufferPointer<TensorViewLayout>
    ) throws -> String {
        let names = model.functionNames
        if names.count == 1, let only = names.first {
            return only
        }
        guard let first = inputs.first, let shapePointer = first.shape, first.rank >= 3 else {
            throw CoreAISessionError.cannotSelectFunction(names)
        }
        let batch = Int(shapePointer[0])
        let frames = Int(shapePointer[Int(first.rank) - 1])
        let vectorName = "b\(batch)_t\(frames)"
        if names.contains(vectorName) {
            return vectorName
        }
        let vocoderName = "t\(frames)"
        if names.contains(vocoderName) {
            return vocoderName
        }
        throw CoreAISessionError.cannotSelectFunction(names)
    }
}

@available(iOS 27.0, macOS 27.0, *)
private struct OwnedCoreAIOutput: Sendable {
    let name: String
    let dataType: Int32
    let shape: [Int64]
    let bytes: Data

    init(name: String, array: NDArray) throws {
        self.name = name
        shape = array.shape.map(Int64.init)
        switch array.scalarType {
        case .float32:
            dataType = tensorFloat32
            let elementCount = array.shape.reduce(1, *)
            let view = array.view(as: Float.self)
            guard view.isContiguous else {
                throw CoreAISessionError.nonContiguousOutput(name)
            }
            bytes = view.withUnsafePointer { pointer, _, _ in
                Data(bytes: pointer, count: elementCount * MemoryLayout<Float>.stride)
            }
        case .int32:
            dataType = tensorInt32
            let elementCount = array.shape.reduce(1, *)
            let view = array.view(as: Int32.self)
            guard view.isContiguous else {
                throw CoreAISessionError.nonContiguousOutput(name)
            }
            bytes = view.withUnsafePointer { pointer, _, _ in
                Data(bytes: pointer, count: elementCount * MemoryLayout<Int32>.stride)
            }
        case .bool:
            dataType = tensorBool
            let elementCount = array.shape.reduce(1, *)
            let view = array.view(as: Bool.self)
            guard view.isContiguous else {
                throw CoreAISessionError.nonContiguousOutput(name)
            }
            bytes = view.withUnsafePointer { pointer, _, _ in
                Data(bytes: pointer, count: elementCount)
            }
        default:
            throw CoreAISessionError.unsupportedOutputType(name, String(describing: array.scalarType))
        }
    }
}

@available(iOS 27.0, macOS 27.0, *)
private func makeArray(
    from view: TensorViewLayout,
    expected: NDArrayDescriptor,
    name: String
) throws -> NDArray {
    guard view.rank >= 0, let shapePointer = view.shape else {
        throw CoreAISessionError.invalidInput(name)
    }
    let shape = (0..<Int(view.rank)).map { Int(shapePointer[$0]) }
    let count = shape.reduce(1, *)
    guard let data = view.data else {
        if count == 0 {
            return NDArray(shape: shape, scalarType: expected.scalarType)
        }
        throw CoreAISessionError.invalidInput(name)
    }

    switch expected.scalarType {
    case .float32:
        guard view.dataType == tensorFloat32 else {
            throw CoreAISessionError.inputType(name)
        }
        let values = Array(
            UnsafeBufferPointer(
                start: data.assumingMemoryBound(to: Float.self),
                count: count
            )
        )
        return NDArray(scalars: values, shape: shape)
    case .int32:
        if view.dataType == tensorInt64 {
            let source = UnsafeBufferPointer(
                start: data.assumingMemoryBound(to: Int64.self),
                count: count
            )
            return NDArray(scalars: source.map(Int32.init), shape: shape)
        }
        guard view.dataType == tensorInt32 else {
            throw CoreAISessionError.inputType(name)
        }
        let values = Array(
            UnsafeBufferPointer(
                start: data.assumingMemoryBound(to: Int32.self),
                count: count
            )
        )
        return NDArray(scalars: values, shape: shape)
    case .bool:
        guard view.dataType == tensorBool else {
            throw CoreAISessionError.inputType(name)
        }
        let source = UnsafeBufferPointer(
            start: data.assumingMemoryBound(to: UInt8.self),
            count: count
        )
        return NDArray(scalars: source.map { $0 != 0 }, shape: shape)
    default:
        throw CoreAISessionError.unsupportedInputType(name, String(describing: expected.scalarType))
    }
}

private final class AsyncResultBox<Value: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Result<Value, Error>?

    func store(_ value: Result<Value, Error>) {
        lock.lock()
        self.value = value
        lock.unlock()
    }

    func take() -> Result<Value, Error>? {
        lock.lock()
        defer { lock.unlock() }
        return value
    }
}

private func waitForAsync<Value: Sendable>(
    _ operation: @escaping @Sendable () async throws -> Value
) throws -> Value {
    let semaphore = DispatchSemaphore(value: 0)
    let box = AsyncResultBox<Value>()
    // This synchronous boundary is required by libscyllasband's C ABI. The
    // caller runs on a user-initiated serial queue, so explicitly match that
    // priority: detached tasks otherwise default to medium/Default QoS and
    // trigger a real priority inversion while this semaphore is waiting.
    Task.detached(priority: .high) {
        do {
            box.store(.success(try await operation()))
        } catch {
            box.store(.failure(error))
        }
        semaphore.signal()
    }
    semaphore.wait()
    guard let result = box.take() else {
        throw CoreAISessionError.missingAsyncResult
    }
    return try result.get()
}

private func reportCoreAIError(_ error: Error) {
    let message = "Core AI graph session: \(error)"
    message.withCString { setNativeGraphError($0) }
}

@_cdecl("scyllasband_coreai_session_create")
@available(iOS 27.0, macOS 27.0, *)
func scyllasbandCoreAISessionCreate(
    _ modelPath: UnsafePointer<CChar>?,
    _ accelerator: Int32,
    _ maxThreads: Int32
) -> UnsafeMutableRawPointer? {
    _ = maxThreads
    guard let modelPath else {
        reportCoreAIError(CoreAISessionError.missingModelPath)
        return nil
    }
    do {
        let session = try CoreAISession(
            modelPath: String(cString: modelPath),
            accelerator: accelerator
        )
        return Unmanaged.passRetained(session).toOpaque()
    } catch {
        reportCoreAIError(error)
        return nil
    }
}

@_cdecl("scyllasband_coreai_session_destroy")
@available(iOS 27.0, macOS 27.0, *)
func scyllasbandCoreAISessionDestroy(_ handle: UnsafeMutableRawPointer?) {
    guard let handle else { return }
    Unmanaged<CoreAISession>.fromOpaque(handle).release()
}

@_cdecl("scyllasband_coreai_session_has_signature")
@available(iOS 27.0, macOS 27.0, *)
func scyllasbandCoreAISessionHasSignature(
    _ handle: UnsafeRawPointer?,
    _ signature: UnsafePointer<CChar>?
) -> Int32 {
    guard handle != nil, let signature else { return 0 }
    return String(cString: signature) == "serving_default" ? 1 : 0
}

@_cdecl("scyllasband_coreai_session_run")
@available(iOS 27.0, macOS 27.0, *)
func scyllasbandCoreAISessionRun(
    _ handle: UnsafeRawPointer?,
    _ signature: UnsafePointer<CChar>?,
    _ inputs: UnsafeRawPointer?,
    _ inputCount: Int32,
    _ outTensors: UnsafeMutablePointer<UnsafeMutableRawPointer?>?,
    _ outTensorCount: UnsafeMutablePointer<Int32>?
) -> Int32 {
    _ = signature
    guard let handle, let inputs, inputCount >= 0, let outTensors, let outTensorCount else {
        reportCoreAIError(CoreAISessionError.invalidRunArguments)
        return -1
    }
    outTensors.pointee = nil
    outTensorCount.pointee = 0
    let session = Unmanaged<CoreAISession>.fromOpaque(handle).takeUnretainedValue()
    do {
        precondition(MemoryLayout<TensorViewLayout>.stride == 48)
        precondition(MemoryLayout<OwnedTensorLayout>.stride == 48)
        let inputBuffer = UnsafeBufferPointer(
            start: inputs.assumingMemoryBound(to: TensorViewLayout.self),
            count: Int(inputCount)
        )
        let outputs = try session.run(inputBuffer)
        guard let storage = calloc(outputs.count, MemoryLayout<OwnedTensorLayout>.stride)?
            .assumingMemoryBound(to: OwnedTensorLayout.self) else {
            throw CoreAISessionError.allocation
        }
        for (index, output) in outputs.enumerated() {
            storage[index].name = strdup(output.name)
            storage[index].dataType = output.dataType
            storage[index].rank = Int32(output.shape.count)
            storage[index].byteLength = UInt64(output.bytes.count)
            if !output.shape.isEmpty {
                let shapeBytes = output.shape.count * MemoryLayout<Int64>.stride
                storage[index].shape = malloc(shapeBytes)?.assumingMemoryBound(to: Int64.self)
                guard let shape = storage[index].shape else {
                    destroyNativeTensors(UnsafeMutableRawPointer(storage), Int32(outputs.count))
                    throw CoreAISessionError.allocation
                }
                _ = output.shape.withUnsafeBytes { source in
                    memcpy(shape, source.baseAddress!, shapeBytes)
                }
            }
            if !output.bytes.isEmpty {
                storage[index].data = malloc(output.bytes.count)
                guard let destination = storage[index].data else {
                    destroyNativeTensors(UnsafeMutableRawPointer(storage), Int32(outputs.count))
                    throw CoreAISessionError.allocation
                }
                output.bytes.copyBytes(to: destination.assumingMemoryBound(to: UInt8.self), count: output.bytes.count)
            }
        }
        outTensors.pointee = UnsafeMutableRawPointer(storage)
        outTensorCount.pointee = Int32(outputs.count)
        return 0
    } catch {
        reportCoreAIError(error)
        return -1
    }
}

@_cdecl("scyllasband_coreai_session_run_resized")
@available(iOS 27.0, macOS 27.0, *)
func scyllasbandCoreAISessionRunResized(
    _ handle: UnsafeRawPointer?,
    _ signature: UnsafePointer<CChar>?,
    _ inputs: UnsafeRawPointer?,
    _ inputCount: Int32,
    _ outTensors: UnsafeMutablePointer<UnsafeMutableRawPointer?>?,
    _ outTensorCount: UnsafeMutablePointer<Int32>?
) -> Int32 {
    scyllasbandCoreAISessionRun(
        handle,
        signature,
        inputs,
        inputCount,
        outTensors,
        outTensorCount
    )
}

private enum CoreAISessionError: Error, CustomStringConvertible {
    case allocation
    case cannotSelectFunction([String])
    case inputCount(function: String, expected: Int, actual: Int)
    case inputType(String)
    case invalidInput(String)
    case invalidRunArguments
    case missingAsyncResult
    case missingDescriptor(String)
    case missingFunction(String)
    case missingModelPath
    case missingOutput(String)
    case nonContiguousOutput(String)
    case unsupportedInput(String)
    case unsupportedInputType(String, String)
    case unsupportedOutputType(String, String)

    var description: String {
        switch self {
        case .allocation: return "memory allocation failed"
        case .cannotSelectFunction(let names): return "no fixed-shape function matches inputs; available=\(names)"
        case .inputCount(let function, let expected, let actual):
            return "\(function) expects \(expected) inputs, received \(actual)"
        case .inputType(let name): return "input \(name) has an incompatible host type"
        case .invalidInput(let name): return "input \(name) has invalid storage or shape"
        case .invalidRunArguments: return "run arguments are invalid"
        case .missingAsyncResult: return "async operation completed without a result"
        case .missingDescriptor(let name): return "function descriptor is missing for \(name)"
        case .missingFunction(let name): return "function could not be loaded: \(name)"
        case .missingModelPath: return "model path is required"
        case .missingOutput(let name): return "output is missing: \(name)"
        case .nonContiguousOutput(let name): return "output is not contiguous: \(name)"
        case .unsupportedInput(let name): return "input is not an NDArray: \(name)"
        case .unsupportedInputType(let name, let type): return "unsupported input type \(type) for \(name)"
        case .unsupportedOutputType(let name, let type): return "unsupported output type \(type) for \(name)"
        }
    }
}

#else  // !canImport(CoreAI)

private func reportCoreAIUnavailable() {
    let message = "Core AI graph session: CoreAI.framework is unavailable in this build (simulator or pre-27 SDK); use an ONNX bundle."
    message.withCString { setNativeGraphError($0) }
}

@_cdecl("scyllasband_coreai_session_create")
func scyllasbandCoreAISessionCreate(
    _ modelPath: UnsafePointer<CChar>?,
    _ accelerator: Int32,
    _ maxThreads: Int32
) -> UnsafeMutableRawPointer? {
    reportCoreAIUnavailable()
    return nil
}

@_cdecl("scyllasband_coreai_session_destroy")
func scyllasbandCoreAISessionDestroy(_ handle: UnsafeMutableRawPointer?) {}

@_cdecl("scyllasband_coreai_session_has_signature")
func scyllasbandCoreAISessionHasSignature(
    _ handle: UnsafeRawPointer?,
    _ signature: UnsafePointer<CChar>?
) -> Int32 {
    return 0
}

@_cdecl("scyllasband_coreai_session_run")
func scyllasbandCoreAISessionRun(
    _ handle: UnsafeRawPointer?,
    _ signature: UnsafePointer<CChar>?,
    _ inputs: UnsafeRawPointer?,
    _ inputCount: Int32,
    _ outTensors: UnsafeMutablePointer<UnsafeMutableRawPointer?>?,
    _ outTensorCount: UnsafeMutablePointer<Int32>?
) -> Int32 {
    reportCoreAIUnavailable()
    return -1
}

@_cdecl("scyllasband_coreai_session_run_resized")
func scyllasbandCoreAISessionRunResized(
    _ handle: UnsafeRawPointer?,
    _ signature: UnsafePointer<CChar>?,
    _ inputs: UnsafeRawPointer?,
    _ inputCount: Int32,
    _ outTensors: UnsafeMutablePointer<UnsafeMutableRawPointer?>?,
    _ outTensorCount: UnsafeMutablePointer<Int32>?
) -> Int32 {
    reportCoreAIUnavailable()
    return -1
}

#endif  // canImport(CoreAI)
