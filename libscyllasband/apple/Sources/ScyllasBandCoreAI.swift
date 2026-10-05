// Core AI graph sessions for libscyllasband (iOS 27 / macOS 27 and later).
//
// Core AI has only a Swift API, so this file exposes plain C entry points that src/scyllasband_graph_apple.mm
// calls. Tensors cross the boundary as parallel arrays of primitives; outputs come back as float32 values in a
// buffer the caller frees with scyllasband_coreai_free. Each asset is specialized once per compute unit through
// the system's persistent cache, so only the first load after installing a bundle or updating the OS is slow.

#if canImport(CoreAI)
import CoreAI
#endif
import Foundation

private func writeError(_ message: String, _ buffer: UnsafeMutablePointer<CChar>?, _ length: Int) {
    guard let buffer, length > 0 else { return }
    let bytes = Array(message.utf8.prefix(length - 1))
    bytes.withUnsafeBufferPointer { source in
        source.baseAddress.map { UnsafeMutableRawPointer(buffer).copyMemory(from: $0, byteCount: bytes.count) }
    }
    buffer[bytes.count] = 0
}

/// Runs an async operation from libscyllasband's synchronous graph calls (a serial worker thread, never the main thread).
private final class Blocking<Value>: @unchecked Sendable {
    private var result: Result<Value, Error>?
    private let done = DispatchSemaphore(value: 0)

    static func run(_ operation: @escaping @Sendable () async throws -> Value) throws -> Value {
        let box = Blocking<Value>()
        Task.detached(priority: .userInitiated) {
            do { box.result = .success(try await operation()) } catch { box.result = .failure(error) }
            box.done.signal()
        }
        box.done.wait()
        return try box.result!.get()
    }
}

#if canImport(CoreAI)

@available(iOS 27.0, macOS 27.0, *)
private final class CoreAIModel: @unchecked Sendable {
    let model: AIModel
    init(model: AIModel) { self.model = model }
}

@available(iOS 27.0, macOS 27.0, *)
private final class CoreAIFunction: @unchecked Sendable {
    let function: InferenceFunction
    let inputTypes: [String: NDArray.ScalarType]
    let output: String

    init(function: InferenceFunction) throws {
        self.function = function
        var types: [String: NDArray.ScalarType] = [:]
        for name in function.descriptor.inputNames {
            guard case .ndArray(let descriptor)? = function.descriptor.inputDescriptor(of: name) else {
                throw BridgeError.message("input \(name) is not a tensor")
            }
            types[name] = descriptor.scalarType
        }
        guard let output = function.descriptor.outputNames.first else { throw BridgeError.message("function has no output") }
        inputTypes = types
        self.output = output
    }
}

private enum BridgeError: Error, CustomStringConvertible {
    case message(String)
    var description: String { if case .message(let text) = self { return text }; return "" }
}

// Input dtypes from libscyllasband (DType): 0 float32, 1 int64, 2 bool (one byte).
@available(iOS 27.0, macOS 27.0, *)
private func makeArray(_ data: UnsafeRawPointer?, dtype: Int32, shape: [Int], as target: NDArray.ScalarType) throws -> NDArray {
    let count = shape.reduce(1, *)
    func values<T>(_ convert: (Double) -> T) -> [T] {
        guard let data, count > 0 else { return [] }
        switch dtype {
        case 0: return UnsafeBufferPointer(start: data.assumingMemoryBound(to: Float.self), count: count).map { convert(Double($0)) }
        case 1: return UnsafeBufferPointer(start: data.assumingMemoryBound(to: Int64.self), count: count).map { convert(Double($0)) }
        default: return UnsafeBufferPointer(start: data.assumingMemoryBound(to: UInt8.self), count: count).map { convert(Double($0)) }
        }
    }
    switch target {
    case .float32: return NDArray(scalars: values { Float($0) }, shape: shape)
    case .float16: return NDArray(scalars: values { Float16($0) }, shape: shape)
    case .int32: return NDArray(scalars: values { Int32($0) }, shape: shape)
    case .int64: return NDArray(scalars: values { Int64($0) }, shape: shape)
    case .bool: return NDArray(scalars: values { $0 != 0 }, shape: shape)
    default: throw BridgeError.message("unsupported input type \(target)")
    }
}

/// Float32 copy of `array` in row-major order, honouring its strides (Neural Engine outputs can be padded or interleaved).
@available(iOS 27.0, macOS 27.0, *)
private func floats(_ array: NDArray) throws -> [Float] {
    func gather<T: BitwiseCopyable>(_ type: T.Type, _ convert: (T) -> Float) -> [Float] {
        array.view(as: type).withUnsafePointer { pointer, shapeSpan, strideSpan in
            let shape = (0..<shapeSpan.count).map { shapeSpan[$0] }, strides = (0..<strideSpan.count).map { strideSpan[$0] }
            let count = shape.reduce(1, *)
            var out = [Float](repeating: 0, count: count)
            guard count > 0 else { return out }
            var index = [Int](repeating: 0, count: shape.count)
            for flat in 0..<count {
                var offset = 0
                for axis in 0..<shape.count { offset += index[axis] * strides[axis] }
                out[flat] = convert(pointer[offset])
                var axis = shape.count - 1
                while axis >= 0 {
                    index[axis] += 1
                    if index[axis] < shape[axis] { break }
                    index[axis] = 0
                    axis -= 1
                }
            }
            return out
        }
    }
    switch array.scalarType {
    case .float32: return gather(Float.self) { $0 }
    case .float16: return gather(Float16.self) { Float($0) }
    default: throw BridgeError.message("unsupported output type \(array.scalarType)")
    }
}

@available(iOS 27.0, macOS 27.0, *)
private func options(_ unit: Int32) -> SpecializationOptions {
    switch unit {   // ScyllasBandAccelerator: 0 cpu, 1 gpu, 2 auto, 3 neural engine
    case 0: return .cpuOnly
    case 1: return SpecializationOptions(preferredComputeUnitKind: .gpu)
    case 3: return SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
    default: return .default
    }
}

#endif

/// 1 when this build and OS can run Core AI assets.
@_cdecl("scyllasband_coreai_available")
public func scyllasbandCoreAIAvailable() -> Int32 {
    #if canImport(CoreAI)
    if #available(iOS 27.0, macOS 27.0, *) { return 1 }
    #endif
    return 0
}

/// Loads (specializing on first use, with the persistent system cache) one `.aimodel` asset.
@_cdecl("scyllasband_coreai_open")
public func scyllasbandCoreAIOpen(_ path: UnsafePointer<CChar>, _ unit: Int32, _ errorBuffer: UnsafeMutablePointer<CChar>?, _ errorLength: Int) -> UnsafeMutableRawPointer? {
    #if canImport(CoreAI)
    if #available(iOS 27.0, macOS 27.0, *) {
        let url = URL(fileURLWithPath: String(cString: path))
        let chosen = options(unit)
        do {
            let model = try Blocking<AIModel>.run {
                try await AIModel.specialize(contentsOf: url, options: chosen, cache: .default, cachePolicy: .persistent)
            }
            return Unmanaged.passRetained(CoreAIModel(model: model)).toOpaque()
        } catch {
            writeError("Core AI cannot load \(url.lastPathComponent): \(error)", errorBuffer, errorLength)
            return nil
        }
    }
    #endif
    writeError("Core AI needs iOS 27 or macOS 27", errorBuffer, errorLength)
    return nil
}

/// One inference function of an open asset (``name`` NULL: the asset's only function).
@_cdecl("scyllasband_coreai_function")
public func scyllasbandCoreAIFunction(_ model: UnsafeMutableRawPointer, _ name: UnsafePointer<CChar>?, _ errorBuffer: UnsafeMutablePointer<CChar>?, _ errorLength: Int) -> UnsafeMutableRawPointer? {
    #if canImport(CoreAI)
    if #available(iOS 27.0, macOS 27.0, *) {
        let asset = Unmanaged<CoreAIModel>.fromOpaque(model).takeUnretainedValue().model
        let names = asset.functionNames
        let functionName = name.map { String(cString: $0) } ?? (names.count == 1 ? names[0] : "main")
        do {
            guard let function = try asset.loadFunction(named: functionName) else {
                throw BridgeError.message("no function \(functionName) (has \(names.joined(separator: ", ")))")
            }
            return Unmanaged.passRetained(try CoreAIFunction(function: function)).toOpaque()
        } catch {
            writeError("Core AI function \(functionName): \(error)", errorBuffer, errorLength)
            return nil
        }
    }
    #endif
    writeError("Core AI needs iOS 27 or macOS 27", errorBuffer, errorLength)
    return nil
}

/// Runs a function. Inputs are parallel arrays: names, dtypes (0 float32, 1 int64, 2 bool), ranks, shapes
/// (concatenated) and data. On success *values (float32, free with scyllasband_coreai_free) and the shape of the
/// first output are set and 0 is returned.
@_cdecl("scyllasband_coreai_run")
public func scyllasbandCoreAIRun(_ handle: UnsafeMutableRawPointer, _ count: Int32, _ names: UnsafePointer<UnsafePointer<CChar>?>,
                                 _ dtypes: UnsafePointer<Int32>, _ ranks: UnsafePointer<Int32>, _ shapes: UnsafePointer<Int64>,
                                 _ data: UnsafePointer<UnsafeRawPointer?>, _ values: UnsafeMutablePointer<UnsafeMutablePointer<Float>?>,
                                 _ valueCount: UnsafeMutablePointer<Int64>, _ outShape: UnsafeMutablePointer<Int64>,
                                 _ outRank: UnsafeMutablePointer<Int32>, _ errorBuffer: UnsafeMutablePointer<CChar>?, _ errorLength: Int) -> Int32 {
    #if canImport(CoreAI)
    if #available(iOS 27.0, macOS 27.0, *) {
        let session = Unmanaged<CoreAIFunction>.fromOpaque(handle).takeUnretainedValue()
        do {
            var inputs: [String: NDArray] = [:]
            var offset = 0
            for i in 0..<Int(count) {
                let name = String(cString: names[i]!)
                let rank = Int(ranks[i])
                let shape = (0..<rank).map { Int(shapes[offset + $0]) }
                offset += rank
                guard let type = session.inputTypes[name] else { continue }   // the graph does not read this input
                inputs[name] = try makeArray(data[i], dtype: dtypes[i], shape: shape, as: type)
            }
            let request = inputs
            let result = try Blocking<(shape: [Int], values: [Float])>.run {
                var outputs = try await session.function.run(inputs: request)
                guard let value = outputs.remove(session.output), let array = value.ndArray else {
                    throw BridgeError.message("missing output \(session.output)")
                }
                return (array.shape, try floats(array))
            }
            guard result.shape.count <= 8 else { throw BridgeError.message("output rank \(result.shape.count) exceeds 8") }
            let buffer = UnsafeMutablePointer<Float>.allocate(capacity: max(1, result.values.count))
            result.values.withUnsafeBufferPointer { source in
                if let base = source.baseAddress { buffer.update(from: base, count: source.count) }
            }
            values.pointee = buffer
            valueCount.pointee = Int64(result.values.count)
            outRank.pointee = Int32(result.shape.count)
            for (axis, size) in result.shape.enumerated() { outShape[axis] = Int64(size) }
            return 0
        } catch {
            writeError("Core AI inference: \(error)", errorBuffer, errorLength)
            return 1
        }
    }
    #endif
    writeError("Core AI needs iOS 27 or macOS 27", errorBuffer, errorLength)
    return 1
}

@_cdecl("scyllasband_coreai_free")
public func scyllasbandCoreAIFree(_ values: UnsafeMutablePointer<Float>?) {
    values?.deallocate()
}

/// Releases a model or function handle.
@_cdecl("scyllasband_coreai_release")
public func scyllasbandCoreAIRelease(_ handle: UnsafeMutableRawPointer?) {
    guard let handle else { return }
    Unmanaged<AnyObject>.fromOpaque(handle).release()
}
