Pod::Spec.new do |spec|
  spec.name = 'ScyllasBandKit'
  spec.version = '1.0.0'
  spec.summary = "An iOS bridge for the native Scylla's Band speech runtime."
  spec.description = <<-DESC
    ScyllasBandKit packages libscyllasband and exposes a small, Swift-friendly
    Objective-C API for manifest discovery, runtime lifetime, warmup, streamed
    long-form synthesis, cancellation, and Float32 PCM delivery.
  DESC
  spec.homepage = 'https://github.com/lowkeylabs/scyllasband'
  spec.license = { :type => 'Apache-2.0' }
  spec.author = { 'Scylla\'s Band contributors' => 'opensource@example.com' }
  spec.source = {
    :git => 'https://github.com/lowkeytea/scyllasband.git',
    :tag => "v#{spec.version}",
  }

  spec.ios.deployment_target = '16.0'
  spec.static_framework = true
  spec.requires_arc = true
  spec.module_name = 'ScyllasBandKit'

  spec.source_files = [
    'include/scyllasband.h',
    'src/*.{h,cpp}',
    'apple/Sources/**/*.{h,mm}',
  ]
  # The ONNX session translation unit implements the graph-session ABI for
  # this pod; the LiteRT translation unit is intentionally empty in this build.
  spec.exclude_files = 'src/scyllasband_litert_session.cpp'
  spec.public_header_files = 'apple/Sources/include/ScyllasBandKit.h'
  spec.private_header_files = ['include/scyllasband.h', 'src/*.h']
  spec.header_mappings_dir = 'apple/Sources/include'
  spec.frameworks = 'Foundation'
  spec.dependency 'onnxruntime-c', '~> 1.21'

  spec.pod_target_xcconfig = {
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17',
    'CLANG_WARN_DOCUMENTATION_COMMENTS' => 'NO',
    'GCC_PREPROCESSOR_DEFINITIONS' => '$(inherited) SCYLLASBAND_WITH_ONNXRUNTIME=1',
    # onnxruntime-c exposes an XCFramework and a sibling C/C++ header folder.
    # CocoaPods links the former automatically but does not add the latter to
    # dependent source targets' header search paths.
    'HEADER_SEARCH_PATHS' => '$(inherited) "${PODS_ROOT}/onnxruntime-c/Headers"',
  }
end
