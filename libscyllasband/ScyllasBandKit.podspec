Pod::Spec.new do |spec|
  spec.name = 'ScyllasBandKit'
  spec.version = '2.0.0'
  spec.summary = "Scylla's Band text-to-speech for iOS on Core ML and Core AI."
  spec.description = <<-DESC
    ScyllasBandKit packages libscyllasband with its Apple graph backend: Core ML bundles on iOS 18 and later and
    Core AI bundles on iOS 27 and later. A small Objective-C API (usable from Swift) loads a bundle, warms it and
    streams long-form speech as mono Float32 PCM, one sentence at a time.
  DESC
  spec.homepage = 'https://github.com/lowkeytea/scyllasband'
  spec.license = { :type => 'Apache-2.0' }
  spec.author = { "Scylla's Band contributors" => 'opensource@example.com' }
  spec.source = { :git => 'https://github.com/lowkeytea/scyllasband.git', :tag => "v#{spec.version}" }

  spec.ios.deployment_target = '18.0'
  spec.static_framework = true
  spec.requires_arc = true
  spec.module_name = 'ScyllasBandKit'
  spec.swift_version = '5.9'

  spec.source_files = [
    'include/scyllasband.h',
    'src/*.{h,cpp,mm,inc}',
    'apple/Sources/**/*.{h,mm,swift}',
  ]
  # One graph backend per build: this pod uses Core ML and Core AI (src/scyllasband_graph_apple.mm).
  spec.exclude_files = ['src/scyllasband_graph_litert.cpp', 'src/scyllasband_graph_onnx.cpp']
  spec.public_header_files = 'apple/Sources/include/ScyllasBandKit.h'
  spec.private_header_files = ['include/scyllasband.h', 'src/*.h']
  spec.header_mappings_dir = 'apple/Sources/include'
  spec.frameworks = 'Foundation', 'CoreML'
  spec.libraries = 'c++'
  # Core AI (iOS 27 and later) is weak-linked, so the app still launches on iOS 18-26 and runs Core ML bundles there.
  # Only the device SDK ships CoreAI.framework; Simulator builds use Core ML alone.
  spec.user_target_xcconfig = { 'OTHER_LDFLAGS[sdk=iphoneos*]' => '$(inherited) -weak_framework CoreAI' }

  spec.pod_target_xcconfig = {
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17',
    'CLANG_WARN_DOCUMENTATION_COMMENTS' => 'NO',
    'GCC_PREPROCESSOR_DEFINITIONS' => '$(inherited) SCYLLASBAND_WITH_APPLE=1 SCYLLASBAND_BUILDING=1',
    # Same float evaluation order as the other builds (no fused multiply-add contraction).
    'OTHER_CPLUSPLUSFLAGS' => '$(inherited) -ffp-contract=off',
  }
end
