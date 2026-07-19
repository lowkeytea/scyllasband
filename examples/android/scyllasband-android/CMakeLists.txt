cmake_minimum_required(VERSION 3.20)
project(scyllasband_android LANGUAGES CXX)

if(NOT DEFINED SCYLLASBAND_ANDROID_ONNXRUNTIME_ROOT)
    message(FATAL_ERROR "SCYLLASBAND_ANDROID_ONNXRUNTIME_ROOT is required")
endif()
set(
    SCYLLASBAND_ONNXRUNTIME_LIBRARY
    "${SCYLLASBAND_ANDROID_ONNXRUNTIME_ROOT}/jni/${ANDROID_ABI}/libonnxruntime.so"
    CACHE FILEPATH "Android ONNX Runtime library" FORCE
)

add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../libscyllasband
    ${CMAKE_CURRENT_BINARY_DIR}/libscyllasband
)

add_library(scyllasband_android_jni SHARED
    src/main/cpp/scyllasband_android_jni.cpp
)

target_compile_features(scyllasband_android_jni PRIVATE cxx_std_17)
target_link_libraries(scyllasband_android_jni PRIVATE scyllasband::native android log)
