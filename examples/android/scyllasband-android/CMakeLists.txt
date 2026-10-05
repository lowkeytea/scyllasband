cmake_minimum_required(VERSION 3.20)
project(scyllasband_android LANGUAGES CXX)

# Gradle passes SCYLLASBAND_BACKEND (litert or onnx). LiteRT links the prebuilt libLiteRt.so staged under
# libscyllasband/third_party/litert/lib/<platform>/ (scripts/stage_litert_sdk.py). ONNX Runtime links the
# libonnxruntime.so from the extracted onnxruntime-android AAR; the headers are libscyllasband's own.
if(SCYLLASBAND_BACKEND STREQUAL "onnx")
    if(NOT DEFINED SCYLLASBAND_ANDROID_ONNXRUNTIME_ROOT)
        message(FATAL_ERROR "SCYLLASBAND_ANDROID_ONNXRUNTIME_ROOT is required for the onnx backend")
    endif()
    set(
        SCYLLASBAND_ONNXRUNTIME_LIBRARY
        "${SCYLLASBAND_ANDROID_ONNXRUNTIME_ROOT}/jni/${ANDROID_ABI}/libonnxruntime.so"
        CACHE FILEPATH "Android ONNX Runtime library" FORCE
    )
endif()

add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../libscyllasband
    ${CMAKE_CURRENT_BINARY_DIR}/libscyllasband
)

add_library(scyllasband_android_jni SHARED
    src/main/cpp/scyllasband_android_jni.cpp
)

target_compile_features(scyllasband_android_jni PRIVATE cxx_std_17)
target_link_libraries(scyllasband_android_jni PRIVATE scyllasband::native android log)
