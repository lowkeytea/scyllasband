cmake_minimum_required(VERSION 3.20)
project(libscyllasband VERSION 2.0.0 LANGUAGES CXX)

set(SCYLLASBAND_BACKEND "litert" CACHE STRING "Graph runtime linked into libscyllasband: litert or onnx")
set_property(CACHE SCYLLASBAND_BACKEND PROPERTY STRINGS litert onnx)
option(SCYLLASBAND_BUILD_TOOLS "Build the scyllasband_speak command-line tool" ON)
option(SCYLLASBAND_BUILD_TESTS "Build the native unit tests" ON)

set(SCYLLASBAND_LITERT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/litert/include" CACHE PATH "LiteRT C API headers")
set(SCYLLASBAND_ONNXRUNTIME_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/include" CACHE PATH "ONNX Runtime C/C++ headers")

# Staged SDK platform directory: third_party/<sdk>/lib/<platform>/. The lookups below ignore CMAKE_FIND_ROOT_PATH,
# which cross-compiling toolchains (the Android NDK) restrict to their sysroot.
if(ANDROID)
    if(ANDROID_ABI STREQUAL "arm64-v8a")
        set(_scyllasband_platform "android-arm64")
    else()
        set(_scyllasband_platform "android-${ANDROID_ABI}")
    endif()
elseif(APPLE)
    if(CMAKE_OSX_SYSROOT MATCHES "iphonesimulator")
        set(_scyllasband_platform "ios-sim-arm64")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "iOS")
        set(_scyllasband_platform "ios-arm64")
    else()
        set(_scyllasband_platform "macos-arm64")
    endif()
elseif(WIN32)
    set(_scyllasband_platform "windows-x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    set(_scyllasband_platform "linux-arm64")
else()
    set(_scyllasband_platform "linux-x86_64")
endif()

if(SCYLLASBAND_BACKEND STREQUAL "litert")
    set(_scyllasband_backend_source src/scyllasband_graph_litert.cpp)
    # Pass SCYLLASBAND_LITERT_LIBRARY to use a libLiteRt outside the staged SDK.
    find_library(SCYLLASBAND_LITERT_LIBRARY NAMES LiteRt libLiteRt
        PATHS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/litert/lib/${_scyllasband_platform}" NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    if(NOT EXISTS "${SCYLLASBAND_LITERT_INCLUDE_DIR}/litert/c/litert_compiled_model.h")
        message(FATAL_ERROR "LiteRT headers not found in ${SCYLLASBAND_LITERT_INCLUDE_DIR}.")
    endif()
    if(NOT SCYLLASBAND_LITERT_LIBRARY)
        message(FATAL_ERROR "libLiteRt not found. Stage it under third_party/litert/lib/${_scyllasband_platform} "
                            "(scripts/stage_litert_sdk.py) or pass SCYLLASBAND_LITERT_LIBRARY.")
    endif()
    set(_scyllasband_sdk_include "${SCYLLASBAND_LITERT_INCLUDE_DIR}")
    set(_scyllasband_sdk_library "${SCYLLASBAND_LITERT_LIBRARY}")
    set(_scyllasband_backend_define SCYLLASBAND_WITH_LITERT=1)
elseif(SCYLLASBAND_BACKEND STREQUAL "onnx")
    set(_scyllasband_backend_source src/scyllasband_graph_onnx.cpp)
    # Pass SCYLLASBAND_ONNXRUNTIME_LIBRARY to use an ONNX Runtime outside the staged SDK.
    find_library(SCYLLASBAND_ONNXRUNTIME_LIBRARY NAMES onnxruntime
        PATHS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/lib/${_scyllasband_platform}" NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    if(NOT EXISTS "${SCYLLASBAND_ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h")
        message(FATAL_ERROR "ONNX Runtime headers not found in ${SCYLLASBAND_ONNXRUNTIME_INCLUDE_DIR}.")
    endif()
    if(NOT SCYLLASBAND_ONNXRUNTIME_LIBRARY)
        message(FATAL_ERROR "libonnxruntime not found. Stage it under third_party/onnxruntime/lib/${_scyllasband_platform} "
                            "or pass SCYLLASBAND_ONNXRUNTIME_LIBRARY.")
    endif()
    set(_scyllasband_sdk_include "${SCYLLASBAND_ONNXRUNTIME_INCLUDE_DIR}")
    set(_scyllasband_sdk_library "${SCYLLASBAND_ONNXRUNTIME_LIBRARY}")
    set(_scyllasband_backend_define SCYLLASBAND_WITH_ONNXRUNTIME=1)
else()
    message(FATAL_ERROR "SCYLLASBAND_BACKEND must be litert or onnx (got '${SCYLLASBAND_BACKEND}').")
endif()
get_filename_component(_scyllasband_sdk_library_dir "${_scyllasband_sdk_library}" DIRECTORY)
message(STATUS "libscyllasband backend: ${SCYLLASBAND_BACKEND} (${_scyllasband_sdk_library})")

# The engine and text frontend; the shared library and the unit tests both link it.
add_library(scyllasband_core STATIC
    src/scyllasband_engine.cpp
    src/scyllasband_events.cpp
    src/scyllasband_g2p.cpp
    src/scyllasband_json.cpp
    src/scyllasband_text.cpp
    src/scyllasband_unicode.cpp
    ${_scyllasband_backend_source}
)
target_compile_features(scyllasband_core PUBLIC cxx_std_17)
target_include_directories(scyllasband_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(scyllasband_core SYSTEM PRIVATE "${_scyllasband_sdk_include}")
target_compile_definitions(scyllasband_core PUBLIC ${_scyllasband_backend_define} SCYLLASBAND_BUILDING)
target_link_libraries(scyllasband_core PUBLIC "${_scyllasband_sdk_library}" ${CMAKE_DL_LIBS})
set_target_properties(scyllasband_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Keep float arithmetic in source order (no fused multiply-add) so results match across platforms.
    target_compile_options(scyllasband_core PUBLIC -ffp-contract=off)
endif()

add_library(scyllasband_native SHARED src/scyllasband.cpp)
add_library(scyllasband::native ALIAS scyllasband_native)
target_link_libraries(scyllasband_native PRIVATE scyllasband_core)
target_include_directories(scyllasband_native PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
set_target_properties(scyllasband_native PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)
if(NOT ANDROID)
    set_target_properties(scyllasband_native PROPERTIES
        BUILD_RPATH "${_scyllasband_sdk_library_dir}"
        INSTALL_RPATH "${_scyllasband_sdk_library_dir}"
    )
endif()

if(SCYLLASBAND_BUILD_TOOLS)
    add_executable(scyllasband_speak tools/scyllasband_speak.cpp)
    target_compile_features(scyllasband_speak PRIVATE cxx_std_17)
    target_link_libraries(scyllasband_speak PRIVATE scyllasband::native)
    set_target_properties(scyllasband_speak PROPERTIES BUILD_RPATH "${_scyllasband_sdk_library_dir}")
endif()

if(SCYLLASBAND_BUILD_TESTS)
    include(CTest)
    if(BUILD_TESTING)
        add_executable(scyllasband_unit_test tests/scyllasband_unit_test.cpp)
        target_link_libraries(scyllasband_unit_test PRIVATE scyllasband_core)
        set_target_properties(scyllasband_unit_test PROPERTIES BUILD_RPATH "${_scyllasband_sdk_library_dir}")
        foreach(_scyllasband_suite events durations context delivery sentences segments normalizer unicode)
            add_test(NAME scyllasband_${_scyllasband_suite}
                     COMMAND scyllasband_unit_test ${_scyllasband_suite} "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures")
        endforeach()

        # Native vs Python reference on a real bundle: configure with SCYLLASBAND_PARITY_BUNDLE (a bundle of this
        # build's flavor) and SCYLLASBAND_PARITY_PYTHON (an interpreter with numpy and onnxruntime or ai-edge-litert).
        set(SCYLLASBAND_PARITY_BUNDLE "" CACHE PATH "Bundle for the parity test")
        set(SCYLLASBAND_PARITY_PYTHON "" CACHE FILEPATH "Python for the parity test")
        if(SCYLLASBAND_PARITY_BUNDLE AND SCYLLASBAND_PARITY_PYTHON)
            add_test(NAME scyllasband_parity
                     COMMAND "${SCYLLASBAND_PARITY_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/parity_harness.py"
                             --library "$<TARGET_FILE:scyllasband_native>" --bundle "${SCYLLASBAND_PARITY_BUNDLE}")
            set_tests_properties(scyllasband_parity PROPERTIES
                ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/.."
                TIMEOUT 3600)
        endif()
    endif()
endif()
