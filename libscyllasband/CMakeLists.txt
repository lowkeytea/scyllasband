cmake_minimum_required(VERSION 3.20)
project(libscyllasband LANGUAGES CXX)

option(SCYLLASBAND_ENABLE_LITERT "Build LiteRT backend support" OFF)
option(SCYLLASBAND_ENABLE_COREML "Build CoreML backend support" OFF)
option(SCYLLASBAND_ENABLE_ONNX "Build ONNX Runtime backend support" OFF)
option(SCYLLASBAND_BUILD_TOOLS "Build native Scylla's Band command-line tools" ON)
option(SCYLLASBAND_BUILD_TESTS "Build native Scylla's Band tests" ON)
if((SCYLLASBAND_ENABLE_LITERT) AND SCYLLASBAND_ENABLE_ONNX)
    message(FATAL_ERROR "LiteRT and ONNX Runtime cannot be enabled in the same libscyllasband build.")
endif()

set(SCYLLASBAND_LITERT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/litert" CACHE PATH "Repo-local LiteRT staging root")
set(SCYLLASBAND_LITERT_INCLUDE_DIR "" CACHE PATH "LiteRT include directory")
set(SCYLLASBAND_LITERT_LIBRARY "" CACHE FILEPATH "Path to libLiteRt shared library")
set(SCYLLASBAND_LITERT_LIB_DIR "" CACHE PATH "Directory containing libLiteRt")
set(SCYLLASBAND_ONNXRUNTIME_INCLUDE_DIR "" CACHE PATH "ONNX Runtime include directory")
set(SCYLLASBAND_ONNXRUNTIME_LIBRARY "" CACHE FILEPATH "Path to the ONNX Runtime library")

set(_scyllasband_litert_stage_platform_fallbacks)
if(ANDROID)
    if(ANDROID_ABI STREQUAL "arm64-v8a")
        set(_scyllasband_litert_stage_platform "android-arm64")
    elseif(ANDROID_ABI STREQUAL "x86_64")
        set(_scyllasband_litert_stage_platform "android-x86_64")
    else()
        set(_scyllasband_litert_stage_platform "android-${ANDROID_ABI}")
    endif()
    set(_scyllasband_litert_lib_names libLiteRt.so)
elseif(APPLE)
    string(TOLOWER "${CMAKE_SYSTEM_NAME}" _scyllasband_apple_system_name)
    string(TOLOWER "${CMAKE_OSX_SYSROOT}" _scyllasband_apple_sysroot)
    if(_scyllasband_apple_sysroot MATCHES "iphonesimulator")
        set(_scyllasband_litert_stage_platform "ios-sim-arm64")
    elseif(_scyllasband_apple_system_name STREQUAL "ios" OR _scyllasband_apple_sysroot MATCHES "iphoneos")
        set(_scyllasband_litert_stage_platform "ios-arm64")
    else()
        set(_scyllasband_litert_stage_platform "macos-arm64")
        set(_scyllasband_litert_stage_platform_fallbacks macos-universal)
    endif()
    set(_scyllasband_litert_lib_names libLiteRt.dylib)
elseif(UNIX)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
        set(_scyllasband_litert_stage_platform "linux-arm64")
    else()
        set(_scyllasband_litert_stage_platform "linux-x86_64")
    endif()
    set(_scyllasband_litert_lib_names libLiteRt.so)
elseif(WIN32)
    set(_scyllasband_litert_stage_platform "windows-x86_64")
    set(_scyllasband_litert_lib_names libLiteRt.dll LiteRt.dll)
else()
    set(_scyllasband_litert_stage_platform "${CMAKE_SYSTEM_NAME}")
    set(_scyllasband_litert_lib_names libLiteRt.so libLiteRt.dylib libLiteRt.dll LiteRt.dll)
endif()

if(NOT SCYLLASBAND_LITERT_INCLUDE_DIR)
    set(_scyllasband_litert_default_include "${SCYLLASBAND_LITERT_ROOT}/include")
    if(EXISTS "${_scyllasband_litert_default_include}/litert/c/litert_compiled_model.h")
        set(SCYLLASBAND_LITERT_INCLUDE_DIR "${_scyllasband_litert_default_include}" CACHE PATH "LiteRT include directory" FORCE)
    endif()
endif()

if(NOT SCYLLASBAND_LITERT_LIBRARY)
    set(_scyllasband_litert_search_dirs)
    if(SCYLLASBAND_LITERT_LIB_DIR)
        if(ANDROID)
            list(APPEND _scyllasband_litert_search_dirs "${SCYLLASBAND_LITERT_LIB_DIR}/${ANDROID_ABI}")
        endif()
        list(APPEND _scyllasband_litert_search_dirs "${SCYLLASBAND_LITERT_LIB_DIR}")
    endif()
    if(SCYLLASBAND_LITERT_ROOT)
        list(APPEND _scyllasband_litert_search_dirs "${SCYLLASBAND_LITERT_ROOT}/lib/${_scyllasband_litert_stage_platform}")
        foreach(_scyllasband_stage_platform IN LISTS _scyllasband_litert_stage_platform_fallbacks)
            list(APPEND _scyllasband_litert_search_dirs "${SCYLLASBAND_LITERT_ROOT}/lib/${_scyllasband_stage_platform}")
        endforeach()
    endif()
    foreach(_scyllasband_dir IN LISTS _scyllasband_litert_search_dirs)
        foreach(_scyllasband_name IN LISTS _scyllasband_litert_lib_names)
            if(EXISTS "${_scyllasband_dir}/${_scyllasband_name}")
                set(SCYLLASBAND_LITERT_LIBRARY "${_scyllasband_dir}/${_scyllasband_name}" CACHE FILEPATH "Path to libLiteRt shared library" FORCE)
                break()
            endif()
        endforeach()
        if(SCYLLASBAND_LITERT_LIBRARY)
            break()
        endif()
    endforeach()
endif()

add_library(scyllasband_native SHARED
    src/scyllasband.cpp
    src/scyllasband_backend.cpp
    src/scyllasband_bundle.cpp
    src/scyllasband_error.cpp
    src/scyllasband_execution_plan.cpp
    src/scyllasband_litert_session.cpp
    src/scyllasband_onnx_session.cpp
    src/scyllasband_request.cpp
    src/scyllasband_tensor.cpp
    src/scyllasband_text_normalizer.cpp
)
add_library(scyllasband::native ALIAS scyllasband_native)

target_compile_features(scyllasband_native PUBLIC cxx_std_17)
target_include_directories(scyllasband_native
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

add_library(scyllasband_litert_sdk INTERFACE)
if(SCYLLASBAND_LITERT_INCLUDE_DIR)
    target_include_directories(scyllasband_litert_sdk INTERFACE "${SCYLLASBAND_LITERT_INCLUDE_DIR}")
endif()
if(SCYLLASBAND_LITERT_LIBRARY)
    target_link_libraries(scyllasband_litert_sdk INTERFACE "${SCYLLASBAND_LITERT_LIBRARY}")
endif()
target_compile_definitions(scyllasband_litert_sdk INTERFACE
    CL_TARGET_OPENCL_VERSION=300
    LITERT_DISABLE_OPENCL_SUPPORT
)

if(SCYLLASBAND_ENABLE_LITERT)
    if(NOT SCYLLASBAND_LITERT_INCLUDE_DIR OR NOT EXISTS "${SCYLLASBAND_LITERT_INCLUDE_DIR}/litert/c/litert_compiled_model.h")
        message(FATAL_ERROR "LiteRT headers not found. Stage them under third_party/litert/include or pass SCYLLASBAND_LITERT_INCLUDE_DIR.")
    endif()
    if(NOT SCYLLASBAND_LITERT_LIBRARY)
        message(FATAL_ERROR "LiteRT shared library not found. Stage it under third_party/litert/lib/${_scyllasband_litert_stage_platform} or pass SCYLLASBAND_LITERT_LIBRARY.")
    endif()
    get_filename_component(_scyllasband_litert_library_dir "${SCYLLASBAND_LITERT_LIBRARY}" DIRECTORY)
    target_compile_definitions(scyllasband_native PRIVATE
        SCYLLASBAND_ENABLE_LITERT=1
        SCYLLASBAND_WITH_LITERT=1
        SCYLLASBAND_LITERT_LIBRARY_DIR="${_scyllasband_litert_library_dir}"
    )
    target_link_libraries(scyllasband_native PRIVATE scyllasband_litert_sdk)
    set_target_properties(scyllasband_native PROPERTIES
        BUILD_RPATH "${_scyllasband_litert_library_dir}"
        INSTALL_RPATH "${_scyllasband_litert_library_dir}"
    )
endif()

if(SCYLLASBAND_ENABLE_COREML)
    target_compile_definitions(scyllasband_native PRIVATE SCYLLASBAND_ENABLE_COREML=1)
endif()

if(SCYLLASBAND_ENABLE_ONNX)
    if(NOT SCYLLASBAND_ONNXRUNTIME_INCLUDE_DIR OR
       NOT EXISTS "${SCYLLASBAND_ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h")
        message(FATAL_ERROR "ONNX Runtime headers not found. Pass SCYLLASBAND_ONNXRUNTIME_INCLUDE_DIR.")
    endif()
    if(NOT SCYLLASBAND_ONNXRUNTIME_LIBRARY OR NOT EXISTS "${SCYLLASBAND_ONNXRUNTIME_LIBRARY}")
        message(FATAL_ERROR "ONNX Runtime library not found. Pass SCYLLASBAND_ONNXRUNTIME_LIBRARY.")
    endif()
    target_compile_definitions(scyllasband_native PRIVATE SCYLLASBAND_WITH_ONNXRUNTIME=1)
    target_include_directories(scyllasband_native PRIVATE "${SCYLLASBAND_ONNXRUNTIME_INCLUDE_DIR}")
    target_link_libraries(scyllasband_native PRIVATE "${SCYLLASBAND_ONNXRUNTIME_LIBRARY}")
    if(NOT ANDROID)
        get_filename_component(_scyllasband_onnxruntime_library_dir "${SCYLLASBAND_ONNXRUNTIME_LIBRARY}" DIRECTORY)
        set_target_properties(scyllasband_native PROPERTIES
            BUILD_RPATH "${_scyllasband_onnxruntime_library_dir}"
            INSTALL_RPATH "${_scyllasband_onnxruntime_library_dir}"
        )
    endif()
endif()

set_target_properties(scyllasband_native PROPERTIES
    OUTPUT_NAME scyllasband_native
    POSITION_INDEPENDENT_CODE ON
)

if(SCYLLASBAND_BUILD_TOOLS)
    add_executable(scyllasband_native_speak tools/scyllasband_native_speak.cpp)
    target_compile_features(scyllasband_native_speak PRIVATE cxx_std_17)
    target_link_libraries(scyllasband_native_speak PRIVATE scyllasband::native)
    if((SCYLLASBAND_ENABLE_LITERT) AND SCYLLASBAND_LITERT_LIBRARY)
        get_filename_component(_scyllasband_tool_litert_library_dir "${SCYLLASBAND_LITERT_LIBRARY}" DIRECTORY)
        set_target_properties(scyllasband_native_speak PROPERTIES
            BUILD_RPATH "${_scyllasband_tool_litert_library_dir}"
            INSTALL_RPATH "${_scyllasband_tool_litert_library_dir}"
        )
    endif()
endif()

if(SCYLLASBAND_BUILD_TESTS)
    include(CTest)
    if(BUILD_TESTING)
        add_executable(scyllasband_native_plan_test tests/scyllasband_native_plan_test.cpp)
        target_compile_features(scyllasband_native_plan_test PRIVATE cxx_std_17)
        target_include_directories(scyllasband_native_plan_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
        target_link_libraries(scyllasband_native_plan_test PRIVATE scyllasband::native)
        add_test(NAME scyllasband_native_plan_test COMMAND scyllasband_native_plan_test)

        add_executable(scyllasband_request_affect_test tests/scyllasband_request_affect_test.cpp)
        target_compile_features(scyllasband_request_affect_test PRIVATE cxx_std_17)
        target_include_directories(scyllasband_request_affect_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
        target_link_libraries(scyllasband_request_affect_test PRIVATE scyllasband::native)
        add_test(NAME scyllasband_request_affect_test COMMAND scyllasband_request_affect_test)

        add_executable(scyllasband_boundary_policy_test tests/scyllasband_boundary_policy_test.cpp)
        target_compile_features(scyllasband_boundary_policy_test PRIVATE cxx_std_17)
        target_include_directories(scyllasband_boundary_policy_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
        add_test(NAME scyllasband_boundary_policy_test COMMAND scyllasband_boundary_policy_test)

        add_executable(scyllasband_target_bucket_policy_test tests/scyllasband_target_bucket_policy_test.cpp)
        target_compile_features(scyllasband_target_bucket_policy_test PRIVATE cxx_std_17)
        target_include_directories(scyllasband_target_bucket_policy_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
        add_test(NAME scyllasband_target_bucket_policy_test COMMAND scyllasband_target_bucket_policy_test)

        add_executable(scyllasband_session_cache_policy_test tests/scyllasband_session_cache_policy_test.cpp)
        target_compile_features(scyllasband_session_cache_policy_test PRIVATE cxx_std_17)
        target_include_directories(scyllasband_session_cache_policy_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
        add_test(NAME scyllasband_session_cache_policy_test COMMAND scyllasband_session_cache_policy_test)

        add_executable(
            scyllasband_text_normalizer_test
            tests/scyllasband_text_normalizer_test.cpp
            src/scyllasband_text_normalizer.cpp
        )
        target_compile_features(scyllasband_text_normalizer_test PRIVATE cxx_std_17)
        target_include_directories(scyllasband_text_normalizer_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
        add_test(NAME scyllasband_text_normalizer_test COMMAND scyllasband_text_normalizer_test)

        add_executable(scyllasband_overflow_plan_test tests/scyllasband_overflow_plan_test.cpp)
        target_compile_features(scyllasband_overflow_plan_test PRIVATE cxx_std_17)
        target_include_directories(scyllasband_overflow_plan_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
        target_link_libraries(scyllasband_overflow_plan_test PRIVATE scyllasband::native)
        add_test(NAME scyllasband_overflow_plan_test COMMAND scyllasband_overflow_plan_test)

        add_executable(
            scyllasband_vector_timing_contract_test
            tests/scyllasband_vector_timing_contract_test.cpp
        )
        target_compile_features(scyllasband_vector_timing_contract_test PRIVATE cxx_std_17)
        target_include_directories(
            scyllasband_vector_timing_contract_test
            PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
        )
        target_link_libraries(
            scyllasband_vector_timing_contract_test
            PRIVATE scyllasband::native
        )
        add_test(
            NAME scyllasband_vector_timing_contract_test
            COMMAND scyllasband_vector_timing_contract_test
        )

        add_executable(
            scyllasband_duration_hierarchy_contract_test
            tests/scyllasband_duration_hierarchy_contract_test.cpp
        )
        target_compile_features(
            scyllasband_duration_hierarchy_contract_test PRIVATE cxx_std_17
        )
        target_include_directories(
            scyllasband_duration_hierarchy_contract_test
            PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
        )
        target_link_libraries(
            scyllasband_duration_hierarchy_contract_test PRIVATE scyllasband::native
        )
        add_test(
            NAME scyllasband_duration_hierarchy_contract_test
            COMMAND scyllasband_duration_hierarchy_contract_test
        )

        find_package(Python3 COMPONENTS Interpreter QUIET)
        if(Python3_Interpreter_FOUND)
            add_test(
                NAME scyllasband_plan_parity_test
                COMMAND "${CMAKE_COMMAND}" -E env
                    "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/.."
                    "SCYLLASBAND_NATIVE_LIBRARY=$<TARGET_FILE:scyllasband_native>"
                    "${Python3_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/scyllasband_plan_parity_test.py"
            )
        endif()
    endif()
endif()
