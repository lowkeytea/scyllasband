// What the C API files share: the runtime object, error mapping and the copies handed to callers.
#pragma once

#include "scyllasband.h"

#include "scyllasband_engine.h"
#include "scyllasband_test_api.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

struct ScyllasBandRuntime {
    std::unique_ptr<scyllasband::Engine> engine;
    std::string voices_json;
    std::mutex mutex;  // serializes graph work; calls that read only the engine's configuration do not take it
    ScyllasBandTestNoise test_noise = nullptr;
    void* test_noise_user = nullptr;
};

namespace scyllasband::abi {

// The calling thread's message for scyllasband_last_error().
std::string& last_error();

// Runs `fn` and maps what it throws to a status, recording the message.
template <typename Fn>
ScyllasBandStatus guarded(Fn&& fn) {
    try {
        fn();
        last_error().clear();
        return SCYLLASBAND_OK;
    } catch (const CancelledError& error) {
        last_error() = error.what();
        return SCYLLASBAND_ERROR_CANCELLED;
    } catch (const std::invalid_argument& error) {
        last_error() = error.what();
        return SCYLLASBAND_ERROR_INVALID_ARGUMENT;
    } catch (const std::exception& error) {
        last_error() = error.what();
        return SCYLLASBAND_ERROR_RUNTIME;
    } catch (...) {
        last_error() = "Unknown native error";
        return SCYLLASBAND_ERROR_RUNTIME;
    }
}

// malloc'd copies, freed by scyllasband_string_free / scyllasband_audio_free.
inline char* copy_string(const std::string& value) {
    char* out = static_cast<char*>(std::malloc(value.size() + 1));
    if (out == nullptr) throw std::bad_alloc();
    std::memcpy(out, value.c_str(), value.size() + 1);
    return out;
}

inline float* copy_floats(const std::vector<float>& values) {
    float* out = static_cast<float*>(std::malloc(std::max<std::size_t>(1, values.size()) * sizeof(float)));
    if (out == nullptr) throw std::bad_alloc();
    std::copy(values.begin(), values.end(), out);
    return out;
}

inline void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

}  // namespace scyllasband::abi
