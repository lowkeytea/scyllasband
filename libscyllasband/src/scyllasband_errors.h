// Errors shared by the engine and the planners (the C API maps them to its status codes).
#pragma once

#include <stdexcept>

namespace scyllasband {

// The sentence or phrase needs more latent frames (or phones) than the largest bucket holds.
class OverlongError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class CancelledError : public std::runtime_error {
public:
    CancelledError() : std::runtime_error("Synthesis was cancelled") {}
};

}  // namespace scyllasband
