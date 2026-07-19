#include "scyllasband_error.h"

namespace scyllasband_detail {
namespace {
thread_local std::string g_last_error;
}  // namespace

void set_error(const std::string& message) {
    g_last_error = message;
}

const char* last_error() {
    return g_last_error.c_str();
}

void clear_error() {
    g_last_error.clear();
}

}  // namespace scyllasband_detail

extern "C" {

const char* scyllasband_last_error(void) {
    return scyllasband_detail::last_error();
}

void scyllasband_clear_error(void) {
    scyllasband_detail::clear_error();
}

}  // extern "C"
