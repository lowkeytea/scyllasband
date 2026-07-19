#pragma once

#include <string>

namespace scyllasband_detail {

void set_error(const std::string& message);
const char* last_error();
void clear_error();

}  // namespace scyllasband_detail
