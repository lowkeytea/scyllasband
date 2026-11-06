// Python's integer and rounding rules, which the singing planner follows exactly so its plans match singing.py.
#pragma once

#include <cmath>
#include <cstdint>

namespace scyllasband {

// a // b and a % b in Python: the quotient rounds down and the remainder takes the divisor's sign.
inline int64_t floor_div(int64_t a, int64_t b) {
    int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}

inline int64_t floor_mod(int64_t a, int64_t b) { return a - floor_div(a, b) * b; }

// int(round(x)): half to even (the default floating-point rounding mode).
inline int64_t round_half_even(double value) { return static_cast<int64_t>(std::nearbyint(value)); }

// round(x, digits), for metadata.
inline double round_to(double value, int digits) {
    const double scale = std::pow(10.0, digits);
    return std::nearbyint(value * scale) / scale;
}

}  // namespace scyllasband
