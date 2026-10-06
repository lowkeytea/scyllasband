"""Delivery requests: energy, tension, valence and assertiveness on 0-4 (2 = neutral).

The graphs keep a fifth delivery input from training (whisper); it is always sent as off."""
from __future__ import annotations

import math
from typing import Any, Mapping

DELIVERY_AXES = ("energy", "tension", "valence", "assertiveness")


def resolve_delivery(value: Mapping[str, Any] | str | None = None) -> dict[str, Any]:
    """Resolve defaults without conflating explicit neutral with omitted axes."""
    defaults = dict.fromkeys(DELIVERY_AXES, 2.0)
    if value is None:
        return defaults
    if isinstance(value, str):
        if value.strip().lower() == "auto":
            return dict.fromkeys(DELIVERY_AXES)
        if value.strip().lower() in ("", "neutral"):
            return defaults
        fields: dict[str, Any] = {}
        for part in value.split(","):
            key, sep, raw = part.strip().partition("=")
            if not sep or key in fields:
                raise ValueError("Delivery must contain unique axis=value entries or 'auto'")
            fields[key] = raw.strip()
        value = fields
    if isinstance(value, Mapping) and "whisper" in value:
        raise ValueError("Whisper is not supported; remove whisper from the delivery")
    if not isinstance(value, Mapping) or set(value) - set(DELIVERY_AXES):
        raise ValueError(f"Delivery accepts only {', '.join(DELIVERY_AXES)}")
    result = {**defaults, **value}
    for axis in DELIVERY_AXES:
        raw = result[axis]
        if raw is None or (isinstance(raw, str) and raw.lower() == "auto"):
            result[axis] = None
            continue
        if isinstance(raw, bool):
            raise ValueError(f"Delivery {axis} must be a number in [0, 4] or auto")
        try:
            number = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Delivery {axis} must be a number in [0, 4] or auto") from exc
        if not math.isfinite(number) or not 0 <= number <= 4:
            raise ValueError(f"Delivery {axis} must be finite and within [0, 4]")
        result[axis] = number
    return result


def delivery_tensors(value: Mapping[str, Any] | str | None = None):
    import numpy as np
    resolved = resolve_delivery(value)
    values = np.zeros((1, 5), dtype=np.float32)
    present = np.zeros((1, 5), dtype=np.bool_)
    for index, axis in enumerate(DELIVERY_AXES):
        if resolved[axis] is not None:
            values[0, index] = (resolved[axis] - 2.0) / 4.0
            present[0, index] = True
    present[0, 4] = True   # whisper: always off
    return values, present, resolved


def delivery_spec(value: Mapping[str, Any] | str | None = None) -> str:
    return ",".join(f"{key}={'auto' if val is None else val}" for key, val in resolve_delivery(value).items())
