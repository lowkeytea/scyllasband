"""Measured delivery requests, independent of legacy emotion/affect controls."""
from __future__ import annotations

import math
from typing import Any, Mapping

DELIVERY_SCHEMA = "scyllasband_measured_delivery_v1"
DELIVERY_RELEASE = "v2-measured-20260922"
DELIVERY_AXES = ("energy", "tension", "valence", "assertiveness")
DELIVERY_CHANNELS = (*DELIVERY_AXES, "whisper")


def delivery_enabled(manifest: Any) -> bool:
    return getattr(manifest, "controls", {}).get("graph_input_contract") == DELIVERY_SCHEMA


def resolve_delivery(value: Mapping[str, Any] | str | None = None) -> dict[str, Any]:
    """Resolve defaults without conflating explicit neutral with omitted axes."""
    defaults = dict.fromkeys(DELIVERY_AXES, 2.0)
    defaults["whisper"] = "off"
    if value is None:
        return defaults
    if isinstance(value, str):
        if value.strip().lower() == "auto":
            return {**dict.fromkeys(DELIVERY_AXES), "whisper": "auto"}
        if value.strip().lower() == "neutral":
            return defaults
        fields: dict[str, Any] = {}
        for part in value.split(","):
            key, sep, raw = part.strip().partition("=")
            if not sep or key in fields:
                raise ValueError("Delivery must contain unique axis=value entries or 'auto'")
            fields[key] = raw.strip()
        value = fields
    if not isinstance(value, Mapping) or set(value) - set(DELIVERY_CHANNELS):
        raise ValueError(f"Delivery accepts only {', '.join(DELIVERY_CHANNELS)}")
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
    whisper = result["whisper"]
    if isinstance(whisper, bool):
        whisper = "on" if whisper else "off"
    elif whisper is None:
        whisper = "auto"
    if whisper not in ("on", "off", "auto"):
        raise ValueError("Delivery whisper must be on, off, or auto")
    result["whisper"] = whisper
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
    if resolved["whisper"] != "auto":
        values[0, 4] = float(resolved["whisper"] == "on")
        present[0, 4] = True
    return values, present, resolved


def delivery_spec(value: Mapping[str, Any] | str | None = None) -> str:
    return ",".join(f"{key}={'auto' if val is None else val}" for key, val in resolve_delivery(value).items())


def validate_delivery_request(manifest: Any, request: Any) -> None:
    value = getattr(request, "delivery", None)
    if not delivery_enabled(manifest):
        if value is not None:
            raise ValueError("This bundle uses legacy emotion controls and does not support measured delivery")
        return
    if (getattr(request, "affect", None) is not None or getattr(request, "emotion", None)
            or getattr(request, "emotion_guidance", None)
            or float(getattr(request, "affect_guidance_scale", 1.0)) != 1.0
            or float(getattr(request, "emotion_embed_scale", 1.0)) != 1.0):
        raise ValueError("This measured v2 bundle uses delivery controls; emotion presets and emotion CFG are unsupported")
    resolve_delivery(value)
