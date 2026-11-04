"""Delivery controls: intensity, pace and pitch.

- intensity: how much the voice puts into a line, from calm (negative) to intense (positive), in the voice's own standard
  deviations; the emotion itself comes from the text.
- pace: how fast the voice speaks, words and pauses together, in the voice's own standard deviations (negative is slower).
- pitch: a shift of the voice's predicted pitch, in semitones.

0 is each voice's own average (a model may set other defaults per voice). Requests are clamped into the ranges the
model supports (manifest ``controls.delivery.ranges``, by default -2 to 2 for each).

A request is a mapping ({"intensity": 1, "pace": -0.5}), a string ("intensity=1,pace=-0.5"), or None / "default" for
the voice's defaults; whatever is left out takes the default.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

CONTROLS = ("intensity", "pace", "pitch")
DEFAULT_RANGES = {"intensity": (-2.0, 2.0), "pace": (-2.0, 2.0), "pitch": (-2.0, 2.0)}
_OLD_KEYS = frozenset({"energy", "tension", "valence", "assertiveness", "whisper", "emotion", "kind"})


def resolve_delivery(value: Mapping[str, Any] | str | None = None,
                     ranges: Mapping[str, Any] | None = None) -> dict[str, float | None]:
    """{intensity, pace, pitch}: a number clamped into its range, or None for the voice's default."""
    result: dict[str, float | None] = dict.fromkeys(CONTROLS)
    if value is None:
        return result
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("", "default", "neutral"):
            return result
        fields: dict[str, Any] = {}
        for part in text.split(","):
            key, sep, raw = part.strip().partition("=")
            if not sep or key in fields:
                raise ValueError("Delivery must be unique control=value entries, e.g. intensity=1,pace=-0.5,pitch=0")
            fields[key] = raw.strip()
        value = fields
    if not isinstance(value, Mapping):
        raise ValueError("Delivery must be a mapping, a control=value string or None")
    keys = set(value)
    if keys & _OLD_KEYS:
        raise ValueError("This model takes intensity, pace and pitch (energy, tension, valence, assertiveness and emotion styles "
                         "belonged to the v2 models)")
    if keys - set(CONTROLS):
        raise ValueError(f"Delivery accepts only {', '.join(CONTROLS)}; got {', '.join(sorted(keys - set(CONTROLS)))}")
    limits = {**DEFAULT_RANGES, **{k: tuple(v) for k, v in (ranges or {}).items() if k in CONTROLS}}
    for key in CONTROLS:
        raw = value.get(key)
        if raw is None or (isinstance(raw, str) and raw.strip().lower() in ("", "default", "auto")):
            continue
        if isinstance(raw, bool):
            raise ValueError(f"Delivery {key} must be a number")
        try:
            number = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Delivery {key} must be a number") from exc
        if not math.isfinite(number):
            raise ValueError(f"Delivery {key} must be finite")
        low, high = limits[key]
        result[key] = min(max(number, float(low)), float(high))
    return result


def with_defaults(request: Mapping[str, float | None], defaults: Mapping[str, Any] | None) -> dict[str, float]:
    """The request with the voice's defaults (0 when the model gives none) filled in."""
    defaults = defaults or {}
    return {key: float(request[key]) if request.get(key) is not None else float(defaults.get(key, 0.0)) for key in CONTROLS}


def delivery_tensors(resolved: Mapping[str, float], *, z_scale: float = 5.0, z_limit: float = 2.5, slots: int = 5):
    """[1, slots] float32 values and bool presence: slot 0 intensity, slot 1 pace, each clip(z, -z_limit, z_limit) / z_scale."""
    import numpy as np
    values = np.zeros((1, slots), dtype=np.float32)
    present = np.zeros((1, slots), dtype=np.bool_)
    for index, key in enumerate(("intensity", "pace")):
        values[0, index] = min(max(float(resolved[key]), -z_limit), z_limit) / z_scale
        present[0, index] = True
    return values, present


def delivery_spec(value: Mapping[str, Any] | str | None = None) -> str:
    resolved = resolve_delivery(value)
    return ",".join(f"{key}={'default' if val is None else repr(float(val))}" for key, val in resolved.items())
