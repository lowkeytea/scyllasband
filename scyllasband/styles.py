"""Delivery styles: an emotion and an intensity, placed per voice.

A bundle with a ``styles`` control (manifest ``controls.styles``) is driven by two choices instead of numeric delivery
values: an emotion (neutral, joy, sadness, ...; the bundle lists its own) and an intensity (less, normal or more). Each
voice has its own preset delivery for every pair, taken from that voice's training data, so every choice is something
the voice is known to do well. Nothing chosen means neutral at normal intensity: a fixed delivery, never a random one.

Requests accept "joy", "more", "joy,more", "emotion=joy,intensity=more" or {"emotion": "joy", "intensity": "more"};
whatever is left out takes the default.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

INTENSITIES = ("less", "normal", "more")
STYLE_KEYS = ("emotion", "intensity")
# Emotion names any styles bundle may use; a bundle offers the ones in its manifest.
KNOWN_EMOTIONS = ("neutral", "joy", "sadness", "anger", "frustration", "fear", "hope", "relief", "disgust", "caring",
                  "sarcasm", "pride", "confusion", "surprise")


def is_style_request(value: Any) -> bool:
    """True for an emotion / intensity request, False for numeric delivery (or nothing)."""
    if value is None:
        return False
    if isinstance(value, Mapping):
        keys = set(value) - {"kind"}
        return value.get("kind") == "style" or (bool(keys) and keys <= set(STYLE_KEYS))
    if isinstance(value, str):
        parts = [p.strip().lower() for p in value.split(",") if p.strip()]
        if not parts:
            return False
        for part in parts:
            key, sep, raw = part.partition("=")
            if sep:
                if key not in STYLE_KEYS:
                    return False
            elif part not in INTENSITIES and part not in KNOWN_EMOTIONS:
                return False
        return True
    return False


def parse_style(value: Any) -> dict[str, str | None]:
    """{"emotion": name or None, "intensity": level or None}; None fields take the bundle default."""
    style: dict[str, str | None] = dict(emotion=None, intensity=None)
    if value is None:
        return style
    if isinstance(value, Mapping):
        items = [(str(k).strip().lower(), v) for k, v in value.items() if k != "kind"]   # "kind": the planner's tag
    else:
        items = []
        for part in (p.strip() for p in str(value).split(",") if p.strip()):
            key, sep, raw = part.partition("=")
            if sep:
                items.append((key.strip().lower(), raw))
            elif part.lower() in INTENSITIES:
                items.append(("intensity", part))
            else:
                items.append(("emotion", part))
    for key, raw in items:
        if key not in STYLE_KEYS:
            raise ValueError(f"A style takes an emotion and an intensity; got {key!r}")
        if style[key] is not None:
            raise ValueError(f"The style names {key} twice")
        text = None if raw is None else str(raw).strip().lower()
        if key == "intensity" and text is not None and text not in INTENSITIES:
            raise ValueError(f"Intensity must be one of {', '.join(INTENSITIES)}; got {raw!r}")
        style[key] = text or None
    return style


def style_spec(style: Mapping[str, Any]) -> str:
    return ",".join(f"{k}={style.get(k) or 'default'}" for k in STYLE_KEYS)


class Styles:
    """The ``controls.styles`` table of a bundle."""

    def __init__(self, control: Mapping[str, Any]) -> None:
        self.emotions = [str(e) for e in control["emotions"]]
        self.intensities = [str(i) for i in control.get("intensities", INTENSITIES)]
        default = control.get("default") or {}
        self.default_emotion = str(default.get("emotion", self.emotions[0]))
        self.default_intensity = str(default.get("intensity", "normal"))
        self.axes = [str(a) for a in control["axes"]]
        rows = control["voice_rows"]
        self.base_voices = int(rows["base_voices"])
        self.presets: dict[str, dict[str, dict[str, dict[str, float]]]] = control["presets"]
        if self.default_emotion != self.emotions[0]:
            raise ValueError("The default emotion must be the first emotion block (plain voice ids)")

    def resolve(self, voice: str, value: Any) -> dict[str, Any]:
        """The emotion, intensity and four delivery values for ``voice``."""
        style = parse_style(value)
        emotion = style["emotion"] or self.default_emotion
        intensity = style["intensity"] or self.default_intensity
        if emotion not in self.emotions:
            raise ValueError(f"Emotion {emotion!r} is not available; choose one of {', '.join(self.emotions)}")
        if intensity not in self.intensities:
            raise ValueError(f"Intensity must be one of {', '.join(self.intensities)}; got {intensity!r}")
        if voice not in self.presets:
            raise ValueError(f"Voice {voice!r} has no styles in this bundle")
        values = self.presets[voice][emotion][intensity]
        return dict(emotion=emotion, intensity=intensity, values={a: float(values[a]) for a in self.axes})

    def row(self, voice_index: int, emotion: str) -> int:
        """The duration / flow voice row for (voice, emotion); the decoder keeps the plain voice index."""
        if not 0 <= int(voice_index) < self.base_voices:
            raise ValueError(f"Voice index {voice_index} is outside the {self.base_voices} voices of this bundle")
        return int(voice_index) + self.base_voices * self.emotions.index(emotion)

    def tensors(self, resolved: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Graph inputs: the four values on the graphs' (value - 2) / 4 scale, all present; whisper always off."""
        from .delivery import DELIVERY_AXES   # the graphs' input order (whisper fifth)
        values = np.zeros((1, 5), dtype=np.float32)
        present = np.ones((1, 5), dtype=np.bool_)
        for index, axis in enumerate(DELIVERY_AXES):
            values[0, index] = (float(resolved["values"][axis]) - 2.0) / 4.0
        return values, present
