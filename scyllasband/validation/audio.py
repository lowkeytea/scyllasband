"""Dependency-light WAV I/O and signal-health measurements."""

from __future__ import annotations

from array import array
import hashlib
import math
import os
from pathlib import Path
import sys
import wave
from typing import Iterable


def write_wav(path: str | Path, audio: Iterable[float], sample_rate: int) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    values = array("h")
    for item in audio:
        value = float(item)
        if not math.isfinite(value):
            value = 0.0
        value = max(-1.0, min(1.0, value))
        values.append(int(round(value * 32767.0)))
    if sys.byteorder != "little":
        values.byteswap()
    with wave.open(str(temporary), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(values.tobytes())
    os.replace(temporary, destination)
    return destination


def read_wav(path: str | Path) -> tuple[list[float], int]:
    source = Path(path)
    with wave.open(str(source), "rb") as handle:
        channels = int(handle.getnchannels())
        width = int(handle.getsampwidth())
        sample_rate = int(handle.getframerate())
        frames = int(handle.getnframes())
        payload = handle.readframes(frames)
    if channels != 1 or width != 2:
        raise ValueError(
            f"Validation WAV must be mono signed-16 PCM, got channels={channels}, width={width}: {source}"
        )
    values = array("h")
    values.frombytes(payload)
    if sys.byteorder != "little":
        values.byteswap()
    return [float(item) / 32768.0 for item in values], sample_rate


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audio_metrics(audio: Iterable[float], sample_rate: int) -> dict[str, float | int | bool]:
    values = [float(item) for item in audio]
    finite = [item for item in values if math.isfinite(item)]
    invalid = len(values) - len(finite)
    if not finite:
        return {
            "sample_rate": int(sample_rate),
            "sample_count": len(values),
            "duration_seconds": 0.0,
            "finite": False,
            "invalid_sample_count": invalid,
            "peak": 0.0,
            "rms": 0.0,
            "dc_offset": 0.0,
            "hard_clipped_sample_count": 0,
            "near_clipped_sample_count": 0,
            "silence_fraction": 1.0,
        }
    peak = max(abs(item) for item in finite)
    rms = math.sqrt(sum(item * item for item in finite) / len(finite))
    dc = sum(finite) / len(finite)
    hard_clipped = sum(abs(item) >= 0.9999 for item in finite)
    near_clipped = sum(abs(item) >= 0.98 for item in finite)
    silence = sum(abs(item) <= 1e-4 for item in finite)
    return {
        "sample_rate": int(sample_rate),
        "sample_count": len(values),
        "duration_seconds": len(values) / float(sample_rate) if sample_rate > 0 else 0.0,
        "finite": invalid == 0,
        "invalid_sample_count": invalid,
        "peak": peak,
        "rms": rms,
        "dc_offset": dc,
        "hard_clipped_sample_count": hard_clipped,
        "near_clipped_sample_count": near_clipped,
        "silence_fraction": silence / float(len(finite)),
    }
