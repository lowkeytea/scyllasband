"""Spoken-form tables, one JSON file per normalizer language (see README.md in this folder)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent
LANGUAGES = ("en", "es", "it", "fr", "de", "vi")


def load(language: str) -> dict[str, Any]:
    """The tables of one normalizer language."""
    return json.loads((DATA_DIR / f"{language}.json").read_text(encoding="utf-8"))


TEXT_DATA: dict[str, dict[str, Any]] = {language: load(language) for language in LANGUAGES}


def forms(written: str | list[str]) -> list[str]:
    """The written forms of an entry: one string or a list."""
    return [written] if isinstance(written, str) else list(written)
