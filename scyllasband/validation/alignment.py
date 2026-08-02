"""Language-safe ASR normalization and edit alignment."""

from __future__ import annotations

import re
import unicodedata
from typing import Any


_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'"})
_HYPHENS = {"-", "‐", "‑", "‒", "–", "—", "―"}
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_for_wer(text: str, language: str | None = None) -> str:
    # Normalize numeric formatting with the same language rules used before G2P.
    value = str(text or "")
    if language:
        from scyllasband.text_normalizer import normalize_spoken_text

        value = normalize_spoken_text(value, language=language)
    value = unicodedata.normalize("NFC", value).translate(_APOSTROPHES).casefold()
    output: list[str] = []
    for character in value:
        category = unicodedata.category(character)
        if character in _HYPHENS or character == "'":
            output.append(" ")
        elif character.isspace():
            output.append(" ")
        elif category.startswith("L") or category.startswith("N") or category.startswith("M"):
            output.append(character)
        else:
            output.append(" ")
    return _WHITESPACE_RE.sub(" ", "".join(output)).strip()


def tokens_for_wer(text: str, language: str | None = None) -> list[str]:
    normalized = normalize_for_wer(text, language=language)
    return normalized.split() if normalized else []


def _distance(reference: list[str], hypothesis: list[str], *, tolerant: bool) -> int:
    n = len(reference)
    m = len(hypothesis)
    large = n + m + 10
    costs = [[large] * (m + 1) for _ in range(n + 1)]
    costs[0][0] = 0
    for i in range(n + 1):
        for j in range(m + 1):
            current = costs[i][j]
            if i < n:
                costs[i + 1][j] = min(costs[i + 1][j], current + 1)
            if j < m:
                costs[i][j + 1] = min(costs[i][j + 1], current + 1)
            if i < n and j < m:
                costs[i + 1][j + 1] = min(
                    costs[i + 1][j + 1],
                    current + (0 if reference[i] == hypothesis[j] else 1),
                )
            if tolerant and i < n and j < m:
                for width in range(2, 5):
                    if j + width <= m and reference[i] == "".join(hypothesis[j : j + width]):
                        costs[i + 1][j + width] = min(costs[i + 1][j + width], current)
                    if i + width <= n and "".join(reference[i : i + width]) == hypothesis[j]:
                        costs[i + width][j + 1] = min(costs[i + width][j + 1], current)
    return int(costs[n][m])


def align_words(reference_text: str, hypothesis_text: str, language: str | None = None) -> dict[str, Any]:
    reference = tokens_for_wer(reference_text, language=language)
    hypothesis = tokens_for_wer(hypothesis_text, language=language)
    n = len(reference)
    m = len(hypothesis)
    costs = [[0] * (m + 1) for _ in range(n + 1)]
    back: list[list[str | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        costs[i][0] = i
        back[i][0] = "delete"
    for j in range(1, m + 1):
        costs[0][j] = j
        back[0][j] = "insert"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            same = reference[i - 1] == hypothesis[j - 1]
            candidates = [
                (costs[i - 1][j - 1] + (0 if same else 1), "match" if same else "substitute"),
                (costs[i - 1][j] + 1, "delete"),
                (costs[i][j - 1] + 1, "insert"),
            ]
            costs[i][j], back[i][j] = min(candidates, key=lambda item: (item[0], item[1] != "match"))

    operations: list[dict[str, Any]] = []
    i, j = n, m
    while i > 0 or j > 0:
        operation = back[i][j]
        if operation in {"match", "substitute"}:
            operations.append(
                {
                    "operation": operation,
                    "reference": reference[i - 1],
                    "hypothesis": hypothesis[j - 1],
                    "reference_index": i - 1,
                    "hypothesis_index": j - 1,
                }
            )
            i -= 1
            j -= 1
        elif operation == "delete":
            operations.append(
                {
                    "operation": "delete",
                    "reference": reference[i - 1],
                    "hypothesis": None,
                    "reference_index": i - 1,
                    "hypothesis_index": None,
                }
            )
            i -= 1
        elif operation == "insert":
            operations.append(
                {
                    "operation": "insert",
                    "reference": None,
                    "hypothesis": hypothesis[j - 1],
                    "reference_index": None,
                    "hypothesis_index": j - 1,
                }
            )
            j -= 1
        else:
            raise RuntimeError(f"Invalid ASR alignment state at ({i}, {j})")
    operations.reverse()
    counts = {
        "matches": sum(item["operation"] == "match" for item in operations),
        "substitutions": sum(item["operation"] == "substitute" for item in operations),
        "deletions": sum(item["operation"] == "delete" for item in operations),
        "insertions": sum(item["operation"] == "insert" for item in operations),
    }
    errors = counts["substitutions"] + counts["deletions"] + counts["insertions"]
    tolerant_errors = _distance(reference, hypothesis, tolerant=True)
    return {
        "reference_normalized": " ".join(reference),
        "hypothesis_normalized": " ".join(hypothesis),
        "reference_words": len(reference),
        "hypothesis_words": len(hypothesis),
        "errors": errors,
        "strict_wer": errors / float(len(reference)) if reference else (0.0 if not hypothesis else 1.0),
        "format_tolerant_errors": tolerant_errors,
        "format_tolerant_wer": (
            tolerant_errors / float(len(reference))
            if reference
            else (0.0 if not hypothesis else 1.0)
        ),
        **counts,
        "operations": operations,
    }


def word_error_rate(
    reference_text: str,
    hypothesis_text: str,
    language: str | None = None,
    *,
    format_tolerant: bool = False,
) -> float:
    reference = tokens_for_wer(reference_text, language=language)
    hypothesis = tokens_for_wer(hypothesis_text, language=language)
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _distance(reference, hypothesis, tolerant=format_tolerant) / float(len(reference))


def _character_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, start=1):
        current = [i]
        for j, right in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (0 if left == right else 1),
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(reference_text: str, hypothesis_text: str, language: str | None = None) -> float:
    reference = normalize_for_wer(reference_text, language=language).replace(" ", "")
    hypothesis = normalize_for_wer(hypothesis_text, language=language).replace(" ", "")
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _character_distance(reference, hypothesis) / float(len(reference))
