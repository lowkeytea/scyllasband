"""Phrase-level text segmentation for Scylla's Band G2P inference.

This mirrors the Scylla's Band phrase G2P contract: split on phrase boundaries,
drop bracketed stage notes by default, and keep chunks under the exported
phrase model's character budget without chopping words.
"""

from __future__ import annotations

import re
import unicodedata


DEFAULT_G2P_PHRASE_MAX_CHARS = 140
PUNCTUATION_PHONE_TOKENS = {
    ",": "<pause_comma>",
    ";": "<pause_semicolon>",
    ":": "<pause_colon>",
    "-": "<pause_dash>",
    "\u2014": "<pause_dash>",
    "\u2026": "<ellipsis>",
    "...": "<ellipsis>",
    ".": "<end_stmt>",
    "?": "<end_question>",
    "!": "<end_exclaim>",
}
BOUNDARY_PHONE_TOKENS = (
    "<pause_comma>",
    "<pause_semicolon>",
    "<pause_colon>",
    "<pause_dash>",
    "<ellipsis>",
    "<end_stmt>",
    "<end_question>",
    "<end_exclaim>",
    "<ctx_sentence_start>",
    "<ctx_continuation>",
    "<ctx_sentence_end>",
    "<ctx_chunk_continue>",
)
CONTEXT_PHONE_TOKENS = (
    "<ctx_sentence_start>",
    "<ctx_continuation>",
    "<ctx_sentence_end>",
    "<ctx_chunk_continue>",
)
NON_ACOUSTIC_IPA_MODIFIERS = frozenset({"ˈ", "ˌ", "ː", "ˑ"})

_WHITESPACE_RE = re.compile(r"\s+")
_WHITESPACE_ONLY_RE = re.compile(r"^\s*$")
_BRACKETED_NOTE_RE = re.compile(r"\[[^\[\]\n]{1,80}\]")
_BOUNDARY_PUNCTUATION = frozenset(".!?;:…")
_SOFT_BOUNDARY_PUNCTUATION = frozenset(",")
_PHONE_BOUNDARY_PUNCTUATION = frozenset(",.!?;:-\u2026\u2014")
_CLOSING_BOUNDARY_CHARS = frozenset("\"')]}»”’")
_PUNCTUATION_FOLD = str.maketrans(
    {
        "\u00a0": " ",
        "\u1680": " ",
        "\u2000": " ",
        "\u2001": " ",
        "\u2002": " ",
        "\u2003": " ",
        "\u2004": " ",
        "\u2005": " ",
        "\u2006": " ",
        "\u2007": " ",
        "\u2008": " ",
        "\u2009": " ",
        "\u200a": " ",
        "\u202f": " ",
        "\u205f": " ",
        "\u3000": " ",
        "\u02bc": "'",
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\uff07": "'",
        "\u00ab": '"',
        "\u00bb": '"',
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\uff02": '"',
        "\u2026": "...",
    }
)


def normalize_g2p_phrase_punctuation(text: str) -> str:
    value = str(text or "")
    for hyphen in ("\u2010", "\u2011"):
        value = value.replace(hyphen, "-")
    for dash in ("\u2012", "\u2013", "\u2014", "\u2015", "\u2212", "\ufe58", "\ufe63", "\uff0d"):
        value = value.replace(dash, " — ")
    value = re.sub(r"-{2,}", " — ", value)
    value = re.sub(r"(?:(?<=\s)-+|-+(?=\s))", " — ", value)
    value = value.translate(_PUNCTUATION_FOLD)
    value = re.sub(r"\.{2,}", "...", value)
    value = re.sub(r"([!?])[!?]+", lambda match: match.group(1), value)
    return _WHITESPACE_RE.sub(" ", value).strip()


def is_non_acoustic_phone_modifier(phone: str) -> bool:
    """Return whether a frontend symbol modifies a phone but owns no audio."""
    value = str(phone or "")
    if value in NON_ACOUSTIC_IPA_MODIFIERS:
        return True
    if not value or value.startswith("<"):
        return False
    return all(unicodedata.category(char).startswith("M") for char in value)


def punctuation_run_phone_token(punctuation: str) -> str | None:
    run = str(punctuation or "")
    if not run:
        return None
    for char in run:
        if char == "?":
            return "<end_question>"
        if char == "!":
            return "<end_exclaim>"
    if "\u2026" in run or run.count(".") >= 2:
        return "<ellipsis>"
    if "." in run:
        return "<end_stmt>"
    if "," in run:
        return "<pause_comma>"
    if ";" in run:
        return "<pause_semicolon>"
    if ":" in run:
        return "<pause_colon>"
    if "-" in run or "\u2014" in run:
        return "<pause_dash>"
    return None


def punctuation_phone_token(text: str) -> str | None:
    value = normalize_g2p_phrase_punctuation(text).strip()
    if not value:
        return None
    end = len(value)
    while end > 0 and value[end - 1].isspace():
        end -= 1
    while end > 0 and value[end - 1] in _CLOSING_BOUNDARY_CHARS:
        end -= 1
    index = end - 1
    while index >= 0 and value[index] in _PHONE_BOUNDARY_PUNCTUATION:
        index -= 1
    if index == end - 1:
        return None
    return punctuation_run_phone_token(value[index + 1 : end])


def g2p_phrase_segments(
    text: str,
    *,
    max_chars: int = DEFAULT_G2P_PHRASE_MAX_CHARS,
    drop_bracketed_notes: bool = True,
) -> list[str]:
    value = normalize_g2p_phrase_punctuation(text)
    if drop_bracketed_notes:
        value = _BRACKETED_NOTE_RE.sub(" ", value)
    value = _WHITESPACE_RE.sub(" ", value).strip()
    if not value:
        return []

    chunks: list[str] = []
    for chunk in _split_on_punctuation(value, boundary_chars=_BOUNDARY_PUNCTUATION):
        chunks.extend(_split_long_chunk(chunk, max_chars=max_chars))
    return [chunk for chunk in chunks if not _WHITESPACE_ONLY_RE.fullmatch(chunk)]


def _consume_boundary(text: str, index: int, boundary_chars: frozenset[str]) -> int:
    end = index + 1
    while end < len(text) and text[end] in boundary_chars:
        end += 1
    while end < len(text) and text[end] in _CLOSING_BOUNDARY_CHARS:
        end += 1
    return end


def _split_on_punctuation(text: str, *, boundary_chars: frozenset[str]) -> list[str]:
    value = _WHITESPACE_RE.sub(" ", str(text or "")).strip()
    if not value:
        return []
    chunks: list[str] = []
    start = 0
    index = 0
    while index < len(value):
        if value[index] in boundary_chars:
            end = _consume_boundary(value, index, boundary_chars)
            chunk = value[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end
            while start < len(value) and value[start].isspace():
                start += 1
            index = start
            continue
        index += 1
    tail = value[start:].strip()
    if tail:
        chunks.append(tail)
    return chunks or [value]


def _split_long_chunk(chunk: str, *, max_chars: int) -> list[str]:
    value = _WHITESPACE_RE.sub(" ", str(chunk or "")).strip()
    if not value:
        return []
    if max_chars <= 0 or len(value) <= max_chars:
        return [value]

    comma_chunks = _split_on_punctuation(value, boundary_chars=_SOFT_BOUNDARY_PUNCTUATION)
    if len(comma_chunks) > 1:
        merged: list[str] = []
        current = ""
        for part in comma_chunks:
            candidate = part if not current else f"{current} {part}"
            if current and len(candidate) > max_chars:
                merged.append(current)
                current = part
            else:
                current = candidate
        if current:
            merged.append(current)

        out: list[str] = []
        for part in merged:
            if len(part) > max_chars and part != value:
                out.extend(_split_long_chunk(part, max_chars=max_chars))
            else:
                out.append(part)
        return out

    out: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in value.split():
        next_len = len(word) if not current else current_len + 1 + len(word)
        if current and next_len > max_chars:
            out.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len = next_len
    if current:
        out.append(" ".join(current))
    return out
