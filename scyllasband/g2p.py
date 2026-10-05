"""Text -> Scylla's Band phones: neural G2P with the training phone layout.

Layout (what the acoustic model was trained on):
- a leading ``<sil>``;
- between two words of a phrase, a word-boundary ``<sil>`` (it may receive zero frames);
- after every punctuation token, a ``<sil>``;
- a trailing ``<sil>``.

``word_starts`` lists the first phone of every word after the first; the flow conditions on them.
"""

from __future__ import annotations

from collections import OrderedDict
import copy
import json
import math
from pathlib import Path
import re
import threading
import unicodedata
from typing import Any, Callable, Mapping

import numpy as np

from .g2p_phrases import (DEFAULT_G2P_PHRASE_MAX_CHARS, g2p_phrase_segments, normalize_g2p_phrase_punctuation,
                          punctuation_run_phone_token)
from .text_normalizer import SPOKEN_TEXT_NORMALIZER_SHA256

CACHE_SIZE = 256
PUNCTUATION_CHARS = frozenset(",.!?;:-…—")
TERMINAL_PRIORITY = ("<end_question>", "<end_exclaim>", "<ellipsis>", "<end_stmt>")
_BRACKETED_NOTE_RE = re.compile(r"\[[^\[\]\n]{1,80}\]")
WORD_BOUNDARY_MARKER = "<word_boundary>"
SILENCE = "<sil>"
SILENCE_PHONES = frozenset({"<sil>", "sil", "sp", "<sp>"})


def load_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


class G2PFrontend:
    """Phonemizer over the bundle's G2P graph. ``infer`` maps int64 [1, tokens] text ids to [1, frames, symbols] logits."""

    def __init__(self, bundle_dir: Path, assets: Mapping[str, str], controls: Mapping[str, Any],
                 phone_to_id: Mapping[str, int], infer: Callable[[np.ndarray], np.ndarray]) -> None:
        self.phone_to_id = dict(phone_to_id)
        self.infer = infer
        self.config = load_json(bundle_dir / assets["g2p_config"])
        self.tokenizer = load_json(bundle_dir / assets["g2p_tokenizer"])
        self.language_map = load_json(bundle_dir / assets["g2p_language_map"])
        normalization = load_json(bundle_dir / assets["g2p_normalization"]) if assets.get("g2p_normalization") else {}
        declared = str(normalization.get("contract_sha256") or "")
        if declared and declared != SPOKEN_TEXT_NORMALIZER_SHA256:
            raise ValueError("Bundle spoken-text normalizer contract does not match this runtime: "
                             f"bundle={declared} runtime={SPOKEN_TEXT_NORMALIZER_SHA256}")
        overrides = assets.get("g2p_pronunciation_overrides")
        override_path = bundle_dir / overrides if overrides else None
        self.overrides = load_json(override_path) if override_path is not None and override_path.exists() else {}
        boundaries = dict(controls.get("word_boundaries") or {})
        self.word_boundary_symbol = str(boundaries.get("g2p_output_symbol", " "))
        fixed = dict(controls.get("fixed_shapes") or {})
        self.text_tokens = int(fixed.get("g2p_text_tokens") or self.config.get("fixed_text_tokens") or 512)
        self.segment_config = _segment_config(self.config, self.tokenizer, self.text_tokens)
        self._symbols = {str(k): int(v) for k, v in self.tokenizer["text_symbols"].items()}
        self._phonemes = {int(k): str(v) for k, v in dict(self.tokenizer["phoneme_symbols"]).items()}
        self._cache: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
        self._word_cache: OrderedDict[tuple[str, str], tuple[str, ...]] = OrderedDict()
        # Segments are predicted independently, so a text that joins already spoken sentences reuses their predictions.
        self._segment_cache: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
        self._lock = threading.RLock()

    def model_language(self, language: str) -> str:
        normalized = str(language).strip().lower().replace("-", "_")
        return str(self.language_map.get(normalized, normalized))

    def phonemize(self, text: str, *, language: str) -> dict[str, Any]:
        key = (str(text or ""), self.model_language(language))
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return copy.deepcopy(cached)
            result = self._phonemize(key[0], language=key[1])
            self._cache[key] = copy.deepcopy(result)
            while len(self._cache) > CACHE_SIZE:
                self._cache.popitem(last=False)
            return result

    def _phonemize(self, text: str, *, language: str) -> dict[str, Any]:
        segments = punctuated_segments(text, max_chars=int(self.segment_config["chunk_max_chars"]))
        phones: list[str] = [SILENCE]
        word_starts: list[int] = []
        candidates: list[int] = []
        punctuation: list[dict[str, Any]] = []
        repairs: list[dict[str, Any]] = []
        applied_overrides: list[dict[str, Any]] = []
        confidences: list[float] = []
        for index, (segment, tokens) in enumerate(segments):
            prediction = self._predict_segment(segment, language=language)
            confidences.append(float(prediction.get("confidence", 0.0)))
            applied_overrides.extend(prediction.get("pronunciation_overrides", []))
            repairs.extend(prediction.get("terminal_tail_repairs", []))
            predicted = [str(p) for p in prediction.get("phones", ())]
            word_open = False
            for position, phone in enumerate(predicted):
                if phone == WORD_BOUNDARY_MARKER:
                    if _is_internal_word_boundary(predicted, position):
                        word_open = False
                    continue
                if phone in SILENCE_PHONES:
                    continue
                if not word_open:
                    if word_starts or len(phones) > 1:   # every word after the first in the chunk
                        if phones[-1] != SILENCE:
                            candidates.append(len(phones))
                            phones.append(SILENCE)
                        word_starts.append(len(phones))
                    word_open = True
                phones.append(phone)
            if len(phones) > 1:
                for token in tokens:
                    if token in self.phone_to_id:
                        punctuation.append(dict(segment_index=index, phone_index=len(phones), phone=token))
                        phones.append(token)
                        phones.append(SILENCE)
        if phones[-1] != SILENCE:
            phones.append(SILENCE)
        if len(phones) <= 2 and not any(p not in SILENCE_PHONES for p in phones):
            raise ValueError("G2P produced no usable Scylla's Band phone symbols")
        unknown = sorted({p for p in phones if p not in self.phone_to_id})
        if unknown:
            raise ValueError(f"G2P produced symbols outside the phone vocabulary: {', '.join(unknown[:20])}")
        result = dict(phones=phones, word_starts=word_starts, word_boundary_candidates=candidates, punctuation=punctuation,
                      g2p_segments=[segment for segment, _ in segments], g2p_confidence=min(confidences) if confidences else 0.0)
        if applied_overrides:
            result["pronunciation_overrides"] = applied_overrides
        if repairs:
            result["terminal_tail_repairs"] = repairs
        return result

    # --- graph prediction -------------------------------------------------------------------------------------------
    def _predict_segment(self, text: str, *, language: str) -> dict[str, Any]:
        key = (text, language)
        with self._lock:
            cached = self._segment_cache.get(key)
            if cached is not None:
                self._segment_cache.move_to_end(key)
                return copy.deepcopy(cached)
        prediction = self._predict_raw(text, language=language)
        prediction = self._apply_overrides(prediction, text=text, language=language)
        prediction = self._repair_terminal_tail(prediction, text=text, language=language)
        with self._lock:
            self._segment_cache[key] = copy.deepcopy(prediction)
            while len(self._segment_cache) > CACHE_SIZE:
                self._segment_cache.popitem(last=False)
        return prediction

    def _predict_raw(self, text: str, *, language: str) -> dict[str, Any]:
        logits = np.asarray(self.infer(self._encode(text, language=language)[np.newaxis, :]), dtype=np.float32)
        return self._decode(logits[0])

    def _encode(self, text: str, *, language: str) -> np.ndarray:
        language_token = f"<{language}>"
        if language_token not in self._symbols:
            raise ValueError(f"G2P language {language!r} is not supported by this bundle")
        repeats = max(1, int(self.tokenizer.get("char_repeats", 1)))
        work = str(text or "")
        if bool(self.tokenizer.get("lowercase", True)):
            work = work.lower()
        sequence = [self._symbols[language_token]]
        emitted = 0
        for char in work:
            token = self._symbols.get(char)
            if token is None:
                continue
            sequence.extend([token] * repeats)
            emitted += 1
        sequence.append(int(self._symbols.get("<end>", self.tokenizer.get("text_pad_index", 0))))
        if emitted <= 0:
            raise ValueError("G2P text has no characters supported by this bundle")
        if len(sequence) > self.text_tokens:
            raise ValueError(f"G2P input encodes to {len(sequence)} tokens, but this bundle supports at most "
                             f"{self.text_tokens}. Split the text into shorter chunks.")
        out = np.full((self.text_tokens,), int(self.tokenizer.get("text_pad_index", 0)), dtype=np.int64)
        out[: len(sequence)] = sequence
        return out

    def _decode(self, logits: np.ndarray) -> dict[str, Any]:
        pad = int(self.tokenizer.get("phoneme_pad_index", 0))
        end = int(self.tokenizer.get("phoneme_end_index", 0))
        probs = _softmax(logits, axis=-1)
        best = np.argmax(probs, axis=-1)
        phones: list[str] = []
        emitted: list[float] = []
        previous = None
        for frame, token in enumerate(best.tolist()):
            if token == previous:
                continue
            previous = token
            if token == pad:
                continue
            if token == end:
                break
            symbol = self._phonemes.get(int(token))
            if symbol == self.word_boundary_symbol:
                phones.append(WORD_BOUNDARY_MARKER)
                emitted.append(float(probs[frame, token]))
                continue
            if _skip_output_symbol(symbol):
                continue
            phones.append(str(symbol))
            emitted.append(float(probs[frame, token]))
        return dict(phones=phones, confidence=_prob_product(emitted))

    def _apply_overrides(self, prediction: dict[str, Any], *, text: str, language: str) -> dict[str, Any]:
        overrides = _overrides_for_language(self.overrides, language=language)
        words = {m.group(0).lower() for m in re.finditer(r"[A-Za-z]+(?:'[A-Za-z]+)?", text or "")}
        if not overrides or not words:
            return prediction
        phones = [str(p) for p in prediction.get("phones", [])]
        applied = []
        for word, spec in sorted(overrides.items()):
            if word not in words:
                continue
            target, source = _phone_list(spec, key="phones"), _phone_list(spec, key="replace")
            if not target or not source:
                continue
            phones, count = _replace_subsequence(phones, source, target)
            if count:
                applied.append(dict(word=word, replace=source, phones=target, count=count))
        if not applied:
            return prediction
        return dict(prediction, phones=phones, pronunciation_overrides=applied)

    def _repair_terminal_tail(self, prediction: dict[str, Any], *, text: str, language: str) -> dict[str, Any]:
        """Trim a short CTC tail hallucination after a correctly decoded final word."""
        word = _terminal_word(text)
        if not word:
            return prediction
        key = (language, word)
        with self._lock:
            isolated = self._word_cache.get(key)
        if isolated is None:
            isolated_prediction = self._apply_overrides(self._predict_raw(word, language=language), text=word, language=language)
            isolated = tuple(str(p) for p in isolated_prediction.get("phones", []))
            with self._lock:
                self._word_cache[key] = isolated
                while len(self._word_cache) > CACHE_SIZE:
                    self._word_cache.popitem(last=False)
        repaired, removed = _trim_terminal_artifacts([str(p) for p in prediction.get("phones", [])], list(isolated))
        if not removed:
            return prediction
        return dict(prediction, phones=repaired, terminal_tail_repairs=[dict(word=word, phones=list(isolated), removed=removed)])


def punctuated_segments(text: str, *, max_chars: int = DEFAULT_G2P_PHRASE_MAX_CHARS) -> list[tuple[str, list[str]]]:
    """Text split at every punctuation run, each with the punctuation tokens that follow it (as in training).

    Consecutive runs separated only by spaces (", --") stack on the preceding phrase; a stack that contains a
    sentence end keeps only that end. Phrases longer than ``max_chars`` (the G2P input budget) are split between
    words, with no punctuation between the parts.
    """
    value = _BRACKETED_NOTE_RE.sub(" ", normalize_g2p_phrase_punctuation(text))
    value = re.sub(r"\s+", " ", value).strip()
    out: list[tuple[str, list[str]]] = []
    current: list[str] = []
    index = 0
    while index < len(value):
        if value[index] in PUNCTUATION_CHARS:
            end = index + 1
            while end < len(value) and value[end] in PUNCTUATION_CHARS:
                end += 1
            token = punctuation_run_phone_token(value[index:end])
            segment = "".join(current).strip()
            if _has_word(segment):
                out.append((segment, [token] if token else []))
            elif out and token:
                out[-1][1].append(token)
            current, index = [], end
            continue
        current.append(value[index])
        index += 1
    if _has_word("".join(current)):
        out.append(("".join(current).strip(), []))
    result: list[tuple[str, list[str]]] = []
    for segment, tokens in out:
        terminal = next((t for t in TERMINAL_PRIORITY if t in tokens), None)
        tokens = [terminal] if terminal else tokens
        parts = _split_words(segment, max_chars)
        result.extend((part, []) for part in parts[:-1])
        result.append((parts[-1], tokens))
    return result


def _has_word(text: str) -> bool:
    return any(char.isalnum() for char in text)


def _split_words(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars:
            parts.append(current)
            current = word
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def g2p_text_segments(text: str, *, config: Mapping[str, Any] | None = None) -> list[str]:
    config_map = dict(config or {})
    if str(config_map.get("input_granularity") or "phrase").strip().lower() == "phrase":
        segments = g2p_phrase_segments(
            text,
            max_chars=_positive_int(config_map.get("chunk_max_chars", config_map.get("phrase_chunk_max_chars")), DEFAULT_G2P_PHRASE_MAX_CHARS),
            drop_bracketed_notes=_bool(config_map.get("drop_bracketed_notes"), True),
            preserve_commas=bool(config_map.get("preserve_commas", False)),
        )
        return segments or [str(text or "").strip()]
    segments = [s.strip() for s in re.split(r"[.!?,;:]+", text) if s.strip()]
    return segments or [str(text or "").strip()]


def _segment_config(config: Mapping[str, Any], tokenizer: Mapping[str, Any], fixed_text_tokens: int) -> dict[str, Any]:
    out = dict(config or {})
    configured = _positive_int(out.get("chunk_max_chars", out.get("phrase_chunk_max_chars")), DEFAULT_G2P_PHRASE_MAX_CHARS)
    try:
        repeats = max(1, int(dict(tokenizer or {}).get("char_repeats", 1)))
    except Exception:
        repeats = 1
    tokens = max(1, int(fixed_text_tokens or 0))
    out["chunk_max_chars"] = min(configured, max(1, (tokens - 2) // repeats))
    out["fixed_text_tokens"] = tokens
    return out


def _is_internal_word_boundary(phones: list[str], index: int) -> bool:
    if phones[index] != WORD_BOUNDARY_MARKER or (index > 0 and phones[index - 1] == WORD_BOUNDARY_MARKER):
        return False
    acoustic = lambda p: p not in {WORD_BOUNDARY_MARKER, *SILENCE_PHONES}  # noqa: E731
    return any(acoustic(p) for p in phones[:index]) and any(acoustic(p) for p in phones[index + 1:])


def _overrides_for_language(data: Mapping[str, Any], *, language: str) -> dict[str, Any]:
    languages = data.get("languages", data) if isinstance(data, Mapping) else {}
    raw = languages.get(str(language)) if isinstance(languages, Mapping) else None
    if not isinstance(raw, Mapping):
        return {}
    return {str(word).strip().lower(): spec for word, spec in raw.items() if str(word).strip()}


def _terminal_word(text: str) -> str:
    matches = list(re.finditer(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", str(text or "").lower(), flags=re.UNICODE))
    return matches[-1].group(0) if matches else ""


def _trim_terminal_artifacts(phrase: list[str], isolated: list[str], *, max_removed: int = 2) -> tuple[list[str], list[str]]:
    """Remove only a short suffix following an exact isolated-word match."""
    if not phrase or not isolated:
        return phrase, []
    count = len(isolated)
    for start in range(len(phrase) - count, max(0, len(phrase) - count - max_removed) - 1, -1):
        end = start + count
        if phrase[start:end] != isolated:
            continue
        removed = phrase[end:]
        if 0 < len(removed) <= max_removed:
            return phrase[:end], removed
    return phrase, []


def _phone_list(spec: Any, *, key: str) -> list[str]:
    value = spec.get(key) if isinstance(spec, Mapping) else (spec if key == "phones" else None)
    if isinstance(value, str):
        return [item for item in value.split() if item]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return []


def _replace_subsequence(phones: list[str], source: list[str], target: list[str]) -> tuple[list[str], int]:
    output, count, index = [], 0, 0
    while index < len(phones):
        if phones[index:index + len(source)] == source:
            output.extend(target)
            index += len(source)
            count += 1
        else:
            output.append(phones[index])
            index += 1
    return output, count


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return int(default)
    return parsed if parsed > 0 else int(default)


def _bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, (bool, int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return True if text in {"1", "true", "yes", "on"} else False if text in {"0", "false", "no", "off"} else bool(default)


def _softmax(values: np.ndarray, *, axis: int) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def _skip_output_symbol(symbol: str | None) -> bool:
    if not symbol or symbol.startswith("<") or symbol == "_" or not symbol.strip():
        return True
    return all(unicodedata.category(char)[0] in {"P", "Z"} for char in symbol)


def _prob_product(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(math.exp(sum(math.log(max(float(v), 1e-12)) for v in values)))
