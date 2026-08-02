"""Public ONNX frontend for the promoted Scylla's Band G2P payload."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np


_MODEL_FILENAME = "model.onnx"
_TOKENIZER_FILENAME = "tokenizer.json"
_PHONEME_DICT_FILENAME = "phoneme_dict.json"
_DEFAULT_PUNCTUATION = "().,:?!/–"
_DEFAULT_PROVIDERS = ("CPUExecutionProvider",)
_INITIALISMS = {
    "AI", "API", "AR", "CPU", "CSS", "DNA", "DNS", "EU", "FBI", "GPU",
    "HTML", "HTTP", "HTTPS", "ID", "IPA", "IRS", "JSON", "MLX", "MRI",
    "ONNX", "RNA", "SQL", "TTS", "TV", "UI", "UK", "US", "URL", "URI",
    "USB", "USA", "VR", "XML",
}
_ENGLISH_LANGUAGES = {"en_us", "en_gb"}
_EN_US_LETTER_NAMES = {
    "A": "ˈeɪ", "B": "bˈiː", "C": "sˈiː", "D": "dˈiː", "E": "ˈiː",
    "F": "ˈɛf", "G": "dʒˈiː", "H": "ˈeɪtʃ", "I": "ˈaɪ", "J": "dʒˈeɪ",
    "K": "kˈeɪ", "L": "ˈɛl", "M": "ˈɛm", "N": "ˈɛn", "O": "ˈoʊ",
    "P": "pˈiː", "Q": "kjˈuː", "R": "ˈɑːɹ", "S": "ˈɛs", "T": "tˈiː",
    "U": "jˈuː", "V": "vˈiː", "W": "dˈʌbəljˌuː", "X": "ˈɛks",
    "Y": "wˈaɪ", "Z": "zˈiː",
}


class ScyllasBandG2P:
    """Word-oriented, dictionary-assisted Scylla's Band ONNX G2P."""

    def __init__(
        self,
        model_dir: Path | str,
        *,
        language: str,
        preserve_punctuation: bool = True,
        providers: Optional[list[str]] = None,
        use_cache: bool = True,
        cache_dir: Path | str | None = None,
    ) -> None:
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.language = _normalize_language(language)
        self.preserve_punctuation = bool(preserve_punctuation)
        self.providers = list(providers or _DEFAULT_PROVIDERS)
        self.model_path = self.model_dir / _MODEL_FILENAME
        self.tokenizer_path = self.model_dir / _TOKENIZER_FILENAME
        self.phoneme_dict_path = self.model_dir / _PHONEME_DICT_FILENAME
        for path in (self.model_path, self.tokenizer_path, self.phoneme_dict_path):
            if not path.exists():
                raise FileNotFoundError(f"ScyllasBandG2P ONNX bundle file not found: {path}")

        tokenizer_cfg = json.loads(self.tokenizer_path.read_text(encoding="utf-8"))
        phoneme_dict_cfg = json.loads(self.phoneme_dict_path.read_text(encoding="utf-8"))
        self.languages = [_normalize_language(item) for item in tokenizer_cfg.get("languages", [])]
        if self.language not in self.languages:
            raise ValueError(
                f"Language {self.language!r} not supported by {self.model_dir}; "
                f"supported={self.languages}"
            )
        self.text_symbols = {str(k): int(v) for k, v in dict(tokenizer_cfg["text_symbols"]).items()}
        self.phoneme_symbols = {int(k): str(v) for k, v in dict(tokenizer_cfg["phoneme_symbols"]).items()}
        self.char_repeats = max(1, int(tokenizer_cfg.get("char_repeats", 1) or 1))
        self._start_index = int(self.text_symbols[f"<{self.language}>"])
        self._end_index = int(self.text_symbols["<end>"])
        self._phoneme_end_index = next(
            (idx for idx, token in self.phoneme_symbols.items() if token == "<end>"),
            2,
        )
        self._blank_index = 0
        self._word_dict = dict(phoneme_dict_cfg.get(self.language, {}))
        self._session = None
        self.use_cache = bool(use_cache)
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None
        model_stat = self.model_path.stat()
        self._model_fingerprint = f"{model_stat.st_size:x}-{model_stat.st_mtime_ns:x}"
        self.cache_path = (
            self.cache_dir
            / f"scyllasband_g2p_{self.language}_{self._model_fingerprint}.json"
            if self.cache_dir is not None
            else None
        )
        self._prediction_cache: dict[str, str] = {}
        self._cache_dirty = False
        if self.use_cache and self.cache_path is not None and self.cache_path.is_file():
            try:
                payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if (
                    payload.get("language") == self.language
                    and payload.get("model_fingerprint") == self._model_fingerprint
                ):
                    self._prediction_cache = {
                        str(key): str(value)
                        for key, value in dict(payload.get("entries") or {}).items()
                    }
            except (OSError, TypeError, ValueError):
                self._prediction_cache = {}

    def _ensure_runtime(self) -> None:
        if self._session is not None:
            return
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise RuntimeError("onnxruntime is required for ScyllasBandG2P ONNX evaluation") from exc
        self._session = ort.InferenceSession(str(self.model_path), providers=self.providers)

    def _normalize_numbers(self, text: str) -> str:
        from .text_normalizer import normalize_spoken_text

        return normalize_spoken_text(str(text or ""), language=self.language)

    def _get_initialism_entry(self, word: str) -> Optional[str]:
        if self.language not in _ENGLISH_LANGUAGES:
            return None
        value = str(word or "")
        if len(value) <= 1 or value.upper() != value or value not in _INITIALISMS:
            return None
        parts = [_EN_US_LETTER_NAMES.get(ch) for ch in value]
        if any(part is None for part in parts):
            return None
        return "".join(str(part) for part in parts)

    def _get_dict_entry(self, word: str, punctuation: set[str]) -> Optional[str]:
        if word in punctuation or not word:
            return word
        for candidate in (word, word.lower(), word.title()):
            if candidate in self._word_dict:
                return str(self._word_dict[candidate])
        return self._get_initialism_entry(word)

    def _encode_word(self, word: str) -> np.ndarray:
        seq = [self._start_index]
        for raw_ch in str(word or ""):
            idx = self.text_symbols.get(raw_ch.lower())
            if idx is None:
                continue
            seq.extend([idx] * self.char_repeats)
        seq.append(self._end_index)
        return np.asarray([seq], dtype=np.int64)

    def _predict_word(self, word: str) -> str:
        cache_key = str(word or "").lower()
        if self.use_cache and cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]

        self._ensure_runtime()
        assert self._session is not None
        input_ids = self._encode_word(word)
        if input_ids.shape[1] <= 2:
            predicted = ""
        else:
            logits = self._session.run(["logits"], {"text": input_ids})[0]
            token_ids = np.argmax(logits[0], axis=-1)
            token_ids = token_ids[token_ids != self._blank_index]

            deduped: list[int] = []
            previous: Optional[int] = None
            for token_id in token_ids.tolist():
                token_id = int(token_id)
                if previous == token_id:
                    continue
                deduped.append(token_id)
                previous = token_id

            out: list[str] = []
            for token_id in deduped:
                if token_id == self._phoneme_end_index:
                    break
                symbol = self.phoneme_symbols.get(int(token_id), "")
                if symbol == "_" or symbol == "<end>" or (
                    symbol.startswith("<") and symbol.endswith(">")
                ):
                    continue
                out.append(symbol)
            predicted = "".join(out)

        if self.use_cache:
            self._prediction_cache[cache_key] = predicted
            self._cache_dirty = True
        return predicted

    def _expand_acronym(self, word: str) -> str:
        subwords: list[str] = []
        for subword in str(word or "").split("-"):
            if not subword.isupper() or subword not in _INITIALISMS:
                subwords.append(subword)
                continue
            expanded: list[str] = []
            rest = subword[1:]
            for a, b in zip(subword, rest + "\0"):
                expanded.append(a)
                if b != "\0" and b.isupper():
                    expanded.append("-")
            subwords.append("".join(expanded))
        return "-".join(subwords)

    def phonemize(self, text: str) -> str:
        value = self._normalize_numbers(str(text or "").strip())
        if not value:
            return ""
        punctuation = _DEFAULT_PUNCTUATION if self.preserve_punctuation else ""
        punc_set = set(punctuation + "- ")
        split_pattern = re.compile(f"([{re.escape(punctuation + ' ')}])") if punctuation else re.compile(r"(\s+)")
        cleaned = "".join(ch for ch in value if ch.isalnum() or ch in punc_set)
        split = [part for part in re.split(split_pattern, cleaned) if part]
        if not split:
            return ""

        word_phonemes = {word: self._get_dict_entry(word, punc_set) for word in split}
        words_to_split = [word for word, phonemes in word_phonemes.items() if phonemes is None]
        word_splits: dict[str, list[str]] = {
            word: re.split(r"([-])", self._expand_acronym(word))
            for word in words_to_split
        }
        subwords = {subword for values in word_splits.values() for subword in values}
        for subword in subwords:
            if subword not in word_phonemes:
                word_phonemes[subword] = self._get_dict_entry(subword, punc_set)
        for word, phonemes in list(word_phonemes.items()):
            if phonemes is None and len(word_splits.get(word, [])) <= 1:
                word_phonemes[word] = self._predict_word(word)

        out: list[str] = []
        for word in split:
            phonemes = word_phonemes[word]
            if phonemes is None:
                phonemes = "".join(word_phonemes[subword] or "" for subword in word_splits.get(word, []))
            out.append(str(phonemes or ""))
        return "".join(out).strip() or str(text or "")

    def phonemize_batch(self, texts: list[str]) -> list[str]:
        return [self.phonemize(text) for text in texts]

    def save_cache(self) -> None:
        if (
            not self.use_cache
            or not self._cache_dirty
            or self.cache_path is None
        ):
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        payload = {
            "schema_version": 1,
            "language": self.language,
            "model_fingerprint": self._model_fingerprint,
            "entries": dict(sorted(self._prediction_cache.items())),
        }
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        temporary_path.replace(self.cache_path)
        self._cache_dirty = False


ScyllasBandG2POnnxPhonemizer = ScyllasBandG2P


def _normalize_language(language: object) -> str:
    return str(language or "").strip().lower().replace("-", "_")
