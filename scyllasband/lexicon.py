"""Runtime pronunciation lexicon: words and initialisms the G2P gets wrong, applied over its word groups.

The G2P reads whole phrases, which keeps connected-speech forms (unstressed "to", linking r, French liaison); a full word
dictionary would undo them. This lexicon is small and targeted:

- initialisms, the G2P's weakest class: in a long phrase it can drop a letter ("GPU" said as "GU"). An initialism matches
  only when written in capitals (also with a plural or possessive "s"), and not inside text written all in capitals;
- words it mispronounces (``WORDS``), matched in any case.

A bundle's ``g2p_phoneme_dict`` asset ({language: {word: phones}}) adds or replaces word entries. The C++ runtime embeds
the same tables (libscyllasband/tools/generate_lexicon.py writes src/scyllasband_lexicon.inc from this module).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

# Letter names as phone templates; "{s}" takes secondary stress, and primary stress on an initialism's last letter, as
# espeak reads them ("GPU" dʒˌiːpˌiːjˈuː).
_EN_LETTERS = {
    "A": "{s}eɪ", "B": "b{s}iː", "C": "s{s}iː", "D": "d{s}iː", "E": "{s}iː", "F": "{s}ɛf", "G": "dʒ{s}iː", "H": "{s}eɪtʃ",
    "I": "{s}aɪ", "J": "dʒ{s}eɪ", "K": "k{s}eɪ", "L": "{s}ɛl", "M": "{s}ɛm", "N": "{s}ɛn", "O": "{s}oʊ", "P": "p{s}iː",
    "Q": "kj{s}uː", "R": "{s}ɑːɹ", "S": "{s}ɛs", "T": "t{s}iː", "U": "j{s}uː", "V": "v{s}iː", "W": "d{s}ʌbəljuː",
    "X": "{s}ɛks", "Y": "w{s}aɪ", "Z": "z{s}iː",
}
LETTERS: dict[str, dict[str, str]] = {
    "en_us": _EN_LETTERS,
    "en_gb": dict(_EN_LETTERS, O="{s}əʊ", R="{s}ɑː", Z="z{s}ɛd"),
}
# Initialisms read letter by letter. Not listed: ones that are also everyday words in capitals ("US", "IT", "AM"), and
# acronyms read as words ("NASA", "JSON", "ONNX").
_EN_INITIALISMS = (
    "AI", "API", "AR", "ASR", "BBC", "CEO", "CLI", "CNN", "CPU", "CSS", "CSV", "DIY", "DNA", "DNS", "EU", "FAQ", "FBI", "GPS",
    "GPU", "HDMI", "HTML", "HTTP", "HTTPS", "ID", "IDE", "IPA", "IRS", "LLM", "MIT", "MRI", "NBA", "NFL", "NLP", "NPU", "NYC",
    "PC", "PDF", "RNA", "SDK", "SQL", "SSD", "TDT", "TPU", "TTS", "TV", "UI", "UK", "URI", "URL", "USA", "USB", "UX", "VPN",
    "VR", "XML",
)
INITIALISMS: dict[str, tuple[str, ...]] = {"en_us": _EN_INITIALISMS, "en_gb": _EN_INITIALISMS}
_EN_WORDS = {
    "git": "ɡˈɪt",                        # unstressed after "new" in a phrase, heard as "NuGet"
    "librispeech": "lˈɪbɹiːspˌiːtʃ",      # G2P: lˈaɪbɹɪspiːtʃ ("LibreSpeech")
    "pytest": "pˈaɪtˌɛst",                # G2P: pˈaɪɾɪst ("pytis")
}
WORDS: dict[str, dict[str, str]] = {"en_us": _EN_WORDS, "en_gb": _EN_WORDS}
PLURAL_SUFFIXES = ("'s", "’s", "s")
PLURAL_PHONE = "z"


def initialism_phones(word: str, letters: Mapping[str, str]) -> str:
    """Phones of an initialism from letter templates: secondary stress on every letter but the last, primary on the last."""
    parts = [letters[ch] for ch in word]
    return "".join(part.replace("{s}", "ˌ" if k < len(parts) - 1 else "ˈ") for k, part in enumerate(parts))


def split_phones(phones: str, vocabulary: Mapping[str, Any]) -> tuple[str, ...] | None:
    """A phone string as vocabulary tokens, longest match first; None when any part is outside the vocabulary."""
    longest = max((len(token) for token in vocabulary), default=1)
    out: list[str] = []
    position = 0
    while position < len(phones):
        for length in range(min(longest, len(phones) - position), 0, -1):
            token = phones[position:position + length]
            if token in vocabulary:
                out.append(token)
                position += length
                break
        else:
            return None
    return tuple(out) if out else None


class Lexicon:
    """Word and initialism entries per G2P language, as phone tuples checked against the acoustic model's phone vocabulary."""

    def __init__(self, vocabulary: Mapping[str, Any], bundle_words: Mapping[str, Any] | None = None) -> None:
        self.words: dict[str, dict[str, tuple[str, ...]]] = {}
        self.initialisms: dict[str, dict[str, tuple[str, ...]]] = {}
        for language, letters in LETTERS.items():
            table = {}
            for word in INITIALISMS.get(language, ()):
                phones = split_phones(initialism_phones(word, letters), vocabulary)
                if phones:
                    table[word] = phones
            self.initialisms[language] = table
        for language, entries in WORDS.items():
            self.words[language] = self._words(entries, vocabulary)
        for language, entries in dict(bundle_words or {}).items():
            if isinstance(entries, Mapping) and entries:
                self.words.setdefault(str(language), {}).update(self._words(entries, vocabulary))

    @staticmethod
    def _words(entries: Mapping[str, Any], vocabulary: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
        table = {}
        for word, phones in entries.items():
            key = str(word).strip().lower()
            split = split_phones(str(phones), vocabulary) if key else None
            if split:
                table[key] = split
        return table

    def lookup(self, token: str, language: str, *, initialisms: bool) -> tuple[str, ...] | None:
        """The entry for one whitespace token, or None. ``initialisms`` is False inside text written all in capitals."""
        words = self.words.get(language)
        if words:
            entry = words.get(token.lower())
            if entry:
                return entry
        table = self.initialisms.get(language) if initialisms else None
        if table:
            if token in table:
                return table[token]
            for suffix in PLURAL_SUFFIXES:
                if token.endswith(suffix) and token[:-len(suffix)] in table:
                    return table[token[:-len(suffix)]] + (PLURAL_PHONE,)
        return None


def has_lowercase(text: str) -> bool:
    """False for text written all in capitals, where "ID" or "TV" may be shouted words rather than initialisms."""
    return any("a" <= ch <= "z" for ch in text)


def word_tokens(text: str) -> list[str]:
    return text.split()


def is_word_token(token: str) -> bool:
    return any(ch.isalnum() for ch in token)


def word_spans(phones: Sequence[str], boundary: str, silences: frozenset[str]) -> list[tuple[int, int]]:
    """The acoustic span [start, end) of every word group in a raw G2P prediction (groups split at ``boundary``)."""
    spans: list[tuple[int, int]] = []
    start = end = -1
    for index, phone in enumerate(list(phones) + [boundary]):
        if phone == boundary:
            if start >= 0:
                spans.append((start, end))
            start = end = -1
        elif phone not in silences:
            if start < 0:
                start = index
            end = index + 1
    return spans
