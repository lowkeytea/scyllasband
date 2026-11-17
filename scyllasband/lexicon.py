"""Runtime pronunciation lexicon: words and initialisms the G2P gets wrong, applied over its word groups.

The G2P reads whole phrases, which keeps connected-speech forms (unstressed "to", linking r, French liaison); a full word
dictionary would undo them. This lexicon is small and targeted:

- initialisms, the G2P's weakest class: in a long phrase it can drop a letter ("GPU" said as "GU"). The English list is
  in scyllasband/text_data/en.json: ``spell`` entries are never words, so they match in capitals or in lower case, also
  inside text written all in capitals; ``capitals_only`` entries are also words ("US", "IT"), so they match only in
  capitals inside mixed-case text, as does any other word of capitals with no vowel. A plural or possessive "s" may
  follow;
- words it mispronounces (``WORDS``, from text_data/en.json "pronunciations"), matched in any case.

A bundle's ``g2p_phoneme_dict`` asset ({language: {word: phones}}) adds or replaces word entries. The C++ runtime embeds
the same tables (libscyllasband/tools/generate_lexicon.py writes src/scyllasband_lexicon.inc from this module).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .text_data import TEXT_DATA

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
_EN_INITIALISM_TABLE = TEXT_DATA["en"]["initialisms"]
_EN_INITIALISMS = tuple(_EN_INITIALISM_TABLE["spell"]) + tuple(_EN_INITIALISM_TABLE["capitals_only"])
_EN_CAPITALS_ONLY = frozenset(_EN_INITIALISM_TABLE["capitals_only"])
# Any other word in capitals with no vowel letter is read letter by letter ("PTWC", "MS", "NPR"), up to this length.
SPELLED_MAX_LETTERS = int(TEXT_DATA["en"]["spell_unlisted"]["max"])
_VOWEL_LETTERS = frozenset(TEXT_DATA["en"]["spell_unlisted"]["vowels"])
INITIALISMS: dict[str, tuple[str, ...]] = {"en_us": _EN_INITIALISMS, "en_gb": _EN_INITIALISMS}
CAPITALS_ONLY: dict[str, frozenset[str]] = {"en_us": _EN_CAPITALS_ONLY, "en_gb": _EN_CAPITALS_ONLY}
# Words the G2P mispronounces, by English dialect (scyllasband/text_data/en.json "pronunciations"), matched in any case:
# "ay" and "eff" (letter names the G2P reads as "eye" and "ee eff eff"), "git" (heard as "NuGet" after "new"),
# "librispeech", "pytest".
WORDS: dict[str, dict[str, str]] = {dialect: dict(entries) for dialect, entries in TEXT_DATA["en"]["pronunciations"].items()}
PLURAL_SUFFIXES = ("'s", "’s", "s")
POSSESSIVE_SUFFIXES = ("'s", "’s")
_SIBILANTS = frozenset("szʃʒ")
_VOICELESS = frozenset("ptkfθ")


def possessive_phones(entry: Sequence[str]) -> tuple[str, ...]:
    """The possessive ending after a word's phones: "ɪz" after a sibilant, "s" after a voiceless consonant, else "z"."""
    last = next((phone for phone in reversed(entry) if phone not in ("ˈ", "ˌ", "ː")), "")
    return ("ɪ", "z") if last in _SIBILANTS else ("s",) if last in _VOICELESS else ("z",)
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
        self._vocabulary = vocabulary
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
        """The entry for one whitespace token, or None. ``initialisms`` is False inside text written all in capitals, where
        only initialisms that are never words are spelled."""
        words = self.words.get(language)
        if words:
            entry = words.get(token.lower())
            if entry:
                return entry
            for suffix in POSSESSIVE_SUFFIXES:
                if token.lower().endswith(suffix):
                    entry = words.get(token.lower()[:-len(suffix)])
                    if entry:
                        return entry + possessive_phones(entry)   # "Garcia's"
        table = self.initialisms.get(language)
        if table:
            entry = self._initialism(token, language, table, initialisms)
            if entry:
                return entry
            for suffix in PLURAL_SUFFIXES:
                if token.endswith(suffix):
                    entry = self._initialism(token[:-len(suffix)], language, table, initialisms)
                    if entry:
                        return entry + (PLURAL_PHONE,)
        return None

    def _initialism(self, token: str, language: str, table: Mapping[str, tuple[str, ...]],
                    mixed_case: bool) -> tuple[str, ...] | None:
        upper = token.upper()
        if upper in table and upper not in CAPITALS_ONLY.get(language, ()) and token in (upper, token.lower()):
            return table[upper]   # never a word: capitals or lower case, also in text written all in capitals
        if not mixed_case or token != upper:
            return None
        if token in table:
            return table[token]
        letters = LETTERS.get(language)
        if (letters and 2 <= len(token) <= SPELLED_MAX_LETTERS and token.isascii() and token.isalpha() and token.isupper()
                and not _VOWEL_LETTERS.intersection(token)):
            return split_phones(initialism_phones(token, letters), self._vocabulary)
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
