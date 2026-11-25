"""Runtime pronunciation lexicon: words and initialisms the G2P gets wrong, applied over its word groups.

The G2P reads whole phrases, which keeps connected-speech forms (unstressed "to", linking r, French liaison); a full word
dictionary would undo them. This lexicon is small and targeted:

- initialisms, the G2P's weakest class: in a long phrase it can drop a letter ("GPU" said as "GU"). The English list is
  in scyllasband/text_data/en.json: ``spell`` entries are never words, so they match in capitals or in lower case, also
  inside text written all in capitals; ``capitals_only`` entries are also words ("US", "IT"), so they match only in
  capitals inside mixed-case text, as does any other word of capitals with no vowel. A plural or possessive "s" may
  follow;
- words it mispronounces (``WORDS``, from text_data/en.json "pronunciations"), matched in any case;
- words whose reading depends on the words around them (``CONTEXT_WORDS``, from text_data/<language>.json
  "context_pronunciations"): French numbers, where the G2P drops the t of "vingt" in "vingt deux" and of a final "huit".

- on a v5 bundle, the non-verbal sounds as the model was trained on them (text_data/en.json "nonverbal_phones", the
  trainer's phone table): a listed word standing alone takes the table's phones, and a sound the normalizer writes as
  several words ("Uh-uh" -> "uh uh") takes one part per word, so the break between its parts is a word boundary.

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
# Words read by context, by G2P language: an entry applies to a word whose previous word is in ``after``, whose next word is
# in ``before``, or (``phrase_end``) that ends its phrase; the first entry that applies wins.
CONTEXT_WORDS: dict[str, tuple[Mapping[str, Any], ...]] = {
    language: tuple(data["context_pronunciations"]) for language, data in TEXT_DATA.items() if data.get("context_pronunciations")}
# Non-verbal sounds of the v5 models: {word: symbols separated by spaces, " | " between the parts of a sound typed as
# several words}, the word sequences that spell such a sound, and the dialects the table covers.
_NONVERBAL = TEXT_DATA["en"]["nonverbal_phones"]
NONVERBAL_WORDS: dict[str, str] = dict(_NONVERBAL["words"])
NONVERBAL_PHRASES: dict[str, str] = dict(_NONVERBAL["phrases"])
NONVERBAL_LANGUAGES = frozenset(_NONVERBAL["locales"])
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

    def __init__(self, vocabulary: Mapping[str, Any], bundle_words: Mapping[str, Any] | None = None, *,
                 nonverbal: bool = False) -> None:
        """``nonverbal``: also hold the v5 non-verbal table (``nonverbal_hits``); every symbol of it must be in the
        vocabulary."""
        self.words: dict[str, dict[str, tuple[str, ...]]] = {}
        self.initialisms: dict[str, dict[str, tuple[str, ...]]] = {}
        self._vocabulary = vocabulary
        self.nonverbal: dict[str, tuple[tuple[str, ...], ...]] = {}   # word -> its parts, each a tuple of phones
        self.nonverbal_phrases: dict[tuple[str, ...], str] = {}
        if nonverbal:
            for word, phones in NONVERBAL_WORDS.items():
                parts = tuple(tuple(part.split()) for part in phones.split("|"))
                unknown = sorted({phone for part in parts for phone in part if phone not in vocabulary})
                if unknown or not all(parts):
                    raise ValueError(f"Non-verbal entry {word!r} needs phones outside this bundle's vocabulary: {' '.join(unknown)}")
                self.nonverbal[word] = parts
            for phrase, word in NONVERBAL_PHRASES.items():
                tokens = tuple(phrase.lower().split())
                if len(self.nonverbal.get(word, ())) != len(tokens):
                    raise ValueError(f"Non-verbal phrase {phrase!r} needs an entry {word!r} with one part per word")
                self.nonverbal_phrases[tokens] = word
        for language, letters in LETTERS.items():
            table = {}
            for word in INITIALISMS.get(language, ()):
                phones = split_phones(initialism_phones(word, letters), vocabulary)
                if phones:
                    table[word] = phones
            self.initialisms[language] = table
        for language, entries in WORDS.items():
            self.words[language] = self._words(entries, vocabulary)
        # language -> [(word, phones, after, before, phrase_end)]
        self.context: dict[str, list[tuple[str, tuple[str, ...], frozenset[str], frozenset[str], bool]]] = {}
        for language, rules in CONTEXT_WORDS.items():
            for rule in rules:
                phones = split_phones(str(rule["phones"]), vocabulary)
                if phones:
                    self.context.setdefault(language, []).append((str(rule["word"]).lower(), phones, frozenset(rule.get("after", ())),
                                                                  frozenset(rule.get("before", ())), bool(rule.get("phrase_end"))))
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

    def nonverbal_hits(self, tokens: Sequence[str], language: str) -> dict[int, tuple[str, ...]]:
        """{token index: phones} for the whitespace tokens that are a listed non-verbal word standing alone, or part of a
        listed phrase (longest first; each word takes the entry's next part). A phrase's sound typed as one word ("haha")
        takes its parts joined. Empty without the table or outside its dialects."""
        if not self.nonverbal or language not in NONVERBAL_LANGUAGES:
            return {}
        split = [_split_token(token) for token in tokens]
        found: dict[int, tuple[str, ...]] = {}
        for phrase in sorted(self.nonverbal_phrases, key=len, reverse=True):
            size = len(phrase)
            for start in range(len(tokens) - size + 1):
                if any(start + k in found for k in range(size)):
                    continue
                if (all(split[start + k][1].lower() == phrase[k] for k in range(size))
                        and not any(split[start + k][2] for k in range(size - 1)) and not any(split[start + k][0] for k in range(1, size))):
                    for k, part in enumerate(self.nonverbal[self.nonverbal_phrases[phrase]]):
                        found[start + k] = part
        for index, (_, core, _) in enumerate(split):
            if index not in found and core.isalpha() and core.lower() in self.nonverbal:
                found[index] = tuple(phone for part in self.nonverbal[core.lower()] for phone in part)
        return found

    def context_hits(self, tokens: Sequence[str], language: str) -> dict[int, tuple[str, ...]]:
        """{token index: phones} for the whitespace tokens of one phrase that a context entry reads."""
        rules = self.context.get(language)
        if not rules:
            return {}
        words = [_split_token(token)[1].lower() for token in tokens]
        found: dict[int, tuple[str, ...]] = {}
        for index, word in enumerate(words):
            previous = words[index - 1] if index > 0 else ""
            following = words[index + 1] if index + 1 < len(words) else ""
            for rule_word, phones, after, before, phrase_end in rules:
                if (word == rule_word and (not after or previous in after) and (not before or following in before)
                        and (not phrase_end or index + 1 == len(words))):
                    found[index] = phones
                    break
        return found

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


def _split_token(token: str) -> tuple[str, str, str]:
    """A whitespace token as (leading marks, word, trailing marks): quotes and brackets around a word are not part of it."""
    start, end = 0, len(token)
    while start < end and not token[start].isalnum():
        start += 1
    while end > start and not token[end - 1].isalnum():
        end -= 1
    return token[:start], token[start:end], token[end:]


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
