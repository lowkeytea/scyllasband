"""Spoken-form text normalization for Scylla's Band.

The G2P model is trained on spoken-form text. Runtime inference applies the same
normalization before G2P for every language declared by the selected bundle.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping
import unicodedata

from .g2p_phrases import mark_tight_dashes
from .text_data import TEXT_DATA, forms as _written_forms


SPOKEN_TEXT_NORMALIZER_CONTRACT = {
    "version": "scyllasband_spoken_text_v2_2026_08_05",
    "training_phone_source": "espeak",
    "ok_spoken_form": "okay",
    "mixed_question_exclamation": "leading_mark",
    "homogeneous_terminal_runs": "single_mark",
    "dot_runs_two_or_more": "ellipsis_three_dots",
    "colon_semicolon_classes": "preserved",
    "lexical_hyphen": "space_no_pause",
    "syntactic_dash": "em_dash_pause",
    "punctuation_silence_target": "explicit_silence",
    "non_acoustic_modifiers": ["stress", "length", "unicode_combining"],
}
SPOKEN_TEXT_NORMALIZER_VERSION = str(SPOKEN_TEXT_NORMALIZER_CONTRACT["version"])
SPOKEN_TEXT_NORMALIZER_SHA256 = hashlib.sha256(
    json.dumps(
        SPOKEN_TEXT_NORMALIZER_CONTRACT,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

try:
    import inflect
except Exception as exc:  # pragma: no cover - environment guard
    inflect = None
    _INFLECT_IMPORT_ERROR = exc
else:
    _INFLECT_IMPORT_ERROR = None

try:
    from num2words import num2words as _num2words_raw
except Exception as exc:  # pragma: no cover - environment guard
    _num2words_raw = None
    _NUM2WORDS_IMPORT_ERROR = exc
else:
    _NUM2WORDS_IMPORT_ERROR = None


_EN_UNDER_20 = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
)
_EN_TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"}
_EN_ORDINALS = {
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth",
    7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth", 11: "eleventh", 12: "twelfth",
    13: "thirteenth", 14: "fourteenth", 15: "fifteenth", 16: "sixteenth",
    17: "seventeenth", 18: "eighteenth", 19: "nineteenth", 20: "twentieth",
    21: "twenty first", 22: "twenty second", 23: "twenty third", 24: "twenty fourth",
    25: "twenty fifth", 26: "twenty sixth", 27: "twenty seventh", 28: "twenty eighth",
    29: "twenty ninth", 30: "thirtieth", 31: "thirty first",
}
_ES_UNDER_30 = (
    "cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve",
    "diez", "once", "doce", "trece", "catorce", "quince", "dieciséis", "diecisiete",
    "dieciocho", "diecinueve", "veinte", "veintiuno", "veintidós", "veintitrés",
    "veinticuatro", "veinticinco", "veintiséis", "veintisiete", "veintiocho", "veintinueve",
)
_ES_TENS = {30: "treinta", 40: "cuarenta", 50: "cincuenta", 60: "sesenta", 70: "setenta", 80: "ochenta", 90: "noventa"}
_ES_HUNDREDS = {100: "cien", 200: "doscientos", 300: "trescientos", 400: "cuatrocientos", 500: "quinientos", 600: "seiscientos", 700: "setecientos", 800: "ochocientos", 900: "novecientos"}
_IT_UNDER_20 = (
    "zero", "uno", "due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove",
    "dieci", "undici", "dodici", "tredici", "quattordici", "quindici", "sedici",
    "diciassette", "diciotto", "diciannove",
)
_IT_TENS = {20: "venti", 30: "trenta", 40: "quaranta", 50: "cinquanta", 60: "sessanta", 70: "settanta", 80: "ottanta", 90: "novanta"}

_FR_UNDER_20 = (
    "zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf",
    "dix", "onze", "douze", "treize", "quatorze", "quinze", "seize", "dix-sept",
    "dix-huit", "dix-neuf",
)
_FR_TENS = {20: "vingt", 30: "trente", 40: "quarante", 50: "cinquante", 60: "soixante"}
_DE_UNDER_20 = (
    "null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun",
    "zehn", "elf", "zwölf", "dreizehn", "vierzehn", "fünfzehn", "sechzehn",
    "siebzehn", "achtzehn", "neunzehn",
)
_DE_TENS = {20: "zwanzig", 30: "dreißig", 40: "vierzig", 50: "fünfzig", 60: "sechzig", 70: "siebzig", 80: "achtzig", 90: "neunzig"}
_VI_DIGITS = ("không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín")


def _fallback_number_to_words(value: int, lang: str, *, ordinal: bool = False) -> str:
    n = int(value)
    if ordinal and lang == "en" and n in _EN_ORDINALS:
        return _EN_ORDINALS[n]
    if ordinal and lang in {"fr", "de", "vi"}:
        return _fallback_ordinal_number(n, lang)
    if lang == "es":
        return _fallback_spanish_number(n)
    if lang == "it":
        return _fallback_italian_number(n)
    if lang == "fr":
        return _fallback_french_number(n)
    if lang == "de":
        return _fallback_german_number(n)
    if lang == "vi":
        return _fallback_vietnamese_number(n)
    return _fallback_english_number(n)


def _fallback_english_number(n: int) -> str:
    if n < 0:
        return "minus " + _fallback_english_number(-n)
    if n < 20:
        return _EN_UNDER_20[n]
    if n < 100:
        tens = (n // 10) * 10
        rest = n % 10
        return _EN_TENS[tens] if rest == 0 else f"{_EN_TENS[tens]} {_fallback_english_number(rest)}"
    if n < 1000:
        rest = n % 100
        head = f"{_fallback_english_number(n // 100)} hundred"
        return head if rest == 0 else f"{head} {_fallback_english_number(rest)}"
    for scale, name in ((1_000_000_000, "billion"), (1_000_000, "million"), (1000, "thousand")):
        if n >= scale:
            rest = n % scale
            head = f"{_fallback_english_number(n // scale)} {name}"
            return head if rest == 0 else f"{head} {_fallback_english_number(rest)}"
    return str(n)


def _fallback_spanish_number(n: int) -> str:
    if n < 0:
        return "menos " + _fallback_spanish_number(-n)
    if n < 30:
        return _ES_UNDER_30[n]
    if n < 100:
        tens = (n // 10) * 10
        rest = n % 10
        return _ES_TENS[tens] if rest == 0 else f"{_ES_TENS[tens]} y {_fallback_spanish_number(rest)}"
    if n < 1000:
        if n in _ES_HUNDREDS:
            return _ES_HUNDREDS[n]
        return f"ciento {_fallback_spanish_number(n % 100)}" if n < 200 else f"{_ES_HUNDREDS[(n // 100) * 100]} {_fallback_spanish_number(n % 100)}"
    if n < 1_000_000:
        rest = n % 1000
        head = "mil" if n // 1000 == 1 else f"{_fallback_spanish_number(n // 1000)} mil"
        return head if rest == 0 else f"{head} {_fallback_spanish_number(rest)}"
    rest = n % 1_000_000
    head = "un millón" if n // 1_000_000 == 1 else f"{_fallback_spanish_number(n // 1_000_000)} millones"
    return head if rest == 0 else f"{head} {_fallback_spanish_number(rest)}"


def _fallback_italian_number(n: int) -> str:
    """Italian writes numbers below a million as one word ("trentacinque", "duemilacinquecento"), with an accent on a final
    "tre" ("ventitré")."""
    words = _italian_number_words(n)
    return words[:-3] + "tré" if len(words) > 3 and words.endswith("tre") and not words.endswith(" tre") else words


def _italian_number_words(n: int) -> str:
    if n < 0:
        return "meno " + _italian_number_words(-n)
    if n < 20:
        return _IT_UNDER_20[n]
    if n < 100:
        tens, rest = (n // 10) * 10, n % 10
        if rest == 0:
            return _IT_TENS[tens]
        return (_IT_TENS[tens][:-1] if rest in (1, 8) else _IT_TENS[tens]) + _IT_UNDER_20[rest]   # "ventuno", "trentotto"
    if n < 1000:
        head = "cento" if n // 100 == 1 else _IT_UNDER_20[n // 100] + "cento"
        tail = _italian_number_words(n % 100) if n % 100 else ""
        return (head[:-1] if tail.startswith("ott") else head) + tail   # "centotto", "duecentottanta"
    if n < 1_000_000:
        head = "mille" if n // 1000 == 1 else _italian_number_words(n // 1000) + "mila"
        return head + (_italian_number_words(n % 1000) if n % 1000 else "")
    rest = n % 1_000_000
    head = "un milione" if n // 1_000_000 == 1 else f"{_italian_number_words(n // 1_000_000)} milioni"
    return head if rest == 0 else f"{head} {_italian_number_words(rest)}"


def _fallback_french_number(n: int) -> str:
    if n < 0:
        return "moins " + _fallback_french_number(-n)
    if n < 20:
        return _FR_UNDER_20[n]
    if n < 70:
        tens = (n // 10) * 10
        rest = n % 10
        if rest == 0:
            return _FR_TENS[tens]
        joiner = " et " if rest == 1 else "-"
        return f"{_FR_TENS[tens]}{joiner}{_fallback_french_number(rest)}"
    if n < 80:
        rest = n - 60
        joiner = " et " if rest == 11 else "-"
        return f"soixante{joiner}{_fallback_french_number(rest)}"
    if n < 100:
        rest = n - 80
        if rest == 0:
            return "quatre-vingts"
        return f"quatre-vingt-{_fallback_french_number(rest)}"
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        head = "cent" if hundreds == 1 else f"{_fallback_french_number(hundreds)} cent"
        if rest == 0:
            return head if hundreds == 1 else head + "s"
        return f"{head} {_fallback_french_number(rest)}"
    for scale, singular, plural in (
        (1_000_000_000, "milliard", "milliards"),
        (1_000_000, "million", "millions"),
        (1000, "mille", "mille"),
    ):
        if n >= scale:
            count, rest = divmod(n, scale)
            if scale == 1000 and count == 1:
                head = singular
            else:
                unit = singular if count == 1 else plural
                head = f"{_fallback_french_number(count)} {unit}"
            return head if rest == 0 else f"{head} {_fallback_french_number(rest)}"
    return str(n)


def _fallback_german_number(n: int) -> str:
    if n < 0:
        return "minus " + _fallback_german_number(-n)
    if n < 20:
        return _DE_UNDER_20[n]
    if n < 100:
        tens = (n // 10) * 10
        rest = n % 10
        if rest == 0:
            return _DE_TENS[tens]
        unit = "ein" if rest == 1 else _fallback_german_number(rest)
        return f"{unit}und{_DE_TENS[tens]}"
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        head = "einhundert" if hundreds == 1 else f"{_fallback_german_number(hundreds)}hundert"
        return head if rest == 0 else f"{head}{_fallback_german_number(rest)}"
    if n < 1_000_000:
        thousands, rest = divmod(n, 1000)
        head = "eintausend" if thousands == 1 else f"{_fallback_german_number(thousands)}tausend"
        return head if rest == 0 else f"{head}{_fallback_german_number(rest)}"
    for scale, singular, plural in (
        (1_000_000_000, "Milliarde", "Milliarden"),
        (1_000_000, "Million", "Millionen"),
    ):
        if n >= scale:
            count, rest = divmod(n, scale)
            head = f"eine {singular}" if count == 1 else f"{_fallback_german_number(count)} {plural}"
            return head if rest == 0 else f"{head} {_fallback_german_number(rest)}"
    return str(n)


def _fallback_vietnamese_below_thousand(n: int, *, force_hundreds: bool = False) -> str:
    hundreds, rest = divmod(n, 100)
    words: list[str] = []
    if hundreds or force_hundreds:
        words.extend((_VI_DIGITS[hundreds], "trăm"))
        if 0 < rest < 10:
            words.append("linh")
    if rest >= 10:
        tens, unit = divmod(rest, 10)
        words.append("mười" if tens == 1 else f"{_VI_DIGITS[tens]} mươi")
        if unit:
            if unit == 1 and tens > 1:
                words.append("mốt")
            elif unit == 4 and tens > 1:
                words.append("tư")
            elif unit == 5:
                words.append("lăm")
            else:
                words.append(_VI_DIGITS[unit])
    elif rest:
        words.append(_VI_DIGITS[rest])
    return " ".join(words) if words else _VI_DIGITS[0]


def _fallback_vietnamese_number(n: int) -> str:
    if n < 0:
        return "âm " + _fallback_vietnamese_number(-n)
    if n < 1000:
        return _fallback_vietnamese_below_thousand(n)
    for scale, name in (
        (1_000_000_000, "tỷ"),
        (1_000_000, "triệu"),
        (1000, "nghìn"),
    ):
        if n >= scale:
            count, rest = divmod(n, scale)
            head = f"{_fallback_vietnamese_number(count)} {name}"
            if rest == 0:
                return head
            tail = (
                _fallback_vietnamese_below_thousand(rest, force_hundreds=True)
                if rest < 100
                else _fallback_vietnamese_number(rest)
            )
            return f"{head} {tail}"
    return str(n)


def _fallback_ordinal_number(n: int, lang: str) -> str:
    if lang == "fr":
        special = {1: "premier", 2: "deuxième", 5: "cinquième", 9: "neuvième"}
        if n in special:
            return special[n]
        cardinal = _fallback_french_number(n)
        if cardinal.endswith("e"):
            cardinal = cardinal[:-1]
        elif cardinal.endswith(("vingts", "cents", "millions", "milliards")):
            # Only plural scale words drop the "s" (quatre-vingtième); words
            # like "trois" keep it (troisième).
            cardinal = cardinal[:-1]
        return cardinal + "ième"
    if lang == "de":
        special = {1: "erste", 2: "zweite", 3: "dritte", 7: "siebte", 8: "achte"}
        if n in special:
            return special[n]
        suffix = "te" if n < 20 else "ste"
        return _fallback_german_number(n) + suffix
    if lang == "vi":
        special = {1: "thứ nhất", 4: "thứ tư"}
        return special.get(n, f"thứ {_fallback_vietnamese_number(n)}")
    return _fallback_number_to_words(n, lang)


# ---- spoken-form tables (text_data/<language>.json): matching shared by the normalizer rules ----


def _word(lang: str, key: str) -> str:
    """A word the rules insert, from the language's ``words`` table (English when the language has none)."""
    return (TEXT_DATA.get(lang) or {}).get("words", {}).get(key) or TEXT_DATA["en"]["words"][key]


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _form_pattern(form: str, left_boundary: bool) -> str:
    """One written form as a pattern for a whole token: a space after a dot is optional, other spaces match any run."""
    parts = []
    for index, char in enumerate(form):
        if char == " ":
            parts.append(r"\s*" if index and form[index - 1] == "." else r"\s+")
        else:
            parts.append(re.escape(char))
    left = r"(?<![\w.'’])" if left_boundary and form[0].isalnum() else ""
    right = r"(?!\w)" if form[-1].isalnum() else ""
    return left + "".join(parts) + right


def _sentence_start(text: str, start: int) -> bool:
    index = start
    while index > 0 and (text[index - 1].isspace() or text[index - 1] in "\"'“‘«(¿¡["):
        index -= 1
    return index == 0 or text[index - 1] in ".!?…" or "\n" in text[index:start]


def _capitals_text(text: str) -> bool:
    """Text written all in capitals (and with at least one letter)."""
    return any(ch.isupper() for ch in text) and not any(ch.islower() for ch in text)


def _case_matches(matched: str, form: str, case: str, text: str, start: int, capitals_text: bool) -> bool:
    """Whether ``matched`` (spaces removed) is an accepted spelling of ``form`` under the entry's ``case`` policy:
    "any" any casing; "exact" as written (in capitals too when the whole text is); "sentence" as written or in lower case,
    capitalised at the start of a sentence, in capitals when the whole text is."""
    if matched == form or case == "any":
        return True
    if case == "exact":
        return capitals_text and matched == form.upper()
    if matched == form.lower() or (capitals_text and matched == form.upper()):
        return True
    if matched in (form[:1].upper() + form[1:], form[:1].upper() + form[1:].lower()):
        return _sentence_start(text, start)
    return False


def _cased(spoken: str, matched: str) -> str:
    """The replacement, with a capital first letter when the written form had one."""
    if matched[:1].isupper() and spoken[:1].islower():
        return spoken[:1].upper() + spoken[1:]
    return spoken


def _after_capitalised_word(text: str, start: int) -> bool:
    before = text[:start].rstrip()
    word = before.split()[-1].strip("\"'“‘«(") if before else ""
    return bool(word) and word[0].isupper() and word[-1].isalpha()


class _FormTable:
    """One section of a language's tables: every written form in one pattern, longest first, and the entries by their
    lower-case form with spaces removed."""

    def __init__(self, entries: Any, default_case: str, *, after_number: bool = False) -> None:
        self.keys: dict[str, list[tuple[Mapping[str, Any], str, str]]] = {}
        written: list[str] = []
        for entry in entries:
            for form in _written_forms(entry["written"]):
                written.append(form)
                self.keys.setdefault(_compact(form).lower(), []).append((entry, form, entry.get("case", default_case)))
        written.sort(key=lambda form: (-len(form), form))
        alternatives = "|".join(_form_pattern(form, not after_number) for form in written)
        if not written:
            self.pattern = None
        elif after_number:
            # A unit after a number, with or without one space: "5 km", "100ms", "1,5 t".
            self.pattern = re.compile(rf"(?<![\w.,])(\d+(?:[.,]\d+)*)( ?)({alternatives})", re.IGNORECASE)
        else:
            self.pattern = re.compile(alternatives, re.IGNORECASE)

    def find(self, text: str, matched: str, start: int, capitals: bool) -> tuple[Mapping[str, Any], str] | None:
        compact = _compact(matched)
        for entry, form, case in self.keys.get(compact.lower(), ()):
            if _case_matches(compact, _compact(form), case, text, start, capitals):
                return entry, form
        return None


_TABLES: dict[str, dict[str, _FormTable]] = {}


def _tables(lang: str) -> dict[str, _FormTable]:
    tables = _TABLES.get(lang)
    if tables is None:
        data = TEXT_DATA.get(lang, {})
        tables = {
            "abbreviations": _FormTable(data.get("abbreviations", ()), "sentence"),
            "titles": _FormTable(data.get("titles", ()), "sentence"),
            "before_number": _FormTable(data.get("before_number", ()), "any"),
            "units": _FormTable(data.get("units", ()), "any", after_number=True),
        }
        _TABLES[lang] = tables
    return tables


_UNIT_FORMS: dict[str, frozenset[str]] = {}


def _unit_words_of(lang: str) -> frozenset[str]:
    """The language's unit forms in lower case: a unit in capitals ("300 MPH") is not an initialism."""
    if lang not in _UNIT_FORMS:
        units = TEXT_DATA.get(lang, {}).get("units", ())
        _UNIT_FORMS[lang] = frozenset(form.lower().rstrip(".") for entry in units for form in _written_forms(entry["written"]))
    return _UNIT_FORMS[lang]


_ORDINAL_SUFFIX_PATTERNS: dict[str, re.Pattern[str]] = {}


def _ordinal_suffix_pattern(lang: str) -> re.Pattern[str]:
    """A number with one of the language's ordinal suffixes: "2nd", "1er", "XXe", "1.º", "3ª"."""
    if lang not in _ORDINAL_SUFFIX_PATTERNS:
        suffixes = sorted(TEXT_DATA[lang]["ordinal_suffixes"], key=lambda suffix: (-len(suffix), suffix))
        _ORDINAL_SUFFIX_PATTERNS[lang] = re.compile(
            r"(?<![\w.,])(\d+)(" + "|".join(map(re.escape, suffixes)) + r")(?!\w)", re.IGNORECASE)
    return _ORDINAL_SUFFIX_PATTERNS[lang]


_CURRENCY_PATTERNS: dict[str, tuple[re.Pattern[str], re.Pattern[str]]] = {}


def _currency_patterns(lang: str) -> tuple[re.Pattern[str], re.Pattern[str]]:
    """A currency symbol before an amount (with an optional scale word: "$5bn") and after one ("40€", "40 EUR")."""
    if lang not in _CURRENCY_PATTERNS:
        data = TEXT_DATA[lang]
        symbols = "".join(re.escape(symbol) for symbol in data["currency"])
        scales = sorted(data.get("currency_scales", {}), key=lambda scale: (-len(scale), scale))
        scale = rf"(?:\s?({'|'.join(map(re.escape, scales))})(?!\w))?" if scales else "()"
        amount = r"([0-9](?:[0-9.,]*[0-9])?)"
        _CURRENCY_PATTERNS[lang] = (
            re.compile(rf"([{symbols}])\s*{amount}{scale}"),
            re.compile(rf"(?<![\w.,]){amount}\s*([{symbols}]|EUR)(?!\w)", re.IGNORECASE),
        )
    return _CURRENCY_PATTERNS[lang]


_ENGLISH_YEAR_PATTERN: list[re.Pattern[str]] = []


def _english_year_pattern() -> re.Pattern[str]:
    """A year 1100-2099 after an English year word ("in 1776", "since 1990", "July 1969"), read "seventeen seventy six"."""
    if not _ENGLISH_YEAR_PATTERN:
        cues = "|".join(map(re.escape, TEXT_DATA["en"].get("year_cues", ())))
        _ENGLISH_YEAR_PATTERN.append(re.compile(rf"\b({cues})\s+(1[1-9]\d\d|20\d\d)(?![\d.,]\d|\w)", re.IGNORECASE))
    return _ENGLISH_YEAR_PATTERN[0]


_GERMAN_YEAR_PATTERN: list[re.Pattern[str]] = []


def _german_year_pattern() -> re.Pattern[str]:
    """A year 1100-1999 after a German month or another year word ("Oktober 1990", "im Jahr 1759")."""
    if not _GERMAN_YEAR_PATTERN:
        cues = "|".join(map(re.escape, TEXT_DATA["de"].get("year_cues", ())))
        _GERMAN_YEAR_PATTERN.append(re.compile(rf"\b({cues})\s+(1[1-9]\d\d)(?![\d.,]\d|\w)"))
    return _GERMAN_YEAR_PATTERN[0]


@dataclass(frozen=True)
class SpokenTextNormalizerConfig:
    expand_currency: bool = True
    expand_numbers: bool = True
    expand_ordinals: bool = True
    expand_percentages: bool = True
    expand_times: bool = True
    expand_dates: bool = True
    normalize_at_sign: bool = True
    normalize_punctuation: bool = True


class SpokenTextNormalizer:
    """Normalize raw text into the form the G2P model should consume."""

    _LANG_BY_MODEL_LANGUAGE = {
        "en": "en",
        "en_us": "en",
        "en_gb": "en",
        "es": "es",
        "es_mx": "es",
        "es_es": "es",
        "it": "it",
        "it_it": "it",
        "fr": "fr",
        "fr_fr": "fr",
        "de": "de",
        "de_de": "de",
        "vi": "vi",
        "vi_vn": "vi",
        "vi_hn": "vi",
    }
    _UNICODE_TRANSLATION = str.maketrans(
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
            "\u2032": "'",
            "\uff07": "'",
            "\u00ab": '"',
            "\u00bb": '"',
            "\u201c": '"',
            "\u201d": '"',
            "\u201e": '"',
            "\u201f": '"',
            "\u2033": '"',
            "\uff02": '"',
            "\u2026": "...",
            "\u2044": "/",
            "\u2215": "/",
            "\u00bc": " 1/4 ",
            "\u00bd": " 1/2 ",
            "\u00be": " 3/4 ",
        }
    )
    _ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200d\ufeff]")
    _WHITESPACE_RE = re.compile(r"\s+")
    _EMAIL_RE = re.compile(r"\b(?P<local>[A-Za-z0-9._%+\-]+)@(?P<domain>[A-Za-z0-9.\-]+\.[A-Za-z]{2,})\b")
    _MIXED_FRACTION_RE = re.compile(r"\b(\d+)\s+(\d+/\d+)\b")
    _SIMPLE_FRACTION_RE = re.compile(r"\b(\d+)\s*/\s*(\d+)\b")
    _DOTTED_INITIALISM_RE = re.compile(r"(?<![A-Za-zÀ-ÖØ-öø-ÿÑñ])(?:[A-Za-zÑñ]\.){2,}")
    _EN_OK_RE = re.compile(r"(?<![A-Za-z0-9])ok(?![A-Za-z0-9])", re.IGNORECASE)
    _MIXED_TERMINAL_PUNCTUATION_RE = re.compile(r"([!?])[!?]+")
    _DOT_RUN_RE = re.compile(r"\.{2,}")
    _LEXICAL_HYPHENS = ("\u2010", "\u2011")
    _SYNTACTIC_DASHES = ("\u2012", "\u2013", "\u2014", "\u2015", "\u2212", "\ufe58", "\ufe63", "\uff0d")
    _EN_MONTH_DATE_RE = re.compile(
        r"\b("
        r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
        r")\.?\s+(\d{1,2})(?:,\s*(\d{2,4}))?\b",
        re.IGNORECASE,
    )
    # Letters glued to digits ("V3", "MP3", "H2O", "4K", "Python3") reach the G2P as one unknown word and come out garbled.
    # They are split into letter and digit runs, the digits left for the number rules that follow ("v3.5" becomes "vee
    # three point five"). A single letter, and a run of up to three letters that is all capitals or has no vowel ("GPT",
    # "fps"), is spelled with letter names; longer runs ("COVID", "iPhone", "Python") stay words. Identifiers joined by "_"
    # are left alone.
    _ALPHANUMERIC_RE = re.compile(r"(?<!\w)[A-Za-z0-9]*(?:[A-Za-z][0-9]|[0-9][A-Za-z])[A-Za-z0-9]*(?!\w)")
    _ALPHANUMERIC_RUN_RE = re.compile(r"[A-Za-z]+|[0-9]+")
    _VOWEL_RE = re.compile(r"[AEIOUYaeiouy]")
    # English numbers glued to "x" or to a plural "s"; the digits are left for the number rules.
    _EN_NUMBER = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
    # Multipliers and sizes: "68x" becomes "68 times", "4x4" becomes "4 by 4".
    _EN_TIMES_RE = re.compile(rf"(?<![\w.,])({_EN_NUMBER})[xX×](\d+(?:\.\d+)?)?(?!\w)")
    # Plural numbers: "4s" fours, "100s" hundreds, "1990s" and "1990's" nineteen nineties, "'80s" eighties; a decimal is
    # seconds ("1.5s"). With an apostrophe only a decade is plural; "66's" keeps its possessive.
    _EN_PLURAL_NUMBER_RE = re.compile(rf"(?<![\w'’.,])(['’]?)({_EN_NUMBER})(['’]?)s(?!\w)")
    _EN_ROUND_PLURALS = {100: "hundreds", 1000: "thousands", 1000000: "millions", 1000000000: "billions"}
    # English clock times without minutes ("10am", "5 PM"); "a.m." is already spelled by the dotted-initialism rule.
    _EN_BARE_AMPM_RE = re.compile(r"(?<![\w.,:])(\d{1,2})\s*([AaPp])[Mm](?!\w)")
    # Callbacks check Unicode word boundaries, including combining marks.
    _DASH_LEFT_SINGLE_LETTER_RE = re.compile(r"([A-Za-zÑñ])\s*-\s*")
    _DASH_RIGHT_SINGLE_LETTER_RE = re.compile(r"\s*-\s*([A-Za-zÑñ])")
    def __init__(self, config: SpokenTextNormalizerConfig | None = None) -> None:
        self.config = config or SpokenTextNormalizerConfig()
        self._inflect = inflect.engine() if inflect is not None else None

    def normalize(self, text: str, *, language: str) -> str:
        lang = self._normalizer_language(language)
        value = unicodedata.normalize("NFC", str(text or ""))
        if self.config.normalize_punctuation:
            value = self._normalize_punctuation(value, lang)
        if lang == "en":
            # eSpeak and the distilled G2P read bare ``ok`` as /oʊk/ ("oak").
            # The spoken lexical form is "okay"; keep following punctuation
            # untouched so ``Ok...`` retains its ellipsis boundary.
            value = self._EN_OK_RE.sub("okay", value)
        if self.config.normalize_at_sign:
            value = self._normalize_at_symbols(value, lang)
        # Text written all in capitals: casing tells nothing about names or initialisms, and the result stays in capitals.
        capitals = _capitals_text(value)
        value = self._expand_abbreviations(value, lang, capitals)
        value = self._expand_symbols(value, lang)
        value = self._expand_dotted_initialisms(value, lang)
        if lang in self._SPACE_GROUPED_LANGUAGES:
            value = self._SPACE_GROUPED_RE.sub(lambda m: m.group(0).replace(" ", ""), value)
        if self.config.expand_currency:
            value = self._expand_currency(value, lang)
        if self.config.expand_numbers:
            # Units before titles and initialisms: "7 SQ FT" and "300 MPH" are units, not "Fort" or a spelled initialism.
            value = self._expand_units(value, lang, capitals)
        value = self._expand_titles(value, lang, capitals)
        value = self._expand_before_number(value, lang, capitals)
        value = self._NAME_INITIAL_RE.sub(lambda m: self._expand_name_initial(m, lang), value)
        value = self._SPACED_INITIALS_RE.sub(lambda m: self._expand_spaced_initials(m, lang), value)
        if lang in self._ROMAN_LANGUAGES and not capitals:
            value = self._expand_roman_numerals(value, lang)
        if lang in self._INITIALISM_LANGUAGES:
            value = self._LETTER_WORD_RE.sub(lambda m: self._expand_initialism(m, lang, capitals), value)
        if lang == "de" and self.config.expand_dates:
            value = self._GERMAN_DATE_RE.sub(self._expand_german_date, value)
        value = (self._EN_NUMBER_RANGE_RE if lang == "en" else self._NUMBER_RANGE_RE).sub(
            lambda m: self._expand_number_range(m, lang), value)
        if self.config.expand_times:
            value = re.sub(r"\b(\d{1,2}):(\d{2})(?:\s*([AaPp][Mm]))?\b", lambda m: self._expand_time(m, lang), value)
            if lang == "en":
                value = self._EN_BARE_AMPM_RE.sub(lambda m: self._expand_bare_ampm(m, lang), value)
            if lang in self._CLOCK_LANGUAGES:
                value = self._CLOCK_RE.sub(lambda m: self._expand_clock(m, lang), value)
        if self.config.expand_percentages:
            # "31%" and, as French and German print it, "31 %"; "0.5 ‰".
            for sign, key in (("%", "percent"), ("‰", "permille")):
                value = re.sub(rf"(?<![\d.,])(\d[\d.,]*) ?{sign}",
                               lambda m, key=key: f"{self._decimal_to_words(m.group(1), lang)} {_word(lang, key)}", value)
        if lang == "en" and self.config.expand_dates:
            value = _english_year_pattern().sub(lambda m: f"{m.group(1)} {self._year_words(int(m.group(2)))}", value)
        if self.config.expand_ordinals:
            value = self._expand_ordinal_suffixes(value, lang)
            if lang == "de":
                value = self._GERMAN_ORDINAL_RE.sub(self._expand_german_ordinal, value)
                value = _german_year_pattern().sub(lambda m: f"{m.group(1)} {self._german_year(int(m.group(2)))}", value)
        if self.config.expand_numbers:
            if lang == "en":
                value = self._EN_TIMES_RE.sub(lambda m: f"{m.group(1)} by {m.group(2)}" if m.group(2) else f"{m.group(1)} times", value)
                value = self._EN_PLURAL_NUMBER_RE.sub(self._expand_plural_number, value)
            value = self._ALPHANUMERIC_RE.sub(lambda m: self._split_alphanumeric(m, lang), value)
            if self.config.expand_dates:
                value = self._EN_MONTH_DATE_RE.sub(lambda m: self._expand_named_month_date(m, lang), value)
                value = re.sub(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b", lambda m: self._expand_slash_date(m, lang), value)
                value = re.sub(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", lambda m: self._expand_date(m.group(2), m.group(3), m.group(1), lang), value)
            value = self._MIXED_FRACTION_RE.sub(lambda m: f"{m.group(1)} {self._and_word(lang)} {m.group(2)}", value)
            value = self._SIMPLE_FRACTION_RE.sub(lambda m: self._expand_fraction(m, lang), value)
            if lang in {"es", "it", "fr", "de", "vi"}:
                value = re.sub(r"\b\d[\d.]*,\d+\b", lambda m: self._decimal_to_words(m.group(0), lang), value)
                value = re.sub(r"\b\d{1,3}(?:\.\d{3})+\b", lambda m: self._to_words(int(self._whole_number_digits(m.group(0))), lang), value)
            value = re.sub(r"\b\d[\d,]*\.\d+\b(?!\.\d)", lambda m: self._decimal_to_words(m.group(0), lang), value)
            value = re.sub(r"\b\d{1,3}(?:,\d{3})+\b", lambda m: self._to_words(int(self._whole_number_digits(m.group(0))), lang), value)
            value = re.sub(r"\b\d+\b", lambda m: self._to_words(int(m.group(0)), lang), value)
        # "to/from", "and/or": the slash is not spoken; left in, the G2P reads it as stray consonants.
        value = self._WORD_SLASH_RE.sub(" ", value)
        value = self._expand_dash_letter_names(value, lang)
        # The target backend is dictionary-based, so hyphenated compounds and
        # number words are safer as separate lookup tokens. Inference uses the
        # same normalizer, preserving the train/runtime contract.
        value = value.replace("-", " ")
        if capitals:
            value = value.upper()
        return self._WHITESPACE_RE.sub(" ", value).strip()

    # A single capital with a dot inside a name: "W. C. Handy", "Kenneth A. Schmied", "D. K. Arya". It is read as a letter
    # and loses its dot, which would otherwise end the sentence. The word after the initials must be capitalised and not
    # one that opens a sentence ("... in group B. The others ...").
    _NAME_INITIAL_RE = re.compile(r"(?<![A-Za-zÀ-ÖØ-öø-ÿ.'’])([A-Z])\.(?=\s+(?:[A-Z]\.\s+)*([A-ZÀ-ÖØ-Þ][a-zß-öø-ÿ'’]+))")
    # Two or three initials not followed by a name: "Washington D. C. was"; "J. S. Bach" is taken by the rule above.
    _SPACED_INITIALS_RE = re.compile(r"(?<![A-Za-zÀ-ÖØ-öø-ÿ.'’])[A-Z]\.(?: [A-Z]\.){1,2}(?![A-Za-zÀ-ÖØ-öø-ÿ])")
    # "4.2-3.9 million", "1939-1945", "pages 10-15": a range is read with the language's "to". Left alone: descending whole
    # numbers (scores, "3-2") and a three-digit group before a four-digit one (telephone numbers).
    _EN_NUMBER_RANGE_RE = re.compile(r"(?<![\w.,-])(\d+(?:\.\d+)?)\s?-\s?(\d+(?:\.\d+)?)(?![\w-])(?!\.\d)")
    _NUMBER_RANGE_RE = re.compile(r"(?<![\w.,-])(\d+(?:,\d+)?)\s?-\s?(\d+(?:,\d+)?)(?![\w-])(?![.,]\d)")
    _DEGREE_SCALE_RE = re.compile(r"\s?°\s?([CF])(?![A-Za-z])")
    # A degree sign right after a letter is the French and Italian number sign ("n° 5"), not degrees.
    _DEGREE_RE = re.compile(r"(?<![^\W\d_])°")
    _WORD_SLASH_RE = re.compile(r"(?<=[^\W\d_])\s?/\s?(?=[^\W\d_])")
    # "5 000 000", "104 500": thousands set apart by spaces, as French, Spanish, Italian and German print them. Not after
    # another number ("0800 123 456").
    _SPACE_GROUPED_LANGUAGES = frozenset({"fr", "es", "it", "de"})
    _SPACE_GROUPED_RE = re.compile(r"(?<![\w.,])(?<!\d )[1-9]\d{0,2}(?: \d{3})+(?![\w])(?! \d)")
    # Clock times written with "h": French "7 h 19", "21 h", "7h30"; Vietnamese "5h30".
    _CLOCK_LANGUAGES = frozenset({"fr", "vi"})
    _CLOCK_RE = re.compile(r"(?<![\w.,])(\d{1,2}) ?h(?: ?(\d{2}))?(?![\w])")
    # Initialisms outside English (English ones are spelled by the pronunciation lexicon): a word of two to eight letters
    # checked against the language's `initialisms` and `spell_unlisted`.
    _INITIALISM_LANGUAGES = frozenset({"de", "fr", "es", "it"})
    _LETTER_WORD_RE = re.compile(r"(?<![\w.])([^\W\d_]{2,8})(?![\w'’]|-?\d)")
    # Roman numerals: centuries ("XVIIe siècle", "siglo XX", "XV secolo") and after a name ("Isabel II", "Henry VIII"). Only
    # I, V and X, up to 39.
    _ROMAN_LANGUAGES = frozenset({"en", "fr", "es", "it"})
    _EN_ROMAN_RE = re.compile(r"(?<![\w.'’])(X{0,3}(?:IX|IV|V?I{0,3}))(?=(?:['’]s)?(?![\w'’]))")
    _ROMAN_RE = re.compile(r"(?<![\w.'’])(X{0,3}(?:IX|IV|V?I{0,3}))(er|re|e|ème)?(?![\w'’])")
    # German ordinals are written with a dot ("am 3. Oktober", "der 2. Weltkrieg"); left alone the dot ends the sentence.
    _GERMAN_ORDINAL_RE = re.compile(r"(?<![\w.,])(\d{1,3})\.(?=\s+([^\W\d_][\w-]*))")
    _GERMAN_DATE_RE = re.compile(r"(?<![\w.,])(\d{1,2})\.\s?(\d{1,2})\.\s?(\d{4})(?!\w|\.\d)")
    _NUMBER_FOLLOWS_RE = re.compile(r"\s*\d")
    _NEXT_WORD_RE = re.compile(r"\s+[\"'“‘«(\[]?([^\W\d_][\w'’-]*)")

    @staticmethod
    def _data(lang: str) -> dict[str, Any]:
        return TEXT_DATA.get(lang) or TEXT_DATA["en"]

    @classmethod
    def _sentence_follows(cls, text: str, end: int, lang: str) -> bool:
        """Whether a sentence ends at ``end``: the text ends there, or a capitalised word follows (in German, where every
        noun is capitalised, a word from the language's ``sentence_openers``)."""
        if not text[end:].strip(" \t\r\n\"'”’»)]"):
            return True
        match = cls._NEXT_WORD_RE.match(text, end)
        if not match or not match.group(1)[0].isupper():
            return False
        following = text[match.start(1):].split(maxsplit=1)[0].rstrip(",;:!?)\"'”’»").lower()
        tables = _tables(lang)
        if any(following in tables[name].keys for name in ("abbreviations", "titles")):
            return False   # "Smith Bros. Inc.": the next word is another abbreviation
        return lang != "de" or match.group(1) in cls._data(lang).get("sentence_openers", ())

    def _expand_abbreviations(self, text: str, lang: str, capitals: bool = False) -> str:
        table = _tables(lang)["abbreviations"]
        if table.pattern is None:
            return text

        def replace(match: re.Match[str]) -> str:
            found = table.find(match.string, match.group(0), match.start(), capitals)
            if found is None:
                return match.group(0)
            entry, form = found
            spoken = _cased(entry["spoken"], match.group(0))
            if entry.get("ends") and form.endswith(".") and self._sentence_follows(match.string, match.end(), lang):
                spoken += "."
            return spoken

        return table.pattern.sub(replace, text)

    def _expand_symbols(self, text: str, lang: str) -> str:
        for symbol, word in self._data(lang).get("symbols", {}).items():
            text = text.replace(symbol, f" {word} ")
        return text

    def _expand_titles(self, text: str, lang: str, capitals: bool = False) -> str:
        table = _tables(lang)["titles"]
        if table.pattern is None:
            return text

        def replace(match: re.Match[str]) -> str:
            found = table.find(match.string, match.group(0), match.start(), capitals)
            if found is None:
                return match.group(0)
            entry, form = found
            if re.search(r"\d\s?$", match.string[:match.start()]):
                return match.group(0)   # "7 FT", "5 MS": a unit after a number, not a title
            name = self._NEXT_WORD_RE.match(match.string, match.end())
            openers = self._data(lang).get("sentence_openers", ())
            if name and name.group(1)[0].isupper() and name.group(1) not in openers:
                return _cased(entry["spoken"], match.group(0))
            spoken = entry.get("after_name") if _after_capitalised_word(match.string, match.start()) else None
            spoken = spoken or entry.get("otherwise")
            if not spoken:
                return match.group(0)
            dot = "." if form.endswith(".") and self._sentence_follows(match.string, match.end(), lang) else ""
            return _cased(spoken, match.group(0)) + dot

        return table.pattern.sub(replace, text)

    def _expand_before_number(self, text: str, lang: str, capitals: bool = False) -> str:
        table = _tables(lang)["before_number"]
        if table.pattern is None:
            return text

        def replace(match: re.Match[str]) -> str:
            found = table.find(match.string, match.group(0), match.start(), capitals)
            if found is None or not self._NUMBER_FOLLOWS_RE.match(match.string, match.end()):
                return match.group(0)
            return _cased(found[0]["spoken"], match.group(0)) + " "

        return table.pattern.sub(replace, text)

    def _expand_units(self, text: str, lang: str, capitals: bool = False) -> str:
        table = _tables(lang)["units"]
        if table.pattern is None:
            return text

        def replace(match: re.Match[str]) -> str:
            found = table.find(match.string, match.group(3), match.start(3), capitals)
            if found is None:
                return match.group(0)
            entry, form = found
            dot = "." if form.endswith(".") and self._sentence_follows(match.string, match.end(), lang) else ""
            return self._unit_words(match.group(1), entry, lang) + dot

        return table.pattern.sub(replace, text)

    _GENDERED_ONE = frozenset({"de", "fr", "es", "it"})

    def _unit_words(self, raw: str, entry: Mapping[str, Any], lang: str) -> str:
        """A number and its unit: the unit in the singular for 1 (French: below 2), and the word for 1 agreeing with the
        unit's gender ("eine Tonne", "veintiún kilómetros", "vingt et une tonnes")."""
        whole, frac = self._split_decimal(raw, lang)
        value = int(self._whole_number_digits(whole) or "0")
        singular = (value == 1 and frac is None) or (lang == "fr" and value < 2)
        number = raw
        if frac is None and lang in self._GENDERED_ONE:
            gender = entry.get("gender", "m")
            if value == 1:
                number = self._data(lang)["one"][gender]
            elif value % 10 == 1 and value % 100 != 11 and lang in ("es", "fr"):
                words = self._to_words(value, lang)
                if lang == "es" and words.endswith("uno"):
                    number = words[:-3] + ("una" if gender == "f" else "ún" if words.endswith("veintiuno") else "un")
                elif lang == "fr" and gender == "f" and words.endswith("un"):
                    number = words + "e"
        return f"{number} {entry['one'] if singular else entry['many']}"

    def _expand_name_initial(self, match: re.Match[str], lang: str) -> str:
        letter, following = match.group(1), match.group(2)
        data = self._data(lang)
        if following in data.get("sentence_openers", ()):
            return match.group(0)
        before = match.string[:match.start()].rstrip()
        previous = before.split()[-1] if before else ""
        another_initial = match.string[match.end():].lstrip()[1:2] == "."
        leader = previous.lower() in data.get("name_leaders", ())
        in_name = not previous or previous[0].isupper() or leader or another_initial
        if not in_name or (lang == "en" and letter == "I" and not (previous and previous[0].isupper())):
            return match.group(0)
        return self._letter_name(letter, lang)

    def _expand_spaced_initials(self, match: re.Match[str], lang: str) -> str:
        spoken = " ".join(self._letter_name(letter, lang) for letter in match.group(0) if letter.isalpha())
        return spoken + "." if self._SENTENCE_FOLLOWS_RE.match(match.string, match.end()) else spoken

    def _expand_number_range(self, match: re.Match[str], lang: str) -> str:
        first, second = match.group(1), match.group(2)
        whole = first.isdigit() and second.isdigit()
        if whole and (int(second) <= int(first) or (len(first) == 3 and len(second) == 4) or first.startswith("0") or second.startswith("0")):
            return match.group(0)
        if lang == "en" and whole and len(first) == len(second) == 4 and 1100 <= int(first) and int(second) <= 2099:
            return f"{self._year_words(int(first))} to {self._year_words(int(second))}"   # "1939-1945"
        return f"{first} {_word(lang, 'range')} {second}"

    def _expand_initialism(self, match: re.Match[str], lang: str, capitals: bool) -> str:
        """Spell a word with letter names when the language's tables say it is an initialism. ``spell`` entries are never
        words, so they match in capitals or in lower case, also in text written all in capitals; ``capitals_only``
        entries and unlisted ones (``spell_unlisted``: capitals, no vowel) only in capitals inside mixed-case text."""
        word = match.group(1)
        upper = word.upper()
        data = self._data(lang)
        initialisms = data.get("initialisms", {})
        if not (upper in initialisms.get("spell", ()) and word in (upper, word.lower())):
            if word != upper or capitals:
                return word
            unlisted = data.get("spell_unlisted", {})
            vowels = unlisted.get("vowels", "")
            spelled = upper in initialisms.get("capitals_only", ()) or (
                unlisted.get("min", 0) <= len(word) <= unlisted.get("max", 0) and not any(ch in vowels for ch in word))
            after_number = re.search(r"\d\s?$", match.string[:match.start()]) is not None
            if not spelled or (after_number and word.lower() in _unit_words_of(lang)):
                return word
        return " ".join(self._letter_name(letter, lang) for letter in upper)

    @staticmethod
    def _roman_value(numeral: str) -> int:
        values = {"I": 1, "V": 5, "X": 10}
        total = 0
        for index, letter in enumerate(numeral):
            value = values[letter]
            total += -value if index + 1 < len(numeral) and values[numeral[index + 1]] > value else value
        return total

    def _ordinal_word(self, value: int, lang: str, kind: str = "m") -> str:
        """An ordinal number word: ``kind`` is the gender ("m", "f"), "short" for the Spanish forms before a noun ("primer",
        "tercer"), or a French "m1"/"f1"/"m2"/"f2" form for "premier"/"première"/"second"/"seconde"."""
        if lang == "fr":
            special = {("m1", 1): "premier", ("f1", 1): "première", ("m2", 2): "second", ("f2", 2): "seconde"}
            return special.get((kind, value)) or self._to_words(value, lang, ordinal=True)
        ordinals = self._data(lang).get("ordinals")
        if lang in ("es", "it") and ordinals:
            if value < 1:
                return self._to_words(value, lang)
            if value <= len(ordinals):
                word = ordinals[value - 1]
            elif lang == "it":
                cardinal = self._to_words(value, lang)
                word = (cardinal if cardinal.endswith(("tre", "tré", "sei")) else cardinal[:-1]).replace("tré", "tre") + "esimo"
            else:
                return self._to_words(value, lang)
            if kind == "short" and value in (1, 3):
                return word[:-1]
            return word[:-1] + "a" if kind == "f" and word.endswith("o") else word
        return self._to_words(value, lang, ordinal=True)

    def _expand_roman_numerals(self, text: str, lang: str) -> str:
        data = self._data(lang)

        def context(match: re.Match[str]) -> tuple[str, str, bool]:
            before = match.string[:match.start()].rstrip()
            tokens = [token.strip("\"'“‘«(,") for token in before.split()[-3:]]
            previous = tokens[-1] if tokens else ""
            named = len(previous) > 2 and previous[0].isupper() and previous[1:].islower()
            return previous, " ".join(tokens[:-1]), named

        if lang == "en":
            titles, names = set(data.get("regnal_titles", ())), set(data.get("regnal_names", ()))

            def english(match: re.Match[str]) -> str:
                numeral = match.group(1)
                previous, earlier, named = context(match)
                regnal = previous in names or any(word in titles for word in earlier.split())
                if not numeral or not named or (len(numeral) < 2 and not regnal):
                    return match.group(0)
                value = self._roman_value(numeral)
                return f"the {self._ordinal_word(value, lang)}" if regnal else self._to_words(value, lang)

            return self._EN_ROMAN_RE.sub(english, text)
        century = data["words"].get("century", "")
        feminine_titles = set(data.get("feminine_titles", ()))
        suffixes = data.get("ordinal_suffixes", {})

        def replace(match: re.Match[str]) -> str:
            numeral, suffix = match.group(1), match.group(2)
            if not numeral or (len(numeral) == 1 and not suffix):
                return match.group(0)
            value = self._roman_value(numeral)
            previous, earlier, named = context(match)
            following = match.string[match.end():].lstrip().split(" ")[0].strip(".,;:!?\"'()")
            if lang == "fr":
                if suffix:
                    return self._ordinal_word(value, lang, suffixes.get(suffix, "m"))
                return self._to_words(value, lang) if named else match.group(0)
            if suffix:
                return match.group(0)
            if previous.lower() == century:
                return self._to_words(value, lang) if lang == "es" else self._ordinal_word(value, lang)
            if lang == "it" and following.lower() == century:
                return self._ordinal_word(value, lang)
            if not named:
                return match.group(0)
            if lang == "es" and value > 10:
                return self._to_words(value, lang)   # Spanish reads rulers above ten with cardinals: "Alfonso trece"
            feminine = previous.endswith("a") or bool(earlier.split()) and earlier.split()[-1].lower() in feminine_titles
            return self._ordinal_word(value, lang, "f" if feminine else "m")

        return self._ROMAN_RE.sub(replace, text)

    def _expand_ordinal_suffixes(self, text: str, lang: str) -> str:
        suffixes = self._data(lang).get("ordinal_suffixes")
        if not suffixes:
            return text
        pattern = _ordinal_suffix_pattern(lang)

        def replace(match: re.Match[str]) -> str:
            return self._ordinal_word(int(match.group(1)), lang, suffixes.get(match.group(2).lower(), suffixes.get(match.group(2), "m")))

        return pattern.sub(replace, text)

    def _expand_german_ordinal(self, match: re.Match[str]) -> str:
        data = self._data("de")
        before = match.string[:match.start()].rstrip()
        if not before:
            return match.group(0)   # a numbered list item ("1. Einleitung")
        previous = before.split()[-1].strip("\"'„“(").lower()
        following = match.group(2)
        leaders = data.get("ordinal_leaders", {})
        months = {month.lower() for month in data["months"]}
        if previous not in leaders and following.lower() not in months and not following[0].islower():
            return match.group(0)
        return self._to_words(int(match.group(1)), "de", ordinal=True) + leaders.get(previous, "n")

    def _german_year(self, year: int) -> str:
        """German years 1100-1999 are read in hundreds: 1990 "neunzehnhundertneunzig"."""
        if 1100 <= year <= 1999:
            high, low = divmod(year, 100)
            return self._to_words(high, "de") + "hundert" + (self._to_words(low, "de") if low else "")
        return self._to_words(year, "de")

    def _expand_german_date(self, match: re.Match[str]) -> str:
        day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if not (1 <= day <= 31 and 1 <= month <= 12):
            return match.group(0)
        months = self._data("de")["months"]
        return f"{self._to_words(day, 'de', ordinal=True)}n {months[month - 1]} {self._german_year(year)}"

    def _expand_clock(self, match: re.Match[str], lang: str) -> str:
        hour, minutes = int(match.group(1)), match.group(2)
        if hour > 24 or (minutes and int(minutes) > 59):
            return match.group(0)
        words = self._to_words(hour, lang)
        if lang == "fr" and words.endswith("un"):
            words += "e"   # "une heure", "vingt et une heures"
        noun = _word(lang, "hour" if hour == 1 else "hours")
        return f"{words} {noun}" + (f" {int(minutes)}" if minutes else "")

    _SENTENCE_FOLLOWS_RE = re.compile(r"\s*$|\s+[\"'“‘(\[]?[A-ZÀ-ÖØ-ÞÑ]")

    def _expand_dotted_initialisms(self, text: str, lang: str) -> str:
        def replace(match: re.Match[str]) -> str:
            letters = re.findall(r"[A-Za-zÑñ]", match.group(0))
            spoken = " ".join(self._letter_name(letter, lang) for letter in letters)
            # The last dot also ends the sentence at the end of the text ("... at 5 p.m.") and after a lower-case
            # initialism followed by a capital ("at 9 a.m. Then ..."). An upper-case initialism before a capital is part
            # of a name ("D.J. Stumpy", "J.K. Rowling") and loses its dot, as in the training normalizer.
            follows = self._SENTENCE_FOLLOWS_RE.match(text, match.end())
            at_end = not text[match.end():].strip()
            ends_sentence = at_end or (follows is not None and all(letter.islower() for letter in letters))
            return spoken + "." if ends_sentence else spoken

        return self._DOTTED_INITIALISM_RE.sub(replace, text)

    @staticmethod
    def _is_dash_word_char(char: str) -> bool:
        return bool(char) and (
            char.isalnum() or unicodedata.category(char).startswith("M")
        )

    def _expand_dash_letter_names(self, text: str, lang: str) -> str:
        def left(match: re.Match[str]) -> str:
            before = match.string[match.start() - 1] if match.start() else ""
            after = match.string[match.end()] if match.end() < len(match.string) else ""
            if before == "_" or self._is_dash_word_char(before) or not self._is_dash_word_char(after):
                return match.group(0)
            return f"{self._letter_name(match.group(1), lang)} "

        def right(match: re.Match[str]) -> str:
            before = match.string[match.start() - 1] if match.start() else ""
            after = match.string[match.end()] if match.end() < len(match.string) else ""
            if not self._is_dash_word_char(before) or after == "_" or self._is_dash_word_char(after):
                return match.group(0)
            return f" {self._letter_name(match.group(1), lang)}"

        value = self._DASH_LEFT_SINGLE_LETTER_RE.sub(left, text)
        return self._DASH_RIGHT_SINGLE_LETTER_RE.sub(right, value)

    def _letter_name(self, letter: str, lang: str) -> str:
        key = str(letter or "").upper()
        return self._data(lang)["letters"].get(key) or TEXT_DATA["en"]["letters"].get(key, letter)

    def _split_alphanumeric(self, match: re.Match[str], lang: str) -> str:
        runs = self._ALPHANUMERIC_RUN_RE.findall(match.group(0))
        # English "MP3s": the final "s" makes the number plural ("em pee threes").
        plural = lang == "en" and len(runs) > 1 and runs[-1] == "s" and runs[-2][0].isdigit()
        if plural:
            runs = runs[:-1]
        parts: list[str] = []
        for run in runs:
            if run[0].isdigit():
                parts.append(run)
            elif len(run) == 1 or (len(run) <= 3 and (run.isupper() or not self._VOWEL_RE.search(run))):
                parts.extend(self._letter_name(letter, lang) for letter in run)
            else:
                parts.append(run)
        if plural:
            parts[-1] = self._plural_number_words(int(parts[-1]))
        return " ".join(parts)

    @staticmethod
    def _plural_words(words: str) -> str:
        words = words.replace("-", " ")
        if words.endswith("y"):
            return words[:-1] + "ies"
        return words + ("es" if words.endswith("x") else "s")

    def _plural_number_words(self, value: int) -> str:
        return self._EN_ROUND_PLURALS.get(value) or self._plural_words(self._to_words(value, "en"))

    def _expand_plural_number(self, match: re.Match[str]) -> str:
        lead, digits, apostrophe = match.group(1), match.group(2), match.group(3)
        if "." in digits:
            return match.group(0) if lead or apostrophe else f"{digits} seconds"
        value = int(digits.replace(",", ""))
        decade = len(digits) in (2, 4) and value % 10 == 0
        if ((lead or apostrophe) and not decade) or (lead and len(digits) != 2):
            return match.group(0)
        if len(digits) == 4 and 1100 <= value <= 2099:
            return self._plural_words(self._year_words(value))
        return self._plural_number_words(value)

    def _year_words(self, year: int) -> str:
        """An English year as spoken: 1990 "nineteen ninety", 1905 "nineteen oh five", 1900 "nineteen hundred", 2005 "two
        thousand five", 2024 "twenty twenty four". Years outside 1100-2099 are read as numbers."""
        if 2000 <= year <= 2009:
            return "two thousand" + (f" {self._to_words(year - 2000, 'en')}" if year > 2000 else "")
        if 1100 <= year <= 2099:
            high, low = divmod(year, 100)
            tail = "hundred" if low == 0 else f"oh {self._to_words(low, 'en')}" if low < 10 else self._to_words(low, "en")
            return f"{self._to_words(high, 'en')} {tail}"
        return self._to_words(year, "en")

    def _ampm_words(self, letter: str, lang: str) -> str:
        return f"{self._letter_name(letter, lang)} {self._letter_name('M', lang)}"

    def _expand_bare_ampm(self, match: re.Match[str], lang: str) -> str:
        hour = int(match.group(1))
        if not 1 <= hour <= 12:
            return match.group(0)
        return f"{self._to_words(hour, lang)} {self._ampm_words(match.group(2), lang)}"

    @classmethod
    def _normalizer_language(cls, language: str) -> str:
        key = str(language or "en").replace("-", "_").lower()
        return cls._LANG_BY_MODEL_LANGUAGE.get(key, key if key in {"en", "es", "it", "fr", "de", "vi"} else "en")

    @classmethod
    def _normalize_punctuation(cls, text: str, lang: str) -> str:
        value = cls._ZERO_WIDTH_RE.sub("", str(text or ""))
        value = value.replace("*", " ")   # scene breaks, *emphasis* and footnote stars are not spoken
        value = cls._normalize_dash_contract(value)
        value = value.translate(cls._UNICODE_TRANSLATION)
        value = cls._DOT_RUN_RE.sub("...", value)
        value = cls._MIXED_TERMINAL_PUNCTUATION_RE.sub(lambda match: match.group(1), value)
        degrees_word = _word(lang, "degrees")
        value = value.replace("=", f" {_word(lang, 'equals')} ")
        # "30 °C" is read "thirty degrees Celsius"; a bare letter after "degrees" is heard as a stray word.
        value = cls._DEGREE_SCALE_RE.sub(lambda m: f" {degrees_word} {_word(lang, 'celsius' if m.group(1) == 'C' else 'fahrenheit')}", value)
        value = cls._DEGREE_RE.sub(f" {degrees_word} ", value)
        return cls._WHITESPACE_RE.sub(" ", value).strip()

    @classmethod
    def _normalize_dash_contract(cls, text: str) -> str:
        value = mark_tight_dashes(text)
        for hyphen in cls._LEXICAL_HYPHENS:
            value = value.replace(hyphen, "-")
        # An en dash between numbers is a range ("1939–1945"), read by the range rule rather than as a pause.
        value = re.sub(r"(?<=\d)\s?[\u2012\u2013]\s?(?=\d)", "-", value)
        for dash in cls._SYNTACTIC_DASHES:
            value = value.replace(dash, " — ")
        value = re.sub(r"-{2,}", " — ", value)
        value = re.sub(r"(?:(?<=\s)-+|-+(?=\s))", " — ", value)
        return cls._WHITESPACE_RE.sub(" ", value).strip()

    @classmethod
    def _normalize_identifier_fragment(cls, fragment: str, lang: str) -> str:
        out = str(fragment or "")
        plus_word = _word(lang, "plus")
        dot_word = _word(lang, "dot")
        out = out.replace("+", f" {plus_word} ")
        out = out.replace("_", " ")
        out = out.replace("-", " ")
        out = out.replace(".", f" {dot_word} ")
        return out

    def _normalize_at_symbols(self, text: str, lang: str) -> str:
        word = _word(lang, "at")

        def replace_email(match: re.Match[str]) -> str:
            local = self._normalize_identifier_fragment(match.group("local"), lang)
            domain = self._normalize_identifier_fragment(match.group("domain"), lang)
            return f"{local} {word} {domain}"

        return self._EMAIL_RE.sub(replace_email, text).replace("@", f" {word} ")

    def _to_words(self, value: int, lang: str, *, ordinal: bool = False) -> str:
        if lang == "en" and not ordinal and self._inflect is not None:
            try:
                return self._inflect.number_to_words(str(value)).replace(",", "")
            except Exception:
                pass
        if _num2words_raw is not None:
            kwargs = {"lang": lang}
            if ordinal:
                kwargs["to"] = "ordinal"
            try:
                return _num2words_raw(value, **kwargs).replace(",", "")
            except (NotImplementedError, TypeError, ValueError):
                pass
        return _fallback_number_to_words(int(value), lang, ordinal=ordinal)

    def _decimal_to_words(self, raw: str, lang: str) -> str:
        whole, frac = self._split_decimal(raw, lang)
        if frac is None:
            return self._to_words(int(self._whole_number_digits(whole) or "0"), lang)
        point = _word(lang, "point")
        words = [self._to_words(int(self._whole_number_digits(whole) or "0"), lang), point]
        words.extend(self._to_words(int(ch), lang) for ch in frac if ch.isdigit())
        return " ".join(words)

    def _expand_currency(self, text: str, lang: str) -> str:
        """"$3.50", "£12", "40€", "50.000 ₫"; with a scale word "$5bn" and "$3.5 million" ("three point five million
        dollars")."""
        currency = self._data(lang).get("currency", {})
        if not currency:
            return text
        prefix, suffix = _currency_patterns(lang)
        scales = self._data(lang).get("currency_scales", {})

        def replace(match: re.Match[str]) -> str:
            symbol, raw, scale = match.group(1), match.group(2), match.group(3)
            if scale:
                return f"{self._decimal_to_words(raw, lang)} {scales[scale]} {currency[symbol][1]}"
            return self._currency_to_words(raw, symbol, lang)

        text = prefix.sub(replace, text)
        return suffix.sub(lambda m: self._currency_to_words(m.group(1), "€" if m.group(2).upper() == "EUR" else m.group(2), lang), text)

    def _currency_to_words(self, raw: str, symbol: str, lang: str) -> str:
        units = self._data(lang).get("currency", {}).get(symbol) or TEXT_DATA["en"]["currency"]["$"]
        major_singular, major_plural, minor_singular, minor_plural = units
        whole, cents = self._split_decimal(raw, lang, currency=True)
        major = int(self._whole_number_digits(whole) or "0")
        major_words = f"{self._to_words(major, lang)} {major_singular if major == 1 else major_plural}"
        if cents is None:
            return major_words
        minor = int((cents + "00")[:2])
        if minor == 0:
            return major_words
        minor_words = f"{self._to_words(minor, lang)} {minor_singular if minor == 1 else minor_plural}"
        return f"{major_words} {_word(lang, 'and')} {minor_words}"

    @classmethod
    def _split_decimal(cls, raw: str, lang: str, *, currency: bool = False) -> tuple[str, str | None]:
        value = str(raw or "").strip().replace(" ", "")
        if not value:
            return "0", None
        comma = value.rfind(",")
        dot = value.rfind(".")
        decimal_sep: str | None = None
        if comma >= 0 and dot >= 0:
            decimal_sep = "," if comma > dot else "."
        elif comma >= 0:
            if not cls._is_grouped_integer(value, ",") and (lang in {"es", "it", "fr", "de", "vi"} or currency):
                decimal_sep = ","
        elif dot >= 0:
            if not cls._is_grouped_integer(value, ".") and (lang == "en" or currency or len(value.rsplit(".", maxsplit=1)[-1]) != 3):
                decimal_sep = "."
        if decimal_sep is None:
            return value, None
        whole, frac = value.rsplit(decimal_sep, maxsplit=1)
        return whole, "".join(ch for ch in frac if ch.isdigit())

    @staticmethod
    def _whole_number_digits(raw: str) -> str:
        return "".join(ch for ch in str(raw or "") if ch.isdigit())

    @staticmethod
    def _is_grouped_integer(raw: str, separator: str) -> bool:
        escaped = re.escape(separator)
        return bool(re.fullmatch(rf"\d{{1,3}}(?:{escaped}\d{{3}})+", str(raw or "")))

    def _expand_time(self, match: re.Match[str], lang: str) -> str:
        hour = int(match.group(1))
        minute = int(match.group(2))
        ampm = (match.group(3) or "").upper()
        if minute == 0 and lang == "en":
            minute_words = "o'clock"
        elif minute < 10 and lang == "en":
            minute_words = f"oh {self._to_words(minute, lang)}"
        else:
            minute_words = self._to_words(minute, lang)
        if ampm and lang == "en":
            ampm = self._ampm_words(ampm[0], lang)   # the G2P reads a bare "AM" as the word "am"
        return f"{self._to_words(hour, lang)} {minute_words} {ampm}".strip()

    def _expand_slash_date(self, match: re.Match[str], lang: str) -> str:
        first, second, year = match.group(1), match.group(2), match.group(3)
        if lang == "en":
            return self._expand_date(first, second, year, lang)
        return self._expand_date(second, first, year, lang)


    def _expand_named_month_date(self, match: re.Match[str], lang: str) -> str:
        if lang != "en":
            return match.group(0)
        raw_month = match.group(1).rstrip(".").lower()
        month_aliases = {
            "jan": 1, "january": 1,
            "feb": 2, "february": 2,
            "mar": 3, "march": 3,
            "apr": 4, "april": 4,
            "may": 5,
            "jun": 6, "june": 6,
            "jul": 7, "july": 7,
            "aug": 8, "august": 8,
            "sep": 9, "sept": 9, "september": 9,
            "oct": 10, "october": 10,
            "nov": 11, "november": 11,
            "dec": 12, "december": 12,
        }
        month = month_aliases.get(raw_month)
        if month is None:
            return match.group(0)
        day = int(match.group(2))
        if day < 1 or day > 31:
            return match.group(0)
        month_name = self._data("en")["months"][month - 1]
        day_words = self._to_words(day, lang, ordinal=True)
        year_s = match.group(3)
        if not year_s:
            return f"{month_name} {day_words}"
        return f"{month_name} {day_words}, {self._year_words(int(year_s))}"

    def _expand_date(self, month_s: str, day_s: str, year_s: str, lang: str) -> str:
        try:
            month = int(month_s)
            day = int(day_s)
            year = int(year_s)
        except ValueError:
            return f"{month_s}/{day_s}/{year_s}"
        if month < 1 or month > 12 or day < 1 or day > 31:
            return f"{month_s}/{day_s}/{year_s}"
        month_name = self._data(lang)["months"][month - 1]
        if lang == "en":
            return f"{month_name} {self._to_words(day, lang, ordinal=True)}, {self._year_words(year)}"
        if lang == "es":
            return f"{day} de {month_name} de {year}"
        if lang == "de":
            return f"{self._to_words(day, lang, ordinal=True)}n {month_name} {self._german_year(year)}"
        if day == 1 and lang in ("fr", "it"):
            return f"{self._ordinal_word(1, lang)} {month_name} {year}"   # "premier mai", "primo maggio"
        return f"{day} {month_name} {year}"

    def _expand_fraction(self, match: re.Match[str], lang: str) -> str:
        numerator = int(match.group(1))
        denominator = int(match.group(2))
        if lang == "en":
            names = {
                2: ("half", "halves"),
                3: ("third", "thirds"),
                4: ("fourth", "fourths"),
                5: ("fifth", "fifths"),
                6: ("sixth", "sixths"),
                7: ("seventh", "sevenths"),
                8: ("eighth", "eighths"),
                9: ("ninth", "ninths"),
                10: ("tenth", "tenths"),
            }
            if denominator in names:
                singular, plural = names[denominator]
                return f"{self._to_words(numerator, lang)} {singular if numerator == 1 else plural}"
            return f"{self._to_words(numerator, lang)} over {self._to_words(denominator, lang)}"
        return f"{self._to_words(numerator, lang)} {_word(lang, 'over')} {self._to_words(denominator, lang)}"

    @staticmethod
    def _and_word(lang: str) -> str:
        return _word(lang, "and")


_DEFAULT_NORMALIZER: SpokenTextNormalizer | None = None


def normalize_spoken_text(text: str, *, language: str, enabled: bool = True) -> str:
    """Normalize text to spoken form, reusing one normalizer per process."""

    if not enabled:
        return str(text or "")
    global _DEFAULT_NORMALIZER
    if _DEFAULT_NORMALIZER is None:
        _DEFAULT_NORMALIZER = SpokenTextNormalizer()
    return _DEFAULT_NORMALIZER.normalize(text, language=language)
