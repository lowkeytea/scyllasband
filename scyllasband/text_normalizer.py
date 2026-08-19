"""Spoken-form text normalization for Scylla's Band.

The G2P model is trained on spoken-form text. Runtime inference applies the same
normalization before G2P for every language declared by the selected bundle.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import unicodedata


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
    if n < 0:
        return "meno " + _fallback_italian_number(-n)
    if n < 20:
        return _IT_UNDER_20[n]
    if n < 100:
        tens = (n // 10) * 10
        rest = n % 10
        return _IT_TENS[tens] if rest == 0 else f"{_IT_TENS[tens]} {_fallback_italian_number(rest)}"
    if n < 1000:
        rest = n % 100
        head = "cento" if n // 100 == 1 else f"{_fallback_italian_number(n // 100)} cento"
        return head if rest == 0 else f"{head} {_fallback_italian_number(rest)}"
    if n < 1_000_000:
        rest = n % 1000
        head = "mille" if n // 1000 == 1 else f"{_fallback_italian_number(n // 1000)} mila"
        return head if rest == 0 else f"{head} {_fallback_italian_number(rest)}"
    rest = n % 1_000_000
    head = "un milione" if n // 1_000_000 == 1 else f"{_fallback_italian_number(n // 1_000_000)} milioni"
    return head if rest == 0 else f"{head} {_fallback_italian_number(rest)}"


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
    _MONTHS = {
        "en": (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ),
        "es": (
            "enero",
            "febrero",
            "marzo",
            "abril",
            "mayo",
            "junio",
            "julio",
            "agosto",
            "septiembre",
            "octubre",
            "noviembre",
            "diciembre",
        ),
        "it": (
            "gennaio",
            "febbraio",
            "marzo",
            "aprile",
            "maggio",
            "giugno",
            "luglio",
            "agosto",
            "settembre",
            "ottobre",
            "novembre",
            "dicembre",
        ),
        "fr": (
            "janvier", "février", "mars", "avril", "mai", "juin",
            "juillet", "août", "septembre", "octobre", "novembre", "décembre",
        ),
        "de": (
            "Januar", "Februar", "März", "April", "Mai", "Juni",
            "Juli", "August", "September", "Oktober", "November", "Dezember",
        ),
        "vi": (
            "tháng một", "tháng hai", "tháng ba", "tháng tư", "tháng năm", "tháng sáu",
            "tháng bảy", "tháng tám", "tháng chín", "tháng mười", "tháng mười một", "tháng mười hai",
        ),
    }
    _CURRENCY_UNITS = {
        "en": {
            "$": ("dollar", "dollars", "cent", "cents"),
            "£": ("pound", "pounds", "penny", "pence"),
            "€": ("euro", "euros", "cent", "cents"),
        },
        "es": {
            "$": ("dólar", "dólares", "centavo", "centavos"),
            "£": ("libra", "libras", "penique", "peniques"),
            "€": ("euro", "euros", "céntimo", "céntimos"),
        },
        "it": {
            "$": ("dollaro", "dollari", "cent", "cent"),
            "£": ("sterlina", "sterline", "penny", "pence"),
            "€": ("euro", "euro", "centesimo", "centesimi"),
        },
        "fr": {
            "$": ("dollar", "dollars", "centime", "centimes"),
            "£": ("livre", "livres", "penny", "pence"),
            "€": ("euro", "euros", "centime", "centimes"),
        },
        "de": {
            "$": ("Dollar", "Dollar", "Cent", "Cent"),
            "£": ("Pfund", "Pfund", "Penny", "Pence"),
            "€": ("Euro", "Euro", "Cent", "Cent"),
        },
        "vi": {
            "$": ("đô la", "đô la", "xu", "xu"),
            "£": ("bảng Anh", "bảng Anh", "xu", "xu"),
            "€": ("euro", "euro", "xu", "xu"),
        },
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
    # Honorific abbreviations the G2P mispronounces. Each spelling here was
    # chosen by matching the model's output for the expansion against espeak's
    # output for the *abbreviation*, which is the target the model was distilled
    # toward -- so "Mrs." expands to "misses" (espeak: mˈɪsɪz) rather than the
    # etymological "missus", which the model renders mˈɪʃəs.
    #
    # Deliberately minimal. "Mr." is NOT listed: the model already handles it
    # correctly before a name (7/8 across a name sample, matching the ceiling set
    # by the names themselves), so expanding it would add risk for no gain.
    # "Prof." and "vs" are likewise already correct. Titles whose expansion does
    # not reproduce espeak's reading of the abbreviation (Ms., Jr., Sr., etc.,
    # Ave., Rd.) are omitted rather than guessed at.
    #
    # This cannot help bare initialisms (FBI, USA, BBC). Spelling them as
    # separate words gives each its own primary stress -- "bee bee see" becomes
    # bˈiːbˈiːsˈiː where espeak has bˌiːbˌiːsˈiː -- so that class needs a G2P-side
    # fix, not a normalizer one.
    _TITLE_ABBREVIATIONS = {
        "en": {"mrs": "misses", "dr": "doctor"},
    }
    # Only expand when a capitalised word follows, i.e. the honorific use
    # ("Dr. Smith"). This leaves the street sense alone, where the abbreviation
    # trails the name ("Main Dr.") and is not followed by a capitalised token.
    _TITLE_ABBREVIATION_RE = re.compile(
        r"(?<![A-Za-zÀ-ÖØ-öø-ÿÑñ])([A-Za-z]{2,4})\.?\s+(?=[A-ZÀ-ÖØ-Þ])"
    )
    _DASH_LEFT_SINGLE_LETTER_RE = re.compile(
        r"(?<![A-Za-zÀ-ÖØ-öø-ÿÑñ])([A-Za-zÑñ])\s*-\s*(?=[A-Za-zÀ-ÖØ-öø-ÿÑñ0-9])"
    )
    _DASH_RIGHT_SINGLE_LETTER_RE = re.compile(
        r"(?<=[A-Za-zÀ-ÖØ-öø-ÿÑñ0-9])\s*-\s*([A-Za-zÑñ])(?![A-Za-zÀ-ÖØ-öø-ÿÑñ])"
    )
    _LETTER_NAMES = {
        "en": {
            "A": "ay", "B": "bee", "C": "see", "D": "dee", "E": "ee", "F": "eff",
            "G": "gee", "H": "aitch", "I": "eye", "J": "jay", "K": "kay", "L": "ell",
            "M": "em", "N": "en", "O": "oh", "P": "pee", "Q": "cue", "R": "are",
            "S": "ess", "T": "tee", "U": "you", "V": "vee", "W": "double you",
            "X": "ex", "Y": "why", "Z": "zee",
        },
        "es": {
            "A": "a", "B": "be", "C": "ce", "D": "de", "E": "e", "F": "efe",
            "G": "ge", "H": "hache", "I": "i", "J": "jota", "K": "ka", "L": "ele",
            "M": "eme", "N": "ene", "Ñ": "eñe", "O": "o", "P": "pe", "Q": "cu",
            "R": "erre", "S": "ese", "T": "te", "U": "u", "V": "uve", "W": "doble uve",
            "X": "equis", "Y": "ye", "Z": "zeta",
        },
        "it": {
            "A": "a", "B": "bi", "C": "ci", "D": "di", "E": "e", "F": "effe",
            "G": "gi", "H": "acca", "I": "i", "J": "i lunga", "K": "cappa", "L": "elle",
            "M": "emme", "N": "enne", "O": "o", "P": "pi", "Q": "cu", "R": "erre",
            "S": "esse", "T": "ti", "U": "u", "V": "vu", "W": "doppia vu",
            "X": "ics", "Y": "ipsilon", "Z": "zeta",
        },
        "fr": {
            "A": "a", "B": "bé", "C": "cé", "D": "dé", "E": "e", "F": "effe",
            "G": "gé", "H": "ache", "I": "i", "J": "ji", "K": "ka", "L": "elle",
            "M": "emme", "N": "enne", "O": "o", "P": "pé", "Q": "ku", "R": "erre",
            "S": "esse", "T": "té", "U": "u", "V": "vé", "W": "double vé",
            "X": "iks", "Y": "i grec", "Z": "zède",
        },
        "de": {
            "A": "a", "B": "be", "C": "tse", "D": "de", "E": "e", "F": "eff",
            "G": "ge", "H": "ha", "I": "i", "J": "jot", "K": "ka", "L": "ell",
            "M": "emm", "N": "enn", "O": "o", "P": "pe", "Q": "ku", "R": "err",
            "S": "ess", "T": "te", "U": "u", "V": "fau", "W": "we",
            "X": "iks", "Y": "ypsilon", "Z": "tset",
        },
        "vi": {
            "A": "a", "B": "bê", "C": "xê", "D": "dê", "E": "e", "F": "ép",
            "G": "giê", "H": "hát", "I": "i", "J": "giây", "K": "ca", "L": "e lờ",
            "M": "e mờ", "N": "e nờ", "O": "o", "P": "pê", "Q": "quy", "R": "e rờ",
            "S": "ét", "T": "tê", "U": "u", "V": "vê", "W": "vê kép",
            "X": "ích", "Y": "i dài", "Z": "dét",
        },
    }

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
        value = self._expand_dotted_initialisms(value, lang)
        value = self._expand_title_abbreviations(value, lang)
        if self.config.expand_currency:
            value = re.sub(
                r"([$£€])\s*([0-9](?:[0-9.,]*[0-9])?)",
                lambda m: self._expand_currency_prefix(m, lang),
                value,
            )
            value = re.sub(
                r"\b([0-9](?:[0-9.,]*[0-9])?)\s*(€|EUR)\b",
                lambda m: self._expand_currency_suffix(m, lang),
                value,
                flags=re.IGNORECASE,
            )
        if self.config.expand_times:
            value = re.sub(r"\b(\d{1,2}):(\d{2})(?:\s*([AaPp][Mm]))?\b", lambda m: self._expand_time(m, lang), value)
        if self.config.expand_percentages:
            value = re.sub(r"(?<![\d.,])(\d[\d.,]*)%", lambda m: f"{self._decimal_to_words(m.group(1), lang)} {self._percent_word(lang)}", value)
        if self.config.expand_ordinals:
            value = re.sub(r"\b(\d+)(st|nd|rd|th|o|a)\b", lambda m: self._to_words(int(m.group(1)), lang, ordinal=True), value, flags=re.IGNORECASE)
            value = re.sub(r"\b(\d+)(?:º|ª)\b", lambda m: self._to_words(int(m.group(1)), lang, ordinal=True), value)
        if self.config.expand_numbers:
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
        value = self._expand_dash_letter_names(value, lang)
        # The target backend is dictionary-based, so hyphenated compounds and
        # number words are safer as separate lookup tokens. Inference uses the
        # same normalizer, preserving the train/runtime contract.
        value = value.replace("-", " ")
        return self._WHITESPACE_RE.sub(" ", value).strip()

    def _expand_dotted_initialisms(self, text: str, lang: str) -> str:
        def replace(match: re.Match[str]) -> str:
            letters = re.findall(r"[A-Za-zÑñ]", match.group(0))
            return " ".join(self._letter_name(letter, lang) for letter in letters)

        return self._DOTTED_INITIALISM_RE.sub(replace, text)

    def _expand_title_abbreviations(self, text: str, lang: str) -> str:
        table = self._TITLE_ABBREVIATIONS.get(lang)
        if not table:
            return text

        def replace(match: re.Match[str]) -> str:
            expansion = table.get(match.group(1).lower())
            return f"{expansion} " if expansion else match.group(0)

        return self._TITLE_ABBREVIATION_RE.sub(replace, text)

    def _expand_dash_letter_names(self, text: str, lang: str) -> str:
        def left(match: re.Match[str]) -> str:
            return f"{self._letter_name(match.group(1), lang)} "

        def right(match: re.Match[str]) -> str:
            return f" {self._letter_name(match.group(1), lang)}"

        value = self._DASH_LEFT_SINGLE_LETTER_RE.sub(left, text)
        return self._DASH_RIGHT_SINGLE_LETTER_RE.sub(right, value)

    def _letter_name(self, letter: str, lang: str) -> str:
        key = str(letter or "").upper()
        names = self._LETTER_NAMES.get(lang, self._LETTER_NAMES["en"])
        return names.get(key, letter)

    @classmethod
    def _normalizer_language(cls, language: str) -> str:
        key = str(language or "en").replace("-", "_").lower()
        return cls._LANG_BY_MODEL_LANGUAGE.get(key, key if key in {"en", "es", "it", "fr", "de", "vi"} else "en")

    @classmethod
    def _normalize_punctuation(cls, text: str, lang: str) -> str:
        value = cls._ZERO_WIDTH_RE.sub("", str(text or ""))
        value = cls._normalize_dash_contract(value)
        value = value.translate(cls._UNICODE_TRANSLATION)
        value = cls._DOT_RUN_RE.sub("...", value)
        value = cls._MIXED_TERMINAL_PUNCTUATION_RE.sub(lambda match: match.group(1), value)
        equals_word = {"en": "equals", "es": "igual", "it": "uguale", "fr": "égal", "de": "gleich", "vi": "bằng"}.get(lang, "equals")
        degrees_word = {"en": "degrees", "es": "grados", "it": "gradi", "fr": "degrés", "de": "Grad", "vi": "độ"}.get(lang, "degrees")
        value = value.replace("=", f" {equals_word} ")
        value = value.replace("°", f" {degrees_word} ")
        return cls._WHITESPACE_RE.sub(" ", value).strip()

    @classmethod
    def _normalize_dash_contract(cls, text: str) -> str:
        value = str(text or "")
        for hyphen in cls._LEXICAL_HYPHENS:
            value = value.replace(hyphen, "-")
        for dash in cls._SYNTACTIC_DASHES:
            value = value.replace(dash, " — ")
        value = re.sub(r"-{2,}", " — ", value)
        value = re.sub(r"(?:(?<=\s)-+|-+(?=\s))", " — ", value)
        return cls._WHITESPACE_RE.sub(" ", value).strip()

    @classmethod
    def _normalize_identifier_fragment(cls, fragment: str, lang: str) -> str:
        out = str(fragment or "")
        plus_word = {"en": "plus", "es": "más", "it": "più", "fr": "plus", "de": "plus", "vi": "cộng"}.get(lang, "plus")
        dot_word = {"en": "dot", "es": "punto", "it": "punto", "fr": "point", "de": "Punkt", "vi": "chấm"}.get(lang, "dot")
        out = out.replace("+", f" {plus_word} ")
        out = out.replace("_", " ")
        out = out.replace("-", " ")
        out = out.replace(".", f" {dot_word} ")
        return out

    def _normalize_at_symbols(self, text: str, lang: str) -> str:
        word = {
            "en": "at", "es": "arroba", "it": "chiocciola",
            "fr": "arobase", "de": "at", "vi": "a còng",
        }.get(lang, "at")

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
        point = {"en": "point", "es": "coma", "it": "virgola", "fr": "virgule", "de": "Komma", "vi": "phẩy"}.get(lang, "point")
        words = [self._to_words(int(self._whole_number_digits(whole) or "0"), lang), point]
        words.extend(self._to_words(int(ch), lang) for ch in frac if ch.isdigit())
        return " ".join(words)

    def _expand_currency_prefix(self, match: re.Match[str], lang: str) -> str:
        return self._currency_to_words(match.group(2), match.group(1), lang)

    def _expand_currency_suffix(self, match: re.Match[str], lang: str) -> str:
        return self._currency_to_words(match.group(1), "€", lang)

    def _currency_to_words(self, raw: str, symbol: str, lang: str) -> str:
        units = self._CURRENCY_UNITS.get(lang, self._CURRENCY_UNITS["en"]).get(symbol, self._CURRENCY_UNITS["en"]["$"])
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
        return f"{major_words} {self._and_word(lang)} {minor_words}"

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
        month_name = self._MONTHS["en"][month - 1]
        day_words = self._to_words(day, lang, ordinal=True)
        year_s = match.group(3)
        if not year_s:
            return f"{month_name} {day_words}"
        return f"{month_name} {day_words}, {self._to_words(int(year_s), lang)}"

    def _expand_date(self, month_s: str, day_s: str, year_s: str, lang: str) -> str:
        try:
            month = int(month_s)
            day = int(day_s)
            year = int(year_s)
        except ValueError:
            return f"{month_s}/{day_s}/{year_s}"
        if month < 1 or month > 12 or day < 1 or day > 31:
            return f"{month_s}/{day_s}/{year_s}"
        month_name = self._MONTHS.get(lang, self._MONTHS["en"])[month - 1]
        if lang == "en":
            return f"{month_name} {self._to_words(day, lang, ordinal=True)}, {self._to_words(year, lang)}"
        if lang == "es":
            return f"{day} de {month_name} de {year}"
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
        if lang == "vi":
            return f"{self._to_words(numerator, lang)} phần {self._to_words(denominator, lang)}"
        over = {"es": "sobre", "it": "su", "fr": "sur", "de": "durch"}.get(lang, "over")
        return f"{self._to_words(numerator, lang)} {over} {self._to_words(denominator, lang)}"

    @staticmethod
    def _and_word(lang: str) -> str:
        return {"en": "and", "es": "y", "it": "e", "fr": "et", "de": "und", "vi": "và"}.get(lang, "and")

    @staticmethod
    def _percent_word(lang: str) -> str:
        return {"en": "percent", "es": "por ciento", "it": "per cento", "fr": "pour cent", "de": "Prozent", "vi": "phần trăm"}.get(lang, "percent")


_DEFAULT_NORMALIZER: SpokenTextNormalizer | None = None


def normalize_spoken_text(text: str, *, language: str, enabled: bool = True) -> str:
    """Normalize text to spoken form, reusing one normalizer per process."""

    if not enabled:
        return str(text or "")
    global _DEFAULT_NORMALIZER
    if _DEFAULT_NORMALIZER is None:
        _DEFAULT_NORMALIZER = SpokenTextNormalizer()
    return _DEFAULT_NORMALIZER.normalize(text, language=language)
