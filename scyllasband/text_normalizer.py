"""Spoken-form text normalization for Scylla's Band.

The Scylla's Band G2P model is trained on spoken-form text. Runtime inference must apply
the same normalization before G2P so numbers, dates, currencies, times, and
common symbols arrive in the form the model is most likely to handle.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

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


def _fallback_number_to_words(value: int, lang: str, *, ordinal: bool = False) -> str:
    n = int(value)
    if ordinal and lang == "en" and n in _EN_ORDINALS:
        return _EN_ORDINALS[n]
    if lang == "es":
        return _fallback_spanish_number(n)
    if lang == "it":
        return _fallback_italian_number(n)
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

    _LANG_BY_MODEL_LANGUAGE = {"en": "en", "en_us": "en", "en_gb": "en", "es": "es", "it": "it"}
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
            "\u2010": "-",
            "\u2011": "-",
            "\u2012": "-",
            "\u2013": "-",
            "\u2014": "-",
            "\u2015": "-",
            "\u2212": "-",
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
    _EN_MONTH_DATE_RE = re.compile(
        r"\b("
        r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
        r")\.?\s+(\d{1,2})(?:,\s*(\d{2,4}))?\b",
        re.IGNORECASE,
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
    }

    def __init__(self, config: SpokenTextNormalizerConfig | None = None) -> None:
        self.config = config or SpokenTextNormalizerConfig()
        self._inflect = inflect.engine() if inflect is not None else None

    def normalize(self, text: str, *, language: str) -> str:
        lang = self._normalizer_language(language)
        value = unicodedata.normalize("NFC", str(text or ""))
        if self.config.normalize_punctuation:
            value = self._normalize_punctuation(value)
        if self.config.normalize_at_sign:
            value = self._normalize_at_symbols(value, lang)
        value = self._expand_dotted_initialisms(value, lang)
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
            if lang in {"es", "it"}:
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
        return cls._LANG_BY_MODEL_LANGUAGE.get(key, key if key in {"en", "es", "it"} else "en")

    @classmethod
    def _normalize_punctuation(cls, text: str) -> str:
        value = cls._ZERO_WIDTH_RE.sub("", str(text or ""))
        value = value.translate(cls._UNICODE_TRANSLATION)
        value = value.replace("=", " equals ")
        value = value.replace("°", " degrees ")
        value = re.sub(r"(?<!\d):|:(?!\d)", ",", value)
        return cls._WHITESPACE_RE.sub(" ", value).strip()

    @classmethod
    def _normalize_identifier_fragment(cls, fragment: str) -> str:
        out = str(fragment or "")
        out = out.replace("+", " plus ")
        out = out.replace("_", " ")
        out = out.replace("-", " ")
        out = out.replace(".", " dot ")
        return out

    def _normalize_at_symbols(self, text: str, lang: str) -> str:
        word = {"en": "at", "es": "arroba", "it": "chiocciola"}.get(lang, "at")

        def replace_email(match: re.Match[str]) -> str:
            local = self._normalize_identifier_fragment(match.group("local"))
            domain = self._normalize_identifier_fragment(match.group("domain"))
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
                try:
                    return _num2words_raw(value, lang="en", **({"to": "ordinal"} if ordinal else {})).replace(",", "")
                except Exception:
                    pass
        return _fallback_number_to_words(int(value), lang, ordinal=ordinal)

    def _decimal_to_words(self, raw: str, lang: str) -> str:
        whole, frac = self._split_decimal(raw, lang)
        if frac is None:
            return self._to_words(int(self._whole_number_digits(whole) or "0"), lang)
        point = {"en": "point", "es": "coma", "it": "virgola"}.get(lang, "point")
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
            if not cls._is_grouped_integer(value, ",") and (lang in {"es", "it"} or currency):
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
        over = "sobre" if lang == "es" else "su"
        return f"{self._to_words(numerator, lang)} {over} {self._to_words(denominator, lang)}"

    @staticmethod
    def _and_word(lang: str) -> str:
        return {"en": "and", "es": "y", "it": "e"}.get(lang, "and")

    @staticmethod
    def _percent_word(lang: str) -> str:
        return {"en": "percent", "es": "por ciento", "it": "per cento"}.get(lang, "percent")


_DEFAULT_NORMALIZER: SpokenTextNormalizer | None = None


def normalize_spoken_text(text: str, *, language: str, enabled: bool = True) -> str:
    """Normalize text to spoken form, reusing one normalizer per process."""

    if not enabled:
        return str(text or "")
    global _DEFAULT_NORMALIZER
    if _DEFAULT_NORMALIZER is None:
        _DEFAULT_NORMALIZER = SpokenTextNormalizer()
    return _DEFAULT_NORMALIZER.normalize(text, language=language)
