"""Opt-in native pronunciation evidence, without lexical or temporal ownership.

Provider units are a provisional representation, not acoustic phones. A trusted
complete emission digest authenticates a producer's output; replaying this
parser proves only its structure and conservative surface interpretation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
from typing import Any, Mapping

SCHEMA = "scyllasband_native_pronunciation_evidence_v1"
PROVIDER_SCHEMA = "scyllasband_espeak_native_provider_v1"
PARSER_VERSION = "paired_native_slots_v1"
SEPARATOR = "\x1f"
MAX_INPUT_CODEPOINTS = 32768
MAX_OUTPUT_CODEPOINTS = 1048576
MAX_CLAUSES = 4096
_SWITCH = re.compile(r"\(([a-z][a-z0-9-]{0,31})\)\Z")


class PronunciationError(ValueError):
    """Unsupported, malformed, mismatched or unavailable provider evidence."""


def _json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _text(value: Any) -> str:
    if not isinstance(value, str):
        raise PronunciationError("Text must be a string")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise PronunciationError("Unpaired surrogate in text") from exc
    return value


def text_sha256(value: str) -> str:
    return hashlib.sha256(_text(value).encode("utf-8")).hexdigest()


def _digest(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise PronunciationError("Expected a lowercase SHA256 digest")
    return value


def _fields(value: Any, names: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(names.split()):
        raise PronunciationError("Unknown or missing contract fields")


@dataclass(frozen=True)
class TextSpan:
    start: int
    end: int

    def __post_init__(self) -> None:
        if type(self.start) is not int or type(self.end) is not int or not 0 <= self.start <= self.end:
            raise PronunciationError("Invalid codepoint span")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TextSpan:
        _fields(value, "start end")
        return cls(**value)


@dataclass(frozen=True)
class AssetBinding:
    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not Path(self.path).is_absolute():
            raise PronunciationError("Asset paths must be absolute")
        _digest(self.sha256)
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise PronunciationError("Invalid asset size")

    @classmethod
    def of(cls, path: str | Path) -> AssetBinding:
        path = Path(path).resolve(strict=True)
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
        return cls(str(path), digest.hexdigest(), size)

    def verify(self) -> None:
        if self != self.of(self.path):
            raise PronunciationError(f"Provider asset changed: {self.path}")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AssetBinding:
        _fields(value, "path sha256 size_bytes")
        return cls(**value)


@dataclass(frozen=True)
class ProviderBinding:
    requested_language: str
    provider_language: str
    voice_language: str
    voice_identifier: str
    voice_name: str
    espeak_version: str
    phonemizer_version: str
    library: AssetBinding
    code_files: tuple[AssetBinding, ...]
    data_root: str
    data_files: tuple[AssetBinding, ...]

    def __post_init__(self) -> None:
        for name in ("requested_language", "provider_language", "voice_language", "voice_identifier",
                     "voice_name", "espeak_version", "phonemizer_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or not value.isprintable():
                raise PronunciationError(f"Invalid provider identity: {name}")
        if not isinstance(self.library, AssetBinding):
            raise PronunciationError("Typed native library binding required")
        if not isinstance(self.data_root, str) or not Path(self.data_root).is_absolute():
            raise PronunciationError("Absolute provider data root required")
        for files in (self.code_files, self.data_files):
            if (type(files) is not tuple or not files or len(files) > 8192
                    or any(not isinstance(x, AssetBinding) for x in files)
                    or tuple(sorted(x.path for x in files)) != tuple(x.path for x in files)
                    or len({x.path for x in files}) != len(files)):
                raise PronunciationError("Provider files must be a nonempty ordered unique bound tuple")
        root = Path(self.data_root)
        if any(not Path(x.path).is_relative_to(root) or Path(x.path) == root for x in self.data_files):
            raise PronunciationError("Provider data file escapes its root")

    def to_dict(self) -> dict:
        return {"schema": PROVIDER_SCHEMA, **asdict(self),
            "code_files": [asdict(x) for x in self.code_files],
            "data_files": [asdict(x) for x in self.data_files], "options": {
            "python_implementation": platform.python_implementation(), "python_version": sys.version,
            "api": "espeak_TextToPhonemes", "input_mode": 1,
            "ipa_mode": (31 << 8) | 2, "mnemonic_mode": 31 << 8,
            "unit_separator": SEPARATOR, "preserve_clause_calls": True,
            "preserve_switches_and_stress": True, "parser_version": PARSER_VERSION,
            "fallback": "raise_no_fallback", "cache": "disabled",
            "provider_spacing_is_lexical_ownership": False}}

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_json(self.to_dict())).hexdigest()

    def verify_assets(self) -> None:
        for binding in (self.library, *self.code_files, *self.data_files):
            binding.verify()
        current = tuple(sorted(str(p.resolve()) for p in Path(self.data_root).rglob("*") if p.is_file()))
        if current != tuple(x.path for x in self.data_files):
            raise PronunciationError("Provider data tree membership changed")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, expected_sha256: str) -> ProviderBinding:
        _fields(value, "schema requested_language provider_language voice_language voice_identifier voice_name espeak_version phonemizer_version library code_files data_root data_files options")
        if value["schema"] != PROVIDER_SCHEMA:
            raise PronunciationError("Unsupported provider schema")
        data = {k: v for k, v in value.items() if k not in ("schema", "options")}
        data["library"] = AssetBinding.from_dict(data["library"])
        for key in ("code_files", "data_files"):
            if type(data[key]) is not list:
                raise PronunciationError("Provider file lists must be arrays")
            data[key] = tuple(AssetBinding.from_dict(x) for x in data[key])
        result = cls(**data)
        if _json(value) != _json(result.to_dict()) or result.sha256 != _digest(expected_sha256):
            raise PronunciationError("Provider options or trusted identity mismatch")
        return result


@dataclass(frozen=True)
class NativeClause:
    input_span: TextSpan
    ipa: str
    mnemonic: str

    def __post_init__(self) -> None:
        if not isinstance(self.input_span, TextSpan) or self.input_span.start == self.input_span.end:
            raise PronunciationError("Native clause requires a nonempty consumed input span")
        for value in (self.ipa, self.mnemonic):
            _text(value)
            if len(value) > MAX_OUTPUT_CODEPOINTS or any((ord(c) < 32 and c != SEPARATOR) or (c.isspace() and c not in (" ", SEPARATOR)) for c in value):
                raise PronunciationError("Unsupported native output control or length")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> NativeClause:
        _fields(value, "input_span ipa mnemonic")
        return cls(TextSpan.from_dict(value["input_span"]), value["ipa"], value["mnemonic"])


@dataclass(frozen=True)
class PronunciationUnit:
    unit_id: int
    clause_index: int
    slot_index: int
    provider_group_index: int
    ipa_span: TextSpan
    mnemonic_span: TextSpan
    ipa: str
    mnemonic: str
    kind: str
    phonetic_language: str
    language_evidence: str
    normalized_word_ids: tuple[int, ...] = field(default=(), init=False)
    lexical_ownership_available: bool = field(default=False, init=False)
    temporal_ownership_available: bool = field(default=False, init=False)
    acoustic_unit_selected: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if any(type(x) is not int or x < 0 for x in (
                self.unit_id, self.clause_index, self.slot_index, self.provider_group_index)):
            raise PronunciationError("Unit indexes must be nonnegative integers")
        if self.kind not in ("pronunciation_unit", "language_switch", "unrendered_native_unit"):
            raise PronunciationError("Unknown provisional native unit kind")
        if self.language_evidence not in ("selected_voice", "explicit_native_switch"):
            raise PronunciationError("Explicit native language provenance required")
        if not isinstance(self.phonetic_language, str) or not self.phonetic_language:
            raise PronunciationError("Native phonetic language code required")
        for value, span in ((self.ipa, self.ipa_span), (self.mnemonic, self.mnemonic_span)):
            if not isinstance(span, TextSpan) or span.end - span.start != len(_text(value)):
                raise PronunciationError("Unit surface differs from its codepoint span")


@dataclass(frozen=True)
class ModifierEvidence:
    surface_unit_id: int
    kind: str
    ipa_span: TextSpan
    mnemonic_span: TextSpan | None
    surface_value: str
    reason: str
    semantic_value: None = field(default=None, init=False)
    semantic_target_available: bool = field(default=False, init=False)
    attachment_unit_ids: tuple[int, ...] = field(default=(), init=False)
    attachment_available: bool = field(default=False, init=False)


def _slots(text: str) -> tuple[tuple[str, ...], tuple[tuple[str, TextSpan], ...]]:
    separators: list[str] = []
    slots: list[tuple[str, TextSpan]] = []
    cursor = 0
    for match in re.finditer("[ " + SEPARATOR + "]", text):
        slots.append((text[cursor:match.start()], TextSpan(cursor, match.start())))
        separators.append(match.group())
        cursor = match.end()
    slots.append((text[cursor:], TextSpan(cursor, len(text))))
    return tuple(separators), tuple(slots)


def _surface_modifiers(unit: PronunciationUnit) -> tuple[ModifierEvidence, ...]:
    if unit.kind != "pronunciation_unit":
        return ()
    result: list[ModifierEvidence] = []
    ipa, mnemonic = unit.ipa, unit.mnemonic
    for symbol, native in (("ˈ", "'"), ("ˌ", ",")):
        if ipa.startswith(symbol) and mnemonic.startswith(native):
            result.append(ModifierEvidence(unit.unit_id, "stress_surface",
                TextSpan(unit.ipa_span.start, unit.ipa_span.start + 1),
                TextSpan(unit.mnemonic_span.start, unit.mnemonic_span.start + 1), symbol,
                "paired_provider_marker_not_observed_stress_or_tone_target"))
    for index, symbol in enumerate(ipa):
        if symbol in ("ː", "ˑ"):
            result.append(ModifierEvidence(unit.unit_id, "length_surface",
                TextSpan(unit.ipa_span.start + index, unit.ipa_span.start + index + 1),
                None, symbol, "provider_surface_only_no_acoustic_duration_or_base_attachment"))
    # Evidence candidate only: no suffix becomes a tone label or duration unit.
    # In particular English native 3:/IPA ɜː under a switch does not enter here.
    tone_surfaces = {"1": "1", "2": "2", "3": "ɜ", "4": "4", "5": "5", "6": "6", "7": "7"}
    if unit.phonetic_language == "vi" and mnemonic and mnemonic[-1] in tone_surfaces:
        native = mnemonic[-1]
        surface = tone_surfaces[native]
        if ipa.endswith(surface):
            result.append(ModifierEvidence(unit.unit_id, "lexical_tone_candidate",
                TextSpan(unit.ipa_span.end - len(surface), unit.ipa_span.end),
                TextSpan(unit.mnemonic_span.end - 1, unit.mnemonic_span.end), surface,
                "unresolved_requires_reviewed_provider_bound_role_table_and_syllable_attachment"))
    return tuple(result)


def _parse(provider: ProviderBinding, clauses: tuple[NativeClause, ...]) -> tuple[tuple[PronunciationUnit, ...], tuple[ModifierEvidence, ...]]:
    units: list[PronunciationUnit] = []
    modifiers: list[ModifierEvidence] = []
    language, language_evidence = provider.voice_language, "selected_voice"
    group = 0
    for clause_index, clause in enumerate(clauses):
        ipa_separators, ipa_slots = _slots(clause.ipa)
        mnemonic_separators, mnemonic_slots = _slots(clause.mnemonic)
        if ipa_separators != mnemonic_separators:
            raise PronunciationError("Paired native lanes disagree on slot topology; no heuristic alignment")
        for slot_index, ((ipa, ipa_span), (mnemonic, mnemonic_span)) in enumerate(zip(ipa_slots, mnemonic_slots, strict=True)):
            if slot_index and ipa_separators[slot_index - 1] == " ":
                group += 1
            if not ipa and not mnemonic:
                continue
            ipa_switch, native_switch = _SWITCH.fullmatch(ipa), _SWITCH.fullmatch(mnemonic)
            if ipa_switch or native_switch or ipa.startswith("(") or mnemonic.startswith("("):
                if not ipa_switch or not native_switch or ipa != mnemonic:
                    raise PronunciationError("Malformed or unmatched native language switch")
                language, language_evidence = ipa_switch.group(1), "explicit_native_switch"
                kind = "language_switch"
            elif not mnemonic:
                raise PronunciationError("IPA slot has no paired native unit")
            elif not ipa:
                kind = "unrendered_native_unit"
            else:
                kind = "pronunciation_unit"
            unit = PronunciationUnit(len(units), clause_index, slot_index, group, ipa_span,
                                     mnemonic_span, ipa, mnemonic, kind, language, language_evidence)
            units.append(unit)
            modifiers.extend(_surface_modifiers(unit))
        if clause_index + 1 < len(clauses) and language != provider.voice_language:
            raise PronunciationError(
                "Unclosed native language switch at clause boundary; next clause language is unproved")
        group += 1  # A native API clause boundary is not an input word boundary.
    return tuple(units), tuple(modifiers)


@dataclass(frozen=True)
class ProviderEmission:
    provider: ProviderBinding
    input_text: str
    clauses: tuple[NativeClause, ...]
    units: tuple[PronunciationUnit, ...] = field(init=False)
    modifiers: tuple[ModifierEvidence, ...] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.provider, ProviderBinding):
            raise PronunciationError("Typed provider binding required")
        _text(self.input_text)
        if len(self.input_text) > MAX_INPUT_CODEPOINTS or "\x00" in self.input_text or SEPARATOR in self.input_text:
            raise PronunciationError("Unsupported input length, NUL or reserved separator")
        if type(self.clauses) is not tuple or len(self.clauses) > MAX_CLAUSES:
            raise PronunciationError("Bounded typed clause tuple required")
        cursor = output_length = 0
        for clause in self.clauses:
            if not isinstance(clause, NativeClause) or clause.input_span.start != cursor or clause.input_span.end > len(self.input_text):
                raise PronunciationError("Native clauses must partition consumed input exactly")
            cursor = clause.input_span.end
            output_length += len(clause.ipa) + len(clause.mnemonic)
        if cursor != len(self.input_text) or output_length > MAX_OUTPUT_CODEPOINTS:
            raise PronunciationError("Incomplete consumed input or excessive native output")
        units, modifiers = _parse(self.provider, self.clauses)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "modifiers", modifiers)

    def to_dict(self) -> dict:
        status = "emitted" if any(x.kind == "pronunciation_unit" for x in self.units) else "no_pronunciation_units_emitted"
        return {"schema": SCHEMA, "provider": self.provider.to_dict(), "provider_sha256": self.provider.sha256,
            "input_text": self.input_text, "input_text_sha256": text_sha256(self.input_text),
            "input_offset_unit": "unicode_codepoint", "output_offset_scope": "clause_lane_codepoint",
            "clauses": [asdict(x) for x in self.clauses], "units": [asdict(x) for x in self.units],
            "modifiers": [asdict(x) for x in self.modifiers], "status": status,
            "lexical_ownership_available": False, "temporal_ownership_available": False,
            "tone_semantics_available": False, "acoustic_representation_selected": False}

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_json(self.to_dict())).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, expected_sha256: str,
                  expected_provider_sha256: str, expected_input_sha256: str,
                  verify_assets: bool = True) -> ProviderEmission:
        _fields(value, "schema provider provider_sha256 input_text input_text_sha256 input_offset_unit output_offset_scope clauses units modifiers status lexical_ownership_available temporal_ownership_available tone_semantics_available acoustic_representation_selected")
        if value["schema"] != SCHEMA or type(value["clauses"]) is not list:
            raise PronunciationError("Unsupported emission schema or clauses")
        provider = ProviderBinding.from_dict(value["provider"], expected_sha256=expected_provider_sha256)
        result = cls(provider, value["input_text"], tuple(NativeClause.from_dict(x) for x in value["clauses"]))
        if (text_sha256(result.input_text) != _digest(expected_input_sha256)
                or result.sha256 != _digest(expected_sha256) or _json(value) != _json(result.to_dict())):
            raise PronunciationError("Emission differs from trusted binding or recomputed conservative evidence")
        if verify_assets:
            provider.verify_assets()
        return result
