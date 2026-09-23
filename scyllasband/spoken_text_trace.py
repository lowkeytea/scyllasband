"""Transform-emitted spoken-text provenance; no inferred string alignment.

The core replay validates edit/provenance structure, not linguistic correctness.
External verbalizers are opaque atomic transforms. Their installed dependency
files are snapshotted; mutable provider state and the whole interpreter are not
claimed to be reproducibly locked by those files.
"""
from __future__ import annotations

from contextvars import ContextVar
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
from typing import Any, Callable, Sequence
import unicodedata

from .normalization_trace import NormalizationTrace, Replacement, Span, TraceBuilder

SCHEMA = "scyllasband_spoken_normalization_trace_v1"
_ACTIVE: ContextVar["NormalizationOperations | None"] = ContextVar("spoken_trace", default=None)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_binding(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    payload = path.read_bytes()
    return {"path": str(path), "size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


@dataclass(frozen=True)
class MappedReplacement:
    """Explicit edits within an actual regex match; gaps are direct copies."""
    text: str
    edits: tuple[Replacement, ...]


class NormalizationOperations:
    """Shared plain/traced primitives using immediate-input codepoint offsets."""
    def __init__(self, builder: TraceBuilder | None = None) -> None:
        self.builder = builder
        self.provider_calls: list[dict[str, Any]] = []
        self._context: dict[str, Any] | None = None

    @staticmethod
    def mapped(match: re.Match[str], edits: Sequence[tuple[int, int, str]]) -> MappedReplacement:
        cursor = match.start()
        pieces: list[str] = []
        replacements: list[Replacement] = []
        for start, end, text in edits:
            if not (cursor <= start <= end <= match.end()):
                raise ValueError("mapped edit must be ordered within its actual match")
            pieces.extend((match.string[cursor:start], text))
            if match.string[start:end] != text:
                replacements.append(Replacement(Span(start, end), text))
            cursor = end
        pieces.append(match.string[cursor:match.end()])
        return MappedReplacement("".join(pieces), tuple(replacements))

    def _apply(self, value: str, edits: Sequence[Replacement], *, rule_id: str) -> str:
        if self.builder is not None:
            if value != self.builder.text:
                raise ValueError("trace input is not the current builder text")
            self.builder.apply_replacements(edits, rule_id=rule_id)
            return self.builder.text
        chunks: list[str] = []
        cursor = 0
        for edit in edits:
            chunks.extend((value[cursor:edit.span.start], edit.text))
            cursor = edit.span.end
        chunks.append(value[cursor:])
        return "".join(chunks)

    def nfc(self, value: str) -> str:
        if self.builder is None:
            return unicodedata.normalize("NFC", value)
        if value != self.builder.text:
            raise ValueError("NFC input is not the current builder text")
        self.builder.normalize_nfc()
        return self.builder.text

    def sub(self, pattern: str | re.Pattern[str], replacement: Any, value: str, *,
            rule_id: str, flags: int = 0) -> str:
        compiled = re.compile(pattern, flags)
        if self.builder is None:
            if callable(replacement):
                def plain(match: re.Match[str]) -> str:
                    result = replacement(match)
                    return result.text if isinstance(result, MappedReplacement) else result
                return compiled.sub(plain, value)
            return compiled.sub(replacement, value)
        edits: list[Replacement] = []
        for match in compiled.finditer(value):
            previous = self._context
            self._context = {"rule_id": rule_id, "input_sha256": hashlib.sha256(value.encode()).hexdigest(),
                             "input_span": [match.start(), match.end()]}
            try:
                result = replacement(match) if callable(replacement) else match.expand(replacement)
            finally:
                self._context = previous
            if isinstance(result, MappedReplacement):
                # The constructor is public: validate its declared output from edits.
                check = self.mapped(match, [(e.span.start, e.span.end, e.text) for e in result.edits])
                if check.text != result.text:
                    raise ValueError("mapped output does not replay its declared edits")
                edits.extend(result.edits)
            elif result != match.group(0):
                edits.append(Replacement(Span(match.start(), match.end()), result))
        return self._apply(value, edits, rule_id=rule_id)

    def replace(self, value: str, old: str, new: str, *, rule_id: str) -> str:
        if not old:
            raise ValueError("empty literal replacement requires an explicit insertion transform")
        return self.sub(re.escape(old), lambda _: new, value, rule_id=rule_id)

    def translate(self, value: str, table: dict[int, str | int | None], *, rule_id: str) -> str:
        if self.builder is None:
            return value.translate(table)
        edits: list[Replacement] = []
        for index, char in enumerate(value):
            translated = char.translate(table)
            if translated != char:
                edits.append(Replacement(Span(index, index + 1), translated))
        return self._apply(value, edits, rule_id=rule_id)

    def strip(self, value: str, *, rule_id: str) -> str:
        left = len(value) - len(value.lstrip())
        right = len(value.rstrip())
        edits = []
        if left:
            edits.append(Replacement(Span(0, left), ""))
        if right < len(value) and right >= left:
            edits.append(Replacement(Span(right, len(value)), ""))
        return self._apply(value, edits, rule_id=rule_id)


def note_provider(provider: str, value: int, language: str, ordinal: bool, *,
                  result: str | None = None, error: Exception | None = None) -> None:
    operations = _ACTIVE.get()
    if operations is not None:
        operations.provider_calls.append({
            "provider": provider, "value": value, "language": language, "ordinal": ordinal,
            "status": "failed_fallback_attempt" if error is not None else "returned",
            "error_type": None if error is None else f"{type(error).__module__}.{type(error).__qualname__}",
            "result": result, "context": operations._context,
            "precision": "opaque_provider_within_atomic_transform",
        })


@lru_cache(maxsize=1)
def _dependency_snapshot() -> tuple[dict[str, Any], ...]:
    """Snapshot optional verbalizers and recursive *installed* requirements once.

    Requirement markers/extras are not an execution graph. Including installed
    requirements is conservative; the scope string deliberately makes no claim
    that this inventory locks all dynamic Python execution or provider state.
    """
    queue = ["inflect", "num2words"]
    visited: set[str] = set()
    records: list[dict[str, Any]] = []
    while queue:
        name = queue.pop(0)
        key = name.lower().replace("_", "-")
        if key in visited:
            continue
        visited.add(key)
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            records.append({"name": name, "installed": False})
            continue
        requirements = distribution.requires or []
        for requirement in requirements:
            match = re.match(r"[A-Za-z0-9_.-]+", requirement)
            if match:
                queue.append(match.group(0))
        files = []
        for entry in distribution.files or []:
            if "__pycache__" in entry.parts or entry.suffix in {".pyc", ".pyo"}:
                continue
            path = Path(distribution.locate_file(entry)).resolve()
            if path.is_file():
                files.append(file_binding(path))
        records.append({"name": distribution.metadata["Name"], "installed": True,
                        "version": distribution.version, "requirements": requirements,
                        "files": sorted(files, key=lambda item: item["path"])})
    return tuple(records)


def normalizer_profile(normalizer: Any, language: str, wrapper_paths: Sequence[str | Path] = ()) -> dict[str, Any]:
    from . import normalization_trace, text_normalizer
    engine = normalizer._inflect
    return {
        "schema": "scyllasband_spoken_normalizer_profile_v1",
        "language_requested": language, "language_resolved": normalizer._normalizer_language(language),
        "config": asdict(normalizer.config),
        "semantic_policy_sha256": text_normalizer.SPOKEN_TEXT_NORMALIZER_SHA256,
        "code": [file_binding(path) for path in (__file__, text_normalizer.__file__, normalization_trace.__file__)],
        "wrapper_code": [file_binding(path) for path in wrapper_paths],
        "normalizer_type": f"{type(normalizer).__module__}.{type(normalizer).__qualname__}",
        "inflect_type": None if engine is None else f"{type(engine).__module__}.{type(engine).__qualname__}",
        "num2words_callable": None if text_normalizer._num2words_raw is None else
            f"{text_normalizer._num2words_raw.__module__}.{text_normalizer._num2words_raw.__qualname__}",
        "dependencies": deepcopy(list(_dependency_snapshot())),
        "unicode_version": unicodedata.unidata_version,
        "reproducibility_scope": "current_source_files_and_process_cached_installed_dependency_snapshot_not_full_execution_lock",
        "mutable_provider_state_bound": False,
        "external_provider_internal_mapping": "opaque_atomic",
        "linguistic_correctness_validated": False,
        "raw_lexical_anchor_resolution": "requires_separate_validated_word_consumer",
    }


def _keys(value: Any, fields: set[str], label: str) -> None:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"invalid {label} fields")


def _string(value: Any, label: str, *, empty: bool = False) -> None:
    if type(value) is not str or (not empty and not value):
        raise ValueError(f"invalid {label} string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"invalid {label} Unicode") from exc


def _sha(value: Any) -> None:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("invalid SHA256")


def _binding(value: Any) -> None:
    _keys(value, {"path", "size_bytes", "sha256"}, "file binding")
    _string(value["path"], "binding path")
    if not Path(value["path"]).is_absolute() or "\x00" in value["path"]:
        raise ValueError("binding path must be absolute without NUL")
    if type(value["size_bytes"]) is not int or value["size_bytes"] < 0:
        raise ValueError("invalid binding size")
    _sha(value["sha256"])


def _bindings(value: Any) -> None:
    if type(value) is not list:
        raise ValueError("bindings must be a list")
    for entry in value:
        _binding(entry)
    if len({entry["path"] for entry in value}) != len(value):
        raise ValueError("duplicate binding path")


def _validate_profile(profile: Any) -> None:
    fields = {"schema", "language_requested", "language_resolved", "config", "semantic_policy_sha256",
              "code", "wrapper_code", "normalizer_type", "inflect_type", "num2words_callable", "dependencies",
              "unicode_version", "reproducibility_scope", "mutable_provider_state_bound",
              "external_provider_internal_mapping", "linguistic_correctness_validated", "raw_lexical_anchor_resolution"}
    _keys(profile, fields, "normalizer profile")
    constants = {"schema": "scyllasband_spoken_normalizer_profile_v1",
                 "reproducibility_scope": "current_source_files_and_process_cached_installed_dependency_snapshot_not_full_execution_lock",
                 "external_provider_internal_mapping": "opaque_atomic",
                 "raw_lexical_anchor_resolution": "requires_separate_validated_word_consumer"}
    if any(profile[key] != value for key, value in constants.items()):
        raise ValueError("invalid profile policy")
    if profile["mutable_provider_state_bound"] is not False or profile["linguistic_correctness_validated"] is not False:
        raise ValueError("unsupported profile correctness/state claim")
    for key in ("language_requested", "language_resolved", "normalizer_type", "unicode_version"):
        _string(profile[key], key)
    if profile["language_resolved"] not in {"en", "es", "it", "fr", "de", "vi"}:
        raise ValueError("invalid resolved language")
    for key in ("inflect_type", "num2words_callable"):
        if profile[key] is not None:
            _string(profile[key], key)
    _sha(profile["semantic_policy_sha256"])
    _keys(profile["config"], {"expand_currency", "expand_numbers", "expand_ordinals", "expand_percentages",
                              "expand_times", "expand_dates", "normalize_at_sign", "normalize_punctuation"}, "config")
    if any(type(value) is not bool for value in profile["config"].values()):
        raise ValueError("config flags must be booleans")
    _bindings(profile["code"])
    _bindings(profile["wrapper_code"])
    if len(profile["code"]) != 3:
        raise ValueError("profile must bind adapter, normalizer and trace core")
    if type(profile["dependencies"]) is not list:
        raise ValueError("dependencies must be a list")
    names = set()
    for record in profile["dependencies"]:
        if type(record) is not dict or type(record.get("installed")) is not bool:
            raise ValueError("invalid dependency availability")
        _keys(record, {"name", "installed", "version", "requirements", "files"} if record["installed"] else
              {"name", "installed"}, "dependency")
        _string(record["name"], "dependency name")
        key = record["name"].lower().replace("_", "-")
        if key in names:
            raise ValueError("duplicate dependency")
        names.add(key)
        if record["installed"]:
            _string(record["version"], "dependency version")
            if type(record["requirements"]) is not list:
                raise ValueError("requirements must be a list")
            for requirement in record["requirements"]:
                _string(requirement, "requirement")
            _bindings(record["files"])


def _validate_provider_calls(calls: Any, trace: NormalizationTrace) -> None:
    if type(calls) is not list:
        raise ValueError("provider calls must be a list")
    for call in calls:
        _keys(call, {"provider", "value", "language", "ordinal", "status", "error_type", "result", "context", "precision"}, "provider call")
        if call["provider"] not in {"inflect", "num2words", "builtin_fallback"} or call["language"] not in {"en", "es", "it", "fr", "de", "vi"}:
            raise ValueError("invalid provider identity/language")
        if type(call["value"]) is not int or type(call["ordinal"]) is not bool:
            raise ValueError("invalid provider value/ordinal")
        if call["precision"] != "opaque_provider_within_atomic_transform":
            raise ValueError("invalid provider precision")
        if call["status"] == "returned":
            _string(call["result"], "provider result", empty=True)
            if call["error_type"] is not None:
                raise ValueError("successful provider cannot have error")
        elif call["status"] == "failed_fallback_attempt":
            _string(call["error_type"], "provider error")
            if call["result"] is not None:
                raise ValueError("failed provider cannot have result")
        else:
            raise ValueError("invalid provider status")
        context = call["context"]
        _keys(context, {"rule_id", "input_sha256", "input_span"}, "provider context")
        _string(context["rule_id"], "provider rule")
        _sha(context["input_sha256"])
        span = context["input_span"]
        if type(span) is not list or len(span) != 2 or any(type(x) is not int for x in span) or not 0 <= span[0] < span[1]:
            raise ValueError("invalid provider span")
        if not any(stage.rule_id == context["rule_id"] and stage.input_sha256 == context["input_sha256"]
                   and span[1] <= stage.input_length for stage in trace.stages):
            raise ValueError("provider call does not bind an actual trace stage")


@dataclass(frozen=True)
class SpokenNormalizationTrace:
    trace: NormalizationTrace
    profile: dict[str, Any]
    provider_calls: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "trace": self.trace.to_dict(), "profile": self.profile,
                "provider_calls": list(self.provider_calls)}

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict())

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, expected_sha256: str) -> "SpokenNormalizationTrace":
        # The trusted complete digest binds profile/callback evidence. Structural
        # replay alone cannot authenticate arbitrary producer rewrite semantics.
        if type(value) is not dict or set(value) != {"schema", "trace", "profile", "provider_calls"}:
            raise ValueError("invalid spoken normalization trace fields")
        if value["schema"] != SCHEMA or _digest(value) != expected_sha256:
            raise ValueError("spoken normalization schema or trusted digest mismatch")
        trace = NormalizationTrace.from_dict(value["trace"])
        _validate_profile(value["profile"])
        _validate_provider_calls(value["provider_calls"], trace)
        if value["profile"]["unicode_version"] != trace.profile.unicode_version:
            raise ValueError("producer/core Unicode profile mismatch")
        return cls(trace, value["profile"], tuple(value["provider_calls"]))

    def verify_profile_files(self, *, include_dependencies: bool = True) -> None:
        """Verify current file bytes; this does not rerun provider semantics."""
        _validate_profile(self.profile)
        bindings = self.profile["code"] + self.profile["wrapper_code"]
        if include_dependencies:
            bindings += [binding for record in self.profile["dependencies"] if record["installed"]
                         for binding in record["files"]]
        for binding in bindings:
            if file_binding(binding["path"]) != binding:
                raise ValueError(f"current profile file differs: {binding['path']}")


def trace_spoken_text(normalizer: Any, text: str, *, language: str,
                      builder: TraceBuilder | None = None,
                      postprocess: Callable[[str, NormalizationOperations], str] | None = None,
                      wrapper_paths: Sequence[str | Path] = ()) -> SpokenNormalizationTrace:
    value = str(text or "")
    builder = builder if builder is not None else TraceBuilder(value)
    if builder.text != value:
        raise ValueError("supplied input differs from composed trace builder")
    operations = NormalizationOperations(builder)
    token = _ACTIVE.set(operations)
    try:
        result = normalizer._normalize_impl(value, language=language, operations=operations)
        if postprocess is not None:
            result = postprocess(result, operations)
        if result != builder.text:
            raise ValueError("normalizer/postprocess returned untraced output")
    finally:
        _ACTIVE.reset(token)
    result = SpokenNormalizationTrace(builder.finish(), normalizer_profile(normalizer, language, wrapper_paths),
                                      tuple(operations.provider_calls))
    _validate_profile(result.profile)
    _validate_provider_calls(list(result.provider_calls), result.trace)
    return result
