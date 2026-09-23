"""Standard-library provenance mechanics; not a wired production normalizer.

Replacements are emitted by a caller that owns the transformation. They retain
atomic input support, not a guessed character alignment. NFC is the one built-in
semantic transform and verifies its tagged result against the pinned Unicode
implementation. Serialized traces require a trusted digest for authenticity;
replay alone verifies mechanics, not the meaning of arbitrary caller rewrites.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
from typing import Any, Iterable, Mapping, Sequence
import unicodedata

SCHEMA = "scyllasband_normalization_trace_core_v1"
_KERNEL_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_RULE = re.compile(r"[a-z][a-z0-9_.:-]{0,127}\Z")


class TraceError(ValueError):
    """Malformed, mismatched or unsupported trace evidence."""


def _text(value: Any) -> str:
    if not isinstance(value, str):
        raise TraceError("Text must be a string")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise TraceError("Unpaired surrogates are not valid trace text") from exc
    return value


def text_sha256(text: str) -> str:
    return hashlib.sha256(_text(text).encode("utf-8")).hexdigest()


def _integer(value: Any) -> int:
    if type(value) is not int:
        raise TraceError("Offsets and counts must be integers, not booleans")
    return value


def _fields(value: Any, names: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(names.split()):
        raise TraceError("Unknown or missing schema fields")


def _list(value: Any) -> list:
    if not isinstance(value, list):
        raise TraceError("Serialized sequences must be arrays")
    return value


def _digest(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise TraceError("Expected a lowercase SHA256 digest")
    return value


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class CoreProfile:
    kernel_sha256: str
    python_implementation: str
    python_version: str
    unicode_version: str

    @classmethod
    def current(cls) -> CoreProfile:
        return cls(_KERNEL_SHA256, platform.python_implementation(), sys.version,
                   unicodedata.unidata_version)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CoreProfile:
        _fields(value, "kernel_sha256 python_implementation python_version unicode_version")
        _digest(value["kernel_sha256"])
        if any(not isinstance(v, str) or not v for v in value.values()):
            raise TraceError("Malformed core profile")
        result = cls(**value)
        if result != cls.current():
            raise TraceError("Trace requires a different kernel/Python/Unicode profile")
        return result


@dataclass(frozen=True)
class Span:
    start: int
    end: int

    def __post_init__(self) -> None:
        if _integer(self.start) < 0 or _integer(self.end) < self.start:
            raise TraceError("Invalid half-open codepoint span")

    @classmethod
    def from_pair(cls, value: Any) -> Span:
        if not isinstance(value, list) or len(value) != 2:
            raise TraceError("Span must contain exactly two offsets")
        return cls(*value)


@dataclass(frozen=True)
class OffsetTable:
    utf8: tuple[int, ...]
    utf16: tuple[int, ...]

    @classmethod
    def for_text(cls, text: str) -> OffsetTable:
        _text(text)
        utf8, utf16 = [0], [0]
        for char in text:
            utf8.append(utf8[-1] + len(char.encode("utf-8")))
            utf16.append(utf16[-1] + len(char.encode("utf-16-le")) // 2)
        return cls(tuple(utf8), tuple(utf16))

    def __post_init__(self) -> None:
        for table in (self.utf8, self.utf16):
            if (type(table) is not tuple or not table or table[0] != 0
                    or any(type(x) is not int for x in table)
                    or any(a >= b for a, b in zip(table, table[1:]))):
                raise TraceError("Malformed offset boundary table")
        if len(self.utf8) != len(self.utf16):
            raise TraceError("Offset tables disagree on codepoint count")

    def from_utf8(self, offset: int) -> int:
        return self._inverse(self.utf8, offset)

    def from_utf16(self, offset: int) -> int:
        return self._inverse(self.utf16, offset)

    @staticmethod
    def _inverse(table: tuple[int, ...], offset: int) -> int:
        _integer(offset)
        index = bisect_left(table, offset)
        if index == len(table) or table[index] != offset:
            raise TraceError("Offset is not an exact encoded codepoint boundary")
        return index

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], text: str) -> OffsetTable:
        _fields(value, "utf8 utf16")
        result = cls(tuple(_list(value["utf8"])), tuple(_list(value["utf16"])))
        if result != cls.for_text(text):
            raise TraceError("Offset tables do not encode the bound text")
        return result


@dataclass(frozen=True)
class NFCResult:
    text: str
    # Ordered original-input codepoint IDs for each resulting codepoint.
    contributors: tuple[tuple[int, ...], ...]


@lru_cache(maxsize=4096)
def _decompose(char: str) -> str:
    return unicodedata.normalize("NFD", char)


@lru_cache(maxsize=8192)
def _compose_pair(left: str, right: str) -> str | None:
    result = unicodedata.normalize("NFC", left + right)
    return result if len(result) == 1 else None


def _unique(values: Iterable[Any]) -> tuple:
    return tuple(dict.fromkeys(values))


def tagged_nfc(text: str) -> NFCResult:
    """Decompose/order/compose with origin tags, never align two output strings."""
    _text(text)
    ordered: list[tuple[str, tuple[int, ...]]] = []
    marks: list[tuple[str, tuple[int, ...]]] = []
    for index, char in enumerate(text):
        for part in _decompose(char):
            unit = (part, (index,))
            if unicodedata.combining(part) == 0:
                ordered.extend(sorted(marks, key=lambda x: unicodedata.combining(x[0])))
                marks.clear()
                ordered.append(unit)
            else:
                marks.append(unit)
    ordered.extend(sorted(marks, key=lambda x: unicodedata.combining(x[0])))
    result: list[tuple[str, tuple[int, ...]]] = []
    starter: int | None = None
    last_class = 0
    for char, origins in ordered:
        current_class = unicodedata.combining(char)
        composite = None
        if starter is not None and (last_class < current_class or last_class == 0):
            composite = _compose_pair(result[starter][0], char)
        if composite is not None:
            result[starter] = (composite, _unique((*result[starter][1], *origins)))
            continue
        if current_class == 0:
            starter = len(result)
        result.append((char, origins))
        last_class = current_class
    normalized = "".join(char for char, _ in result)
    if normalized != unicodedata.normalize("NFC", text):
        raise TraceError("Tagged Unicode transform disagrees with pinned NFC")
    return NFCResult(normalized, tuple(origins for _, origins in result))


@dataclass(frozen=True)
class Replacement:
    span: Span
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.span, Span):
            raise TraceError("Replacement requires a typed input span")
        _text(self.text)
        if self.span.start == self.span.end and not self.text:
            raise TraceError("Empty insertion is not a transformation")


@dataclass(frozen=True)
class Event:
    kind: str
    span: Span
    input_sha256: str
    text: str
    contributors: tuple[tuple[int, ...], ...] = ()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Event:
        _fields(value, "kind span input_sha256 text contributors")
        if value["kind"] not in ("copy", "rewrite", "delete", "insert", "nfc"):
            raise TraceError("Unknown event kind")
        return cls(value["kind"], Span.from_pair(value["span"]), _digest(value["input_sha256"]),
                   _text(value["text"]), tuple(tuple(_list(x)) for x in _list(value["contributors"])))

    def to_dict(self) -> dict:
        return {"kind": self.kind, "span": [self.span.start, self.span.end],
                "input_sha256": self.input_sha256, "text": self.text,
                "contributors": [list(x) for x in self.contributors]}


@dataclass(frozen=True)
class Stage:
    rule_id: str
    input_sha256: str
    output_sha256: str
    input_length: int
    output_length: int
    events: tuple[Event, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Stage:
        _fields(value, "rule_id input_sha256 output_sha256 input_length output_length events")
        return cls(value["rule_id"], _digest(value["input_sha256"]), _digest(value["output_sha256"]),
                   _integer(value["input_length"]), _integer(value["output_length"]),
                   tuple(Event.from_dict(x) for x in _list(value["events"])))

    def to_dict(self) -> dict:
        return {"rule_id": self.rule_id, "input_sha256": self.input_sha256,
                "output_sha256": self.output_sha256, "input_length": self.input_length,
                "output_length": self.output_length, "events": [e.to_dict() for e in self.events]}


@dataclass(frozen=True)
class ProvenanceRun:
    span: Span
    raw_spans: tuple[Span, ...]
    precision: str
    group_ids: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"span": [self.span.start, self.span.end],
                "raw_spans": [[s.start, s.end] for s in self.raw_spans],
                "precision": self.precision, "group_ids": list(self.group_ids)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProvenanceRun:
        _fields(value, "span raw_spans precision group_ids")
        return cls(Span.from_pair(value["span"]), tuple(Span.from_pair(x) for x in _list(value["raw_spans"])),
                   value["precision"], tuple(_list(value["group_ids"])))


def _ordered_spans(spans: Iterable[Span]) -> tuple[Span, ...]:
    # Preserve first-contributor order, not a sorted bounding hull.
    indexes = _unique(i for span in spans for i in range(span.start, span.end))
    result: list[Span] = []
    for index in indexes:
        if result and result[-1].end == index:
            result[-1] = Span(result[-1].start, index + 1)
        else:
            result.append(Span(index, index + 1))
    return tuple(result)


def _slice(runs: tuple[ProvenanceRun, ...], span: Span,
           starts: tuple[int, ...]) -> list[ProvenanceRun]:
    result = []
    index = max(0, bisect_right(starts, span.start) - 1)
    while index < len(runs):
        run = runs[index]
        index += 1
        if run.span.start >= span.end:
            break
        start, end = max(span.start, run.span.start), min(span.end, run.span.end)
        if start >= end:
            continue
        support = run.raw_spans
        if run.precision == "copy_exact":
            raw_start = support[0].start + start - run.span.start
            support = (Span(raw_start, raw_start + end - start),)
        result.append(ProvenanceRun(Span(start, end), support, run.precision, run.group_ids))
    return result


def _merge(runs: Iterable[ProvenanceRun]) -> tuple[ProvenanceRun, ...]:
    result: list[ProvenanceRun] = []
    for run in runs:
        if result:
            prev = result[-1]
            if prev.span.end == run.span.start and prev.precision == run.precision and prev.group_ids == run.group_ids:
                if prev.raw_spans == run.raw_spans and run.precision != "copy_exact":
                    result[-1] = ProvenanceRun(Span(prev.span.start, run.span.end), run.raw_spans, run.precision, run.group_ids)
                    continue
                if (run.precision == "copy_exact" and prev.raw_spans[0].end == run.raw_spans[0].start):
                    result[-1] = ProvenanceRun(Span(prev.span.start, run.span.end),
                        (Span(prev.raw_spans[0].start, run.raw_spans[0].end),), "copy_exact", ())
                    continue
        result.append(run)
    return tuple(result)


def _nfc_components(contributors: tuple[tuple[int, ...], ...], length: int) -> list[int]:
    parents = list(range(length))
    def find(x: int) -> int:
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x
    for origins in contributors:
        head = find(origins[0])
        for item in origins[1:]:
            left, right = find(head), find(item)
            parents[max(left, right)] = min(left, right)
            head = min(left, right)
    return [find(i) for i in range(length)]


def _replay_stage(text: str, runs: tuple[ProvenanceRun, ...], stage: Stage,
                  stage_index: int) -> tuple[str, tuple[ProvenanceRun, ...]]:
    if (not isinstance(stage.rule_id, str) or _RULE.fullmatch(stage.rule_id) is None
            or stage.input_sha256 != text_sha256(text)
            or _integer(stage.input_length) != len(text)):
        raise TraceError("Stage rule or input binding mismatch")
    if stage.rule_id == "unicode.nfc":
        if len(stage.events) != 1 or stage.events[0].kind != "nfc":
            raise TraceError("Reserved NFC stage requires the tagged NFC event")
    elif any(event.kind == "nfc" for event in stage.events):
        raise TraceError("Tagged NFC event requires its reserved stage identity")
    cursor = output_cursor = 0
    starts = tuple(run.span.start for run in runs)
    chunks: list[str] = []
    emitted: list[ProvenanceRun] = []
    for event_index, event in enumerate(stage.events):
        start, end = event.span.start, event.span.end
        if start != cursor or end > len(text) or event.input_sha256 != text_sha256(text[start:end]):
            raise TraceError("Events must exactly partition the immediate input")
        _text(event.text)
        identity = f"s{stage_index}:e{event_index}"
        support_runs = _slice(runs, event.span, starts)
        if event.kind == "copy":
            if start == end or event.text != text[start:end] or event.contributors:
                raise TraceError("Copy event differs from its exact input")
            for run in support_runs:
                emitted.append(ProvenanceRun(Span(output_cursor + run.span.start - start,
                    output_cursor + run.span.end - start), run.raw_spans, run.precision, run.group_ids))
        elif event.kind in ("rewrite", "delete", "insert"):
            if (event.contributors or (event.kind == "insert") != (start == end)
                    or (event.kind == "delete") != (not event.text)):
                raise TraceError("Malformed rewrite/delete/insert event")
            if event.kind != "delete":
                support = _ordered_spans(s for r in support_runs for s in r.raw_spans)
                groups = _unique((*[g for r in support_runs for g in r.group_ids], identity))
                emitted.append(ProvenanceRun(Span(output_cursor, output_cursor + len(event.text)), support,
                    "rewrite_group" if support else "inserted", groups))
        elif event.kind == "nfc":
            if len(stage.events) != 1 or start != 0 or end != len(text):
                raise TraceError("NFC event must consume the whole immediate input")
            expected = tagged_nfc(text)
            if event.text != expected.text or event.contributors != expected.contributors:
                raise TraceError("NFC output/contributors differ from the tagged transform")
            if any(type(i) is not int for ids in event.contributors for i in ids):
                raise TraceError("NFC contributor IDs must be integers, not booleans")
            components = _nfc_components(event.contributors, len(text))
            for index, (char, ids) in enumerate(zip(event.text, event.contributors, strict=True)):
                parts = [part for i in ids for part in _slice(runs, Span(i, i + 1), starts)]
                if len(ids) == 1 and char == text[ids[0]]:
                    part = parts[0]
                    emitted.append(ProvenanceRun(Span(index, index + 1), part.raw_spans,
                                                   part.precision, part.group_ids))
                else:
                    support = _ordered_spans(s for r in parts for s in r.raw_spans)
                    groups = _unique((*[g for r in parts for g in r.group_ids],
                                      f"{identity}:nfc{components[ids[0]]}"))
                    emitted.append(ProvenanceRun(Span(index, index + 1), support,
                        "rewrite_group" if support else "inserted", groups))
        else:
            raise TraceError("Unknown event kind")
        cursor = end
        output_cursor += len(event.text)
        chunks.append(event.text)
    result = "".join(chunks)
    if (cursor != len(text) or stage.output_sha256 != text_sha256(result)
            or _integer(stage.output_length) != len(result)):
        raise TraceError("Uncovered input or stage output binding mismatch")
    return result, _merge(emitted)


@dataclass(frozen=True)
class NormalizationTrace:
    raw_text: str
    normalized_text: str
    profile: CoreProfile
    raw_offsets: OffsetTable
    normalized_offsets: OffsetTable
    stages: tuple[Stage, ...]
    provenance: tuple[ProvenanceRun, ...]

    def validate(self, *, expected_raw_sha256: str | None = None,
                 expected_trace_sha256: str | None = None) -> None:
        _text(self.raw_text)
        _text(self.normalized_text)
        if self.profile != CoreProfile.current():
            raise TraceError("Unexpected core profile")
        if self.raw_offsets != OffsetTable.for_text(self.raw_text) or self.normalized_offsets != OffsetTable.for_text(self.normalized_text):
            raise TraceError("Offset table mismatch")
        if expected_raw_sha256 is not None and _digest(expected_raw_sha256) != text_sha256(self.raw_text):
            raise TraceError("Unexpected original text")
        text = self.raw_text
        runs = (ProvenanceRun(Span(0, len(text)), (Span(0, len(text)),), "copy_exact", ()),) if text else ()
        for index, stage in enumerate(self.stages):
            text, runs = _replay_stage(text, runs, stage, index)
        if text != self.normalized_text or runs != self.provenance:
            raise TraceError("Final text/provenance differs from replay")
        if expected_trace_sha256 is not None and _digest(expected_trace_sha256) != self.sha256:
            raise TraceError("Trace does not match its trusted digest")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_json_bytes(self.to_dict())).hexdigest()

    def to_dict(self) -> dict:
        return {"schema": SCHEMA, "raw_text": self.raw_text,
                "raw_text_sha256": text_sha256(self.raw_text),
                "normalized_text": self.normalized_text,
                "normalized_text_sha256": text_sha256(self.normalized_text),
                "profile": asdict(self.profile),
                "raw_offsets": {"utf8": list(self.raw_offsets.utf8), "utf16": list(self.raw_offsets.utf16)},
                "normalized_offsets": {"utf8": list(self.normalized_offsets.utf8), "utf16": list(self.normalized_offsets.utf16)},
                "stages": [s.to_dict() for s in self.stages],
                "provenance": [r.to_dict() for r in self.provenance]}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, expected_raw_sha256: str | None = None,
                  expected_trace_sha256: str | None = None) -> NormalizationTrace:
        _fields(value, "schema raw_text raw_text_sha256 normalized_text normalized_text_sha256 profile raw_offsets normalized_offsets stages provenance")
        if value["schema"] != SCHEMA:
            raise TraceError("Unknown trace schema")
        for name in ("raw_text", "normalized_text"):
            if text_sha256(value[name]) != _digest(value[f"{name}_sha256"]):
                raise TraceError("Text hash mismatch")
        result = cls(value["raw_text"], value["normalized_text"], CoreProfile.from_dict(value["profile"]),
                     OffsetTable.from_dict(value["raw_offsets"], value["raw_text"]),
                     OffsetTable.from_dict(value["normalized_offsets"], value["normalized_text"]),
                     tuple(Stage.from_dict(x) for x in _list(value["stages"])),
                     tuple(ProvenanceRun.from_dict(x) for x in _list(value["provenance"])))
        result.validate(expected_raw_sha256=expected_raw_sha256, expected_trace_sha256=expected_trace_sha256)
        return result

    @classmethod
    def from_json(cls, payload: str | bytes, **kwargs: Any) -> NormalizationTrace:
        def pairs(items: list[tuple[str, Any]]) -> dict:
            result = {}
            for key, value in items:
                if key in result:
                    raise TraceError("Duplicate JSON field")
                result[key] = value
            return result
        def constant(value: str) -> None:
            raise TraceError(f"Non-finite JSON constant: {value}")
        try:
            data = json.loads(payload, object_pairs_hook=pairs, parse_constant=constant)
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
            raise TraceError("Malformed trace JSON") from exc
        return cls.from_dict(data, **kwargs)


class TraceBuilder:
    """Emit literal transformation decisions; no search/diff-based provenance."""
    def __init__(self, raw_text: str) -> None:
        self.raw_text = _text(raw_text)
        self.text = raw_text
        self._stages: list[Stage] = []
        self._runs = (ProvenanceRun(Span(0, len(raw_text)), (Span(0, len(raw_text)),), "copy_exact", ()),) if raw_text else ()

    def _apply(self, rule_id: str, events: Sequence[Event]) -> None:
        output = "".join(e.text for e in events)
        stage = Stage(rule_id, text_sha256(self.text), text_sha256(output), len(self.text), len(output), tuple(events))
        output, runs = _replay_stage(self.text, self._runs, stage, len(self._stages))
        self._stages.append(stage)
        self.text, self._runs = output, runs

    def apply_replacements(self, replacements: Sequence[Replacement], *, rule_id: str) -> TraceBuilder:
        cursor = 0
        events: list[Event] = []
        for replacement in replacements:
            if not isinstance(replacement, Replacement):
                raise TraceError("Typed replacements required")
            start, end = replacement.span.start, replacement.span.end
            if start < cursor or end > len(self.text):
                raise TraceError("Replacements must be ordered, nonoverlapping and in bounds")
            if start > cursor:
                copied = self.text[cursor:start]
                events.append(Event("copy", Span(cursor, start), text_sha256(copied), copied))
            kind = "insert" if start == end else ("rewrite" if replacement.text else "delete")
            events.append(Event(kind, replacement.span, text_sha256(self.text[start:end]), replacement.text))
            cursor = end
        if cursor < len(self.text):
            copied = self.text[cursor:]
            events.append(Event("copy", Span(cursor, len(self.text)), text_sha256(copied), copied))
        self._apply(rule_id, events)
        return self

    def replace(self, start: int, end: int, text: str, *, rule_id: str) -> TraceBuilder:
        return self.apply_replacements([Replacement(Span(start, end), text)], rule_id=rule_id)

    def normalize_nfc(self) -> TraceBuilder:
        result = tagged_nfc(self.text)
        self._apply("unicode.nfc", [Event("nfc", Span(0, len(self.text)), text_sha256(self.text),
                                          result.text, result.contributors)])
        return self

    def finish(self) -> NormalizationTrace:
        trace = NormalizationTrace(self.raw_text, self.text, CoreProfile.current(),
            OffsetTable.for_text(self.raw_text), OffsetTable.for_text(self.text), tuple(self._stages), self._runs)
        trace.validate()
        return trace
