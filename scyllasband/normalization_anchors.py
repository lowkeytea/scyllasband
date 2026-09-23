"""Exact text provenance for lexical delivery requests; no time allocation.

The scanner labels lexical runs, not linguistic/morphological word boundaries.
Only complete words resolve automatically. Atomic normalization groups never
supply an invented internal alignment. Consumers must supply the trusted trace
digest from their producer; a self-reported digest is not authentication.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import unicodedata

from .normalization_trace import NormalizationTrace, Span, TraceError, text_sha256

WORDS_SCHEMA = "scyllasband_normalized_lexical_words_v1"
ANCHOR_SCHEMA = "scyllasband_raw_lexical_anchor_v1"
PROJECTION_SCHEMA = "scyllasband_lexical_anchor_projection_v1"
_SCANNER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_APOSTROPHES = frozenset(("'", "’"))


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise TraceError("Expected a lowercase SHA256 digest")
    return value


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128 or not value.isprintable():
        raise TraceError("Expected a nonempty printable identifier of at most 128 characters")
    return value


def _spans(indexes: set[int]) -> tuple[Span, ...]:
    result: list[Span] = []
    for index in sorted(indexes):
        if result and result[-1].end == index:
            result[-1] = Span(result[-1].start, index + 1)
        else:
            result.append(Span(index, index + 1))
    return tuple(result)


def _indexes(spans: Sequence[Span]) -> set[int]:
    return {i for span in spans for i in range(span.start, span.end)}


@dataclass(frozen=True)
class LexicalWord:
    word_id: int
    text: str
    start_codepoint: int
    end_codepoint: int


def _scan(text: str) -> tuple[LexicalWord, ...]:
    """UCD letters/numbers, attached marks and internal straight/curly apostrophes.

    Symbols, separators, orphan marks and leading/trailing quotes are not words.
    Adjacent CJK characters form a lexical run, not an asserted language token.
    """
    def base(char: str) -> bool:
        return unicodedata.category(char)[0] in "LN"
    words: list[LexicalWord] = []
    i = 0
    while i < len(text):
        if not base(text[i]):
            i += 1
            continue
        start = i
        i += 1
        while i < len(text):
            if base(text[i]) or unicodedata.category(text[i]).startswith("M"):
                i += 1
            elif text[i] in _APOSTROPHES and i + 1 < len(text) and base(text[i + 1]):
                i += 1
            else:
                break
        words.append(LexicalWord(len(words), text[start:i], start, i))
    return tuple(words)


@dataclass(frozen=True)
class NormalizedLexicalWords:
    trace_sha256: str
    raw_text_sha256: str
    normalized_text: str
    words: tuple[LexicalWord, ...]

    @classmethod
    def from_trace(cls, trace: NormalizationTrace, *, expected_trace_sha256: str) -> NormalizedLexicalWords:
        trace.validate(expected_trace_sha256=_digest(expected_trace_sha256))
        return cls(trace.sha256, text_sha256(trace.raw_text), trace.normalized_text, _scan(trace.normalized_text))

    def to_dict(self) -> dict:
        return {"schema": WORDS_SCHEMA, "trace_sha256": self.trace_sha256,
                "raw_text_sha256": self.raw_text_sha256,
                "normalized_text": self.normalized_text,
                "normalized_text_sha256": text_sha256(self.normalized_text),
                "scanner": {"implementation_sha256": _SCANNER_SHA256,
                            "unicode_version": unicodedata.unidata_version,
                            "policy": "letters_numbers_attached_marks_internal_apostrophes_v1"},
                "words": [asdict(word) for word in self.words]}

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_json(self.to_dict()).encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, trace: NormalizationTrace,
                  expected_trace_sha256: str) -> NormalizedLexicalWords:
        expected = cls.from_trace(trace, expected_trace_sha256=expected_trace_sha256)
        # Strict JSON equality also distinguishes bool offsets from integer 0/1.
        if not isinstance(value, Mapping) or _json(value) != _json(expected.to_dict()):
            raise TraceError("Lexical words/profile differ from the verified trace and scanner")
        return expected


@dataclass(frozen=True)
class RawLexicalAnchor:
    request_id: str
    raw_text_sha256: str
    span: Span

    def __post_init__(self) -> None:
        _identifier(self.request_id)
        _digest(self.raw_text_sha256)
        if not isinstance(self.span, Span) or self.span.start == self.span.end:
            raise TraceError("Lexical requests need a nonempty span; boundary anchors require a separate contract")

    @classmethod
    def from_offsets(cls, trace: NormalizationTrace, *, request_id: str,
                     raw_text_sha256: str, start: int, end: int,
                     unit: str = "codepoint") -> RawLexicalAnchor:
        if _digest(raw_text_sha256) != text_sha256(trace.raw_text):
            raise TraceError("Anchor belongs to different original text")
        if unit == "utf8":
            start, end = trace.raw_offsets.from_utf8(start), trace.raw_offsets.from_utf8(end)
        elif unit == "utf16":
            start, end = trace.raw_offsets.from_utf16(start), trace.raw_offsets.from_utf16(end)
        elif unit != "codepoint":
            raise TraceError("Unknown raw offset unit")
        result = cls(request_id, raw_text_sha256, Span(start, end))
        if result.span.end > len(trace.raw_text):
            raise TraceError("Anchor lies outside the original text")
        return result

    def to_dict(self) -> dict:
        return {"schema": ANCHOR_SCHEMA, "request_id": self.request_id,
                "raw_text_sha256": self.raw_text_sha256,
                "span": [self.span.start, self.span.end], "offset_unit": "codepoint"}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RawLexicalAnchor:
        fields = {"schema", "request_id", "raw_text_sha256", "span", "offset_unit"}
        if (not isinstance(value, Mapping) or set(value) != fields
                or value["schema"] != ANCHOR_SCHEMA or value["offset_unit"] != "codepoint"):
            raise TraceError("Malformed raw lexical anchor")
        return cls(value["request_id"], value["raw_text_sha256"], Span.from_pair(value["span"]))


@dataclass(frozen=True)
class AnchorProjection:
    anchor: RawLexicalAnchor
    words_sha256: str
    policy: str
    status: str
    candidate_word_ids: tuple[int, ...]
    target_word_ids: tuple[int, ...]
    group_ids: tuple[str, ...]
    partial_group_ids: tuple[str, ...]
    partial_word_ids: tuple[int, ...]
    normalized_support: tuple[Span, ...]
    raw_without_descendants: tuple[Span, ...]

    @property
    def resolved(self) -> bool:
        return self.status in ("resolved_exact", "resolved_group", "resolved_expanded_group")

    def to_dict(self) -> dict:
        result = asdict(self)
        result["schema"] = PROJECTION_SCHEMA
        result["anchor"] = self.anchor.to_dict()
        for key in ("candidate_word_ids", "target_word_ids", "group_ids", "partial_group_ids", "partial_word_ids"):
            result[key] = list(getattr(self, key))
        for key in ("normalized_support", "raw_without_descendants"):
            result[key] = [[span.start, span.end] for span in getattr(self, key)]
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, trace: NormalizationTrace,
                  expected_trace_sha256: str) -> AnchorProjection:
        if not isinstance(value, Mapping) or "anchor" not in value or "policy" not in value:
            raise TraceError("Malformed anchor projection")
        expected = project_raw_anchor(trace, RawLexicalAnchor.from_dict(value["anchor"]),
                                      expected_trace_sha256=expected_trace_sha256, policy=value["policy"])
        if _json(value) != _json(expected.to_dict()):
            raise TraceError("Projection differs from verified source ownership")
        return expected


def project_raw_anchor(trace: NormalizationTrace, anchor: RawLexicalAnchor, *,
                       expected_trace_sha256: str, policy: str = "strict") -> AnchorProjection:
    """Map a raw lexical selection, conservatively preserving atomic groups.

    ``expand_groups`` is an explicit caller opt-in to include an entire touched
    normalization group. It never expands an arbitrary copied word fragment.
    Punctuation/deleted selections and boundaries never choose a nearest word.
    """
    words = NormalizedLexicalWords.from_trace(trace, expected_trace_sha256=expected_trace_sha256)
    if not isinstance(anchor, RawLexicalAnchor):
        raise TraceError("Typed raw lexical anchor required")
    if anchor.raw_text_sha256 != words.raw_text_sha256 or anchor.span.end > len(trace.raw_text):
        raise TraceError("Anchor does not bind this original text")
    if policy not in ("strict", "expand_groups"):
        raise TraceError("Unknown atomic group inheritance policy")
    selected = set(range(anchor.span.start, anchor.span.end))
    output: set[int] = set()
    surviving_raw: set[int] = set()
    touched_groups: set[str] = set()
    run_support = [_indexes(run.raw_spans) for run in trace.provenance]
    for run, support in zip(trace.provenance, run_support):
        surviving_raw.update(support)
        overlap = selected & support
        if not overlap:
            continue
        if run.precision == "copy_exact":
            output.update(run.span.start + i - run.raw_spans[0].start for i in overlap)
        else:
            output.update(range(run.span.start, run.span.end))
            touched_groups.update(run.group_ids)
    # Inherited groups can overlap a later rewrite. Follow real group identities
    # transitively, retaining all descendants; never approximate a bounding hull.
    changed = True
    while changed:
        changed = False
        for run in trace.provenance:
            if touched_groups.intersection(run.group_ids):
                new = set(run.group_ids) - touched_groups
                if new:
                    touched_groups.update(new)
                    changed = True
                output.update(range(run.span.start, run.span.end))
    group_support: dict[str, set[int]] = {group: set() for group in touched_groups}
    for run, support in zip(trace.provenance, run_support):
        for group in touched_groups.intersection(run.group_ids):
            group_support[group].update(support)
    partial_groups = tuple(sorted(group for group, support in group_support.items() if not support <= selected))
    candidates = tuple(word.word_id for word in words.words
                       if any(i in output for i in range(word.start_codepoint, word.end_codepoint)))
    partial_words = tuple(word.word_id for word in words.words if word.word_id in candidates
                          and any(i not in output for i in range(word.start_codepoint, word.end_codepoint)))
    if not candidates:
        status = "deleted" if not (selected & surviving_raw) else "no_lexical_target"
    elif partial_groups and policy == "strict":
        status = "ambiguous_group"
    elif partial_words:
        status = "partial_word"
    elif partial_groups:
        status = "resolved_expanded_group"
    elif touched_groups:
        status = "resolved_group"
    else:
        status = "resolved_exact"
    targets = candidates if status.startswith("resolved_") else ()
    return AnchorProjection(anchor, words.sha256, policy, status, candidates, targets,
                            tuple(sorted(touched_groups)), partial_groups, partial_words,
                            _spans(output), _spans(selected - surviving_raw))


@dataclass(frozen=True)
class LexicalDirective:
    projection: AnchorProjection
    axis: str
    value: float

    def __post_init__(self) -> None:
        if not isinstance(self.projection, AnchorProjection):
            raise TraceError("Directive requires a typed projection")
        _identifier(self.axis)
        if type(self.value) not in (int, float) or not math.isfinite(self.value):
            raise TraceError("Directive value must be finite and numeric")


@dataclass(frozen=True)
class DirectiveConflict:
    left_request_id: str
    right_request_id: str
    axis: str
    word_ids: tuple[int, ...]
    kind: str


def directive_conflicts(directives: Sequence[LexicalDirective]) -> tuple[DirectiveConflict, ...]:
    """Report opposing requests; never average, reorder or choose a winner.

    Inputs must be validated projections from a trusted producer/reader. This
    detects conflicts, not acceptable control ranges or acoustic feasibility.
    """
    ids: set[tuple[str, str]] = set()
    word_bindings: set[str] = set()
    for directive in directives:
        key = (directive.projection.anchor.request_id, directive.axis)
        if key in ids:
            raise TraceError("Duplicate request ID and axis")
        ids.add(key)
        word_bindings.add(directive.projection.words_sha256)
    if len(word_bindings) > 1:
        raise TraceError("Directives bind different normalized word contracts")
    conflicts = []
    for i, left in enumerate(directives):
        for right in directives[i + 1:]:
            if left.axis != right.axis or left.value == right.value:
                continue
            a, b = left.projection, right.projection
            overlap = tuple(sorted(set(a.candidate_word_ids) & set(b.candidate_word_ids)))
            if overlap:
                conflicts.append(DirectiveConflict(a.anchor.request_id, b.anchor.request_id, left.axis,
                    overlap, "conflict" if a.resolved and b.resolved else "potential_conflict"))
    return tuple(conflicts)
