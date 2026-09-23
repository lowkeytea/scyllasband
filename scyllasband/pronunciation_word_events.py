"""Native word-event evidence, independent of translation units and source time.

An exact WORD event span is a lexical anchor. It is not proof that all following
PHONEME events belong exclusively to that word: native multiword pronunciations
may report only the first word. No event-to-translation-unit alignment is implied.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import hashlib
from typing import Any, Mapping

from .normalization_anchors import NormalizedLexicalWords
from .normalization_trace import NormalizationTrace
from .pronunciation_contract import (
    AssetBinding, ProviderBinding, PronunciationError, _digest, _fields, _json, text_sha256,
)

SCHEMA = "scyllasband_native_word_events_v1"
BINDING_SCHEMA = "scyllasband_native_word_event_provider_v1"
SPAN_SCHEMA = "scyllasband_native_word_event_lexical_spans_v1"
MAX_INPUT = 8192
MAX_EVENTS = 32768
MAX_SYNTHETIC_SAMPLES = 2_000_000


def _integer(value: Any, minimum: int = 0, maximum: int = 2**31 - 1) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise PronunciationError("Bounded integer required")
    return value


@dataclass(frozen=True)
class EventProviderBinding:
    translation_provider: ProviderBinding
    event_code: tuple[AssetBinding, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.translation_provider, ProviderBinding):
            raise PronunciationError("Verified underlying native provider required")
        if (type(self.event_code) is not tuple or not self.event_code or len(self.event_code) > 32
                or any(not isinstance(x, AssetBinding) for x in self.event_code)
                or tuple(x.path for x in self.event_code) != tuple(sorted({x.path for x in self.event_code}))):
            raise PronunciationError("Unique sorted event implementation bindings required")

    def to_dict(self) -> dict:
        return {"schema": BINDING_SCHEMA, "translation_provider": self.translation_provider.to_dict(),
            "translation_provider_sha256": self.translation_provider.sha256,
            "event_code": [asdict(x) for x in self.event_code], "options": {
                "api": "espeak_Synth", "audio_output": 2, "initialize_flags": 0x8001,
                "buffer_ms": 0, "input_flags": 1, "position": 0, "position_type": 1,
                "end_position": 0, "native_event_names": True, "ipa_event_names": False,
                "ssml": False, "inline_phonemes": False, "end_pause_requested": False,
                "abi": "linux_little_endian_int32_pointer64_event40_offsets_0_4_8_12_16_20_24_32",
                "native_instance": "new_library_instance_per_capture",
                "synthetic_waveform": "discarded_no_source_timing",
                "max_input_codepoints": MAX_INPUT, "max_events": MAX_EVENTS,
                "max_synthetic_samples": MAX_SYNTHETIC_SAMPLES}}

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_json(self.to_dict())).hexdigest()

    def verify_assets(self) -> None:
        self.translation_provider.verify_assets()
        for asset in self.event_code:
            asset.verify()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, expected_sha256: str) -> EventProviderBinding:
        _fields(value, "schema translation_provider translation_provider_sha256 event_code options")
        if value["schema"] != BINDING_SCHEMA or type(value["event_code"]) is not list:
            raise PronunciationError("Unsupported native event provider")
        result = cls(ProviderBinding.from_dict(value["translation_provider"],
            expected_sha256=value["translation_provider_sha256"]),
            tuple(AssetBinding.from_dict(x) for x in value["event_code"]))
        if result.sha256 != _digest(expected_sha256) or _json(result.to_dict()) != _json(value):
            raise PronunciationError("Event provider differs from trusted binding/options")
        return result


@dataclass(frozen=True)
class NativeEvent:
    kind: int
    text_position: int
    length: int
    synthetic_audio_ms: int
    synthetic_sample: int
    number: int | None = None
    native_name_hex: str | None = None

    def __post_init__(self) -> None:
        if _integer(self.kind, 1, 8) in (3, 4):
            raise PronunciationError("MARK/PLAY events unsupported with SSML disabled")
        for value in (self.text_position, self.length, self.synthetic_audio_ms, self.synthetic_sample):
            _integer(value)
        if self.kind in (1, 2, 8):
            _integer(self.number, 1 if self.kind == 8 else 0)
        elif self.number is not None:
            raise PronunciationError("Only WORD/SENTENCE/SAMPLERATE events have a numeric payload")
        if self.kind == 7:
            if (not isinstance(self.native_name_hex, str) or len(self.native_name_hex) != 16
                    or any(x not in "0123456789abcdef" for x in self.native_name_hex)):
                raise PronunciationError("PHONEME event requires all eight raw name bytes")
        elif self.native_name_hex is not None:
            raise PronunciationError("Only PHONEME events carry native names")

    @property
    def native_name(self) -> str | None:
        if self.native_name_hex is None:
            return None
        try:
            return bytes.fromhex(self.native_name_hex).split(b"\0", 1)[0].decode("utf-8", "strict")
        except UnicodeDecodeError:
            return None  # Preserve raw bytes; truncation is not repaired.

    @property
    def native_name_terminated(self) -> bool:
        return self.native_name_hex is not None and b"\0" in bytes.fromhex(self.native_name_hex)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> NativeEvent:
        _fields(value, "kind text_position length synthetic_audio_ms synthetic_sample number native_name_hex")
        return cls(**value)


@dataclass(frozen=True)
class NativeWordEvents:
    provider: EventProviderBinding
    input_text: str
    events: tuple[NativeEvent, ...]
    sample_rate: int
    discarded_synthetic_samples: int
    callback_calls: int

    def __post_init__(self) -> None:
        if not isinstance(self.provider, EventProviderBinding):
            raise PronunciationError("Typed event provider required")
        text_sha256(self.input_text)
        if len(self.input_text) > MAX_INPUT or "\0" in self.input_text or "\x1f" in self.input_text:
            raise PronunciationError("Unsupported native event input")
        if (type(self.events) is not tuple or len(self.events) > MAX_EVENTS
                or any(not isinstance(x, NativeEvent) for x in self.events)):
            raise PronunciationError("Bounded native event tuple required")
        _integer(self.sample_rate, 1, 192000)
        _integer(self.discarded_synthetic_samples, 0, MAX_SYNTHETIC_SAMPLES)
        _integer(self.callback_calls, 0, MAX_EVENTS)

    def to_dict(self) -> dict:
        return {"schema": SCHEMA, "provider": self.provider.to_dict(), "provider_sha256": self.provider.sha256,
            "input_text": self.input_text, "input_text_sha256": text_sha256(self.input_text),
            "events": [asdict(x) for x in self.events], "sample_rate": self.sample_rate,
            "discarded_synthetic_samples": self.discarded_synthetic_samples, "callback_calls": self.callback_calls,
            "text_position_convention": "native_one_based_characters_unrepaired",
            "exclusive_word_phone_ownership_available": False,
            "translation_unit_association_available": False, "source_timing_available": False,
            "tone_targets_available": False}

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_json(self.to_dict())).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, expected_sha256: str,
                  expected_provider_sha256: str, expected_input_sha256: str,
                  verify_assets: bool = True) -> NativeWordEvents:
        _fields(value, "schema provider provider_sha256 input_text input_text_sha256 events sample_rate discarded_synthetic_samples callback_calls text_position_convention exclusive_word_phone_ownership_available translation_unit_association_available source_timing_available tone_targets_available")
        if value["schema"] != SCHEMA or type(value["events"]) is not list:
            raise PronunciationError("Unsupported native event evidence")
        result = cls(EventProviderBinding.from_dict(value["provider"], expected_sha256=expected_provider_sha256),
            value["input_text"], tuple(NativeEvent.from_dict(x) for x in value["events"]),
            value["sample_rate"], value["discarded_synthetic_samples"], value["callback_calls"])
        if (result.sha256 != _digest(expected_sha256) or text_sha256(result.input_text) != _digest(expected_input_sha256)
                or _json(result.to_dict()) != _json(value)):
            raise PronunciationError("Native event evidence differs from trusted/recomputed fields")
        if verify_assets:
            result.provider.verify_assets()
        return result


@dataclass(frozen=True)
class WordSpanCandidate:
    event_index: int
    native_position: int
    native_length: int
    status: str
    span_codepoints: tuple[int, int] | None
    candidate_word_ids: tuple[int, ...]
    exact_word_ids: tuple[int, ...]


@dataclass(frozen=True)
class LexicalEventSpans:
    trace_sha256: str
    words_sha256: str
    events_sha256: str
    candidates: tuple[WordSpanCandidate, ...]
    words_without_exact_event: tuple[int, ...]
    words_with_duplicate_exact_events: tuple[int, ...]
    exclusive_word_phone_ownership_available: bool = field(default=False, init=False)
    translation_unit_association_available: bool = field(default=False, init=False)

    def to_dict(self) -> dict:
        return {"schema": SPAN_SCHEMA, **asdict(self),
            "candidate_semantics": "reported_word_span_only_not_following_phone_ownership"}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, trace: NormalizationTrace,
                  expected_trace_sha256: str, events: NativeWordEvents,
                  expected_events_sha256: str) -> LexicalEventSpans:
        result = inspect_word_event_spans(trace, expected_trace_sha256=expected_trace_sha256,
            events=events, expected_events_sha256=expected_events_sha256)
        if _json(value) != _json(result.to_dict()):
            raise PronunciationError("Lexical event spans differ from recomputed input positions")
        return result


def inspect_word_event_spans(trace: NormalizationTrace, *, expected_trace_sha256: str,
                             events: NativeWordEvents, expected_events_sha256: str) -> LexicalEventSpans:
    """Check reported spans, retaining partial/duplicate/missing/nonlexical cases.

    This explicitly does not assign following phonemes to an exact word span, or
    widen a coalesced/length-capped span to the next reported word boundary.
    """
    words = NormalizedLexicalWords.from_trace(trace, expected_trace_sha256=expected_trace_sha256)
    if (not isinstance(events, NativeWordEvents) or events.sha256 != _digest(expected_events_sha256)
            or events.input_text != words.normalized_text):
        raise PronunciationError("Events must bind this exact normalized lexical input")
    candidates = []
    counts: Counter[int] = Counter()
    for index, event in enumerate(events.events):
        if event.kind != 1:
            continue
        start, end = event.text_position - 1, event.text_position - 1 + event.length
        if start < 0 or event.length == 0 or end > len(events.input_text):
            status, span, overlapping, exact = "invalid_or_out_of_input_span", None, (), ()
        else:
            span = (start, end)
            overlapping = tuple(w.word_id for w in words.words if w.start_codepoint < end and w.end_codepoint > start)
            exact = tuple(w.word_id for w in words.words if (w.start_codepoint, w.end_codepoint) == span)
            status = "exact_lexical_span" if exact else ("partial_or_multiword_span" if overlapping else "nonlexical_span")
        counts.update(exact)
        candidates.append(WordSpanCandidate(index, event.text_position, event.length, status, span, overlapping, exact))
    return LexicalEventSpans(words.trace_sha256, words.sha256, events.sha256, tuple(candidates),
        tuple(w.word_id for w in words.words if not counts[w.word_id]),
        tuple(w.word_id for w in words.words if counts[w.word_id] > 1))
