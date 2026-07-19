"""Long-form planning helpers for Scylla's Band synthesis."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Mapping

from .runtime import SynthesisRequest


DEFAULT_LONG_FORM_CHUNK_MAX_CHARS = 220
DEFAULT_LONG_FORM_CHUNK_MIN_CHARS = 48
DEFAULT_LONG_FORM_BOUNDARY_FADE_MS = 8.0
DEFAULT_MIN_SENTENCE_PUNCTUATION_PAUSE_MS = 320.0
DEFAULT_MIN_CLAUSE_PUNCTUATION_PAUSE_MS = 160.0
DEFAULT_ADAPTIVE_CHUNK_SCHEDULE = (120, 160, 220)
DEFAULT_ADAPTIVE_CHUNK_MIN_CHARS = 32
DEFAULT_ADAPTIVE_BUFFER_MS = 6000
DEFAULT_ADAPTIVE_REALTIME_FACTOR = 1.15
DEFAULT_ADAPTIVE_MIN_CHUNKS_PER_STAGE = 2
DEFAULT_LOOKAHEAD_CHUNKS = 1

PLAN_VERSION = "scyllasband.streaming.plan.v1"

_SENTENCE_DOT_PLACEHOLDER = "<SCYLLASBAND_DOT>"
_DECIMAL_DOT_RE = re.compile(r"(?<=\d)\.(?=\d)")
_DOTTED_INITIALISM_RE = re.compile(r"(?<![A-Za-z])(?:[A-Za-z]\.){2,}")
_CONTEXT_TEXT_CHARS = 220
_TERMINAL_BOUNDARY_CHARS = frozenset(".?!")
_STRONG_CONTINUATION_CHARS = frozenset(";:")
_SENTENCE_END_BOUNDARIES = {"sentence_end", "paragraph_end"}
_OVERLONG_LATENT_RE = re.compile(r"Predicted\s+(\d+)\s+latent frames.*supports at most\s+(\d+)")
_OVERLONG_PHONE_RE = re.compile(r"Phone sequence has\s+(\d+)\s+phones.*supports at most\s+(\d+)")
_OVERLONG_G2P_RE = re.compile(r"G2P input encodes to\s+(\d+)\s+tokens.*supports at most\s+(\d+)")


@dataclass(frozen=True)
class PlannerOptions:
    max_chunk_chars: int = DEFAULT_LONG_FORM_CHUNK_MAX_CHARS
    min_chunk_chars: int = DEFAULT_LONG_FORM_CHUNK_MIN_CHARS
    no_auto_split_overlong: bool = False
    no_preflight_chunks: bool = False
    no_normalize_text: bool = False
    seed: int | None = None
    steps: int = 8
    sampler: str = "heun"
    speed: float = 1.0
    backend: str = "litert"
    emotion: str | None = None
    affect: Mapping[str, float] | str | None = None
    affect_guidance_scale: float = 1.0
    emotion_guidance: str | None = None
    guidance_keep_reference: bool = False
    emotion_embed_scale: float = 1.0
    pause_ms: int = 0
    continuation_pause_ms: int = 0
    boundary_fade_ms: float = DEFAULT_LONG_FORM_BOUNDARY_FADE_MS
    min_sentence_pause_ms: float = DEFAULT_MIN_SENTENCE_PUNCTUATION_PAUSE_MS
    min_clause_pause_ms: float = DEFAULT_MIN_CLAUSE_PUNCTUATION_PAUSE_MS
    adaptive_chunking: bool = False
    adaptive_chunk_schedule: tuple[int, ...] = DEFAULT_ADAPTIVE_CHUNK_SCHEDULE
    adaptive_min_chunk_chars: int = DEFAULT_ADAPTIVE_CHUNK_MIN_CHARS
    adaptive_buffer_ms: int = DEFAULT_ADAPTIVE_BUFFER_MS
    adaptive_realtime_factor: float = DEFAULT_ADAPTIVE_REALTIME_FACTOR
    adaptive_min_chunks_per_stage: int = DEFAULT_ADAPTIVE_MIN_CHUNKS_PER_STAGE
    adaptive_preflight_chunks: bool = False
    lookahead_chunks: int = DEFAULT_LOOKAHEAD_CHUNKS
    lazy_preflight: bool = True
    pipeline_preflight: bool = True


@dataclass(frozen=True)
class PlanRecord:
    record_id: str
    voice: str
    language: str
    emotion: str | None
    emotion_guidance: str | None
    text: str
    normalized_text: str
    start_char: int = 0
    end_char: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "voice": self.voice,
            "language": self.language,
            "emotion": self.emotion,
            "emotion_guidance": self.emotion_guidance,
            "text": self.text,
            "normalized_text": self.normalized_text,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


@dataclass(frozen=True)
class PlanChunk:
    chunk_id: str
    record_id: str
    chain_id: str
    order_index: int
    text: str
    voice: str
    language: str
    emotion: str | None
    emotion_guidance: str | None
    context_before: str | None
    context_after: str | None
    boundary_before: str
    boundary_after: str
    starts_sentence: bool
    ends_sentence: bool
    g2p_token_count: int | None = None
    phone_count: int | None = None
    predicted_latent_frames: int | None = None
    fixed_latent_frames: int | None = None
    target_duration_seconds: float | None = None
    split_reason: str | None = None
    reference_key: str | None = None
    prefix_policy: str = "previous_chunk"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "record_id": self.record_id,
            "chain_id": self.chain_id,
            "order_index": self.order_index,
            "text": self.text,
            "voice": self.voice,
            "language": self.language,
            "emotion": self.emotion,
            "emotion_guidance": self.emotion_guidance,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "boundary_before": self.boundary_before,
            "boundary_after": self.boundary_after,
            "starts_sentence": self.starts_sentence,
            "ends_sentence": self.ends_sentence,
            "g2p_token_count": self.g2p_token_count,
            "phone_count": self.phone_count,
            "predicted_latent_frames": self.predicted_latent_frames,
            "fixed_latent_frames": self.fixed_latent_frames,
            "target_duration_seconds": self.target_duration_seconds,
            "split_reason": self.split_reason,
            "reference_key": self.reference_key,
            "prefix_policy": self.prefix_policy,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SynthesisPlan:
    source_kind: str
    normalized_text: str
    sample_rate: int | None
    backend: str
    fixed_shape_budget: dict[str, int]
    records: list[PlanRecord]
    chunks: list[PlanChunk]
    scheduler_hints: dict[str, Any] = field(default_factory=dict)
    version: str = PLAN_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source_kind": self.source_kind,
            "normalized_text": self.normalized_text,
            "sample_rate": self.sample_rate,
            "backend": self.backend,
            "fixed_shape_budget": dict(self.fixed_shape_budget),
            "records": [record.to_dict() for record in self.records],
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "scheduler_hints": dict(self.scheduler_hints),
        }


def planner_options_from(options: object | None = None, **overrides: Any) -> PlannerOptions:
    values: dict[str, Any] = {}
    defaults = PlannerOptions()
    for name in defaults.__dataclass_fields__:
        if options is None:
            values[name] = getattr(defaults, name)
        elif isinstance(options, PlannerOptions):
            values[name] = getattr(options, name)
        elif isinstance(options, Mapping):
            values[name] = options.get(name, getattr(defaults, name))
        else:
            values[name] = getattr(options, name, getattr(defaults, name))
    for name, value in overrides.items():
        if value is not None and name in values:
            values[name] = value
    values["adaptive_chunk_schedule"] = _coerce_positive_int_tuple(
        values.get("adaptive_chunk_schedule"),
        fallback=DEFAULT_ADAPTIVE_CHUNK_SCHEDULE,
    )
    values["adaptive_min_chunk_chars"] = max(0, int(values.get("adaptive_min_chunk_chars") or 0))
    values["adaptive_buffer_ms"] = max(0, int(values.get("adaptive_buffer_ms") or 0))
    values["adaptive_realtime_factor"] = max(0.0, float(values.get("adaptive_realtime_factor") or 0.0))
    values["adaptive_min_chunks_per_stage"] = max(1, int(values.get("adaptive_min_chunks_per_stage") or 1))
    values["lookahead_chunks"] = max(0, int(values.get("lookahead_chunks") or 0))
    return PlannerOptions(**values)


def _coerce_positive_int_tuple(value: object, *, fallback: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return tuple(fallback)
    if isinstance(value, str):
        items: list[object] = [item.strip() for item in value.split(",")]
    elif isinstance(value, int):
        items = [value]
    else:
        try:
            items = list(value)  # type: ignore[arg-type]
        except TypeError:
            items = []
    out: list[int] = []
    for item in items:
        if item in (None, ""):
            continue
        try:
            number = int(item)
        except (TypeError, ValueError):
            continue
        if number > 0 and number not in out:
            out.append(number)
    return tuple(out or fallback)


def plan_text(
    runtime: Any,
    text: str,
    *,
    voice: str,
    language: str | None = None,
    emotion: str | None = None,
    affect: Mapping[str, float] | str | None = None,
    affect_guidance_scale: float | None = None,
    emotion_guidance: str | None = None,
    options: object | None = None,
    **overrides: Any,
) -> SynthesisPlan:
    return plan_records(
        runtime,
        [
            {
                "voice": voice,
                "language": language,
                "emotion": emotion,
                "affect": affect,
                "affect_guidance_scale": affect_guidance_scale,
                "emotion_guidance": emotion_guidance,
                "text": text,
            }
        ],
        options=options,
        source_kind="text",
        **overrides,
    )


def plan_records(
    runtime: Any,
    records: list[dict[str, Any]],
    *,
    options: object | None = None,
    source_kind: str = "records",
    **overrides: Any,
) -> SynthesisPlan:
    opts = planner_options_from(options, **overrides)
    prepared = prepare_render_chunks(runtime, records, opts)
    return synthesis_plan_from_prepared(
        runtime,
        records,
        prepared,
        opts,
        source_kind=source_kind,
    )


def synthesis_plan_from_prepared(
    runtime: Any,
    records: list[dict[str, Any]],
    prepared: list[dict[str, Any]],
    options: object | None,
    *,
    source_kind: str = "records",
) -> SynthesisPlan:
    opts = planner_options_from(options)
    plan_records_out = _plan_records_from_input(runtime, records, opts)
    chunks = plan_chunks_from_prepared(runtime, prepared, opts)
    return SynthesisPlan(
        source_kind=source_kind,
        normalized_text="\n\n".join(record.normalized_text for record in plan_records_out if record.normalized_text),
        sample_rate=_runtime_sample_rate(runtime),
        backend=str(opts.backend),
        fixed_shape_budget=_fixed_shape_budget(runtime),
        records=plan_records_out,
        chunks=chunks,
        scheduler_hints={
            "serial_within_chain": True,
            "can_preflight_ahead": not bool(opts.no_preflight_chunks),
            "lookahead_chunks": int(opts.lookahead_chunks),
            "parallelizable_chains": sorted({chunk.chain_id for chunk in chunks}),
        },
    )


def plan_chunks_from_prepared(
    runtime: Any,
    chunks: list[dict[str, Any]],
    options: object | None = None,
) -> list[PlanChunk]:
    opts = planner_options_from(options)
    out: list[PlanChunk] = []
    chain_keys = [_chain_key(chunk, opts) for chunk in chunks]
    last_chain_key: tuple[Any, ...] | None = None
    chain_serial = -1
    for index, chunk in enumerate(chunks):
        chain_key = chain_keys[index]
        if chain_key != last_chain_key:
            chain_serial += 1
            last_chain_key = chain_key
        chain_id = _chain_id(chain_serial)
        metadata = dict(chunk)
        text = str(chunk.get("text") or "")
        predicted = _int_or_none(chunk.get("preflight_predicted_latent_frames"))
        fixed = _int_or_none(chunk.get("preflight_fixed_latent_frames"))
        out.append(
            PlanChunk(
                chunk_id=f"chunk-{index:04d}",
                record_id=str(chunk.get("record_id") or "record-0000"),
                chain_id=chain_id,
                order_index=index,
                text=text,
                voice=str(chunk.get("voice") or ""),
                language=str(chunk.get("language") or ""),
                emotion=_optional_text(chunk.get("emotion")),
                emotion_guidance=_optional_text(chunk.get("emotion_guidance")),
                context_before=_chain_context_before(chunks, chain_keys, index),
                context_after=_chain_context_after(
                    chunks,
                    chain_keys,
                    index,
                    lookahead_chunks=opts.lookahead_chunks,
                ),
                boundary_before=str(chunk.get("boundary_before") or ""),
                boundary_after=str(chunk.get("boundary_after") or ""),
                starts_sentence=bool(chunk.get("starts_sentence", False)),
                ends_sentence=bool(chunk.get("ends_sentence", False)),
                g2p_token_count=_int_or_none(chunk.get("g2p_retry_encoded_tokens")),
                phone_count=_int_or_none(chunk.get("phone_retry_phone_count")),
                predicted_latent_frames=predicted,
                fixed_latent_frames=fixed,
                target_duration_seconds=_target_duration_seconds(runtime, predicted),
                split_reason=_optional_text(chunk.get("split_reason")),
                reference_key=_reference_key(chunk),
                prefix_policy="none" if index == 0 or chain_id != (out[-1].chain_id if out else None) else "previous_chunk",
                metadata=metadata,
            )
        )
    return out


def prepare_render_chunks(
    runtime: Any,
    records: list[dict[str, Any]],
    options: object | None,
) -> list[dict[str, Any]]:
    opts = planner_options_from(options)
    prepared: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        voice = str(record["voice"])
        language = runtime.resolve_language_for_voice(voice, record.get("language"))
        emotion = _optional_text(record.get("emotion")) or opts.emotion
        emotion_guidance = _optional_text(record.get("emotion_guidance")) or opts.emotion_guidance
        source_text = str(record["text"])
        record_id = str(record.get("record_id") or f"record-{record_index:04d}")
        if opts.no_normalize_text:
            split_records = split_text_chunk_records(
                source_text,
                max_chars=opts.max_chunk_chars,
                min_chars=opts.min_chunk_chars,
            )
            for split_record in split_records:
                prepared.append(
                    {
                        **split_record,
                        "record_id": record_id,
                        "voice": voice,
                        "language": language,
                        "emotion": emotion,
                        "emotion_guidance": emotion_guidance,
                        "source_text": source_text,
                        "pre_normalized": False,
                        "normalize_text": False,
                    }
                )
            continue
        normalized = runtime.normalize_text(source_text, language=language, voice_id=voice)
        for split_record in split_text_chunk_records(
            normalized,
            max_chars=opts.max_chunk_chars,
            min_chars=opts.min_chunk_chars,
        ):
            prepared.append(
                {
                    **split_record,
                    "record_id": record_id,
                    "voice": voice,
                    "language": language,
                    "emotion": emotion,
                    "emotion_guidance": emotion_guidance,
                    "source_text": source_text,
                    "pre_normalized": True,
                    "normalize_text": False,
                }
            )
    if opts.no_preflight_chunks:
        return prepared
    return preflight_split_overlong_chunks(runtime, prepared, opts)


def preflight_split_overlong_chunks(
    runtime: Any,
    chunks: list[dict[str, Any]],
    options: object | None,
) -> list[dict[str, Any]]:
    opts = planner_options_from(options)
    output: list[dict[str, Any]] = []
    queue = [dict(chunk) for chunk in chunks]
    split_budget = max(1, len(queue) * 16)
    while queue:
        chunk = queue.pop(0)
        try:
            metadata = estimate_chunk_duration_metadata(runtime, chunk, opts)
        except (ValueError, RuntimeError) as exc:
            budget_name, replacements = (
                (None, [])
                if opts.no_auto_split_overlong
                else split_chunk_for_overlong_retry(
                    exc,
                    chunk,
                    max_chars=opts.max_chunk_chars,
                    min_chars=opts.min_chunk_chars,
                )
            )
            if replacements and split_budget > 0:
                split_budget -= 1
                queue = [dict(item) for item in replacements] + queue
                continue
            raise
        chunk["preflight_predicted_latent_frames"] = int(metadata.get("predicted_latent_frames", 0))
        chunk["preflight_fixed_latent_frames"] = int(metadata.get("fixed_latent_frames", 0))
        predicted = int(metadata.get("predicted_latent_frames", 0))
        fixed = int(metadata.get("fixed_latent_frames", 0))
        if fixed <= 0 or predicted <= fixed:
            output.append(chunk)
            continue
        replacements = split_chunk_for_latent_retry(
            chunk,
            max_chars=opts.max_chunk_chars,
            min_chars=opts.min_chunk_chars,
            predicted_latent_frames=predicted,
            fixed_latent_frames=fixed,
        )
        if not replacements or split_budget <= 0:
            output.append(chunk)
            continue
        split_budget -= 1
        queue = [dict(item) for item in replacements] + queue
    return output


def estimate_chunk_duration_metadata(
    runtime: Any,
    chunk: dict[str, Any],
    options: object | None,
) -> dict[str, Any]:
    opts = planner_options_from(options)
    return runtime.estimate_latent_frames(
        SynthesisRequest(
            text=str(chunk["text"]),
            voice_id=str(chunk["voice"]),
            language=str(chunk["language"]),
            emotion=_optional_text(chunk.get("emotion")) or opts.emotion,
            affect=chunk.get("affect") if chunk.get("affect") is not None else opts.affect,
            affect_guidance_scale=float(
                chunk.get("affect_guidance_scale")
                if chunk.get("affect_guidance_scale") is not None
                else opts.affect_guidance_scale
            ),
            emotion_guidance=_optional_text(chunk.get("emotion_guidance")) or opts.emotion_guidance,
            guidance_null_reference=not bool(opts.guidance_keep_reference),
            emotion_embed_scale=float(opts.emotion_embed_scale),
            normalize_text=bool(chunk.get("normalize_text", not opts.no_normalize_text)),
            seed=opts.seed,
            steps=opts.steps,
            sampler=opts.sampler,
            speed=opts.speed,
            context_before=None,
            context_after=None,
            boundary_before=str(chunk["boundary_before"]),
            boundary_after=str(chunk["boundary_after"]),
            min_sentence_pause_ms=float(opts.min_sentence_pause_ms),
            min_clause_pause_ms=float(opts.min_clause_pause_ms),
        )
    )


def split_text_chunks(text: str, *, max_chars: int, min_chars: int = 0) -> list[str]:
    return [
        str(item["text"])
        for item in split_text_chunk_records(
            text,
            max_chars=max_chars,
            min_chars=min_chars,
        )
    ]


def split_text_chunk_records(
    text: str,
    *,
    max_chars: int,
    min_chars: int = 0,
) -> list[dict[str, Any]]:
    if max_chars <= 0:
        raise ValueError("--max-chunk-chars must be positive")
    min_chars = max(0, min(int(min_chars), max_chars))
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n+", str(text or "")) if item.strip()]
    records: list[dict[str, Any]] = []
    previous_after: str | None = None
    for paragraph_index, paragraph in enumerate(paragraphs):
        units = _sentence_units(paragraph)
        for unit_index, unit in enumerate(units):
            pieces = _split_oversized_unit(unit, max_chars=max_chars, min_chars=min_chars)
            for piece_index, piece in enumerate(pieces):
                is_last_piece = piece_index == len(pieces) - 1
                boundary_before = _boundary_before(
                    previous_after,
                    paragraph_start=unit_index == 0 and piece_index == 0,
                )
                boundary_after = _boundary_after(
                    piece,
                    final_piece=is_last_piece,
                    paragraph_end=unit_index == len(units) - 1,
                )
                split_reason = "unit" if len(pieces) == 1 else ("unit_tail" if is_last_piece else "budget")
                records.append(
                    {
                        "text": piece,
                        "source_text": text,
                        "paragraph_index": paragraph_index,
                        "unit_index": unit_index,
                        "piece_index": piece_index,
                        "piece_count": len(pieces),
                        "split_reason": split_reason,
                        "boundary_before": boundary_before,
                        "boundary_after": boundary_after,
                        "starts_sentence": boundary_before in {"paragraph_start", "sentence_start"},
                        "ends_sentence": boundary_after in _SENTENCE_END_BOUNDARIES,
                    }
                )
                previous_after = boundary_after
    records = _merge_adjacent_chunk_records(records, max_chars=max_chars)
    return _merge_short_chunk_records(records, max_chars=max_chars, min_chars=min_chars)


def pause_after_ms(chunk: dict[str, Any], *, options: object | None, is_last: bool) -> int:
    if is_last:
        return 0
    opts = planner_options_from(options)
    boundary_after = str(chunk.get("boundary_after") or "")
    if boundary_after in _SENTENCE_END_BOUNDARIES:
        return max(0, int(opts.pause_ms))
    if boundary_after == "chunk_continue":
        return 0
    return max(0, int(opts.continuation_pause_ms))


def chunk_context_before(chunks: list[dict[str, Any]], index: int) -> str | None:
    if index <= 0:
        return None
    value = str(chunks[index - 1].get("text") or "")
    return value[-_CONTEXT_TEXT_CHARS:] or None


def chunk_context_after(
    chunks: list[dict[str, Any]],
    index: int,
    *,
    lookahead_chunks: int = DEFAULT_LOOKAHEAD_CHUNKS,
) -> str | None:
    if index >= len(chunks) - 1 or int(lookahead_chunks) <= 0:
        return None
    end = min(len(chunks), index + 1 + int(lookahead_chunks))
    value = " ".join(str(chunks[item].get("text") or "") for item in range(index + 1, end)).strip()
    return value[:_CONTEXT_TEXT_CHARS] or None


def is_overlong_latent_error(exc: Exception) -> bool:
    return overlong_latent_counts(exc) != (None, None)


def overlong_latent_counts(exc: Exception) -> tuple[int | None, int | None]:
    match = _OVERLONG_LATENT_RE.search(str(exc))
    if match is None:
        return None, None
    return int(match.group(1)), int(match.group(2))


def is_overlong_phone_error(exc: Exception) -> bool:
    return overlong_phone_counts(exc) != (None, None)


def overlong_phone_counts(exc: Exception) -> tuple[int | None, int | None]:
    match = _OVERLONG_PHONE_RE.search(str(exc))
    if match is None:
        return None, None
    return int(match.group(1)), int(match.group(2))


def is_overlong_g2p_error(exc: Exception) -> bool:
    return overlong_g2p_counts(exc) != (None, None)


def overlong_g2p_counts(exc: Exception) -> tuple[int | None, int | None]:
    match = _OVERLONG_G2P_RE.search(str(exc))
    if match is None:
        return None, None
    return int(match.group(1)), int(match.group(2))


def split_chunk_for_overlong_retry(
    exc: Exception,
    chunk: dict[str, Any],
    *,
    max_chars: int,
    min_chars: int = 0,
) -> tuple[str | None, list[dict[str, Any]]]:
    latent_predicted, latent_fixed = overlong_latent_counts(exc)
    if latent_predicted is not None and latent_fixed is not None:
        return "latent", split_chunk_for_latent_retry(
            chunk,
            max_chars=max_chars,
            min_chars=min_chars,
            predicted_latent_frames=latent_predicted,
            fixed_latent_frames=latent_fixed,
        )
    phone_count, phone_fixed = overlong_phone_counts(exc)
    if phone_count is not None and phone_fixed is not None:
        return "phone", split_chunk_for_phone_retry(
            chunk,
            max_chars=max_chars,
            min_chars=min_chars,
            phone_count=phone_count,
            fixed_frames=phone_fixed,
        )
    g2p_count, g2p_fixed = overlong_g2p_counts(exc)
    if g2p_count is not None and g2p_fixed is not None:
        return "G2P", split_chunk_for_g2p_retry(
            chunk,
            max_chars=max_chars,
            min_chars=min_chars,
            encoded_tokens=g2p_count,
            fixed_tokens=g2p_fixed,
        )
    return None, []


def split_chunk_for_latent_retry(
    chunk: dict[str, Any],
    *,
    max_chars: int,
    min_chars: int = 0,
    predicted_latent_frames: int | None = None,
    fixed_latent_frames: int | None = None,
) -> list[dict[str, Any]]:
    return _split_chunk_for_budget_retry(
        chunk,
        max_chars=max_chars,
        min_chars=min_chars,
        predicted_count=predicted_latent_frames,
        fixed_count=fixed_latent_frames,
        reason_prefix="latent_budget",
        predicted_key="latent_retry_predicted_frames",
        fixed_key="latent_retry_fixed_frames",
    )


def split_chunk_for_phone_retry(
    chunk: dict[str, Any],
    *,
    max_chars: int,
    min_chars: int = 0,
    phone_count: int | None = None,
    fixed_frames: int | None = None,
) -> list[dict[str, Any]]:
    return _split_chunk_for_budget_retry(
        chunk,
        max_chars=max_chars,
        min_chars=min_chars,
        predicted_count=phone_count,
        fixed_count=fixed_frames,
        reason_prefix="phone_budget",
        predicted_key="phone_retry_phone_count",
        fixed_key="phone_retry_fixed_frames",
    )


def split_chunk_for_g2p_retry(
    chunk: dict[str, Any],
    *,
    max_chars: int,
    min_chars: int = 0,
    encoded_tokens: int | None = None,
    fixed_tokens: int | None = None,
) -> list[dict[str, Any]]:
    return _split_chunk_for_budget_retry(
        chunk,
        max_chars=max_chars,
        min_chars=min_chars,
        predicted_count=encoded_tokens,
        fixed_count=fixed_tokens,
        reason_prefix="g2p_budget",
        predicted_key="g2p_retry_encoded_tokens",
        fixed_key="g2p_retry_fixed_tokens",
    )


def _plan_records_from_input(runtime: Any, records: list[dict[str, Any]], opts: PlannerOptions) -> list[PlanRecord]:
    out: list[PlanRecord] = []
    for index, record in enumerate(records):
        voice = str(record["voice"])
        language = runtime.resolve_language_for_voice(voice, record.get("language"))
        text = str(record["text"])
        normalized = text if opts.no_normalize_text else runtime.normalize_text(text, language=language, voice_id=voice)
        out.append(
            PlanRecord(
                record_id=str(record.get("record_id") or f"record-{index:04d}"),
                voice=voice,
                language=language,
                emotion=_optional_text(record.get("emotion")) or opts.emotion,
                emotion_guidance=_optional_text(record.get("emotion_guidance")) or opts.emotion_guidance,
                text=text,
                normalized_text=normalized,
                start_char=0,
                end_char=len(text),
            )
        )
    return out


def _merge_adjacent_chunk_records(records: list[dict[str, Any]], *, max_chars: int) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for record in records:
        if merged and _can_merge_chunk_records(merged[-1], record, max_chars=max_chars):
            previous = merged[-1]
            previous["text"] = f"{previous['text']} {record['text']}"
            previous["boundary_after"] = record["boundary_after"]
            previous["ends_sentence"] = record["ends_sentence"]
            previous["split_reason"] = "merged"
            continue
        merged.append(dict(record))
    return merged


def _can_merge_chunk_records(
    previous: dict[str, Any],
    current: dict[str, Any],
    *,
    max_chars: int,
) -> bool:
    if previous.get("paragraph_index") != current.get("paragraph_index"):
        return False
    if previous.get("boundary_after") == "chunk_continue":
        return False
    if current.get("boundary_before") == "chunk_continue":
        return False
    candidate = f"{previous.get('text', '')} {current.get('text', '')}"
    return len(candidate) <= max_chars


def _merge_short_chunk_records(
    records: list[dict[str, Any]],
    *,
    max_chars: int,
    min_chars: int,
) -> list[dict[str, Any]]:
    if min_chars <= 0 or len(records) <= 1:
        return records
    output = [dict(record) for record in records]
    index = 0
    while index < len(output):
        text_value = str(output[index].get("text") or "")
        if len(text_value) >= min_chars:
            index += 1
            continue
        merged = False
        if index > 0 and output[index - 1].get("paragraph_index") == output[index].get("paragraph_index"):
            candidate = f"{output[index - 1].get('text', '')} {text_value}".strip()
            if len(candidate) <= max_chars or index == len(output) - 1:
                previous = output[index - 1]
                previous["text"] = candidate
                previous["boundary_after"] = output[index].get("boundary_after")
                previous["ends_sentence"] = output[index].get("ends_sentence")
                previous["split_reason"] = "min_chunk_merged"
                output.pop(index)
                merged = True
        if merged:
            continue
        if index + 1 < len(output) and output[index + 1].get("paragraph_index") == output[index].get("paragraph_index"):
            candidate = f"{text_value} {output[index + 1].get('text', '')}".strip()
            if len(candidate) <= max_chars:
                current = output[index]
                current["text"] = candidate
                current["boundary_after"] = output[index + 1].get("boundary_after")
                current["ends_sentence"] = output[index + 1].get("ends_sentence")
                current["split_reason"] = "min_chunk_merged"
                output.pop(index + 1)
                continue
        index += 1
    return output


def _sentence_units(text: str) -> list[str]:
    value = _protect_sentence_breaks(str(text or ""))
    units = re.findall(r"[^.!?;:\n]+(?:[.!?;:]+[\"']?)?", value)
    return [_restore_sentence_breaks(item.strip()) for item in units if item.strip()]


def _protect_sentence_breaks(text: str) -> str:
    def protect_initialism(match: re.Match[str]) -> str:
        return match.group(0).replace(".", _SENTENCE_DOT_PLACEHOLDER)

    value = _DOTTED_INITIALISM_RE.sub(protect_initialism, text)
    return _DECIMAL_DOT_RE.sub(_SENTENCE_DOT_PLACEHOLDER, value)


def _restore_sentence_breaks(text: str) -> str:
    return text.replace(_SENTENCE_DOT_PLACEHOLDER, ".")


def _split_oversized_unit(text: str, *, max_chars: int, min_chars: int = 0) -> list[str]:
    value = str(text or "").strip()
    if len(value) <= max_chars:
        return [value] if value else []
    words = value.split()
    chunks: list[str] = []
    current = ""
    for word in words:
        pieces = [word[index : index + max_chars] for index in range(0, len(word), max_chars)] or [word]
        for piece in pieces:
            if not current:
                current = piece
                continue
            candidate = f"{current} {piece}"
            if len(candidate) <= max_chars:
                current = candidate
            else:
                chunks.append(current)
                current = piece
    if current:
        chunks.append(current)
    return _rebalance_short_text_pieces(chunks, max_chars=max_chars, min_chars=min_chars)


def _rebalance_short_text_pieces(
    chunks: list[str],
    *,
    max_chars: int,
    min_chars: int,
) -> list[str]:
    min_chars = max(0, min(int(min_chars), int(max_chars)))
    if min_chars <= 0 or len(chunks) <= 1:
        return chunks
    output = [chunk for chunk in chunks if chunk]
    if len(output) <= 1:
        return output
    last = output[-1]
    if len(last) >= min_chars:
        return output
    previous_words = output[-2].split()
    last_words = last.split()
    while len(" ".join(last_words)) < min_chars and len(previous_words) > 1:
        last_words.insert(0, previous_words.pop())
    if previous_words and len(" ".join(last_words)) >= min_chars:
        output[-2] = " ".join(previous_words)
        output[-1] = " ".join(last_words)
        return output
    output[-2] = f"{output[-2]} {last}".strip()
    output.pop()
    return output


def _boundary_before(previous_after: str | None, *, paragraph_start: bool) -> str:
    if previous_after is None:
        return "paragraph_start"
    if paragraph_start:
        return "paragraph_start"
    if previous_after == "paragraph_end":
        return "paragraph_start"
    if previous_after == "sentence_end":
        return "sentence_start"
    if previous_after == "chunk_continue":
        return "chunk_continue"
    return "clause_continue"


def _boundary_after(text: str, *, final_piece: bool, paragraph_end: bool) -> str:
    value = _strip_closing_boundary(str(text or ""))
    if not final_piece:
        return "chunk_continue"
    if value and value[-1] in _TERMINAL_BOUNDARY_CHARS:
        return "paragraph_end" if paragraph_end else "sentence_end"
    if value and value[-1] in _STRONG_CONTINUATION_CHARS:
        return "clause_continue"
    return "paragraph_end" if paragraph_end else "clause_continue"


def _strip_closing_boundary(text: str) -> str:
    value = text.rstrip()
    while value and value[-1] in "\"')]}»”’":
        value = value[:-1].rstrip()
    return value


def _split_chunk_for_budget_retry(
    chunk: dict[str, Any],
    *,
    max_chars: int,
    min_chars: int = 0,
    predicted_count: int | None = None,
    fixed_count: int | None = None,
    reason_prefix: str,
    predicted_key: str,
    fixed_key: str,
) -> list[dict[str, Any]]:
    text = str(chunk.get("text") or "").strip()
    if not text:
        return []
    retry_max_chars = _retry_split_max_chars(
        text,
        max_chars=max_chars,
        min_chars=min_chars,
        predicted_latent_frames=predicted_count,
        fixed_latent_frames=fixed_count,
    )
    pieces = _split_units_min_max(
        text,
        max_chars=retry_max_chars,
        min_chars=min_chars,
    )
    if not pieces and min_chars > 0:
        # Preserve sentence boundaries even when that leaves a short tail.
        pieces = _split_units_min_max(
            text,
            max_chars=retry_max_chars,
            min_chars=0,
        )
    if not pieces:
        pieces = _split_words_min_max(
            text,
            max_chars=retry_max_chars,
            min_chars=min_chars,
        )
    if not pieces:
        pieces = _split_oversized_unit(
            text,
            max_chars=retry_max_chars,
            min_chars=min_chars,
        )
    if not _retry_pieces_are_usable(
        pieces,
        max_chars=retry_max_chars,
        min_chars=0,
    ):
        pieces = _split_words_near_half(
            text,
            max_chars=retry_max_chars,
            min_chars=min_chars,
        )
    if not _retry_pieces_are_usable(
        pieces,
        max_chars=retry_max_chars,
        min_chars=0,
    ):
        return []

    out: list[dict[str, Any]] = []
    previous_after: str | None = None
    for piece_index, piece in enumerate(pieces):
        first = piece_index == 0
        last = piece_index == len(pieces) - 1
        record = dict(chunk)
        record["text"] = piece
        record["piece_index"] = piece_index
        record["piece_count"] = len(pieces)
        record["split_reason"] = f"{reason_prefix}_tail" if last else reason_prefix
        record["boundary_before"] = (
            chunk.get("boundary_before")
            if first
            else _boundary_before(previous_after, paragraph_start=False)
        )
        record["boundary_after"] = (
            chunk.get("boundary_after")
            if last
            else _retry_piece_boundary_after(piece)
        )
        record["starts_sentence"] = (
            bool(chunk.get("starts_sentence"))
            if first
            else record["boundary_before"] in {"paragraph_start", "sentence_start"}
        )
        record["ends_sentence"] = record["boundary_after"] in _SENTENCE_END_BOUNDARIES
        if predicted_count is not None:
            record[predicted_key] = int(predicted_count)
        if fixed_count is not None:
            record[fixed_key] = int(fixed_count)
        out.append(record)
        previous_after = str(record["boundary_after"])
    return out


def _retry_piece_boundary_after(text: str) -> str:
    value = _strip_closing_boundary(text)
    if value and value[-1] in _TERMINAL_BOUNDARY_CHARS:
        return "sentence_end"
    if value and value[-1] in _STRONG_CONTINUATION_CHARS:
        return "clause_continue"
    return "chunk_continue"


def _split_units_min_max(text: str, *, max_chars: int, min_chars: int) -> list[str]:
    units = _sentence_units(str(text or ""))
    if len(units) <= 1:
        return []
    max_chars = max(1, int(max_chars))
    pieces: list[str] = []
    current = ""
    for unit in units:
        candidates = [unit] if len(unit) <= max_chars else _split_oversized_unit(
            unit,
            max_chars=max_chars,
            min_chars=min_chars,
        )
        for candidate_piece in candidates:
            if not candidate_piece:
                continue
            if not current:
                current = candidate_piece
                continue
            candidate = f"{current} {candidate_piece}"
            if len(candidate) <= max_chars:
                current = candidate
            else:
                pieces.append(current)
                current = candidate_piece
    if current:
        pieces.append(current)
    return pieces if _retry_pieces_are_usable(pieces, max_chars=max_chars, min_chars=min_chars) else []


def _path_balance_score(pieces: list[str]) -> tuple[int, int]:
    lengths = [len(piece) for piece in pieces]
    if not lengths:
        return (0, 0)
    return (max(lengths) - min(lengths), max(lengths))


def _retry_split_max_chars(
    text: str,
    *,
    max_chars: int,
    min_chars: int,
    predicted_latent_frames: int | None = None,
    fixed_latent_frames: int | None = None,
) -> int:
    requested_max = max(1, int(max_chars))
    requested_min = max(0, min(int(min_chars), requested_max))
    predicted = int(predicted_latent_frames or 0)
    fixed = int(fixed_latent_frames or 0)
    if predicted > 0 and fixed > 0 and predicted > fixed:
        text_len = len(str(text or "").strip())
        scaled = int(max(1, text_len) * float(fixed) / float(predicted) * 0.92)
        requested_max = min(requested_max, max(requested_min or 1, scaled))
    return max(requested_min or 1, requested_max)


def _split_words_min_max(text: str, *, max_chars: int, min_chars: int) -> list[str]:
    words = str(text or "").split()
    if len(words) <= 1:
        return []
    max_chars = max(1, int(max_chars))
    min_chars = max(0, min(int(min_chars), max_chars))
    n = len(words)
    prefix = [0]
    for word in words:
        prefix.append(prefix[-1] + len(word))

    def segment_len(start: int, end: int) -> int:
        return prefix[end] - prefix[start] + max(0, end - start - 1)

    paths: list[list[str] | None] = [None for _ in range(n + 1)]
    paths[0] = []
    for start in range(n):
        if paths[start] is None:
            continue
        for end in range(start + 1, n + 1):
            length = segment_len(start, end)
            if length > max_chars:
                break
            if min_chars > 0 and length < min_chars:
                continue
            segment = " ".join(words[start:end])
            candidate = [*paths[start], segment]
            existing = paths[end]
            if (
                existing is None
                or len(candidate) < len(existing)
                or (len(candidate) == len(existing) and _path_balance_score(candidate) < _path_balance_score(existing))
            ):
                paths[end] = candidate
    result = paths[n] or []
    return result if len(result) > 1 else []


def _retry_pieces_are_usable(
    pieces: list[str],
    *,
    max_chars: int,
    min_chars: int,
) -> bool:
    if len(pieces) <= 1:
        return False
    requested_min = max(0, min(int(min_chars), int(max_chars)))
    for piece in pieces:
        length = len(str(piece).strip())
        if length <= 0 or length > max_chars:
            return False
        if requested_min > 0 and length < requested_min:
            return False
    return True


def _split_words_near_half(text: str, *, max_chars: int, min_chars: int = 0) -> list[str]:
    words = text.split()
    if len(words) <= 1:
        return []
    midpoint = max(1, len(words) // 2)
    pieces = [" ".join(words[:midpoint]), " ".join(words[midpoint:])]
    if _retry_pieces_are_usable(pieces, max_chars=max_chars, min_chars=min_chars):
        return pieces
    return []


def _chain_key(chunk: dict[str, Any], opts: PlannerOptions) -> tuple[Any, ...]:
    return (
        str(chunk.get("record_id") or "record-0000"),
        str(chunk.get("voice") or ""),
        str(chunk.get("language") or ""),
        _optional_text(chunk.get("emotion")),
        _stable_affect_key(chunk.get("affect") if chunk.get("affect") is not None else opts.affect),
        float(
            chunk.get("affect_guidance_scale")
            if chunk.get("affect_guidance_scale") is not None
            else opts.affect_guidance_scale
        ),
        _optional_text(chunk.get("emotion_guidance")),
        float(opts.speed),
        int(opts.steps),
        str(opts.sampler),
        bool(opts.guidance_keep_reference),
        float(opts.emotion_embed_scale),
        _reference_key(chunk),
    )


def _stable_affect_key(value: object) -> object:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), float(item)) for key, item in value.items()))
    return _optional_text(value)


def _chain_context_before(
    chunks: list[dict[str, Any]],
    chain_keys: list[tuple[Any, ...]],
    index: int,
) -> str | None:
    if index <= 0 or chain_keys[index - 1] != chain_keys[index]:
        return None
    return chunk_context_before(chunks, index)


def _chain_context_after(
    chunks: list[dict[str, Any]],
    chain_keys: list[tuple[Any, ...]],
    index: int,
    *,
    lookahead_chunks: int,
) -> str | None:
    if index >= len(chunks) - 1 or int(lookahead_chunks) <= 0:
        return None
    current_key = chain_keys[index]
    following: list[str] = []
    end = min(len(chunks), index + 1 + int(lookahead_chunks))
    for item in range(index + 1, end):
        if chain_keys[item] != current_key:
            break
        following.append(str(chunks[item].get("text") or ""))
    value = " ".join(following).strip()
    return value[:_CONTEXT_TEXT_CHARS] or None


def _chain_id(serial: int) -> str:
    return f"chain-{serial:04d}"


def _reference_key(chunk: dict[str, Any]) -> str | None:
    for key in ("reference_key", "voice_pack", "voice_pack_id", "speaker_key"):
        value = _optional_text(chunk.get(key))
        if value:
            return value
    return str(chunk.get("voice") or "") or None


def _target_duration_seconds(runtime: Any, predicted_latent_frames: int | None) -> float | None:
    if predicted_latent_frames is None:
        return None
    manifest = getattr(runtime, "manifest", None)
    audio = getattr(manifest, "audio", None)
    sample_rate = getattr(audio, "sample_rate", None)
    latent_hop = getattr(audio, "latent_hop_length", None)
    if not sample_rate or not latent_hop:
        return None
    return float(predicted_latent_frames) * float(latent_hop) / float(sample_rate)


def _runtime_sample_rate(runtime: Any) -> int | None:
    manifest = getattr(runtime, "manifest", None)
    audio = getattr(manifest, "audio", None)
    sample_rate = getattr(audio, "sample_rate", None)
    return int(sample_rate) if sample_rate else None


def _fixed_shape_budget(runtime: Any) -> dict[str, int]:
    manifest = getattr(runtime, "manifest", None)
    controls = getattr(manifest, "controls", {}) or {}
    fixed = dict(controls.get("fixed_shapes", {}) or {})
    export_status = dict(controls.get("export_status", {}) or {})
    if not fixed:
        fixed = dict(export_status.get("fixed_shapes", {}) or {})
    return {str(key): int(value) for key, value in fixed.items() if _int_or_none(value) is not None}


def _int_or_none(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
