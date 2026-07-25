"""Evented long-form synthesis for Scylla's Band."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
import sys
import time
from typing import Any, Iterator, Mapping

from .planner import (
    PlanChunk,
    PlannerOptions,
    SynthesisPlan,
    pause_after_ms,
    plan_chunks_from_prepared,
    preflight_split_overlong_chunks,
    planner_options_from,
    prepare_render_chunks,
    split_chunk_for_overlong_retry,
    synthesis_plan_from_prepared,
)
from .runtime import SynthesisRequest


@dataclass(frozen=True)
class StreamingEvent:
    type: str
    plan: SynthesisPlan | None = None
    chunk: PlanChunk | None = None
    audio: Any | None = None
    sample_rate: int | None = None
    start_sample: int | None = None
    end_sample: int | None = None
    metadata: dict[str, Any] | None = None
    message: str | None = None

    def to_dict(self, *, include_audio: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "type": self.type,
            "sample_rate": self.sample_rate,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "metadata": dict(self.metadata or {}),
            "message": self.message,
        }
        if self.plan is not None:
            data["plan"] = self.plan.to_dict()
        if self.chunk is not None:
            data["chunk"] = self.chunk.to_dict()
        if include_audio:
            data["audio"] = self.audio
        return data


def synthesize_text_stream(
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
) -> Iterator[StreamingEvent]:
    yield from synthesize_records_stream(
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


def synthesize_records_stream(
    runtime: Any,
    records: list[dict[str, Any]],
    *,
    options: object | None = None,
    source_kind: str = "records",
    progress_stream: Any | None = None,
    **overrides: Any,
) -> Iterator[StreamingEvent]:
    opts = planner_options_from(options, **overrides)
    stream_started_at = time.perf_counter()
    first_audio_ms: int | None = None
    adaptive_schedule = _adaptive_schedule(opts)
    adaptive_stage = 0
    adaptive_buffer_ms = 0.0
    adaptive_chunks_completed_at_stage = 0
    if opts.adaptive_chunking and adaptive_schedule:
        opts = _adaptive_options_for_stage(opts, adaptive_schedule[0])

    yield StreamingEvent(
        type="plan_started",
        metadata={
            "planning_started_ms": 0,
            "lookahead_chunks": int(opts.lookahead_chunks),
        },
    )
    use_lazy_preflight = bool(opts.lazy_preflight and not opts.no_preflight_chunks)
    planning_opts = replace(opts, no_preflight_chunks=True) if use_lazy_preflight else opts
    render_chunks = prepare_render_chunks(runtime, records, planning_opts)
    if not render_chunks:
        raise ValueError("No speakable chunks found after text splitting")
    plan = synthesis_plan_from_prepared(
        runtime,
        records,
        render_chunks,
        opts,
        source_kind=source_kind,
    )
    plan_ready_ms = int(round((time.perf_counter() - stream_started_at) * 1000.0))
    yield StreamingEvent(
        type="plan_ready",
        plan=plan,
        metadata={
            "chunk_count": len(plan.chunks),
            "planning_ms": plan_ready_ms,
            "first_chunk_ready_ms": plan_ready_ms if plan.chunks else None,
            "lookahead_chunks": int(opts.lookahead_chunks),
            "lookahead_chunks_ready": min(int(opts.lookahead_chunks), max(0, len(plan.chunks) - 1)),
            "lazy_preflight": bool(use_lazy_preflight),
            "pipeline_preflight": bool(use_lazy_preflight and opts.pipeline_preflight),
            "preflight_window": _preflight_window(opts, len(render_chunks)),
            "preflight_queue_depth": 0,
        },
    )

    sample_rate = int(plan.sample_rate or _runtime_sample_rate(runtime) or 0)
    previous_latents: object | None = None
    previous_chain_id: str | None = None
    previous_pause_samples = 0
    rendered_samples = 0
    preflight_pipeline = _PreflightPipeline(
        runtime,
        opts,
        enabled=bool(use_lazy_preflight and opts.pipeline_preflight),
        window=_preflight_window(opts, len(render_chunks)),
    )
    index = 0
    try:
        while index < len(render_chunks):
            if _needs_lazy_preflight(render_chunks, index, opts):
                old_count = len(render_chunks)
                yield StreamingEvent(
                    type="chunk_preflight_started",
                    metadata={
                        "index": index,
                        "chunk_count": old_count,
                        "lazy_preflight": True,
                        "pipeline_preflight": bool(preflight_pipeline.enabled),
                        "preflight_queued": preflight_pipeline.has_index(render_chunks, index),
                        "preflight_queue_depth": preflight_pipeline.pending_count(),
                    },
                )
                preflight_info = _preflight_chunk_at(
                    runtime,
                    render_chunks,
                    index,
                    opts,
                    pipeline=preflight_pipeline,
                )
                yield StreamingEvent(
                    type="chunk_preflight_finished",
                    metadata=preflight_info,
                )
                if len(render_chunks) != old_count:
                    preflight_pipeline.clear()
                    plan = synthesis_plan_from_prepared(
                        runtime,
                        records,
                        render_chunks,
                        opts,
                        source_kind=source_kind,
                    )
                    yield StreamingEvent(
                        type="plan_updated",
                        plan=plan,
                        metadata={
                            "reason": "lazy_preflight_split",
                            "index": index,
                            "old_chunk_count": old_count,
                            "chunk_count": len(render_chunks),
                        },
                    )
                    preflight_pipeline.update_options(
                        opts,
                        window=_preflight_window(opts, len(render_chunks)),
                    )
            plan_chunks = plan_chunks_from_prepared(runtime, render_chunks, opts)
            chunk_plan = plan_chunks[index]
            chunk = render_chunks[index]
            seed = None if opts.seed is None else int(opts.seed) + index
            text_value = str(chunk["text"])
            emotion = _optional_text(chunk.get("emotion")) or opts.emotion
            affect = chunk.get("affect") if chunk.get("affect") is not None else opts.affect
            emotion_guidance = _optional_text(chunk.get("emotion_guidance")) or opts.emotion_guidance
            prefix_latents = previous_latents if previous_chain_id == chunk_plan.chain_id else None
            preflight_pipeline.fill(render_chunks, start_index=index + 1)
            _print_progress(
                progress_stream,
                f"[scyllasband] chunk {index + 1}/{len(render_chunks)} voice={chunk['voice']} "
                f"language={chunk['language']} emotion={emotion or 'default'} chars={len(text_value)} "
                f"before={chunk['boundary_before']} after={chunk['boundary_after']}",
            )
            yield StreamingEvent(type="chunk_started", chunk=chunk_plan, metadata={"index": index})
            synthesize_started_at = time.perf_counter()
            try:
                result = runtime.synthesize(
                    SynthesisRequest(
                        text=text_value,
                        voice_id=str(chunk["voice"]),
                        language=str(chunk["language"]),
                        emotion=emotion,
                        affect=affect,
                        affect_guidance_scale=float(
                            chunk.get("affect_guidance_scale")
                            if chunk.get("affect_guidance_scale") is not None
                            else opts.affect_guidance_scale
                        ),
                        emotion_guidance=emotion_guidance,
                        guidance_null_reference=not bool(opts.guidance_keep_reference),
                        emotion_embed_scale=float(opts.emotion_embed_scale),
                        prefix_latents=prefix_latents,
                        normalize_text=bool(chunk.get("normalize_text", not opts.no_normalize_text)),
                        seed=seed,
                        steps=opts.steps,
                        sampler=opts.sampler,
                        speed=opts.speed,
                        context_before=chunk_plan.context_before,
                        context_after=chunk_plan.context_after,
                        chunk_index=index,
                        chunk_count=len(render_chunks),
                        boundary_before=str(chunk["boundary_before"]),
                        boundary_after=str(chunk["boundary_after"]),
                        min_sentence_pause_ms=float(opts.min_sentence_pause_ms),
                        min_clause_pause_ms=float(opts.min_clause_pause_ms),
                    )
                )
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
                if replacements:
                    message = (
                        f"chunk {index + 1} exceeded fixed {budget_name} budget; "
                        f"retrying as {len(replacements)} smaller chunks"
                    )
                    _print_progress(progress_stream, f"[scyllasband] {message}")
                    render_chunks[index : index + 1] = replacements
                    preflight_pipeline.clear()
                    preflight_pipeline.update_options(
                        opts,
                        window=_preflight_window(opts, len(render_chunks)),
                    )
                    yield StreamingEvent(type="warning", message=message, metadata={"budget": budget_name})
                    plan = synthesis_plan_from_prepared(
                        runtime,
                        records,
                        render_chunks,
                        opts,
                        source_kind=source_kind,
                    )
                    continue
                raise

            generation_ms = int(round((time.perf_counter() - synthesize_started_at) * 1000.0))
            elapsed_ms = int(round((time.perf_counter() - stream_started_at) * 1000.0))
            if first_audio_ms is None:
                first_audio_ms = elapsed_ms
            sample_rate = int(result.sample_rate)
            previous_latents = result.latents
            previous_chain_id = chunk_plan.chain_id
            pause_ms = pause_after_ms(chunk, options=opts, is_last=index >= len(render_chunks) - 1)
            pause_samples = (
                max(0, int(round(sample_rate * pause_ms / 1000.0)))
                if index < len(render_chunks) - 1
                else 0
            )
            audio_piece = [float(item) for item in result.audio]
            audio_ms = (float(len(audio_piece)) * 1000.0 / float(sample_rate)) if sample_rate > 0 else 0.0
            if pause_samples and sample_rate > 0:
                audio_ms += float(pause_samples) * 1000.0 / float(sample_rate)
            if opts.adaptive_chunking:
                adaptive_buffer_ms = max(0.0, adaptive_buffer_ms - float(generation_ms)) + audio_ms
            start_sample = rendered_samples
            end_sample = rendered_samples + len(audio_piece)
            rendered_samples = end_sample + pause_samples
            chunk_metadata = _chunk_metadata(
                chunk,
                index=index,
                seed=seed,
                emotion=emotion,
                emotion_guidance=emotion_guidance,
                pause_ms=pause_ms,
                start_sample=start_sample,
                end_sample=end_sample,
                result=result,
                generation_ms=generation_ms,
                elapsed_ms=elapsed_ms,
                first_audio_ms=first_audio_ms,
                lookahead_chunks=opts.lookahead_chunks,
                audio_ms=audio_ms,
                buffer_ms=adaptive_buffer_ms if opts.adaptive_chunking else None,
                adaptive_stage=adaptive_stage if opts.adaptive_chunking else None,
                adaptive_max_chunk_chars=opts.max_chunk_chars if opts.adaptive_chunking else None,
            )
            yield StreamingEvent(
                type="audio_chunk",
                chunk=chunk_plan,
                audio=audio_piece,
                sample_rate=sample_rate,
                start_sample=start_sample,
                end_sample=end_sample,
                metadata=chunk_metadata,
            )
            yield StreamingEvent(
                type="chunk_finished",
                chunk=chunk_plan,
                sample_rate=sample_rate,
                start_sample=start_sample,
                end_sample=end_sample,
                metadata=chunk_metadata,
            )
            if opts.adaptive_chunking and adaptive_stage + 1 < len(adaptive_schedule):
                adaptive_chunks_completed_at_stage += 1
                realtime_factor = (audio_ms / float(generation_ms)) if generation_ms > 0 else float("inf")
                should_advance = (
                    adaptive_chunks_completed_at_stage >= opts.adaptive_min_chunks_per_stage
                    and adaptive_buffer_ms >= float(opts.adaptive_buffer_ms)
                    and realtime_factor >= float(opts.adaptive_realtime_factor)
                )
                if should_advance:
                    old_count = len(render_chunks)
                    next_stage = adaptive_stage + 1
                    next_opts = _adaptive_options_for_stage(opts, adaptive_schedule[next_stage])
                    updated_chunks = _merge_future_chunks_for_adaptive(
                        render_chunks,
                        start_index=index + 1,
                        max_chars=next_opts.max_chunk_chars,
                    )
                    opts = next_opts
                    adaptive_stage = next_stage
                    adaptive_chunks_completed_at_stage = 0
                    preflight_pipeline.update_options(
                        opts,
                        window=_preflight_window(opts, len(updated_chunks)),
                    )
                    if updated_chunks is not render_chunks:
                        render_chunks = updated_chunks
                        plan = synthesis_plan_from_prepared(
                            runtime,
                            records,
                            render_chunks,
                            opts,
                            source_kind=source_kind,
                        )
                        yield StreamingEvent(
                            type="plan_updated",
                            plan=plan,
                            metadata={
                                "adaptive_stage": adaptive_stage,
                                "adaptive_max_chunk_chars": opts.max_chunk_chars,
                                "adaptive_buffer_ms": int(round(adaptive_buffer_ms)),
                                "adaptive_realtime_factor": realtime_factor,
                                "old_chunk_count": old_count,
                                "chunk_count": len(render_chunks),
                            },
                        )
            previous_pause_samples = pause_samples
            index += 1
    finally:
        preflight_pipeline.close()
    final_plan = synthesis_plan_from_prepared(
        runtime,
        records,
        render_chunks,
        opts,
        source_kind=source_kind,
    )
    yield StreamingEvent(type="done", plan=final_plan, sample_rate=sample_rate)


def render_text_records(
    runtime: Any,
    records: list[dict[str, Any]],
    options: object | None = None,
    *,
    source_kind: str = "records",
    progress_stream: Any | None = sys.stderr,
    **overrides: Any,
) -> tuple[list[float], int, dict[str, Any]]:
    opts = planner_options_from(options, **overrides)
    rendered: list[float] = []
    metadata_chunks: list[dict[str, Any]] = []
    sample_rate = int(_runtime_sample_rate(runtime) or 0)
    previous_pause_samples = 0
    final_plan: SynthesisPlan | None = None
    for event in synthesize_records_stream(
        runtime,
        records,
        options=opts,
        source_kind=source_kind,
        progress_stream=progress_stream,
    ):
        if event.plan is not None:
            final_plan = event.plan
        if event.type != "audio_chunk":
            continue
        sample_rate = int(event.sample_rate or sample_rate)
        fade_samples = _fade_sample_count(float(opts.boundary_fade_ms), sample_rate)
        next_gap_samples = _gap_after_event(event, sample_rate)
        start_sample, end_sample = _append_rendered_audio(
            rendered,
            event.audio,
            fade_samples=fade_samples,
            previous_gap_samples=previous_pause_samples,
            next_gap_samples=next_gap_samples,
        )
        chunk_metadata = dict(event.metadata or {})
        chunk_metadata["start_sample"] = start_sample
        chunk_metadata["end_sample"] = end_sample
        metadata_chunks.append(chunk_metadata)
        if next_gap_samples:
            rendered.extend([0.0] * next_gap_samples)
            previous_pause_samples = next_gap_samples
        else:
            previous_pause_samples = 0
    if final_plan is None:
        raise ValueError("No synthesis plan was produced")
    return rendered, sample_rate, _render_metadata(
        opts,
        final_plan=final_plan,
        sample_rate=sample_rate,
        chunks=metadata_chunks,
    )


def _chunk_metadata(
    chunk: dict[str, Any],
    *,
    index: int,
    seed: int | None,
    emotion: str | None,
    emotion_guidance: str | None,
    pause_ms: int,
    start_sample: int,
    end_sample: int,
    result: Any,
    generation_ms: int | None = None,
    elapsed_ms: int | None = None,
    first_audio_ms: int | None = None,
    lookahead_chunks: int | None = None,
    audio_ms: float | None = None,
    buffer_ms: float | None = None,
    adaptive_stage: int | None = None,
    adaptive_max_chunk_chars: int | None = None,
) -> dict[str, Any]:
    metadata = {
        "index": index,
        "voice": chunk["voice"],
        "language": chunk["language"],
        "emotion": emotion,
        "affect": chunk.get("affect"),
        "affect_guidance_scale": chunk.get("affect_guidance_scale"),
        "emotion_guidance": emotion_guidance,
        "seed": seed,
        "text": chunk["text"],
        "source_text": chunk.get("source_text"),
        "pre_normalized": bool(chunk.get("pre_normalized", False)),
        "split_reason": chunk.get("split_reason"),
        "boundary_before": chunk.get("boundary_before"),
        "boundary_after": chunk.get("boundary_after"),
        "starts_sentence": bool(chunk.get("starts_sentence", False)),
        "ends_sentence": bool(chunk.get("ends_sentence", False)),
        "pause_after_ms": pause_ms,
        "start_sample": start_sample,
        "end_sample": end_sample,
        "duration_seconds": float(result.duration_seconds),
        "metadata": result.metadata,
    }
    if generation_ms is not None:
        metadata["stream_generation_ms"] = int(generation_ms)
    if elapsed_ms is not None:
        metadata["stream_elapsed_ms"] = int(elapsed_ms)
    if first_audio_ms is not None:
        metadata["first_audio_ms"] = int(first_audio_ms)
    if lookahead_chunks is not None:
        metadata["lookahead_chunks"] = int(lookahead_chunks)
    if audio_ms is not None:
        metadata["stream_audio_ms"] = int(round(audio_ms))
    if buffer_ms is not None:
        metadata["stream_buffer_ms"] = int(round(buffer_ms))
    if adaptive_stage is not None:
        metadata["adaptive_stage"] = int(adaptive_stage)
    if adaptive_max_chunk_chars is not None:
        metadata["adaptive_max_chunk_chars"] = int(adaptive_max_chunk_chars)
    for key in (
        "preflight_predicted_latent_frames",
        "preflight_fixed_latent_frames",
        "lazy_preflight_ms",
        "lazy_preflight_total_ms",
        "lazy_preflight_replacement_count",
        "lazy_preflight_source_index",
        "lazy_preflight_source",
        "lazy_preflight_wait_ms",
        "lazy_preflight_pipeline_queued",
        "latent_retry_predicted_frames",
        "latent_retry_fixed_frames",
        "phone_retry_phone_count",
        "phone_retry_fixed_frames",
        "g2p_retry_encoded_tokens",
        "g2p_retry_fixed_tokens",
    ):
        if key in chunk:
            metadata[key] = chunk.get(key)
    return metadata


def _render_metadata(
    opts: PlannerOptions,
    *,
    final_plan: SynthesisPlan,
    sample_rate: int,
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "mode": "chunked",
        "backend": opts.backend,
        "sampler": opts.sampler,
        "steps": int(opts.steps),
        "speed": float(opts.speed),
        "emotion": opts.emotion,
        "affect": opts.affect,
        "affect_guidance_scale": float(opts.affect_guidance_scale),
        "emotion_guidance": opts.emotion_guidance,
        "guidance_null_reference": not bool(opts.guidance_keep_reference),
        "emotion_embed_scale": float(opts.emotion_embed_scale),
        "sample_rate": sample_rate,
        "pause_ms": int(opts.pause_ms),
        "continuation_pause_ms": int(opts.continuation_pause_ms),
        "boundary_fade_ms": float(opts.boundary_fade_ms),
        "min_sentence_pause_ms": float(opts.min_sentence_pause_ms),
        "min_clause_pause_ms": float(opts.min_clause_pause_ms),
        "chunk_max_chars": int(opts.max_chunk_chars),
        "chunk_min_chars": int(opts.min_chunk_chars),
        "preflight_chunks": not bool(opts.no_preflight_chunks),
        "auto_split_overlong": not bool(opts.no_auto_split_overlong),
        "adaptive_chunking": bool(opts.adaptive_chunking),
        "adaptive_chunk_schedule": list(opts.adaptive_chunk_schedule),
        "adaptive_min_chunk_chars": int(opts.adaptive_min_chunk_chars),
        "adaptive_buffer_ms": int(opts.adaptive_buffer_ms),
        "adaptive_realtime_factor": float(opts.adaptive_realtime_factor),
        "adaptive_min_chunks_per_stage": int(opts.adaptive_min_chunks_per_stage),
        "adaptive_preflight_chunks": bool(opts.adaptive_preflight_chunks),
        "lazy_preflight": bool(opts.lazy_preflight and not opts.no_preflight_chunks),
        "pipeline_preflight": bool(opts.pipeline_preflight and opts.lazy_preflight and not opts.no_preflight_chunks),
        "preflight_window": _preflight_window(opts, len(chunks)),
        "lookahead_chunks": int(opts.lookahead_chunks),
        "first_audio_ms": _first_chunk_value(chunks, "first_audio_ms"),
        "plan": final_plan.to_dict(),
        "chunk_count": len(chunks),
        "chunks": chunks,
    }


def _first_chunk_value(chunks: list[dict[str, Any]], key: str) -> Any | None:
    if not chunks:
        return None
    return chunks[0].get(key)


def _preflight_window(opts: PlannerOptions, chunk_count: int) -> int:
    if bool(opts.no_preflight_chunks) or not bool(opts.lazy_preflight) or chunk_count <= 0:
        return 0
    return min(int(chunk_count), max(1, int(opts.lookahead_chunks) + 1))


def _needs_lazy_preflight(chunks: list[dict[str, Any]], index: int, opts: PlannerOptions) -> bool:
    if bool(opts.no_preflight_chunks) or not bool(opts.lazy_preflight) or index >= len(chunks):
        return False
    return not bool(chunks[index].get("_lazy_preflight_done"))


@dataclass(frozen=True)
class _LazyPreflightResult:
    source_index: int
    key: tuple[Any, ...]
    replacements: list[dict[str, Any]]
    preflight_ms: int


class _PreflightPipeline:
    def __init__(
        self,
        runtime: Any,
        opts: PlannerOptions,
        *,
        enabled: bool,
        window: int,
    ) -> None:
        self.runtime = runtime
        self.opts = opts
        self.enabled = bool(enabled and window > 0)
        self.window = max(0, int(window))
        self._executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="scyllasband-preflight")
            if self.enabled
            else None
        )
        self._futures: dict[int, tuple[tuple[Any, ...], Future[_LazyPreflightResult]]] = {}

    def fill(self, chunks: list[dict[str, Any]], *, start_index: int) -> None:
        if not self.enabled:
            return
        end = min(len(chunks), max(0, int(start_index)) + self.window)
        for index in range(max(0, int(start_index)), end):
            self.schedule(chunks, index)

    def schedule(self, chunks: list[dict[str, Any]], index: int) -> None:
        if not self.enabled or self._executor is None or not _needs_lazy_preflight(chunks, index, self.opts):
            return
        key = _preflight_key(chunks[index])
        existing = self._futures.get(index)
        if existing is not None:
            existing_key, existing_future = existing
            if existing_key == key:
                return
            existing_future.cancel()
        chunk = dict(chunks[index])
        self._futures[index] = (
            key,
            self._executor.submit(_run_lazy_preflight, self.runtime, chunk, index, self.opts, key),
        )

    def pop_result(
        self,
        chunks: list[dict[str, Any]],
        index: int,
    ) -> tuple[_LazyPreflightResult, int, bool] | None:
        if not self.enabled:
            return None
        item = self._futures.pop(index, None)
        if item is None:
            return None
        key, future = item
        if index >= len(chunks) or _preflight_key(chunks[index]) != key:
            future.cancel()
            return None
        wait_started_at = time.perf_counter()
        result = future.result()
        wait_ms = int(round((time.perf_counter() - wait_started_at) * 1000.0))
        if index >= len(chunks) or result.key != _preflight_key(chunks[index]):
            return None
        return result, wait_ms, True

    def has_index(self, chunks: list[dict[str, Any]], index: int) -> bool:
        item = self._futures.get(index)
        return item is not None and index < len(chunks) and item[0] == _preflight_key(chunks[index])

    def pending_count(self) -> int:
        return len(self._futures)

    def update_options(self, opts: PlannerOptions, *, window: int) -> None:
        window = max(0, int(window))
        if self.opts == opts and self.window == window:
            return
        self.clear()
        self.opts = opts
        self.window = window
        self.enabled = bool(self._executor is not None and window > 0)

    def clear(self) -> None:
        for _, future in self._futures.values():
            future.cancel()
        self._futures.clear()

    def close(self) -> None:
        self.clear()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None


def _preflight_chunk_at(
    runtime: Any,
    chunks: list[dict[str, Any]],
    index: int,
    opts: PlannerOptions,
    *,
    pipeline: _PreflightPipeline | None = None,
) -> dict[str, Any]:
    original_count = len(chunks)
    pipelined = pipeline.pop_result(chunks, index) if pipeline is not None else None
    if pipelined is None:
        result = _run_lazy_preflight(runtime, dict(chunks[index]), index, opts, _preflight_key(chunks[index]))
        wait_ms = 0
        source = "sync"
        queued = False
    else:
        result, wait_ms, queued = pipelined
        source = "pipeline"
    replacements = [dict(item) for item in result.replacements] or [dict(chunks[index])]
    replacement_count = len(replacements)
    for offset, item in enumerate(replacements):
        item["_lazy_preflight_done"] = True
        item["lazy_preflight_ms"] = result.preflight_ms if offset == 0 else 0
        item["lazy_preflight_total_ms"] = result.preflight_ms
        item["lazy_preflight_replacement_count"] = replacement_count
        item["lazy_preflight_source_index"] = index
        item["lazy_preflight_source"] = source
        item["lazy_preflight_wait_ms"] = wait_ms
        item["lazy_preflight_pipeline_queued"] = queued
    chunks[index : index + 1] = replacements
    return {
        "index": index,
        "lazy_preflight": True,
        "pipeline_preflight": source == "pipeline",
        "preflight_source": source,
        "preflight_ms": result.preflight_ms,
        "preflight_wait_ms": wait_ms,
        "replacement_count": replacement_count,
        "old_chunk_count": original_count,
        "chunk_count": len(chunks),
        "preflight_queue_depth": pipeline.pending_count() if pipeline is not None else 0,
    }


def _run_lazy_preflight(
    runtime: Any,
    chunk: dict[str, Any],
    index: int,
    opts: PlannerOptions,
    key: tuple[Any, ...],
) -> _LazyPreflightResult:
    started_at = time.perf_counter()
    replacements = preflight_split_overlong_chunks(runtime, [dict(chunk)], opts)
    elapsed_ms = int(round((time.perf_counter() - started_at) * 1000.0))
    return _LazyPreflightResult(
        source_index=index,
        key=key,
        replacements=[dict(item) for item in (replacements or [chunk])],
        preflight_ms=elapsed_ms,
    )


def _preflight_key(chunk: dict[str, Any]) -> tuple[Any, ...]:
    return (
        chunk.get("record_id"),
        chunk.get("voice"),
        chunk.get("language"),
        chunk.get("emotion"),
        chunk.get("emotion_guidance"),
        chunk.get("text"),
        chunk.get("boundary_before"),
        chunk.get("boundary_after"),
        chunk.get("normalize_text"),
    )


_ADAPTIVE_INVALIDATED_METADATA_KEYS = frozenset(
    {
        "preflight_predicted_latent_frames",
        "preflight_fixed_latent_frames",
        "_lazy_preflight_done",
        "lazy_preflight_ms",
        "lazy_preflight_total_ms",
        "lazy_preflight_replacement_count",
        "lazy_preflight_source_index",
        "lazy_preflight_source",
        "lazy_preflight_wait_ms",
        "lazy_preflight_pipeline_queued",
        "latent_retry_predicted_frames",
        "latent_retry_fixed_frames",
        "phone_retry_phone_count",
        "phone_retry_fixed_frames",
        "g2p_retry_encoded_tokens",
        "g2p_retry_fixed_tokens",
    }
)


def _adaptive_schedule(opts: PlannerOptions) -> tuple[int, ...]:
    if not bool(opts.adaptive_chunking):
        return ()
    output: list[int] = []
    last = 0
    for value in opts.adaptive_chunk_schedule:
        number = int(value)
        if number <= last:
            continue
        output.append(number)
        last = number
    return tuple(output)


def _adaptive_options_for_stage(opts: PlannerOptions, max_chars: int) -> PlannerOptions:
    stage_max = max(1, int(max_chars))
    requested_min = int(opts.adaptive_min_chunk_chars or opts.min_chunk_chars)
    stage_min = max(0, min(stage_max, requested_min))
    return replace(
        opts,
        max_chunk_chars=stage_max,
        min_chunk_chars=stage_min,
        no_preflight_chunks=bool(opts.no_preflight_chunks or not opts.adaptive_preflight_chunks),
    )


def _merge_future_chunks_for_adaptive(
    chunks: list[dict[str, Any]],
    *,
    start_index: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    if start_index >= len(chunks) - 1:
        return chunks
    output = [dict(chunk) for chunk in chunks[:start_index]]
    index = start_index
    changed = False
    while index < len(chunks):
        current = dict(chunks[index])
        index += 1
        while index < len(chunks):
            candidate = chunks[index]
            if not _can_adaptively_merge_chunks(current, candidate):
                break
            merged_text = f"{current.get('text', '')} {candidate.get('text', '')}".strip()
            if len(merged_text) > max_chars:
                break
            current = _merge_adaptive_chunks(current, candidate, text=merged_text)
            changed = True
            index += 1
        output.append(current)
    return output if changed else chunks


def _can_adaptively_merge_chunks(left: dict[str, Any], right: dict[str, Any]) -> bool:
    for key in (
        "record_id",
        "voice",
        "language",
        "emotion",
        "emotion_guidance",
        "paragraph_index",
        "pre_normalized",
        "normalize_text",
    ):
        if left.get(key) != right.get(key):
            return False
    if left.get("boundary_after") == "chunk_continue":
        return False
    if right.get("boundary_before") == "chunk_continue":
        return False
    if left.get("boundary_after") == "paragraph_end":
        return False
    if right.get("boundary_before") == "paragraph_start":
        return False
    return True


def _merge_adaptive_chunks(left: dict[str, Any], right: dict[str, Any], *, text: str) -> dict[str, Any]:
    merged = dict(left)
    merged["text"] = text
    merged["boundary_after"] = right.get("boundary_after")
    merged["ends_sentence"] = bool(right.get("ends_sentence", False))
    merged["piece_index"] = left.get("piece_index", 0)
    merged["piece_count"] = None
    merged["split_reason"] = "adaptive_merged"
    merged["adaptive_merged"] = True
    merged["adaptive_merged_text_count"] = int(left.get("adaptive_merged_text_count") or 1) + int(
        right.get("adaptive_merged_text_count") or 1
    )
    for key in _ADAPTIVE_INVALIDATED_METADATA_KEYS:
        merged.pop(key, None)
    return merged


def _gap_after_event(event: StreamingEvent, sample_rate: int) -> int:
    metadata = event.metadata or {}
    pause_ms = int(metadata.get("pause_after_ms") or 0)
    return max(0, int(round(sample_rate * pause_ms / 1000.0)))


def _fade_sample_count(boundary_fade_ms: float, sample_rate: int) -> int:
    return max(0, int(round(float(boundary_fade_ms) * int(sample_rate) / 1000.0)))


def _smoothstep_weight(index: int, count: int) -> float:
    if count <= 1:
        return 1.0
    value = float(index) / float(count - 1)
    return value * value * (3.0 - 2.0 * value)


def _prepare_audio_piece(
    audio: list[float],
    *,
    fade_samples: int,
    previous_gap_samples: int,
    next_gap_samples: int,
) -> list[float]:
    return _fade_audio_edges(
        audio,
        fade_samples,
        fade_in=previous_gap_samples > 0,
        fade_out=next_gap_samples > 0,
    )


def _fade_audio_edges(
    audio: list[float],
    fade_samples: int,
    *,
    fade_in: bool,
    fade_out: bool,
) -> list[float]:
    if fade_samples <= 0 or not audio:
        return list(audio)
    out = list(audio)
    local = min(int(fade_samples), len(out) // 2)
    if local <= 0:
        return out
    if fade_in:
        for offset in range(local):
            out[offset] *= _smoothstep_weight(offset, local)
    if fade_out:
        start = len(out) - local
        for offset in range(local):
            out[start + offset] *= 1.0 - _smoothstep_weight(offset, local)
    return out


def _append_rendered_audio(
    rendered: list[float],
    audio: object,
    *,
    fade_samples: int,
    previous_gap_samples: int,
    next_gap_samples: int,
) -> tuple[int, int]:
    piece = [float(item) for item in audio]
    if not piece:
        start = len(rendered)
        return start, start
    if previous_gap_samples > 0 or next_gap_samples > 0:
        piece = _fade_audio_edges(
            piece,
            fade_samples,
            fade_in=previous_gap_samples > 0,
            fade_out=next_gap_samples > 0,
        )
    if previous_gap_samples <= 0 and rendered and fade_samples > 0:
        local = min(int(fade_samples), len(rendered), len(piece))
        if local > 0:
            start = len(rendered) - local
            for offset in range(local):
                weight = _smoothstep_weight(offset, local)
                rendered[start + offset] = rendered[start + offset] * (1.0 - weight) + piece[offset] * weight
            rendered.extend(piece[local:])
            return start, len(rendered)
    start = len(rendered)
    rendered.extend(piece)
    return start, len(rendered)


def _runtime_sample_rate(runtime: Any) -> int | None:
    manifest = getattr(runtime, "manifest", None)
    audio = getattr(manifest, "audio", None)
    sample_rate = getattr(audio, "sample_rate", None)
    return int(sample_rate) if sample_rate else None


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _print_progress(stream: Any | None, message: str) -> None:
    if stream is None:
        return
    print(message, file=stream, flush=True)
