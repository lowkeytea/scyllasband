"""Speak a plan sentence by sentence, yielding audio as each sentence is ready.

Within a chain, every sentence gets the neighbouring sentences' phones as context, the last latent frames
already spoken as its acoustic prefix, and the preceding latents as the vocoder's left context, so the
emitted pieces join into one continuous waveform with no added pauses or crossfades.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
import time
from typing import Any, Iterator, TextIO

import numpy as np

from .delivery import delivery_tensors
from .engine import Engine, OverlongError, Sentence
from .planner import PlanChunk, SynthesisPlan, split_for_retry


@dataclass
class StreamingEvent:
    type: str                                  # plan | chunk_started | audio_chunk | done
    chunk: PlanChunk | None = None
    audio: np.ndarray | None = None
    sample_rate: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_audio: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = dict(type=self.type, metadata=self.metadata)
        if self.chunk is not None:
            payload["chunk"] = self.chunk.to_dict()
        if self.sample_rate is not None:
            payload["sample_rate"] = self.sample_rate
        if self.audio is not None:
            payload["samples"] = int(self.audio.size)
            if include_audio:
                payload["audio"] = self.audio.tolist()
        return payload


@dataclass
class StreamOptions:
    steps: int = 8
    sampler: str = "heun"
    speed: float = 1.0
    seed: int | None = None
    temperature: float = 1.0
    pause_ms: float = 0.0          # extra silence where the voice, language or delivery changes


def synthesize_plan_stream(engine: Engine, plan: SynthesisPlan, options: StreamOptions | None = None, *,
                           progress: TextIO | None = None) -> Iterator[StreamingEvent]:
    options = options or StreamOptions()
    sample_rate = int(engine.manifest["audio"]["sample_rate"])
    yield StreamingEvent(type="plan", metadata=plan.to_dict())
    pending = deque(plan.chunks)
    spoken: list[PlanChunk] = []
    tails: dict[str, np.ndarray] = {}
    started = time.perf_counter()
    first_audio_ms: float | None = None
    previous_chain: str | None = None
    while pending:
        chunk = pending.popleft()
        sentence = _sentence(engine, chunk, spoken, list(pending))
        values, present, _ = delivery_tensors(chunk.delivery)
        seed = None if options.seed is None else int(options.seed) + chunk.index
        tail = tails.get(chunk.chain_id)
        yield StreamingEvent(type="chunk_started", chunk=chunk)
        _progress(progress, f"[scyllasband] {chunk.chunk_id} {chunk.voice}/{chunk.language}: {chunk.text[:80]}")
        try:
            result = engine.synthesize_sentence(sentence, voice=chunk.voice, language=chunk.language, delivery=(values, present),
                                                prefix=tail, steps=options.steps, sampler=options.sampler, speed=options.speed,
                                                noise_scale=options.temperature, rng=np.random.default_rng(seed))
        except OverlongError:
            pieces = split_for_retry(chunk.text)
            for offset, piece in enumerate(reversed(pieces)):
                pending.appendleft(replace(chunk, chunk_id=f"{chunk.chunk_id}_{len(pieces) - offset}", text=piece))
            _progress(progress, f"[scyllasband] {chunk.chunk_id} too long for one pass; split in {len(pieces)}")
            continue
        audio = engine.decode(result.latents, voice=chunk.voice, language=chunk.language, left=tail)
        if previous_chain is not None and previous_chain != chunk.chain_id and options.pause_ms > 0:
            audio = np.concatenate([np.zeros(int(sample_rate * options.pause_ms / 1000), np.float32), audio])
        joined = result.latents if tail is None else np.concatenate([tail, result.latents], -1)
        tails[chunk.chain_id] = joined[:, -max(engine.prefix_frames, engine.decode_context):]
        previous_chain = chunk.chain_id
        spoken.append(chunk)
        if first_audio_ms is None:
            first_audio_ms = (time.perf_counter() - started) * 1000
        yield StreamingEvent(type="audio_chunk", chunk=chunk, audio=audio, sample_rate=sample_rate,
                             metadata=dict(result.metadata, seed=seed, first_audio_ms=round(first_audio_ms, 1)))
    yield StreamingEvent(type="done", sample_rate=sample_rate,
                         metadata=dict(chunks=len(spoken), first_audio_ms=None if first_audio_ms is None else round(first_audio_ms, 1),
                                       elapsed_ms=round((time.perf_counter() - started) * 1000, 1)))


def _sentence(engine: Engine, chunk: PlanChunk, spoken: list[PlanChunk], pending: list[PlanChunk]) -> Sentence:
    """Phones of ``chunk`` plus up to the engine's context budget of neighbouring phones from the same chain."""
    phonemized = engine.phonemize(chunk.text, language=chunk.language)
    before: list[int] = []
    for neighbour in reversed([c for c in spoken if c.chain_id == chunk.chain_id]):
        if len(before) >= engine.context_phones:
            break
        before = engine.context_ids(engine.phonemize(neighbour.text, language=neighbour.language)["phones"]) + before
    after: list[int] = []
    for neighbour in (c for c in pending if c.chain_id == chunk.chain_id):
        if len(after) >= engine.context_phones:
            break
        after += engine.context_ids(engine.phonemize(neighbour.text, language=neighbour.language)["phones"])
    return Sentence(phones=phonemized["phones"], word_starts=phonemized["word_starts"], before_ids=before, after_ids=after)


def _progress(stream: TextIO | None, message: str) -> None:
    if stream is not None:
        print(message, file=stream, flush=True)


def render_plan(engine: Engine, plan: SynthesisPlan, options: StreamOptions | None = None, *,
                progress: TextIO | None = None) -> tuple[np.ndarray, int, dict[str, Any]]:
    pieces, chunks, sample_rate, summary = [], [], int(engine.manifest["audio"]["sample_rate"]), {}
    for event in synthesize_plan_stream(engine, plan, options, progress=progress):
        if event.type == "audio_chunk" and event.audio is not None:
            pieces.append(event.audio)
            chunks.append(dict(chunk=event.chunk.to_dict() if event.chunk else None, **event.metadata))
        elif event.type == "done":
            summary = event.metadata
    audio = np.concatenate(pieces) if pieces else np.zeros(0, np.float32)
    return audio, sample_rate, dict(summary, chunks=chunks)

