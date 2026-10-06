"""Speak a plan as packed targets, yielding audio as each target is ready.

Within a chain, the plan's sentences are read as one stream of phones, the way the model saw connected text in
training. Each synthesized target is a span of that stream from one silence to another, sized by the durations the
model predicts for its own delivery and speed:

- the first target of a request ends at the first sentence end past the trained minimum, so audio starts quickly (at a
  clause, else between words, only when the opening sentence is too long for one pass);
- every later target reaches as far toward the trained maximum as the text allows, ending at a sentence boundary, else
  at a clause, else between words;
- a paragraph or a new record ends a target once it is long enough, and a change of voice, language or delivery always
  does; a short remainder is never left on its own;
- a target over the trained maximum gives way to an earlier boundary before it is synthesized.

Every target gets the neighbouring phones as context, the last latent frames already spoken as its acoustic prefix, and
the preceding latents as the vocoder's left context, so the emitted pieces join into one continuous waveform with no
added pauses or crossfades.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import time
from typing import Any, Iterator, TextIO

import numpy as np

from .delivery import delivery_tensors
from .engine import Engine, Sentence
from .events import SILENCE
from .planner import PlanChunk, SynthesisPlan


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


SENTENCE_END = frozenset({"<end_stmt>", "<end_question>", "<end_exclaim>", "<ellipsis>"})
CLAUSE_PAUSE = frozenset({"<pause_comma>", "<pause_semicolon>", "<pause_colon>", "<pause_dash>"})
WINDOW_PHONES = 200        # phones measured ahead to choose a target's end; past the trained maximum at any speed
FIRST_WINDOW_PHONES = 48   # the first target looks only this far ahead at first, so little text is phonemized early
SENTENCE_STOPS = ("sentence", "paragraph", "end")
FILL_SHARE = 0.5           # a later target prefers a sentence end in the upper half of the trained range


def synthesize_plan_stream(engine: Engine, plan: SynthesisPlan, options: StreamOptions | None = None, *,
                           progress: TextIO | None = None) -> Iterator[StreamingEvent]:
    options = options or StreamOptions()
    sample_rate = int(engine.manifest["audio"]["sample_rate"])
    yield StreamingEvent(type="plan", metadata=plan.to_dict())
    started = time.perf_counter()
    first_audio_ms: float | None = None
    spoken = 0
    previous_chain: str | None = None
    for chain in _chains(plan.chunks):
        stream = _ChainStream(engine, chain)
        values, present, _ = delivery_tensors(chain[0].delivery)
        tail: np.ndarray | None = None
        start = 0
        while True:
            end, sentence, durations = _next_target(engine, stream, start, spoken == 0, (values, present), options.speed)
            chunk = stream.target_chunk(start, end)
            seed = None if options.seed is None else int(options.seed) + chunk.index
            yield StreamingEvent(type="chunk_started", chunk=chunk)
            _progress(progress, f"[scyllasband] {chunk.chunk_id} {chunk.voice}/{chunk.language}: {chunk.text[:80]}")
            result = engine.synthesize_sentence(sentence, voice=chunk.voice, language=chunk.language, delivery=(values, present),
                                                prefix=tail, steps=options.steps, sampler=options.sampler, speed=options.speed,
                                                noise_scale=options.temperature, rng=np.random.default_rng(seed), durations=durations)
            audio = engine.decode(result.latents, voice=chunk.voice, language=chunk.language, left=tail)
            if previous_chain is not None and previous_chain != chunk.chain_id and options.pause_ms > 0:
                audio = np.concatenate([np.zeros(int(sample_rate * options.pause_ms / 1000), np.float32), audio])
            joined = result.latents if tail is None else np.concatenate([tail, result.latents], -1)
            tail = joined[:, -max(engine.prefix_frames, engine.decode_context):]
            previous_chain = chunk.chain_id
            spoken += 1
            if first_audio_ms is None:
                first_audio_ms = (time.perf_counter() - started) * 1000
            yield StreamingEvent(type="audio_chunk", chunk=chunk, audio=audio, sample_rate=sample_rate,
                                 metadata=dict(result.metadata, seed=seed, first_audio_ms=round(first_audio_ms, 1)))
            if stream.at_end(end):
                break
            start = end
    yield StreamingEvent(type="done", sample_rate=sample_rate,
                         metadata=dict(chunks=spoken, first_audio_ms=None if first_audio_ms is None else round(first_audio_ms, 1),
                                       elapsed_ms=round((time.perf_counter() - started) * 1000, 1)))


def _chains(chunks: list[PlanChunk]) -> Iterator[list[PlanChunk]]:
    """Consecutive plan sentences that share a chain."""
    group: list[PlanChunk] = []
    for chunk in chunks:
        if group and chunk.chain_id != group[-1].chain_id:
            yield group
            group = []
        group.append(chunk)
    if group:
        yield group


@dataclass
class _Piece:
    chunk: PlanChunk
    start: int                 # stream index of the sentence's leading silence
    end: int                   # stream index of its trailing silence
    word_phones: list[int]     # stream index of each word's first phone
    parts: int = 0             # targets that took part of the sentence so far


class _ChainStream:
    """One chain's sentences joined as connected text (one shared silence between sentences), phonemized only as far
    as the targets and their context need. ``cuts`` maps silences that can end a target to their kind: end, paragraph
    (a paragraph or record ends there), sentence, clause or word."""

    def __init__(self, engine: Engine, chunks: list[PlanChunk]):
        self.engine, self.chunks = engine, chunks
        self.phones: list[str] = []
        self.word_starts: list[int] = []
        self.cuts: dict[int, str] = {}
        self.pieces: list[_Piece] = []

    @property
    def complete(self) -> bool:
        return len(self.pieces) == len(self.chunks)

    def extend(self, length: int) -> None:
        while len(self.phones) < length and not self.complete:
            self._append(self.chunks[len(self.pieces)])

    def extend_context(self, end: int) -> None:
        """Phonemize until ``reach`` context phones follow ``end`` (or the chain is complete)."""
        while not self.complete and sum(p != SILENCE for p in self.phones[end + 1:]) < self.engine.context_phones:
            self._append(self.chunks[len(self.pieces)])

    def at_end(self, index: int) -> bool:
        self.extend(index + 2)
        return self.complete and index == len(self.phones) - 1

    def _append(self, chunk: PlanChunk) -> None:
        result = self.engine.phonemize(chunk.text, language=chunk.language)
        phones = list(result["phones"])
        base = len(self.phones) - 1 if self.phones else 0
        if self.phones:
            previous = self.pieces[-1].chunk
            self.cuts[base] = "paragraph" if previous.paragraph_end or previous.record_index != chunk.record_index else "sentence"
            self.phones.extend(phones[1:])
        else:
            self.phones.extend(phones)
        first_word = base + next(j for j, phone in enumerate(phones) if phone != SILENCE)
        starts = [base + j for j in result["word_starts"]]
        self.word_starts.extend(([first_word] if self.pieces else []) + starts)
        candidates = set(result["word_boundary_candidates"])
        for j in range(1, len(phones) - 1):
            if phones[j] == SILENCE:
                kind = ("sentence" if phones[j - 1] in SENTENCE_END else "clause" if phones[j - 1] in CLAUSE_PAUSE
                        else "word" if j in candidates else None)
                if kind:
                    self.cuts[base + j] = kind
        self.cuts[base + len(phones) - 1] = "end"
        self.pieces.append(_Piece(chunk, base, base + len(phones) - 1, [first_word, *starts]))

    def sentence(self, start: int, end: int) -> Sentence:
        """The target ``[start, end]`` (silence to silence) with its context."""
        reach = self.engine.context_phones
        self.extend_context(end)
        first_word = next(i for i in range(start, end + 1) if self.phones[i] != SILENCE)
        word_starts = [w - start for w in self.word_starts if start < w <= end and w != first_word]
        before = self.engine.context_ids(self.phones[max(0, start - 3 * reach):start])[-reach:]
        after = self.engine.context_ids(self.phones[end + 1:end + 1 + 3 * reach])[:reach]
        return Sentence(phones=self.phones[start:end + 1], word_starts=word_starts, before_ids=before, after_ids=after)

    def target_chunk(self, start: int, end: int) -> PlanChunk:
        """A plan chunk describing the target: its sentences' ids and text (part of a sentence gets ``id.n``)."""
        labels, texts = [], []
        pieces = [p for p in self.pieces if p.start < end and p.end > start]
        for piece in pieces:
            words = piece.chunk.text.split()
            inside = [i for i, w in enumerate(piece.word_phones) if start < w < end]
            if len(inside) == len(piece.word_phones):
                labels.append(piece.chunk.chunk_id)
                texts.append(piece.chunk.text)
                continue
            piece.parts += 1
            labels.append(f"{piece.chunk.chunk_id}.{piece.parts}")
            if len(words) == len(piece.word_phones):
                texts.append(" ".join(words[inside[0]:inside[-1] + 1]) if inside else "")
            else:                       # the text's words do not line up with the G2P's: share them out evenly
                share = len(words) / max(1, len(piece.word_phones))
                texts.append(" ".join(words[round(inside[0] * share):round((inside[-1] + 1) * share)]) if inside else "")
        chunk_id = labels[0] if len(labels) == 1 else f"{labels[0]}+{labels[-1]}"
        return replace(pieces[-1].chunk, chunk_id=chunk_id, text=" ".join(t for t in texts if t))


def _next_target(engine: Engine, stream: _ChainStream, start: int, first: bool,
                 delivery: tuple[np.ndarray, np.ndarray], speed: float) -> tuple[int, Sentence, list[int]]:
    """The end of the next target, its sentence and durations."""
    shortest, longest = engine.min_target_frames, engine.max_target_frames
    voice, language = engine.voice_index(stream.chunks[0].voice), engine.language_index(stream.chunks[0].language)

    def measure(end: int) -> tuple[Sentence, list[int]]:
        sentence = stream.sentence(start, end)
        return sentence, engine.durations(sentence, voice=voice, language=language, delivery=delivery, speed=speed)

    window = FIRST_WINDOW_PHONES if first else WINDOW_PHONES
    while True:
        stream.extend(start + window + 2)
        last = min(len(stream.phones) - 1, start + window)
        stops = [(i, stream.cuts[i]) for i in sorted(stream.cuts) if start < i <= last]
        if not stops:                   # no silence within the window: run to the next one
            stops = [next((i, kind) for i, kind in sorted(stream.cuts.items()) if i > start)]
        if (first and window < WINDOW_PHONES and not stream.at_end(stops[-1][0])
                and all(kind not in SENTENCE_STOPS for _, kind in stops)):
            window *= 2                 # no sentence end in view yet: widen before measuring
            continue
        _, durations = measure(stops[-1][0])
        reach = np.cumsum(durations)
        frames = {i: int(reach[i - start]) for i, _ in stops}
        # The first target widens its window until a sentence end past the minimum is in view.
        if not first or window >= WINDOW_PHONES or stream.at_end(stops[-1][0]) or any(
                kind in SENTENCE_STOPS and frames[i] >= shortest for i, kind in stops):
            break
        window *= 2
    candidates = []
    for i, kind in stops:               # a paragraph or record ends the target once it is long enough
        candidates.append((i, kind))
        if kind == "end" or (kind == "paragraph" and frames[i] >= shortest):
            break
    while True:
        end = _choose(candidates, frames, first, shortest, longest)
        sentence, durations = measure(end)
        earlier = [c for c in candidates if c[0] < end]
        if sum(durations) <= longest or not earlier:
            return end, sentence, durations
        candidates, first = earlier, False     # over the maximum: the furthest earlier boundary that fits


def _choose(candidates: list[tuple[int, str]], frames: dict[int, int], first: bool, shortest: int, longest: int) -> int:
    fit = [c for c in candidates if frames[c[0]] <= longest] or candidates[:1]
    long_enough = [c for c in fit if frames[c[0]] >= shortest]
    stop = candidates[-1] if candidates[-1][1] in ("end", "paragraph") else None
    if stop and stop in fit and not first:
        return stop[0]
    marks = [c for c in long_enough if c[1] != "word"]
    if first:
        ends = [c for c in long_enough if c[1] in SENTENCE_STOPS]
        choice = ends[0] if ends else stop if stop and stop in fit else marks[0] if marks else (long_enough or fit)[0]
    else:
        full = [c for c in marks if frames[c[0]] >= FILL_SHARE * longest]
        sentences = [c for c in full if c[1] in ("sentence", "paragraph")]
        choice = (sentences or full or marks or long_enough or fit)[-1]
    if stop and choice != stop and frames[stop[0]] - frames[choice[0]] < shortest:   # never leave a short remainder
        if stop in fit:
            return stop[0]
        balanced = [c for c in long_enough if frames[stop[0]] - frames[c[0]] >= shortest]
        if balanced:
            choice = ([c for c in balanced if c[1] != "word"] or balanced)[-1]
    return choice[0]


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

