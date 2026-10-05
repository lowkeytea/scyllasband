"""Scylla's Band synthesis engine: one sentence at a time, with passage context and acoustic continuity.

A passage is spoken sentence by sentence. For each sentence:

1. its phones (from :mod:`scyllasband.g2p`) form the target of a before/target/after span whose context is up
   to 180 phones (silences excluded) of the neighbouring text;
2. the duration graph predicts frames for every span position; the target slice is rounded to integer frames;
3. per-frame phone ids and text events are built (:mod:`scyllasband.events`);
4. the flow integrates noise to latents in the smallest fitting fixed-shape bucket, conditioned on the span and
   on the last 96 latent frames already spoken (right-aligned prefix);
5. the vocoder decodes the sentence's latents with up to 48 frames of the preceding latents as left context, so
   consecutive sentences join as one continuous decode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import threading
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .events import frame_events, frames_to_durations, modifier_bits
from .g2p import SILENCE, G2PFrontend

GRAPH_CONTRACT = "scyllasband_measured_delivery_v2"
LATENT_HOP = 512


def balanced_context(before: Sequence[int], target: Sequence[int], after: Sequence[int], max_phones: int) -> tuple[list[int], list[int]]:
    """Span ids and segments (0 before, 1 target, 2 after), context split evenly with spare given to either side."""
    target = list(target)[:max_phones]
    if len(target) >= max_phones:
        return target, [1] * len(target)
    remaining = max_phones - len(target)
    before_keep = min(len(before), (remaining + 1) // 2)
    after_keep = min(len(after), remaining - before_keep)
    spare = remaining - before_keep - after_keep
    if spare > 0 and before_keep < len(before):
        extra = min(spare, len(before) - before_keep)
        before_keep += extra
        spare -= extra
    if spare > 0 and after_keep < len(after):
        after_keep += min(spare, len(after) - after_keep)
    kept_before = list(before)[-before_keep:] if before_keep else []
    kept_after = list(after)[:after_keep] if after_keep else []
    return kept_before + target + kept_after, [0] * len(kept_before) + [1] * len(target) + [2] * len(kept_after)


class OverlongError(ValueError):
    """The sentence needs more latent frames (or phones) than the largest bucket holds."""

    def __init__(self, message: str, *, frames: int | None = None, limit: int | None = None):
        super().__init__(message)
        self.frames, self.limit = frames, limit


@dataclass
class Sentence:
    """Phones of one sentence plus the context the engine needs around it."""
    phones: list[str]
    word_starts: list[int]
    before_ids: list[int] = field(default_factory=list)   # preceding passage phones, silences excluded
    after_ids: list[int] = field(default_factory=list)    # following passage phones, silences excluded


@dataclass
class SentenceResult:
    latents: np.ndarray            # [24, T]
    durations: list[int]
    bucket: int
    metadata: dict[str, Any]


class OnnxSession:
    def __init__(self, path: Path, *, threads: int | None = None, providers: Sequence[str] | None = None):
        import onnxruntime as ort
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = int(threads)
        self.session = ort.InferenceSession(str(path), options, providers=list(providers or ["CPUExecutionProvider"]))
        self.inputs = [i.name for i in self.session.get_inputs()]

    def run(self, feeds: Mapping[str, np.ndarray]) -> np.ndarray:
        return self.session.run(None, {name: feeds[name] for name in self.inputs})[0]


class Engine:
    def __init__(self, bundle_dir: str | Path, *, backend: str = "onnx", threads: int | None = None,
                 session_factory: Callable[[str, Mapping[str, Any]], Any] | None = None) -> None:
        self.bundle_dir = Path(bundle_dir)
        self.manifest = json.loads((self.bundle_dir / "manifest.json").read_text(encoding="utf-8"))
        controls = self.manifest["controls"]
        if controls.get("graph_input_contract") != GRAPH_CONTRACT:
            raise ValueError(f"Bundle graph contract {controls.get('graph_input_contract')!r} is not supported by this runtime "
                             f"(expected {GRAPH_CONTRACT!r}); download the current model.")
        self.backend, self.threads = backend, threads
        self.controls = controls
        assets = self.manifest["assets"]
        vocab = json.loads((self.bundle_dir / assets["phone_vocab"]).read_text(encoding="utf-8"))
        self.phone_to_id: dict[str, int] = {str(k): int(v) for k, v in dict(vocab.get("token_to_id", vocab)).items()}
        self.voice_to_id = _index(self.bundle_dir / assets["voice_index"])
        self.language_to_id = _index(self.bundle_dir / assets["language_index"])
        self.bits = modifier_bits(self.phone_to_id)
        span = controls["span_conditioning"]
        self.span_width = int(span["context_max_phones"])
        self.context_phones = int(span.get("context_phones_each_side", 180))
        self.prefix_frames = int(controls["prefix_conditioning"]["max_frames"])
        self.decode_context = int(controls.get("decoding", {}).get("context_frames", 48))
        self.latent_dim = int(self.manifest["audio"]["latent_dim"])
        buckets = controls["target_buckets"]["buckets"]
        self.buckets = sorted((int(b["latent_frames"]), str(b["vector_estimator"]), str(b["vocoder"])) for b in buckets)
        # Optional graphs that run every flow step in one call, for one sampler and step count: {latent_frames: component}.
        fused = controls.get("fused_flow") or {}
        self.fused_flow = (str(fused.get("sampler", "")).lower(), int(fused.get("steps", 0)))
        self.flow_graphs = {int(b["latent_frames"]): str(b["vector_flow"]) for b in buckets if b.get("vector_flow")}
        # Optional narrower G2P inputs, smallest first: a phrase runs at the smallest width that holds it.
        self.g2p_buckets = sorted((int(b["text_tokens"]), str(b["component"])) for b in controls.get("g2p_buckets") or [])
        self._session_factory = session_factory or self._onnx_session
        self._sessions: dict[str, Any] = {}
        self._lock = threading.RLock()
        self.g2p = G2PFrontend(self.bundle_dir, assets, controls, self.phone_to_id, infer=self._g2p)

    # --- sessions -----------------------------------------------------------------------------------------------------
    def _onnx_session(self, name: str, spec: Mapping[str, Any]) -> OnnxSession:
        return OnnxSession(self.bundle_dir / spec["artifacts"]["onnx"]["path"], threads=self.threads)

    def session(self, name: str) -> Any:
        with self._lock:
            if name not in self._sessions:
                self._sessions[name] = self._session_factory(name, self.manifest["components"][name])
            return self._sessions[name]

    def _run(self, name: str, **feeds: np.ndarray) -> np.ndarray:
        return self.session(name).run(feeds)

    def _g2p(self, text: np.ndarray) -> np.ndarray:
        """G2P logits for padded text ids [1, tokens]; padding is masked, so a narrower graph gives the same logits."""
        if self.g2p_buckets:
            pad = int(self.g2p.tokenizer.get("text_pad_index", 0))
            used = np.flatnonzero(np.asarray(text)[0] != pad)
            length = int(used[-1]) + 1 if used.size else 1
            for width, name in self.g2p_buckets:
                if width >= length:
                    return self._run(name, text=np.asarray(text)[:, :width])
        return self._run("g2p", text=text)

    @property
    def max_frames(self) -> int:
        return self.buckets[-1][0]

    def bucket(self, frames: int) -> tuple[int, str, str]:
        for bucket in self.buckets:
            if bucket[0] >= frames:
                return bucket
        raise OverlongError(f"Predicted {frames} latent frames, but this bundle supports at most {self.max_frames}. "
                            "Split the text into shorter sentences.", frames=frames, limit=self.max_frames)

    # --- text ---------------------------------------------------------------------------------------------------------
    def phonemize(self, text: str, *, language: str) -> dict[str, Any]:
        return self.g2p.phonemize(text, language=language)

    def context_ids(self, phones: Sequence[str]) -> list[int]:
        return [self.phone_to_id[p] for p in phones if p != SILENCE]

    def voice_index(self, voice: str) -> int:
        if voice not in self.voice_to_id:
            raise ValueError(f"Unknown voice {voice!r}; available: {', '.join(sorted(self.voice_to_id))}")
        return int(self.voice_to_id[voice])

    def language_index(self, language: str) -> int:
        if language not in self.language_to_id:
            raise ValueError(f"Unknown language {language!r}; available: {', '.join(sorted(self.language_to_id))}")
        return int(self.language_to_id[language])

    # --- synthesis ----------------------------------------------------------------------------------------------------
    def durations(self, sentence: Sentence, *, voice: int, language: int, delivery: tuple[np.ndarray, np.ndarray],
                  speed: float = 1.0) -> list[int]:
        target = [self.phone_to_id[p] for p in sentence.phones]
        if len(target) > self.span_width:
            raise OverlongError(f"Sentence has {len(target)} phones, but this bundle supports at most {self.span_width}.",
                                frames=len(target), limit=self.span_width)
        span, segments = balanced_context(sentence.before_ids[-self.context_phones:], target,
                                          sentence.after_ids[:self.context_phones], self.span_width)
        ids, segs, mask = _pad(span, self.span_width), _pad(segments, self.span_width), _pad([True] * len(span), self.span_width, False)
        values, present = delivery
        frames = self._run("duration_predictor", span_phone_ids=ids, span_segment_ids=segs, span_mask=mask,
                           voice_id=_scalar(voice), language_id=_scalar(language), delivery_values=values, delivery_present=present)
        offset = int(((segs == 0) & mask).sum())
        return frames_to_durations(frames[0, offset:offset + len(target)].tolist(), sentence.phones, scale=1.0 / float(speed))

    def synthesize_sentence(self, sentence: Sentence, *, voice: str, language: str, delivery: tuple[np.ndarray, np.ndarray],
                            prefix: np.ndarray | None = None, steps: int = 8, sampler: str = "heun", speed: float = 1.0,
                            noise_scale: float = 1.0, rng: np.random.Generator | None = None, noise: np.ndarray | None = None,
                            durations: list[int] | None = None) -> SentenceResult:
        voice_id, language_id = self.voice_index(voice), self.language_index(language)
        if durations is None:
            durations = self.durations(sentence, voice=voice_id, language=language_id, delivery=delivery, speed=speed)
        length = int(sum(durations))
        if length <= 0:
            raise ValueError("Predicted zero latent frames; cannot synthesize")
        frames, vector_name, _ = self.bucket(length)
        target = [self.phone_to_id[p] for p in sentence.phones]
        events = frame_events(sentence.phones, durations, sentence.word_starts,
                              [self._id_to_token(i) for i in sentence.after_ids[:self.context_phones]], self.phone_to_id, self.bits)
        span, segments = balanced_context(sentence.before_ids[-self.context_phones:], target, sentence.after_ids[:self.context_phones],
                                          self.span_width)
        hidden = self._run("vector_context_encoder", span_context_phone_ids=_pad(span, self.span_width),
                           span_context_segment_ids=_pad(segments, self.span_width),
                           span_context_mask=_pad([True] * len(span), self.span_width, False))
        prefix_latents = np.zeros((1, self.latent_dim, self.prefix_frames), np.float32)
        prefix_mask = np.zeros((1, self.prefix_frames), bool)
        if prefix is not None and prefix.shape[-1]:
            tail = np.asarray(prefix, np.float32).reshape(self.latent_dim, -1)[:, -self.prefix_frames:]
            prefix_latents[0, :, self.prefix_frames - tail.shape[-1]:] = tail
            prefix_mask[0, self.prefix_frames - tail.shape[-1]:] = True
        mask = _pad([True] * length, frames, False)
        feeds = dict(expanded_phone_ids=_pad(np.repeat(target, durations), frames), voice_id=_scalar(voice_id),
                     language_id=_scalar(language_id), delivery_values=delivery[0], delivery_present=delivery[1], latent_mask=mask,
                     span_context_hidden=hidden, prefix_latents=prefix_latents, prefix_mask=prefix_mask,
                     **{k: _pad(v, frames, 0) for k, v in events.items()})
        x = np.zeros((1, self.latent_dim, frames), np.float32)
        if noise is None:
            rng = rng if rng is not None else np.random.default_rng()
            noise = rng.standard_normal((self.latent_dim, length)).astype(np.float32) * float(noise_scale)
        x[0, :, :length] = noise
        heun = str(sampler).lower() == "heun"
        fused = self.flow_graphs.get(frames) if (str(sampler).lower(), int(steps)) == self.fused_flow else None
        if fused is not None:
            x = self._run(fused, noise=x, **feeds)
        for step in range(0 if fused is not None else int(steps)):
            a = self._run(vector_name, noise=x, time=np.array([step / steps], np.float32), **feeds)
            if heun:
                b = self._run(vector_name, noise=x + a / steps, time=np.array([(step + 1) / steps], np.float32), **feeds)
                x = x + (a + b) / (2 * steps)
            else:
                x = x + a / steps
            x = x * mask[:, None, :]
        metadata = dict(phones=sentence.phones, durations=durations, latent_frames=length, bucket=frames, steps=int(steps), sampler=sampler,
                        fused_flow=fused is not None,
                        prefix_frames=int(prefix_mask.sum()), context_before=min(len(sentence.before_ids), self.context_phones),
                        context_after=min(len(sentence.after_ids), self.context_phones))
        return SentenceResult(latents=x[0, :, :length].copy(), durations=durations, bucket=frames, metadata=metadata)

    def decode(self, latents: np.ndarray, *, voice: str, language: str, left: np.ndarray | None = None) -> np.ndarray:
        """Waveform for ``latents`` [24, T], decoded after up to ``decoding.context_frames`` frames of ``left`` context."""
        latents = np.asarray(latents, np.float32).reshape(self.latent_dim, -1)
        context = np.zeros((self.latent_dim, 0), np.float32) if left is None else np.asarray(left, np.float32).reshape(self.latent_dim, -1)
        # Left context gives way to the sentence itself when both do not fit the largest bucket.
        keep = min(self.decode_context, max(0, self.max_frames - 1 - latents.shape[-1]))
        context = context[:, context.shape[-1] - min(keep, context.shape[-1]):]
        joined = np.concatenate([context, latents], -1)
        # The vocoder returns frames * 512 - 256 samples, so keep one spare frame to emit the sentence's last frame in full;
        # a sentence that fills the largest bucket has its last 256 samples padded with silence instead.
        frames, _, vocoder_name = self.bucket(min(joined.shape[-1] + 1, self.max_frames))
        padded = np.zeros((1, self.latent_dim, frames), np.float32)
        padded[0, :, :joined.shape[-1]] = joined
        audio = self._run(vocoder_name, latents=padded, latent_mask=_pad([True] * joined.shape[-1], frames, False),
                          voice_id=_scalar(self.voice_index(voice)), language_id=_scalar(self.language_index(language)),
                          emotion_id=_scalar(0))
        audio = np.asarray(audio, np.float32).reshape(-1)
        start, length = context.shape[-1] * LATENT_HOP, latents.shape[-1] * LATENT_HOP
        audio = audio[start:start + length]
        return np.clip(np.pad(audio, (0, length - audio.size)), -1.0, 1.0)

    def _id_to_token(self, index: int) -> str:
        if not hasattr(self, "_tokens"):
            self._tokens = {v: k for k, v in self.phone_to_id.items()}
        return self._tokens[int(index)]


def _index(path: Path) -> dict[str, int]:
    """Rows of {"id", "index"}."""
    return {str(row["id"]): int(row["index"]) for row in json.loads(path.read_text(encoding="utf-8"))}


def _scalar(value: int) -> np.ndarray:
    return np.array([int(value)], np.int64)


def _pad(values: Sequence[Any] | np.ndarray, width: int, fill: Any = 0) -> np.ndarray:
    values = np.asarray(values)
    dtype = bool if values.dtype == bool else values.dtype if values.size else (bool if isinstance(fill, bool) else np.int64)
    if dtype not in (bool, np.float32) and np.issubdtype(dtype, np.integer):
        dtype = np.int64
    out = np.full((1, width), fill, dtype=dtype)
    out[0, :values.shape[0]] = values
    return out
