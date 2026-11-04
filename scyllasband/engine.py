"""Scylla's Band synthesis engine: one target at a time, with passage context and acoustic continuity.

Long text is spoken as targets of 64 to 420 latent frames cut by :mod:`scyllasband.streaming`. For each target:

1. its phones (from :mod:`scyllasband.g2p`) form the target of a before/target/after span whose context is up
   to 180 phones (silences excluded) of the neighbouring text;
2. the duration graph predicts frames for every span position; the target slice is rounded to integer frames, with
   pauses scaled by the requested pace (``controls.pace_pauses``);
3. per-frame phone ids and text events are built (:mod:`scyllasband.events`);
4. the pitch predictor integrates noise to a pitch and voicing contour for those frames; the requested pitch shift is
   added and the voice's soft limit applied (``controls.pitch``), giving the flow's melody input;
5. the flow integrates noise to latents in the smallest fitting fixed-shape bucket, conditioned on the span, the melody
   and the last 96 latent frames already spoken (right-aligned prefix);
6. the vocoder decodes the target's latents with up to 48 frames of the preceding latents as left context, so
   consecutive targets join as one continuous decode.

Delivery is intensity and pace (graph slots 0 and 1, in the voice's standard deviations / z_scale) plus a pitch shift in
semitones. A target with fewer than ``min_spoken_phones`` spoken phones gets neither slot, as in training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import functools
import json
from pathlib import Path
import threading
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .events import frame_events, frames_to_durations, modifier_bits
from .g2p_phrases import is_non_acoustic_phone_modifier
from .g2p import SILENCE, G2PFrontend

GRAPH_CONTRACT = "scyllasband_measured_controls_v3"
LATENT_HOP = 512
# Target lengths v2 was trained on, in latent frames (about 1.4 to 9 seconds): shorter targets lose their prosody and
# longer ones their intelligibility. A bundle can state its own as controls.chunking.min/max_target_frames.
MIN_TARGET_FRAMES, MAX_TARGET_FRAMES = 64, 420


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


CUDA_HINT = ("ONNX Runtime with CUDA is needed for --compute-units gpu: install the runtime with the cuda extra "
             "(pip install -e \".[cuda]\") in an environment without the CPU-only onnxruntime package")


def onnx_providers(compute_units: str | Mapping[str, str] | None) -> Callable[[str], list[str]]:
    """ONNX Runtime execution providers for each graph: ``cpu`` (the default), ``gpu`` (NVIDIA GPUs through CUDA, also
    written ``cuda``), or a default plus per-graph choices, e.g. ``gpu,g2p=cpu``."""
    from .coreml import parse_compute_units
    value = None if compute_units in (None, "", "auto") else compute_units
    choose = parse_compute_units(value, default="cpu", units=("cpu", "gpu", "cuda"))
    return lambda graph: ["CUDAExecutionProvider", "CPUExecutionProvider"] if choose(graph) in ("gpu", "cuda") else ["CPUExecutionProvider"]


@functools.cache
def _preload_cuda() -> None:
    """Load CUDA and cuDNN from the nvidia pip packages that onnxruntime-gpu[cuda,cudnn] installs (ONNX Runtime 1.21+)."""
    import onnxruntime as ort
    try:
        ort.preload_dlls()
    except Exception:
        pass


class OnnxSession:
    def __init__(self, path: Path, *, threads: int | None = None, providers: Sequence[str] | None = None):
        import onnxruntime as ort
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = int(threads)
        providers = list(providers or ["CPUExecutionProvider"])
        if "CUDAExecutionProvider" in providers:
            _preload_cuda()
        self.session = ort.InferenceSession(str(path), options, providers=providers)
        if providers[0] not in self.session.get_providers():
            # ONNX Runtime falls back to the CPU when the CUDA provider cannot load (for example without cuDNN).
            raise RuntimeError(f"{path.name}: ONNX Runtime could not use {providers[0]}. {CUDA_HINT}.")
        self.inputs = [i.name for i in self.session.get_inputs()]

    def run(self, feeds: Mapping[str, np.ndarray]) -> np.ndarray:
        return self.session.run(None, {name: feeds[name] for name in self.inputs})[0]


class Engine:
    def __init__(self, bundle_dir: str | Path, *, backend: str = "onnx", threads: int | None = None,
                 session_factory: Callable[[str, Mapping[str, Any]], Any] | None = None,
                 providers: Callable[[str], Sequence[str]] | None = None) -> None:
        self.bundle_dir = Path(bundle_dir)
        self.manifest = json.loads((self.bundle_dir / "manifest.json").read_text(encoding="utf-8"))
        controls = self.manifest["controls"]
        if controls.get("graph_input_contract") != GRAPH_CONTRACT:
            raise ValueError(f"Bundle graph contract {controls.get('graph_input_contract')!r} is not supported by this runtime "
                             f"(expected {GRAPH_CONTRACT!r}); download the current model.")
        self.backend, self.threads, self.providers = backend, threads, providers
        self.controls = controls
        assets = self.manifest["assets"]
        vocab = json.loads((self.bundle_dir / assets["phone_vocab"]).read_text(encoding="utf-8"))
        self.phone_to_id: dict[str, int] = {str(k): int(v) for k, v in dict(vocab.get("token_to_id", vocab)).items()}
        self.voice_to_id = _index(self.bundle_dir / assets["voice_index"])
        self.language_to_id = _index(self.bundle_dir / assets["language_index"])
        delivery = controls["delivery"]
        self.z_scale, self.z_limit = float(delivery.get("z_scale", 5.0)), float(delivery.get("z_limit", 2.5))
        self.min_spoken_phones = int(delivery.get("min_spoken_phones", 4))
        self.control_ranges = {str(k): tuple(v) for k, v in (delivery.get("ranges") or {}).items()}
        self.voice_defaults = {str(k): dict(v) for k, v in (delivery.get("defaults") or {}).items()}
        self.pace_pause_per_sd = float((controls.get("pace_pauses") or {}).get("per_sd", 0.0))
        self.pitch = dict(controls["pitch"])
        self.pitch_sampler = str(self.pitch.get("sampler", {}).get("default", "euler")).lower()
        self.pitch_steps = int(self.pitch.get("steps", {}).get("default", 4))
        self.pitch_context = str(self.pitch.get("context_encoder", "pitch_context_encoder"))
        self.bits = modifier_bits(self.phone_to_id)
        span = controls["span_conditioning"]
        self.span_width = int(span["context_max_phones"])
        self.context_phones = int(span.get("context_phones_each_side", 180))
        self.prefix_frames = int(controls["prefix_conditioning"]["max_frames"])
        self.decode_context = int(controls.get("decoding", {}).get("context_frames", 48))
        self.latent_dim = int(self.manifest["audio"]["latent_dim"])
        buckets = controls["target_buckets"]["buckets"]
        self.buckets = sorted((int(b["latent_frames"]), str(b["vector_estimator"]), str(b["vocoder"])) for b in buckets)
        self.pitch_graphs = {int(b["latent_frames"]): str(b["pitch_estimator"]) for b in buckets}
        # Optional graphs that run every flow step in one call, for one sampler and step count: {latent_frames: component}.
        fused = controls.get("fused_flow") or {}
        self.fused_flow = (str(fused.get("sampler", "")).lower(), int(fused.get("steps", 0)))
        self.flow_graphs = {int(b["latent_frames"]): str(b["vector_flow"]) for b in buckets if b.get("vector_flow")}
        # Optional graphs without the prefix tokens, for targets with no prefix (a chain's first): masked prefix keys
        # contribute nothing, so they return what the full graphs return there, for less work.
        self.first_graphs = {int(b["latent_frames"]): str(b["vector_estimator_first"]) for b in buckets if b.get("vector_estimator_first")}
        self.first_flow_graphs = {int(b["latent_frames"]): str(b["vector_flow_first"]) for b in buckets if b.get("vector_flow_first")}
        chunking = controls.get("chunking") or {}
        self.min_target_frames = int(chunking.get("min_target_frames", MIN_TARGET_FRAMES))
        self.max_target_frames = min(int(chunking.get("max_target_frames", MAX_TARGET_FRAMES)), self.buckets[-1][0])
        # Optional narrower G2P inputs, smallest first: a phrase runs at the smallest width that holds it.
        self.g2p_buckets = sorted((int(b["text_tokens"]), str(b["component"])) for b in controls.get("g2p_buckets") or [])
        # Optional narrower span graphs: padding is masked and positions are absolute, so the narrowest that holds the span
        # gives the same values. [(phones, duration component, context encoder component)], smallest first.
        self.span_buckets = sorted((int(b["phones"]), str(b.get("duration_predictor") or ""), str(b.get("vector_context_encoder") or ""),
                                    str(b.get("pitch_context_encoder") or "")) for b in controls.get("span_buckets") or [])
        self._session_factory = session_factory or self._onnx_session
        self._sessions: dict[str, Any] = {}
        self._lock = threading.RLock()
        self.g2p = G2PFrontend(self.bundle_dir, assets, controls, self.phone_to_id, infer=self._g2p)

    # --- sessions -----------------------------------------------------------------------------------------------------
    def _onnx_session(self, name: str, spec: Mapping[str, Any]) -> OnnxSession:
        return OnnxSession(self.bundle_dir / spec["artifacts"]["onnx"]["path"], threads=self.threads,
                           providers=self.providers(name) if self.providers else None)

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

    def _span_graph(self, phones: int, which: int, default: str) -> tuple[int, str]:
        for width, *names in self.span_buckets:
            if phones <= width < self.span_width and names[which]:
                return width, names[which]
        return self.span_width, default

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

    def delivery_inputs(self, voice: str, request: Mapping[str, Any] | str | None) -> tuple[np.ndarray, np.ndarray, int, dict[str, Any]]:
        """(delivery values, present, the voice row, the resolved {intensity, pace, pitch}) for a planned request; whatever the
        request leaves out takes the voice's default."""
        from .delivery import delivery_tensors, resolve_delivery, with_defaults
        index = self.voice_index(voice)
        if isinstance(request, Mapping) and request.get("kind") == "style":
            raise ValueError("This model takes intensity, pace and pitch; emotion styles belonged to the v2 models")
        resolved = with_defaults(resolve_delivery(request, self.control_ranges), self.voice_defaults.get(voice))
        values, present = delivery_tensors(resolved, z_scale=self.z_scale, z_limit=self.z_limit)
        return values, present, index, resolved

    def target_delivery(self, phones: Sequence[str], delivery: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
        """The graph's delivery for a target: none at all when it has fewer than ``min_spoken_phones`` spoken phones (training
        never set intensity or pace on such short views)."""
        values, present = np.asarray(delivery[0], np.float32), np.asarray(delivery[1], bool)
        spoken = sum(1 for p in phones if not str(p).startswith("<") and not is_non_acoustic_phone_modifier(p))
        return (values, np.zeros_like(present)) if spoken < self.min_spoken_phones else (values, present)

    def pause_scale(self, delivery: Sequence[Any]) -> float:
        """Pause frames move with pace: exp(per_sd * pace), pace in standard deviations (graph slot 1 x z_scale)."""
        values, present = np.asarray(delivery[0], np.float32), np.asarray(delivery[1], bool)
        pace = float(values[0, 1]) * self.z_scale if present[0, 1] else 0.0
        return float(np.exp(self.pace_pause_per_sd * pace))

    def melody(self, contour: np.ndarray, voice: str, shift: float = 0.0) -> np.ndarray:
        """Pitch contour [2, T] -> the flow's melody input [T, 3]: pitch shifted, soft-limited inside the voice's edges, on the
        melody scale where voiced; voicing probability; voiced flag (``controls.pitch.rule``)."""
        p = self.pitch
        st = contour[0].astype(np.float64) * float(p["scale"]) + float(p["center"]) + float(shift)
        low, high = (float(e) for e in p["edges"][voice])
        knee = min(float(p["knee"]), (high - low) / 4)
        top, bottom = high - knee, low + knee
        st = np.where(st > top, top + knee * np.tanh((st - top) / knee), st)
        st = np.where(st < bottom, bottom - knee * np.tanh((bottom - st) / knee), st)
        probability = np.clip((contour[1].astype(np.float64) + 1.0) / 2.0, 0.0, 1.0)
        voiced = (probability > float(p["voiced_threshold"])).astype(np.float64)
        pitch = (st - float(p["melody_center"])) / float(p["melody_scale"]) * voiced
        return np.stack([pitch, probability, voiced], -1).astype(np.float32)

    # --- synthesis ----------------------------------------------------------------------------------------------------
    def durations(self, sentence: Sentence, *, voice: int, language: int, delivery: tuple[np.ndarray, np.ndarray],
                  speed: float = 1.0) -> list[int]:
        target = [self.phone_to_id[p] for p in sentence.phones]
        if len(target) > self.span_width:
            raise OverlongError(f"Sentence has {len(target)} phones, but this bundle supports at most {self.span_width}.",
                                frames=len(target), limit=self.span_width)
        span, segments = balanced_context(sentence.before_ids[-self.context_phones:], target,
                                          sentence.after_ids[:self.context_phones], self.span_width)
        width, graph = self._span_graph(len(span), 0, "duration_predictor")
        ids, segs, mask = _pad(span, width), _pad(segments, width), _pad([True] * len(span), width, False)
        values, present = self.target_delivery(sentence.phones, delivery)
        frames = self._run(graph, span_phone_ids=ids, span_segment_ids=segs, span_mask=mask,
                           voice_id=_scalar(voice), language_id=_scalar(language), delivery_values=values, delivery_present=present)
        offset = int(((segs == 0) & mask).sum())
        return frames_to_durations(frames[0, offset:offset + len(target)].tolist(), sentence.phones, scale=1.0 / float(speed),
                                   pause_scale=self.pause_scale(delivery))

    def synthesize_sentence(self, sentence: Sentence, *, voice: str, language: str, delivery: Sequence[Any],
                            prefix: np.ndarray | None = None, steps: int = 8, sampler: str = "heun", speed: float = 1.0,
                            noise_scale: float = 1.0, rng: np.random.Generator | None = None, noise: np.ndarray | None = None,
                            durations: list[int] | None = None, voice_row: int | None = None, pitch_noise: np.ndarray | None = None,
                            pitch_steps: int | None = None, pitch_sampler: str | None = None) -> SentenceResult:
        """``delivery``: (values, present[, pitch shift in semitones]) from delivery_inputs. ``voice_row``: the duration / flow
        voice row; defaults to the voice index. The pitch predictor draws its noise from ``rng`` before the flow does."""
        voice_id = self.voice_index(voice) if voice_row is None else int(voice_row)
        language_id = self.language_index(language)
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
        width, graph = self._span_graph(len(span), 1, "vector_context_encoder")
        span_feeds = dict(span_context_phone_ids=_pad(span, width), span_context_segment_ids=_pad(segments, width),
                          span_context_mask=_pad([True] * len(span), width, False))
        hidden = self._run(graph, **span_feeds)
        pitch_width, pitch_graph = self._span_graph(len(span), 2, self.pitch_context)
        pitch_hidden = (self._run(pitch_graph, **span_feeds) if pitch_width == width else
                        self._run(pitch_graph, span_context_phone_ids=_pad(span, pitch_width), span_context_segment_ids=_pad(segments, pitch_width),
                                  span_context_mask=_pad([True] * len(span), pitch_width, False)))
        values, present = self.target_delivery(sentence.phones, delivery)
        shift = float(delivery[2]) if len(delivery) > 2 and delivery[2] is not None else 0.0
        prefix_latents = np.zeros((1, self.latent_dim, self.prefix_frames), np.float32)
        prefix_mask = np.zeros((1, self.prefix_frames), bool)
        if prefix is not None and prefix.shape[-1]:
            tail = np.asarray(prefix, np.float32).reshape(self.latent_dim, -1)[:, -self.prefix_frames:]
            prefix_latents[0, :, self.prefix_frames - tail.shape[-1]:] = tail
            prefix_mask[0, self.prefix_frames - tail.shape[-1]:] = True
        mask = _pad([True] * length, frames, False)
        common = dict(expanded_phone_ids=_pad(np.repeat(target, durations), frames), voice_id=_scalar(voice_id),
                      language_id=_scalar(language_id), delivery_values=values, delivery_present=present, latent_mask=mask,
                      **{k: _pad(v, frames, 0) for k, v in events.items()})
        rng = rng if rng is not None else np.random.default_rng()
        if pitch_noise is None:
            pitch_noise = rng.standard_normal((2, length)).astype(np.float32)
        contour = self._integrate(self.pitch_graphs[frames], np.asarray(pitch_noise, np.float32).reshape(2, length), frames, mask,
                                  dict(common, span_context_hidden=pitch_hidden), int(pitch_steps or self.pitch_steps),
                                  str(pitch_sampler or self.pitch_sampler).lower() == "heun")
        melody = np.zeros((1, frames, 3), np.float32)
        melody[0, :length] = self.melody(contour, voice, shift)
        feeds = dict(common, span_context_hidden=hidden, prefix_latents=prefix_latents, prefix_mask=prefix_mask, expanded_melody=melody)
        x = np.zeros((1, self.latent_dim, frames), np.float32)
        if noise is None:
            noise = rng.standard_normal((self.latent_dim, length)).astype(np.float32) * float(noise_scale)
        x[0, :, :length] = noise
        heun = str(sampler).lower() == "heun"
        no_prefix = not prefix_mask.any()
        if no_prefix and frames in self.first_graphs:
            vector_name = self.first_graphs[frames]
        flows = self.first_flow_graphs if no_prefix and frames in self.first_flow_graphs else self.flow_graphs
        fused = flows.get(frames) if (str(sampler).lower(), int(steps)) == self.fused_flow else None
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
                        pitch_steps=int(pitch_steps or self.pitch_steps), pitch_sampler=str(pitch_sampler or self.pitch_sampler),
                        pitch_shift=shift, delivery_set=bool(present.any()),
                        fused_flow=fused is not None, flow_graph=fused if fused is not None else vector_name,
                        prefix_frames=int(prefix_mask.sum()), context_before=min(len(sentence.before_ids), self.context_phones),
                        context_after=min(len(sentence.after_ids), self.context_phones))
        return SentenceResult(latents=x[0, :, :length].copy(), durations=durations, bucket=frames, metadata=metadata)

    def _integrate(self, graph: str, noise: np.ndarray, frames: int, mask: np.ndarray, feeds: Mapping[str, np.ndarray], steps: int,
                   heun: bool) -> np.ndarray:
        """The pitch predictor's contour [2, T] for ``noise`` [2, T] in a ``frames`` bucket, Euler or Heun on a uniform grid."""
        length = noise.shape[-1]
        x = np.zeros((1, noise.shape[0], frames), np.float32)
        x[0, :, :length] = noise
        for step in range(steps):
            a = self._run(graph, noise=x, time=np.array([step / steps], np.float32), **feeds)
            if heun:
                b = self._run(graph, noise=x + a / steps, time=np.array([(step + 1) / steps], np.float32), **feeds)
                x = x + (a + b) / (2 * steps)
            else:
                x = x + a / steps
            x = x * mask[:, None, :]
        return x[0, :, :length]

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
