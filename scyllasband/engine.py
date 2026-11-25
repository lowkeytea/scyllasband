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

A v4 bundle (:mod:`scyllasband.contract`) adds singing: the flow also takes a delivery mode id (0 spoken, 1 sung, 2 rap,
3 ballad), a sung target brings its own melody from a score (:mod:`scyllasband.singing`), which replaces steps 2 and 4
(durations come from the score, the pitch predictor and its soft limit are skipped), and the bundle may carry a second
decoder (``controls.decoders``). On a v3 bundle the mode input is never fed and everything above is unchanged.

A v5 bundle adds each voice's identity vector as a graph input and a table of style rows (``controls.delivery_modes.ids``,
read from the bundle): the duration, pitch and flow graphs all take the target's row. Speech uses the rows the duration
and pitch graphs were trained on (``duration_styles_trained``: spoken, and rap, which is spoken with intensity and pace
unset); the other rows are sung from a score. The pitch predictor continues from the contour rendered so far, pauses
after punctuation carry the sounded-pause mark, and an opening "Uh-uh" keeps minimum lengths (:mod:`scyllasband.events`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import functools
import json
import os
from pathlib import Path
import re
import sys
import threading
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .contract import DELIVERY_MODE_INPUT, GRAPH_CONTRACTS, GRAPH_CONTRACTS_V5, GRAPH_CONTRACT_V4, IDENTITY_INPUT, STYLE_INPUT
from .events import frame_events, frames_to_durations, modifier_bits, opening_minimums
from .g2p_phrases import is_non_acoustic_phone_modifier
from .g2p import SILENCE, G2PFrontend

GRAPH_CONTRACT = "scyllasband_measured_controls_v3"
LATENT_HOP = 512
# The decoder every bundle has, as v3 manifests (which have no controls.decoders) describe it.
DEFAULT_DECODERS = {"vocos": dict(component="vocoder", samples_per_frame=LATENT_HOP, output_samples="latent_frames * 512 - 256")}
_OUTPUT_RULE = re.compile(r"^\s*latent_frames\s*\*\s*(\d+)\s*(?:-\s*(\d+))?\s*$")
# Target lengths v2 was trained on, in latent frames (about 1.4 to 9 seconds): shorter targets lose their prosody and
# longer ones their intelligibility. A bundle can state its own as controls.chunking.min/max_target_frames.
MIN_TARGET_FRAMES, MAX_TARGET_FRAMES = 64, 420
# v5: a target never holds fewer phones than this (stress and length marks and <tokens> not counted) while the passage
# has a neighbour to join; shorter than anything the flow trained on, a lone "Really." came out squeezed. A bundle can
# state its own as controls.chunking.min_target_phones.
MIN_TARGET_PHONES_V5 = 12
# A dash typed with no spaces around it ("a—dream") interrupts the word before it: its pause lasts at least this long.
TIGHT_DASH_PAUSE_SECONDS, LATENT_SAMPLE_RATE = 0.3, 24000


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
    min_pauses: list[int] = field(default_factory=list)   # silences held at least TIGHT_DASH_PAUSE_SECONDS (unspaced dashes)
    sentence_starts: list[int] = field(default_factory=lambda: [0])   # the silence in front of each sentence that begins here


@dataclass
class SentenceResult:
    latents: np.ndarray            # [24, T]
    durations: list[int]
    bucket: int
    metadata: dict[str, Any]
    contour: np.ndarray | None = None   # [2, T] the pitch contour as rendered (soft-limited, before any pitch shift); v5 chains it


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


# The graphs a target calls once per sampler evaluation: the flow's and the pitch predictor's estimators, in every bucket.
CUDA_GRAPH_COMPONENTS = ("vector_estimator", "pitch_estimator")
_OFF = ("0", "off", "false", "no")


def cuda_graph_default() -> bool:
    """Whether the estimators run as captured CUDA graphs on the CUDA provider: yes unless SCYLLASBAND_CUDA_GRAPH is 0."""
    return os.environ.get("SCYLLASBAND_CUDA_GRAPH", "").strip().lower() not in _OFF


class CudaGraphSession:
    """A fixed-shape graph on the CUDA provider, replayed as one captured CUDA graph: each input lives in a device buffer
    that is overwritten in place when its values change, the output in another, so a call launches the whole graph at once
    instead of one kernel at a time. The values are those of :class:`OnnxSession` up to the GPU's own rounding.

    The graph must have fixed shapes and run wholly on the CUDA provider; ONNX Runtime refuses to create the session
    otherwise. If a call fails, the graph goes back to an :class:`OnnxSession` for good, with one line on stderr."""

    def __init__(self, path: Path, *, threads: int | None = None, providers: Sequence[str]):
        import onnxruntime as ort
        self.path, self.threads, self.providers = path, threads, list(providers)
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = int(threads)
        _preload_cuda()
        capture = ("CUDAExecutionProvider", {"enable_cuda_graph": "1"})
        self.session = ort.InferenceSession(str(path), options,
                                            providers=[capture if name == capture[0] else name for name in self.providers])
        if self.session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError("the CUDA provider did not load")
        outputs = self.session.get_outputs()
        tensors = list(self.session.get_inputs()) + outputs[:1]
        if len(outputs) != 1 or any(not all(isinstance(dim, int) and dim > 0 for dim in tensor.shape) for tensor in tensors):
            raise RuntimeError("its shapes are not fixed")
        if any(tensor.type not in _ONNX_DTYPES for tensor in tensors):
            raise RuntimeError("it has a tensor type the device buffers do not hold")
        self.inputs = [i.name for i in self.session.get_inputs()]
        # Host copies of what the device buffers hold: an input is uploaded again only when its bytes differ.
        self.host = {i.name: np.zeros(i.shape, _ONNX_DTYPES[i.type]) for i in self.session.get_inputs()}
        self.device = {name: ort.OrtValue.ortvalue_from_numpy(value, "cuda", 0) for name, value in self.host.items()}
        self.output = ort.OrtValue.ortvalue_from_shape_and_type(list(outputs[0].shape), _ONNX_DTYPES[outputs[0].type], "cuda", 0)
        self.binding = self.session.io_binding()
        for name, value in self.device.items():
            self.binding.bind_ortvalue_input(name, value)
        self.binding.bind_ortvalue_output(outputs[0].name, self.output)
        self.plain: OnnxSession | None = None
        self._lock = threading.Lock()   # one set of buffers: one call at a time

    def run(self, feeds: Mapping[str, np.ndarray]) -> np.ndarray:
        with self._lock:
            if self.plain is None:
                try:
                    return self._replay(feeds)
                except Exception as exc:
                    _cuda_graph_off(self.path, exc)
                    self.session = self.binding = self.device = self.output = None   # frees the buffers and the captured graph
                    self.plain = OnnxSession(self.path, threads=self.threads, providers=self.providers)
            return self.plain.run(feeds)

    def _replay(self, feeds: Mapping[str, np.ndarray]) -> np.ndarray:
        for name in self.inputs:
            value, held = np.asarray(feeds[name]), self.host[name]
            if value.dtype != held.dtype or value.shape != held.shape:
                raise ValueError(f"input {name} is {value.dtype}{list(value.shape)}, not {held.dtype}{list(held.shape)}")
            if not np.array_equal(value.view(_BITS[held.dtype.itemsize]), held.view(_BITS[held.dtype.itemsize])):
                np.copyto(held, value)
                self.device[name].update_inplace(held)
        self.session.run_with_iobinding(self.binding)
        return self.output.numpy()


_ONNX_DTYPES = {"tensor(float)": np.float32, "tensor(int64)": np.int64, "tensor(int32)": np.int32, "tensor(bool)": np.bool_}
_BITS = {1: np.uint8, 4: np.uint32, 8: np.uint64}


def _cuda_graph_off(path: Path, reason: BaseException) -> None:
    why = (str(reason).strip() or type(reason).__name__).splitlines()[0]
    print(f"{path.name}: running without CUDA graph capture ({why}); SCYLLASBAND_CUDA_GRAPH=0 turns it off for every graph.",
          file=sys.stderr)


def cuda_graph_session(path: Path, *, threads: int | None = None, providers: Sequence[str]) -> "CudaGraphSession | OnnxSession":
    """``path`` as a captured CUDA graph, or as a plain session (with one line on stderr) when this ONNX Runtime or this
    graph cannot be captured."""
    try:
        return CudaGraphSession(path, threads=threads, providers=providers)
    except Exception as exc:
        _cuda_graph_off(path, exc)
        return OnnxSession(path, threads=threads, providers=providers)


class Engine:
    def __init__(self, bundle_dir: str | Path, *, backend: str = "onnx", threads: int | None = None,
                 session_factory: Callable[[str, Mapping[str, Any]], Any] | None = None,
                 providers: Callable[[str], Sequence[str]] | None = None, cuda_graph: bool | None = None) -> None:
        self.bundle_dir = Path(bundle_dir)
        self.manifest = json.loads((self.bundle_dir / "manifest.json").read_text(encoding="utf-8"))
        controls = self.manifest["controls"]
        if controls.get("graph_input_contract") not in GRAPH_CONTRACTS:
            raise ValueError(f"Bundle graph contract {controls.get('graph_input_contract')!r} is not supported by this runtime "
                             f"(expected one of {', '.join(GRAPH_CONTRACTS)}); download the current model.")
        self.backend, self.threads, self.providers = backend, threads, providers
        # ONNX Runtime on CUDA: the estimators run as captured CUDA graphs (None: unless SCYLLASBAND_CUDA_GRAPH is 0).
        self.cuda_graph = cuda_graph_default() if cuda_graph is None else bool(cuda_graph)
        self.controls = controls
        self.graph_contract = str(controls["graph_input_contract"])
        # v5: each voice's fixed identity vector is a graph input, the duration and pitch graphs take the style row too, the
        # pitch predictor takes the contour rendered so far as a prefix, and pauses after punctuation carry the
        # sounded-pause mark by default (controls.pause_marks).
        self.v5 = self.graph_contract in GRAPH_CONTRACTS_V5
        # v4 and v5 flows take the delivery mode id on every call (the spoken row for speech); v3 flows have no such input.
        self.feeds_delivery_mode = self.graph_contract == GRAPH_CONTRACT_V4 or self.v5
        self.identity: dict[int, np.ndarray] = {}
        self.pitch_prefix_frames = 0
        self.pause_mark: str | None = None
        modes = controls.get("delivery_modes") or {}
        self.delivery_modes: dict[str, int] = ({str(k): int(v) for k, v in (modes.get("ids") or {}).items()} if modes.get("enabled") else {})
        self.singing: dict[str, Any] = dict(modes.get("singing") or {}) if modes.get("enabled") else {}
        self.spoken_mode = int(self.delivery_modes.get("spoken", 0))
        # The styles speech may ask for: the rows the duration and pitch graphs were trained on (v5); spoken alone before.
        self.speech_styles: tuple[str, ...] = ("spoken",)
        if self.v5:
            if "spoken" not in self.delivery_modes:
                raise ValueError("This v5 bundle gives no style rows (controls.delivery_modes.ids with a 'spoken' row)")
            trained = [str(style) for style in modes.get("duration_styles_trained") or ()]
            unknown = [style for style in trained if style not in self.delivery_modes]
            if "spoken" not in trained or unknown:
                raise ValueError("This v5 bundle must list the styles its duration graph was trained on "
                                 f"(controls.delivery_modes.duration_styles_trained); got {trained}"
                                 + (f", with no row for {', '.join(unknown)}" if unknown else ""))
            pitched = modes.get("pitch_styles_trained")   # the pitch graph's own list, where the bundle gives one
            self.speech_styles = tuple(style for style in trained if pitched is None or style in pitched)
        # Speech styles trained with the intensity and pace slots absent (the score-free rap rows).
        self.uncontrolled_styles = frozenset(str(style) for style in (modes.get("controls_absent") or {}).get("styles") or ())
        self.decoders = {str(k): dict(v) for k, v in (controls.get("decoders") or DEFAULT_DECODERS).items()}
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
        if self.v5:
            table = json.loads((self.bundle_dir / assets["voice_identity"]).read_text(encoding="utf-8"))
            dim = int(table["dim"])
            for name, vector in table["voices"].items():
                if name in self.voice_to_id:
                    self.identity[int(self.voice_to_id[name])] = np.asarray(vector, np.float32).reshape(1, dim)
            prefix = self.pitch.get("prefix") or {}
            self.pitch_prefix_frames = int(prefix.get("max_frames", 0)) if prefix.get("enabled") else 0
            marks = controls.get("pause_marks") or {}
            if marks.get("default") not in ("all", "none"):
                raise ValueError("This v5 bundle does not say how pauses are marked (controls.pause_marks.default: all or none)")
            if marks["default"] == "all":
                if marks.get("token") not in self.phone_to_id:
                    raise ValueError(f"This bundle's pause mark {marks.get('token')!r} is not in its phone vocabulary")
                self.pause_mark = str(marks["token"])
        span = controls["span_conditioning"]
        self.span_width = int(span["context_max_phones"])
        self.context_phones = int(span.get("context_phones_each_side", 180))
        self.prefix_frames = int(controls["prefix_conditioning"]["max_frames"])
        self.decode_context = int(controls.get("decoding", {}).get("context_frames", 48))
        self.latent_dim = int(self.manifest["audio"]["latent_dim"])
        buckets = controls["target_buckets"]["buckets"]
        self.buckets = sorted((int(b["latent_frames"]), str(b["vector_estimator"]), str(b["vocoder"])) for b in buckets)
        self.pitch_graphs = {int(b["latent_frames"]): str(b["pitch_estimator"]) for b in buckets}
        # Decoder graphs per bucket, by decoder name: a bucket names a decoder's component under that component's base
        # name ("vocoder", "vocoder_bigvgan"); an optional decoder may be missing from some or all buckets.
        self.decoder_graphs = {name: {int(b["latent_frames"]): str(b[key]) for b in buckets if b.get(key)}
                               for name, spec in self.decoders.items() for key in [str(spec.get("component") or "vocoder")]}
        self.decoder_windows = self._decoder_windows()
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
        self.min_target_phones = int(chunking.get("min_target_phones", MIN_TARGET_PHONES_V5 if self.v5 else 0))
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
    def _onnx_session(self, name: str, spec: Mapping[str, Any]) -> "OnnxSession | CudaGraphSession":
        path, providers = self.bundle_dir / spec["artifacts"]["onnx"]["path"], self.providers(name) if self.providers else None
        if self.cuda_graph and providers and providers[0] == "CUDAExecutionProvider" and name.startswith(CUDA_GRAPH_COMPONENTS):
            return cuda_graph_session(path, threads=self.threads, providers=providers)
        return OnnxSession(path, threads=self.threads, providers=providers)

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

    def decoder_bucket(self, decoder: str, frames: int) -> tuple[int, str]:
        """The smallest bucket of at least ``frames`` that has a graph for ``decoder``, and that graph."""
        graphs = self.decoder_graphs.get(decoder)
        if graphs is None:
            raise ValueError(f"Unknown decoder {decoder!r}; this bundle declares {', '.join(sorted(self.decoders))}")
        if not graphs:
            raise ValueError(f"This bundle has no {decoder} decoder graphs")
        for size in sorted(graphs):
            if size >= frames:
                return size, graphs[size]
        limit = max(graphs)
        raise OverlongError(f"{frames} latent frames to decode, but the {decoder} decoder supports at most {limit}.",
                            frames=frames, limit=limit)

    def _decoder_windows(self) -> dict[str, tuple[int, int]]:
        """Decoders that decode in fixed windows through one bucket graph, as libscyllasband does: {name: (the window's
        bucket in frames, frames of context either side of each cut)}, from ``controls.decoders.<name>.window`` {frames,
        context} or SCYLLASBAND_DECODER_WINDOW ("bigvgan=256:16"; "bigvgan=0" turns it off). A windowed decoder decodes any
        length, so a bundle may carry it at the window's bucket only."""
        requested = {name: (int((spec.get("window") or {}).get("frames", 0)), int((spec.get("window") or {}).get("context", 0)))
                     for name, spec in self.decoders.items()}
        for entry in filter(None, os.environ.get("SCYLLASBAND_DECODER_WINDOW", "").split(",")):
            name, _, value = entry.partition("=")
            frames, _, context = value.partition(":")
            if name not in self.decoders or not frames.isdigit() or (context and not context.isdigit()):
                raise ValueError(f"SCYLLASBAND_DECODER_WINDOW entry {entry!r} is not decoder=frames[:context] for a decoder of this bundle")
            requested[name] = (int(frames), int(context or 0))
        windows = {}
        for name, (frames, context) in requested.items():
            if frames <= 0:
                continue
            if self.decoder_rule(name)[1] != 0:
                raise ValueError(f"The {name} decoder cannot decode in windows (its output is not whole frames)")
            graphs = self.decoder_graphs.get(name) or {}
            if graphs:
                frames = self.decoder_bucket(name, min(frames, max(graphs)))[0]
            if context < 0 or 2 * context >= frames:
                raise ValueError(f"The {name} decoder's window context must leave frames to keep: {frames} frames, {context} context")
            windows[name] = (frames, context)
        return windows

    def delivery_mode_id(self, mode: str | int | None) -> int:
        """A delivery mode's id (``controls.delivery_modes.ids``); the spoken row (0 when the bundle has no table) when none
        is asked."""
        if mode is None or mode == "spoken":
            return self.spoken_mode
        if isinstance(mode, str):
            if mode not in self.delivery_modes:
                raise ValueError(f"Unknown delivery mode {mode!r}; this bundle has {', '.join(sorted(self.delivery_modes)) or 'spoken only'}")
            return int(self.delivery_modes[mode])
        if int(mode) != 0 and not self.feeds_delivery_mode:
            raise ValueError("This bundle has no delivery modes (a v3 bundle); singing needs a v4 bundle")
        return int(mode)

    def speech_style(self, style: str | None) -> tuple[int, bool]:
        """(the style's row, whether intensity and pace are set) for speech in ``style``. Only the styles the duration and
        pitch graphs were trained on can be spoken (``controls.delivery_modes.duration_styles_trained`` and
        ``pitch_styles_trained``; v5): "spoken", and score-free styles such as "rap", whose training rows carried neither
        intensity nor pace (``controls.delivery_modes.controls_absent.styles``)."""
        name = "spoken" if style is None else str(style).strip().lower()
        if name == "spoken":
            return self.spoken_mode, True
        if not self.v5:
            raise ValueError(f"This bundle speaks in one style; --style {name} needs a v5 bundle")
        if name not in self.speech_styles:
            sung = " (it is a sung style: give a score to sing)" if name in self.delivery_modes else ""
            raise ValueError(f"Style {name!r} cannot be spoken{sung}: this bundle's duration and pitch models were trained on "
                             f"{', '.join(self.speech_styles)}")
        return int(self.delivery_modes[name]), name not in self.uncontrolled_styles

    # --- text ---------------------------------------------------------------------------------------------------------
    def phonemize(self, text: str, *, language: str, nonverbal: bool = True) -> dict[str, Any]:
        return self.g2p.phonemize(text, language=language, nonverbal=nonverbal)

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

    def rendered_contour(self, contour: np.ndarray, voice: str) -> np.ndarray:
        """The contour [2, T] as the flow heard it, in the predictor's own units: the pitch line soft-limited inside the
        voice's edges (no pitch shift: a shift is added after the predictor, so chaining it would add it again), the
        voicing channel as sampled. This is what the next target's predictor continues from (``controls.pitch.prefix``)."""
        p = self.pitch
        st = contour[0].astype(np.float64) * float(p["scale"]) + float(p["center"])
        low, high = (float(e) for e in p["edges"][voice])
        knee = min(float(p["knee"]), (high - low) / 4)
        top, bottom = high - knee, low + knee
        st = np.where(st > top, top + knee * np.tanh((st - top) / knee), st)
        st = np.where(st < bottom, bottom - knee * np.tanh((bottom - st) / knee), st)
        return np.stack([(st - float(p["center"])) / float(p["scale"]), contour[1].astype(np.float64)]).astype(np.float32)

    def identity_feeds(self, voice_row: int) -> dict[str, np.ndarray]:
        """{identity_reference: the voice's fixed vector} on a v5 bundle, {} otherwise."""
        if not self.v5:
            return {}
        if int(voice_row) not in self.identity:
            raise ValueError(f"This bundle has no identity vector for voice row {voice_row}")
        return {IDENTITY_INPUT: self.identity[int(voice_row)]}

    def style_feeds(self, style: int | None) -> dict[str, np.ndarray]:
        """{style_id: the row} for the duration graph of a v5 bundle (the spoken row when none is given), {} otherwise."""
        return {STYLE_INPUT: _scalar(self.spoken_mode if style is None else style)} if self.v5 else {}

    # --- synthesis ----------------------------------------------------------------------------------------------------
    def durations(self, sentence: Sentence, *, voice: int, language: int, delivery: tuple[np.ndarray, np.ndarray],
                  speed: float = 1.0, style: int | None = None) -> list[int]:
        """Frames per phone of the target. ``style`` (v5): the style row fed to the duration graph; the spoken row by default."""
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
                           voice_id=_scalar(voice), language_id=_scalar(language), delivery_values=values, delivery_present=present,
                           **self.identity_feeds(voice), **self.style_feeds(style))
        offset = int(((segs == 0) & mask).sum())
        durations = frames_to_durations(frames[0, offset:offset + len(target)].tolist(), sentence.phones, scale=1.0 / float(speed),
                                        pause_scale=self.pause_scale(delivery), min_pauses=sentence.min_pauses,
                                        min_pause_frames=TIGHT_DASH_PAUSE_SECONDS * LATENT_SAMPLE_RATE / LATENT_HOP)
        return opening_minimums(sentence.phones, durations, sentence.sentence_starts) if self.v5 else durations

    def synthesize_sentence(self, sentence: Sentence, *, voice: str, language: str, delivery: Sequence[Any],
                            prefix: np.ndarray | None = None, steps: int = 8, sampler: str = "heun", speed: float = 1.0,
                            noise_scale: float = 1.0, rng: np.random.Generator | None = None, noise: np.ndarray | None = None,
                            durations: list[int] | None = None, voice_row: int | None = None, pitch_noise: np.ndarray | None = None,
                            pitch_steps: int | None = None, pitch_sampler: str | None = None, melody: np.ndarray | None = None,
                            delivery_mode: int | str | None = None, guidance: float = 1.0, pitch_prefix: np.ndarray | None = None,
                            guidance_until: float | None = None) -> SentenceResult:
        """``delivery``: (values, present[, pitch shift in semitones]) from delivery_inputs. ``voice_row``: the duration / flow
        voice row; defaults to the voice index. The pitch predictor draws its noise from ``rng`` before the flow does.

        ``melody``: a ready melody input [T, 3] (T = the sum of ``durations``) in the flow's layout, as a score gives it
        (:mod:`scyllasband.singing`); it replaces the pitch predictor's contour and the voice's soft limit entirely, so the
        pitch graphs do not run and the pitch shift is ignored. ``delivery_mode``: the mode id fed to a v4 flow (0 spoken;
        a v3 bundle takes only 0); on a v5 bundle the style row, fed to the duration and pitch graphs as well (the spoken
        row by default). ``pitch_prefix`` (v5): the contours rendered so far in this chain [2, n]
        (``SentenceResult.contour`` of the targets before, joined); the predictor continues from its tail. ``guidance`` > 1 contrasts the melody and mode against both switched off
        (v = v_off + guidance * (v_on - v_off), classifier-free guidance; off = no melody and the spoken row); 1 runs the
        flow once per evaluation. ``guidance_until``: flow time (0 to 1) from which evaluations run unguided; None guides all.
        """
        voice_id = self.voice_index(voice) if voice_row is None else int(voice_row)
        language_id = self.language_index(language)
        mode_id = self.delivery_mode_id(delivery_mode)
        if durations is None:
            durations = self.durations(sentence, voice=voice_id, language=language_id, delivery=delivery, speed=speed, style=mode_id)
        length = int(sum(durations))
        if length <= 0:
            raise ValueError("Predicted zero latent frames; cannot synthesize")
        if melody is not None:
            melody = np.asarray(melody, np.float32)
            if melody.shape != (length, 3):
                raise ValueError(f"melody must be [{length}, 3] (one row per latent frame); got {melody.shape}")
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
        if melody is None:
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
                      **{k: _pad(v, frames, 0) for k, v in events.items()}, **self.identity_feeds(voice_id))
        rng = rng if rng is not None else np.random.default_rng()
        expanded_melody = np.zeros((1, frames, 3), np.float32)
        rendered = None
        pitch_prefix_used = 0
        if melody is None:
            if pitch_noise is None:
                pitch_noise = rng.standard_normal((2, length)).astype(np.float32)
            pitch_feeds = dict(common, span_context_hidden=pitch_hidden)
            if self.v5:
                width = self.pitch_prefix_frames
                contour_prefix = np.zeros((1, 2, width), np.float32)
                contour_mask = np.zeros((1, width), bool)
                if pitch_prefix is not None and width:
                    tail = np.asarray(pitch_prefix, np.float32).reshape(2, -1)[:, -width:]
                    if tail.shape[-1] and (tail[1] > 0.0).any():   # a tail with no voiced frame says nothing about pitch
                        contour_prefix[0, :, width - tail.shape[-1]:] = tail
                        contour_mask[0, width - tail.shape[-1]:] = True
                        pitch_prefix_used = int(tail.shape[-1])
                pitch_feeds.update(prefix_latents=contour_prefix, prefix_mask=contour_mask)
                pitch_feeds[DELIVERY_MODE_INPUT] = _scalar(mode_id)
            contour = self._integrate(self.pitch_graphs[frames], np.asarray(pitch_noise, np.float32).reshape(2, length), frames, mask,
                                      pitch_feeds, int(pitch_steps or self.pitch_steps),
                                      str(pitch_sampler or self.pitch_sampler).lower() == "heun")
            expanded_melody[0, :length] = self.melody(contour, voice, shift)
            rendered = self.rendered_contour(contour, voice)
        else:
            expanded_melody[0, :length] = melody
        feeds = dict(common, span_context_hidden=hidden, prefix_latents=prefix_latents, prefix_mask=prefix_mask, expanded_melody=expanded_melody)
        if self.feeds_delivery_mode:
            feeds[DELIVERY_MODE_INPUT] = _scalar(mode_id)
        x = np.zeros((1, self.latent_dim, frames), np.float32)
        if noise is None:
            noise = rng.standard_normal((self.latent_dim, length)).astype(np.float32) * float(noise_scale)
        x[0, :, :length] = noise
        heun = str(sampler).lower() == "heun"
        no_prefix = not prefix_mask.any()
        if no_prefix and frames in self.first_graphs:
            vector_name = self.first_graphs[frames]
        flows = self.first_flow_graphs if no_prefix and frames in self.first_flow_graphs else self.flow_graphs
        # Guidance contrasts the conditioned velocity with the melody and mode switched off; it needs the stepped graph.
        guided = float(guidance) != 1.0 and (melody is not None or mode_id != self.spoken_mode)
        off = dict(feeds, expanded_melody=np.zeros_like(expanded_melody)) if guided else None
        if off is not None and self.feeds_delivery_mode:
            off[DELIVERY_MODE_INPUT] = _scalar(self.spoken_mode)
        fused = flows.get(frames) if (str(sampler).lower(), int(steps)) == self.fused_flow and not guided else None

        def velocity(current: np.ndarray, t: float) -> np.ndarray:
            on = self._run(vector_name, noise=current, time=np.array([t], np.float32), **feeds)
            if off is None or (guidance_until is not None and t >= float(guidance_until) - 1e-6):
                return on
            base = self._run(vector_name, noise=current, time=np.array([t], np.float32), **off)
            return base + float(guidance) * (on - base)

        if fused is not None:
            x = self._run(fused, noise=x, **feeds)
        for step in range(0 if fused is not None else int(steps)):
            a = velocity(x, step / steps)
            if heun:
                b = velocity(x + a / steps, (step + 1) / steps)
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
        if self.feeds_delivery_mode or melody is not None:
            metadata.update(delivery_mode=mode_id, melody_source="pitch_predictor" if melody is None else "score", guidance=float(guidance))
        if self.v5:
            metadata.update(pitch_prefix_frames=pitch_prefix_used)
        return SentenceResult(latents=x[0, :, :length].copy(), durations=durations, bucket=frames, metadata=metadata, contour=rendered)

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

    def decoder_rule(self, decoder: str) -> tuple[int, int]:
        """(samples per latent frame, samples the decoder's output falls short of frames * samples_per_frame) from
        ``controls.decoders[decoder].output_samples`` ("latent_frames * 512 - 256"; Vocos) or "latent_frames * 512" (BigVGAN)."""
        spec = self.decoders.get(decoder)
        if spec is None:
            raise ValueError(f"Unknown decoder {decoder!r}; this bundle declares {', '.join(sorted(self.decoders))}")
        per_frame = int(spec.get("samples_per_frame", LATENT_HOP))
        match = _OUTPUT_RULE.match(str(spec.get("output_samples") or ""))
        if match:
            per_frame, short = int(match.group(1)), int(match.group(2) or 0)
        else:
            short = 0
        return per_frame, short

    def decode(self, latents: np.ndarray, *, voice: str, language: str, left: np.ndarray | None = None, decoder: str = "vocos") -> np.ndarray:
        """Waveform for ``latents`` [24, T], decoded after up to ``decoding.context_frames`` frames of ``left`` context.
        ``decoder`` names one of ``controls.decoders`` (vocos, the default every bundle has; bigvgan where exported)."""
        latents = np.asarray(latents, np.float32).reshape(self.latent_dim, -1)
        context = np.zeros((self.latent_dim, 0), np.float32) if left is None else np.asarray(left, np.float32).reshape(self.latent_dim, -1)
        per_frame, short = self.decoder_rule(decoder)
        if decoder in self.decoder_windows:
            return self._decode_windowed(latents, context, decoder=decoder, voice=voice, language=language)
        graphs = self.decoder_graphs.get(decoder) or {}
        limit = max(graphs) if graphs else self.max_frames
        if latents.shape[-1] > limit:   # (BigVGAN may stop below the flow's largest bucket)
            raise OverlongError(f"{latents.shape[-1]} latent frames to decode, but the {decoder} decoder supports at most {limit}.",
                                frames=latents.shape[-1], limit=limit)
        # Left context gives way to the sentence itself when both do not fit the largest bucket.
        keep = min(self.decode_context, max(0, limit - 1 - latents.shape[-1]))
        context = context[:, context.shape[-1] - min(keep, context.shape[-1]):]
        joined = np.concatenate([context, latents], -1)
        # Vocos returns frames * 512 - 256 samples, so keep one spare frame to emit the sentence's last frame in full; a
        # sentence that fills the largest bucket has its last 256 samples padded with silence instead. BigVGAN returns
        # frames * 512 samples and needs no spare frame.
        spare = 1 if short > 0 else 0
        frames, vocoder_name = self.decoder_bucket(decoder, min(joined.shape[-1] + spare, limit))
        padded = np.zeros((1, self.latent_dim, frames), np.float32)
        padded[0, :, :joined.shape[-1]] = joined
        audio = self._run(vocoder_name, latents=padded, latent_mask=_pad([True] * joined.shape[-1], frames, False),
                          voice_id=_scalar(self.voice_index(voice)), language_id=_scalar(self.language_index(language)),
                          emotion_id=_scalar(0))
        audio = np.asarray(audio, np.float32).reshape(-1)
        start, length = context.shape[-1] * per_frame, latents.shape[-1] * per_frame
        audio = audio[start:start + length]
        return np.clip(np.pad(audio, (0, length - audio.size)), -1.0, 1.0)

    def _decode_windowed(self, latents: np.ndarray, context: np.ndarray, *, decoder: str, voice: str, language: str) -> np.ndarray:
        """decode() through the window's one bucket graph (libscyllasband's Engine::decode_windowed): the sequence (up to
        ``decoding.context_frames`` of left context, then the target) is decoded ``window`` frames at a time, each call
        keeping the frames at least ``context`` from a cut inside the sequence; the last window ends at the sequence end and
        moves left to decode a full window where it can."""
        window_frames, overlap = self.decoder_windows[decoder]
        window, graph = self.decoder_bucket(decoder, window_frames)
        per_frame, _ = self.decoder_rule(decoder)
        left = min(context.shape[-1], self.decode_context)
        sequence = np.concatenate([context[:, context.shape[-1] - left:], latents], -1)
        total = sequence.shape[-1]
        out = np.zeros(latents.shape[-1] * per_frame, np.float32)
        ids = dict(voice_id=_scalar(self.voice_index(voice)), language_id=_scalar(self.language_index(language)), emotion_id=_scalar(0))
        start = left
        while start < total:
            first = max(0, min(start - overlap, total - window))
            last = min(total, first + window)
            keep = total if last == total else last - overlap
            padded = np.zeros((1, self.latent_dim, window), np.float32)
            padded[0, :, :last - first] = sequence[:, first:last]
            audio = np.asarray(self._run(graph, latents=padded, latent_mask=_pad([True] * (last - first), window, False), **ids),
                               np.float32).reshape(-1)
            piece = audio[(start - first) * per_frame:(keep - first) * per_frame]
            out[(start - left) * per_frame:(start - left) * per_frame + piece.size] = piece
            start = keep
        return np.clip(out, -1.0, 1.0)

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
