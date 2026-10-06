#!/usr/bin/env python3
"""Native libscyllasband vs the Python reference runtime on a real bundle.

Checks, for the build's own backend (ONNX Runtime, LiteRT, or Core ML / Core AI for Apple builds, on the same compute unit
as the Python reference):
- text frontend: spoken-text normalization, sentence split, punctuation segments, Unicode lower/NFC tables;
- G2P: identical phones, word starts and word-boundary candidates;
- per target the Python streaming loop speaks (with its context and the previous target's latents as prefix):
  identical durations, latents within --latent-tolerance relative error given identical noise, decoded audio SNR
  above --min-snr;
- long-form: identical plans; synthesis with a fixed seed runs end to end; with the reference's noise injected, the
  native loop speaks the same targets (ids, texts, phones, durations, context and prefix sizes, seeds) and the whole
  waveform matches the reference.

Usage: PYTHONPATH=<repo> python parity_harness.py --library libscyllasband_native.so --bundle <bundle> [--report out.json]
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

TEXTS = (
    ("exclamatory", "You thought the clock had run out, but the referee was still watching the ball! There's one more play! We're still in this!"),
    ("trailing", "I kept the letter for years... I never opened it. Maybe I was afraid of what it would say..."),
    ("staccato", "You. Are. Not. It. Not this time, and not ever again."),
    ("question", "Did you really leave the gate open all night? What were you thinking?"),
    ("statement", "Please put the small blue box beside the kitchen window, then close the door behind you."),
)
# Long-form only, as paragraphs of their own: short sentences the streaming loop packs together, one sentence longer
# than the model was trained on, which it cuts at clauses, and a passage without punctuation, which it cuts between words.
LONG_FORM_EXTRA = (
    "Hey. Hey! You, with the headphones. Come closer... the others don't know you're here yet.",
    "When the caravan finally reached the old stone bridge at the edge of the valley, where the river widened into a slow "
    "brown mirror and the herons stood like forgotten statues among the reeds, the travellers set down their packs and "
    "argued for an hour about whether to cross before dark or wait for the ferryman who had promised, three days earlier "
    "and with considerable ceremony, to meet them there at noon.",
    "the old men sat by the water all afternoon and talked about the weather and the price of grain and the long winter "
    "that was coming and the children who had gone away to the city and never written home and nobody listened to them "
    "at all but they kept on talking anyway because talking was what they had left and the river did not mind",
)
# Per long-form target, the metadata both loops report.
TARGET_KEYS = ("phones", "durations", "latent_frames", "context_before", "context_after", "prefix_frames", "seed")
VOICES = ("scylla", "ink", "max")
SEED = 2027

FRONTEND_CORPUS = (
    "Meet me at 9 a.m. Then we leave.", "It ends at 5 p.m.", "The U.S. Army and the U.K. Navy met.", "Dr. Smith met Mrs. Jones on Main Dr. today.",
    "I paid $3.50, then £12 and 40€ — or 1.234,56 EUR?", "On Jan. 5, 2024 we had 25% more; by 3/4/2025 it was 1/2.",
    "Call me at 10:30pm or 7:05 AM.", "She came 1st, he was 22nd, and they were 103rd.", "Email jane.doe+tts@example.co.uk or ping @scylla.",
    "Ok... OK?! ok!!", "Well -- that's it - right?", "e-mail, X-ray, T-shirt and A - B.", "The year 2024-03-07 was fine.",
    "Café, naïve, café and Ångström.", "It's 3.14159 or 2,718.28 or 1,000,000.", "Thirty-three point five, 33.5, and 0.5.",
    "“Quoted,” she said… ‘really’?", "[laughs] Okay [sighs] fine.", "A—B–C − D", "x = 5 and 20°",
)
FRONTEND_LANGUAGES = ("en_us", "en_gb", "es", "it", "fr", "de", "vi")
FRONTEND_EXTRA = {
    "es": ("Tengo 1.234 euros y 3,5 kilos; el 2º día, 15/03/2024.",),
    "fr": ("J'ai 81 ans, 1.000.000 et 3,75 € le 1er mai.",),
    "de": ("Am 3. Mai 2024 kostet es 21,50 € oder 1.234 Euro.",),
    "it": ("Il 2º posto costa 1.234,50 € e 99%.",),
    "vi": ("Tôi có 21 quyển sách và 105 cây bút, 1.000.000 đồng.",),
}


class Native:
    def __init__(self, path: str):
        lib = ctypes.CDLL(path)
        self.lib = lib
        c = ctypes
        lib.scyllasband_backend.restype = c.c_char_p
        lib.scyllasband_last_error.restype = c.c_char_p
        lib.scyllasband_voices_json.restype = c.c_char_p
        lib.scyllasband_voices_json.argtypes = [c.c_void_p]
        lib.scyllasband_runtime_create.argtypes = [c.c_void_p, c.POINTER(c.c_void_p)]
        lib.scyllasband_runtime_destroy.argtypes = [c.c_void_p]
        lib.scyllasband_test_free.argtypes = [c.c_void_p]
        for name in ("scyllasband_test_normalize",):
            getattr(lib, name).argtypes = [c.c_char_p, c.c_char_p, c.POINTER(c.c_void_p)]
        lib.scyllasband_test_split_sentences.argtypes = [c.c_char_p, c.POINTER(c.c_void_p)]
        lib.scyllasband_test_punctuated_segments.argtypes = [c.c_char_p, c.c_int32, c.POINTER(c.c_void_p)]
        lib.scyllasband_test_unicode.argtypes = [c.c_char_p, c.c_char_p, c.POINTER(c.c_void_p)]
        lib.scyllasband_test_phonemize.argtypes = [c.c_void_p, c.c_char_p, c.c_char_p, c.POINTER(c.c_void_p)]
        lib.scyllasband_test_synthesize_sentence.argtypes = [
            c.c_void_p, c.c_char_p, c.POINTER(c.c_float), c.c_int64, c.POINTER(c.c_float), c.c_int64, c.POINTER(c.c_void_p),
            c.POINTER(c.c_void_p), c.POINTER(c.c_int64), c.POINTER(c.c_void_p), c.POINTER(c.c_int64)]
        lib.scyllasband_plan_json.argtypes = [c.c_void_p, c.c_void_p, c.POINTER(c.c_void_p)]
        lib.scyllasband_synthesize.argtypes = [c.c_void_p, c.c_void_p, c.c_void_p]
        lib.scyllasband_audio_free.argtypes = [c.c_void_p]
        lib.scyllasband_request_init.argtypes = [c.c_void_p]
        self.noise_type = c.CFUNCTYPE(None, c.c_int32, c.c_uint64, c.c_int32, c.c_int32, c.c_int64, c.POINTER(c.c_float), c.c_void_p)
        lib.scyllasband_test_set_noise.argtypes = [c.c_void_p, self.noise_type, c.c_void_p]
        self.runtime = None

    @property
    def backend(self) -> str:
        return self.lib.scyllasband_backend().decode()

    def error(self) -> str:
        return (self.lib.scyllasband_last_error() or b"").decode()

    def _take_text(self, pointer: ctypes.c_void_p) -> str:
        value = ctypes.cast(pointer, ctypes.c_char_p).value.decode("utf-8")
        self.lib.scyllasband_test_free(pointer)
        return value

    def _check(self, status: int, what: str) -> None:
        if status != 0:
            raise RuntimeError(f"{what} failed ({status}): {self.error()}")

    def open(self, bundle: str, accelerator: int = 0) -> None:
        class Options(ctypes.Structure):
            _fields_ = [("bundle_dir", ctypes.c_char_p), ("threads", ctypes.c_int32), ("accelerator", ctypes.c_int)]
        options = Options(bundle.encode(), 0, accelerator)
        runtime = ctypes.c_void_p()
        self._check(self.lib.scyllasband_runtime_create(ctypes.byref(options), ctypes.byref(runtime)), "runtime_create")
        self.runtime = runtime

    def close(self) -> None:
        if self.runtime is not None:
            self.lib.scyllasband_runtime_destroy(self.runtime)
            self.runtime = None

    def normalize(self, text: str, language: str) -> str:
        out = ctypes.c_void_p()
        self._check(self.lib.scyllasband_test_normalize(text.encode(), language.encode(), ctypes.byref(out)), "normalize")
        return self._take_text(out)

    def split_sentences(self, text: str) -> list[str]:
        out = ctypes.c_void_p()
        self._check(self.lib.scyllasband_test_split_sentences(text.encode(), ctypes.byref(out)), "split_sentences")
        return json.loads(self._take_text(out))

    def segments(self, text: str, max_chars: int) -> list:
        out = ctypes.c_void_p()
        self._check(self.lib.scyllasband_test_punctuated_segments(text.encode(), max_chars, ctypes.byref(out)), "segments")
        return [(segment, tokens) for segment, tokens in json.loads(self._take_text(out))]

    def unicode(self, text: str, operation: str) -> str:
        out = ctypes.c_void_p()
        self._check(self.lib.scyllasband_test_unicode(text.encode("utf-8", "surrogatepass"), operation.encode(), ctypes.byref(out)), "unicode")
        return self._take_text(out)

    def phonemize(self, text: str, language: str) -> dict:
        out = ctypes.c_void_p()
        self._check(self.lib.scyllasband_test_phonemize(self.runtime, text.encode(), language.encode(), ctypes.byref(out)), "phonemize")
        return json.loads(self._take_text(out))

    def sentence(self, request: dict, noise: np.ndarray | None, prefix: np.ndarray | None):
        out_json, latents, audio = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
        latent_count, sample_count = ctypes.c_int64(), ctypes.c_int64()
        fp = ctypes.POINTER(ctypes.c_float)
        noise_c = np.ascontiguousarray(noise, np.float32) if noise is not None else None
        prefix_c = np.ascontiguousarray(prefix, np.float32) if prefix is not None and prefix.shape[-1] else None
        status = self.lib.scyllasband_test_synthesize_sentence(
            self.runtime, json.dumps(request).encode(),
            noise_c.ctypes.data_as(fp) if noise_c is not None else None, 0 if noise_c is None else noise_c.shape[-1],
            prefix_c.ctypes.data_as(fp) if prefix_c is not None else None, 0 if prefix_c is None else prefix_c.shape[-1],
            ctypes.byref(out_json), ctypes.byref(latents), ctypes.byref(latent_count), ctypes.byref(audio), ctypes.byref(sample_count))
        self._check(status, "synthesize_sentence")
        meta = json.loads(self._take_text(out_json))
        if noise is None:
            return meta, None, None
        lat = np.ctypeslib.as_array(ctypes.cast(latents, fp), shape=(latent_count.value,)).copy()
        wav = np.ctypeslib.as_array(ctypes.cast(audio, fp), shape=(sample_count.value,)).copy()
        self.lib.scyllasband_test_free(latents)
        self.lib.scyllasband_test_free(audio)
        return meta, lat.reshape(noise.shape[0], -1), wav

    def _request(self, text: str, voice: str, seed: int | None):
        class Request(ctypes.Structure):
            _fields_ = [("text", ctypes.c_char_p), ("voice_id", ctypes.c_char_p), ("language", ctypes.c_char_p), ("delivery", ctypes.c_char_p),
                        ("speed", ctypes.c_float), ("steps", ctypes.c_int32), ("sampler", ctypes.c_int), ("seed", ctypes.c_uint64),
                        ("has_seed", ctypes.c_int32), ("temperature", ctypes.c_float), ("normalize_text", ctypes.c_int32)]
        request = Request()
        self.lib.scyllasband_request_init(ctypes.byref(request))
        request.text, request.voice_id = text.encode(), voice.encode()
        if seed is not None:
            request.seed, request.has_seed = seed, 1
        return request

    def plan(self, text: str, voice: str) -> dict:
        out = ctypes.c_void_p()
        request = self._request(text, voice, None)
        self._check(self.lib.scyllasband_plan_json(self.runtime, ctypes.byref(request), ctypes.byref(out)), "plan_json")
        value = ctypes.cast(out, ctypes.c_char_p).value.decode()
        self.lib.scyllasband_test_free(out)
        return json.loads(value)

    def synthesize(self, text: str, voice: str, seed: int, noise=None):
        class Audio(ctypes.Structure):
            _fields_ = [("samples", ctypes.POINTER(ctypes.c_float)), ("sample_count", ctypes.c_int64), ("sample_rate", ctypes.c_int32),
                        ("metadata_json", ctypes.c_char_p)]
        callback = self.noise_type(noise) if noise is not None else self.noise_type()
        self.lib.scyllasband_test_set_noise(self.runtime, callback, None)
        request, audio = self._request(text, voice, seed), Audio()
        try:
            self._check(self.lib.scyllasband_synthesize(self.runtime, ctypes.byref(request), ctypes.byref(audio)), "synthesize")
            samples = np.ctypeslib.as_array(audio.samples, shape=(audio.sample_count,)).copy() if audio.sample_count else np.zeros(0, np.float32)
            meta = json.loads(audio.metadata_json.decode())
        finally:
            self.lib.scyllasband_audio_free(ctypes.byref(audio))
            self.lib.scyllasband_test_set_noise(self.runtime, self.noise_type(), None)
        return samples, meta


def snr_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    n = min(reference.size, candidate.size)
    if reference.size != candidate.size:
        return float("-inf")
    error = float(np.sum((reference[:n].astype(np.float64) - candidate[:n]) ** 2))
    signal = float(np.sum(reference[:n].astype(np.float64) ** 2))
    return float("inf") if error == 0 else 10 * math.log10(max(signal, 1e-30) / error)


def rel_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference.astype(np.float64)))
    return float(np.linalg.norm(reference.astype(np.float64) - candidate)) / max(denominator, 1e-30)


ACCELERATORS = {"cpu": 0, "gpu": 1, "ane": 3}


def reference_runtime(bundle: str, backend: str, compute_units: str = "cpu"):
    from scyllasband.engine import Engine
    from scyllasband.runtime import ScyllasBandRuntime
    if backend == "apple":   # the bundle names Core ML or Core AI; run it on the native side's compute unit
        return ScyllasBandRuntime.from_bundle(Path(bundle), compute_units=compute_units)
    factory = None
    if backend == "litert":
        from scyllasband.litert import litert_session_factory
        factory = litert_session_factory(Path(bundle))
    engine = Engine(bundle, backend=backend, session_factory=factory)
    return ScyllasBandRuntime(engine, bundle_dir=Path(bundle), backend=backend)


def check_frontend(native: Native, report: dict, failures: list[str]) -> None:
    from scyllasband.g2p import punctuated_segments
    from scyllasband.planner import split_sentences
    from scyllasband.text_normalizer import normalize_spoken_text
    mismatches = []
    cases = 0
    for language in FRONTEND_LANGUAGES:
        corpus = list(FRONTEND_CORPUS) + [t for _, t in TEXTS] + list(FRONTEND_EXTRA.get(language, ()))
        for text in corpus:
            cases += 1
            expected, got = normalize_spoken_text(text, language=language), native.normalize(text, language)
            if expected != got:
                mismatches.append(dict(kind="normalize", language=language, text=text, python=expected, native=got))
            for source in (text, expected):
                if split_sentences(source) != native.split_sentences(source):
                    mismatches.append(dict(kind="split_sentences", text=source))
                for max_chars in (140, 12):
                    want = [(segment, list(tokens)) for segment, tokens in punctuated_segments(source, max_chars=max_chars)]
                    if want != native.segments(source, max_chars):
                        mismatches.append(dict(kind="segments", text=source, max_chars=max_chars))
    # Unicode tables: every scalar value through lower() and NFC.
    scalar = lambda cp: cp and not 0xD800 <= cp <= 0xDFFF  # noqa: E731 (NUL ends a C string)
    chars = "".join(chr(cp) for cp in range(0x110000) if scalar(cp))
    unicode_mismatch = 0
    step = 4096
    for start in range(0, len(chars), step):
        block = chars[start:start + step]
        if native.unicode(block, "lower") != block.lower():
            unicode_mismatch += 1
    import random
    rng = random.Random(0)
    pool = ["\u03a3", "\u0391", "\u03c3", "'", "\u0301", "\u00ad", ".", " ", "1", "a", "\u02b0", "\u200d", ":", "\u1fbc", "\u01c5"]
    for _ in range(2000):  # context-dependent lowercasing (final sigma)
        text = "".join(rng.choice(pool) for _ in range(rng.randint(1, 8)))
        if native.unicode(text, "lower") != text.lower():
            unicode_mismatch += 1
    import unicodedata
    samples = ["é", "Å", "Å", "ẛ̣", "ą́", "q̣̇", "각", "각",
               "Việt Nam", "Việt Nam", "Ą́", "Ǻ"]
    samples += ["".join(chr(cp) for cp in range(base, base + 256) if scalar(cp)) for base in range(0, 0x30000, 256)]
    nfc_mismatch = sum(native.unicode(s, "nfc") != unicodedata.normalize("NFC", s) for s in samples)
    decomposed = [unicodedata.normalize("NFD", "".join(chr(cp) for cp in range(base, base + 256) if scalar(cp))) for base in range(0, 0x30000, 256)]
    nfc_mismatch += sum(native.unicode(s, "nfc") != unicodedata.normalize("NFC", s) for s in decomposed)
    report["frontend"] = dict(cases=cases, mismatches=mismatches[:20], mismatch_count=len(mismatches), unicode_lower_block_mismatches=unicode_mismatch,
                              nfc_mismatches=nfc_mismatch)
    if mismatches or unicode_mismatch or nfc_mismatch:
        failures.append(f"frontend: {len(mismatches)} mismatches, lower blocks {unicode_mismatch}, nfc {nfc_mismatch}")


def python_targets(engine, plan, speed: float = 1.0):
    """The targets the Python streaming loop speaks for ``plan``, without synthesizing them:
    (chunk, sentence, durations, delivery, first of its chain)."""
    from scyllasband.delivery import delivery_tensors
    from scyllasband.streaming import _chains, _ChainStream, _next_target
    spoken = 0
    for chain in _chains(plan.chunks):
        stream = _ChainStream(engine, chain)
        values, present, _ = delivery_tensors(chain[0].delivery)
        start = 0
        while True:
            end, sentence, durations = _next_target(engine, stream, start, spoken == 0, (values, present), speed)
            yield stream.target_chunk(start, end), sentence, durations, (values, present), start == 0
            spoken += 1
            if stream.at_end(end):
                break
            start = end


def compare_targets(native: Native, engine, plan, voice: str, label: str, failures: list[str], args) -> list[dict]:
    """Native durations, latents and audio for each Python target, given the same sentence, noise and prefix."""
    from scyllasband.delivery import delivery_spec
    language = plan.chunks[0].language
    rows = []
    tail = None
    for index, (chunk, sentence, python_durations, delivery, chain_start) in enumerate(python_targets(engine, plan)):
        tail = None if chain_start else tail
        request = dict(phones=sentence.phones, word_starts=sentence.word_starts, before_ids=sentence.before_ids, after_ids=sentence.after_ids,
                       voice=voice, language=language, delivery=delivery_spec(chunk.delivery), steps=args.steps, sampler="heun", speed=1.0)
        native_durations = native.sentence(request, None, None)[0]["durations"]
        length = int(sum(python_durations))
        noise = np.random.default_rng(SEED + 31 * index).standard_normal((engine.latent_dim, length)).astype(np.float32)
        result = engine.synthesize_sentence(sentence, voice=voice, language=language, delivery=delivery, prefix=tail, steps=args.steps,
                                            sampler="heun", noise=noise, durations=python_durations)
        python_audio = engine.decode(result.latents, voice=voice, language=language, left=tail)
        native_meta, native_latents, native_audio = native.sentence(dict(request, durations=python_durations), noise, tail)
        latent_rel = rel_error(result.latents, native_latents)
        snr = snr_db(python_audio, native_audio)
        rows.append(dict(voice=voice, text=label, target=index, chunk_id=chunk.chunk_id, phones=len(sentence.phones), frames=length,
                         bucket=result.bucket, durations_identical=python_durations == native_durations, duration_frames_python=length,
                         duration_frames_native=int(sum(native_durations)), latent_rel=latent_rel, snr_db=snr,
                         prefix_frames=native_meta["prefix_frames"]))
        if python_durations != native_durations:
            failures.append(f"durations differ: {voice}/{label}/{chunk.chunk_id}")
        if latent_rel > args.latent_tolerance:
            failures.append(f"latents differ: {voice}/{label}/{chunk.chunk_id}: rel {latent_rel:.3g}")
        if snr < args.min_snr:
            failures.append(f"audio SNR {snr:.1f} dB: {voice}/{label}/{chunk.chunk_id}")
        joined = result.latents if tail is None else np.concatenate([tail, result.latents], -1)
        tail = joined[:, -max(engine.prefix_frames, engine.decode_context):]
    return rows


def compare_g2p(native: Native, engine, plan, label: str, failures: list[str]) -> bool:
    identical = True
    for chunk in plan.chunks:
        python_g2p = engine.phonemize(chunk.text, language=chunk.language)
        native_g2p = native.phonemize(chunk.text, chunk.language)
        if any(python_g2p[key] != native_g2p[key] for key in ("phones", "word_starts", "word_boundary_candidates")):
            identical = False
            failures.append(f"g2p differs: {label}/{chunk.chunk_id}: {python_g2p['phones']} vs {native_g2p['phones']}")
    return identical


def worst_of(rows: list[dict]) -> dict:
    return dict(latent_rel=max((r["latent_rel"] for r in rows), default=0.0), snr_db=min((r["snr_db"] for r in rows), default=float("inf")))


def check_sentences(native: Native, runtime, report: dict, failures: list[str], args) -> None:
    engine = runtime.engine
    rows = []
    g2p_identical = True
    for voice in args.voices:
        for kind, text in TEXTS:
            plan = runtime.plan_text(text, voice_id=voice)
            if native.plan(text, voice) != json.loads(json.dumps(plan.to_dict())):
                failures.append(f"plan differs: {voice}/{kind}")
            g2p_identical &= compare_g2p(native, engine, plan, f"{voice}/{kind}", failures)
            rows += compare_targets(native, engine, plan, voice, kind, failures, args)
    report["sentences"] = dict(rows=rows, count=len(rows), worst=worst_of(rows), g2p_identical=g2p_identical,
                               durations_identical=all(r["durations_identical"] for r in rows))


def end_kind(chunk_id: str, text: str, plan) -> str:
    """Where a target ends: a paragraph (or the passage), a sentence, a clause or between words."""
    label = chunk_id.split("+")[-1]
    index = next(i for i, c in enumerate(plan.chunks) if c.chunk_id == label.split(".")[0])
    sentence = plan.chunks[index]
    if "." not in label or sentence.text.endswith(text):
        return "paragraph" if sentence.paragraph_end or index == len(plan.chunks) - 1 else "sentence"
    return "clause" if text.rstrip("\"'”’»)]")[-1:] in ",;:—-" else "word"


def check_long_form(native: Native, runtime, report: dict, failures: list[str], args) -> None:
    from scyllasband.runtime import SynthesisRequest
    text = "\n\n".join([" ".join(t for _, t in TEXTS), *LONG_FORM_EXTRA])
    out = dict()
    # Built-in noise: runs end to end with a fixed seed and is reproducible.
    started = time.perf_counter()
    first, meta = native.synthesize(text, "scylla", SEED)
    second, _ = native.synthesize(text, "scylla", SEED)
    out["native_seconds"] = round(time.perf_counter() - started, 2)
    out["native_samples"] = int(first.size)
    out["native_chunks"] = len(meta["chunks"])
    out["native_reproducible"] = bool(np.array_equal(first, second))
    if first.size == 0 or not out["native_reproducible"] or not np.isfinite(first).all():
        failures.append("long-form native synthesis did not produce reproducible finite audio")
    # The reference's noise injected: the native loop should speak the same targets and the whole passage should match.
    reference = runtime.synthesize(SynthesisRequest(text=text, voice_id="scylla", seed=SEED, steps=args.steps))

    def noise(index, seed, has_seed, dim, frames, pointer, _user):
        values = np.random.default_rng(int(seed) if has_seed else None).standard_normal((dim, frames)).astype(np.float32) * np.float32(1.0)
        ctypes.memmove(pointer, values.ctypes.data, values.nbytes)

    injected, injected_meta = native.synthesize(text, "scylla", SEED, noise=noise)
    python_targets_ = reference.metadata["chunks"]
    native_targets = injected_meta["chunks"]
    out["reference_samples"] = int(reference.audio.size)
    out["injected_samples"] = int(injected.size)
    python_chunks = [(c["chunk"]["chunk_id"], c["chunk"]["text"]) for c in python_targets_]
    native_chunks = [(c["chunk"]["chunk_id"], c["chunk"]["text"]) for c in native_targets]
    out["chunks_identical"] = python_chunks == native_chunks
    out["durations_identical"] = [c["durations"] for c in python_targets_] == [c["durations"] for c in native_targets]
    differing = sorted({key for key in TARGET_KEYS for p, n in zip(python_targets_, native_targets) if p[key] != n[key]})
    out["target_metadata_identical"] = len(python_targets_) == len(native_targets) and not differing
    frames = [sum(c["durations"]) for c in python_targets_]
    out["targets"] = len(python_targets_)
    out["target_frames"] = [min(frames), max(frames)]
    out["joined"] = sum("+" in chunk_id for chunk_id, _ in python_chunks)
    out["partial"] = sum("." in chunk_id for chunk_id, _ in python_chunks)
    plan = runtime.plan_text(text, voice_id="scylla")
    ends = [end_kind(chunk_id, t, plan) for chunk_id, t in python_chunks]
    out["ends"] = {kind: ends.count(kind) for kind in ("paragraph", "sentence", "clause", "word")}
    out["chunks"] = [dict(chunk_id=chunk_id, text=t, frames=f) for (chunk_id, t), f in zip(python_chunks, frames)]
    if not out["chunks_identical"]:
        failures.append("long-form chunks differ from the reference")
    if not out["durations_identical"]:
        failures.append("long-form durations differ from the reference")
    if not out["target_metadata_identical"]:
        failures.append(f"long-form target metadata differs from the reference: {differing}")
    out["snr_db"] = snr_db(reference.audio, injected)
    if out["snr_db"] < args.min_snr:
        failures.append(f"long-form SNR {out['snr_db']:.1f} dB")
    # Per target, the native graphs given the reference's sentence, noise and prefix.
    rows = compare_targets(native, runtime.engine, plan, "scylla", "long_form", failures, args)
    out["per_target"] = dict(count=len(rows), worst=worst_of(rows), durations_identical=all(r["durations_identical"] for r in rows))
    report["long_form"] = out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--library", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--voices", nargs="+", default=list(VOICES))
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--latent-tolerance", type=float, default=1e-4)
    parser.add_argument("--min-snr", type=float, default=50.0)
    parser.add_argument("--accelerator", choices=sorted(ACCELERATORS), default="cpu", help="Apple builds: compute unit on both sides")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--skip-frontend", action="store_true")
    args = parser.parse_args()

    native = Native(args.library)
    report: dict = dict(backend=native.backend, bundle=args.bundle)
    failures: list[str] = []
    if not args.skip_frontend:
        check_frontend(native, report, failures)
    native.open(args.bundle, ACCELERATORS[args.accelerator])
    runtime = reference_runtime(args.bundle, native.backend, args.accelerator)
    try:
        check_sentences(native, runtime, report, failures, args)
        check_long_form(native, runtime, report, failures, args)
    finally:
        native.close()
    report["failures"] = failures
    summary = dict(backend=report["backend"], bundle=args.bundle, failures=len(failures),
                   sentences=report["sentences"]["count"], g2p_identical=report["sentences"]["g2p_identical"],
                   durations_identical=report["sentences"]["durations_identical"], worst=report["sentences"]["worst"],
                   long_form={k: v for k, v in report["long_form"].items() if k != "chunks"},
                   frontend={k: v for k, v in report.get("frontend", {}).items() if k != "mismatches"})
    print(json.dumps(summary, indent=2))
    for failure in failures[:40]:
        print("FAIL", failure, file=sys.stderr)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, default=float))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
