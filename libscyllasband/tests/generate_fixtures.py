#!/usr/bin/env python3
"""Regenerate tests/fixtures/*.json from the Python reference runtime (no model files needed).

Usage: PYTHONPATH=<repo> python generate_fixtures.py
"""

from __future__ import annotations

import importlib.util
import json
import random
import tempfile
import unicodedata
from pathlib import Path

import numpy as np

from scyllasband.delivery import delivery_spec, resolve_delivery
from scyllasband.engine import balanced_context
from scyllasband.events import frame_events, frames_to_durations, modifier_bits
from scyllasband.g2p import _marked_segments, punctuated_segments
from scyllasband.g2p_phrases import normalize_g2p_phrase_punctuation
from scyllasband.planner import _PARAGRAPH_RE, plan_records, records_from_text, split_for_retry, split_sentences
from scyllasband.streaming import StreamOptions, _choose, synthesize_plan_stream
from scyllasband.text_data import TEXT_DATA, forms
from scyllasband.text_normalizer import normalize_spoken_text

OUT = Path(__file__).resolve().parent / "fixtures"
REPO = Path(__file__).resolve().parents[2]
VOCAB = ["<pad>", "<unk>", "<sil>", "<pause_comma>", "<pause_semicolon>", "<pause_colon>", "<pause_dash>", "<ellipsis>", "<end_stmt>",
         "<end_question>", "<end_exclaim>", "<ctx_sentence_start>", "<ctx_continuation>", "<ctx_sentence_end>", "<ctx_chunk_continue>",
         "a", "b", "d", "e", "h", "i", "k", "l", "n", "o", "s", "t", "ə", "ɪ", "ʊ", "ˈ", "ˌ", "ː", "̃", "̩", "̪"]
TOKEN_TO_ID = {token: index for index, token in enumerate(VOCAB)}
PUNCTUATION = ["<pause_comma>", "<pause_semicolon>", "<pause_colon>", "<pause_dash>", "<ellipsis>", "<end_stmt>", "<end_question>", "<end_exclaim>"]
ACOUSTIC = ["a", "b", "d", "e", "h", "i", "k", "l", "n", "o", "s", "t", "ə", "ɪ", "ʊ"]
MODIFIERS = ["ˈ", "ˌ", "ː", "̃", "̩", "̪"]
TEXTS = [
    'You. Are. Not. It. "Really?!" she asked... Fine', "I kept the letter for years... I never opened it. Maybe I was afraid…  ok",
    "Wait — what was that? I heard it, too.", "for Death, — He kindly stopped —", "Really?! she asked.", "one two three four five six.",
    "Hello (there). [laughs] Fine!) Next? 'Quoted.' “Curly.” End", "No terminal punctuation here", "...", "Ok!!! Sure?? Yes?!?! Fine..",
    "Para one.\n\nPara two line one.\nline two.\n \n\t\nPara three.", "A -- B - C-D — E – F", "x y z, ‘a’ «b»",
    "one, two; three: four - five", "Dr. Smith arrived at 9 a.m. Then he left.", "",
    "It was only a\u2014dream, he said. A\u2014B \u2014 c--d e---f g\u2013h 3\u20135 x\u2014\u2014y \u00e9t\u00e9\u2014hiver a\u2e3ab door\u2014\u2019 well-known",
    "Well, the engineer who rebuilt the clockwork orchestra in the basement finally invited everyone to hear it play, at last.",
    "quatre-vingt-dix-sept, well-known - no; COVID-19 a\u2011b-c x--y",
]


def events_cases(rng: random.Random) -> list[dict]:
    cases = [
        dict(phones=["<sil>", "ˈ", "a", "b", "<pause_comma>", "<sil>", "h", "ː", "<end_stmt>", "<sil>"], durations=[2, 0, 3, 2, 0, 3, 2, 0, 0, 2],
             word_starts=[6], following=[]),
        dict(phones=["<sil>", "a", "<pause_comma>", "<sil>", "b", "<sil>"], durations=[1, 2, 0, 0, 2, 1], word_starts=[4], following=[]),
    ]
    for _ in range(60):
        phones, word_starts = ["<sil>"], []
        for word in range(rng.randint(1, 6)):
            if word:
                if phones[-1] != "<sil>":
                    phones.append("<sil>")
                word_starts.append(len(phones))
            for _ in range(rng.randint(1, 5)):
                if rng.random() < 0.3:
                    phones.append(rng.choice(["ˈ", "ˌ"]))
                phones.append(rng.choice(ACOUSTIC))
                if rng.random() < 0.25:
                    phones.append(rng.choice(["ː", "̃", "̩", "̪"]))
            if rng.random() < 0.4:
                for _ in range(rng.randint(1, 2)):
                    phones.append(rng.choice(PUNCTUATION))
                phones.append("<sil>")
        if phones[-1] != "<sil>":
            phones.append("<sil>")
        values = [rng.uniform(-0.5, 6.0) for _ in phones]
        durations = frames_to_durations(values, phones)
        if sum(durations) == 0:
            continue
        following = [rng.choice(ACOUSTIC + PUNCTUATION) for _ in range(rng.randint(0, 6))]
        cases.append(dict(phones=phones, durations=durations, word_starts=word_starts + ([99] if rng.random() < 0.1 else []), following=following))
    bits = modifier_bits(TOKEN_TO_ID)
    for case in cases:
        events = frame_events(case["phones"], case["durations"], case["word_starts"], case["following"], TOKEN_TO_ID, bits)
        case["expected"] = {key: value.tolist() for key, value in events.items()}
    return cases


def durations_cases(rng: random.Random) -> list[dict]:
    cases = []
    floors = random.Random(20261110)   # its own stream, so the other suites' draws stay as they were
    for _ in range(40):
        phones = [rng.choice(ACOUSTIC + MODIFIERS + PUNCTUATION + ["<sil>"]) for _ in range(rng.randint(1, 20))]
        values = [rng.choice([rng.uniform(-1, 8), float(rng.randint(0, 6)) + 0.5, 2.5, 0.5, 1.5, 0.4999]) for _ in phones]
        values = [float(f"{v:.4f}") if rng.random() < 0.5 else v for v in values]
        import numpy as np
        values = np.asarray(values, np.float32).tolist()
        scale = rng.choice([1.0, 1 / 0.75, 1 / 1.25, 1 / 1.1])
        pause_scale = rng.choice([1.0, 1.0, float(np.exp(0.13)), float(np.exp(-0.26)), 1.3])
        silences = [i for i, p in enumerate(phones) if p == "<sil>"]
        min_pauses = sorted(floors.sample(silences, floors.randint(0, min(2, len(silences))))) if floors.random() < 0.4 else []
        min_pause_frames = floors.choice([0.3 * 24000 / 512, 6.5, 3.0])
        cases.append(dict(phones=phones, values=values, scale=scale, pause_scale=pause_scale, min_pauses=min_pauses,
                          min_pause_frames=min_pause_frames,
                          expected=frames_to_durations(values, phones, scale=scale, pause_scale=pause_scale, min_pauses=min_pauses,
                                                       min_pause_frames=min_pause_frames)))
    return cases


def context_cases(rng: random.Random) -> list[dict]:
    cases = [dict(before=list(range(10)), target=[99] * 4, after=list(range(20, 30)), width=10),
             dict(before=[1, 2], target=[9], after=list(range(20, 30)), width=8),
             dict(before=list(range(300)), target=list(range(600)), after=[], width=512)]
    for _ in range(40):
        cases.append(dict(before=[rng.randint(3, 80) for _ in range(rng.randint(0, 200))], target=[rng.randint(3, 80) for _ in range(rng.randint(1, 60))],
                          after=[rng.randint(3, 80) for _ in range(rng.randint(0, 200))], width=rng.choice([16, 64, 128, 512])))
    for case in cases:
        span, segments = balanced_context(case["before"], case["target"], case["after"], case["width"])
        case["expected"] = dict(span=span, segments=segments)
    return cases


def delivery_cases() -> list[dict]:
    specs = [None, "default", " Neutral ", "intensity=1", "intensity=1,pace=-0.5,pitch=2", "pace=-3", "pitch=2.5", " intensity = 1 ",
             "intensity=1 , pace=0.25", "energy=3", "whisper=on", "intensity=abc", "intensity=1,intensity=2", "tempo=1", "intensity", "",
             "intensity=1e0", "pace=.5", "pace=2.", "intensity=nan", "intensity=inf", "pace=Default", "pitch=auto", "Intensity=-1.5"]
    cases = []
    for spec in specs:
        try:
            resolved = resolve_delivery(spec)
            cases.append(dict(spec=spec, error=False, resolved=resolved, spec_text=delivery_spec(resolved)))
        except ValueError:
            cases.append(dict(spec=spec, error=True))
    return cases


def text_cases() -> dict:
    sentences, retry, paragraphs, segments, phrase = [], [], [], [], []
    for text in TEXTS:
        sentences.append(dict(text=text, expected=split_sentences(text)))
        paragraphs.append(dict(text=text, expected=[p for p in _PARAGRAPH_RE.split(text) if p.strip()]))
        phrase.append(dict(text=text, expected=normalize_g2p_phrase_punctuation(text)))
        for max_chars in (140, 10):
            segments.append(dict(text=text, max_chars=max_chars,
                                 expected=[[segment, list(tokens)] for segment, tokens in punctuated_segments(text, max_chars=max_chars)],
                                 tight=[tight for _, _, tight in _marked_segments(text, max_chars=max_chars)]))
        for min_share in (0.0, 0.1, 0.3):
            try:
                retry.append(dict(text=text, min_share=min_share, expected=split_for_retry(text, min_share=min_share)))
            except ValueError:
                retry.append(dict(text=text, min_share=min_share, error=True))
    return dict(sentences=sentences, retry=retry, paragraphs=paragraphs, segments=segments, phrase=phrase)


def choose_cases(rng: random.Random) -> list[dict]:
    """Target-end choices over candidate silences built as the streaming loop builds them."""
    cases = []
    for _ in range(400):
        shortest, longest = rng.choice([(6, 12), (10, 40), (64, 420)])
        first = rng.random() < 0.35
        index, frames, stops = rng.randint(0, 4), 0, []
        for _ in range(rng.randint(1, 12)):
            index += rng.randint(1, 30)
            frames += rng.randint(0, longest // 3 + 1)
            stops.append((index, rng.choices(["sentence", "clause", "word", "paragraph"], weights=[3, 3, 4, 1])[0], frames))
        if rng.random() < 0.4:
            stops[-1] = (stops[-1][0], "end", stops[-1][2])
        candidates = []
        for stop in stops:               # a paragraph or record ends the target once it is long enough
            candidates.append(stop)
            if stop[1] == "end" or (stop[1] == "paragraph" and stop[2] >= shortest):
                break
        expected = _choose([(i, kind) for i, kind, _ in candidates], {i: f for i, _, f in candidates}, first, shortest, longest)
        cases.append(dict(candidates=[list(c) for c in candidates], first=first, shortest=shortest, longest=longest, expected=expected))
    return cases


def stream_cases() -> dict:
    """Targets the Python streaming loop chooses with the fake graphs of tests/test_runtime.py (one frame per phone and
    silence, none for punctuation; trained range 6 to 12 frames; a span wide enough for the measuring window)."""
    spec = importlib.util.spec_from_file_location("scyllasband_test_runtime", REPO / "tests" / "test_runtime.py")
    fakes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fakes)
    text = lambda value: records_from_text(value, voice="scylla", language=None)  # noqa: E731
    scenarios = [
        ("short sentences join", text("ab. ab. ab ab ab. ab ab."), 1.0),
        ("short last sentence", text("ab ab. ab."), 1.0),
        ("short first sentence", text("ab. ab ab ab ab."), 1.0),
        ("fill", text("ab. ab. ab. ab. ab. ab. ab. ab."), 1.0),
        ("no punctuation", text("ab ab ab ab ab ab ab ab"), 1.0),
        ("slow", text("ab. ab. ab. ab. ab. ab."), 2.0),
        ("short paragraphs", text("ab.\n\nab."), 1.0),
        ("short records", [dict(text="ab.", voice="scylla"), dict(text="ab.", voice="scylla")], 1.0),
        ("chains", [dict(text="ab.", voice="scylla"), dict(text="ab.", voice="scylla", delivery="intensity=1")], 1.0),
        ("long paragraph", text("ab ab. ab ab.\n\nab ab."), 1.0),
        ("clauses", text("ab ab, ab ab ab; ab ab: ab ab ab, ab. ab ab ab ab, ab ab ab ab ab. ab, ab ab ab ab ab ab, ab ab."), 1.0),
        ("clauses fast", text("ab ab, ab ab ab; ab ab: ab ab ab, ab. ab ab ab ab, ab ab ab ab ab. ab, ab ab ab ab ab ab, ab ab."), 0.5),
        ("words", text(" ".join(["ab"] * 40)), 0.6),
        ("mixed", [dict(text="ab ab ab. ab, ab.\n\nab ab ab ab ab ab ab ab ab ab ab ab ab.", voice="scylla"),
                   dict(text="ab. ab ab ab ab ab ab ab, ab ab ab.", voice="scylla"),
                   dict(text="ab ab ab ab ab.", voice="scylla", delivery="intensity=1")], 1.0),
    ]
    controls = dict(chunking=dict(min_target_frames=6, max_target_frames=12), span_conditioning=dict(context_max_phones=256, context_phones_each_side=8))
    cases = []
    for name, records, frames_per_phone in scenarios:
        engine = fakes._fake_engine(frames_per_phone, **controls)
        plan = plan_records(records, resolve_language=lambda voice, language: "en_us")
        targets = []
        for event in synthesize_plan_stream(engine, plan, StreamOptions(steps=1, seed=0)):
            if event.type == "audio_chunk":
                meta = event.metadata
                targets.append(dict(chunk_id=event.chunk.chunk_id, text=event.chunk.text, index=event.chunk.index, phones=meta["phones"],
                                    durations=meta["durations"], latent_frames=meta["latent_frames"], context_before=meta["context_before"],
                                    context_after=meta["context_after"], prefix_frames=meta["prefix_frames"], seed=meta["seed"]))
        chunks = [dict(chunk.to_dict(), delivery=delivery_spec(chunk.delivery)) for chunk in plan.chunks]
        cases.append(dict(name=name, frames_per_phone=frames_per_phone, chunks=chunks, targets=targets))
    with tempfile.TemporaryDirectory() as tmp:     # the fake bundle, for the native fake graphs to load
        bundle = fakes._fake_bundle(Path(tmp), controls=controls)
        files = {str(path.relative_to(bundle)): json.loads(path.read_text()) if path.suffix == ".json" else None
                 for path in sorted(bundle.rglob("*")) if path.is_file()}
    return dict(bundle=files, cases=cases)


def records_cases() -> dict:
    """Plans of records (planner.plan_records through the runtime: voice and language checks, delivery forms, chains) and a
    records stream with pause_ms, from the Python runtime on the fake bundle (one voice, scylla / en_us)."""
    spec = importlib.util.spec_from_file_location("scyllasband_test_runtime", REPO / "tests" / "test_runtime.py")
    fakes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fakes)
    from scyllasband.runtime import ScyllasBandRuntime
    controls = dict(chunking=dict(min_target_frames=6, max_target_frames=12), span_conditioning=dict(context_max_phones=256, context_phones_each_side=8))
    S = "scylla"
    scenarios = [
        ("one record", [dict(text="ab. ab ab.\n\nab.", voice=S)], True),
        ("records join one chain", [dict(text="ab.", voice=S), dict(text="ab ab.", voice=S, language="en_us")], True),
        ("language spellings", [dict(text="ab.", voice=S, language="EN-US"), dict(text="ab.", voice=S, language="english"),
                                dict(text="ab.", voice=S, language="en"), dict(text="ab.", voice=S, language=" ")], True),
        ("delivery string", [dict(text="ab.", voice=S, delivery="intensity=1,pace=-0.5"), dict(text="ab.", voice=S, delivery="pace=-0.5,intensity=1")], True),
        ("delivery default", [dict(text="ab.", voice=S), dict(text="ab.", voice=S, delivery="default"), dict(text="ab.", voice=S, delivery={}),
                              dict(text="ab.", voice=S, delivery=dict(intensity=None, pace="default", pitch=" Auto "))], True),
        ("delivery mapping", [dict(text="ab.", voice=S, delivery=dict(intensity=1, pace="-0.5")), dict(text="ab.", voice=S, delivery=dict(pitch=5))], True),
        ("delivery fields", [dict(text="ab.", voice=S, intensity=1, pace=-0.5), dict(text="ab.", voice=S, delivery="intensity=1,pace=-0.5"),
                             dict(text="ab.", voice=S, pitch=-9)], True),
        ("empty record between", [dict(text="ab.", voice=S), dict(text="", voice=S, delivery="intensity=2"), dict(text="ab.", voice=S)], True),
        ("empty record same key", [dict(text="ab.", voice=S), dict(text=None, voice=S), dict(text="ab.", voice=S)], True),
        ("voice padded", [dict(text="ab.", voice="  scylla ")], True),
        ("no normalize", [dict(text="ab 12.", voice=S)], False),
        ("normalize", [dict(text="ab 12.", voice=S)], True),
        ("no voice", [dict(text="ab.", voice=S), dict(text="ab.", voice=" ")], True),
        ("missing voice", [dict(text="ab.")], True),
        ("unknown voice", [dict(text="ab.", voice="nobody")], True),
        ("unsupported language", [dict(text="ab.", voice=S, language="fr")], True),
        ("emotion", [dict(text="ab.", voice=S, emotion="happy")], True),
        ("fields and delivery", [dict(text="ab.", voice=S, intensity=1, delivery="pace=1")], True),
        ("old keys", [dict(text="ab.", voice=S, delivery=dict(energy=1))], True),
        ("old keys string", [dict(text="ab.", voice=S, delivery="valence=1")], True),
        ("extra keys", [dict(text="ab.", voice=S, delivery=dict(tempo=1, speed=2, intensity=1))], True),
        ("boolean", [dict(text="ab.", voice=S, delivery=dict(intensity=True))], True),
        ("not a number", [dict(text="ab.", voice=S, delivery=dict(pace="fast"))], True),
        ("not finite", [dict(text="ab.", voice=S, delivery=dict(pitch="nan"))], True),
        ("not finite field", [dict(text="ab.", voice=S, pace="-inf")], True),
        ("delivery number", [dict(text="ab.", voice=S, delivery=5)], True),
        ("duplicate string", [dict(text="ab.", voice=S, delivery="intensity=1,intensity=2")], True),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        bundle = fakes._fake_bundle(Path(tmp), controls=controls)
        files = _bundle_files(bundle)
        runtime = ScyllasBandRuntime(fakes._fake_engine(1.0, **controls), bundle_dir=bundle, backend="onnx")
        cases = []
        for name, records, normalize in scenarios:
            try:
                plan = runtime.plan_records(records, normalize=normalize).to_dict()
                cases.append(dict(name=name, records=records, normalize=normalize, plan=plan))
            except ValueError as exc:
                cases.append(dict(name=name, records=records, normalize=normalize, error=str(exc)))
        records = [dict(text="ab. ab.", voice=S), dict(text="ab.", voice=S, delivery="intensity=1"), dict(text="", voice=S, pace=1),
                   dict(text="ab ab.", voice=S), dict(text="ab.", voice=S)]
        targets = []
        for event in runtime.synthesize_stream(records, pause_ms=100.0, steps=1, seed=0):
            if event.type == "audio_chunk":
                zeros = 0
                while zeros < event.audio.size and event.audio[zeros] == 0.0:
                    zeros += 1
                targets.append(dict(chunk_id=event.chunk.chunk_id, chain_id=event.chunk.chain_id, samples=int(event.audio.size), leading_zeros=zeros,
                                    seed=event.metadata["seed"]))
    return dict(bundle=files, cases=cases, pause=dict(records=records, pause_ms=100.0, targets=targets))


def _bundle_files(bundle: Path) -> dict:
    return {str(path.relative_to(bundle)): json.loads(path.read_text()) if path.suffix == ".json" else None
            for path in sorted(bundle.rglob("*")) if path.is_file()}


def _run_length(values) -> list:
    """[[value, count], ...] of a float32 waveform (the fake decoders return constants)."""
    out = []
    for value in (float(v) for v in values):
        if out and out[-1][0] == value:
            out[-1][1] += 1
        else:
            out.append([value, 1])
    return out


def singing_cases() -> dict:
    """The singing planner and renderer of scyllasband/singing.py on the fake v4 bundle of tests/test_runtime.py: scores, the
    phrase plan (phones, durations, melody), what a sung phrase feeds the graphs and the song it returns. The fake G2P spells
    letters (e -> o), so words use a, b, h, l, o, w and e; at 1200 bpm a beat is 0.05 s, so phrases fit the 16-frame buckets."""
    import numpy as np
    from scyllasband import singing
    spec = importlib.util.spec_from_file_location("scyllasband_test_runtime", REPO / "tests" / "test_runtime.py")
    fakes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fakes)
    score = dict(text="hello bob wow", bpm=1200, style="pop",
                 notes=[["hello", "C4", 2], ["-", "rest", 6], ["bob~1", "D4", 1], ["bob~2", "E4", 1], ["wow", "G4", 2]])
    scores = dict(
        basic=score,
        ballad=dict(score, style="ballad"),
        syllables=dict(text="hellow how", bpm=1200, style="pop", notes=[["hel~1", "A3", 1], ["low~2", "B3", 1], ["how", "C#4", 2]]),
        melisma=dict(bpm=1200, style="sung", notes=[["how~1", "Bb3", 1], ["how~2", "F#4", 1], ["-", "rest", 0.5], ["wow", "G5", 1.5]]),
        continuous=dict(text="how bob wow hello below allow", bpm=1200, style="pop",
                        notes=[["how", "C4", 1], ["bob", "D4", 1], ["wow", "E4", 1], ["hello", "F4", 1.5], ["below", "G4", 1], ["allow", "A4", 1]]),
        string_numbers=dict(text=" Bob  WOW ", bpm="1200", style=" Pop ", notes=[["bob", "c4", "2"], ["wow", "d#4", 2.5]]),
        many_rests=dict(text="bob how wow", bpm=1200, style="ballad",
                        notes=[["-", "rest", 1], ["bob", "C3", 1], ["-", "r", 2], ["how", "E3", 1], ["-", "", 6], ["wow", "G3", 1], ["-", "rest", 1]]),
    )
    errors = [
        [1, 2], dict(bpm=0, notes=[["a", "C4", 1]]), dict(bpm="fast", notes=[["a", "C4", 1]]), dict(notes=[["a", "C4", 1]]),
        dict(bpm=100, style="jazz", notes=[["a", "C4", 1]]), dict(bpm=100, notes=[]), dict(bpm=100, notes="abc"),
        dict(bpm=100, notes=[["a", "C4"]]), dict(bpm=100, notes=[["a", "C4", "long"]]), dict(bpm=100, notes=[["a", "C4", -1]]),
        dict(bpm=100, notes=[["a", "H4", 1]]), dict(bpm=100, notes=[["a", "C", 1]]), dict(bpm=100, notes=[["a~2", "C4", 1]]),
        dict(bpm=100, notes=[["a~x", "C4", 1]]), dict(bpm=100, notes=[["~1", "C4", 1]]), dict(bpm=100, notes=[["-", "rest", 1]]),
        dict(text="other words", bpm=100, notes=[["a", "C4", 1]]), dict(bpm=100, notes=[["a~1", "C4", 1], ["-", "rest", 1], ["a~2", "D4", 1]]),
    ]
    parse = []
    for name, data in scores.items():
        parse.append(dict(name=name, score=data, expected=singing.parse_score(data).to_dict(), error=False))
    for data in errors:
        try:
            singing.parse_score(data)
            raise AssertionError(f"expected a ScoreError for {data}")
        except singing.ScoreError:
            parse.append(dict(name=json.dumps(data)[:60], score=data, error=True))
    notes = []
    for name in ["C4", "c4", "D#4", "Bb3", "bb3", "F#3", "E♭5", "G♯2", "B#3", "Cb4", "A-1", "C10", " G4 ", "rest", "-", "r", "", "REST",
                 "H4", "C", "4C", "C#", "Cx4", "é4"]:
        try:
            notes.append(dict(name=name, midi=singing.note_to_midi(name), error=False))
        except singing.ScoreError:
            notes.append(dict(name=name, error=True))
    names = [dict(midi=m, name=singing.midi_to_note(m)) for m in (-13, -1, 0, 11, 12, 59, 60, 61, 127, 128)]
    registers = []
    for pitched in ([60, 62, 64], [48], [72, 76, 79], [36, 40], [84, 88], [55, 67]):
        for center in (None, 58.35, 50.0, 49.0, 59.0, 63.0):
            data = dict(bpm=120, notes=[[f"w{i}", singing.midi_to_note(m), 1] for i, m in enumerate(pitched)])
            registers.append(dict(score=data, center=center, shift=singing.register_shift(singing.parse_score(data), center)))
    controls = dict(chunking=dict(min_target_frames=6, max_target_frames=12), span_conditioning=dict(context_max_phones=256, context_phones_each_side=8))
    guided = json.loads(json.dumps(fakes.V4_DELIVERY_MODES))
    guided["singing"].update(guidance=2.0, voice_midi_center=dict(scylla=63.0))
    plans = []
    for name, data in scores.items():
        for register, transpose, pad, max_seconds in ((True, 0, 0.02, None), (False, 0, 0.02, None), (True, -5, 0.02, None), (True, 0, 0.01, 0.12)):
            engine = fakes._fake_engine(v4=True)
            try:
                plan = singing.plan_song(data, engine, "scylla", "en_us", register=register, transpose=transpose, pad=pad, max_seconds=max_seconds)
            except Exception as error:   # noqa: BLE001 - a phrase longer than the fake buckets
                plans.append(dict(score=data, register=register, transpose=transpose, pad=pad, max_seconds=max_seconds, error=type(error).__name__))
                continue
            plans.append(dict(score=data, register=register, transpose=transpose, pad=pad, max_seconds=max_seconds, error=None,
                              register_shift=plan.register_shift, mode_id=plan.mode_id, seconds=plan.seconds,
                              phrases=[dict(start=p.phrase.start, end=p.phrase.end, text=p.phrase.text, phones=list(p.sentence.phones),
                                            word_starts=list(p.sentence.word_starts), durations=list(p.durations), frames=p.frames,
                                            melody=[float(v) for v in np.asarray(p.melody, np.float32).reshape(-1)]) for p in plan.phrases]))
    sings = []
    for bundle, data, kwargs in (("v4", score, {}), ("v4", dict(score, style="ballad"), {}), ("v4", score, dict(steps=2, sampler="euler", decoder="vocos")),
                                 ("v4_guided", score, dict(steps=1, sampler="euler")), ("v4_guided", score, dict(steps=1, sampler="heun", guidance=1.0)),
                                 ("v4_guided", scores["melisma"], dict(steps=1, sampler="euler", transpose=3))):
        engine = fakes._fake_engine(v4=True, **({"delivery_modes": guided} if bundle == "v4_guided" else {}))
        engine._fake.calls.clear()
        result = singing.sing(engine, data, "scylla", "en_us", seed=5, pad=0.02, **kwargs)
        calls = []
        for call in engine._fake.calls:
            row = dict(name=call["name"])
            if call["name"].startswith("vector_estimator"):
                row.update(mode=int(call["delivery_mode_id"][0]), melody_zero=bool((call["expanded_melody"] == 0).all()), time=float(call["time"][0]))
            calls.append(row)
        meta = result.metadata
        sings.append(dict(bundle=bundle, score=data, options=kwargs, calls=calls, audio=_run_length(result.audio),
                          metadata=dict(decoder=meta["decoder"], steps=meta["steps"], sampler=meta["sampler"], guidance=meta["guidance"],
                                        mode=meta["mode"], mode_id=meta["mode_id"], register_shift=meta["register_shift"], transpose=meta["transpose"],
                                        seconds=meta["seconds"], phrases=[dict(text=p["text"], frames=p["frames"], bucket=p["bucket"], seed=p["seed"],
                                                                               seconds=p["seconds"], durations=p["durations"]) for p in meta["phrases"]])))
    # Speech on the v4 bundle: the targets of the v3 loop, with delivery_mode_id = 0 on every flow call.
    speech = []
    for text, frames_per_phone in (("ab. ab ab ab ab.", 1.0), ("ab ab, ab ab ab; ab ab: ab ab ab, ab.", 1.0)):
        engine = fakes._fake_engine(frames_per_phone, v4=True, **controls)
        plan = plan_records(records_from_text(text, voice="scylla", language=None), resolve_language=lambda voice, language: "en_us")
        targets = []
        for event in synthesize_plan_stream(engine, plan, StreamOptions(steps=1, seed=0)):
            if event.type == "audio_chunk":
                meta = event.metadata
                targets.append(dict(chunk_id=event.chunk.chunk_id, text=event.chunk.text, phones=meta["phones"], durations=meta["durations"],
                                    latent_frames=meta["latent_frames"], prefix_frames=meta["prefix_frames"], delivery_mode=meta["delivery_mode"]))
        modes = sorted({int(c["delivery_mode_id"][0]) for c in engine._fake.calls if c["name"].startswith("vector_estimator")})
        speech.append(dict(chunks=[dict(chunk.to_dict(), delivery=delivery_spec(chunk.delivery)) for chunk in plan.chunks],
                           frames_per_phone=frames_per_phone, targets=targets, flow_modes=modes))
    bundles = {}
    with tempfile.TemporaryDirectory() as tmp:
        bundles["v4"] = _bundle_files(fakes._fake_bundle(Path(tmp) / "v4", v4=True))
        bundles["v4_guided"] = _bundle_files(fakes._fake_bundle(Path(tmp) / "guided", v4=True, controls=dict(delivery_modes=guided)))
        bundles["v4_speech"] = _bundle_files(fakes._fake_bundle(Path(tmp) / "speech", v4=True, controls=controls))
        bundles["v3"] = _bundle_files(fakes._fake_bundle(Path(tmp) / "v3"))
    return dict(bundles=bundles, parse=parse, notes=notes, names=names, registers=registers, plans=plans, sings=sings, speech=speech)


def v5_cases(rng: random.Random) -> dict:
    """The v5 rules against the Python runtime on the fake v5 bundle of tests/test_v5.py: the non-verbal spellings and phones,
    the minimum lengths of an opening "Uh-uh", bundles the contract refuses, the targets of the streaming loop (the phone
    minimum, pause marks, styles and what each graph is fed), and singing (lyrics read as a phrase and cut into words, their
    timing, styles, the register floor, guidance and the decoder rule)."""
    import sys
    from scyllasband import singing
    from scyllasband.contract import BundleError, validate_bundle_layout
    from scyllasband.events import opening_minimums
    from scyllasband.runtime import ScyllasBandRuntime
    from scyllasband.text_normalizer import nonverbal_spellings
    sys.path.insert(0, str(REPO / "tests"))
    import test_v5 as fakes

    # --- spellings: every variant in every casing, alone and between characters that do and do not end a word ---------------
    variants = ["uhuh", "nuhuh", "hehheh", "ah-ha-ha", "a-ha-ha", "ah ha ha", "ah‑ha‑ha", "a‑ha-ha", "a-hem", "a‑hem", "pst", "sh",
                "uh-uh", "ahaha", "ah-ha ha", "ah  ha ha", "a ha ha", "ahh-ha-ha", "shh", "psst", "ahem", "uhuhuh", "hehhehheh",
                "ah-ha-ha-ha", "ſh", "pſt", "ahem-", "ps t", "s h"]
    casings = [str.lower, str.upper, str.capitalize, lambda v: v[:-1].lower() + v[-1:].upper(), lambda v: v[:1].lower() + v[1:].upper()]
    befores = ["", " ", "x", "X", "'", "’", "-", "‑", "(", '"', "1", "é", "\n", "—", ". "]
    afters = ["", " ", "x", "Z", "'", "’", "-", "‑", ".", ",", "!", "?", "…", ";", ":", " .", "  !", " x", "\n", " \n", "\t,", ")", "1", "é", " and", " — "]
    texts = []
    for variant in variants:
        for casing in casings:
            word = casing(variant)
            texts.append(word)
            texts.extend(rng.choice(befores) + word + rng.choice(afters) for _ in range(10))
    texts += ["Uhuh, no. Nuhuh! HEHHEH... A-hem. Pst! Sh.", "sh, sh, she said sh", "Ah-ha-ha and ah ha ha, a-ha-ha", "Pst pst PST pSt",
              "push ah-ha-han uhuhs", "Sh", "sh\n", "sh \n\n", "Sh  ", "sh x", "", "   ", "UHUH NUHUH", "uhuh‑nuhuh", "It's uhuh's"]
    spellings = [dict(text=text, language="en_us", expected=nonverbal_spellings(text, language="en_us")) for text in dict.fromkeys(texts)]
    for language in ("en_gb", "en", "", "de", "vi", "es"):
        spellings += [dict(text=text, language=language, expected=nonverbal_spellings(text, language=language)) for text in texts[-15:]]

    # --- opening minimums ---------------------------------------------------------------------------------------------------
    word = ["ʌ", "̃", "ʔ", "<sil>", "ʌ", "̃"]
    pool = ["<sil>", "n", "ˈ", "a", "b", "<end_stmt>", "<nonverbal>", "ʌ"]
    opening = []
    for _ in range(400):
        phones = []
        for _ in range(rng.randint(1, 4)):
            phones += [rng.choice(pool) for _ in range(rng.randint(0, 4))]
            phones += word if rng.random() < 0.8 else word[:rng.randint(1, 5)]
        phones += [rng.choice(pool) for _ in range(rng.randint(0, 3))]
        durations = [rng.randint(0, 14) for _ in phones]
        starts = sorted(rng.sample(range(len(phones)), k=rng.randint(0, min(5, len(phones)))))
        opening.append(dict(phones=phones, durations=durations, sentence_starts=starts, expected=opening_minimums(phones, durations, starts)))

    bundles: dict[str, dict] = {}

    def engine_of(name: str, frames_per_phone: float = 1.0, *, mutate=None, **controls):
        engine = fakes._v5_engine(frames_per_phone, mutate=mutate, **controls)
        bundle = Path(engine._tmp.name) / "bundle"
        bundles.setdefault(name, _bundle_files(bundle))
        return engine, bundle

    # --- the G2P with and without the non-verbal table ----------------------------------------------------------------------
    phonemize = []
    engine, _ = engine_of("default")
    for text in ["aha", "haha", "ha ha", "ha ha ha", "ha ha ha ha", "ha ha ha ha ha", '"uh uh" hello', "uh uh, hello.", "(hmm) hello", "hmm, ha, ha!",
                 "HAHA hello", "ha-ha", "hello aha's", "nuh uh", "uh huh", "'ha' ha", "ha (ha)", "ha, ha", "uhuh", "hello mm ow", "nuh uh uh",
                 "ha) ha", "ha (ha ha", "Ha HA ha.", "hello hahaha hello", "ah ahh aw aww", "hello", "below allow"]:
        for nonverbal in (True, False):
            result = engine.phonemize(text, language="en_us", nonverbal=nonverbal)
            phonemize.append(dict(text=text, nonverbal=nonverbal, phones=result["phones"], word_starts=result["word_starts"],
                                  word_boundary_candidates=result["word_boundary_candidates"]))

    # --- bundles the contract refuses ---------------------------------------------------------------------------------------
    def edit(path, value=None, *, delete=False):
        def mutate(manifest):
            node = manifest
            for key in path[:-1]:
                node = node[key]
            if delete:
                del node[path[-1]]
            else:
                node[path[-1]] = value
        return mutate

    def v4_inputs(manifest):
        manifest["components"]["duration_predictor"]["inputs"].remove("style_id")

    refused = []
    for name, mutate, same_message in (
            ("named v5", edit(["controls", "graph_input_contract"], "scyllasband_measured_controls_v5"), True),
            ("no identity asset", edit(["assets", "voice_identity"], delete=True), False),
            ("identity file missing", edit(["assets", "voice_identity"], "assets/none.json"), True),
            ("modes disabled", edit(["controls", "delivery_modes", "enabled"], False), True),
            ("no spoken row", edit(["controls", "delivery_modes", "ids"], dict(rap=1)), True),
            ("no trained list", edit(["controls", "delivery_modes", "duration_styles_trained"], delete=True), True),
            ("spoken not trained", edit(["controls", "delivery_modes", "duration_styles_trained"], ["rap"]), True),
            ("trained without a row", edit(["controls", "delivery_modes", "duration_styles_trained"], ["spoken", "chant"]), True),
            ("no pause marks", edit(["controls", "pause_marks"], delete=True), True),
            ("pause marks unsaid", edit(["controls", "pause_marks"], dict(token="<nonverbal>")), True),
            ("pause mark outside the vocabulary", edit(["controls", "pause_marks"], dict(token="<hum>", default="all")), True),
            ("quiet pauses need no token", edit(["controls", "pause_marks"], dict(default="none")), True),
            ("no style input", v4_inputs, False),
            ("unknown contract", edit(["controls", "graph_input_contract"], "scyllasband_measured_controls_v6"), False)):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = fakes._v5_bundle(Path(tmp), mutate=mutate)
            try:
                validate_bundle_layout(bundle)
                error = None
            except BundleError as exc:
                error = str(exc)
            refused.append(dict(name=name, bundle=_bundle_files(bundle), error=error, same_message=same_message and error is not None))
    with tempfile.TemporaryDirectory() as tmp:   # an identity table without the bundle's voice
        bundle = fakes._v5_bundle(Path(tmp))
        (bundle / "assets/voice_identity.json").write_text(json.dumps(dict(dim=4, voices=dict(other=[0.0] * 4))))
        try:
            validate_bundle_layout(bundle)
            raise AssertionError("expected a BundleError")
        except BundleError as exc:
            refused.append(dict(name="no vector for the voice", bundle=_bundle_files(bundle), error=str(exc), same_message=True))

    # --- the streaming loop -------------------------------------------------------------------------------------------------
    S = "scylla"
    passage = fakes.PASSAGE
    three = dict(passage, chunking={**passage["chunking"], "min_target_phones": 3})

    def wide(manifest):   # room for an opening "Uh-uh" at its minimum lengths
        manifest["controls"]["target_buckets"]["buckets"][1]["latent_frames"] = 96

    long_targets = dict(chunking=dict(min_target_frames=3, max_target_frames=90), span_conditioning=dict(context_max_phones=128, context_phones_each_side=8))
    variants_ = dict(three=dict(controls=three), passage=dict(controls=passage), quiet=dict(controls=dict(passage, pause_marks=dict(default="none"))),
                     wide=dict(controls=dict(long_targets, chunking={**long_targets["chunking"], "min_target_phones": 3}), mutate=wide),
                     twelve=dict(controls=long_targets, mutate=wide))
    text = lambda value: records_from_text(value, voice=S, language=None)   # noqa: E731
    scenarios = [
        ("short joins the next", "three", text("ab. ab ab."), {}, 1.0),
        ("short last joins the one before", "three", text("ab ab. ab."), {}, 1.0),
        ("short takes part of the next", "three", text("ab. ab ab ab ab."), {}, 1.0),
        ("short last takes part of the one before", "three", text("ab ab ab ab. ab."), {}, 1.0),
        ("lone short", "three", text("ab."), {}, 1.0),
        ("new chain", "three", [dict(text="ab.", voice=S), dict(text="ab.", voice=S, delivery="intensity=1")], {}, 1.0),
        ("paragraphs", "three", text("ab.\n\nab ab."), {}, 1.0),
        ("several short", "three", text("ab. ab. ab. ab ab ab. ab. ab."), {}, 1.0),
        ("marks are not counted", "three", text("aha, ab. ab."), {}, 1.0),
        ("frames alone", "passage", text("ab. ab ab. ab ab. ab."), {}, 1.0),
        ("pause marks", "passage", text("ab, ab. ab ab? ab."), {}, 1.0),
        ("quiet pauses", "quiet", text("ab, ab. ab ab? ab."), {}, 1.0),
        ("rap", "passage", text("ab ab. ab ab, ab."), dict(style="rap"), 1.0),
        ("rap in capitals with a pitch shift", "passage", [dict(text="ab ab. ab ab.", voice=S, pitch=1.5)], dict(style=" RAP "), 1.0),
        ("rap takes no intensity", "passage", [dict(text="ab ab.", voice=S, intensity=1)], dict(style="rap"), 1.0),
        ("rap takes no pace in a later record", "passage", [dict(text="ab ab.", voice=S), dict(text="ab.", voice=S, pace=1, intensity=-1)],
         dict(style="rap"), 1.0),
        ("a sung style is not spoken", "passage", text("ab ab."), dict(style="jazz"), 1.0),
        ("an unknown style", "passage", text("ab ab."), dict(style="opera"), 1.0),
        ("spoken by name", "passage", text("ab ab."), dict(style="spoken"), 1.0),
        ("the delivery changes", "passage", [dict(text="ab ab ab.", voice=S), dict(text="ab ab ab.", voice=S, delivery="pitch=1")], {}, 1.0),
        ("a pause between chains", "passage", [dict(text="ab ab ab.", voice=S), dict(text="ab ab ab.", voice=S, delivery="pitch=1")],
         dict(pause_ms=50.0), 1.0),
        ("opening uh-uh", "wide", text("Uh-uh, ab ab. Nuhuh! ab uhuh ab. Uh uh."), {}, 2.0),
        ("opening uh-uh in later sentences", "wide", text("ab ab ab. Uh-uh. ab ab, nuh-uh ab.\n\nNuh-uh, ab ab ab ab."), {}, 2.0),
        ("typed variants", "wide", text("A-hem. Hehheh, ab! Ah-ha-ha... ab uhuh."), {}, 1.0),
        ("twelve phones", "twelve", text("ab. ab ab ab ab ab ab ab ab. ab ab. ab ab ab ab ab ab ab ab ab ab. ab."), {}, 1.0),
        ("twelve phones in paragraphs", "twelve", text("ab ab.\n\nab ab ab ab ab ab ab.\n\nab."), {}, 1.0),
    ]
    streams = []
    for name, variant, records, options, frames_per_phone in scenarios:
        spec = variants_[variant]
        engine, bundle = engine_of(variant, frames_per_phone, mutate=spec.get("mutate"), **spec["controls"])
        runtime = ScyllasBandRuntime(engine, bundle_dir=bundle, backend="onnx")
        case = dict(name=name, bundle=variant, records=records, options=options, frames_per_phone=frames_per_phone,
                    plan=runtime.plan_records(records).to_dict())
        engine._fake.calls.clear()
        try:
            targets = []
            for event in runtime.synthesize_stream(records, steps=1, seed=0, **options):
                if event.type == "audio_chunk":
                    meta = event.metadata
                    targets.append(dict(chunk_id=event.chunk.chunk_id, text=event.chunk.text, samples=int(event.audio.size),
                                        **{key: meta[key] for key in ("phones", "durations", "latent_frames", "prefix_frames", "pitch_prefix_frames",
                                                                      "delivery_mode", "delivery_set", "delivery", "seed", "context_before",
                                                                      "context_after")}))
            case.update(targets=targets, calls=_graph_calls(engine))
        except ValueError as exc:
            case.update(error=str(exc))
        streams.append(case)

    # --- singing: lyrics as a phrase, cut into the score's words ------------------------------------------------------------
    lines = []
    table = dict(fakes.SungLyricsV5Test.LINES)
    table.update({"i know": [["a", "ɪ"], ["n", "ˈ", "o", "ʊ"]], "I know": [["a", "ɪ"], ["n", "ˈ", "o", "ʊ"]],
                  "for i know": [["f", "ɔ", "ː", "ɹ", "a", "ɪ"], ["n", "ˈ", "o", "ʊ"]], "for": [["f", "ɔ", "ː"]], "i": [["ˈ", "a", "ɪ"]],
                  "know": [["n", "ˈ", "o", "ʊ"]], "i'm i'll i'd i've": [["a", "ɪ", "m"], ["a", "ɪ", "l"], ["a", "ɪ", "d"], ["a", "ɪ", "v"]],
                  "it is i": [["ɪ", "t"], ["ɪ", "z"], ["ˈ", "a", "ɪ"]], "si i": [["s", "i"], ["i"]], "it": [["ˈ", "ɪ", "t"]], "is": [["ˈ", "ɪ", "z"]],
                  "and i i": [["æ", "n", "d", "a", "ɪ", "a", "ɪ"]], "and": [["ˈ", "æ", "n", "d"]],
                  "mm i": [["m", "̩", "ː"], ["ˌ", "a", "ɪ"]], "mm": [["m", "̩", "ː"]],
                  "the the way on": [["ð", "ə", "ð", "ə", "w", "ˈ", "e", "ɪ", "ɒ", "n"]], "see see ah ah": [["s", "ˈ", "i", "ː", "s", "ˈ", "i", "ː", "ɑ", "ː"], ["ɑ", "ː"]]})
    g2p = fakes._line_g2p(table)
    for words, language in ((["to", "me"], "en_us"), (["on", "the", "way"], "en_us"), (["far", "away"], "en_gb"), (["see", "ah"], "en_us"),
                            (["a", "b", "c"], "en_us"), (["i", "know"], "en_us"), (["I", "know"], "en_gb"), (["for", "i", "know"], "en_gb"),
                            (["for", "i", "know"], "de"), (["i'm", "i'll", "i'd", "i've"], "en_us"), (["it", "is", "i"], "en_us"), (["si", "i"], "it"),
                            (["and", "i", "i"], "en_us"), (["mm", "i"], "en_us"), (["the", "the", "way", "on"], "en_us"),
                            (["see", "see", "ah", "ah"], "en_us")):
        try:
            lines.append(dict(words=words, language=language, expected=singing.line_phones(words, g2p, language)))
        except singing.ScoreError as exc:
            lines.append(dict(words=words, language=language, error=str(exc)))
    timing = []
    sounds = ["a", "e", "o", "ᵻ", "ɪ", "b", "t", "s", "m", "ɹ", "ˈ", "ː", "̩", "ˌ"]
    for index in range(300):
        groups = [[rng.choice(sounds) for _ in range(rng.randint(1, 7))] for _ in range(rng.randint(1, 5))]
        phones, starts = singing.word_layout(groups)
        cursor, words = rng.randint(0, 4), []
        for group in groups:
            first = cursor + rng.randint(0, 3)
            cursor = first + rng.randint(1, 20)
            words.append((first, cursor))
        if index % 25 == 0:
            phones[rng.randrange(1, len(phones) - 1)] = "<sil>"   # not one word between two silences (or the same layout)
            groups = None
        total, share = cursor + rng.randint(-3, 6), rng.choice([0.51, 0.42, 0.3, 0.8])
        case = dict(groups=groups, phones=phones, word_starts=starts, words=[[f"w{i}", *span] for i, span in enumerate(words)], total=total, share=share)
        try:
            case.update(expected=singing.score_durations(phones, starts, [fakes._word(f"w{i}", *span) for i, span in enumerate(words)], total, share))
        except singing.ScoreError as exc:
            case.update(error=str(exc))
        timing.append(case)

    # --- singing: plans and songs on the fake bundle ------------------------------------------------------------------------
    modes = json.loads(json.dumps(fakes.V5_DELIVERY_MODES))
    floor = json.loads(json.dumps(modes))
    floor["singing"]["register_floor"] = "F2"
    half = json.loads(json.dumps(modes))
    half["singing"]["guidance_steps"] = "first_half"
    last = json.loads(json.dumps(modes))
    last["singing"]["guidance_steps"] = "last"
    unsaid = json.loads(json.dumps(modes))
    del unsaid["singing"]["guidance"]
    moved = dict(modes, ids=dict(fakes.STYLE_IDS, spoken=4, jazz=0, rap=6))

    def renamed(required, graphs=True):
        def mutate(manifest):
            spec = dict(manifest["controls"]["decoders"].pop("bigvgan"), component="vocoder_wave")
            spec.pop("required", None)
            if required is not None:
                spec["required"] = required
            manifest["controls"]["decoders"]["wave"] = spec
            manifest["controls"]["delivery_modes"]["singing"]["decoder"] = "wave"
            for bucket in manifest["controls"]["target_buckets"]["buckets"]:
                graph = bucket.pop("vocoder_bigvgan")
                if graphs:
                    bucket["vocoder_wave"] = graph
        return mutate

    sung = dict(default=dict(), floor=dict(delivery_modes=floor), half=dict(delivery_modes=half), last=dict(delivery_modes=last),
                unsaid=dict(delivery_modes=unsaid), moved=dict(delivery_modes=moved), wave=dict(mutate=renamed(None)),
                wave_needed=dict(mutate=renamed(True, graphs=False)), wave_spare=dict(mutate=renamed(False, graphs=False)))

    def sung_engine(name):
        spec = dict(sung[name])
        return engine_of("sung_" + name, mutate=spec.pop("mutate", None), **spec)[0]

    score = fakes.SCORE
    low = dict(bpm=1200, notes=[["hello", "C2", 2], ["bob", "C5", 2]])
    scores = dict(pop=score, folk=dict(score, style="folk"), lullaby=dict(score, style="lullaby"), ballad=dict(score, style=" Ballad "),
                  sung=dict(score, style="sung"), rap=dict(score, style="rap"), spoken=dict(score, style="spoken"), low=low,
                  unvoiced=dict(text="hello bob wow how", bpm=1200, style="rock",
                                notes=[["hello", "C4", 2], ["bob", "x", 1], ["-", "x", 1], ["wow~1", "X", 1], ["wow~2", "E4", 1], ["-", "rest", 6],
                                       ["how", " x ", 2]]),
                  nonverbal=dict(bpm=1200, style="pop", notes=[["aha", "C4", 2], ["hello", "D4", 2], ["haha", "E4", 2]]),
                  syllables=dict(text="hellow how", bpm=1200, style="jazz", notes=[["hel~1", "A3", 1], ["low~2", "B3", 1], ["how", "C#4", 2]]),
                  continuous=dict(text="how bob wow hello below allow", bpm=1200, style="soul",
                                  notes=[["how", "C4", 1], ["bob", "D4", 1], ["wow", "E4", 1], ["hello", "F4", 1.5], ["below", "G4", 1], ["allow", "A4", 1]]))
    plans = []
    for bundle in ("default", "floor", "moved"):
        engine = sung_engine(bundle)
        for name, data in scores.items():
            for register, transpose in ((True, 0), (False, 0), (True, -5)):
                case = dict(bundle="sung_" + bundle, name=name, score=data, register=register, transpose=transpose)
                try:
                    plan = singing.plan_song(data, engine, S, "en_us", register=register, transpose=transpose, pad=fakes.PAD)
                    case.update(register_shift=plan.register_shift, mode=plan.mode, mode_id=plan.mode_id, seconds=plan.seconds,
                                phrases=[dict(start=p.phrase.start, end=p.phrase.end, text=p.phrase.text, phones=list(p.sentence.phones),
                                              word_starts=list(p.sentence.word_starts), durations=list(p.durations), frames=p.frames,
                                              notes=[n.name for n in p.phrase.notes],
                                              melody=[float(v) for v in np.asarray(p.melody, np.float32).reshape(-1)]) for p in plan.phrases])
                except ValueError as exc:   # ScoreError, or a phrase longer than the fake buckets
                    case.update(error=str(exc), error_type=type(exc).__name__)
                plans.append(case)
    sings = []
    for bundle, data, kwargs in (("default", dict(score, style="folk"), dict(steps=2)), ("default", score, dict(steps=1, sampler="euler", guidance=1.0)),
                                 ("default", scores["unvoiced"], dict(steps=1, decoder="vocos")), ("half", score, dict(steps=4)),
                                 ("half", score, dict(steps=3, sampler="euler")), ("last", score, {}), ("unsaid", score, {}),
                                 ("unsaid", score, dict(steps=1, guidance=1.0)), ("moved", dict(score, style="jazz"), dict(steps=1)),
                                 ("wave", score, dict(steps=1, guidance=1.0)), ("wave", score, dict(steps=1, guidance=1.0, decoder="vocos")),
                                 ("wave_needed", score, dict(steps=1, guidance=1.0)), ("wave_spare", score, dict(steps=1, guidance=1.0)),
                                 ("floor", low, dict(steps=1))):
        engine = sung_engine(bundle)
        engine._fake.calls.clear()
        case = dict(bundle="sung_" + bundle, score=data, options=kwargs)
        try:
            result = singing.sing(engine, data, S, "en_us", seed=5, pad=fakes.PAD, **kwargs)
            meta = result.metadata
            case.update(calls=_graph_calls(engine), audio=_run_length(result.audio),
                        metadata=dict(decoder=meta["decoder"], steps=meta["steps"], sampler=meta["sampler"], guidance=meta["guidance"],
                                      guidance_steps=meta.get("guidance_steps"), mode=meta["mode"], mode_id=meta["mode_id"],
                                      register_shift=meta["register_shift"], seconds=meta["seconds"],
                                      phrases=[dict(text=p["text"], frames=p["frames"], bucket=p["bucket"], seed=p["seed"], seconds=p["seconds"],
                                                    durations=p["durations"]) for p in meta["phrases"]]))
        except ValueError as exc:
            case.update(error=str(exc))
        sings.append(case)
    styles = {"sung_" + name: singing.singing_styles(sung_engine(name)) for name in ("default", "moved")}
    return dict(spellings=spellings, opening=opening, phonemize=phonemize, refused=refused, bundles=bundles, streams=streams, lines=lines,
                line_table=table, timing=timing, plans=plans, sings=sings, styles=styles, identity=fakes.IDENTITY)


def _graph_calls(engine) -> list[dict]:
    """What the fake graphs were called with, the G2P aside: the component, the style row (style_id or delivery_mode_id),
    whether intensity and pace were set, the identity vector's first value, and for flow calls whether the melody was all
    zero and the time."""
    calls = []
    for call in engine._fake.calls:
        name = call["name"]
        if name.startswith("g2p"):
            continue
        row = dict(name=name)
        mode = call.get("delivery_mode_id", call.get("style_id"))
        if mode is not None:
            row.update(mode=int(mode[0]))
        if "delivery_present" in call:
            row.update(present=bool(call["delivery_present"].any()))
        if "identity_reference" in call:
            row.update(identity=float(call["identity_reference"][0, 0]))
        if name.startswith("vector_estimator"):
            row.update(melody_zero=bool((call["expanded_melody"] == 0).all()), time=float(call["time"][0]))
        calls.append(row)
    return calls


def normalizer_cases() -> list[dict]:
    corpus = ["Meet me at 9 a.m. Then we leave.", "It ends at 5 p.m.", "Dr. Smith met Mrs. Jones on Main Dr. today.",
              "I paid $3.50, then £12 and 40€.", "On Jan. 5, 2024 we had 25% more; by 3/4/2025 it was 1/2.", "Call me at 10:30pm or 7:05 AM.",
              "She came 1st and 22nd.", "Email jane.doe+tts@example.co.uk or ping @scylla.", "Ok... OK?! ok!!", "Well -- that's it - right?", "It was only a\u2014dream. Well--no; \u00e9t\u00e9\u2014hiver, wait\u2014 and a \u2e3a b.",
              "e-mail, X-ray, T-shirt and A - B.", "Café café 2024-03-07.", "It's 3.14159 or 2,718.28 or 1,000,000.", "x = 5 and 20°",
              "12345678901234567890123 and ٣٤", "D.J. Stumpy charged $99.99. I met J.K. Rowling.", "He moved to the U.S. Then he left.",
              "She lives in the U.S.", 'Call me at 9 a.m. "Sure," he said.',
              "Anni 108, 180, 1880 e il 0º; el 0.º y 0ª.", "Whisper Large V3 beat v3.5 and v2.", "Play the MP3 in 4K on a 3D TV; H2O, B2B, 2FA and e2e.", "GPT4 on an RTX4090 and an A100, Python3, COVID19, iPhone15, x86 at 1080p and 60fps.", "Run it at 10am or 5 PM; the 1990s, the '80s, her 20s, the 2000s; wait 30s.", "On March 4, 1990 and 3/4/2005; test_voice2.py, user_42 and 3.5GB of 16GB.", "100s of 4s, two MP3s, Route 66's history, the 1500s and 1.5s; 68x faster on a 4x4 grid at 1920x1080, 1.5x, 100ms, 24 kHz, 1 GB, 5 mins and 30 mph.", "Edge cases: 4x4.5, 4x45a, 2X, 1.5.s, '1.5s, '1990s, 6's, 0x1F, 1,000s, 10 ms. and 1 fps.", "1,000s of people and 10,000s more, 1,000 ms, 1,500.5x and 1,0005s.",
              # Initials, abbreviations and titles.
              "Kenneth A. Schmied met W. C. Handy in Washington D. C. in 1939-1945.", "He chose plan B. The others left 3-2.",
              "Er kam in Gruppe B. Die anderen gingen.", "Washington D. C. debe su nombre", "J. C. Bach.", "D. K. Arya sagte", "Selon M. Pittman",
              "Inspekteur D. K. Arya fuhr 480 km/h, 1 km weit, mit 90 kg und 30 % Last.", "I. M. Pei and I. Then B. He left. Ask for A. Smith or J. R. R. Tolkien.",
              "Dr. Lee und John Baldwin Jr. fuhren nach St. Louis.", "Mme Curie, Mlle Dupont, le Dr. Damadian et Prof. Rossi; Dra. Ruiz, Dott. Bianchi e il Sig. Verdi.",
              "EE. UU.", "EE.UU.", "Los EE UU y EE. UU. Luego", "z.B.", "z. B.", "usw. Der Rest", "etc.", "etc., bzw. ca. 5, Nr. 7, ggf. evtl. inkl. bspw. d. h. u. a. hier",
              "av. J.-C.", "n.º 11", "le n° 5, nº 6", "Sig.ra Rossi", "En el 44 a. C. y el 14 d. C. Roma, p. ej. Ud. y Uds. con la Srta. Paz, el Sr. Gil y la Sra. Gil.",
              "Nel 44 a.C. e nel 14 d.C. Roma, ecc. Poi Jr. e Sr.", "La SNCF et le Dr. Damadian, 5 000 ans av. J.-C.", "Metc. etc.x a.etc.",
              # Initialisms in capitals and Roman numerals.
              "Das UN-Mitgliedsland nutzt z.B. UMTS bzw. LTE usw. Der NDR und die NATO senden.", "ACHTUNG, DIE SPD TAGT.",
              "J. C. Bach y el NDP: 104 500 barriles a 11 km/h, el n.º 11.", "La ONU, la UE y el PIB; l'ONG, la TVA, le B-52, FBI-4 et un TGV's.",
              "papa Giovanni XXIII e Elisabetta II", "el rey Juan Carlos I y Alfonso XIII", "la reina Isabel II.", "François Ier et Napoléon III", "XVe siècle",
              "siglo IX", "nel XV secolo", "secolo XV", "Au XVIIIe siècle, François Ier et Elizabeth II roulaient à 56-64 km/h, etc.",
              "En los EE. UU. del siglo XX, la reina Isabel II y Alfonso XIII, etc.", "Nel XV secolo Elisabetta II, e/o Kier Starmer QC, a 11 km/h, ecc.",
              "La Ire armée, le IIe et la XXIème; Luis XIV, Pío XII, Benedetto XVI, (Carlo V), secolo XXX, MIX, Vitamin XI e Maria I.",
              "World War II ended in Queen Elizabeth II's youth; Gordon R. Dickson, Malcolm X.", "Le TGV, une ONG et le NPWS.", "Pagine 21, 38 e 45.",
              "Henry VIII’s six wives, Louis XIV'sx, Rocky IV, Rocky IVs, part II, Apollo XIII. MARK II AND Mark XL, Type XXXIX-b, 28, 81, 88 e 121.",
              # Grouped numbers, ranges, clock times, percent, units, slashes and degrees.
              "Appelez le 0800 123 456 vite", "capacidad de 104 500 barriles", "5 000 000 de visiteurs", "1 000 et 12 345 678 mais 1 2345, 1234 567 et 1 000 5",
              "4,2-3,9 milioni", "(1644-1912)", "1995- 96", "56-64 km/h", "7-2", "pages 10-15", "555-1234", "4.2-3.9 million, 0-5, 5-05, 3 - 4, 1-2-3, 2-3.5.1 and 10-20%",
              "Selon M. Pittman, 31 % des 5 000 000 de visiteurs partent à 7 h 19 ou 21 h.", "1 h 05", "21 h", "25 h 99", "7h30, 0 h, 7 h 195, 123 h et 7 hommes",
              "31 %", "31%", "bis zu 480 km/h (133 m/s; 300 MPH)", "1 km", "1 m, 2 m/s, 3 mm, 4 cm, 5 kg, 6 mph, 7 km/hx, 8 m/s2 and 1,5 km",
              "Drive to/from work at 40 mph (64 km/h) in 30 °C heat.", "5 kph, 1 km/h and 3 m/s", "to/from", "and/or", "e/o", "seiner/ihrer", "a / b, 1/x, x/2 and n/a/b",
              "30 °C", "90 °F", "20°C, 5 ° F, 30 °Cx and 40 °K", "A *bold* word * and a footnote*.",
              # Non-verbal words: one spelling by shape, whole tokens, never after a number.
              "Hm, I had not thought of that. Hhhmmm... maybe. HMMMM.", "The HMM model, H.M.M., M.H.M. and S.H.H.", "HMMMM.", "MM-HMM, SHHH! UH HUH.",
              "Mm-hmm, go on; mm hmm, mmhm, MM-HMM, mmm hmmm, mm-hm, mhmm and m-hm.", "Uh huh, sure. UH-HUH! uhhuh, uhuh, uh-uh, uh-oh and uhhh-oh.",
              "Shhh, listen. Sssh! Sh! Quiet. Sh is a symbol? Shh. sh, Sh... shhhip and sshh.", "Uhh, give me a second, ummm, or two. UM, UH, Uh.",
              "Errr, no. Errm, yes. I err on purpose. We all err. Er sagte, er, erm; the ER. ER, ERM!", "Heh, hehh, heeeh, he, heh's. Tssk tsk, TSK.",
              "Mmm, that is good. Mm? mm, m, Mmmmmmmmmm.", "The bolt is 5 mm wide, the film 35 MM, the gap 2mm and 5 hmm, 3 uh, 1.5 um.",
              "Call at 9 a.m. Hmm, then go, etc. Uhh, fine. Dr. Hmm and Mrs. Umm.", "'Hmm,' she said. \"Uhh\" Well...hmmm. (hm) [mhm] hmm—uh\u2014um",
              "I hummed to mom; um's, main.mm, src/mm, mm/s, mm-wave, hmm_x, hmm2, 2hmm, ühm, hmmé, .hmm ..hmm ...hmm hmm. hmm.x hmm..x",
              "Aha, there. Ah-hah! Ah hah, I see. Ahah. AH-HA, ahhh haaa! ahha aha's a-ha a ha, A-ha. a hah of joy; ah, ha. ah  ha ah - ha",
              "Ha ha, the Sahara, Utah-haters, aah, ah, aha-moment, 3 aha, ah ha ha ha, hah ah hah",
              "Listen, Stephen, I have the keys. Mary I think so. John I am sure. Queen Elizabeth I was crowned. Charles I of England, under James I.",
              "Henry I's court, Henry I 's, Stephen, II and Henry VIII; George I) Paul I\" Peter I offer, Louis I ofter, William I", "Pope John Paul I said. Mary, I. Mary I",
              "mm hmm hmm", "uh huh huh uh", "mm  hmm", "mm - hmm", "uh\u2010huh and mm\u2011hmm", "hmm@uhh.com and @umm", "HMM", "Hmm", "u h, h m, s h h",
              # Table entries in context: abbreviations, titles, symbols, currency with scales, units, ordinals and dates.
              "Dr. Smith met Mrs. Jones on Main Dr. today, e.g. at 5 p.m. etc.", "DR. SMITH AND MR JONES, ETC. THE FBI SAID.",
              "Mr Smith & Co. paid $3.5 billion, £2bn and $5m to Smith Bros. Inc. in 2020.",
              "See No. 5, pp. 12–15 and § 3; ~5 km away at 40 mph, 1 kg, 1.5 kg, 10 lbs, 3 ft and 5 m.", "Max. 5 people. I saw Max. Then he left.",
              "Queen Elizabeth II's reign, Henry VIII, World War II, Rocky II and Pope John Paul II; King George V.",
              "St Paul's and Baker St. The St. Louis arch.", "It was 1939–1945 and the score was 7–2.", "vs. vs VS and aka AKA a.k.a. w/ w/o",
              "Dr. Lee, z.B. Herr Müller u. a., fuhr am 3. Oktober 1990 mit 120 km/h, 1 t und 1 km. Usw. Der Rest.",
              "Am 03.10.2026 kamen ca. 5 Mio. Menschen, d. h. 21 % mehr; der 2. Weltkrieg; S. 12; Nr. 5; 1 h.",
              "Die EU, die USA, die UN, die CDU, die ICE-Strecke und die AfD; die GmbH & Co. KG.", "Er war 5. Danach ging er. 1. Einleitung",
              "Selon M. Pittman, le Dr Martin et Mme Dupont, n° 5, au XVIIIe siècle, le 1er mai et la 2e fois, 21 h 05.",
              "Il pèse 1 t, 21 t, 1,5 km, 2 km; c.-à-d. 5 % du PIB, l'UE et les USA, etc. Fin.",
              "El Sr. García y la Dra. López, en EE. UU., el siglo XX, la reina Isabel II, Alfonso XIII, el 1.º y la 3.ª.",
              "Pesa 1 t, 21 t, 21 km, 31 km, 1 km; p. ej., 5 % del PIB, n.º 11, pág. 4, C/ Mayor s/n.",
              "Il Sig. Rossi e la Sig.ra Bianchi, Dott.ssa Verdi, nel XV secolo, papa Giovanni XXIII, 1º maggio, 23, 38, 135, 2023, 103.",
              "A 11 km/h, 1 t, 21 km, ecc., p. es. il 5 % e 3,5 mln.",
              "TP.HCM và TP. Hà Nội, UBND, THPT, GS.TS. Nguyễn Văn An, 50.000đ, 5h30, 30°C, Q. 1, v.v.", '" , Carlo VI e ( , Carlos VI',
              # Minus signs, clock times, French decimals and thousands.
              "Il fait -3,5 °C à 15:00, 15 h 00, 0 h 15 et 00:05; x = -2, −4, (-7), 10 -12, 5-3 et 2026-11-24.",
              "3,14, 0,05, 21,75 %, 9,30, 2,125 kg, 1,000,000, 1,000 € et 1 500 kWh; 80 000, 200 000, 1 280 000 et 5,1,2.",
              "Um 1:00 und 21:15; a las 15:00; alle 15:00 e 3:30; IL FAIT -3 DEGRÉS.", "-3,5 °C",
              # "One" agreeing with the noun after it, dates by dialect, times, decimals and year ranges by language.
              "Tienes 1 mensaje nuevo y 1 alarma; 21 años, 31 días, 21 casas, 101 perros, 21 000 personas, 31 millones; el capítulo 1 "
              "sección 2, 1 de cada 3, 1 vez y 1 día.",
              "Hai 1 nuovo messaggio, 1 studente, 1 ora, 1 casa, 1 milione; capitolo 1 sezione 2; il 1 settembre alle 15:30; 2.500.000.000.",
              "In 1 Stunde: 1 neue Nachricht, 1 Termin, 1 Million, um 1 Uhr, seit 1 Jahr, mit 1 neuen Nachricht, mit 1 neuen Termin, 12:00 Uhr und 00:30 Uhr; Kapitel 1 Abschnitt 2; 1939-1945.",
              "On 1/9/2026, 24/11/2026, the 1st September, The 1st May, 12 May-14 May; at 15:00, 00:00 and 3:00 PM; 5 may be.",
              "Ngày 1/9/2026, giá 41 093,38 $, 3,14, 0,05, 1500 kWh và lúc 15:30 giờ.", "3,14 et 0,05; 1,75 m; le 1 Septembre; 1 Mai."]
    cases = []
    for language in ("en_us", "en_gb", "es", "it", "fr", "de", "vi"):
        for text in corpus:
            cases.append(dict(text=text, language=language, expected=normalize_spoken_text(text, language=language)))
    for language, text in text_data_cases():
        cases.append(dict(text=text, language=language, expected=normalize_spoken_text(text, language=language)))
    return cases


def text_data_cases() -> list[tuple[str, str]]:
    """Every written form of every table entry (scyllasband/text_data), in its own language: as written, in lower and upper
    case, in a sentence written all in capitals, capitalised at the start of a sentence, and without the space after a dot."""
    sentences = {"abbreviations": lambda form: f"aa {form} bb", "titles": lambda form: f"aa {form} Zorro bb",
                 "before_number": lambda form: f"aa {form} 7 bb", "units": lambda form: f"aa 7 {form} bb"}
    openings = {"abbreviations": lambda form: f"{form} bb", "titles": lambda form: f"{form} Zorro bb",
                "before_number": lambda form: f"{form} 7 bb", "units": lambda form: f"7 {form} bb"}
    cases = []
    for lang, data in TEXT_DATA.items():
        language = "en_us" if lang == "en" else lang
        for section, sentence in sentences.items():
            for entry in data.get(section, ()):
                for form in forms(entry["written"]):
                    texts = [sentence(form), sentence(form).upper(), sentence(form.lower()), sentence(form.upper()),
                             openings[section](form[:1].upper() + form[1:]), sentence(form.replace(". ", "."))]
                    cases.extend((language, text) for text in dict.fromkeys(texts))
    return cases


def unicode_cases() -> list[dict]:
    samples = ["ÀÉÎÕÜ Ñ İstanbul ΣΑΣ Москва ẞ", "é Å Å ẛ̣ q̣̇ 각", "Việt Nam", "Straße ﬁne ŉ ΐ ǆǅ ı ſ µ ᾳ"]
    return [dict(text=text, lower=text.lower(), upper=text.upper(), nfc=unicodedata.normalize("NFC", text)) for text in samples]


def main() -> None:
    rng = random.Random(20261005)
    OUT.mkdir(parents=True, exist_ok=True)
    fixtures = dict(events=dict(token_to_id=TOKEN_TO_ID, cases=events_cases(rng)), durations=dict(cases=durations_cases(rng)),
                    context=dict(cases=context_cases(rng)), delivery=dict(cases=delivery_cases()), text=text_cases(),
                    normalizer=dict(cases=normalizer_cases()), unicode=dict(cases=unicode_cases()),
                    targets=dict(choose=choose_cases(rng), streams=stream_cases()), singing=singing_cases(), records=records_cases(),
                    v5=v5_cases(random.Random(20261123)))
    for name, data in fixtures.items():
        (OUT / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=None, separators=(",", ":")) + "\n", encoding="utf-8")
        print(f"{OUT / name}.json")


if __name__ == "__main__":
    main()
