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

from scyllasband.delivery import delivery_spec, resolve_delivery
from scyllasband.engine import balanced_context
from scyllasband.events import frame_events, frames_to_durations, modifier_bits
from scyllasband.g2p import _marked_segments, punctuated_segments
from scyllasband.g2p_phrases import normalize_g2p_phrase_punctuation
from scyllasband.planner import _PARAGRAPH_RE, plan_records, records_from_text, split_for_retry, split_sentences
from scyllasband.streaming import StreamOptions, _choose, synthesize_plan_stream
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


def normalizer_cases() -> list[dict]:
    corpus = ["Meet me at 9 a.m. Then we leave.", "It ends at 5 p.m.", "Dr. Smith met Mrs. Jones on Main Dr. today.",
              "I paid $3.50, then £12 and 40€.", "On Jan. 5, 2024 we had 25% more; by 3/4/2025 it was 1/2.", "Call me at 10:30pm or 7:05 AM.",
              "She came 1st and 22nd.", "Email jane.doe+tts@example.co.uk or ping @scylla.", "Ok... OK?! ok!!", "Well -- that's it - right?", "It was only a\u2014dream. Well--no; \u00e9t\u00e9\u2014hiver, wait\u2014 and a \u2e3a b.",
              "e-mail, X-ray, T-shirt and A - B.", "Café café 2024-03-07.", "It's 3.14159 or 2,718.28 or 1,000,000.", "x = 5 and 20°",
              "12345678901234567890123 and ٣٤", "D.J. Stumpy charged $99.99. I met J.K. Rowling.", "He moved to the U.S. Then he left.",
              "She lives in the U.S.", 'Call me at 9 a.m. "Sure," he said.',
              "Whisper Large V3 beat v3.5 and v2.", "Play the MP3 in 4K on a 3D TV; H2O, B2B, 2FA and e2e.", "GPT4 on an RTX4090 and an A100, Python3, COVID19, iPhone15, x86 at 1080p and 60fps.", "Run it at 10am or 5 PM; the 1990s, the '80s, her 20s, the 2000s; wait 30s.", "On March 4, 1990 and 3/4/2005; test_voice2.py, user_42 and 3.5GB of 16GB.", "100s of 4s, two MP3s, Route 66's history, the 1500s and 1.5s; 68x faster on a 4x4 grid at 1920x1080, 1.5x, 100ms, 24 kHz, 1 GB, 5 mins and 30 mph.", "Edge cases: 4x4.5, 4x45a, 2X, 1.5.s, '1.5s, '1990s, 6's, 0x1F, 1,000s, 10 ms. and 1 fps.", "1,000s of people and 10,000s more, 1,000 ms, 1,500.5x and 1,0005s."]
    cases = []
    for language in ("en_us", "en_gb", "es", "it", "fr", "de", "vi"):
        for text in corpus:
            cases.append(dict(text=text, language=language, expected=normalize_spoken_text(text, language=language)))
    return cases


def unicode_cases() -> list[dict]:
    samples = ["ÀÉÎÕÜ Ñ İstanbul ΣΑΣ Москва ẞ", "é Å Å ẛ̣ q̣̇ 각", "Việt Nam"]
    return [dict(text=text, lower=text.lower(), nfc=unicodedata.normalize("NFC", text)) for text in samples]


def main() -> None:
    rng = random.Random(20261005)
    OUT.mkdir(parents=True, exist_ok=True)
    fixtures = dict(events=dict(token_to_id=TOKEN_TO_ID, cases=events_cases(rng)), durations=dict(cases=durations_cases(rng)),
                    context=dict(cases=context_cases(rng)), delivery=dict(cases=delivery_cases()), text=text_cases(),
                    normalizer=dict(cases=normalizer_cases()), unicode=dict(cases=unicode_cases()),
                    targets=dict(choose=choose_cases(rng), streams=stream_cases()), singing=singing_cases(), records=records_cases())
    for name, data in fixtures.items():
        (OUT / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=None, separators=(",", ":")) + "\n", encoding="utf-8")
        print(f"{OUT / name}.json")


if __name__ == "__main__":
    main()
