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

from scyllasband.delivery import delivery_spec, delivery_tensors
from scyllasband.engine import balanced_context
from scyllasband.events import frame_events, frames_to_durations, modifier_bits
from scyllasband.g2p import punctuated_segments
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
    for _ in range(40):
        phones = [rng.choice(ACOUSTIC + MODIFIERS + PUNCTUATION + ["<sil>"]) for _ in range(rng.randint(1, 20))]
        values = [rng.choice([rng.uniform(-1, 8), float(rng.randint(0, 6)) + 0.5, 2.5, 0.5, 1.5, 0.4999]) for _ in phones]
        values = [float(f"{v:.4f}") if rng.random() < 0.5 else v for v in values]
        import numpy as np
        values = np.asarray(values, np.float32).tolist()
        scale = rng.choice([1.0, 1 / 0.75, 1 / 1.25, 1 / 1.1])
        cases.append(dict(phones=phones, values=values, scale=scale, expected=frames_to_durations(values, phones, scale=scale)))
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
    specs = [None, "neutral", " Neutral ", "auto", "AUTO", "energy=3", "energy=2.5,tension=2,valence=3,assertiveness=1,whisper=off", "whisper=on",
             "energy=auto,whisper=auto", "energy=0,tension=4", " energy = 3", "energy=3 , valence=1.25", "energy=5", "energy=-1", "energy=abc",
             "energy=3,energy=2", "calm=2", "whisper=yes", "whisper=On", "energy", "", "energy=1e0", "energy=.5", "energy=2.", "energy=nan",
             "energy=inf", "valence=Auto"]
    cases = []
    for spec in specs:
        try:
            values, present, resolved = delivery_tensors(spec)
            cases.append(dict(spec=spec, error=False, values=values[0].tolist(), present=present[0].astype(int).tolist(),
                              resolved=resolved))
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
                                 expected=[[segment, list(tokens)] for segment, tokens in punctuated_segments(text, max_chars=max_chars)]))
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
        ("chains", [dict(text="ab.", voice="scylla"), dict(text="ab.", voice="scylla", delivery="whisper=on")], 1.0),
        ("long paragraph", text("ab ab. ab ab.\n\nab ab."), 1.0),
        ("clauses", text("ab ab, ab ab ab; ab ab: ab ab ab, ab. ab ab ab ab, ab ab ab ab ab. ab, ab ab ab ab ab ab, ab ab."), 1.0),
        ("clauses fast", text("ab ab, ab ab ab; ab ab: ab ab ab, ab. ab ab ab ab, ab ab ab ab ab. ab, ab ab ab ab ab ab, ab ab."), 0.5),
        ("words", text(" ".join(["ab"] * 40)), 0.6),
        ("mixed", [dict(text="ab ab ab. ab, ab.\n\nab ab ab ab ab ab ab ab ab ab ab ab ab.", voice="scylla"),
                   dict(text="ab. ab ab ab ab ab ab ab, ab ab ab.", voice="scylla"),
                   dict(text="ab ab ab ab ab.", voice="scylla", delivery="energy=3")], 1.0),
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


def normalizer_cases() -> list[dict]:
    corpus = ["Meet me at 9 a.m. Then we leave.", "It ends at 5 p.m.", "Dr. Smith met Mrs. Jones on Main Dr. today.",
              "I paid $3.50, then £12 and 40€.", "On Jan. 5, 2024 we had 25% more; by 3/4/2025 it was 1/2.", "Call me at 10:30pm or 7:05 AM.",
              "She came 1st and 22nd.", "Email jane.doe+tts@example.co.uk or ping @scylla.", "Ok... OK?! ok!!", "Well -- that's it - right?",
              "e-mail, X-ray, T-shirt and A - B.", "Café café 2024-03-07.", "It's 3.14159 or 2,718.28 or 1,000,000.", "x = 5 and 20°",
              "12345678901234567890123 and ٣٤", "D.J. Stumpy charged $99.99. I met J.K. Rowling.", "He moved to the U.S. Then he left.",
              "She lives in the U.S.", 'Call me at 9 a.m. "Sure," he said.']
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
                    targets=dict(choose=choose_cases(rng), streams=stream_cases()))
    for name, data in fixtures.items():
        (OUT / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=None, separators=(",", ":")) + "\n", encoding="utf-8")
        print(f"{OUT / name}.json")


if __name__ == "__main__":
    main()
