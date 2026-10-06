"""Runtime tests that need no model files: fake graphs stand in for the bundle's sessions."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scyllasband.contract import (BundleError, CONTEXT_INPUTS, DURATION_INPUTS, GRAPH_CONTRACT, VECTOR_INPUTS, VOCODER_INPUTS,
                                  validate_bundle_layout)
from scyllasband.delivery import delivery_tensors, resolve_delivery
from scyllasband.engine import Engine, OverlongError, Sentence, balanced_context
from scyllasband.events import frame_events, frames_to_durations, modifier_bits, sentence_types
from scyllasband.g2p import punctuated_segments
from scyllasband.planner import parse_group_lines, plan_records, records_from_text, split_for_retry, split_sentences
from scyllasband.streaming import StreamOptions, synthesize_plan_stream

VOCAB = ["<pad>", "<unk>", "<sil>", "<pause_comma>", "<pause_semicolon>", "<pause_colon>", "<pause_dash>", "<ellipsis>", "<end_stmt>",
         "<end_question>", "<end_exclaim>", "a", "b", "h", "l", "o", "w", "ˈ", "ː"]
PHONE_TO_ID = {p: i for i, p in enumerate(VOCAB)}


class PunctuationTest(unittest.TestCase):
    def test_segments_carry_their_punctuation(self):
        self.assertEqual(punctuated_segments("Wait — what was that? I heard it, too."),
                         [("Wait", ["<pause_dash>"]), ("what was that", ["<end_question>"]), ("I heard it", ["<pause_comma>"]),
                          ("too", ["<end_stmt>"])])

    def test_stacked_marks_stay_on_the_previous_phrase(self):
        self.assertEqual(punctuated_segments("for Death, — He kindly stopped —"),
                         [("for Death", ["<pause_comma>", "<pause_dash>"]), ("He kindly stopped", ["<pause_dash>"])])

    def test_a_sentence_end_wins_over_other_stacked_marks(self):
        self.assertEqual(punctuated_segments("Really?! she asked."), [("Really", ["<end_question>"]), ("she asked", ["<end_stmt>"])])

    def test_long_phrases_split_between_words_without_punctuation(self):
        segments = punctuated_segments("one two three four five six.", max_chars=10)
        self.assertEqual([s for s, _ in segments], ["one two", "three four", "five six"])
        self.assertEqual([t for _, t in segments], [[], [], ["<end_stmt>"]])


class FakeG2PTest(unittest.TestCase):
    def test_layout_word_starts_and_punctuation_silences(self):
        engine = _fake_engine()
        result = engine.phonemize("hello bob, ah.", language="en_us")
        phones = result["phones"]
        self.assertEqual(phones[0], "<sil>")
        self.assertEqual(phones[-1], "<sil>")
        self.assertIn("<pause_comma>", phones)
        comma = phones.index("<pause_comma>")
        self.assertEqual(phones[comma + 1], "<sil>")
        self.assertEqual([phones[i - 1] for i in result["word_starts"]], ["<sil>", "<sil>"])   # each later word follows a <sil>
        self.assertEqual(len(result["word_boundary_candidates"]), 1)                          # only between hello and bob


class DurationAndEventTest(unittest.TestCase):
    def test_rounding_matches_training(self):
        phones = ["<sil>", "ˈ", "a", "<pause_comma>", "<sil>", "b", "<sil>"]
        self.assertEqual(frames_to_durations([0.4, 3, 0.2, 5, 2.5, 3.5, 0.0], phones), [0, 0, 1, 0, 2, 4, 0])

    def test_events(self):
        phones = ["<sil>", "ˈ", "a", "b", "<pause_comma>", "<sil>", "h", "ː", "<end_stmt>", "<sil>"]
        durations = [2, 0, 3, 2, 0, 3, 2, 0, 0, 2]
        bits = modifier_bits(PHONE_TO_ID)
        events = frame_events(phones, durations, word_starts=[6], following=[], token_to_id=PHONE_TO_ID, bits=bits)
        boundary = events["expanded_boundary_event_ids"]
        self.assertTrue((boundary[7:10] == PHONE_TO_ID["<pause_comma>"]).all())     # the silence after the comma
        self.assertTrue((boundary[10:12] == len(PHONE_TO_ID)).all())                # word start: next word's first phone
        self.assertTrue((boundary[12:14] == PHONE_TO_ID["<end_stmt>"]).all())       # the final silence
        modifier = events["expanded_modifier_event_ids"]
        self.assertTrue((modifier[2:5] == bits["ˈ"]).all())                        # stress -> next phone
        self.assertTrue((modifier[10:12] == bits["ː"]).all())                      # length -> preceding phone
        np.testing.assert_allclose(events["expanded_phone_phase"][2:5], [1 / 6, 0.5, 5 / 6], rtol=1e-6)
        types = events["expanded_sentence_type_ids"]
        self.assertTrue((types[:-2] == 1).all())   # up to the statement end
        self.assertTrue((types[-2:] == 0).all())   # the silence after it belongs to the next sentence (none here)

    def test_zero_frame_silence_moves_the_event_to_the_next_phone(self):
        phones = ["<sil>", "a", "<pause_comma>", "<sil>", "b", "<sil>"]
        events = frame_events(phones, [1, 2, 0, 0, 2, 1], word_starts=[4], following=[], token_to_id=PHONE_TO_ID, bits={})
        self.assertEqual(events["expanded_boundary_event_ids"].tolist(), [0, 0, 0, PHONE_TO_ID["<pause_comma>"], 0, 0])

    def test_sentence_type_reads_the_following_context(self):
        self.assertEqual(sentence_types(["a", "<pause_comma>", "b"], ["c", "<end_question>"]), [2, 2, 2])
        self.assertEqual(sentence_types(["a"], []), [0])


class ContextTest(unittest.TestCase):
    def test_balanced_context(self):
        span, segments = balanced_context(list(range(10)), [99] * 4, list(range(20, 30)), 10)
        self.assertEqual(segments, [0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
        self.assertEqual(span[:3], [7, 8, 9])
        self.assertEqual(span[-3:], [20, 21, 22])

    def test_spare_room_goes_to_the_longer_side(self):
        span, segments = balanced_context([1, 2], [9], list(range(20, 30)), 8)
        self.assertEqual(segments, [0, 0, 1, 2, 2, 2, 2, 2])


class NormalizerTest(unittest.TestCase):
    def test_decorative_asterisks_are_not_spoken(self):
        from scyllasband.text_normalizer import normalize_spoken_text
        self.assertEqual(normalize_spoken_text("* * * * * The Pool of Tears. It was *very* odd.*", language="en_gb"),
                         "The Pool of Tears. It was very odd.")


class InitialismTest(unittest.TestCase):
    def test_names_stay_in_their_sentence_and_times_can_end_one(self):
        from scyllasband.text_normalizer import normalize_spoken_text
        self.assertEqual(normalize_spoken_text("D.J. Stumpy met J.K. Rowling.", language="en_us"), "dee jay Stumpy met jay kay Rowling.")
        self.assertEqual(normalize_spoken_text("We left at 5 p.m. Then it rained.", language="en_us"),
                         "We left at five pee em. Then it rained.")
        self.assertEqual(normalize_spoken_text("She lives in the U.S.", language="en_us"), "She lives in the you ess.")


class PlannerTest(unittest.TestCase):
    def test_sentence_split(self):
        self.assertEqual(split_sentences('You. Are. Not. It. "Really?!" she asked... Fine'),
                         ["You.", "Are.", "Not.", "It.", '"Really?!"', "she asked...", "Fine"])

    def test_retry_split_prefers_clause_punctuation(self):
        self.assertEqual(split_for_retry("one two three, four five six seven"), ["one two three,", "four five six seven"])

    def test_retry_split_leaves_each_piece_a_minimum_share(self):
        self.assertEqual(split_for_retry("Well, one two three four five six"), ["Well,", "one two three four five six"])
        self.assertEqual(split_for_retry("Well, one two three four five six", min_share=0.25), ["Well, one two", "three four five six"])

    def test_chains_follow_voice_language_and_delivery(self):
        records = parse_group_lines("[scylla:en_us] Hi there. How are you?\n[ink:en_gb:energy=3] Fine. [scylla] Good.",
                                    default_voice=None, default_language=None)
        plan = plan_records(records, resolve_language=lambda voice, language: language or "en_us", normalize=False)
        self.assertEqual([c.chain_id for c in plan.chunks], ["chain_001", "chain_001", "chain_002", "chain_003"])
        self.assertEqual(plan.chunks[2].delivery["energy"], 3.0)


class DeliveryTest(unittest.TestCase):
    def test_neutral_default_and_auto(self):
        values, present, _ = delivery_tensors(None)
        self.assertTrue(present.all())
        self.assertEqual(values[0, :4].tolist(), [0, 0, 0, 0])
        values, present, _ = delivery_tensors("auto")
        self.assertFalse(present[0, :4].any())
        self.assertEqual(resolve_delivery("energy=3")["energy"], 3.0)
        self.assertEqual(resolve_delivery(""), resolve_delivery(None))

    def test_whisper_input_is_always_off(self):
        for spec in (None, "auto", "energy=3,valence=1"):
            values, present, resolved = delivery_tensors(spec)
            self.assertEqual((float(values[0, 4]), bool(present[0, 4])), (0.0, True))
            self.assertNotIn("whisper", resolved)
        for spec in ("whisper=on", "energy=2,whisper=off", {"whisper": False}):
            with self.assertRaisesRegex(ValueError, "Whisper is not supported"):
                resolve_delivery(spec)


class ContractTest(unittest.TestCase):
    def test_valid_and_invalid_bundles(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = _fake_bundle(Path(tmp))
            summary = validate_bundle_layout(bundle)
            self.assertEqual(summary["buckets"], [8, 16])
            manifest = json.loads((bundle / "manifest.json").read_text())
            manifest["components"]["vector_estimator"]["inputs"] = list(VECTOR_INPUTS[:-1])
            (bundle / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(BundleError):
                validate_bundle_layout(bundle)


class EngineTest(unittest.TestCase):
    def test_bucket_padding_prefix_and_decode(self):
        engine = _fake_engine()
        sentence = Sentence(phones=["<sil>", "a", "b", "<end_stmt>", "<sil>"], word_starts=[], before_ids=[11, 12], after_ids=[13])
        delivery = delivery_tensors(None)[:2]
        result = engine.synthesize_sentence(sentence, voice="scylla", language="en_us", delivery=delivery, steps=2,
                                            rng=np.random.default_rng(0), prefix=np.ones((24, 3), np.float32))
        self.assertEqual(result.latents.shape, (24, sum(result.durations)))
        self.assertEqual(result.bucket, 8)
        self.assertEqual(result.metadata["prefix_frames"], 3)
        calls = engine._fake.calls
        self.assertTrue(all(call["noise"].shape == (1, 24, 8) for call in calls if call["name"] == "vector_estimator_8"))
        audio = engine.decode(result.latents, voice="scylla", language="en_us", left=np.zeros((24, 20), np.float32))
        self.assertEqual(audio.shape, (result.latents.shape[1] * 512,))

    def test_decode_gives_left_context_way_to_long_sentences(self):
        engine = _fake_engine()
        left = np.zeros((24, 20), np.float32)
        for frames in (15, 16):   # 16 fills the largest bucket, so its last 256 samples are padding
            audio = engine.decode(np.zeros((24, frames), np.float32), voice="scylla", language="en_us", left=left)
            self.assertEqual(audio.shape, (frames * 512,))
            self.assertEqual(engine._fake.calls[-1]["latents"].shape, (1, 24, 16))
            self.assertEqual(int(engine._fake.calls[-1]["latent_mask"].sum()), frames)

    def test_fused_flow_replaces_the_step_loop(self):
        engine = _fake_engine(apple=True)
        sentence = Sentence(phones=["<sil>", "a", "b", "<end_stmt>", "<sil>"], word_starts=[])
        delivery = delivery_tensors(None)[:2]
        fused = engine.synthesize_sentence(sentence, voice="scylla", language="en_us", delivery=delivery, steps=8, sampler="heun",
                                           rng=np.random.default_rng(0))
        names = [call["name"] for call in engine._fake.calls]
        self.assertEqual(names.count("vector_flow_8"), 1)
        self.assertFalse(any(name.startswith("vector_estimator") for name in names))
        self.assertNotIn("time", engine._fake.calls[names.index("vector_flow_8")])
        self.assertTrue(fused.metadata["fused_flow"])
        engine._fake.calls.clear()
        stepped = engine.synthesize_sentence(sentence, voice="scylla", language="en_us", delivery=delivery, steps=2, sampler="heun",
                                             rng=np.random.default_rng(0))
        names = [call["name"] for call in engine._fake.calls]
        self.assertEqual(names.count("vector_estimator_8"), 4)
        self.assertFalse(stepped.metadata["fused_flow"])

    def test_g2p_runs_the_narrowest_bucket_that_fits(self):
        plain, bucketed = _fake_engine(), _fake_engine(apple=True)
        self.assertEqual(bucketed.phonemize("hello", language="en_us")["phones"], plain.phonemize("hello", language="en_us")["phones"])
        call = next(c for c in bucketed._fake.calls if c["name"].startswith("g2p"))
        self.assertEqual((call["name"], call["text"].shape), ("g2p_16", (1, 16)))
        bucketed.phonemize("hello hello hello hello", language="en_us")
        self.assertEqual(bucketed._fake.calls[-1]["name"], "g2p_32")

    def test_overlong(self):
        engine = _fake_engine(frames_per_phone=9.0)
        sentence = Sentence(phones=["<sil>", "a", "b", "a", "<sil>"], word_starts=[])
        with self.assertRaises(OverlongError):
            engine.synthesize_sentence(sentence, voice="scylla", language="en_us", delivery=delivery_tensors(None)[:2], steps=1)


class StreamingTargetTest(unittest.TestCase):
    """Fake durations: one frame per phone and silence, none for punctuation ("ab." 4 frames, "ab ab." 7); trained
    target range 6 to 12 frames."""

    def targets(self, records, frames_per_phone: float = 1.0):
        engine = _fake_engine(frames_per_phone, chunking=dict(min_target_frames=6, max_target_frames=12),
                              span_conditioning=dict(context_max_phones=64, context_phones_each_side=8))
        plan = plan_records(records, resolve_language=lambda voice, language: "en_us")
        events = [e for e in synthesize_plan_stream(engine, plan, StreamOptions(steps=1, seed=0)) if e.type == "audio_chunk"]
        return [(e.chunk.chunk_id, e.chunk.text, e.metadata["latent_frames"]) for e in events]

    def test_short_sentences_join_the_following_ones_until_long_enough(self):
        self.assertEqual(self.targets(records_from_text("ab. ab. ab ab ab. ab ab.", voice="scylla", language=None)),
                         [("chunk_0001+chunk_0002", "ab. ab.", 7), ("chunk_0003", "ab ab ab.", 10), ("chunk_0004", "ab ab.", 7)])

    def test_a_short_last_sentence_joins_the_one_before(self):
        self.assertEqual(self.targets(records_from_text("ab ab. ab.", voice="scylla", language=None)),
                         [("chunk_0001+chunk_0002", "ab ab. ab.", 10)])

    def test_a_short_first_sentence_takes_the_next_word_rather_than_speak_alone(self):
        # "ab." (4 frames) cannot take all of the next sentence (13); it takes its first word, and the rest is one target.
        self.assertEqual(self.targets(records_from_text("ab. ab ab ab ab.", voice="scylla", language=None)),
                         [("chunk_0001+chunk_0002.1", "ab. ab", 7), ("chunk_0002.2", "ab ab ab.", 10)])

    def test_later_targets_fill_toward_the_trained_maximum(self):
        self.assertEqual(self.targets(records_from_text("ab. ab. ab. ab. ab. ab. ab. ab.", voice="scylla", language=None)),
                         [("chunk_0001+chunk_0002", "ab. ab.", 7), ("chunk_0003+chunk_0005", "ab. ab. ab.", 10),
                          ("chunk_0006+chunk_0008", "ab. ab. ab.", 10)])

    def test_text_without_punctuation_splits_between_words(self):
        self.assertEqual(self.targets(records_from_text("ab ab ab ab ab ab ab ab", voice="scylla", language=None)),
                         [("chunk_0001.1", "ab ab", 7), ("chunk_0001.2", "ab ab ab", 10), ("chunk_0001.3", "ab ab ab", 10)])

    def test_slow_delivery_never_exceeds_the_trained_maximum(self):
        frames = [f for _, _, f in self.targets(records_from_text("ab. ab. ab. ab. ab. ab.", voice="scylla", language=None),
                                                frames_per_phone=2.0)]
        self.assertEqual(frames, [8] * 6)

    def test_short_paragraphs_and_records_join_and_chains_never_do(self):
        paragraphs = records_from_text("ab.\n\nab.", voice="scylla", language=None)
        records = [dict(text="ab.", voice="scylla"), dict(text="ab.", voice="scylla")]
        chains = [dict(text="ab.", voice="scylla"), dict(text="ab.", voice="scylla", delivery="energy=3")]
        self.assertEqual([text for _, text, _ in self.targets(paragraphs)], ["ab. ab."])
        self.assertEqual([text for _, text, _ in self.targets(records)], ["ab. ab."])
        self.assertEqual([text for _, text, _ in self.targets(chains)], ["ab.", "ab."])

    def test_a_long_enough_paragraph_ends_its_target(self):
        self.assertEqual([text for _, text, _ in self.targets(records_from_text("ab ab. ab ab.\n\nab ab.", voice="scylla",
                                                                                 language=None))],
                         ["ab ab.", "ab ab.", "ab ab."])


# --- fakes --------------------------------------------------------------------------------------------------------------
class _FakeGraphs:
    """Graph stand-ins with the bundle's input names. The vocoder returns frames * 512 - 256 samples, like the real one."""

    def __init__(self, frames_per_phone: float):
        self.frames_per_phone, self.calls = frames_per_phone, []

    def factory(self, name, spec):
        graphs = self

        class Session:
            def run(self, feeds):
                graphs.calls.append(dict(feeds, name=name))
                if name.startswith("g2p"):
                    return _fake_g2p_logits(feeds["text"])
                if name.startswith("vector_flow"):
                    return feeds["noise"] * 0.5
                if name == "duration_predictor":
                    return np.full((1, feeds["span_phone_ids"].shape[1]), graphs.frames_per_phone, np.float32)
                if name == "vector_context_encoder":
                    return np.zeros((1, 512), np.float32)
                if name.startswith("vector_estimator"):
                    return -feeds["noise"]
                frames = feeds["latents"].shape[-1]
                return np.ones((1, frames * 512 - 256), np.float32) * 0.1
        return Session()


# G2P fake: characters map to phones of the same letter, spaces to the word-boundary symbol.
_PHONEME_SYMBOLS = {0: "<pad>", 1: "<end>", 2: " ", 3: "a", 4: "b", 5: "h", 6: "l", 7: "o", 8: "w"}
_TEXT_SYMBOLS = {"<pad>": 0, "<end>": 1, "<en_us>": 2, " ": 3, "a": 4, "b": 5, "h": 6, "l": 7, "o": 8, "w": 9, "e": 10}


def _fake_g2p_logits(encoded: np.ndarray) -> np.ndarray:
    inverse = {v: k for k, v in _TEXT_SYMBOLS.items()}
    phone_of = {s: i for i, s in _PHONEME_SYMBOLS.items()}
    frames = []
    for token in encoded[0].tolist():
        char = inverse.get(int(token), "<pad>")
        if char in ("<pad>", "<en_us>"):
            continue
        if char == "<end>":
            frames.append(1)
            break
        frames.append(phone_of.get(char, phone_of.get("a")) if char != "e" else phone_of["o"])
        frames.append(0)   # CTC blank between characters so repeats survive
    logits = np.full((1, len(frames), len(_PHONEME_SYMBOLS)), -10.0, np.float32)
    for index, symbol in enumerate(frames):
        logits[0, index, symbol] = 10.0
    return logits


def _fake_bundle(root: Path, *, apple: bool = False, controls: dict | None = None) -> Path:
    """``apple``: also the optional fused-flow graphs and narrower G2P inputs of the Core ML and Core AI bundles.
    ``controls``: manifest controls to replace."""
    bundle = root / "bundle"
    (bundle / "assets/g2p").mkdir(parents=True)
    (bundle / "onnx").mkdir()
    write = lambda rel, value: (bundle / rel).write_text(json.dumps(value))  # noqa: E731
    write("assets/phone_vocab.json", dict(token_to_id=PHONE_TO_ID))
    write("assets/voices.json", [dict(id="scylla", index=0)])
    write("assets/languages.json", [dict(id="en_us", index=0)])
    write("assets/g2p/export_config.json", dict(input_granularity="phrase", chunk_max_chars=100))
    write("assets/g2p/tokenizer.json", dict(text_symbols=_TEXT_SYMBOLS, phoneme_symbols={str(k): v for k, v in _PHONEME_SYMBOLS.items()},
                                            char_repeats=1, lowercase=True, text_pad_index=0, phoneme_pad_index=0, phoneme_end_index=1))
    write("assets/g2p/language_map.json", {"en_us": "en_us"})
    components = {}
    for name, inputs in (("g2p", ["text"]), ("duration_predictor", list(DURATION_INPUTS)), ("vector_context_encoder", list(CONTEXT_INPUTS)),
                         ("vector_estimator_8", list(VECTOR_INPUTS)), ("vocoder_8", list(VOCODER_INPUTS)),
                         ("vector_estimator", list(VECTOR_INPUTS)), ("vocoder", list(VOCODER_INPUTS))):
        (bundle / f"onnx/{name}.onnx").write_bytes(b"")
        components[name] = dict(artifacts=dict(onnx=dict(path=f"onnx/{name}.onnx", format="onnx")), inputs=inputs, outputs=["out"])
    manifest = dict(contract_version="1.0.0", audio=dict(sample_rate=24000, latent_dim=24, latent_hop_length=512, hop_length=256),
                    assets=dict(phone_vocab="assets/phone_vocab.json", voice_index="assets/voices.json", language_index="assets/languages.json",
                                g2p_config="assets/g2p/export_config.json", g2p_tokenizer="assets/g2p/tokenizer.json",
                                g2p_language_map="assets/g2p/language_map.json"),
                    components=components,
                    voices=[dict(id="scylla", languages=["en_us"], default_language="en_us")],
                    controls=dict(graph_input_contract=GRAPH_CONTRACT, word_boundaries=dict(enabled=True, g2p_output_symbol=" "),
                                  fixed_shapes=dict(g2p_text_tokens=64), span_conditioning=dict(context_max_phones=16, context_phones_each_side=4),
                                  prefix_conditioning=dict(max_frames=6), decoding=dict(context_frames=4),
                                  target_buckets=dict(buckets=[dict(latent_frames=8, vector_estimator="vector_estimator_8", vocoder="vocoder_8"),
                                                               dict(latent_frames=16, vector_estimator="vector_estimator", vocoder="vocoder")])))
    if apple:
        flow_inputs = [n for n in VECTOR_INPUTS if n != "time"]
        for name, inputs in (("vector_flow_8", flow_inputs), ("vector_flow", flow_inputs), ("g2p_16", ["text"]), ("g2p_32", ["text"])):
            (bundle / f"onnx/{name}.onnx").write_bytes(b"")
            components[name] = dict(artifacts=dict(onnx=dict(path=f"onnx/{name}.onnx", format="onnx")), inputs=inputs, outputs=["out"])
        controls = manifest["controls"]
        controls["fused_flow"] = dict(sampler="heun", steps=8)
        controls["g2p_buckets"] = [dict(text_tokens=16, component="g2p_16"), dict(text_tokens=32, component="g2p_32")]
        for entry, flow in zip(controls["target_buckets"]["buckets"], ("vector_flow_8", "vector_flow")):
            entry["vector_flow"] = flow
    manifest["controls"].update(controls or {})
    write("manifest.json", manifest)
    return bundle


def _fake_engine(frames_per_phone: float = 1.0, *, apple: bool = False, **controls) -> Engine:
    tmp = tempfile.TemporaryDirectory()
    bundle = _fake_bundle(Path(tmp.name), apple=apple, controls=controls)
    graphs = _FakeGraphs(frames_per_phone)
    engine = Engine(bundle, session_factory=graphs.factory)
    engine._fake, engine._tmp = graphs, tmp
    return engine


if __name__ == "__main__":
    unittest.main()
