"""v5 bundle tests on a fake bundle: the contract, style rows for speech and song, non-verbal text, and the render rules."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from scyllasband.contract import (BundleError, DURATION_INPUTS, DURATION_INPUTS_V5, GRAPH_CONTRACTS_V5, PITCH_INPUTS, PITCH_INPUTS_V5,
                                  VECTOR_INPUTS, VECTOR_INPUTS_V5, VOCODER_INPUTS, validate_bundle_layout)
from scyllasband.engine import Engine, MIN_TARGET_PHONES_V5, Sentence
from scyllasband.events import mark_sounded_pauses, opening_minimums
from scyllasband.lexicon import Lexicon
from scyllasband.planner import plan_records, records_from_text
from scyllasband.runtime import ScyllasBandRuntime, SynthesisRequest
from scyllasband.singing import (HOP, SAMPLE_RATE, ScoreError, Word, line_phones, parse_score, phone_durations, plan_song, score_durations,
                                 sing, singing_styles, stressed_pronoun, style_modes, word_layout, word_phones)
from scyllasband.streaming import StreamOptions, synthesize_plan_stream
from scyllasband.text_data import TEXT_DATA
from scyllasband.text_normalizer import nonverbal_spellings
from test_runtime import PHONE_TO_ID, V4_DECODERS, VOCAB, _FakeGraphs, _fake_bundle, _fake_engine

# The style rows of a v5 manifest (row 0 spoken, 9 rap, the sung styles around them), as the manifest gives them.
STYLE_IDS = dict(spoken=0, ballad=1, pop=2, folk=3, jazz=4, soul=5, country=6, punk=7, indie=8, rap=9, rock=10)
SUNG = [name for name in STYLE_IDS if name not in ("spoken", "rap")]
V5_DELIVERY_MODES = dict(enabled=True, input="delivery_mode_id", duration_input="style_id", ids=STYLE_IDS, default="spoken",
                         aliases=dict(lullaby="ballad"), duration_styles_trained=["spoken", "rap"],
                         pitch_styles_trained=["spoken", "rap"], controls_absent=dict(styles=["rap"]),
                         singing=dict(modes=SUNG, decoder="bigvgan", guidance=3.0, sampler="heun", steps=8,
                                      melody=dict(layout="note_phase_presence", midi_center=60, midi_scale=24)))
# The trainer's phone table for the v5 non-verbal sounds (nonverbal_en_v2): what text_data/en.json must reproduce.
NONVERBAL_TABLE = {
    "hmm": "h ə m", "mhm": "m ̩ ː h ˈ m ̩ ː", "uhhuh": "ˈ ʌ ̃ h ʌ ̃", "shh": "ʃ ː ʃ ː", "aha": "ɑ ː h ˈ ɑ ː", "heh": "h ˈ ɛ h",
    "uh": "ˈ ʌ", "um": "ˈ ʌ m", "mm": "m ̩ ː", "hehe": "h ɛ h ˈ ɛ", "haha": "h ɑ | h ˈ ɑ", "ahaha": "ɑ h ɑ h ˈ ɑ",
    "phew": "f j ˈ u ː", "whew": "h j ˈ u ː", "ugh": "ˈ ʌ ɡ", "ahh": "ˈ ɑ ː", "ohh": "ˈ o ʊ", "pfft": "p f ː t",
    "hmph": "h m ̩ f", "grr": "ɡ ɹ ̩ ː", "brr": "b ɹ ̩ ː", "psst": "p s ː t", "tsk": "t s k", "ow": "ˈ a ʊ", "ew": "ˈ i ː w",
    "aw": "ˈ ɔ ː", "uhuh": "ˈ ʌ ̃ ʔ | ʌ ̃", "nuhuh": "n ˈ ʌ ̃ ʔ | ʌ ̃", "ahem": "ə h ˈ ɛ m", "hahaha": "h ɑ | h ɑ | h ˈ ɑ",
    "eww": "ˈ i ː w", "aww": "ˈ ɔ ː", "uhm": "ˈ ʌ m"}
NONVERBAL_PHRASES = {"uh uh": "uhuh", "nuh uh": "nuhuh", "ha ha": "haha", "ha ha ha": "hahaha"}
V5_VOCAB = VOCAB + ["<nonverbal>"] + sorted({p for phones in NONVERBAL_TABLE.values() for p in phones.split() if p != "|"} - set(VOCAB))
V5_PHONE_TO_ID = {p: i for i, p in enumerate(V5_VOCAB)}
IDENTITY = [0.25, -0.5, 0.75, 1.0]


def _v5_bundle(root: Path, *, contract: str = GRAPH_CONTRACTS_V5[1], mutate=None, **controls) -> Path:
    """The fake bundle under the v5 contract: identity and style inputs, the style table, the pause mark, an identity vector
    per voice and both decoders. ``mutate(manifest)`` edits the manifest before it is written."""
    bundle = _fake_bundle(root, v4=True)
    manifest = json.loads((bundle / "manifest.json").read_text())
    for name, spec in manifest["components"].items():
        if name == "duration_predictor":
            spec["inputs"] = list(DURATION_INPUTS_V5)
        elif name.startswith("vector_estimator"):
            spec["inputs"] = list(VECTOR_INPUTS_V5)
        elif name.startswith("pitch_estimator"):
            spec["inputs"] = list(PITCH_INPUTS_V5)
    (bundle / "assets/phone_vocab.json").write_text(json.dumps(dict(token_to_id=V5_PHONE_TO_ID)))
    (bundle / "assets/voice_identity.json").write_text(json.dumps(dict(dim=len(IDENTITY), voices=dict(scylla=IDENTITY))))
    manifest["assets"]["voice_identity"] = "assets/voice_identity.json"
    manifest["controls"].update(graph_input_contract=contract, delivery_modes=json.loads(json.dumps(V5_DELIVERY_MODES)),
                                decoders=json.loads(json.dumps(V4_DECODERS)), pause_marks=dict(token="<nonverbal>", default="all"))
    manifest["controls"]["pitch"]["prefix"] = dict(enabled=True, max_frames=6)
    manifest["controls"].update(json.loads(json.dumps(controls)))
    if mutate is not None:
        mutate(manifest)
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    return bundle


def _v5_engine(frames_per_phone: float = 1.0, *, mutate=None, **controls) -> Engine:
    tmp = tempfile.TemporaryDirectory()
    graphs = _FakeGraphs(frames_per_phone)
    engine = Engine(_v5_bundle(Path(tmp.name), mutate=mutate, **controls), session_factory=graphs.factory)
    engine._fake, engine._tmp = graphs, tmp
    return engine


# Room for a few short sentences in one passage of the fake bundle: targets of 3 to 12 frames, no minimum in phones.
PASSAGE = dict(chunking=dict(min_target_frames=3, max_target_frames=12, min_target_phones=0),
               span_conditioning=dict(context_max_phones=64, context_phones_each_side=8))


def _calls(engine: Engine, prefix: str) -> list[dict]:
    return [c for c in engine._fake.calls if c["name"].startswith(prefix)]


def _stream(engine: Engine, records, **options):
    plan = plan_records(records, resolve_language=lambda voice, language: "en_us", nonverbal_spellings=engine.v5)
    return [e for e in synthesize_plan_stream(engine, plan, StreamOptions(steps=1, seed=0, **options)) if e.type == "audio_chunk"]


class ContractV5Test(unittest.TestCase):
    def _check(self, mutate=None, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            return validate_bundle_layout(_v5_bundle(Path(tmp), mutate=mutate, **kwargs))

    def test_both_contract_names_validate(self):
        for contract in GRAPH_CONTRACTS_V5:
            summary = self._check(contract=contract)
            self.assertEqual((summary["graph_contract"], summary["singing"], summary["buckets"]), (contract, True, [8, 16]))

    def test_input_lists(self):
        self.assertEqual(DURATION_INPUTS_V5, (*DURATION_INPUTS, "identity_reference", "style_id"))
        self.assertEqual(PITCH_INPUTS_V5, (*PITCH_INPUTS, "prefix_latents", "prefix_mask", "identity_reference", "delivery_mode_id"))
        self.assertEqual(VECTOR_INPUTS_V5, (*VECTOR_INPUTS, "identity_reference", "delivery_mode_id"))

    def test_graphs_without_the_style_or_identity_inputs_are_rejected(self):
        def without(component, name):
            def mutate(manifest):
                for key, spec in manifest["components"].items():
                    if key.startswith(component):
                        spec["inputs"] = [i for i in spec["inputs"] if i != name]
            return mutate
        for component, name in (("duration_predictor", "style_id"), ("pitch_estimator", "delivery_mode_id"),
                                ("vector_estimator", "delivery_mode_id"), ("vector_estimator", "identity_reference"),
                                ("pitch_estimator", "prefix_mask")):
            with self.assertRaisesRegex(BundleError, component):
                self._check(without(component, name))

    def test_identity_vectors_style_table_and_pause_mark_are_required(self):
        def drop_asset(manifest):
            del manifest["assets"]["voice_identity"]
        with self.assertRaisesRegex(BundleError, "voice_identity"):
            self._check(drop_asset)

        def other_voice(manifest):
            manifest["voices"].append(dict(id="ink", languages=["en_us"], default_language="en_us"))
        with self.assertRaisesRegex(BundleError, "vector for voice 'ink'"):
            self._check(other_voice)
        with self.assertRaisesRegex(BundleError, "delivery_modes"):
            self._check(delivery_modes=dict(enabled=False))
        for trained in (None, [], ["rap"], ["spoken", "grunge"]):
            with self.assertRaisesRegex(BundleError, "duration_styles_trained"):
                self._check(delivery_modes=dict(V5_DELIVERY_MODES, duration_styles_trained=trained))
        with self.assertRaisesRegex(BundleError, "pause_marks"):
            self._check(pause_marks=None)
        with self.assertRaisesRegex(BundleError, "not in the phone vocabulary"):
            self._check(pause_marks=dict(token="<hum>", default="all"))
        self.assertTrue(self._check(pause_marks=dict(default="none"))["singing"])

    def test_a_bucket_may_name_any_declared_decoder(self):
        def rename(keep_components):
            def mutate(manifest):
                manifest["controls"]["decoders"]["wave"] = dict(manifest["controls"]["decoders"].pop("bigvgan"), component="vocoder_wave")
                for bucket in manifest["controls"]["target_buckets"]["buckets"]:
                    old = bucket.pop("vocoder_bigvgan")
                    bucket["vocoder_wave"] = old.replace("bigvgan", "wave")
                    spec = manifest["components"].pop(old)
                    if keep_components:
                        manifest["components"][bucket["vocoder_wave"]] = spec
            return mutate
        self.assertTrue(self._check(rename(True))["singing"])
        with self.assertRaisesRegex(BundleError, "vocoder_wave"):
            self._check(rename(False))

    def test_the_engine_refuses_a_bundle_without_what_v5_needs(self):
        for controls, message in ((dict(delivery_modes=dict(enabled=False)), "style rows"),
                                  (dict(delivery_modes=dict(V5_DELIVERY_MODES, duration_styles_trained=None)), "duration_styles_trained"),
                                  (dict(delivery_modes=dict(V5_DELIVERY_MODES, duration_styles_trained=["spoken", "trap"])), "no row for trap"),
                                  (dict(pause_marks=None), "pause_marks"), (dict(pause_marks=dict(token="<hum>", default="all")), "<hum>")):
            with self.assertRaisesRegex(ValueError, message):
                _v5_engine(**controls)


class SpeechStyleTest(unittest.TestCase):
    def _speak(self, engine, **kwargs):
        sentence = Sentence(phones=["<sil>", "a", "b", "h", "l", "<sil>"], word_starts=[], before_ids=[], after_ids=[])
        delivery = (*engine.delivery_inputs("scylla", dict(intensity=0.5))[:2], 1.0)
        return engine.synthesize_sentence(sentence, voice="scylla", language="en_us", delivery=delivery, steps=2, sampler="heun",
                                          rng=np.random.default_rng(7), **kwargs)

    def test_spoken_feeds_the_spoken_row_and_the_identity_to_all_three_models(self):
        engine = _v5_engine()
        result = self._speak(engine)
        duration, pitch, flow = (_calls(engine, name) for name in ("duration_predictor", "pitch_estimator", "vector_estimator"))
        self.assertEqual(set(duration[0]) - {"name"}, set(DURATION_INPUTS_V5))
        self.assertEqual((duration[0]["style_id"].tolist(), duration[0]["style_id"].dtype), ([0], np.int64))
        self.assertEqual((len(pitch), len(flow)), (2, 4))   # Euler x 2 for the contour, Heun x 2 for the latents
        for call in pitch:
            self.assertEqual(set(call) - {"name"}, set(PITCH_INPUTS_V5))
            self.assertEqual(call["prefix_latents"].shape, (1, 2, 6))
        for call in flow:
            self.assertEqual(set(call) - {"name"}, set(VECTOR_INPUTS_V5))
        for call in duration + pitch + flow:
            np.testing.assert_array_equal(call["identity_reference"], np.asarray([IDENTITY], np.float32))
            self.assertEqual(call.get("delivery_mode_id", call.get("style_id")).tolist(), [0])
            self.assertTrue(call["delivery_present"][0, :2].all())
        self.assertEqual(result.metadata["delivery_mode"], 0)

    def test_the_rows_come_from_the_bundle(self):
        moved = dict(V5_DELIVERY_MODES, ids=dict(STYLE_IDS, spoken=4, jazz=0, rap=6))
        engine = _v5_engine(delivery_modes=moved)
        self.assertEqual((engine.speech_style("spoken"), engine.speech_style(None), engine.speech_style("rap")), ((4, True), (4, True), (6, False)))
        self._speak(engine, delivery_mode="rap")
        for name in ("duration_predictor", "pitch_estimator", "vector_estimator"):
            for call in _calls(engine, name):
                self.assertEqual(call.get("delivery_mode_id", call.get("style_id")).tolist(), [6])
        engine._fake.calls.clear()
        self._speak(engine)
        self.assertEqual(_calls(engine, "duration_predictor")[0]["style_id"].tolist(), [4])

    def test_rap_sends_its_row_with_intensity_and_pace_absent(self):
        engine = _v5_engine(delivery=dict(z_scale=5.0, z_limit=2.5, min_spoken_phones=1,
                                          defaults=dict(scylla=dict(intensity=0.5, pace=-0.5, pitch=0.0))), **PASSAGE)
        events = _stream(engine, records_from_text("ab ab ab. ab ab.", voice="scylla", language=None, pitch=1.0), style="rap")
        calls = [c for name in ("duration_predictor", "pitch_estimator", "vector_estimator") for c in _calls(engine, name)]
        self.assertTrue(calls and all(c.get("delivery_mode_id", c.get("style_id")).tolist() == [9] for c in calls))
        self.assertFalse(any(c["delivery_present"].any() for c in calls))
        self.assertEqual({e.metadata["delivery_mode"] for e in events}, {9})
        self.assertEqual({e.metadata["pitch_shift"] for e in events}, {1.0})   # the pitch shift is not a model input: it stays
        self.assertEqual(events[0].metadata["delivery"], dict(intensity=None, pace=None, pitch=1.0))
        # The same text spoken: the voice's defaults are set and the row is the spoken one.
        engine._fake.calls.clear()
        _stream(engine, records_from_text("ab ab ab. ab ab.", voice="scylla", language=None))
        flows = _calls(engine, "vector_estimator")
        self.assertTrue(all(c["delivery_mode_id"].tolist() == [0] and c["delivery_present"][0, :2].all() for c in flows))

    def test_rap_refuses_intensity_and_pace(self):
        engine = _v5_engine(**PASSAGE)
        for field in ("intensity", "pace"):
            with self.assertRaisesRegex(ValueError, f"trained without intensity and pace; leave {field} unset"):
                _stream(engine, records_from_text("ab ab.", voice="scylla", language=None, **{field: 1.0}), style="rap")
        self.assertEqual(engine._fake.calls, [])

    def test_only_trained_styles_can_be_spoken(self):
        engine = _v5_engine()
        self.assertEqual(engine.speech_styles, ("spoken", "rap"))
        with self.assertRaisesRegex(ValueError, "'pop' cannot be spoken \\(it is a sung style.*trained on spoken, rap"):
            engine.speech_style("pop")
        with self.assertRaisesRegex(ValueError, "'grunge' cannot be spoken: .*trained on spoken, rap"):
            engine.speech_style("grunge")
        # A style the pitch model did not train on is not a speech style either.
        narrow = _v5_engine(delivery_modes=dict(V5_DELIVERY_MODES, pitch_styles_trained=["spoken"]))
        self.assertEqual(narrow.speech_styles, ("spoken",))
        with self.assertRaisesRegex(ValueError, "cannot be spoken"):
            narrow.speech_style("rap")

    def test_earlier_bundles_speak_as_before_and_refuse_a_style(self):
        for v4 in (False, True):
            engine = _fake_engine(v4=v4, **PASSAGE)
            self.assertEqual((engine.speech_styles, engine.speech_style("spoken")), (("spoken",), (0, True)))
            self.assertEqual(_fake_engine(v4=v4).min_target_phones, 0)
            with self.assertRaisesRegex(ValueError, "speaks only in the spoken style"):
                engine.speech_style("rap")
            plain = _stream(engine, records_from_text("ab ab ab. ab.", voice="scylla", language=None))
            named = _fake_engine(v4=v4, **PASSAGE)
            styled = _stream(named, records_from_text("ab ab ab. ab.", voice="scylla", language=None), style="spoken")
            self.assertEqual(len(engine._fake.calls), len(named._fake.calls))
            for a, b in zip(engine._fake.calls, named._fake.calls):
                self.assertEqual(set(a), set(b))
                self.assertFalse({"style_id", "identity_reference"} & set(a))
            for a, b in zip(plain, styled):
                np.testing.assert_array_equal(a.audio, b.audio)

    def test_requests_and_the_cli_carry_the_style(self):
        from scyllasband.cli import _options, build_parser
        with tempfile.TemporaryDirectory() as tmp:
            graphs = _FakeGraphs(1.0)
            bundle = _v5_bundle(Path(tmp), **PASSAGE)
            runtime = ScyllasBandRuntime(Engine(bundle, session_factory=graphs.factory), bundle_dir=bundle, backend="onnx")
            self.assertEqual(runtime.speech_styles(), ["spoken", "rap"])
            result = runtime.synthesize(SynthesisRequest(text="ab ab ab.", voice_id="scylla", style="rap", pitch=0.5, steps=1, seed=0))
            self.assertEqual((result.metadata["style"], result.metadata["delivery"]), ("rap", dict(intensity=None, pace=None, pitch=0.5)))
            self.assertEqual({c["style_id"].tolist()[0] for c in graphs.calls if c["name"] == "duration_predictor"}, {9})
            graphs.calls.clear()
            for request in (SynthesisRequest(text="ab.", voice_id="scylla", style="pop"),
                            SynthesisRequest(text="ab.", voice_id="scylla", style="rap", intensity=1.0)):
                with self.assertRaises(ValueError):
                    runtime.synthesize(request)
            with self.assertRaisesRegex(ValueError, "cannot be spoken"):
                runtime.synthesize_stream(records_from_text("ab.", voice="scylla", language=None), style="rock")
            with self.assertRaisesRegex(ValueError, "leave pace unset"):
                runtime.render_records(records_from_text("ab.", voice="scylla", language=None, pace=1.0), style="rap")
            self.assertEqual(graphs.calls, [])
            self.assertEqual(runtime.runtime_status()["speech_styles"], ["spoken", "rap"])
        args = build_parser().parse_args(["speak", "ab.", "--style", "rap"])
        self.assertEqual(_options(args)["style"], "rap")
        self.assertIsNone(_options(build_parser().parse_args(["speak", "ab."]))["style"])

    def test_the_native_engine_takes_a_style_from_abi_3_2(self):
        from types import SimpleNamespace
        from scyllasband.native import NativeRuntime
        old = SimpleNamespace(library=SimpleNamespace(has_styles=False, path="libscyllasband_native.so", version="3.1.0"))
        self.assertIsNone(NativeRuntime._style(old, None))
        self.assertIsNone(NativeRuntime._style(old, " Spoken "))
        with self.assertRaisesRegex(ValueError, "ABI 3.2 or later"):
            NativeRuntime._style(old, "rap")
        new = SimpleNamespace(library=SimpleNamespace(has_styles=True))
        self.assertEqual(NativeRuntime._style(new, " Rap "), b"rap")
        self.assertEqual(NativeRuntime.speech_styles(SimpleNamespace(info={})), ["spoken"])
        self.assertEqual(NativeRuntime.speech_styles(SimpleNamespace(info=dict(speech_styles=["spoken", "rap"]))), ["spoken", "rap"])


class OpeningMinimumTest(unittest.TestCase):
    UH_UH = ["ˈ", "ʌ", "̃", "ʔ", "<sil>", "ʌ", "̃"]

    def test_an_opening_uh_uh_keeps_its_minimum_lengths(self):
        phones = ["<sil>", *self.UH_UH, "<nonverbal>", "<pause_comma>", "<sil>", "a", "<end_stmt>", "<sil>"]
        durations = [3, 0, 2, 0, 4, 0, 5, 0, 0, 0, 9, 7, 0, 6]
        self.assertEqual(opening_minimums(phones, durations), [3, 0, 6, 0, 4, 2, 12, 0, 0, 0, 9, 7, 0, 6])
        longer = [3, 0, 8, 0, 4, 3, 14, 0, 0, 0, 9, 7, 0, 6]
        self.assertEqual(opening_minimums(phones, longer), longer)
        nuh = ["<sil>", "n", *self.UH_UH, "<end_exclaim>", "<sil>"]
        self.assertEqual(opening_minimums(nuh, [2, 3, 0, 1, 0, 2, 0, 3, 0, 0, 4]), [2, 3, 0, 6, 0, 2, 2, 12, 0, 0, 4])

    def test_inside_a_sentence_the_lengths_stay(self):
        phones = ["<sil>", "a", "<sil>", *self.UH_UH, "<end_stmt>", "<sil>"]
        durations = [2, 3, 1, 0, 2, 0, 3, 0, 4, 0, 0, 5]
        self.assertEqual(opening_minimums(phones, durations), durations)
        self.assertEqual(opening_minimums(phones, durations, [0, 2]), [2, 3, 1, 0, 6, 0, 3, 2, 12, 0, 0, 5])   # a sentence starts at 2
        self.assertEqual(opening_minimums(["<sil>", *self.UH_UH, "<sil>"], [1] * 9, []), [1] * 9)                # a target cut inside one

    def test_the_duration_step_applies_it_at_the_sentence_starts_of_a_target(self):
        from scyllasband.streaming import _ChainStream
        engine = _v5_engine(2.0, **PASSAGE)
        plan = plan_records(records_from_text("uh uh, ab. uh uh ab uh uh. nuh uh!", voice="scylla", language=None),
                            resolve_language=lambda voice, language: "en_us")
        stream = _ChainStream(engine, plan.chunks)
        stream.extend(1000)
        delivery = engine.delivery_inputs("scylla", None)[:2]

        def lengths(start):
            sentence = stream.sentence(start, len(stream.phones) - 1)
            frames = engine.durations(sentence, voice=0, language=0, delivery=delivery)
            found = [i for i in range(len(sentence.phones) - 5) if sentence.phones[i:i + 6] == ["ʌ", "̃", "ʔ", "<sil>", "ʌ", "̃"]]
            return sentence.sentence_starts, [(frames[i], frames[i + 3], frames[i + 4]) for i in found]

        starts, found = lengths(0)
        self.assertEqual([stream.phones[i] for i in starts], ["<sil>"] * 3)
        self.assertEqual(found, [(6, 2, 12), (6, 2, 12), (2, 2, 2), (6, 2, 12)])   # the third is inside its sentence
        cut = stream.phones.index("ʌ") + 3                                        # a target cut between the two parts of the first
        self.assertEqual(stream.phones[cut], "<sil>")
        self.assertEqual(lengths(cut)[1], [(6, 2, 12), (2, 2, 2), (6, 2, 12)])
        self.assertNotIn(0, lengths(cut)[0])


class TargetPhonesTest(unittest.TestCase):
    """Fake durations: one frame per phone and silence ("ab." 4 frames, 2 phones). Trained range 3 to 12 frames, so a
    two-phone sentence is long enough in frames to be spoken alone; the v5 rule wants three phones here."""

    def targets(self, records, engine=None, **chunking):
        engine = engine or _v5_engine(**dict(PASSAGE, chunking={**PASSAGE["chunking"], "min_target_phones": 3, **chunking}))
        return [e.chunk.text for e in _stream(engine, records)]

    def test_the_default_is_twelve_phones_on_v5(self):
        self.assertEqual((MIN_TARGET_PHONES_V5, _v5_engine().min_target_phones), (12, 12))
        self.assertEqual(_v5_engine(chunking=dict(min_target_phones=5)).min_target_phones, 5)

    def test_a_short_sentence_joins_the_next_one(self):
        text = records_from_text("ab. ab ab.", voice="scylla", language=None)
        self.assertEqual(self.targets(text), ["ab. ab ab."])
        self.assertEqual(self.targets(text, min_target_phones=0), ["ab.", "ab ab."])   # frames alone do not join them

    def test_a_short_last_sentence_joins_the_one_before(self):
        text = records_from_text("ab ab. ab.", voice="scylla", language=None)
        self.assertEqual(self.targets(text), ["ab ab. ab."])
        self.assertEqual(self.targets(text, min_target_phones=0), ["ab ab.", "ab."])

    def test_a_short_sentence_takes_part_of_a_neighbour_that_does_not_fit(self):
        self.assertEqual(self.targets(records_from_text("ab. ab ab ab ab.", voice="scylla", language=None)), ["ab. ab", "ab ab ab."])
        self.assertEqual(self.targets(records_from_text("ab ab ab ab. ab.", voice="scylla", language=None)), ["ab ab", "ab ab. ab."])

    def test_a_lone_short_sentence_and_a_new_chain_are_spoken_as_they_are(self):
        self.assertEqual(self.targets(records_from_text("ab.", voice="scylla", language=None)), ["ab."])
        chains = [dict(text="ab.", voice="scylla"), dict(text="ab.", voice="scylla", delivery="intensity=1")]
        self.assertEqual(self.targets(chains), ["ab.", "ab."])
        paragraphs = records_from_text("ab.\n\nab ab.", voice="scylla", language=None)
        self.assertEqual(self.targets(paragraphs), ["ab. ab ab."])

    def test_marks_and_stress_are_not_counted(self):
        from scyllasband.streaming import _ChainStream
        engine = _v5_engine()
        plan = plan_records(records_from_text("aha, ab. ab.", voice="scylla", language=None), resolve_language=lambda v, l: "en_us")
        stream = _ChainStream(engine, plan.chunks)
        stream.extend(100)
        self.assertEqual(stream.phones, ["<sil>", "ɑ", "ː", "h", "ˈ", "ɑ", "ː", "<nonverbal>", "<pause_comma>", "<sil>", "a", "b",
                                         "<nonverbal>", "<end_stmt>", "<sil>", "a", "b", "<end_stmt>", "<sil>"])
        self.assertEqual((stream.spoken_phones(0), stream.spoken_phones(0, 9), stream.spoken_phones(14)), (7, 3, 2))
        self.assertTrue(stream.short_rest(14, 3))
        self.assertFalse(stream.short_rest(9, 3))
        self.assertFalse(stream.short_rest(18, 3))   # nothing left: the chain ends there


class PauseMarkTest(unittest.TestCase):
    def test_every_pause_after_punctuation_inside_the_text_is_marked(self):
        phones = ["<sil>", "a", "<pause_comma>", "<sil>", "b", "<sil>", "a", "<end_stmt>", "<sil>", "b", "<end_question>", "<sil>"]
        marked, moved = mark_sounded_pauses(phones, last=False)
        self.assertEqual(marked, ["<sil>", "a", "<nonverbal>", "<pause_comma>", "<sil>", "b", "<sil>", "a", "<nonverbal>", "<end_stmt>",
                                  "<sil>", "b", "<end_question>", "<sil>"])
        self.assertEqual(moved, [0, 1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13])
        self.assertEqual(mark_sounded_pauses(phones, last=True)[0][-3:], ["<nonverbal>", "<end_question>", "<sil>"])

    def test_the_stream_marks_all_but_the_last_silence_and_none_when_the_bundle_says_so(self):
        text = records_from_text("ab, ab. ab ab? ab.", voice="scylla", language=None)
        phones = [p for e in _stream(_v5_engine(**PASSAGE), text) for p in e.metadata["phones"]]
        self.assertEqual([phones[i + 1] for i, p in enumerate(phones) if p == "<nonverbal>"], ["<pause_comma>", "<end_stmt>", "<end_question>"])
        self.assertEqual(phones[-3:], ["b", "<end_stmt>", "<sil>"])
        quiet = _v5_engine(pause_marks=dict(default="none"), **PASSAGE)
        self.assertNotIn("<nonverbal>", [p for e in _stream(quiet, text) for p in e.metadata["phones"]])
        self.assertNotIn("<nonverbal>", [p for e in _stream(_fake_engine(v4=True, **PASSAGE), text) for p in e.metadata["phones"]])


class NonverbalTextTest(unittest.TestCase):
    def test_typed_variants_take_the_trained_spelling(self):
        for typed, written in (("Uhuh, no.", "Uh-uh, no."), ("uhuh", "uh-uh"), ("UHUH!", "UH-UH!"), ("Nuhuh! You did.", "Nuh-uh! You did."),
                               ("Hehheh, sure.", "Heh heh, sure."), ("A-ha-ha, very good.", "Ahaha, very good."),
                               ("ah-ha-ha", "ahaha"), ("Ah ha ha!", "Ahaha!"), ("A‑hem. Excuse me.", "Ahem. Excuse me."),
                               ("Pst, over here.", "Psst, over here."), ("pst", "psst"), ("Sh, quiet.", "Shh, quiet."),
                               ("Be quiet, sh.", "Be quiet, shh.")):
            self.assertEqual(nonverbal_spellings(typed, language="en_us"), written)
        for kept in ("It is 9 PST now.", "SH!", "a ha ha", "Sh is a shell.", "The uhuh-sayer.", "pstree", "Uh-uh, no.", "Ahem."):
            self.assertEqual(nonverbal_spellings(kept, language="en_gb"), kept)
        self.assertEqual(nonverbal_spellings("Uhuh, pst.", language="de"), "Uhuh, pst.")   # English only

    def test_the_plan_respells_only_for_v5_and_keeps_the_source_text(self):
        records = records_from_text("Uhuh, no. Pst!", voice="scylla", language=None)
        resolve = lambda voice, language: "en_us"   # noqa: E731
        v5 = plan_records(records, resolve_language=resolve, nonverbal_spellings=True)
        self.assertEqual([c.text for c in v5.chunks], ["Uh uh, no.", "Psst!"])
        self.assertEqual(v5.chunks[0].source_text, "Uhuh, no. Pst!")
        self.assertEqual([c.text for c in plan_records(records, resolve_language=resolve).chunks], ["Uhuh, no.", "Pst!"])
        raw = plan_records(records, resolve_language=resolve, normalize=False, nonverbal_spellings=True)
        self.assertEqual([c.text for c in raw.chunks], ["Uhuh, no.", "Pst!"])

    def test_the_text_data_holds_the_trained_table(self):
        section = TEXT_DATA["en"]["nonverbal_phones"]
        self.assertEqual((len(NONVERBAL_TABLE), len(NONVERBAL_PHRASES)), (33, 4))
        self.assertEqual(section["words"], NONVERBAL_TABLE)
        self.assertEqual(section["phrases"], NONVERBAL_PHRASES)
        self.assertEqual(section["locales"], ["en_us", "en_gb"])

    def test_listed_words_and_phrases_give_the_table_phones(self):
        lexicon = Lexicon(V5_PHONE_TO_ID, nonverbal=True)
        for language in ("en_us", "en_gb"):
            for word, phones in NONVERBAL_TABLE.items():
                joined = tuple(p for p in phones.split() if p != "|")
                for token in (word, word.capitalize(), word.upper(), f'"{word}'):
                    self.assertEqual(lexicon.nonverbal_hits(["so", token, "then"], language), {1: joined}, token)
            for phrase, word in NONVERBAL_PHRASES.items():
                parts = [tuple(part.split()) for part in NONVERBAL_TABLE[word].split("|")]
                tokens = phrase.capitalize().split()
                self.assertEqual(lexicon.nonverbal_hits(tokens, language), dict(enumerate(parts)), phrase)
        hits = lexicon.nonverbal_hits("ha ha ha ha ha oh uh uh uh".split(), "en_us")
        self.assertEqual([" ".join(hits[i]) for i in sorted(hits)], ["h ɑ", "h ɑ", "h ˈ ɑ", "h ɑ", "h ˈ ɑ", "ˈ ʌ ̃ ʔ", "ʌ ̃", "ˈ ʌ"])
        self.assertEqual(lexicon.nonverbal_hits(['uh"', "uh"], "en_us"), {0: ("ˈ", "ʌ"), 1: ("ˈ", "ʌ")})   # a mark between: two words
        self.assertEqual(lexicon.nonverbal_hits(["hmm"], "es"), {})
        self.assertEqual(Lexicon(V5_PHONE_TO_ID).nonverbal_hits(["hmm"], "en_us"), {})
        with self.assertRaisesRegex(ValueError, "outside this bundle's vocabulary"):
            Lexicon(PHONE_TO_ID, nonverbal=True)

    def test_the_g2p_uses_the_table_on_v5_only(self):
        v5 = _v5_engine()
        result = v5.phonemize("uh uh, haha. ha ha ab", language="en_us")
        self.assertEqual(result["phones"], ["<sil>", "ˈ", "ʌ", "̃", "ʔ", "<sil>", "ʌ", "̃", "<pause_comma>", "<sil>", "h", "ɑ", "h", "ˈ", "ɑ",
                                            "<end_stmt>", "<sil>", "h", "ɑ", "<sil>", "h", "ˈ", "ɑ", "<sil>", "a", "b", "<sil>"])
        self.assertEqual(result["word_starts"], [6, 10, 17, 20, 24])
        self.assertEqual(_fake_engine(v4=True).phonemize("haha. ha ha ab", language="en_us")["phones"],
                         ["<sil>", "h", "a", "h", "a", "<end_stmt>", "<sil>", "h", "a", "<sil>", "h", "a", "<sil>", "a", "b", "<sil>"])


SCORE = dict(text="hello bob wow", bpm=1200, style="pop",
             notes=[["hello", "C4", 2], ["-", "rest", 6], ["bob~1", "D4", 1], ["bob~2", "E4", 1], ["wow", "G4", 2]])
PAD = 0.02


class SingingV5Test(unittest.TestCase):
    def test_styles_come_from_the_bundle_table(self):
        engine = _v5_engine()
        self.assertEqual(singing_styles(engine), [*SUNG, "lullaby"])
        self.assertEqual(style_modes(engine)["lullaby"], "ballad")
        for style, row in (("pop", 2), ("ballad", 1), ("rock", 10), ("lullaby", 1)):
            plan = plan_song(dict(SCORE, style=style), engine, "scylla", "en_us", pad=PAD)
            self.assertEqual((plan.score.style, plan.mode_id), (style, row))
        self.assertEqual(plan_song(parse_score(SCORE), engine, "scylla", "en_us", pad=PAD).mode, "pop")   # parsed without the bundle
        for style in ("sung", "rap", "spoken"):   # v4's mode name; a speech style; the spoken row
            with self.assertRaisesRegex(ScoreError, "Unknown style"):
                plan_song(dict(SCORE, style=style), engine, "scylla", "en_us", pad=PAD)
        moved = _v5_engine(delivery_modes=dict(V5_DELIVERY_MODES, ids=dict(STYLE_IDS, pop=7, punk=2)))
        self.assertEqual(plan_song(SCORE, moved, "scylla", "en_us", pad=PAD).mode_id, 7)
        self.assertEqual(singing_styles(_fake_engine(v4=True)), ["pop", "sung", "ballad"])

    def test_a_sung_phrase_feeds_its_row_identity_and_the_manifest_guidance(self):
        engine = _v5_engine()
        result = sing(engine, dict(SCORE, style="folk"), "scylla", "en_us", seed=0, steps=2, pad=PAD)
        names = {c["name"].rsplit("_", 1)[0] if c["name"][-1].isdigit() else c["name"] for c in engine._fake.calls}
        self.assertFalse({"duration_predictor", "pitch_estimator", "pitch_context_encoder"} & names)
        flows = _calls(engine, "vector_estimator")
        self.assertEqual(len(flows), 2 * 8)                                # two phrases, two Heun steps, each evaluation guided
        self.assertEqual([c["delivery_mode_id"].tolist()[0] for c in flows], [3, 0] * 8)
        for on, off in zip(flows[::2], flows[1::2]):
            self.assertTrue(on["expanded_melody"].any() and not off["expanded_melody"].any())
            self.assertFalse(on["delivery_present"].any())
            np.testing.assert_array_equal(off["identity_reference"], np.asarray([IDENTITY], np.float32))
        self.assertEqual((result.metadata["guidance"], result.metadata["decoder"], result.metadata["mode_id"]), (3.0, "bigvgan", 3))
        self.assertNotIn("guidance_steps", result.metadata)

    def test_guidance_on_the_first_half_of_the_steps(self):
        modes = json.loads(json.dumps(V5_DELIVERY_MODES))
        modes["singing"]["guidance_steps"] = "first_half"
        engine = _v5_engine(delivery_modes=modes)
        result = sing(engine, SCORE, "scylla", "en_us", seed=0, steps=4, pad=PAD)
        flows = _calls(engine, "vector_estimator")
        # Heun, 4 steps: evaluations at t = 0, .25, .25, .5, .5, .75, .75, 1; those before 0.5 run twice.
        self.assertEqual(len(flows), 2 * (8 + 3))
        self.assertEqual([round(float(c["time"][0]), 2) for c in flows[:11]], [0, 0, .25, .25, .25, .25, .5, .5, .75, .75, 1.0])
        self.assertEqual([c["delivery_mode_id"].tolist()[0] for c in flows[:11]], [2, 0, 2, 0, 2, 0, 2, 2, 2, 2, 2])
        self.assertEqual(result.metadata["guidance_steps"], "first_half")
        modes["singing"]["guidance_steps"] = "last"
        with self.assertRaisesRegex(ValueError, "guidance_steps"):
            sing(_v5_engine(delivery_modes=modes), SCORE, "scylla", "en_us", pad=PAD)
        # The engine's own switch: no window guides every evaluation, as on v4.
        v4 = _fake_engine(v4=True)
        sentence = Sentence(phones=["<sil>", "a", "b", "h", "l", "<sil>"], word_starts=[])
        delivery = (np.zeros((1, 5), np.float32), np.zeros((1, 5), bool))
        v4.synthesize_sentence(sentence, voice="scylla", language="en_us", delivery=delivery, steps=2, sampler="heun", durations=[1] * 6,
                               melody=np.full((6, 3), 0.5, np.float32), delivery_mode=1, guidance=2.0, rng=np.random.default_rng(0))
        self.assertEqual(len(_calls(v4, "vector_estimator")), 8)

    def test_a_v5_bundle_must_state_its_guidance(self):
        modes = json.loads(json.dumps(V5_DELIVERY_MODES))
        del modes["singing"]["guidance"]
        engine = _v5_engine(delivery_modes=modes)
        with self.assertRaisesRegex(ValueError, "singing guidance"):
            sing(engine, SCORE, "scylla", "en_us", pad=PAD)
        self.assertEqual(sing(engine, SCORE, "scylla", "en_us", seed=0, steps=1, guidance=1.0, pad=PAD).metadata["guidance"], 1.0)

    def test_the_singing_decoder_is_the_one_the_manifest_names(self):
        def renamed(required):
            def mutate(manifest):
                spec = dict(manifest["controls"]["decoders"].pop("bigvgan"), component="vocoder_wave")
                spec.pop("required", None)
                if required is not None:
                    spec["required"] = required
                manifest["controls"]["decoders"]["wave"] = spec
                manifest["controls"]["delivery_modes"]["singing"]["decoder"] = "wave"
                for bucket in manifest["controls"]["target_buckets"]["buckets"]:
                    bucket["vocoder_wave"] = bucket.pop("vocoder_bigvgan")
            return mutate
        engine = _v5_engine(mutate=renamed(None))
        result = sing(engine, SCORE, "scylla", "en_us", seed=0, steps=1, guidance=1.0, pad=PAD)
        self.assertEqual(result.metadata["decoder"], "wave")
        self.assertTrue(_calls(engine, "vocoder_bigvgan") and not [c for c in engine._fake.calls if c["name"] in ("vocoder", "vocoder_8")])
        self.assertEqual(sing(engine, SCORE, "scylla", "en_us", seed=0, steps=1, guidance=1.0, decoder="vocos", pad=PAD).metadata["decoder"],
                         "vocos")

        def without_graphs(required):
            def mutate(manifest):
                renamed(required)(manifest)
                for bucket in manifest["controls"]["target_buckets"]["buckets"]:
                    del bucket["vocoder_wave"]
            return mutate
        with self.assertRaisesRegex(ValueError, "no 'wave' decoder"):
            sing(_v5_engine(mutate=without_graphs(None)), SCORE, "scylla", "en_us", guidance=1.0, pad=PAD)
        optional = sing(_v5_engine(mutate=without_graphs(False)), SCORE, "scylla", "en_us", seed=0, steps=1, guidance=1.0, pad=PAD)
        self.assertEqual(optional.metadata["decoder"], "vocos")

    def test_the_register_keeps_the_song_above_the_floor(self):
        low = dict(bpm=1200, notes=[["hello", "C2", 2], ["bob", "C5", 2]])   # mean 54: the register leaves it, C2 is under F2
        self.assertEqual(plan_song(low, _v5_engine(), "scylla", "en_us", pad=PAD).register_shift, 0)
        modes = json.loads(json.dumps(V5_DELIVERY_MODES))
        modes["singing"]["register_floor"] = "F2"
        engine = _v5_engine(delivery_modes=modes)
        plan = plan_song(low, engine, "scylla", "en_us", pad=PAD)
        self.assertEqual((plan.register_shift, min(plan.score.pitched)), (12, 48))
        self.assertEqual(plan_song(low, engine, "scylla", "en_us", register=False, pad=PAD).register_shift, 0)
        self.assertEqual(plan_song(SCORE, engine, "scylla", "en_us", pad=PAD).register_shift, 0)
        modes["singing"]["register_floor"] = 41
        self.assertEqual(plan_song(low, _v5_engine(delivery_modes=modes), "scylla", "en_us", pad=PAD).register_shift, 12)


def _line_g2p(table):
    """A phonemizer that returns ``table[text]`` (phone groups, one per word it hears) in the G2P's layout."""
    def phonemize(text, language):
        phones, starts = ["<sil>"], []
        for index, group in enumerate(table[text]):
            if index:
                phones.append("<sil>")
                starts.append(len(phones))
            phones.extend(group)
        return dict(phones=[*phones, "<end_stmt>", "<sil>"], word_starts=starts)
    return phonemize


def _word(text, first, end):
    return Word(text, first * HOP / SAMPLE_RATE, end * HOP / SAMPLE_RATE)


class SungLyricsV5Test(unittest.TestCase):
    LINES = {"on the way": [["ɒ", "n", "ð", "ə"], ["w", "ˈ", "e", "ɪ"]], "on": [["ˈ", "ɒ", "n"]], "the": [["ð", "ˈ", "i", "ː"]],
             "way": [["w", "ˈ", "e", "ɪ"]], "far away": [["f", "ˈ", "ɑ", "ː", "ɹ", "ɐ", "w", "ˈ", "e", "ɪ"]], "far": [["f", "ˈ", "ɑ", "ː"]],
             "away": [["ɐ", "w", "ˈ", "e", "ɪ"]], "see ah": [["s", "ˈ", "i", "ː", "ɑ", "ː"]], "see": [["s", "ˈ", "i", "ː"]],
             "ah": [["ɑ", "ː"]], "to me": [["t", "ə"], ["m", "ˈ", "i", "ː"]], "a b c": [["a"], ["b"]], "a": [["a"]], "b": [["b"]],
             "c": [["s", "ˈ", "i", "ː"]]}

    def test_one_group_per_word_is_used_as_it_is(self):
        calls = []

        def phonemize(text, language):
            calls.append(text)
            return _line_g2p(self.LINES)(text, language)
        self.assertEqual(line_phones(["to", "me"], phonemize, "en_us"), [["t", "ə"], ["m", "ˈ", "i", "ː"]])
        self.assertEqual(calls, ["to me"])   # the phrase, once: no word is read alone

    def test_joined_words_are_cut_from_the_phrase_reading(self):
        g2p = _line_g2p(self.LINES)
        # "the" keeps its weak vowel from the phrase, not the "thee" it has alone; the linking r goes to the next word.
        self.assertEqual(line_phones(["on", "the", "way"], g2p, "en_us"), [["ɒ", "n"], ["ð", "ə"], ["w", "ˈ", "e", "ɪ"]])
        self.assertEqual(line_phones(["far", "away"], g2p, "en_us"), [["f", "ˈ", "ɑ", "ː"], ["ɹ", "ɐ", "w", "ˈ", "e", "ɪ"]])
        self.assertEqual(line_phones(["see", "ah"], g2p, "en_us"), [["s", "ˈ", "i", "ː"], ["ɑ", "ː"]])   # no piece starts on a length mark
        with self.assertRaisesRegex(ScoreError, "Could not cut"):
            line_phones(["a", "b", "c"], g2p, "en_us")

    def test_the_pronoun_i_is_stressed_as_the_sung_rows_have_it(self):
        table = {"i know": [["a", "ɪ"], ["n", "ˈ", "o", "ʊ"]], "I know": [["a", "ɪ"], ["n", "ˈ", "o", "ʊ"]],
                 "for i know": [["f", "ɔ", "ː", "ɹ", "a", "ɪ"], ["n", "ˈ", "o", "ʊ"]], "for": [["f", "ɔ", "ː"]], "i": [["ˈ", "a", "ɪ"]],
                 "know": [["n", "ˈ", "o", "ʊ"]], "i'm i'll i'd i've": [["a", "ɪ", "m"], ["a", "ɪ", "l"], ["a", "ɪ", "d"], ["a", "ɪ", "v"]],
                 "it is i": [["ɪ", "t"], ["ɪ", "z"], ["ˈ", "a", "ɪ"]], "si i": [["s", "i"], ["i"]]}
        g2p = _line_g2p(table)
        for language in ("en_us", "en_gb"):
            self.assertEqual(line_phones(["i", "know"], g2p, language), [["ˈ", "a", "ɪ"], ["n", "ˈ", "o", "ʊ"]])
            self.assertEqual(line_phones(["I", "know"], g2p, language), [["ˈ", "a", "ɪ"], ["n", "ˈ", "o", "ʊ"]])
            # After a linking r the consonant stays in front of the stress mark; this line goes through the cut.
            self.assertEqual(line_phones(["for", "i", "know"], g2p, language),
                             [["f", "ɔ", "ː"], ["ɹ", "ˈ", "a", "ɪ"], ["n", "ˈ", "o", "ʊ"]])
            # Contractions keep the G2P's reading, and a pronoun that is already stressed is not marked twice.
            self.assertEqual(line_phones(["i'm", "i'll", "i'd", "i've"], g2p, language),
                             [["a", "ɪ", "m"], ["a", "ɪ", "l"], ["a", "ɪ", "d"], ["a", "ɪ", "v"]])
            self.assertEqual(line_phones(["it", "is", "i"], g2p, language)[-1], ["ˈ", "a", "ɪ"])
        self.assertEqual(line_phones(["si", "i"], g2p, "it"), [["s", "i"], ["i"]])   # English only
        self.assertEqual(stressed_pronoun(["<sil>"]), ["<sil>"])
        self.assertEqual(stressed_pronoun(["ɹ", "ˌ", "a", "ɪ"]), ["ɹ", "ˈ", "a", "ɪ"])   # one stress mark, the primary one

    def test_speech_does_not_stress_the_pronoun(self):
        engine = _v5_engine()
        with mock.patch.object(engine.g2p, "_predict_raw", return_value=dict(phones=["a", "i"], confidence=1.0)):
            self.assertEqual(engine.phonemize("i", language="en_us")["phones"], ["<sil>", "a", "i", "<sil>"])

    def test_timing_follows_where_the_words_really_end(self):
        phones, starts = word_layout([["b", "a", "b"], ["a"], ["s", "t", "ɹ", "ˈ", "ᵻ", "t", "s"]])
        words = [_word("bab", 2, 3), _word("a", 4, 8), _word("streets", 8, 12)]
        # "bab" needs 3 frames for its 1: "a" starts a frame late and ends on time; "streets" overruns and the last
        # silence still reaches the phrase end.
        self.assertEqual(score_durations(phones, starts, words, 20, 0.51), [2, 1, 1, 1, 0, 3, 0, 1, 1, 1, 0, 1, 1, 1, 6])
        self.assertEqual(phone_durations(phones, starts, words, 20, 0.51), [2, 1, 1, 1, 0, 4, 0, 1, 1, 1, 0, 1, 1, 1, 6])   # the v4 rule
        self.assertEqual(score_durations(phones, starts, words, 10, 0.51)[-1], 2)   # at least 2 frames close the phrase

    def test_vowels_consonants_and_marks(self):
        phones, starts = word_layout([["m", "̩", "ː"], ["l", "ˈ", "a", "ː"], ["s", "ᵻ", "z"]])
        words = [_word("mm", 1, 5), _word("la", 6, 16), _word("ses", 16, 26)]
        self.assertEqual(score_durations(phones, starts, words, 27, 0.51), [1, 4, 0, 0, 1, 5, 0, 5, 0, 0, 2, 6, 2, 2])
        # A consonant gives way while the vowels would be left under one frame each (rap share: 2 frames, then 1).
        phones, starts = word_layout([["t", "a", "e", "o"]])
        self.assertEqual(score_durations(phones, starts, [_word("tao", 0, 4)], 6, 0.42), [0, 1, 1, 1, 1, 2])
        with self.assertRaisesRegex(ScoreError, "not one word"):
            score_durations(["<sil>", "a", "<sil>", "b", "<sil>"], [], [_word("ab", 0, 4)], 6, 0.51)

    def test_lyrics_are_read_without_the_nonverbal_table(self):
        engine = _v5_engine()
        self.assertEqual(engine.phonemize("aha", language="en_us")["phones"], ["<sil>", "ɑ", "ː", "h", "ˈ", "ɑ", "ː", "<sil>"])
        self.assertEqual(engine.phonemize("aha", language="en_us", nonverbal=False)["phones"], ["<sil>", "a", "h", "a", "<sil>"])
        score = dict(bpm=1200, style="pop", notes=[["aha", "C4", 2], ["hello", "D4", 2]])
        with mock.patch.object(engine, "phonemize", wraps=engine.phonemize) as phonemize:
            plan = plan_song(score, engine, "scylla", "en_us", pad=PAD)
        self.assertEqual(plan.phrases[0].sentence.phones, ["<sil>", "a", "h", "a", "<sil>", "h", "o", "l", "l", "o", "<sil>"])
        self.assertEqual(phonemize.call_args_list, [mock.call("aha hello", language="en_us", nonverbal=False)])   # the phrase, once

    def test_a_note_x_is_sung_without_pitch_on_v5_only(self):
        score = dict(bpm=1200, style="pop", notes=[["hello", "C4", 1], ["bob", "x", 1], ["wow", "G4", 1]])
        plan = plan_song(score, _v5_engine(), "scylla", "en_us", pad=PAD)
        phrase = plan.phrases[0]
        self.assertEqual([w.text for w in phrase.phrase.words], ["hello", "bob", "wow"])
        self.assertEqual([n.name for n in phrase.phrase.notes], ["C4", "x", "G4", "rest"])
        centres = (np.arange(phrase.frames) * HOP + HOP // 2) / SAMPLE_RATE
        first, silent, last = phrase.phrase.notes[:3]
        under = lambda note: (centres >= note.start) & (centres < note.end)   # noqa: E731
        self.assertTrue(under(silent).any() and not phrase.melody[under(silent)].any())
        self.assertTrue(phrase.melody[under(first), 2].all() and phrase.melody[under(last), 2].all())
        self.assertEqual(sum(phrase.durations[phrase.sentence.phones.index("b"):][:3]), 3)   # the word keeps its frames (one per phone here)
        self.assertEqual(plan.score.pitched, [60, 67])
        with self.assertRaisesRegex(ScoreError, "Unknown note"):
            plan_song(score, _fake_engine(v4=True), "scylla", "en_us", pad=PAD)
        with self.assertRaisesRegex(ScoreError, "Unknown note"):
            parse_score(score)

    def test_a_v4_bundle_sings_by_its_own_rules(self):
        from scyllasband import singing
        engine = _fake_engine(v4=True)
        refuse = mock.Mock(side_effect=AssertionError("a v5 rule ran on a v4 bundle"))
        with mock.patch.object(singing, "line_phones", refuse), mock.patch.object(singing, "score_durations", refuse), \
                mock.patch.object(singing, "word_layout", refuse), mock.patch.object(engine, "phonemize", wraps=engine.phonemize) as phonemize:
            result = sing(engine, SCORE, "scylla", "en_us", seed=3, steps=2, pad=PAD)
        calls = list(engine._fake.calls)
        self.assertFalse(any("nonverbal" in c or "identity_reference" in c or "style_id" in c for c in calls))
        # Phones and lengths are what the v4 functions give for each phrase, from one G2P call per phrase.
        plan = plan_song(SCORE, engine, "scylla", "en_us", pad=PAD)
        for item, rendered in zip(plan.phrases, result.metadata["phrases"]):
            words = [w.text for w in item.phrase.words]
            phones, starts = word_phones(words, engine.phonemize, "en_us")
            total = int(round(item.phrase.notes[-1].end * SAMPLE_RATE / HOP)) + 2
            self.assertEqual((rendered["phones"], rendered["durations"]),
                             (phones, phone_durations(phones, starts, item.phrase.words, total, 0.51)))
        self.assertEqual(phonemize.call_args_list, [mock.call(item.phrase.text, language="en_us") for item in plan.phrases])
        # The same song again: the same graphs get the same inputs, call for call, and the audio is the same.
        again = _fake_engine(v4=True)
        repeat = sing(again, SCORE, "scylla", "en_us", seed=3, steps=2, pad=PAD)
        self.assertEqual([c["name"] for c in calls], [c["name"] for c in again._fake.calls])
        for a, b in zip(calls, again._fake.calls):
            for key in a:
                if key != "name":
                    np.testing.assert_array_equal(a[key], b[key], key)
        np.testing.assert_array_equal(result.audio, repeat.audio)


if __name__ == "__main__":
    unittest.main()
