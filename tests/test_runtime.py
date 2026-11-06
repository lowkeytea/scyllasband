"""Runtime tests that need no model files: fake graphs stand in for the bundle's sessions."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from scyllasband.contract import (BundleError, CONTEXT_INPUTS, DURATION_INPUTS, GRAPH_CONTRACT, GRAPH_CONTRACT_V4, PITCH_INPUTS,
                                  VECTOR_INPUTS, VECTOR_INPUTS_V4, VOCODER_INPUTS, validate_bundle_layout)
from scyllasband.delivery import CONTROLS, DEFAULT_RANGES, delivery_tensors, resolve_delivery, with_defaults
from scyllasband.engine import Engine, OverlongError, Sentence, balanced_context
from scyllasband.events import frame_events, frames_to_durations, modifier_bits, sentence_types
from scyllasband.g2p import punctuated_segments
from scyllasband.planner import parse_group_lines, plan_records, records_from_text, split_for_retry, split_sentences
from scyllasband.streaming import StreamOptions, synthesize_plan_stream

VOCAB = ["<pad>", "<unk>", "<sil>", "<pause_comma>", "<pause_semicolon>", "<pause_colon>", "<pause_dash>", "<ellipsis>", "<end_stmt>",
         "<end_question>", "<end_exclaim>", "a", "b", "h", "l", "o", "w", "ˈ", "ː"]
PHONE_TO_ID = {p: i for i, p in enumerate(VOCAB)}
# The v4 manifest controls of the singing release, as the fake bundle declares them (tests copy and adjust them).
V4_DELIVERY_MODES = dict(enabled=True, input="delivery_mode_id", ids=dict(spoken=0, sung=1, rap=2, ballad=3), default="spoken",
                         singing=dict(modes=["sung", "ballad"], sampler="heun", steps=8, decoder="bigvgan",
                                      melody=dict(layout="note_phase_presence", midi_center=60, midi_scale=24),
                                      pitch_predictor="bypassed", pitch_soft_limit="bypassed"))
V4_DECODERS = dict(vocos=dict(component="vocoder", samples_per_frame=512, output_samples="latent_frames * 512 - 256"),
                   bigvgan=dict(component="vocoder_bigvgan", samples_per_frame=512, output_samples="latent_frames * 512", required=False))


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

    def test_pause_scale_moves_only_silences(self):
        phones = ["<sil>", "a", "<sil>", "b", "<sil>"]
        self.assertEqual(frames_to_durations([4, 4, 0.4, 4, 2], phones, pause_scale=1.3), [5, 4, 1, 4, 3])
        self.assertEqual(frames_to_durations([4, 4, 0.4, 4, 2], phones, pause_scale=0.7), [3, 4, 0, 4, 1])

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

    def test_letters_glued_to_digits_are_split(self):
        from scyllasband.text_normalizer import normalize_spoken_text
        self.assertEqual(normalize_spoken_text("Whisper Large V3 beat v3.5.", language="en_us"),
                         "Whisper Large vee three beat vee three point five.")
        self.assertEqual(normalize_spoken_text("MP3 in 4K, H2O, Python3 and COVID19.", language="en_us"),
                         "em pee three in four kay, aitch two oh, Python three and COVID nineteen.")
        self.assertEqual(normalize_spoken_text("La V3 y el MP3.", language="es"), "La uve tres y el eme pe tres.")
        self.assertEqual(normalize_spoken_text("test_voice2.py", language="en_us"), "test_voice2.py")

    def test_english_decades_clock_times_and_years(self):
        from scyllasband.text_normalizer import normalize_spoken_text
        self.assertEqual(normalize_spoken_text("The 1990s, the '80s and her 20s.", language="en_us"),
                         "The nineteen nineties, the eighties and her twenties.")
        self.assertEqual(normalize_spoken_text("100s of 4s, two MP3s, Route 66's and 1.5s.", language="en_us"),
                         "hundreds of fours, two em pee threes, Route sixty six's and one point five seconds.")
        self.assertEqual(normalize_spoken_text("68x faster on a 4x4 grid, 1.5x, 20ms, 24kHz, 1 GB and 60fps.", language="en_us"),
                         "sixty eight times faster on a four by four grid, one point five times, twenty milliseconds, "
                         "twenty four kilohertz, one gigabyte and sixty frames per second.")
        self.assertEqual(normalize_spoken_text("At 10am, 5 PM or 7:05 AM.", language="en_us"),
                         "At ten ay em, five pee em or seven oh five ay em.")
        self.assertEqual(normalize_spoken_text("On March 4, 1990 and 3/4/2005.", language="en_us"),
                         "On March fourth, nineteen ninety and March fourth, two thousand five.")


class LexiconTest(unittest.TestCase):
    def test_entries_and_matching(self):
        from scyllasband.lexicon import LETTERS, WORDS, Lexicon, initialism_phones, split_phones
        self.assertEqual(initialism_phones("GPU", LETTERS["en_us"]), "dʒˌiːpˌiːjˈuː")
        spelled = "".join(t for table in LETTERS.values() for t in table.values()) + "".join(p for w in WORDS.values() for p in w.values())
        vocabulary = {ch: 1 for ch in spelled.replace("{s}", "") + "ˈˌzsˈɪlə"}
        self.assertEqual(split_phones("ɡˈɪt", vocabulary), ("ɡ", "ˈ", "ɪ", "t"))
        self.assertIsNone(split_phones("ɡˈɪt§", vocabulary))
        lexicon = Lexicon(vocabulary, {"en_us": {"Scylla": "sˈɪlə"}})
        self.assertEqual("".join(lexicon.lookup("GPUs", "en_us", initialisms=True)), "dʒˌiːpˌiːjˈuːz")
        self.assertEqual("".join(lexicon.lookup("Git", "en_us", initialisms=True)), "ɡˈɪt")
        self.assertEqual("".join(lexicon.lookup("scylla", "en_us", initialisms=True)), "sˈɪlə")
        self.assertIsNone(lexicon.lookup("GPU", "en_us", initialisms=False))   # text written all in capitals
        self.assertIsNone(lexicon.lookup("US", "en_us", initialisms=True))
        self.assertIsNone(lexicon.lookup("GPU", "es", initialisms=True))

    def test_native_lexicon_table_is_current(self):
        import importlib.util
        path = Path(__file__).resolve().parents[1] / "libscyllasband" / "tools" / "generate_lexicon.py"
        spec = importlib.util.spec_from_file_location("generate_lexicon", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.OUT.read_text(encoding="utf-8"), module.render())


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
        records = parse_group_lines("[scylla:en_us] Hi there. How are you?\n[ink:en_gb:intensity=1,pace=-0.5] Fine. [scylla] Good.",
                                    default_voice=None, default_language=None)
        plan = plan_records(records, resolve_language=lambda voice, language: language or "en_us", normalize=False)
        self.assertEqual([c.chain_id for c in plan.chunks], ["chain_001", "chain_001", "chain_002", "chain_003"])
        self.assertEqual(plan.chunks[2].delivery, dict(intensity=1.0, pace=-0.5, pitch=None))
        self.assertIsNone(plan.chunks[0].delivery)

    def test_delivery_fields_or_string_not_both(self):
        plan = plan_records(records_from_text("Hi.", voice="scylla", language=None, pace=-1), resolve_language=lambda v, l: "en_us")
        self.assertEqual(plan.chunks[0].delivery, dict(intensity=None, pace=-1.0, pitch=None))
        with self.assertRaisesRegex(ValueError, "not both"):
            plan_records(records_from_text("Hi.", voice="scylla", language=None, delivery="pace=1", pace=-1),
                         resolve_language=lambda v, l: "en_us")
        with self.assertRaisesRegex(ValueError, "emotion comes from the text"):
            plan_records([dict(text="Hi.", voice="scylla", emotion="joy")], resolve_language=lambda v, l: "en_us")


class DeliveryTest(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(resolve_delivery(None), dict.fromkeys(CONTROLS))
        self.assertEqual(resolve_delivery(""), resolve_delivery("default"))
        self.assertEqual(with_defaults(resolve_delivery("pace=-1"), dict(pace=0.5, pitch=1)), dict(intensity=0.0, pace=-1.0, pitch=1.0))
        values, present = delivery_tensors(with_defaults({}, None))
        self.assertEqual(present[0].tolist(), [True, True, False, False, False])
        self.assertEqual(values[0].tolist(), [0, 0, 0, 0, 0])

    def test_values_are_z_over_five_and_clamped_into_the_ranges(self):
        resolved = resolve_delivery("intensity=5,pace=-3,pitch=2.5")
        self.assertEqual(resolved, dict(intensity=2.0, pace=-2.0, pitch=2.0))
        self.assertEqual(resolve_delivery({"pace": 1}, ranges=dict(pace=[-0.5, 0.5]))["pace"], 0.5)
        values, _ = delivery_tensors(dict(intensity=1.0, pace=-2.0, pitch=0.0))
        np.testing.assert_allclose(values[0, :2], [0.2, -0.4], rtol=1e-6)
        for low, high in DEFAULT_RANGES.values():
            self.assertTrue(low <= 0.0 <= high)

    def test_v2_axes_and_unknown_keys_are_rejected(self):
        for spec in ("energy=2", "tension=2,pace=1", {"whisper": False}, {"emotion": "joy"}):
            with self.assertRaisesRegex(ValueError, "intensity, pace and pitch"):
                resolve_delivery(spec)
        with self.assertRaisesRegex(ValueError, "only intensity"):
            resolve_delivery("tempo=1")
        with self.assertRaisesRegex(ValueError, "must be a number"):
            resolve_delivery("pace=fast")


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

    def test_v4_bundles_take_the_mode_input_and_optional_decoders(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = _fake_bundle(Path(tmp), v4=True)
            summary = validate_bundle_layout(bundle)
            self.assertEqual((summary["graph_contract"], summary["singing"]), (GRAPH_CONTRACT_V4, True))
            manifest = json.loads((bundle / "manifest.json").read_text())
            # A v3 flow under the v4 contract (or the reverse) is rejected.
            broken = json.loads(json.dumps(manifest))
            broken["components"]["vector_estimator"]["inputs"] = list(VECTOR_INPUTS)
            (bundle / "manifest.json").write_text(json.dumps(broken))
            with self.assertRaisesRegex(BundleError, "vector_estimator"):
                validate_bundle_layout(bundle)
            broken = json.loads(json.dumps(manifest))
            broken["controls"]["graph_input_contract"] = GRAPH_CONTRACT
            (bundle / "manifest.json").write_text(json.dumps(broken))
            with self.assertRaisesRegex(BundleError, "vector_estimator"):
                validate_bundle_layout(bundle)
            # The second decoder need not exist for every backend, but a bucket may not name a component that is absent.
            tolerant = json.loads(json.dumps(manifest))
            tolerant["components"]["vocoder_bigvgan_8"]["artifacts"] = {}
            (bundle / "manifest.json").write_text(json.dumps(tolerant))
            self.assertTrue(validate_bundle_layout(bundle)["singing"])
            del tolerant["components"]["vocoder_bigvgan"]
            (bundle / "manifest.json").write_text(json.dumps(tolerant))
            with self.assertRaisesRegex(BundleError, "vocoder_bigvgan"):
                validate_bundle_layout(bundle)

    def test_v3_bundles_validate_without_the_mode_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = validate_bundle_layout(_fake_bundle(Path(tmp)))
            self.assertEqual((summary["graph_contract"], summary["singing"]), (GRAPH_CONTRACT, False))


class LiteRTThreadsTest(unittest.TestCase):
    def test_default_threads_follow_the_native_runtime(self):
        from unittest import mock
        from scyllasband import litert
        with mock.patch.object(litert.os, "sched_getaffinity", return_value=set(range(16)), create=True):
            self.assertEqual(litert.default_threads(), 4)
        with mock.patch.object(litert.os, "sched_getaffinity", return_value={0, 1}, create=True):
            self.assertEqual(litert.default_threads(), 2)

    def test_interpreters_get_the_default_unless_threads_are_given(self):
        from unittest import mock
        from scyllasband import litert
        made = []

        class Interpreter:
            def __init__(self, model_path, num_threads):
                made.append(num_threads)

        spec = {"artifacts": {"litert": {"path": "g2p.tflite"}}, "inputs": []}
        with mock.patch.object(litert, "_interpreter_class", return_value=Interpreter), \
                mock.patch.object(litert, "LiteRTSession"), mock.patch.object(litert, "default_threads", return_value=3):
            litert.litert_session_factory(Path("."))("g2p", spec)
            litert.litert_session_factory(Path("."), threads=8)("g2p", spec)
        self.assertEqual(made, [3, 8])


class OnnxProvidersTest(unittest.TestCase):
    def test_compute_units_pick_execution_providers(self):
        from scyllasband.engine import onnx_providers
        cpu, cuda = ["CPUExecutionProvider"], ["CUDAExecutionProvider", "CPUExecutionProvider"]
        for value in (None, "", "auto", "cpu"):
            self.assertEqual(onnx_providers(value)("vector_estimator_256"), cpu)
        per_graph = onnx_providers("gpu,g2p=cpu")
        self.assertEqual(per_graph("vector_estimator_256"), cuda)
        self.assertEqual(per_graph("g2p"), cpu)
        self.assertEqual(onnx_providers("cuda")("vocoder"), cuda)
        self.assertEqual(onnx_providers({"vocoder": "gpu"})("vocoder"), cuda)
        with self.assertRaises(ValueError):
            onnx_providers("ane")

    def test_a_cuda_session_that_falls_back_to_the_cpu_is_an_error(self):
        import sys
        import types
        from unittest import mock
        from scyllasband.engine import OnnxSession, _preload_cuda

        class Session:
            def __init__(self, path, options, providers):
                pass

            def get_providers(self):
                return ["CPUExecutionProvider"]

            def get_inputs(self):
                return []

        fake = types.SimpleNamespace(SessionOptions=lambda: types.SimpleNamespace(), InferenceSession=Session,
                                     preload_dlls=lambda: None)
        _preload_cuda.cache_clear()
        with mock.patch.dict(sys.modules, {"onnxruntime": fake}):
            OnnxSession(Path("g2p.onnx"))   # CPU only: nothing to check
            with self.assertRaisesRegex(RuntimeError, r"\.\[cuda\]"):
                OnnxSession(Path("g2p.onnx"), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        _preload_cuda.cache_clear()

    def test_gpu_requests_use_the_webgpu_bundle(self):
        import argparse
        from scyllasband.cli import _default_bundle, wants_gpu
        self.assertTrue(wants_gpu("gpu") and wants_gpu("cuda") and wants_gpu("cpu,vector_estimator=gpu"))
        self.assertFalse(wants_gpu(None) or wants_gpu("cpu") or wants_gpu("gpu_like=cpu"))
        with tempfile.TemporaryDirectory() as tmp:
            for flavor in ("onnx", "webgpu"):
                Path(tmp, flavor).mkdir()
                Path(tmp, flavor, "manifest.json").write_text("{}")
            args = lambda units: argparse.Namespace(models_dir=tmp, backend="onnx", compute_units=units)
            self.assertEqual(_default_bundle(args("gpu")).name, "webgpu")
            self.assertEqual(_default_bundle(args(None)).name, "onnx")


class EngineTest(unittest.TestCase):
    def test_bucket_padding_prefix_and_decode(self):
        engine = _fake_engine()
        sentence = Sentence(phones=["<sil>", "a", "b", "<end_stmt>", "<sil>"], word_starts=[], before_ids=[11, 12], after_ids=[13])
        delivery = engine.delivery_inputs("scylla", None)[:2]
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
        delivery = engine.delivery_inputs("scylla", None)[:2]
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
            engine.synthesize_sentence(sentence, voice="scylla", language="en_us", delivery=engine.delivery_inputs("scylla", None)[:2],
                                       steps=1)

    def test_short_targets_leave_intensity_and_pace_unset(self):
        engine = _fake_engine()
        delivery = engine.delivery_inputs("scylla", dict(intensity=1.0, pace=-1.0))[:2]
        short = Sentence(phones=["<sil>", "ˈ", "a", "b", "a", "<end_stmt>", "<sil>"], word_starts=[])   # 3 spoken phones
        long = Sentence(phones=["<sil>", "a", "b", "a", "b", "<end_stmt>", "<sil>"], word_starts=[])     # 4
        for sentence, expected in ((short, [False] * 5), (long, [True, True, False, False, False])):
            engine._fake.calls.clear()
            engine.synthesize_sentence(sentence, voice="scylla", language="en_us", delivery=delivery, steps=1, rng=np.random.default_rng(0))
            for call in engine._fake.calls:
                if "delivery_present" in call:
                    self.assertEqual(call["delivery_present"][0].tolist(), expected, call["name"])

    def test_pace_moves_pauses_even_when_unset(self):
        engine = _fake_engine(frames_per_phone=4.0)
        sentence = Sentence(phones=["<sil>", "a", "b", "<end_stmt>", "<sil>"], word_starts=[])
        normal = engine.durations(sentence, voice=0, language=0, delivery=engine.delivery_inputs("scylla", None)[:2])
        slow = engine.durations(sentence, voice=0, language=0, delivery=engine.delivery_inputs("scylla", dict(pace=-2.0))[:2])
        self.assertEqual(normal, [4, 4, 4, 0, 4])
        self.assertEqual(slow, [5, 4, 4, 0, 5])      # 4 x exp(0.26), the words left as the graph gives them

    def test_pitch_contour_feeds_the_flow_melody(self):
        engine = _fake_engine()
        sentence = Sentence(phones=["<sil>", "a", "b", "a", "b", "<end_stmt>", "<sil>"], word_starts=[])
        result = engine.synthesize_sentence(sentence, voice="scylla", language="en_us", delivery=(*engine.delivery_inputs("scylla", None)[:2], 1.5),
                                            steps=1, rng=np.random.default_rng(0), pitch_noise=np.full((2, 6), 0.25, np.float32))
        calls = engine._fake.calls
        pitch = [c for c in calls if c["name"] == "pitch_estimator_8"]
        self.assertEqual(len(pitch), engine.pitch_steps)
        self.assertEqual(pitch[0]["noise"].shape, (1, 2, 8))
        self.assertEqual(pitch[0]["span_context_hidden"].shape, (1, 384))
        flow = next(c for c in calls if c["name"] == "vector_estimator_8")
        melody = flow["expanded_melody"]
        self.assertEqual(melody.shape, (1, 8, 3))
        self.assertTrue((melody[0, 6:] == 0).all())                       # bucket padding
        contour = np.full((2, 6), 0.25 * (1 - 1 / engine.pitch_steps) ** engine.pitch_steps, np.float32)
        np.testing.assert_allclose(melody[0, :6], engine.melody(contour, "scylla", 1.5), rtol=1e-5)
        self.assertEqual(result.metadata["pitch_shift"], 1.5)

    def test_melody_soft_limit_and_voicing(self):
        engine = _fake_engine()
        low, high = engine.pitch["edges"]["scylla"]
        contour = np.array([[(high + 5 - 10) / 6, (low - 5 - 10) / 6, 0.0], [1.0, 1.0, -1.0]], np.float32)
        melody = engine.melody(contour, "scylla")
        st = melody[:2, 0] * 24 + 16.65
        self.assertTrue(high - 2 < st[0] < high and low < st[1] < low + 2)
        self.assertEqual(melody[:, 2].tolist(), [1.0, 1.0, 0.0])
        self.assertEqual(float(melody[2, 0]), 0.0)                       # unvoiced: no pitch

    def _spoken(self, engine, **kwargs):
        sentence = Sentence(phones=["<sil>", "a", "b", "a", "b", "<end_stmt>", "<sil>"], word_starts=[3], before_ids=[11], after_ids=[12])
        delivery = (*engine.delivery_inputs("scylla", dict(intensity=0.5))[:2], 1.0)
        return engine.synthesize_sentence(sentence, voice="scylla", language="en_us", delivery=delivery, steps=2, sampler="heun",
                                          rng=np.random.default_rng(7), prefix=np.ones((24, 2), np.float32), **kwargs)

    def test_v3_speech_feeds_exactly_the_v3_inputs_and_v4_adds_mode_zero(self):
        v3, v4 = _fake_engine(), _fake_engine(v4=True)
        spoken_v3, spoken_v4 = self._spoken(v3), self._spoken(v4)
        flows_v3 = [c for c in v3._fake.calls if c["name"].startswith("vector_estimator")]
        flows_v4 = [c for c in v4._fake.calls if c["name"].startswith("vector_estimator")]
        self.assertEqual(len(flows_v3), 4)
        for call in flows_v3:
            self.assertEqual(set(call) - {"name"}, set(VECTOR_INPUTS))
        self.assertFalse(any("delivery_mode_id" in call for call in v3._fake.calls))
        self.assertNotIn("delivery_mode", spoken_v3.metadata)
        # Spoken on v4: the same graphs are called with the same values, plus delivery_mode_id = 0 on the flow.
        self.assertEqual([c["name"] for c in v3._fake.calls], [c["name"] for c in v4._fake.calls])
        for call in flows_v4:
            self.assertEqual(set(call) - {"name"}, set(VECTOR_INPUTS_V4))
            self.assertEqual(call["delivery_mode_id"].tolist(), [0])
            self.assertEqual(call["delivery_mode_id"].dtype, np.int64)
        for a, b in zip(v3._fake.calls, v4._fake.calls):
            for key in a:
                if key != "name":
                    np.testing.assert_array_equal(a[key], b[key], key)
        np.testing.assert_array_equal(spoken_v3.latents, spoken_v4.latents)
        self.assertEqual(spoken_v4.metadata["delivery_mode"], 0)
        self.assertEqual(spoken_v4.metadata["melody_source"], "pitch_predictor")

    def test_a_score_melody_replaces_the_pitch_predictor(self):
        engine = _fake_engine(v4=True)
        melody = np.zeros((6, 3), np.float32)
        melody[1:5] = [[2 / 24, 0.5, 1.0]] * 4
        result = self._spoken(engine, melody=melody, delivery_mode=3)
        names = [c["name"] for c in engine._fake.calls]
        self.assertFalse(any(n.startswith("pitch_") for n in names))
        flow = next(c for c in engine._fake.calls if c["name"].startswith("vector_estimator"))
        np.testing.assert_array_equal(flow["expanded_melody"][0, :6], melody)
        self.assertTrue((flow["expanded_melody"][0, 6:] == 0).all())
        self.assertEqual(flow["delivery_mode_id"].tolist(), [3])
        self.assertEqual((result.metadata["delivery_mode"], result.metadata["melody_source"]), (3, "score"))
        with self.assertRaisesRegex(ValueError, r"\[6, 3\]"):
            self._spoken(engine, melody=np.zeros((5, 3), np.float32))
        with self.assertRaisesRegex(ValueError, "v4"):
            self._spoken(_fake_engine(), delivery_mode=1)

    def test_guidance_contrasts_the_melody_and_mode_against_off(self):
        engine = _fake_engine(v4=True)
        melody = np.full((6, 3), 0.5, np.float32)
        self._spoken(engine, melody=melody, delivery_mode=1, guidance=2.0)
        flows = [c for c in engine._fake.calls if c["name"].startswith("vector_estimator")]
        self.assertEqual(len(flows), 8)                                  # two evaluations per Heun stage
        self.assertEqual([c["delivery_mode_id"].tolist()[0] for c in flows], [1, 0] * 4)
        self.assertTrue((flows[1]["expanded_melody"] == 0).all() and (flows[0]["expanded_melody"][0, :6] == 0.5).all())

    def test_bigvgan_decoder_needs_no_spare_frame(self):
        engine = _fake_engine(v4=True)
        for frames in (9, 16):
            audio = engine.decode(np.zeros((24, frames), np.float32), voice="scylla", language="en_us", decoder="bigvgan")
            self.assertEqual(audio.shape, (frames * 512,))
            self.assertEqual(engine._fake.calls[-1]["name"], "vocoder_bigvgan")
            self.assertEqual(int(engine._fake.calls[-1]["latent_mask"].sum()), frames)
        engine.decode(np.zeros((24, 7), np.float32), voice="scylla", language="en_us", decoder="bigvgan")
        self.assertEqual(engine._fake.calls[-1]["name"], "vocoder_bigvgan_8")
        engine.decode(np.zeros((24, 7), np.float32), voice="scylla", language="en_us")
        self.assertEqual(engine._fake.calls[-1]["name"], "vocoder_8")    # vocos keeps its spare frame: 7 + 1 fits 8
        engine.decode(np.zeros((24, 8), np.float32), voice="scylla", language="en_us")
        self.assertEqual(engine._fake.calls[-1]["name"], "vocoder")
        with self.assertRaisesRegex(ValueError, "Unknown decoder"):
            _fake_engine().decode(np.zeros((24, 4), np.float32), voice="scylla", language="en_us", decoder="bigvgan")


class DecoderWindowTest(unittest.TestCase):
    """Windowed decoding (controls.decoders.<name>.window), as libscyllasband does it: with a BigVGAN stand-in whose
    samples are their frame's first latent channel, windows stitched from the frames each keeps equal one whole decode."""

    @staticmethod
    def _engine(window=None, *, single_bucket=False):
        tmp = tempfile.TemporaryDirectory()
        bundle = _fake_bundle(Path(tmp.name), v4=True)
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        if window is not None:
            manifest["controls"]["decoders"]["bigvgan"]["window"] = window
        if single_bucket:   # BigVGAN only at the 8-frame bucket
            for entry in manifest["controls"]["target_buckets"]["buckets"]:
                if entry["latent_frames"] != 8:
                    entry.pop("vocoder_bigvgan")
            del manifest["components"]["vocoder_bigvgan"]
        (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        graphs = _FakeGraphs(1.0)
        graphs.bigvgan_by_frame = True
        engine = Engine(bundle, session_factory=graphs.factory)
        engine._fake, engine._tmp = graphs, tmp
        return engine

    @staticmethod
    def _latents(frames, base):
        latents = np.full((24, frames), 0.5, np.float32)
        latents[0] = base + np.arange(frames, dtype=np.float32) / 100
        return latents

    def test_windows_stitch_to_the_whole_decode(self):
        whole, windowed = self._engine(), self._engine(dict(frames=8, context=2))
        self.assertEqual(windowed.decoder_windows, {"bigvgan": (8, 2)})
        latents = self._latents(13, 0.1)
        want = whole.decode(latents, voice="scylla", language="en_us", decoder="bigvgan")
        got = windowed.decode(latents, voice="scylla", language="en_us", decoder="bigvgan")
        np.testing.assert_array_equal(got, want)
        self.assertEqual([c["name"] for c in windowed._fake.calls], ["vocoder_bigvgan_8"] * 3)   # keeps [0, 6), [6, 10), [10, 13)
        left = self._latents(5, 0.7)
        np.testing.assert_array_equal(windowed.decode(latents, voice="scylla", language="en_us", left=left, decoder="bigvgan"),
                                      whole.decode(latents, voice="scylla", language="en_us", left=left, decoder="bigvgan"))
        short = self._latents(5, 0.3)
        windowed._fake.calls.clear()
        np.testing.assert_array_equal(windowed.decode(short, voice="scylla", language="en_us", decoder="bigvgan"),
                                      whole.decode(short, voice="scylla", language="en_us", decoder="bigvgan"))
        self.assertEqual([c["name"] for c in windowed._fake.calls], ["vocoder_bigvgan_8"])
        # Vocos is unchanged by a BigVGAN window.
        np.testing.assert_array_equal(windowed.decode(latents, voice="scylla", language="en_us"), whole.decode(latents, voice="scylla", language="en_us"))

    def test_a_single_bucket_decodes_any_length(self):
        engine = self._engine(dict(frames=8, context=2), single_bucket=True)
        latents = self._latents(13, 0.1)
        np.testing.assert_array_equal(engine.decode(latents, voice="scylla", language="en_us", decoder="bigvgan"),
                                      self._engine().decode(latents, voice="scylla", language="en_us", decoder="bigvgan"))
        with self.assertRaises(OverlongError):   # without its window the one bucket refuses longer decodes
            self._engine(single_bucket=True).decode(latents, voice="scylla", language="en_us", decoder="bigvgan")

    def test_bad_windows_and_the_environment(self):
        with self.assertRaisesRegex(ValueError, "leave frames to keep"):
            self._engine(dict(frames=8, context=4))
        with mock.patch.dict(os.environ, {"SCYLLASBAND_DECODER_WINDOW": "bigvgan=8:3"}):
            self.assertEqual(self._engine().decoder_windows, {"bigvgan": (8, 3)})
        with mock.patch.dict(os.environ, {"SCYLLASBAND_DECODER_WINDOW": "bigvgan=0"}):
            self.assertEqual(self._engine(dict(frames=8, context=2)).decoder_windows, {})
        with mock.patch.dict(os.environ, {"SCYLLASBAND_DECODER_WINDOW": "vocos=8"}), self.assertRaisesRegex(ValueError, "whole frames"):
            self._engine()


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
        chains = [dict(text="ab.", voice="scylla"), dict(text="ab.", voice="scylla", delivery="intensity=1")]
        self.assertEqual([text for _, text, _ in self.targets(paragraphs)], ["ab. ab."])
        self.assertEqual([text for _, text, _ in self.targets(records)], ["ab. ab."])
        self.assertEqual([text for _, text, _ in self.targets(chains)], ["ab.", "ab."])

    def test_a_long_enough_paragraph_ends_its_target(self):
        self.assertEqual([text for _, text, _ in self.targets(records_from_text("ab ab. ab ab.\n\nab ab.", voice="scylla",
                                                                                 language=None))],
                         ["ab ab.", "ab ab.", "ab ab."])


# --- fakes --------------------------------------------------------------------------------------------------------------
class _FakeGraphs:
    """Graph stand-ins with the bundle's input names. The vocoder returns frames * 512 - 256 samples, like the real one;
    the BigVGAN decoder of a v4 bundle returns frames * 512."""

    def __init__(self, frames_per_phone: float):
        self.frames_per_phone, self.calls = frames_per_phone, []
        self.bigvgan_by_frame = False   # BigVGAN samples = their frame's first latent channel (0 where masked)

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
                if name == "pitch_context_encoder":
                    return np.zeros((1, 384), np.float32)
                if name.startswith("pitch_estimator"):
                    return -feeds["noise"]
                if name.startswith("vector_estimator"):
                    return -feeds["noise"]
                frames = feeds["latents"].shape[-1]
                if name.startswith("vocoder_bigvgan"):
                    if graphs.bigvgan_by_frame:
                        values = np.where(feeds["latent_mask"][0], feeds["latents"][0, 0], 0.0).astype(np.float32)
                        return np.repeat(values, 512)[None]
                    return np.ones((1, frames * 512), np.float32) * 0.2
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


def _fake_bundle(root: Path, *, apple: bool = False, controls: dict | None = None, v4: bool = False) -> Path:
    """``apple``: also the optional fused-flow graphs and narrower G2P inputs of the Core ML and Core AI bundles.
    ``controls``: manifest controls to replace. ``v4``: the singing contract (the flow takes delivery_mode_id, delivery
    modes and decoders are declared, every bucket has a BigVGAN decoder)."""
    bundle = root / "bundle"
    (bundle / "assets/g2p").mkdir(parents=True)
    (bundle / "onnx").mkdir()
    flow_inputs = list(VECTOR_INPUTS_V4 if v4 else VECTOR_INPUTS)
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
                         ("pitch_context_encoder", list(CONTEXT_INPUTS)), ("vector_estimator_8", flow_inputs),
                         ("pitch_estimator_8", list(PITCH_INPUTS)), ("vocoder_8", list(VOCODER_INPUTS)),
                         ("vector_estimator", flow_inputs), ("pitch_estimator", list(PITCH_INPUTS)), ("vocoder", list(VOCODER_INPUTS)),
                         *((("vocoder_bigvgan_8", list(VOCODER_INPUTS)), ("vocoder_bigvgan", list(VOCODER_INPUTS))) if v4 else ())):
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
                                  delivery=dict(z_scale=5.0, z_limit=2.5, min_spoken_phones=4, ranges=dict(DEFAULT_RANGES),
                                                defaults=dict(scylla=dict(intensity=0.0, pace=0.0, pitch=0.0))),
                                  pace_pauses=dict(phone="<sil>", per_sd=-0.13),
                                  pitch=dict(channels=2, context_encoder="pitch_context_encoder", sampler=dict(default="euler"),
                                             steps=dict(default=2), center=10.0, scale=6.0, melody_center=16.65, melody_scale=24.0,
                                             voiced_threshold=0.5, knee=2.0, edges=dict(scylla=[5.0, 25.0])),
                                  target_buckets=dict(buckets=[dict(latent_frames=8, vector_estimator="vector_estimator_8", vocoder="vocoder_8",
                                                                    pitch_estimator="pitch_estimator_8"),
                                                               dict(latent_frames=16, vector_estimator="vector_estimator", vocoder="vocoder",
                                                                    pitch_estimator="pitch_estimator")])))
    if apple:
        fused_inputs = [n for n in flow_inputs if n != "time"]
        for name, inputs in (("vector_flow_8", fused_inputs), ("vector_flow", fused_inputs), ("g2p_16", ["text"]), ("g2p_32", ["text"])):
            (bundle / f"onnx/{name}.onnx").write_bytes(b"")
            components[name] = dict(artifacts=dict(onnx=dict(path=f"onnx/{name}.onnx", format="onnx")), inputs=inputs, outputs=["out"])
        manifest["controls"]["fused_flow"] = dict(sampler="heun", steps=8)
        manifest["controls"]["g2p_buckets"] = [dict(text_tokens=16, component="g2p_16"), dict(text_tokens=32, component="g2p_32")]
        for entry, flow in zip(manifest["controls"]["target_buckets"]["buckets"], ("vector_flow_8", "vector_flow")):
            entry["vector_flow"] = flow
    if v4:
        manifest["controls"]["graph_input_contract"] = GRAPH_CONTRACT_V4
        manifest["controls"]["delivery_modes"] = json.loads(json.dumps(V4_DELIVERY_MODES))
        manifest["controls"]["decoders"] = json.loads(json.dumps(V4_DECODERS))
        for entry, decoder in zip(manifest["controls"]["target_buckets"]["buckets"], ("vocoder_bigvgan_8", "vocoder_bigvgan")):
            entry["vocoder_bigvgan"] = decoder
    manifest["controls"].update(controls or {})
    write("manifest.json", manifest)
    return bundle


def _fake_engine(frames_per_phone: float = 1.0, *, apple: bool = False, v4: bool = False, **controls) -> Engine:
    tmp = tempfile.TemporaryDirectory()
    bundle = _fake_bundle(Path(tmp.name), apple=apple, controls=controls, v4=v4)
    graphs = _FakeGraphs(frames_per_phone)
    engine = Engine(bundle, session_factory=graphs.factory)
    engine._fake, engine._tmp = graphs, tmp
    return engine


if __name__ == "__main__":
    unittest.main()
