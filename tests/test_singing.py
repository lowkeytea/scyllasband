"""Singing tests on the fake bundle: scores, the phrase plan, and what a sung phrase feeds the graphs."""

from __future__ import annotations

import unittest

import numpy as np

from scyllasband.engine import OverlongError
from scyllasband.singing import (HOP, SAMPLE_RATE, Note, ScoreError, frame_melody, midi_to_note, note_to_midi, parse_score, plan_song,
                                 register_shift, sing, singing_styles, split_phrases, voice_sung_center)
from test_runtime import V4_DELIVERY_MODES, _fake_engine


def _v4_engine(**singing):
    """A v4 fake engine whose manifest's delivery_modes.singing carries extra keys (guidance, voice_midi_center, modes)."""
    import json
    modes = json.loads(json.dumps(V4_DELIVERY_MODES))
    modes["singing"].update(singing)
    return _fake_engine(v4=True, delivery_modes=modes)

# The fake G2P maps letters to phones of the same letter (e -> o), so these words use the fake vocabulary's letters.
# 1200 bpm: a beat is 0.05 s, so the song fits the fake bundle's 16-frame buckets (0.34 s) once padded by 0.02 s.
SCORE = dict(text="hello bob wow", bpm=1200, style="pop",
             notes=[["hello", "C4", 2], ["-", "rest", 6], ["bob~1", "D4", 1], ["bob~2", "E4", 1], ["wow", "G4", 2]])
PAD = 0.02


class ScoreTest(unittest.TestCase):
    def test_notes_and_midi(self):
        self.assertEqual([note_to_midi(n) for n in ("C4", "D#4", "Bb3", "F#3", "rest", "-")], [60, 63, 58, 54, None, None])
        self.assertEqual([midi_to_note(m) for m in (60, 63, 58)], ["C4", "D#4", "A#3"])
        for bad in ("H4", "C", "4", "Cx4"):
            with self.assertRaisesRegex(ScoreError, "Unknown note"):
                note_to_midi(bad)

    def test_words_parts_rests_and_times(self):
        score = parse_score(SCORE)
        self.assertEqual((score.style, score.mode, score.bpm), ("pop", "sung", 1200.0))
        self.assertEqual([(w.text, w.parts) for w in score.words], [("hello", 1), ("bob", 2), ("wow", 1)])
        self.assertEqual([n.name for n in score.notes], ["C4", "rest", "D4", "E4", "G4"])
        np.testing.assert_allclose([(w.start, w.end) for w in score.words], [(0.0, 0.1), (0.4, 0.5), (0.5, 0.6)], atol=1e-9)
        self.assertAlmostEqual(score.seconds, 0.6)
        self.assertEqual(parse_score(dict(SCORE, style="ballad")).mode, "ballad")

    def test_syllable_parts_join_and_whole_word_parts_do_not_repeat(self):
        syllables = parse_score(dict(bpm=100, notes=[["ba~1", "E4", 0.5], ["by~2", "D4", 0.5], ["home~1", "C4", 1], ["home~2", "D4", 1]]))
        self.assertEqual([w.text for w in syllables.words], ["baby", "home"])
        self.assertEqual(syllables.text, "baby home")
        repeated = parse_score(dict(text="row row row", bpm=100, notes=[["row", "C4", 1], ["row", "D4", 1], ["row", "E4", 1]]))
        self.assertEqual(len(repeated.words), 3)

    def test_errors(self):
        with self.assertRaisesRegex(ScoreError, "bpm"):
            parse_score(dict(notes=[["a", "C4", 1]]))
        with self.assertRaisesRegex(ScoreError, "non-empty list"):
            parse_score(dict(bpm=100, notes=[]))
        with self.assertRaisesRegex(ScoreError, "consecutive"):
            parse_score(dict(bpm=100, notes=[["bob~1", "C4", 1], ["bob~3", "D4", 1]]))
        with self.assertRaisesRegex(ScoreError, "consecutive"):
            parse_score(dict(bpm=100, notes=[["bob~1", "C4", 1], ["-", "rest", 1], ["bob~2", "D4", 1]]))
        with self.assertRaisesRegex(ScoreError, "does not match"):
            parse_score(dict(SCORE, text="hello wow"))
        with self.assertRaisesRegex(ScoreError, "positive"):
            parse_score(dict(bpm=100, notes=[["a", "C4", 0]]))
        with self.assertRaisesRegex(ScoreError, "Unknown style"):
            parse_score(dict(SCORE, style="opera"))
        with self.assertRaisesRegex(ScoreError, "no sung notes"):
            parse_score(dict(bpm=100, notes=[["-", "rest", 1]]))

    def test_register_shift_moves_the_mean_by_octaves_into_the_band(self):
        score = parse_score(SCORE)                         # mean of C4 D4 E4 G4 = 63.25
        self.assertEqual(register_shift(score, 58.35), 0)
        self.assertEqual(register_shift(score.transposed(12), 58.35), -12)
        self.assertEqual(register_shift(score.transposed(-24), 58.35), 24)
        self.assertEqual(register_shift(score, None), 0)
        self.assertEqual([n.name for n in score.transposed(-12).notes][:3], ["C3", "rest", "D3"])

    def test_voice_centre_comes_from_the_pitch_edges(self):
        engine = _fake_engine(v4=True)                     # edges [5, 25] on the melody scale whose centre 16.65 is C4
        self.assertAlmostEqual(voice_sung_center(engine, "scylla"), 15 - 16.65 + 60)
        self.assertIsNone(voice_sung_center(engine, "nobody"))

    def test_phrases_split_at_rests_and_pack_up_to_the_limit(self):
        score = parse_score(SCORE)
        phrases = split_phrases(score, max_seconds=0.25, pad=PAD)
        self.assertEqual([p.text for p in phrases], ["hello", "bob wow"])
        self.assertAlmostEqual(phrases[0].start, -PAD)
        self.assertAlmostEqual(phrases[1].start, 0.4 - PAD)
        self.assertEqual([n.name for n in phrases[1].notes], ["D4", "E4", "G4", "rest"])
        self.assertAlmostEqual(phrases[1].notes[0].start, PAD)
        self.assertEqual([p.text for p in split_phrases(score, max_seconds=10, pad=PAD)], ["hello bob wow"])
        long_run = parse_score(dict(bpm=60, notes=[["a", "C4", 1], ["b", "C4", 1], ["c", "C4", 1], ["d", "C4", 1]]))
        self.assertEqual([p.text for p in split_phrases(long_run, max_seconds=2.5, pad=PAD)], ["a b", "c d"])

    def test_frame_melody_layout(self):
        notes = [Note("a", 62, 0.02, 0.12), Note("-", None, 0.12, 0.14)]
        melody = frame_melody(notes, 9)
        self.assertEqual(melody.shape, (9, 3))
        self.assertEqual(melody[:, 2].tolist(), [0, 1, 1, 1, 1, 1, 0, 0, 0])   # frame centres 0.032 .. 0.117 s fall inside the note
        np.testing.assert_allclose(melody[1:6, 0], 2 / 24)
        centre = (1 * HOP + HOP // 2) / SAMPLE_RATE
        self.assertAlmostEqual(float(melody[1, 1]), (centre - 0.02) / 0.1, places=6)
        self.assertTrue((np.diff(melody[1:6, 1]) > 0).all())
        self.assertTrue((melody[[0, 6, 7, 8]] == 0).all())


class PlanTest(unittest.TestCase):
    def test_phone_frames_sum_to_the_phrase_frames(self):
        engine = _fake_engine(v4=True)
        plan = plan_song(SCORE, engine, "scylla", "en_us", pad=PAD)
        self.assertEqual((plan.mode, plan.mode_id, plan.register_shift), ("sung", 1, 0))
        self.assertEqual([p.phrase.text for p in plan.phrases], ["hello", "bob wow"])
        first = plan.phrases[0]
        self.assertEqual(first.sentence.phones, ["<sil>", "h", "o", "l", "l", "o", "<sil>"])
        self.assertEqual(first.frames, 9)                                  # round(0.14 s * 24000 / 512) + 2
        self.assertEqual(sum(first.durations), first.frames)
        self.assertEqual(first.durations, [1, 1, 1, 1, 1, 1, 3])            # 5 frames for 5 phones; the rest is the release
        second = plan.phrases[1]
        self.assertEqual(second.sentence.word_starts, [5])
        self.assertEqual(sum(second.durations), second.frames)
        self.assertEqual(second.durations[0], 1)                             # the lead-in
        self.assertEqual(second.durations[4], 0)                             # bob (2 frames for 3 phones) runs into the next word
        for item in plan.phrases:
            self.assertEqual(item.melody.shape, (item.frames, 3))
            self.assertTrue((item.melody[0] == 0).all() and (item.melody[-1] == 0).all())
        self.assertEqual(first.melody[:, 2].tolist(), [0, 1, 1, 1, 1, 1, 0, 0, 0])
        np.testing.assert_allclose(first.melody[1:6, 0], 0.0)              # C4 is the centre
        sung = second.melody[second.melody[:, 2] > 0]
        self.assertEqual(sorted(set(np.round(sung[:, 0] * 24).tolist())), [2.0, 4.0, 7.0])   # D4, E4, G4

    def test_the_longest_phrase_follows_the_bundle(self):
        engine = _fake_engine(v4=True)
        too_long = dict(bpm=1200, notes=[["hello", "C4", 20]])              # one note of 1 s: no rest to cut at
        with self.assertRaises(OverlongError):
            plan_song(too_long, engine, "scylla", "en_us", pad=PAD)
        plan = plan_song(dict(bpm=1200, notes=[["hello", "C4", 2], ["bob", "D4", 2], ["wow", "E4", 2], ["bob", "G4", 2]]),
                         engine, "scylla", "en_us", pad=PAD)
        self.assertGreater(len(plan.phrases), 1)
        self.assertTrue(all(p.frames <= engine.max_frames for p in plan.phrases))

    def test_singing_needs_a_v4_bundle(self):
        with self.assertRaisesRegex(ValueError, "v4"):
            plan_song(SCORE, _fake_engine(), "scylla", "en_us", pad=PAD)
        with self.assertRaisesRegex(ValueError, "v4"):
            sing(_fake_engine(), SCORE, "scylla", "en_us", pad=PAD)

    def test_styles_follow_the_manifest_singing_modes(self):
        engine = _fake_engine(v4=True)                                       # singing.modes: sung, ballad (rap has an id)
        self.assertEqual(singing_styles(engine), ["pop", "sung", "ballad"])
        self.assertEqual(plan_song(dict(SCORE, style="ballad"), engine, "scylla", "en_us", pad=PAD).mode_id, 3)
        with self.assertRaisesRegex(ScoreError, r"Style 'rap' \(rap\) is not a singing style.*pop, sung, ballad"):
            plan_song(dict(SCORE, style="rap"), engine, "scylla", "en_us", pad=PAD)
        with self.assertRaisesRegex(ScoreError, "choose one of pop, sung$"):
            plan_song(dict(SCORE, style="ballad"), _v4_engine(modes=["sung"]), "scylla", "en_us", pad=PAD)
        self.assertEqual(singing_styles(_v4_engine(modes=[])), ["pop", "sung", "ballad", "rap"])   # nothing listed: every mode with an id

    def test_manifest_voice_midi_center_wins_over_the_pitch_edges(self):
        engine = _v4_engine(voice_midi_center=dict(scylla=72.0))
        self.assertEqual(voice_sung_center(engine, "scylla"), 72.0)
        self.assertEqual(plan_song(SCORE, engine, "scylla", "en_us", pad=PAD).register_shift, 12)   # mean 63.25 -> 75.25 in [69, 77]
        self.assertEqual(plan_song(SCORE, _fake_engine(v4=True), "scylla", "en_us", pad=PAD).register_shift, 0)

    def test_transpose_moves_the_score_by_semitones_before_the_register(self):
        engine = _v4_engine(voice_midi_center=dict(scylla=72.0))
        plain = plan_song(SCORE, engine, "scylla", "en_us", pad=PAD)
        plan = plan_song(SCORE, engine, "scylla", "en_us", pad=PAD, transpose=-5)
        self.assertEqual((plan.transpose, plan.register_shift), (-5, 12))   # mean 63.25 - 5 + 12 = 70.25, still in [69, 77]
        self.assertEqual(plan.score.pitched, [m - 5 for m in plain.score.pitched])
        self.assertEqual(plan.to_dict()["transpose"], -5)
        self.assertEqual(plan_song(SCORE, engine, "scylla", "en_us", pad=PAD, transpose=-5, register=False).score.pitched,
                         [m - 5 for m in parse_score(SCORE).pitched])


class SingTest(unittest.TestCase):
    def test_a_sung_phrase_feeds_the_mode_and_melody_and_decodes_with_bigvgan(self):
        for style, mode_id in (("pop", 1), ("ballad", 3)):
            engine = _fake_engine(v4=True)
            plan = plan_song(dict(SCORE, style=style), engine, "scylla", "en_us", pad=PAD)
            engine._fake.calls.clear()
            result = sing(engine, dict(SCORE, style=style), "scylla", "en_us", seed=5, pad=PAD)
            calls = engine._fake.calls
            names = [c["name"] for c in calls]
            self.assertFalse(any(n.startswith("pitch_") or n == "duration_predictor" for n in names))
            flows = [c for c in calls if c["name"].startswith("vector_estimator")]
            self.assertEqual(len(flows), 2 * 8 * len(plan.phrases))        # the manifest's singing sampler and steps: Heun, 8
            self.assertTrue(all(c["delivery_mode_id"].tolist() == [mode_id] for c in flows))
            self.assertTrue(all(not c["delivery_present"].any() for c in flows))
            first = flows[0]
            np.testing.assert_array_equal(first["expanded_melody"][0, :plan.phrases[0].frames], plan.phrases[0].melody)
            self.assertTrue((first["expanded_melody"][0, plan.phrases[0].frames:] == 0).all())
            self.assertEqual([n for n in names if n.startswith("vocoder")], ["vocoder_bigvgan"] * len(plan.phrases))
            self.assertEqual(result.sample_rate, 24000)
            self.assertEqual(result.audio.shape, (int(np.ceil(0.62 * 24000)),))   # the song plus the last phrase's release
            self.assertEqual(result.metadata["decoder"], "bigvgan")
            self.assertEqual((result.metadata["mode"], result.metadata["mode_id"], result.metadata["steps"], result.metadata["sampler"]),
                             (plan.mode, mode_id, 8, "heun"))
            self.assertEqual([p["seed"] for p in result.metadata["phrases"]], [5, 6])
            self.assertEqual([p["text"] for p in result.metadata["phrases"]], ["hello", "bob wow"])

    def test_phrases_land_on_the_timeline_with_silence_between(self):
        engine = _fake_engine(v4=True)
        result = sing(engine, SCORE, "scylla", "en_us", seed=0, pad=PAD)
        audio = result.audio
        self.assertTrue((audio[int(0.2 * 24000):int(0.3 * 24000)] == 0).all())     # the rest between the phrases
        self.assertTrue((audio[:int(0.05 * 24000)] != 0).any())                     # hello
        self.assertTrue((audio[int(0.45 * 24000):int(0.55 * 24000)] != 0).any())    # bob wow

    def test_requested_decoder_steps_and_sampler_win(self):
        engine = _fake_engine(v4=True)
        result = sing(engine, SCORE, "scylla", "en_us", seed=0, steps=2, sampler="euler", decoder="vocos", pad=PAD)
        names = [c["name"] for c in engine._fake.calls]
        self.assertEqual(names.count("vector_estimator"), 2 * 2)             # Euler, 2 steps, 2 phrases
        self.assertEqual([n for n in names if n.startswith("vocoder")], ["vocoder"] * 2)
        self.assertEqual((result.metadata["steps"], result.metadata["sampler"], result.metadata["decoder"]), (2, "euler", "vocos"))
        with self.assertRaisesRegex(ValueError, "no 'nothing' decoder"):
            sing(engine, SCORE, "scylla", "en_us", decoder="nothing", pad=PAD)

    def test_manifest_guidance_is_the_default_and_a_request_overrides_it(self):
        engine = _v4_engine(guidance=2.0)
        result = sing(engine, SCORE, "scylla", "en_us", seed=0, steps=1, sampler="euler", pad=PAD)
        flows = [c for c in engine._fake.calls if c["name"].startswith("vector_estimator")]
        self.assertEqual(len(flows), 2 * 2)                                   # on/off pairs for 2 phrases at 1 Euler step
        self.assertEqual([c["delivery_mode_id"].tolist()[0] for c in flows], [1, 0, 1, 0])
        self.assertTrue((flows[1]["expanded_melody"] == 0).all())
        self.assertEqual(result.metadata["guidance"], 2.0)
        engine._fake.calls.clear()
        result = sing(engine, SCORE, "scylla", "en_us", seed=0, steps=1, sampler="euler", guidance=1.0, pad=PAD)
        self.assertEqual(sum(c["name"].startswith("vector_estimator") for c in engine._fake.calls), 2)
        self.assertEqual(result.metadata["guidance"], 1.0)
        plain = _fake_engine(v4=True)                                         # no guidance in the manifest: off
        self.assertEqual(sing(plain, SCORE, "scylla", "en_us", seed=0, steps=1, sampler="euler", pad=PAD).metadata["guidance"], 1.0)
        self.assertEqual(sum(c["name"].startswith("vector_estimator") for c in plain._fake.calls), 2)

    def test_runtime_sing_resolves_the_language_and_rejects_v3(self):
        from pathlib import Path
        from scyllasband.runtime import ScyllasBandRuntime
        engine = _fake_engine(v4=True)
        runtime = ScyllasBandRuntime(engine, bundle_dir=Path(engine.bundle_dir), backend="onnx")
        self.assertTrue(runtime.supports_singing)
        short = dict(text="hello", bpm=1200, style="pop", notes=[["hello", "C4", 1]])   # fits 16 frames with the default pads
        result = runtime.sing(short, "scylla", seed=1, decoder="vocos")
        self.assertEqual((result.metadata["language"], result.metadata["voice_id"], result.metadata["backend"]), ("en_us", "scylla", "onnx"))
        self.assertEqual(runtime.runtime_status()["decoders"], ["bigvgan", "vocos"])
        spoken = _fake_engine()
        v3 = ScyllasBandRuntime(spoken, bundle_dir=Path(spoken.bundle_dir), backend="onnx")
        self.assertFalse(v3.supports_singing)
        with self.assertRaisesRegex(ValueError, "v4"):
            v3.sing(SCORE, "scylla")


if __name__ == "__main__":
    unittest.main()
