"""The native binding (scyllasband.native): engine selection without a library, request checks, and, with a built library
and a bundle (SCYLLASBAND_TEST_NATIVE_LIB, SCYLLASBAND_TEST_BUNDLE: a LiteRT build and a LiteRT v4 bundle), the runtime
itself: plans as the Python engine's, streams that cancel cleanly, and errors raised after the call."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from scyllasband import cli, native
from scyllasband.native import NativeUnavailable, check_seed, library_backend, open_runtime
from scyllasband.runtime import ScyllasBandRuntime
from test_runtime import _fake_bundle


class _Unavailable:
    """Stands in for NativeRuntime when no library is found."""

    def __init__(self, *args, **kwargs):
        raise NativeUnavailable("no native litert library (none found)")


class EngineChoiceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.bundle = _fake_bundle(Path(self.tmp.name))
        manifest = json.loads((self.bundle / "manifest.json").read_text())
        manifest["preferred_backends"] = ["litert"]
        (self.bundle / "manifest.json").write_text(json.dumps(manifest))
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()
        os.environ.pop(native.ENGINE_ENV, None)

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def open(self, **kwargs):
        log = io.StringIO()
        runtime = open_runtime(self.bundle, validate=False, log=log, **kwargs)
        return runtime, log.getvalue()

    def test_python_engine_on_request(self):
        runtime, log = self.open(engine="python")
        self.assertIsInstance(runtime, ScyllasBandRuntime)
        self.assertEqual(log, "")

    def test_without_a_library_auto_falls_back_with_one_notice(self):
        with mock.patch.object(native, "NativeRuntime", _Unavailable):
            runtime, log = self.open()
        self.assertIsInstance(runtime, ScyllasBandRuntime)
        self.assertEqual(log.count("\n"), 1)
        self.assertIn("Python engine", log)
        self.assertIn("build-native --backend litert", log)

    def test_native_on_request_without_a_library_is_an_error(self):
        with mock.patch.object(native, "NativeRuntime", _Unavailable):
            with self.assertRaisesRegex(NativeUnavailable, "build-native"):
                self.open(engine="native")

    def test_environment_chooses_the_engine(self):
        os.environ[native.ENGINE_ENV] = "python"
        with mock.patch.object(native, "NativeRuntime", side_effect=AssertionError("native was tried")):
            runtime, _ = self.open()
        self.assertIsInstance(runtime, ScyllasBandRuntime)
        os.environ[native.ENGINE_ENV] = "quantum"
        with self.assertRaisesRegex(ValueError, "Unknown engine"):
            self.open()

    def test_auto_uses_native_when_it_opens(self):
        sentinel = object()
        with mock.patch.object(native, "NativeRuntime", return_value=sentinel) as made:
            runtime, log = self.open()
        self.assertIs(runtime, sentinel)
        self.assertEqual(made.call_args.kwargs["backend"], "litert")
        self.assertEqual(log, "")

    def test_onnx_stays_on_the_python_engine_unless_native_is_asked_for(self):
        with mock.patch.object(native, "NativeRuntime", side_effect=AssertionError("native was tried")):
            runtime, log = self.open(backend="onnx")
        self.assertIsInstance(runtime, ScyllasBandRuntime)
        self.assertEqual(log, "")
        with mock.patch.object(native, "NativeRuntime", return_value="native") as made:
            runtime, _ = self.open(backend="onnx", engine="native")
        self.assertEqual(runtime, "native")

    def test_cli_runtime_is_the_one_selection_point(self):
        args = cli.build_parser().parse_args(["speak", "hello", "--bundle", str(self.bundle), "--engine", "python", "--no-validate-bundle"])
        with mock.patch("scyllasband.native.open_runtime", return_value="runtime") as opened:
            self.assertEqual(cli._runtime(args), "runtime")
        self.assertEqual(opened.call_args.kwargs["engine"], "python")
        self.assertEqual(opened.call_args.args[0], self.bundle)


class RequestCheckTest(unittest.TestCase):
    def test_seeds(self):
        self.assertEqual(check_seed(None), (0, 0))
        self.assertEqual(check_seed(7), (7, 1))
        self.assertEqual(check_seed(2 ** 64 - 1), (2 ** 64 - 1, 1))
        for bad in (-1, 2 ** 64, 1.5, True, "7"):
            with self.assertRaises(ValueError):
                check_seed(bad)

    def test_nul_characters_are_refused(self):
        with self.assertRaisesRegex(ValueError, "NUL"):
            native._json_bytes([dict(text="a\x00b", voice="scylla")], "records")
        with self.assertRaisesRegex(ValueError, "NUL"):
            native._text("sc\x00ylla", "voice")
        self.assertEqual(json.loads(native._json_bytes([dict(text="é", pace=1)], "records")), [dict(text="é", pace=1)])

    def test_library_backends(self):
        self.assertEqual([library_backend(f) for f in ("litert", "coreml", "coreai", "onnx", "webgpu")], ["litert", "apple", "apple", "onnx", "onnx"])
        with self.assertRaises(ValueError):
            library_backend("tensorrt")

    def test_a_file_that_is_not_a_library_is_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / native.library_name()
            fake.write_text("not a library")
            with self.assertRaises(NativeUnavailable):
                native.load_library("litert", fake)

    def test_candidates_start_with_the_environment(self):
        with mock.patch.dict(os.environ, {native.LIBRARY_ENV: "/somewhere/lib.dylib"}):
            candidates = native.library_candidates("litert")
        self.assertEqual(candidates[0], Path("/somewhere/lib.dylib"))
        self.assertEqual(candidates[1], native.PACKAGE_DIR / "litert" / native.library_name())


LIBRARY, BUNDLE = os.environ.get("SCYLLASBAND_TEST_NATIVE_LIB"), os.environ.get("SCYLLASBAND_TEST_BUNDLE")


@unittest.skipUnless(LIBRARY and BUNDLE, "set SCYLLASBAND_TEST_NATIVE_LIB and SCYLLASBAND_TEST_BUNDLE (a LiteRT library and bundle)")
class NativeRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = native.NativeRuntime(BUNDLE, library=native.load_library("litert", LIBRARY))

    @classmethod
    def tearDownClass(cls):
        cls.runtime.close()

    def test_plans_match_the_python_engine(self):
        records = [dict(text="Hello there. It has been a while.", voice="scylla"), dict(text="", voice="scylla", pace=1),
                   dict(text="It has.", voice="ink", language="en_gb", delivery={"intensity": -1})]
        python = ScyllasBandRuntime.from_bundle(BUNDLE, backend="litert")
        self.assertEqual(self.runtime.plan_records(records).to_dict(), python.plan_records(records).to_dict())
        with self.assertRaisesRegex(ValueError, "Unknown voice 'nobody'"):
            self.runtime.plan_records([dict(text="Hi.", voice="nobody")])

    def test_render_and_metadata(self):
        result = self.runtime.render_records([dict(text="One line.", voice="scylla")], seed=3, steps=1)
        self.assertGreater(result.audio.size, 0)
        self.assertEqual((result.metadata["engine"], result.metadata["noise"]), ("native", "xoshiro256ss"))
        again = self.runtime.render_records([dict(text="One line.", voice="scylla")], seed=3, steps=1)
        self.assertTrue((result.audio == again.audio).all())

    def test_closing_a_stream_early_cancels_it(self):
        threads = threading.active_count()
        stream = self.runtime.synthesize_stream([dict(text="One. Two. Three. Four. Five. Six.", voice="scylla")], seed=1, steps=1)
        for event in stream:
            if event.type == "audio_chunk":
                break
        stream.close()
        self.assertEqual(threading.active_count(), threads)
        self.assertGreater(self.runtime.render_records([dict(text="Still here.", voice="scylla")], steps=1).audio.size, 0)

    def test_an_exception_in_the_consumer_cancels_the_call(self):
        threads = threading.active_count()
        with self.assertRaises(KeyboardInterrupt):
            for event in self.runtime.synthesize_stream([dict(text="One. Two. Three. Four.", voice="scylla")], seed=1, steps=1):
                if event.type == "chunk_started":
                    raise KeyboardInterrupt
        self.assertEqual(threading.active_count(), threads)
        self.assertGreater(self.runtime.render_records([dict(text="Still here.", voice="scylla")], steps=1).audio.size, 0)

    def test_an_exception_in_the_callback_is_raised_after_the_call(self):
        original, calls = native.copy_event, []

        def failing(event):
            calls.append(event.type)
            if len(calls) == 3:
                raise RuntimeError("copy failed")
            return original(event)

        with mock.patch.object(native, "copy_event", failing):
            with self.assertRaisesRegex(RuntimeError, "copy failed"):
                self.runtime.render_records([dict(text="One. Two. Three. Four.", voice="scylla")], steps=1)
        self.assertGreater(self.runtime.render_records([dict(text="Still here.", voice="scylla")], steps=1).audio.size, 0)

    def test_scores(self):
        score = dict(bpm=100, notes=[["la", "C4", 1], ["la", "H9", 1]])
        with self.assertRaises(native.NativeInvalidArgument) as caught:
            self.runtime.score_parse(score)
        self.assertEqual(caught.exception.detail["note_index"], 1)
        good = dict(bpm=120, notes=[["la", "C5", 1], ["-", "rest", 1], ["la", "D5", 1]])
        parsed = self.runtime.score_parse(good, transpose=2)
        self.assertEqual([n["midi"] for n in parsed["notes"]], [74, None, 76])
        if not self.runtime.supports_singing:   # a spoken-only (v3) bundle parses scores but refuses to fit or sing them
            with self.assertRaisesRegex(native.NativeInvalidArgument, "v4"):
                self.runtime.sing_fit(good, "scylla")
            return
        fit = self.runtime.sing_fit(good, "gwen")
        self.assertEqual(fit["register_shift"], -12)
        preview = self.runtime.sing_preview(good, "gwen")
        self.assertEqual((preview.sample_rate, preview.metadata["register_shift"]), (24000, -12))
        self.assertEqual(preview.audio.size, int(1.5 * 24000))


if __name__ == "__main__":
    unittest.main()
