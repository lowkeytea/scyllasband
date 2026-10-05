"""First-run setup decisions (no packages are installed by these tests)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from unittest import mock

from scyllasband import download
from scyllasband.bootstrap import plan_setup, wanted_backend, wants_onnx

MISSING = {"ai_edge_litert": "ai-edge-litert"}


def plan(**overrides):
    state = dict(checkout=True, venv_ready=False, in_venv=False, in_repo_venv=False, python_ok=True, previous=None)
    state.update(overrides)
    return plan_setup(MISSING, **state)


class PlanSetupTest(unittest.TestCase):
    def test_nothing_missing(self):
        self.assertEqual(plan_setup({}, checkout=True, venv_ready=False, in_venv=False, in_repo_venv=False, python_ok=True), "ok")

    def test_fresh_checkout_creates_the_venv(self):
        self.assertEqual(plan(), "create-venv")

    def test_existing_venv_is_used_without_asking(self):
        self.assertEqual(plan(venv_ready=True), "use-venv")
        self.assertEqual(plan(venv_ready=True, in_venv=True), "use-venv")   # another environment is active

    def test_inside_a_virtualenv_installs_there(self):
        self.assertEqual(plan(venv_ready=True, in_venv=True, in_repo_venv=True), "install-here")
        self.assertEqual(plan(in_venv=True), "install-here")

    def test_old_python_is_refused_before_creating_anything(self):
        self.assertEqual(plan(python_ok=False), "fail-python")

    def test_installed_copy_only_gets_advice(self):
        self.assertEqual(plan(checkout=False), "fail-installed")

    def test_never_loops(self):
        self.assertEqual(plan(in_venv=True, in_repo_venv=True, venv_ready=True, previous="installed"), "fail-loop")
        self.assertEqual(plan(venv_ready=True, previous="use-venv"), "fail-loop")
        self.assertEqual(plan(venv_ready=True, in_venv=True, in_repo_venv=True, previous="use-venv"), "install-here")


class WantsOnnxTest(unittest.TestCase):
    def test_backend_flag(self):
        self.assertTrue(wants_onnx(["speak", "--backend", "onnx", "hi"]))
        self.assertTrue(wants_onnx(["speak", "--backend=onnx"]))
        self.assertFalse(wants_onnx(["speak", "--backend", "litert"]))
        self.assertFalse(wants_onnx(["download", "--flavor", "onnx"]))

    def test_bundle_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "manifest.json").write_text(json.dumps(dict(preferred_backends=["onnx"])))
            self.assertTrue(wants_onnx(["speak", "--bundle", tmp]))
            self.assertFalse(wants_onnx(["speak", "--bundle", tmp, "--backend", "litert"]))


class WantedBackendTest(unittest.TestCase):
    def test_apple_bundles_name_their_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "manifest.json").write_text(json.dumps(dict(preferred_backends=["coreml"])))
            self.assertEqual(wanted_backend(["speak", "--bundle", tmp]), "coreml")
        self.assertEqual(wanted_backend(["speak", "--backend=coreai"]), "coreai")
        self.assertIsNone(wanted_backend(["download"]))

    def test_model_commands_use_the_installed_bundle_this_machine_picks(self):
        with tempfile.TemporaryDirectory() as tmp:
            for flavor in ("litert", "coreml"):
                Path(tmp, flavor).mkdir()
                Path(tmp, flavor, "manifest.json").write_text("{}")
            with mock.patch.object(download, "macos_major", return_value=26):
                self.assertEqual(wanted_backend(["speak", "--models-dir", tmp, "hi"]), "coreml")
            with mock.patch.object(download, "macos_major", return_value=None):
                self.assertEqual(wanted_backend(["speak", "--models-dir", tmp, "hi"]), "litert")


class ChooseFlavorTest(unittest.TestCase):
    def choose(self, major, sdk=None, answers=None):
        replies = iter(answers or [])
        with mock.patch.object(download, "macos_major", return_value=major), \
                mock.patch.object(download, "xcode_sdk_major", return_value=sdk), open(os.devnull, "w") as quiet:
            return download.choose_flavors(interactive=answers is not None, ask=lambda _: next(replies), out=quiet)

    def test_macs_run_core_ml_and_other_machines_litert(self):
        for major in (15, 26, 27):
            with mock.patch.object(download, "macos_major", return_value=major):
                self.assertEqual(download.recommended_flavor(), "coreml")
        with mock.patch.object(download, "macos_major", return_value=None):
            self.assertEqual(download.recommended_flavor(), "litert")
        self.assertEqual(self.choose(None), ["litert"])

    def test_core_ai_is_offered_only_to_setups_that_build_for_os_27(self):
        self.assertEqual(self.choose(26, sdk=26, answers=[]), ["coreml"])            # not asked
        self.assertEqual(self.choose(26, sdk=27, answers=["y"]), ["coreml", "coreai"])   # Xcode with the iOS 27 SDK
        self.assertEqual(self.choose(27, sdk=None, answers=[""]), ["coreml"])        # macOS 27, default no
        self.assertEqual(self.choose(27, answers=["maybe", "yes"]), ["coreml", "coreai"])
        self.assertEqual(self.choose(27), ["coreml"])                                # --yes or no terminal

    def test_a_mac_with_both_bundles_uses_core_ml(self):
        with tempfile.TemporaryDirectory() as tmp:
            for flavor in ("coreai", "coreml"):
                Path(tmp, flavor).mkdir()
                Path(tmp, flavor, "manifest.json").write_text("{}")
            with mock.patch.object(download, "macos_major", return_value=27):
                self.assertEqual(download.default_installed_flavor(tmp), "coreml")


if __name__ == "__main__":
    unittest.main()
