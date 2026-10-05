"""First-run setup decisions (no packages are installed by these tests)."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scyllasband.bootstrap import plan_setup, wants_onnx

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


if __name__ == "__main__":
    unittest.main()
