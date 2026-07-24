from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest

from scyllasband.cli import (
    _default_bundle_path,
    _normalize_requested_bundle_subdirs,
    _parse_cli_args,
    build_parser,
)
from scyllasband.native import SCYLLASBAND_BACKEND_ONNX, NativeScyllasBandRuntime, _backend_id
from scyllasband.runtime import ScyllasBandRuntime, _validate_backends


class BackendDefaultsTest(unittest.TestCase):
    def parse_speak(self, *arguments: str):
        return _parse_cli_args(build_parser(), ["speak", *arguments])

    def test_cli_defaults_to_onnx(self) -> None:
        args = self.parse_speak("--voice", "scylla", "Hello.")
        self.assertEqual(args.backend, "onnx")
        self.assertEqual(_default_bundle_path(args.backend).name, "onnx")

    def test_download_accepts_int8_and_preserves_existing_bundle_groups(self) -> None:
        args = _parse_cli_args(
            build_parser(),
            ["download", "--runtime-bundles", "onnx-int8"],
        )
        self.assertEqual(args.runtime_bundles, "onnx-int8")
        self.assertEqual(
            _normalize_requested_bundle_subdirs(("both",)),
            ("onnx", "litert"),
        )
        self.assertEqual(
            _normalize_requested_bundle_subdirs(("all",)),
            ("onnx", "onnx-int8", "litert"),
        )

    def test_android_sample_defaults_to_int8_assets(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        gradle = (repository_root / "examples/android/app/build.gradle.kts").read_text(
            encoding="utf-8"
        )
        installer = (
            repository_root
            / "examples/android/scyllasband-android/src/main/java/org/scyllasband/android/ScyllasBandAssetInstaller.kt"
        ).read_text(encoding="utf-8")
        self.assertIn("scyllasband/models/onnx-int8", gradle)
        self.assertIn("into(\"scyllasband/onnx-int8\")", gradle)
        self.assertIn("DEFAULT_ASSET_ROOT = \"scyllasband/onnx-int8\"", installer)

    def test_litert_requires_explicit_selection(self) -> None:
        args = self.parse_speak(
            "--voice",
            "scylla",
            "--backend",
            "litert",
            "Hello.",
        )
        self.assertEqual(args.backend, "litert")
        self.assertEqual(_default_bundle_path(args.backend).name, "litert")

    def test_runtime_auto_is_an_onnx_compatibility_alias(self) -> None:
        runtime = ScyllasBandRuntime(
            ".",
            SimpleNamespace(preferred_backends=("litert",)),
            backends=["auto"],
        )
        runtime._has_backend_artifacts = lambda backend: backend == "onnx"
        self.assertEqual(runtime._select_backend(), "onnx")
        self.assertEqual(_validate_backends([]), ["onnx"])

    def test_native_binding_defaults_to_onnx(self) -> None:
        parameters = inspect.signature(NativeScyllasBandRuntime).parameters
        backend_default = parameters["backend"].default
        self.assertEqual(backend_default, "onnx")
        self.assertEqual(parameters["max_cached_target_buckets"].default, 0)
        self.assertEqual(_backend_id("onnx"), SCYLLASBAND_BACKEND_ONNX)


if __name__ == "__main__":
    unittest.main()
