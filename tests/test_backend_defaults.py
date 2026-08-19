from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

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

    def test_cli_defaults_to_auto_with_platform_aware_bundle(self) -> None:
        args = self.parse_speak("--voice", "scylla", "Hello.")
        self.assertEqual(args.backend, "auto")
        with mock.patch("scyllasband.download.coreai_host_supported", return_value=False):
            self.assertIn(_default_bundle_path(args.backend).name, ("onnx-int8", "onnx"))
        self.assertIn(_default_bundle_path("onnx").name, ("onnx-int8", "onnx"))

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
            ("onnx", "onnx-int8", "litert", "coreai", "coreai-fp32"),
        )

    def test_download_default_group_is_platform_aware_and_int8_first(self) -> None:
        with mock.patch("scyllasband.download.coreai_host_supported", return_value=True):
            self.assertEqual(
                _normalize_requested_bundle_subdirs(("default",)),
                ("coreai", "onnx-int8"),
            )
        with mock.patch("scyllasband.download.coreai_host_supported", return_value=False):
            self.assertEqual(
                _normalize_requested_bundle_subdirs(("default",)),
                ("onnx-int8",),
            )
        # fp32 onnx stays available but only when explicitly requested.
        self.assertEqual(_normalize_requested_bundle_subdirs(("onnx",)), ("onnx",))

    def test_android_sample_defaults_to_int8_assets(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        gradle = (repository_root / "examples/android/app/build.gradle.kts").read_text(
            encoding="utf-8"
        )
        installer = (
            repository_root
            / "examples/android/scyllasband-android/src/main/java/org/scyllasband/android/ScyllasBandAssetInstaller.kt"
        ).read_text(encoding="utf-8")
        self.assertIn("scyllasband/models", gradle)
        self.assertIn('"v2/onnx-int8"', gradle)
        self.assertIn('"v2/onnx"', gradle)
        self.assertIn('"v1/onnx-int8"', gradle)
        self.assertLess(gradle.index('"v2/onnx-int8"'), gradle.index('"v1/onnx-int8"'))
        self.assertLess(gradle.index('"v1/onnx-int8"'), gradle.index('"onnx-int8",'))
        self.assertIn("into(\"scyllasband/onnx-int8\")", gradle)
        self.assertNotIn("coreai", gradle)
        self.assertIn("DEFAULT_ASSET_ROOT = \"scyllasband/onnx-int8\"", installer)


    def test_ios_asset_packager_keeps_backends_on_v2_first_release(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        script = (repository_root / "examples/ios/Scripts/prepare_assets.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('"${models_dir}/v2" "${models_dir}/v1" "${models_dir}"', script)
        self.assertIn('selected_models_dir="${candidate}"', script)
        self.assertIn('${selected_models_dir}/coreai/manifest.json', script)
        self.assertIn('${selected_models_dir}/${onnx_name}/manifest.json', script)

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

    def test_runtime_auto_prefers_coreai_only_on_supported_hosts(self) -> None:
        runtime = ScyllasBandRuntime(
            ".",
            SimpleNamespace(preferred_backends=("litert",)),
            backends=["auto"],
        )
        runtime._has_backend_artifacts = lambda backend: backend == "onnx"
        self.assertEqual(runtime._select_backend(), "onnx")

        runtime._has_backend_artifacts = lambda backend: backend in ("onnx", "coreai")
        with mock.patch("scyllasband.runtime.coreai_host_supported", return_value=True):
            self.assertEqual(runtime._select_backend(), "coreai")
        with mock.patch("scyllasband.runtime.coreai_host_supported", return_value=False):
            self.assertEqual(runtime._select_backend(), "onnx")

    def test_native_binding_defaults_to_onnx(self) -> None:
        parameters = inspect.signature(NativeScyllasBandRuntime).parameters
        backend_default = parameters["backend"].default
        self.assertEqual(backend_default, "onnx")
        self.assertEqual(parameters["max_cached_target_buckets"].default, 0)
        self.assertEqual(_backend_id("onnx"), SCYLLASBAND_BACKEND_ONNX)


if __name__ == "__main__":
    unittest.main()
