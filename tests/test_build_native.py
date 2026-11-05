"""build-native's SDK staging, without building or downloading anything."""

from __future__ import annotations

import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from scyllasband import build_native


class StageSdkTest(unittest.TestCase):
    def stage(self, backend, staged=()):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp)
            lib = checkout / "third_party" / "litert" / "lib" / build_native.sdk_platform()
            lib.mkdir(parents=True)
            for name in staged:
                (lib / name).write_bytes(b"")
            with mock.patch.object(build_native, "CHECKOUT_DIR", checkout), mock.patch.object(build_native.subprocess, "run") as run:
                build_native.stage_sdk(backend, log=io.StringIO())
            return run, checkout

    def test_a_missing_litert_runtime_is_downloaded_for_this_machine(self):
        run, checkout = self.stage("litert")
        run.assert_called_once_with([sys.executable, str(checkout / "scripts" / "stage_litert_sdk.py"), "--platform",
                                     build_native.sdk_platform(), "--download-runtime"], check=True)

    def test_a_staged_runtime_is_kept(self):
        for name in ("libLiteRt.so", "libLiteRt.dylib", "libLiteRt.dll"):
            run, _ = self.stage("litert", staged=[name])
            run.assert_not_called()

    def test_other_backends_stage_nothing(self):
        for backend in ("apple", "onnx"):
            run, _ = self.stage(backend)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
