"""Download tests without the network: a fake snapshot_download stands in for the Hub."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scyllasband import download
from scyllasband.download import RELEASE, REPO_ID, default_revision, download_bundle


class _FakeHub:
    """Records snapshot_download calls; ``repos`` maps a repository to the flavours it publishes or to an exception."""

    def __init__(self, repos):
        self.repos, self.calls = repos, []

    def __call__(self, *, repo_id, repo_type, allow_patterns, local_dir, force_download, token, revision):
        self.calls.append((repo_id, revision, allow_patterns[0].split("/")[0], token))
        published = self.repos.get(repo_id)
        if isinstance(published, BaseException):
            raise published
        for flavor in published or ():
            if flavor == "melodies" and "melodies/**" in allow_patterns:
                (Path(local_dir) / "melodies").mkdir(parents=True)
                (Path(local_dir) / "melodies" / "song.json").write_text(json.dumps(dict(source=repo_id)))
            elif allow_patterns[0] == f"{flavor}/**":
                bundle = Path(local_dir) / flavor
                bundle.mkdir(parents=True)
                (bundle / "manifest.json").write_text(json.dumps(dict(controls=dict(graph_input_contract=f"{repo_id}:{revision}"))))


class DownloadTest(unittest.TestCase):
    def run_download(self, repos, **kwargs):
        hub = _FakeHub(repos)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(download, "_require_snapshot_download", return_value=hub):
            path = download_bundle(flavor="litert", models_dir=tmp, validate=False, **kwargs)
            provenance = json.loads((path / "download_provenance.json").read_text())
            manifest = json.loads((path / "manifest.json").read_text())
        return hub.calls, provenance, manifest["controls"]["graph_input_contract"]

    def test_the_release_repository_at_its_release_tag(self):
        calls, provenance, origin = self.run_download({REPO_ID: ["litert", "onnx"]}, token="hf_x")
        self.assertEqual(calls, [(REPO_ID, RELEASE, "litert", "hf_x")])
        self.assertEqual(origin, f"{REPO_ID}:{RELEASE}")
        self.assertEqual((provenance["repo_id"], provenance["revision"], provenance["flavor"]), (REPO_ID, RELEASE, "litert"))
        self.assertEqual(provenance["graph_contract"], origin)

    def test_default_revisions(self):
        self.assertEqual(default_revision(REPO_ID), RELEASE)
        self.assertIsNone(default_revision("me/fork"))

    def test_an_earlier_release_by_tag(self):
        calls, provenance, _ = self.run_download({REPO_ID: ["litert"]}, revision="v3-20261104-apple")
        self.assertEqual(calls, [(REPO_ID, "v3-20261104-apple", "litert", None)])
        self.assertEqual(provenance["revision"], "v3-20261104-apple")

    def test_explicit_repository_is_used_as_given(self):
        calls, provenance, _ = self.run_download({"me/fork": ["litert"]}, repo_id="me/fork", revision="dev")
        self.assertEqual(calls, [("me/fork", "dev", "litert", None)])
        self.assertEqual((provenance["repo_id"], provenance["revision"]), ("me/fork", "dev"))
        calls, _, _ = self.run_download({"me/fork": ["litert"]}, repo_id="me/fork")
        self.assertEqual(calls, [("me/fork", None, "litert", None)])

    def test_melodies_are_installed_beside_the_bundles(self):
        with tempfile.TemporaryDirectory() as tmp:
            hub = _FakeHub({REPO_ID: ["litert", "melodies"], "me/fork": ["litert"]})
            with mock.patch.object(download, "_require_snapshot_download", return_value=hub):
                download_bundle(flavor="litert", models_dir=tmp, validate=False)
                song = Path(tmp) / "melodies" / "song.json"
                self.assertEqual(json.loads(song.read_text())["source"], REPO_ID)
                download_bundle(flavor="litert", models_dir=tmp, validate=False, repo_id="me/fork")
                self.assertTrue(song.is_file())                      # a repository without melodies keeps the installed copy

    def test_errors_and_a_missing_flavour_are_raised(self):
        with self.assertRaises(OSError):
            self.run_download({REPO_ID: OSError("network")})
        with self.assertRaisesRegex(FileNotFoundError, "no 'litert' bundle"):
            self.run_download({REPO_ID: ["onnx"]})

    def test_cli_download_options(self):
        from scyllasband.cli import build_parser
        args = build_parser().parse_args(["download", "--flavor", "onnx", "--revision", "v3-x"])
        self.assertEqual((args.repo_id, args.revision), (None, "v3-x"))
        args = build_parser().parse_args(["download", "--repo", "me/fork"])
        self.assertEqual((args.repo_id, args.revision), ("me/fork", None))
        self.assertEqual(build_parser().parse_args(["download", "--repo-id", "me/fork"]).repo_id, "me/fork")
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            build_parser().parse_args(["download", "--public"])


if __name__ == "__main__":
    unittest.main()
