"""Download tests without the network: a fake snapshot_download stands in for the Hub."""

from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scyllasband import download
from scyllasband.download import RELEASE, REPO_ID, SING_RELEASE, SING_REPO_ID, download_bundle, download_candidates, inaccessible


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
            if allow_patterns[0] == f"{flavor}/**":
                bundle = Path(local_dir) / flavor
                bundle.mkdir(parents=True)
                (bundle / "manifest.json").write_text(json.dumps(dict(controls=dict(graph_input_contract=f"{repo_id}:{revision}"))))


def _hub_error(cls, status=None):
    from huggingface_hub.utils import HfHubHTTPError
    exc = cls("denied") if cls is not HfHubHTTPError else HfHubHTTPError("denied")
    if status is not None:
        exc.response = mock.Mock(status_code=status)
    return exc


class CandidateTest(unittest.TestCase):
    def test_order(self):
        self.assertEqual(download_candidates(), [(SING_REPO_ID, SING_RELEASE), (REPO_ID, RELEASE)])
        self.assertEqual(download_candidates(revision="main"), [(SING_REPO_ID, "main"), (REPO_ID, RELEASE)])
        self.assertEqual(download_candidates(public=True), [(REPO_ID, RELEASE)])
        self.assertEqual(download_candidates(public=True, revision="v3-x"), [(REPO_ID, "v3-x")])
        self.assertEqual(download_candidates(repo_id="me/fork"), [("me/fork", None)])
        self.assertEqual(download_candidates(repo_id="me/fork", revision="dev"), [("me/fork", "dev")])
        self.assertEqual(download_candidates(repo_id=SING_REPO_ID, public=True), [(SING_REPO_ID, SING_RELEASE)])

    def test_inaccessible_errors(self):
        from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError, RevisionNotFoundError
        for cls in (RepositoryNotFoundError, GatedRepoError, RevisionNotFoundError):
            self.assertTrue(inaccessible(_hub_error(cls)))
        for status in (401, 403, 404):
            self.assertTrue(inaccessible(_hub_error(HfHubHTTPError, status)))
        self.assertFalse(inaccessible(_hub_error(HfHubHTTPError, 500)))
        self.assertFalse(inaccessible(OSError("disk full")))
        self.assertFalse(inaccessible(ValueError("bad")))


class FallbackTest(unittest.TestCase):
    def run_download(self, repos, **kwargs):
        hub = _FakeHub(repos)
        log = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(download, "_require_snapshot_download", return_value=hub):
            path = download_bundle(flavor="litert", models_dir=tmp, validate=False, log=log, **kwargs)
            provenance = json.loads((path / "download_provenance.json").read_text())
            manifest = json.loads((path / "manifest.json").read_text())
        return hub.calls, provenance, manifest["controls"]["graph_input_contract"], log.getvalue()

    def test_accessible_sing_repository_is_used_first(self):
        calls, provenance, origin, log = self.run_download({SING_REPO_ID: ["litert", "onnx"], REPO_ID: ["litert"]}, token="hf_x")
        self.assertEqual(calls, [(SING_REPO_ID, SING_RELEASE, "litert", "hf_x")])
        self.assertEqual(origin, f"{SING_REPO_ID}:{SING_RELEASE}")
        self.assertEqual((provenance["repo_id"], provenance["revision"], provenance["flavor"]), (SING_REPO_ID, SING_RELEASE, "litert"))
        self.assertEqual(provenance["graph_contract"], origin)
        self.assertEqual(log, "")

    def test_inaccessible_sing_repository_falls_back_to_the_public_one(self):
        from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
        for error in (_hub_error(RepositoryNotFoundError), _hub_error(GatedRepoError), _hub_error(HfHubHTTPError, 401),
                      _hub_error(HfHubHTTPError, 403)):
            calls, provenance, origin, log = self.run_download({SING_REPO_ID: error, REPO_ID: ["litert"]})
            self.assertEqual(calls, [(SING_REPO_ID, SING_RELEASE, "litert", None), (REPO_ID, RELEASE, "litert", None)])
            self.assertEqual((provenance["repo_id"], provenance["revision"]), (REPO_ID, RELEASE))
            self.assertEqual(origin, f"{REPO_ID}:{RELEASE}")
            self.assertEqual(log.count("\n"), 1)
            self.assertIn("not accessible", log)

    def test_a_flavour_the_sing_repository_lacks_falls_back(self):
        calls, provenance, _, log = self.run_download({SING_REPO_ID: ["litert", "onnx"], REPO_ID: ["litert", "coreml"]})
        self.assertEqual(provenance["repo_id"], SING_REPO_ID)
        hub = _FakeHub({SING_REPO_ID: ["onnx"], REPO_ID: ["litert"]})
        log = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(download, "_require_snapshot_download", return_value=hub):
            path = download_bundle(flavor="litert", models_dir=tmp, validate=False, log=log)
            self.assertEqual(json.loads((path / "download_provenance.json").read_text())["repo_id"], REPO_ID)
        self.assertEqual([c[0] for c in hub.calls], [SING_REPO_ID, REPO_ID])
        self.assertIn("has no 'litert' bundle", log.getvalue())

    def test_public_flag_skips_the_sing_repository(self):
        calls, provenance, _, log = self.run_download({SING_REPO_ID: ["litert"], REPO_ID: ["litert"]}, public=True)
        self.assertEqual(calls, [(REPO_ID, RELEASE, "litert", None)])
        self.assertEqual(provenance["repo_id"], REPO_ID)

    def test_explicit_repository_is_used_as_given(self):
        calls, provenance, _, _ = self.run_download({"me/fork": ["litert"], SING_REPO_ID: ["litert"]}, repo_id="me/fork", revision="dev")
        self.assertEqual(calls, [("me/fork", "dev", "litert", None)])
        self.assertEqual((provenance["repo_id"], provenance["revision"]), ("me/fork", "dev"))
        calls, _, _, _ = self.run_download({"me/fork": ["litert"]}, repo_id="me/fork")
        self.assertEqual(calls, [("me/fork", None, "litert", None)])

    def test_other_errors_and_a_missing_public_flavour_are_raised(self):
        from huggingface_hub.utils import HfHubHTTPError
        with self.assertRaises(HfHubHTTPError):
            self.run_download({SING_REPO_ID: _hub_error(HfHubHTTPError, 500), REPO_ID: ["litert"]})
        with self.assertRaises(OSError):
            self.run_download({SING_REPO_ID: OSError("network"), REPO_ID: ["litert"]})
        with self.assertRaisesRegex(FileNotFoundError, "no 'litert' bundle"):
            self.run_download({SING_REPO_ID: ["onnx"], REPO_ID: ["onnx"]})
        with self.assertRaises(FileNotFoundError):
            self.run_download({SING_REPO_ID: ["litert"], REPO_ID: ["onnx"]}, public=True)

    def test_cli_download_options(self):
        from scyllasband.cli import build_parser
        args = build_parser().parse_args(["download", "--flavor", "onnx", "--public", "--revision", "v3-x"])
        self.assertEqual((args.repo_id, args.revision, args.public), (None, "v3-x", True))
        args = build_parser().parse_args(["download", "--repo", "me/fork"])
        self.assertEqual((args.repo_id, args.revision, args.public), ("me/fork", None, False))
        self.assertEqual(build_parser().parse_args(["download", "--repo-id", "me/fork"]).repo_id, "me/fork")


if __name__ == "__main__":
    unittest.main()
