import dataclasses
import hashlib
import os
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import download_model
from config import load_model_spec
from tests.helpers import ROOT, tmpdir

PAYLOAD = b"pretend-weights" * 1000


class FakeResponse:
    def __init__(self, data):
        self._data, self._pos = data, 0
        self.headers = {"Content-Length": str(len(data))}

    def read(self, n=-1):
        chunk = self._data[self._pos:self._pos + n] if n >= 0 else self._data[self._pos:]
        self._pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class DownloadTests(unittest.TestCase):
    def _spec(self, payload):
        spec = load_model_spec(ROOT / "models" / "model_config.json", env={})
        return dataclasses.replace(spec, sha256=hashlib.sha256(payload).hexdigest(), expected_size_bytes=len(payload))

    def _run(self, tmp, payload, urlopen, extra_args=()):
        spec = self._spec(payload)
        with mock.patch.dict(os.environ, {"MODEL_DIR": tmp}), \
             mock.patch.object(download_model, "load_model_spec", return_value=spec), \
             mock.patch("urllib.request.urlopen", urlopen):
            return download_model.main(list(extra_args)), Path(tmp) / spec.local_filename

    def test_success_verifies_and_is_idempotent(self):
        with tmpdir() as d:
            code, dest = self._run(d, PAYLOAD, lambda *a, **k: FakeResponse(PAYLOAD))
            self.assertEqual(code, 0)
            self.assertEqual(dest.read_bytes(), PAYLOAD)
            self.assertFalse(dest.with_suffix(dest.suffix + ".part").exists())
            code2, _ = self._run(d, PAYLOAD, mock.Mock(side_effect=AssertionError("must not re-download")))
            self.assertEqual(code2, 0)

    def test_corrupted_download_is_rejected_and_removed(self):
        with tmpdir() as d:
            code, dest = self._run(d, PAYLOAD, lambda *a, **k: FakeResponse(b"x" * len(PAYLOAD)))
            self.assertEqual(code, 3)
            self.assertFalse(dest.exists())

    def test_network_failure_returns_clear_error(self):
        with tmpdir() as d:
            code, dest = self._run(d, PAYLOAD, mock.Mock(side_effect=urllib.error.URLError("no network")))
            self.assertEqual(code, 2)
            self.assertFalse(dest.exists())

    def test_http_403_returns_error(self):
        err = urllib.error.HTTPError("u", 403, "Forbidden", {}, None)
        with tmpdir() as d:
            code, _ = self._run(d, PAYLOAD, mock.Mock(side_effect=err))
            self.assertEqual(code, 2)

    def test_verify_only_flag(self):
        with tmpdir() as d:
            code, _ = self._run(d, PAYLOAD, mock.Mock(side_effect=AssertionError("no download")), ["--verify"])
            self.assertEqual(code, 1)

    def test_url_construction_and_real_spec_pins_hash(self):
        spec = load_model_spec(ROOT / "models" / "model_config.json", env={})
        self.assertEqual(spec.sha256, "e833f69d110f167cad4a6c38d385564bdb2f6de63d246e45cb03ff9aa17f0349")
        self.assertEqual(spec.expected_size_bytes, 12268364)
        url = download_model.build_url(spec.repo_id, spec.revision, spec.filename)
        self.assertTrue(url.startswith("https://huggingface.co/visionlab-ai/school-violence-detection-models/resolve/main/final/final_x3d_realtime.pt"))


if __name__ == "__main__":
    unittest.main()
