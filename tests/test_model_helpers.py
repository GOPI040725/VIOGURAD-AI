import dataclasses
import hashlib
import unittest
from pathlib import Path

import numpy as np

from tests.helpers import default_spec, make_settings, tmpdir
from violence_model import (ModelUnavailableError, ViolenceModel, detect_head_layout, probabilities_from_logits,
                            sha256_file, strip_wrapper_prefixes, unwrap_checkpoint)

try:
    import torch  # noqa: F401
    import pytorchvideo  # noqa: F401
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


class CheckpointHelperTests(unittest.TestCase):
    def test_unwrap_variants(self):
        sd = {"a": 1}
        self.assertIs(unwrap_checkpoint({"model": sd}), sd)
        self.assertIs(unwrap_checkpoint({"model_state_dict": sd, "epoch": 3}), sd)
        self.assertIs(unwrap_checkpoint({"state_dict": sd}), sd)
        self.assertIs(unwrap_checkpoint(sd), sd)
        with self.assertRaises(ModelUnavailableError):
            unwrap_checkpoint([1, 2, 3])

    def test_strip_prefixes(self):
        out = strip_wrapper_prefixes({"module.backbone.blocks.0.conv.weight": 1, "blocks.5.proj.weight": 2, "backbone.blocks.1.x": 3})
        self.assertEqual(set(out), {"blocks.0.conv.weight", "blocks.5.proj.weight", "blocks.1.x"})

    def test_detect_head_layouts(self):
        self.assertEqual(detect_head_layout({"blocks.5.proj.weight": (2, 2048)}), ("linear", 2, 2048))
        self.assertEqual(detect_head_layout({"blocks.5.proj.1.weight": (2, 2048)}), ("sequential", 2, 2048))
        self.assertEqual(detect_head_layout({"blocks.5.proj.weight": (1, 2048)})[1], 1)
        with self.assertRaises(ModelUnavailableError) as cm:
            detect_head_layout({"fc.weight": (10, 5)})
        self.assertEqual(cm.exception.code, "bad_checkpoint")

    def test_probabilities(self):
        p = probabilities_from_logits(np.array([[0.0, 0.0], [0.0, 2.0], [5.0, -5.0]]), 1)
        np.testing.assert_allclose(p[0], 0.5)
        self.assertGreater(p[1], 0.85)
        self.assertLess(p[2], 0.001)
        big = probabilities_from_logits(np.array([[1000.0, -1000.0]]), 1)  # numerically stable
        self.assertTrue(np.isfinite(big).all())
        s = probabilities_from_logits(np.array([[0.0], [3.0]]), 1)  # single-logit sigmoid
        np.testing.assert_allclose(s, 1 / (1 + np.exp(-np.array([0.0, 3.0]))))

    def test_sha256_file(self):
        with tmpdir() as d:
            p = Path(d) / "f.bin"
            p.write_bytes(b"hello")
            self.assertEqual(sha256_file(p), hashlib.sha256(b"hello").hexdigest())


class ModelStateTests(unittest.TestCase):
    def test_status_without_weights(self):
        with tmpdir() as d:
            m = ViolenceModel(default_spec(), make_settings(Path(d)))
            st = m.status()
            self.assertIn(st["state"], ("weights_missing", "dependency_missing"))
            self.assertFalse(st["available"])
            self.assertFalse(m.is_loaded)

    def test_load_without_weights_raises_clear_error(self):
        with tmpdir() as d:
            m = ViolenceModel(default_spec(), make_settings(Path(d)))
            with self.assertRaises(ModelUnavailableError) as cm:
                m.load()
            self.assertIn(cm.exception.code, ("weights_missing", "dependency_missing"))
            self.assertIn("python download_model.py" if cm.exception.code == "weights_missing" else "pip install", str(cm.exception))

    def test_predict_rejects_wrong_shape(self):
        with tmpdir() as d:
            m = ViolenceModel(default_spec(), make_settings(Path(d)))
            with self.assertRaises(ValueError):
                m.predict_clips(np.zeros((1, 8, 224, 224, 3), np.uint8))

    @unittest.skipUnless(HAVE_TORCH, "torch + pytorchvideo not installed in this environment")
    def test_truncated_or_wrong_file_is_rejected(self):
        with tmpdir() as d:
            settings = make_settings(Path(d))
            spec = default_spec()
            (settings.model_dir / spec.local_filename).write_bytes(b"x" * 1000)
            with self.assertRaises(ModelUnavailableError) as cm:
                ViolenceModel(spec, settings).load()
            self.assertEqual(cm.exception.code, "bad_checkpoint")


@unittest.skipUnless(HAVE_TORCH, "torch + pytorchvideo not installed in this environment")
class RandomWeightsLoadingTests(unittest.TestCase):
    """Software test only: RANDOMLY initialised weights exercise the loader and the forward pass.
    These say nothing about detection accuracy."""

    def _roundtrip(self, head_kind, wrap):
        import torch
        from pytorchvideo.models.hub import x3d_m
        from violence_model import _install_torchvision_shim
        _install_torchvision_shim()
        net = x3d_m(pretrained=False)
        lin = torch.nn.Linear(net.blocks[5].proj.in_features, 2)
        net.blocks[5].proj = lin if head_kind == "linear" else torch.nn.Sequential(torch.nn.Dropout(0.3), lin)
        state = net.state_dict()
        if wrap:
            state = {f"backbone.{k}": v for k, v in state.items()}
        with tmpdir() as d:
            settings = make_settings(Path(d))
            spec = dataclasses.replace(default_spec(), sha256="", expected_size_bytes=0)
            torch.save({"model_state_dict": state}, settings.model_dir / spec.local_filename)
            model = ViolenceModel(spec, settings)
            model.load()
            self.assertTrue(model.is_loaded)
            probs = model.predict_clips(np.random.default_rng(0).integers(0, 256, (2, 16, 224, 224, 3), dtype=np.uint8))
            self.assertEqual(probs.shape, (2,))
            self.assertTrue(((probs >= 0) & (probs <= 1)).all())

    def test_linear_head_wrapped_prefix(self):
        self._roundtrip("linear", True)

    def test_sequential_head_plain_keys(self):
        self._roundtrip("sequential", False)


if __name__ == "__main__":
    unittest.main()
