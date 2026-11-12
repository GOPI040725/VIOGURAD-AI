"""Test utilities. NOTHING here is used by the application at runtime.

``FakeMotionModel`` is a deterministic stand-in used ONLY to test the plumbing
(decoding, windowing, aggregation, HTTP, cleanup). Its scores say nothing about
real violence detection accuracy.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Settings, load_model_spec  # noqa: E402


def write_video(path, seconds=3.0, fps=30, size=(160, 120), kind="static", fourcc="MJPG", color=(40, 90, 200)):
    """Write a small synthetic video. kind: static | motion | red."""
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot write {fourcc} video here")
    rng = np.random.default_rng(0)
    n = int(seconds * fps)
    for i in range(n):
        if kind == "static":
            frame = np.full((h, w, 3), color, np.uint8)
        elif kind == "red":
            frame = np.zeros((h, w, 3), np.uint8)
            frame[:, :, 2] = 255  # BGR red
        else:  # motion: strong random change every frame
            frame = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return n


class FakeMotionModel:
    """TEST ONLY. Score = amount of frame-to-frame change, squashed into [0, 1]."""

    def __init__(self, state="ready"):
        self._state = state
        self.calls = 0

    def status(self):
        ok = self._state == "ready"
        return {"state": self._state, "message": "test model" if ok else "Model weights not found. Run: python download_model.py",
                "available": ok, "device": "cpu", "name": "fake", "architecture": "fake", "threshold": 0.4, "weights_file": "none"}

    def predict_clips(self, clips):
        self.calls += 1
        x = clips.astype(np.float32) / 255.0
        diff = np.abs(np.diff(x, axis=1)).mean(axis=(1, 2, 3, 4))
        return np.clip(diff * 4.0, 0.0, 1.0)


def make_settings(tmp: Path, **overrides) -> Settings:
    base = dict(upload_dir=tmp / "uploads", output_dir=tmp / "outputs", model_dir=tmp / "models", log_dir=tmp / "logs",
                model_preload=False, max_upload_mb=1, job_ttl_seconds=60)
    base.update(overrides)
    s = Settings(**base)
    s.ensure_dirs()
    return s


def default_spec():
    return load_model_spec(ROOT / "models" / "model_config.json", env={})


def tmpdir() -> tempfile.TemporaryDirectory:
    return tempfile.TemporaryDirectory(prefix="vd_test_")
