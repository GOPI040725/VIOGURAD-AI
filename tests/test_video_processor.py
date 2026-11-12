import dataclasses
import unittest
from pathlib import Path

import numpy as np

from tests.helpers import FakeMotionModel, default_spec, tmpdir, write_video
from video_processor import (VideoAnalyzer, VideoDecodeError, VideoInfo, VideoTooLongError, aggregate_scores,
                             certainty_label, extract_segment_clips, plan_segments, probe_video, sample_indices)


class ProbeTests(unittest.TestCase):
    def test_valid_video(self):
        with tmpdir() as d:
            p = Path(d) / "a.avi"
            n = write_video(p, seconds=2, fps=30)
            info = probe_video(p)
            self.assertEqual(info.frame_count, n)
            self.assertAlmostEqual(info.fps, 30, delta=1)
            self.assertEqual((info.width, info.height), (160, 120))
            self.assertAlmostEqual(info.duration, 2.0, delta=0.1)

    def test_corrupted_file(self):
        with tmpdir() as d:
            p = Path(d) / "bad.mp4"
            p.write_bytes(np.random.default_rng(1).bytes(5000))
            with self.assertRaises(VideoDecodeError):
                probe_video(p)

    def test_missing_and_empty_file(self):
        with tmpdir() as d:
            with self.assertRaises(VideoDecodeError):
                probe_video(Path(d) / "nope.mp4")
            e = Path(d) / "empty.avi"
            e.write_bytes(b"")
            with self.assertRaises(VideoDecodeError):
                probe_video(e)

    def test_too_few_frames_and_too_long(self):
        with tmpdir() as d:
            p = Path(d) / "short.avi"
            write_video(p, seconds=5 / 30, fps=30)
            with self.assertRaises(VideoDecodeError):
                probe_video(p, min_frames=8)
            q = Path(d) / "ok.avi"
            write_video(q, seconds=3)
            with self.assertRaises(VideoTooLongError):
                probe_video(q, max_seconds=1)


class PlanTests(unittest.TestCase):
    spec = default_spec()

    def test_sample_indices(self):
        idx = sample_indices(0, 150, 16)
        self.assertEqual(len(idx), 16)
        self.assertEqual((idx[0], idx[-1]), (0, 149))
        self.assertEqual(idx, sorted(idx))

    def test_short_video_single_segment(self):
        segs = plan_segments(VideoInfo(120, 30.0, 320, 240), self.spec)
        self.assertEqual(len(segs), 1)
        self.assertEqual((segs[0].start_frame, segs[0].end_frame), (0, 120))

    def test_long_video_overlapping_windows_cover_end(self):
        info = VideoInfo(30 * 60, 30.0, 320, 240)  # 60 s
        segs = plan_segments(info, self.spec)
        self.assertGreater(len(segs), 5)
        self.assertEqual(segs[0].start_frame, 0)
        self.assertEqual(segs[-1].end_frame, info.frame_count)
        self.assertTrue(all(len(s.frame_indices) == 16 for s in segs))
        self.assertTrue(all(b.start_frame > a.start_frame for a, b in zip(segs, segs[1:])))
        self.assertTrue(all(s.end_frame - s.start_frame == 150 for s in segs))

    def test_very_long_video_is_capped(self):
        info = VideoInfo(30 * 3600, 30.0, 1920, 1080)
        segs = plan_segments(info, self.spec)
        self.assertLessEqual(len(segs), self.spec.max_segments)
        self.assertEqual(segs[-1].end_frame, info.frame_count)

    def test_video_with_few_frames_still_gives_16_indices(self):
        segs = plan_segments(VideoInfo(10, 30.0, 64, 64), self.spec)
        self.assertEqual(len(segs[0].frame_indices), 16)
        self.assertTrue(max(segs[0].frame_indices) <= 9)


class ExtractionTests(unittest.TestCase):
    spec = default_spec()

    def test_shape_dtype_and_rgb_order(self):
        with tmpdir() as d:
            p = Path(d) / "red.avi"
            write_video(p, seconds=2, kind="red")
            info = probe_video(p)
            segs = plan_segments(info, self.spec)
            clips = extract_segment_clips(p, segs, 224)
            self.assertEqual(clips.shape, (1, 16, 224, 224, 3))
            self.assertEqual(clips.dtype, np.uint8)
            r, g, b = (clips[..., c].mean() for c in range(3))
            self.assertGreater(r, 200)  # RGB order: a red video has a high R channel
            self.assertLess(b, 60)

    def test_metadata_longer_than_real_stream_is_handled(self):
        with tmpdir() as d:
            p = Path(d) / "s.avi"
            write_video(p, seconds=2)
            fake_info = VideoInfo(frame_count=600, fps=30.0, width=160, height=120)  # claims 20 s
            segs = plan_segments(fake_info, self.spec)
            clips = extract_segment_clips(p, segs, 224)
            self.assertEqual(clips.shape[0], len(segs))
            self.assertEqual(clips.shape[1:], (16, 224, 224, 3))

    def test_progress_callback_reports_decoding(self):
        with tmpdir() as d:
            p = Path(d) / "s.avi"
            write_video(p, seconds=2)
            seen = []
            extract_segment_clips(p, plan_segments(probe_video(p), self.spec), 224, lambda s, f, e: seen.append((s, f)))
            self.assertTrue(seen)
            self.assertEqual(seen[-1], ("decoding", 1.0))


class AggregationTests(unittest.TestCase):
    def test_modes(self):
        p = [0.1, 0.9, 0.2, 0.8]
        self.assertEqual(aggregate_scores(p, "max", 0.4)[0], 0.9)
        self.assertAlmostEqual(aggregate_scores(p, "mean", 0.4)[0], 0.5)
        self.assertAlmostEqual(aggregate_scores(p, "topk", 0.4, topk=2)[0], 0.85)
        self.assertAlmostEqual(aggregate_scores([0.1, 0.9, 0.2, 0.8], "consecutive", 0.4, min_consecutive=2)[0], 0.2)
        self.assertAlmostEqual(aggregate_scores([0.1, 0.9, 0.8, 0.2], "consecutive", 0.4, min_consecutive=2)[0], 0.8)

    def test_threshold_boundary_and_errors(self):
        self.assertTrue(aggregate_scores([0.4], "max", 0.4)[1])
        self.assertFalse(aggregate_scores([0.399], "max", 0.4)[1])
        with self.assertRaises(ValueError):
            aggregate_scores([], "max", 0.4)
        with self.assertRaises(ValueError):
            aggregate_scores([0.5], "bogus", 0.4)

    def test_certainty_label(self):
        self.assertEqual(certainty_label(0.41, 0.4), "borderline")
        self.assertEqual(certainty_label(0.99, 0.4), "strong")
        self.assertEqual(certainty_label(0.02, 0.4), "strong")


class AnalyzerTests(unittest.TestCase):
    def test_pipeline_end_to_end_with_stand_in_model(self):
        spec = default_spec()
        with tmpdir() as d:
            moving, still = Path(d) / "m.avi", Path(d) / "s.avi"
            write_video(moving, seconds=3, kind="motion")
            write_video(still, seconds=3, kind="static")
            model = FakeMotionModel()
            an = VideoAnalyzer(model, spec)
            stages = []
            r1 = an.analyze(moving, lambda s, f, e: stages.append(s))
            r2 = an.analyze(still)
            self.assertEqual(r1["label"], "violence")
            self.assertEqual(r1["verdict"], "VIOLENCE DETECTED")
            self.assertEqual(r2["label"], "non_violence")
            self.assertEqual(r2["verdict"], "NON-VIOLENCE DETECTED")
            self.assertEqual(r1["frames_analyzed"], 16)
            self.assertEqual(r1["segment_count"], len(r1["segments"]))
            self.assertGreater(r1["processing_time_sec"], 0)
            self.assertIn("inference", stages)
            self.assertEqual(stages[-1], "done")

    def test_long_video_produces_multiple_segments(self):
        spec = default_spec()
        with tmpdir() as d:
            p = Path(d) / "long.avi"
            write_video(p, seconds=20, fps=10, size=(64, 48), kind="static")
            r = VideoAnalyzer(FakeMotionModel(), spec, batch_size=3).analyze(p)
            self.assertGreaterEqual(r["segment_count"], 5)
            self.assertAlmostEqual(r["video"]["duration_sec"], 20, delta=0.5)
            self.assertEqual(r["segments"][-1]["end_sec"], r["video"]["duration_sec"])

    def test_corrupted_video_raises(self):
        with tmpdir() as d:
            p = Path(d) / "bad.mov"
            p.write_bytes(b"not a video" * 100)
            with self.assertRaises(VideoDecodeError):
                VideoAnalyzer(FakeMotionModel(), default_spec()).analyze(p)


if __name__ == "__main__":
    unittest.main()
