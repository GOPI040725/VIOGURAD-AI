import unittest
from pathlib import Path

from evaluate import (build_report, choose_setting, confusion, discover_videos, evaluate_setting, label_from_folder,
                      metrics_from_counts, stratified_split, wilson_interval)
from tests.helpers import tmpdir


class MetricTests(unittest.TestCase):
    def test_confusion_and_metrics(self):
        y = [1, 1, 1, 0, 0, 0, 0, 1]
        p = [1, 1, 0, 0, 0, 1, 0, 1]
        c = confusion(y, p)
        self.assertEqual(c, {"tp": 3, "tn": 3, "fp": 1, "fn": 1})
        m = metrics_from_counts(c)
        self.assertAlmostEqual(m["accuracy"], 0.75)
        self.assertAlmostEqual(m["precision"], 0.75)
        self.assertAlmostEqual(m["recall"], 0.75)
        self.assertAlmostEqual(m["f1"], 0.75)
        self.assertAlmostEqual(m["false_positive_rate"], 0.25)

    def test_degenerate_counts_do_not_crash(self):
        m = metrics_from_counts({"tp": 0, "tn": 0, "fp": 0, "fn": 0})
        self.assertEqual(m["accuracy"], 0.0)
        m = metrics_from_counts({"tp": 0, "tn": 5, "fp": 0, "fn": 5})
        self.assertEqual((m["precision"], m["recall"], m["f1"]), (0.0, 0.0, 0.0))

    def test_wilson_interval(self):
        lo, hi = wilson_interval(95, 100)
        self.assertAlmostEqual(lo, 0.8882, places=3)
        self.assertAlmostEqual(hi, 0.9785, places=3)
        self.assertEqual(wilson_interval(0, 0), (0.0, 0.0))


class DatasetTests(unittest.TestCase):
    def test_label_from_folder(self):
        for name, want in [("Violence", 1), ("NonViolence", 0), ("non_violence", 0), ("Fight", 1), ("NonFight", 0),
                           ("violent", 1), ("Normal", 0), ("random_stuff", None)]:
            self.assertEqual(label_from_folder(name), want, name)

    def test_discover_and_split_are_disjoint_and_stratified(self):
        with tmpdir() as d:
            root = Path(d)
            for folder, n in (("Violence", 20), ("NonViolence", 30)):
                (root / folder).mkdir()
                for i in range(n):
                    (root / folder / f"{i}.mp4").write_bytes(b"x")
            (root / "Violence" / "readme.txt").write_text("ignore me")
            items = discover_videos(root)
            self.assertEqual(len(items), 50)
            self.assertEqual(sum(l for _, l in items), 20)
            val, test = stratified_split(items, 0.3, seed=7)
            self.assertFalse({p for p, _ in val} & {p for p, _ in test})
            self.assertEqual(len(val) + len(test), 50)
            self.assertEqual(sum(1 for _, l in val if l == 1), 6)
            self.assertEqual(sum(1 for _, l in val if l == 0), 9)
            self.assertEqual(stratified_split(items, 0.3, seed=7), (val, test))  # reproducible


class SettingSelectionTests(unittest.TestCase):
    def test_selection_uses_validation_and_reports_on_test(self):
        val = [{"label": 1, "probs": [0.7, 0.9]}, {"label": 1, "probs": [0.6]}, {"label": 0, "probs": [0.1, 0.2]}, {"label": 0, "probs": [0.15]}]
        chosen = choose_setting(val, 0.4)
        self.assertEqual(chosen["validation_metrics"]["accuracy"], 1.0)
        test = [{"label": 1, "probs": [0.8]}, {"label": 0, "probs": [0.55]}]  # hard negative
        m = evaluate_setting(test, chosen["mode"], chosen["threshold"])
        self.assertEqual(m["n"], 2)
        self.assertEqual(m["fp"], 1)  # honest: an error on held-out data is reported, not hidden

    def test_report_never_claims_target_when_missed(self):
        low = metrics_from_counts({"tp": 40, "tn": 40, "fp": 10, "fn": 10})
        meta = {"data": "x", "found": 100, "skipped": 0, "seed": 1, "n_violent": 50, "n_nonviolent": 50, "n_val": 0, "n_test": 100, "model": "m"}
        text = build_report(meta, {"mode": "max", "threshold": 0.4, "validation_metrics": low}, low, low, low)
        self.assertIn("does NOT meet", text)
        self.assertNotIn("| meets", text)
        high = metrics_from_counts({"tp": 50, "tn": 50, "fp": 0, "fn": 0})
        self.assertIn("meets", build_report(meta, {"mode": "max", "threshold": 0.4, "validation_metrics": high}, high, high, high))


if __name__ == "__main__":
    unittest.main()
