import unittest

from config import Settings, load_model_spec
from tests.helpers import ROOT

CFG = ROOT / "models" / "model_config.json"


class ConfigTests(unittest.TestCase):
    def test_defaults_match_verified_model_facts(self):
        s = load_model_spec(CFG, env={})
        self.assertEqual((s.num_frames, s.frame_size, s.threshold), (16, 224, 0.4))
        self.assertEqual(s.mean, (0.45, 0.45, 0.45))
        self.assertEqual(s.std, (0.225, 0.225, 0.225))
        self.assertEqual(s.violent_index, 1)

    def test_env_overrides(self):
        s = load_model_spec(CFG, env={"VIOLENCE_THRESHOLD": "0.55", "AGGREGATION": "TopK"})
        self.assertEqual((s.threshold, s.aggregation), (0.55, "topk"))

    def test_invalid_values_rejected(self):
        with self.assertRaises(ValueError):
            load_model_spec(CFG, env={"AGGREGATION": "median"})
        with self.assertRaises(ValueError):
            load_model_spec(CFG, env={"VIOLENCE_THRESHOLD": "1.5"})

    def test_settings_safe_defaults(self):
        s = Settings.from_env()
        self.assertFalse(s.debug)
        self.assertEqual(s.host, "127.0.0.1")
        self.assertEqual(s.allowed_extensions, ("mp4", "avi", "mov"))


if __name__ == "__main__":
    unittest.main()
