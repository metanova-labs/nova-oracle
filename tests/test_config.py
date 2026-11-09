import os
import unittest
from unittest import mock

from oracle.config import Config

MACHINE = {"ORACLE_MODEL": "boltz2", "ORACLE_GPUS": "0,1", "ORACLE_HOST": "127.0.0.1",
           "ORACLE_PORT": "8000", "ORACLE_TOKEN": "t", "ORACLE_DB": "x.db"}
TUNING = {"shards_per_gpu": 3, "stagger_s": 2, "request_timeout_s": 900,
          "recycle_rss_gb": 17, "max_batch": 256}


class ConfigTest(unittest.TestCase):
    def test_machine_settings_are_required(self):
        for name in MACHINE:
            env = {k: v for k, v in MACHINE.items() if k != name}
            with self.subTest(name), mock.patch.dict(os.environ, env, clear=True):
                with self.assertRaises(SystemExit):
                    Config.from_env(TUNING)

    def test_tuning_defaults_to_the_model(self):
        with mock.patch.dict(os.environ, MACHINE, clear=True):
            cfg = Config.from_env(TUNING)
        self.assertEqual((cfg.shards_per_gpu, cfg.max_batch, cfg.shards), (3, 256, 6))
        self.assertIn("max_batch=256 (default)", cfg.describe())

    def test_tuning_can_be_overridden(self):
        with mock.patch.dict(os.environ, MACHINE | {"ORACLE_MAX_BATCH": "512"}, clear=True):
            cfg = Config.from_env(TUNING)
        self.assertEqual(cfg.max_batch, 512)
        self.assertTrue(cfg.describe().endswith("max_batch=512"))


if __name__ == "__main__":
    unittest.main()
