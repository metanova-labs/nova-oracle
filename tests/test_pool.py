import os
import types
import unittest

from oracle.pool import Pool

FAKE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_worker.py")
MODEL = types.SimpleNamespace(worker=FAKE, probe=("SEQ", "input"))


def item(smiles, id_="a"):
    return [{"id": id_, "sequence": "SEQ", "input": "input", "smiles": smiles}]


class PoolTest(unittest.TestCase):
    def setUp(self):
        cfg = types.SimpleNamespace(gpus=(0, 1), shards_per_gpu=2, stagger_s=0,
                                    request_timeout_s=1, recycle_rss_gb=1e9)
        self.pool = Pool(cfg, MODEL)

    def tearDown(self):
        self.pool.shutdown()

    def test_shards_spread_breadth_first(self):
        self.assertEqual([w.gpu for w in self.pool.workers], [0, 1, 0, 1])

    def test_reply_after_other_output(self):
        message = self.pool.workers[0].score(item("CCO"), 5)
        self.assertEqual(message["results"], {"a": {"affinity_pred_value": 3.0,
                                                    "affinity_probability_binary": 0.5}})

    def test_slow_shard_times_out(self):
        with self.assertRaises(TimeoutError):
            self.pool.workers[0].score(item("SLOW"), 1)

    def test_crash_carries_stderr(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA error"):
            self.pool.workers[1].score(item("CRASH"), 5)

    def test_failed_warm_kills_the_shard(self):
        self.pool.PROBE_LIGAND = "SLOW"
        self.pool._warm_one(self.pool.workers[2])
        self.assertFalse(self.pool.workers[2].alive())

    def test_dead_shard_is_replaced(self):
        worker = self.pool.workers[3]
        worker.kill()
        self.pool._replace(worker, "died")
        self.assertTrue(worker.alive())


if __name__ == "__main__":
    unittest.main()
