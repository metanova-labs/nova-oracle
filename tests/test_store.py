import os
import sqlite3
import tempfile
import unittest

from oracle.store import Store

REQUEST = {"request_id": "r1", "miner": "m", "epoch": "1", "molecules": 1,
           "targets": 1, "shards": 1, "elapsed_s": 0.1}


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "oracle.db")

    def tearDown(self):
        self.dir.cleanup()

    def test_records_one_column_per_metric(self):
        store = Store(self.path, ("a", "b"))
        store.record_proteins([("h", "SEQ", 3, None, "label")])
        store.record(REQUEST, [("h", "C", 0, {"a": 1.0, "b": 2.0})])
        row = sqlite3.connect(self.path).execute("SELECT smiles, a, b FROM predictions").fetchone()
        self.assertEqual(row, ("C", 1.0, 2.0))

    def test_refuses_another_models_database(self):
        Store(self.path, ("a", "b"))
        with self.assertRaises(RuntimeError):
            Store(self.path, ("a", "c"))


if __name__ == "__main__":
    unittest.main()
