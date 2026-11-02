import importlib
import json
import os
import socket
import sqlite3
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest import mock

import uvicorn

from oracle import sequence
from oracle.models import boltz2

FAKE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_worker.py")
A, B = "MKTAYIAKQR", "GSHMLEDPAA"


class FakePreparation:
    """Stands in for Boltz-2's alignment search."""

    def __init__(self):
        self.prepared = set()

    def prepare(self, sequences):
        statuses = [{"sequence_hash": sequence.digest(s), "residues": len(s), "depth": 1,
                     "cached": s in self.prepared} for s in sequences]
        self.prepared.update(sequences)
        return statuses

    def target_input(self, target):
        return "input" if target in self.prepared else None


class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        cls.base = f"http://127.0.0.1:{port}"
        cls.db = os.path.join(cls.dir.name, "oracle.db")
        fake = FakePreparation()
        cls.patches = [
            mock.patch.dict(os.environ, {
                "ORACLE_MODEL": "boltz2", "ORACLE_GPUS": "0,1", "ORACLE_HOST": "127.0.0.1",
                "ORACLE_PORT": str(port), "ORACLE_TOKEN": "t", "ORACLE_DB": cls.db,
                "BOLTZ2_MSA_DIR": cls.dir.name, "ORACLE_SHARDS_PER_GPU": "1",
                "ORACLE_STAGGER_S": "0", "ORACLE_MAX_BATCH": "8"}),
            mock.patch.object(boltz2.Model, "worker", FAKE),
            mock.patch.object(boltz2.Model, "known_proteins", lambda self: []),
            mock.patch.object(boltz2.Model, "prepare", fake.prepare),
            mock.patch.object(boltz2.Model, "target_input", fake.target_input),
        ]
        for patch in cls.patches:
            patch.start()
        server = importlib.reload(importlib.import_module("oracle.server"))
        cls.server = uvicorn.Server(uvicorn.Config(server.app, port=port, log_level="warning"))
        cls.thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.thread.start()
        deadline = time.time() + 30
        while not cls.server.started and time.time() < deadline:
            time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True
        cls.thread.join(timeout=30)
        for patch in reversed(cls.patches):
            patch.stop()
        cls.dir.cleanup()

    def call(self, path, body=None, token="t"):
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(self.base + path, data, headers)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read() or b"{}")

    def test_auth(self):
        self.assertEqual(self.call("/v1/score", {}, token=None)[0], 401)
        self.assertEqual(self.call("/v1/score", {}, token="wrong")[0], 401)
        self.assertEqual(self.call("/health", token=None)[0], 200)

    def test_limits(self):
        score = {"miner": "m", "targets": [A]}
        self.assertEqual(self.call("/v1/score", score | {"smiles": ["C" * 2001]})[0], 422)
        self.assertEqual(self.call("/v1/score", score | {"smiles": ["C"] * 10001})[0], 422)
        self.assertEqual(self.call("/v1/proteins", {"sequences": ["not a protein"]})[0], 422)

    def test_register_then_score(self):
        score = {"miner": "m", "epoch": "1", "targets": [A, B], "smiles": ["C", "CCO", "c1ccccc1"]}
        self.assertEqual(self.call("/v1/score", score)[0], 409)

        code, body = self.call("/v1/proteins", {"sequences": [A, B]})
        self.assertEqual((code, [p["cached"] for p in body["proteins"]]), (200, [False, False]))
        code, body = self.call("/v1/proteins", {"sequences": [A, B]})
        self.assertEqual([p["cached"] for p in body["proteins"]], [True, True])

        code, body = self.call("/v1/score", score)
        self.assertEqual((code, body["predictions"]), (200, 6))
        self.assertEqual([[s["affinity_pred_value"] for s in r["scores"]] for r in body["results"]],
                         [[1.0, 1.0], [3.0, 3.0], [8.0, 8.0]])
        self.assertEqual(self.call("/v1/score", score | {"smiles": ["C"] * 5})[0], 413)

        rows = sqlite3.connect(self.db).execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        self.assertEqual(rows, 6)


if __name__ == "__main__":
    unittest.main()
