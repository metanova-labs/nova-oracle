import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from oracle import sequence
from oracle.models import nesso
from oracle.models.nesso import embeddings

A, B = "MKTAYIAKQR", "GSHMLEDPAA"


def write(path: Path, protein: str) -> None:
    """A tensorless safetensors file carrying the sequence, as embed.py writes."""
    header = json.dumps({"__metadata__": {"sequence": protein}}).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header)


class EmbeddingsTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name)
        self.embedded: list[list[str]] = []
        self.gpus: list[str] = []

        def embed(command, input, env, **kwargs):
            batch = json.loads(input)
            self.embedded.append(batch)
            self.gpus.append(env["CUDA_VISIBLE_DEVICES"])
            for protein in batch:
                write(embeddings.path(Path(command[-1]), protein), protein)

        patch = mock.patch.object(embeddings.subprocess, "run", side_effect=embed)
        patch.start()
        self.addCleanup(patch.stop)
        self.addCleanup(self.dir.cleanup)

    def test_files_are_named_as_nessos_featurizer_looks_them_up(self):
        self.assertEqual(embeddings.path(self.path, A).name,
                         hashlib.md5(A.encode()).hexdigest() + ".safetensors")

    def test_embeds_only_what_is_missing_in_one_batch(self):
        first = embeddings.ensure(self.path, [A, B, A], "0")
        self.assertEqual(self.embedded, [[A, B]])
        self.assertEqual([s["cached"] for s in first], [False, False, False])
        second = embeddings.ensure(self.path, [B], "0")
        self.assertEqual(self.embedded, [[A, B]])
        self.assertEqual(second, [{"sequence_hash": sequence.digest(B), "residues": len(B),
                                   "depth": None, "cached": True}])

    def test_resident_reads_the_sequence_back(self):
        embeddings.ensure(self.path, [A], "0")
        self.assertEqual(embeddings.resident(self.path), [
            {"sequence_hash": sequence.digest(A), "sequence": A, "residues": len(A),
             "depth": None}])

    def test_model_prepares_its_probe_but_does_not_report_it(self):
        env = {"NESSO_EMBEDDINGS_DIR": self.dir.name, "ORACLE_GPUS": "3,4"}
        with mock.patch.dict(os.environ, env):
            model = nesso.Model()
        self.assertEqual(self.embedded, [[nesso.PROBE]])
        self.assertEqual(self.gpus, ["3"])
        self.assertTrue(Path(model.probe[1]).exists())
        self.assertEqual(model.known_proteins(), [])
        self.assertIsNone(model.target_input(A))
        model.prepare([A])
        self.assertEqual(model.target_input(A), str(embeddings.path(self.path, A)))
        self.assertEqual([p["sequence"] for p in model.known_proteins()], [A])


if __name__ == "__main__":
    unittest.main()
