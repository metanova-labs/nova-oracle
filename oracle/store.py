"""Audit trail: who asked, in which epoch, which molecule against which target,
and every score Boltz returned. One writer, so SQLite; WAL so reads never
block it."""
from __future__ import annotations

import sqlite3
import threading
import time

# Every scalar Boltz reports, plus the three the worker flattens out of its
# two per-chain maps. The rest of those maps duplicate columns already here.
METRICS = (
    "affinity_pred_value", "affinity_probability_binary",
    "affinity_pred_value1", "affinity_probability_binary1",
    "affinity_pred_value2", "affinity_probability_binary2",
    "confidence_score", "ptm", "iptm", "ligand_iptm", "protein_iptm",
    "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde",
    "chains_ptm_protein", "chains_ptm_ligand", "pair_chains_iptm_protein_ligand",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS proteins (
  sequence_hash TEXT PRIMARY KEY,
  sequence      TEXT NOT NULL,
  residues      INTEGER NOT NULL,
  msa_depth     INTEGER,
  label         TEXT,           -- caller-supplied, e.g. a UniProt accession
  first_seen    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS requests (
  request_id  TEXT PRIMARY KEY,
  ts          REAL NOT NULL,
  miner       TEXT NOT NULL,
  epoch       TEXT,
  molecules   INTEGER NOT NULL,
  targets     INTEGER NOT NULL,
  predictions INTEGER NOT NULL,
  shards      INTEGER NOT NULL,
  elapsed_s   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
  id         INTEGER PRIMARY KEY,
  ts         REAL NOT NULL,
  request_id TEXT NOT NULL REFERENCES requests(request_id),
  miner      TEXT NOT NULL,
  epoch      TEXT,
  target     TEXT NOT NULL REFERENCES proteins(sequence_hash),
  smiles     TEXT NOT NULL,
  gpu        INTEGER,
  %s
);

CREATE INDEX IF NOT EXISTS ix_pred_mol   ON predictions(target, smiles);
CREATE INDEX IF NOT EXISTS ix_pred_miner ON predictions(miner, epoch);
CREATE INDEX IF NOT EXISTS ix_pred_ts    ON predictions(ts);
CREATE INDEX IF NOT EXISTS ix_req_miner  ON requests(miner, epoch);
""" % ",\n  ".join(f"{m:<28} REAL" for m in METRICS)

_INSERT = (
    "INSERT INTO predictions (ts,request_id,miner,epoch,target,smiles,gpu,"
    + ",".join(METRICS) + ") VALUES (" + ",".join("?" * (7 + len(METRICS))) + ")")


class Store:
    def __init__(self, path: str):
        self.lock = threading.Lock()
        self.con = sqlite3.connect(path, check_same_thread=False)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA synchronous=NORMAL")
        self.con.execute("PRAGMA foreign_keys=ON")
        self.con.executescript(_SCHEMA)
        self.con.commit()

    def record_proteins(self, rows: list[tuple]) -> None:
        """rows: (sequence_hash, sequence, residues, msa_depth, label).

        Re-registering keeps first_seen and only overwrites a label if given."""
        now = time.time()
        with self.lock:
            self.con.executemany(
                "INSERT INTO proteins (sequence_hash,sequence,residues,msa_depth,"
                "label,first_seen) VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(sequence_hash) DO UPDATE SET "
                "label=COALESCE(excluded.label, proteins.label)",
                [(*row, now) for row in rows])
            self.con.commit()

    def record(self, request: dict, rows: list[tuple]) -> int:
        """rows: (target, smiles, gpu, metrics), one per (molecule, target)."""
        now = time.time()
        values = [
            (now, request["request_id"], request["miner"], request["epoch"],
             target, smiles, gpu, *((metrics or {}).get(m) for m in METRICS))
            for target, smiles, gpu, metrics in rows]
        with self.lock:
            self.con.execute(
                "INSERT INTO requests (request_id,ts,miner,epoch,molecules,targets,"
                "predictions,shards,elapsed_s) VALUES (?,?,?,?,?,?,?,?,?)",
                (request["request_id"], now, request["miner"], request["epoch"],
                 request["molecules"], request["targets"], len(rows),
                 request["shards"], request["elapsed_s"]))
            self.con.executemany(_INSERT, values)
            self.con.commit()
        return len(values)
