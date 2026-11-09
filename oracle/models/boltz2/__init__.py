"""Boltz-2: shards hold both checkpoints; each target needs an alignment."""
from __future__ import annotations

from pathlib import Path

from ...config import required
from . import msa

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


class Model:
    worker = str(Path(__file__).with_name("worker.py"))
    metrics = METRICS
    # Ubiquitin, single-sequence mode: a cold start needs no network.
    probe = ("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQ"
             "KESTLHLVLRLRGG", "empty")
    # Measured on 8x RTX 4090; sizing is in the README.
    tuning = {
        "shards_per_gpu": 3,
        "stagger_s": 2,
        "request_timeout_s": 900,
        "recycle_rss_gb": 17,
        "max_batch": 512,
    }

    def __init__(self):
        self.msa_dir = Path(required("BOLTZ2_MSA_DIR"))

    def known_proteins(self) -> list[dict]:
        """Targets prepared by an earlier run."""
        return msa.resident(self.msa_dir)

    def prepare(self, sequences: list[str]) -> list[dict]:
        """Search and cache an alignment for each target; one status per input."""
        return msa.ensure(self.msa_dir, sequences)

    def target_input(self, sequence: str) -> str | None:
        """What a shard needs for this target, or None if it is not prepared."""
        path = msa.path(self.msa_dir, sequence)
        return str(path) if path.exists() else None
