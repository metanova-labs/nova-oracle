"""Nesso-1: shards hold the model; each target needs a protein embedding."""
from __future__ import annotations

from pathlib import Path

from ...config import required
from . import embeddings

METRICS = (
    "affinity_pred_value", "affinity_probability_binary",
    "affinity_pred_value1", "affinity_pred_value2", "affinity_logits_binary",
    "entropy_pp", "entropy_pl", "entropy_ll",
    "entropy_crop_pp", "entropy_crop_pl", "entropy_crop_ll",
)

# Ubiquitin.
PROBE = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


class Model:
    worker = str(Path(__file__).with_name("worker.py"))
    metrics = METRICS
    tuning = {
        "shards_per_gpu": 4,
        "stagger_s": 2,
        "request_timeout_s": 900,
        "recycle_rss_gb": 8,
        "max_batch": 512,
    }

    def __init__(self):
        self.embeddings_dir = Path(required("NESSO_EMBEDDINGS_DIR"))
        self.gpu = required("ORACLE_GPUS").split(",")[0]
        embeddings.ensure(self.embeddings_dir, [PROBE], self.gpu)
        self.probe = (PROBE, str(embeddings.path(self.embeddings_dir, PROBE)))

    def known_proteins(self) -> list[dict]:
        """Targets prepared by an earlier run."""
        return [p for p in embeddings.resident(self.embeddings_dir) if p["sequence"] != PROBE]

    def prepare(self, sequences: list[str]) -> list[dict]:
        """Compute and cache an embedding for each target; one status per input."""
        return embeddings.ensure(self.embeddings_dir, sequences, self.gpu)

    def target_input(self, sequence: str) -> str | None:
        """What a shard needs for this target, or None if it is not prepared."""
        path = embeddings.path(self.embeddings_dir, sequence)
        return str(path) if path.exists() else None
