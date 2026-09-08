"""Runtime configuration, entirely from the environment."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _ints(name: str, default: str) -> tuple[int, ...]:
    return tuple(int(x) for x in os.environ.get(name, default).split(",") if x != "")


@dataclass(frozen=True)
class Config:
    gpus: tuple[int, ...]
    shards_per_gpu: int
    stagger_s: float
    host: str
    port: int
    token: str
    db_path: Path
    msa_dir: Path
    request_timeout_s: float
    recycle_rss_gb: float
    max_batch: int

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            gpus=_ints("ORACLE_GPUS", "0"),
            # Bounded by host RAM (~14 GB/shard) or VRAM (~5.7 GB/shard).
            shards_per_gpu=int(os.environ.get("ORACLE_SHARDS_PER_GPU", "3")),
            stagger_s=float(os.environ.get("ORACLE_STAGGER_S", "2")),
            host=os.environ.get("ORACLE_HOST", "127.0.0.1"),
            port=int(os.environ.get("ORACLE_PORT", "8000")),
            token=os.environ.get("ORACLE_TOKEN", ""),
            db_path=Path(os.environ.get("ORACLE_DB", "oracle.db")),
            msa_dir=Path(os.environ.get("ORACLE_MSA_DIR", "msa")),
            request_timeout_s=float(os.environ.get("ORACLE_REQUEST_TIMEOUT_S", "900")),
            # Must exceed the ~14 GB plateau shard RSS settles at, or shards
            # are rebuilt forever; and stay under hostRAM/shards.
            recycle_rss_gb=float(os.environ.get("ORACLE_RECYCLE_RSS_GB", "17")),
            max_batch=int(os.environ.get("ORACLE_MAX_BATCH", "256")),
        )

    @property
    def shards(self) -> int:
        return len(self.gpus) * self.shards_per_gpu
