"""Runtime configuration, entirely from the environment.

Machine settings have no default. Tuning settings default to the model's
measured values and can be overridden.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

TUNING = {
    "shards_per_gpu": int,
    "stagger_s": float,
    "request_timeout_s": float,
    "recycle_rss_gb": float,
    "max_batch": int,
}


def required(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise SystemExit(f"{name} is not set; copy the model's env.example to .env")
    return value


@dataclass(frozen=True)
class Config:
    model: str
    gpus: tuple[int, ...]
    host: str
    port: int
    token: str
    db_path: Path
    shards_per_gpu: int
    stagger_s: float
    request_timeout_s: float
    recycle_rss_gb: float
    max_batch: int
    overridden: tuple[str, ...] = field(default=())

    @classmethod
    def from_env(cls, tuning: dict) -> "Config":
        """tuning: the model's default for every TUNING setting."""
        values, overridden = {}, []
        for name, kind in TUNING.items():
            raw = os.environ.get(f"ORACLE_{name.upper()}", "")
            values[name] = kind(raw or tuning[name])
            if raw:
                overridden.append(name)
        return cls(
            model=required("ORACLE_MODEL"),
            gpus=tuple(int(x) for x in required("ORACLE_GPUS").split(",")),
            host=required("ORACLE_HOST"),
            port=int(required("ORACLE_PORT")),
            token=required("ORACLE_TOKEN"),
            db_path=Path(required("ORACLE_DB")),
            overridden=tuple(overridden),
            **values,
        )

    @property
    def shards(self) -> int:
        return len(self.gpus) * self.shards_per_gpu

    def describe(self) -> str:
        machine = (f"model={self.model} gpus={','.join(map(str, self.gpus))} "
                   f"host={self.host} port={self.port} db={self.db_path}")
        tuning = " ".join(
            f"{name}={getattr(self, name)}{'' if name in self.overridden else ' (default)'}"
            for name in TUNING)
        return f"{machine} {tuning}"
