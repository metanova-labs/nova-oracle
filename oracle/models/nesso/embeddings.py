"""Sequence -> ESM-2 embedding, computed once and shared by every shard.

Files are named as Nesso's featurizer looks them up, by the sequence's MD5, and
carry the sequence in their metadata.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from ...sequence import digest

log = logging.getLogger("oracle.embeddings")

EMBED = Path(__file__).with_name("embed.py")


def path(embeddings_dir: Path, sequence: str) -> Path:
    name = hashlib.md5(sequence.encode("utf-8")).hexdigest()
    return Path(embeddings_dir) / f"{name}.safetensors"


def describe(embedding: Path) -> dict:
    with embedding.open("rb") as handle:
        header = json.loads(handle.read(int.from_bytes(handle.read(8), "little")))
    sequence = header["__metadata__"]["sequence"]
    return {"sequence_hash": digest(sequence), "sequence": sequence,
            "residues": len(sequence), "depth": None}


def resident(embeddings_dir: Path) -> list[dict]:
    return [describe(p) for p in sorted(Path(embeddings_dir).glob("*.safetensors"))]


def ensure(embeddings_dir: Path, sequences: list[str], gpu: str) -> list[dict]:
    """Make an embedding resident for each sequence; one status per input."""
    embeddings_dir = Path(embeddings_dir)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    missing = [s for s in dict.fromkeys(sequences) if not path(embeddings_dir, s).exists()]
    if missing:
        log.info("embedding %d sequence(s) on gpu%s", len(missing), gpu)
        subprocess.run(
            [sys.executable, "-u", str(EMBED), str(embeddings_dir)],
            input=json.dumps(missing), text=True, check=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu))

    return [{"sequence_hash": digest(s), "residues": len(s), "depth": None,
             "cached": s not in missing}
            for s in sequences]
