"""Sequence -> alignment, content addressed and never expiring.

Alignments come from boltz's own ColabFold client. Boltz only pairs MSAs for
multi-chain complexes; every request here is one protein and one ligand, so
the unpaired search is the whole job.
"""
from __future__ import annotations

import hashlib
import logging
import tempfile
from pathlib import Path

log = logging.getLogger("oracle.msa")

# The 20 standard residues plus the ambiguity codes UniProt actually emits.
RESIDUES = set("ACDEFGHIKLMNPQRSTVWYXUBZO")
MAX_RESIDUES = 4096


def digest(sequence: str) -> str:
    return hashlib.sha256(sequence.encode()).hexdigest()[:16]


def path(msa_dir: Path, sequence: str) -> Path:
    return Path(msa_dir) / f"{digest(sequence)}.a3m"


def invalid(sequence: str) -> str | None:
    """Why this is not a protein sequence, or None if it is one."""
    if not sequence:
        return "empty sequence"
    if len(sequence) > MAX_RESIDUES:
        return f"{len(sequence)} residues exceeds {MAX_RESIDUES}"
    bad = sorted(set(sequence) - RESIDUES)
    if bad:
        return f"not residue codes: {''.join(bad)}"
    return None


def describe(a3m: Path) -> dict:
    depth, query = 0, ""
    with a3m.open() as handle:
        for line in handle:
            if line.startswith(">"):
                depth += 1
            elif not query and not line.startswith("#"):
                query = line.strip()
    return {"sequence_hash": a3m.stem, "sequence": query,
            "residues": len(query), "depth": depth}


def resident(msa_dir: Path) -> list[dict]:
    """Every alignment on disk; the store is backfilled from this at startup."""
    return [describe(p) for p in sorted(Path(msa_dir).glob("*.a3m"))]


def ensure(msa_dir: Path, sequences: list[str]) -> list[dict]:
    """Make an MSA resident for each sequence; one status per input.

    Cached sequences cost nothing; missing ones go out in one batched search."""
    msa_dir = Path(msa_dir)
    msa_dir.mkdir(parents=True, exist_ok=True)

    missing = [s for s in dict.fromkeys(sequences) if not path(msa_dir, s).exists()]
    if missing:
        log.info("searching %d sequence(s) against the msa server", len(missing))
        _search(msa_dir, missing)

    return [{"sequence_hash": d["sequence_hash"], "residues": d["residues"],
             "depth": d["depth"], "cached": s not in missing}
            for s, d in ((s, describe(path(msa_dir, s))) for s in sequences)]


def _search(msa_dir: Path, sequences: list[str]) -> None:
    """Published atomically, so an interrupted search leaves no partial file."""
    # Local import: the server process has no other reason to pull in torch.
    from boltz.data.msa.mmseqs2 import run_mmseqs2

    with tempfile.TemporaryDirectory(prefix="msa_") as scratch:
        # One a3m per query, in order. The type annotation claims a tuple;
        # boltz's own callers take the list, and so do we.
        alignments = run_mmseqs2(
            sequences, Path(scratch) / "search", use_env=True, use_pairing=False)
        for sequence, a3m in zip(sequences, alignments):
            staged = Path(scratch) / f"{digest(sequence)}.a3m"
            staged.write_text(a3m)
            staged.replace(path(msa_dir, sequence))

