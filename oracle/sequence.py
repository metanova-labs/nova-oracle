"""Protein sequences: validation, and the digest they are addressed by."""
from __future__ import annotations

import hashlib

# The 20 standard residues plus the ambiguity codes UniProt actually emits.
RESIDUES = set("ACDEFGHIKLMNPQRSTVWYXUBZO")
MAX_RESIDUES = 4096


def digest(sequence: str) -> str:
    return hashlib.sha256(sequence.encode()).hexdigest()[:16]


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
