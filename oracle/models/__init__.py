"""Scoring models, one package each, selected by ORACLE_MODEL."""
from __future__ import annotations

import importlib

MODELS = ("boltz2",)


def load(name: str):
    if name not in MODELS:
        raise SystemExit(f"ORACLE_MODEL must be one of {MODELS}, not {name!r}")
    return importlib.import_module(f"{__name__}.{name}").Model()
