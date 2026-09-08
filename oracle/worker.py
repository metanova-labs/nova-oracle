"""One resident Boltz-2 shard, served over a newline JSON protocol on stdin.

Caching the checkpoint loads is what makes an interactive API possible: it
turns a 117.6 s per-call fixed cost into roughly zero. Items carry their own
sequence and alignment path, so a batch may span several targets.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Must precede any torch/CUDA import.
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("WORKER_GPU", "0")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import yaml  # noqa: E402
import boltz.main as bm  # noqa: E402
from boltz.model.models.boltz2 import Boltz2  # noqa: E402

_checkpoints: dict[str, object] = {}
_canonicals: dict[str, object] = {}
_load_checkpoint = Boltz2.load_from_checkpoint
_load_canonicals = bm.load_canonicals


def _cached_checkpoint(path, **kwargs):
    key = str(path)
    if key not in _checkpoints:
        _checkpoints[key] = _load_checkpoint(path, **kwargs)
    return _checkpoints[key]


def _cached_canonicals(mol_dir):
    key = str(mol_dir)
    if key not in _canonicals:
        _canonicals[key] = _load_canonicals(mol_dir)
    return _canonicals[key]


Boltz2.load_from_checkpoint = staticmethod(_cached_checkpoint)
bm.load_canonicals = _cached_canonicals

# Fixed here so every caller scores identically, and deliberately not boltz's
# CLI defaults: step counts halved from 200, molecular-weight correction on,
# seed pinned where boltz leaves it random.
PREDICT = {
    "recycling_steps": 3,
    "sampling_steps": 100,
    "diffusion_samples": 1,
    "sampling_steps_affinity": 100,
    "diffusion_samples_affinity": 3,
    "affinity_mw_correction": True,
    "output_format": "mmcif",
    "seed": 68,
}


def score(items: list[dict]) -> dict:
    """items: [{"id", "sequence", "msa", "smiles"}] -> {id: metrics}.

    "msa" is a path to an .a3m, or "empty" for single-sequence mode."""
    work = Path(tempfile.mkdtemp(prefix="oracle_", dir="/dev/shm"))
    inputs, outputs = work / "inputs", work / "out"
    inputs.mkdir(parents=True)
    outputs.mkdir(parents=True)
    try:
        for item in items:
            document = {
                "version": 1,
                "sequences": [
                    {"protein": {"id": "A", "sequence": item["sequence"],
                                 "msa": item["msa"]}},
                    {"ligand": {"id": "B", "smiles": item["smiles"]}},
                ],
                "properties": [{"affinity": {"binder": "B"}}],
            }
            (inputs / f"{item['id']}.yaml").write_text(
                yaml.safe_dump(document, sort_keys=False))

        bm.predict(data=str(inputs), out_dir=str(outputs),
                   override=True, num_workers=0, **PREDICT)

        results = {}
        for directory in (outputs / "boltz_results_inputs" / "predictions").glob("*"):
            metrics = {}
            for f in directory.glob("*.json"):
                if f.name.startswith(("affinity", "confidence")):
                    metrics.update(json.loads(f.read_text()))
            results[directory.name] = _flatten(metrics)
        return results
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _flatten(metrics: dict) -> dict:
    """Reduce boltz's two per-chain maps to the values they alone carry.

    Chain 0 is the protein and chain 1 the ligand, fixed by the document built
    above. The pair matrix is asymmetric; of its four entries the diagonal
    repeats chains_ptm and [1][0] repeats ligand_iptm, so only [0][1] is new.
    """
    chains = metrics.pop("chains_ptm", None) or {}
    pairs = metrics.pop("pair_chains_iptm", None) or {}
    metrics["chains_ptm_protein"] = chains.get("0")
    metrics["chains_ptm_ligand"] = chains.get("1")
    metrics["pair_chains_iptm_protein_ligand"] = (pairs.get("0") or {}).get("1")
    return metrics


def main() -> None:
    print(json.dumps({"event": "ready"}), flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        started = time.time()
        try:
            results, error = score(json.loads(line)["items"]), None
        except Exception as exc:  # noqa: BLE001 - reported to the caller, not swallowed
            results, error = {}, f"{type(exc).__name__}: {exc}"
        print(json.dumps({
            "event": "result", "results": results, "error": error,
            "elapsed_s": round(time.time() - started, 2),
        }), flush=True)


if __name__ == "__main__":
    main()
