"""One resident Nesso-1 shard, served over a newline JSON protocol on stdin.

The model and the residue dictionary load once per process, and each item is
featurized in memory with Nesso's own parser and dataset code.
"""
from __future__ import annotations

import gc
import json
import os
import random
import shutil
import sys
import tempfile
import time
import zlib
from pathlib import Path

# Must precede any torch/CUDA import.
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("WORKER_GPU", "0")
os.environ["HF_HUB_OFFLINE"] = "1"
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import cuequivariance_torch  # noqa: E402,F401 - refuse to start without the kernels
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

import nesso.data.yaml_input as yi  # noqa: E402
from nesso.data import const  # noqa: E402
from nesso.data.featurizer import NessoFeaturizer  # noqa: E402
from nesso.data.inference import STANDARD_AA, InferenceDataset, inference_collate  # noqa: E402
from nesso.data.types import Manifest  # noqa: E402
from nesso.main import ensure_cache, get_cache_path, resolve_model_revision  # noqa: E402
from nesso.model.models.nesso1 import Nesso1  # noqa: E402

torch.set_num_threads(1)
torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("highest")

# Fixed here so every caller scores identically; these are Nesso's CLI defaults.
PREDICT = {
    "pose_protein_cutoff": 15.0,
    "recycling_steps": 5,
    "affinity_protein_cutoff": 15.0,
    "refine_protein_inference": True,
    "refine_protein_cutoff": 22.0,
    "refine_protein_tokens_budget": 256,
    "save_metadata": False,
}

_seed = 0
_native_conformer = yi.get_conformer


def _seeded_conformer(mol):
    """Nesso's get_conformer, seeded per molecule."""
    if mol.GetNumConformers() > 0:
        return _native_conformer(mol)
    options = AllChem.ETKDGv3()
    options.clearConfs = False
    options.randomSeed = _seed
    if AllChem.EmbedMolecule(mol, options) < 0:
        options.useRandomCoords = True
        AllChem.EmbedMolecule(mol, options)
    return mol.GetConformer(0)


yi.get_conformer = _seeded_conformer

DEVICE = torch.device("cuda")
CCD_PKL, MODEL_DIR = ensure_cache(get_cache_path(), revision=resolve_model_revision(None))
# Proteins only need the standard residues; the full dictionary costs ~3 GB per shard.
_ccd = yi.load_ccd_mol_dict(CCD_PKL)
CCD = {name: _ccd[name] for name in const.ref_atoms if name in _ccd}
del _ccd
gc.collect()

MODEL = Nesso1.from_pretrained(MODEL_DIR)
MODEL.use_kernels = True
MODEL.predict_args.update(PREDICT)
MODEL.eval().to(DEVICE)

FEATURIZER = NessoFeaturizer(esm_emb_dir=Path(os.environ["NESSO_EMBEDDINGS_DIR"]),
                             esm_emb_dim=1280, esm_num_layers=33)
WORK = Path(tempfile.mkdtemp(prefix="nesso_", dir="/dev/shm"))
(WORK / "structures").mkdir()
DATASET = InferenceDataset(manifest=Manifest([]), target_dir=WORK, ligand_dir=WORK,
                           ccd_pkl=None, featurizer=FEATURIZER)
DATASET._standard_aa_mols = {aa: CCD.get(aa) for aa in STANDARD_AA}


def _features(record_id: str, sequence: str, smiles: str) -> dict:
    global _seed
    # Seeded per molecule, so its conformer and random rotation repeat on every read.
    _seed = zlib.crc32(smiles.encode("utf-8")) & 0x7FFFFFFF
    random.seed(_seed)
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    schema = {
        "version": 1,
        "sequences": [{"protein": {"id": "A", "sequence": sequence}},
                      {"ligand": {"id": "B", "smiles": smiles}}],
        "properties": [{"affinity": {"binder": "B"}}],
    }
    structure, record, _, _ = yi.parse_schema(schema, WORK, ccd_dict=CCD, record_id=record_id)
    structure.dump(WORK / "structures" / f"{record_id}.npz")
    DATASET.manifest = Manifest([record])
    features = DATASET[0]
    if features.get("exception"):
        raise ValueError("featurization failed")
    return inference_collate([features])


def _metrics(prediction: dict) -> dict:
    """What NessoWriter reports: every affinity_* and entropy_* scalar."""
    metrics = {}
    for key, value in prediction.items():
        if key.startswith(("affinity_", "entropy_")) and key != "entropy_pair":
            if isinstance(value, torch.Tensor):
                metrics[key] = float(value.item())
            elif isinstance(value, (int, float)):
                metrics[key] = float(value)
    return metrics


def score(items: list[dict]) -> tuple[dict, list[str]]:
    """items: [{"id", "sequence", "input", "smiles"}] -> ({id: metrics}, errors).

    "input" is the target's embedding, which the featurizer finds by sequence."""
    results, errors = {}, []
    for item in items:
        stage = "featurize"
        try:
            batch = _features(item["id"], item["sequence"], item["smiles"])
            stage = "predict"
            batch = {k: v.to(DEVICE, non_blocking=True) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = MODEL.predict_step(batch, 0)
            if prediction.get("exception"):
                raise RuntimeError("prediction failed")
            results[item["id"]] = _metrics(prediction)
        except Exception as exc:
            if "CUDA" in str(exc):
                raise
            errors.append(f"{item['id']} {stage}: {type(exc).__name__}: {exc}")
        finally:
            for leftover in WORK.glob(f"{item['id']}*"):
                leftover.unlink(missing_ok=True)
            (WORK / "structures" / f"{item['id']}.npz").unlink(missing_ok=True)
    return results, errors


def main() -> None:
    print(json.dumps({"event": "ready"}), flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        started = time.time()
        fault = None
        try:
            results, errors = score(json.loads(line)["items"])
        except Exception as exc:  # noqa: BLE001 - reported to the caller, not swallowed
            results, errors, fault = {}, [f"{type(exc).__name__}: {exc}"], exc
        print(json.dumps({
            "event": "result", "results": results, "error": "; ".join(errors) or None,
            "elapsed_s": round(time.time() - started, 2),
        }), flush=True)
        if fault is not None and "CUDA" in str(fault):
            sys.exit(1)  # the pool replaces a shard whose device failed


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(WORK, ignore_errors=True)
