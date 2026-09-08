"""Boltz-2 affinity scoring over a pool of resident shards.

One process per shard, each holding both checkpoints. Every request uses the
whole pool and no per-caller state is kept, so throughput is set by how deeply
callers batch. Proteins are addressed by sequence; a round is prepared by
POSTing its targets, which caches their alignments and warms the shards.
"""
from __future__ import annotations

import concurrent.futures as futures
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import AfterValidator, BaseModel, Field, model_validator

from . import msa
from . import __version__
from .config import Config
from .pool import Pool
from .store import Store

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("oracle")

CFG = Config.from_env()


class State:
    pool: Pool
    store: Store


S = State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("starting %d shards on gpus %s", CFG.shards, list(CFG.gpus))
    started = time.time()
    S.pool = Pool(CFG)
    log.info("shards spawned in %.0fs; warming", time.time() - started)
    log.info("warmed in %.0fs", S.pool.warm())
    S.store = Store(str(CFG.db_path))
    S.store.record_proteins([
        (p["sequence_hash"], p["sequence"], p["residues"], p["depth"], None)
        for p in msa.resident(CFG.msa_dir)])
    yield
    S.pool.shutdown()


app = FastAPI(title="nova-oracle", version=__version__, lifespan=lifespan)


@app.middleware("http")
async def authorize(request: Request, call_next):
    # Network policy belongs at the edge; this only checks the token.
    if request.url.path != "/health":
        if CFG.token and request.headers.get("authorization") != f"Bearer {CFG.token}":
            return JSONResponse({"detail": "bad token"}, status_code=401)
    return await call_next(request)


def _residues(sequences: list[str]) -> list[str]:
    for sequence in sequences:
        why = msa.invalid(sequence)
        if why:
            raise ValueError(why)
    return sequences


# One-letter codes. Targets are given in full on every request so a stored
# prediction identifies its own input.
Sequences = Annotated[list[str], AfterValidator(_residues)]


class ProteinRequest(BaseModel):
    sequences: Sequences = Field(..., min_length=1, max_length=16)
    # Positional against `sequences`; the oracle only ever sees the sequence.
    labels: list[str] | None = None

    @model_validator(mode="after")
    def _labels_align(self):
        if self.labels is not None and len(self.labels) != len(self.sequences):
            raise ValueError("labels must be the same length as sequences")
        return self


class ScoreRequest(BaseModel):
    miner: str = Field(..., min_length=1, max_length=128)
    targets: Sequences = Field(..., min_length=1, max_length=8)
    smiles: list[str] = Field(..., min_length=1)
    epoch: str | None = None


@app.post("/v1/proteins")
async def register_proteins(req: ProteinRequest):
    """Cache every target's alignment, then warm the shards on the first.

    Idempotent and cheap once cached, so callers run it at the top of every
    round. An uncached sequence costs an MSA search of tens of seconds."""
    started = time.time()
    try:
        proteins = await anyio.to_thread.run_sync(
            msa.ensure, CFG.msa_dir, req.sequences)
    except Exception as exc:  # noqa: BLE001 - an upstream failure, not a bad request
        log.warning("alignment search failed: %s", exc)
        raise HTTPException(
            502, f"alignment search failed: {type(exc).__name__}: {exc}") from exc
    labels = req.labels or [None] * len(req.sequences)
    S.store.record_proteins([
        (p["sequence_hash"], sequence, p["residues"], p["depth"], label)
        for p, sequence, label in zip(proteins, req.sequences, labels)])
    warm_s = await anyio.to_thread.run_sync(
        S.pool.warm, req.sequences[0], str(msa.path(CFG.msa_dir, req.sequences[0])))
    log.info("proteins n=%d built=%d warm=%.0fs total=%.0fs", len(proteins),
             sum(1 for p in proteins if not p["cached"]), warm_s,
             time.time() - started)
    return {"proteins": proteins, "warm_s": round(warm_s, 1),
            "elapsed_s": round(time.time() - started, 1)}


@app.post("/v1/score")
async def score(req: ScoreRequest):
    alignments = [msa.path(CFG.msa_dir, target) for target in req.targets]
    missing = [msa.digest(t) for t, a in zip(req.targets, alignments)
               if not a.exists()]
    if missing:
        raise HTTPException(
            409, f"no alignment for {missing}; register with POST /v1/proteins")

    # One prediction is one (molecule, target) pair -- the unit of GPU work.
    units = [{"id": f"q{i:05d}t{t}", "sequence": target, "msa": str(alignment),
              "smiles": smi}
             for i, smi in enumerate(req.smiles)
             for t, (target, alignment) in enumerate(zip(req.targets, alignments))]
    if len(units) > CFG.max_batch:
        raise HTTPException(413, f"{len(units)} predictions exceeds {CFG.max_batch}")

    workers = S.pool.live()
    if not workers:
        raise HTTPException(503, "no live shards")

    request_id = uuid.uuid4().hex[:12]
    started = time.time()
    scored = await anyio.to_thread.run_sync(_dispatch, workers, units, request_id)
    elapsed = time.time() - started

    # Positional: results[i].scores[t] is smiles[i] against targets[t], null
    # where that prediction failed. Sequences are long, so we return digests.
    digests = [msa.digest(target) for target in req.targets]
    results, rows = [], []
    for i, smi in enumerate(req.smiles):
        scores = []
        for t, target_digest in enumerate(digests):
            hit = scored.get(f"q{i:05d}t{t}")
            scores.append(None if hit is None else hit[1])
            if hit is not None:
                rows.append((target_digest, smi, hit[0], hit[1]))
        results.append({"smiles": smi, "scores": scores})

    S.store.record({"request_id": request_id, "miner": req.miner,
                    "epoch": req.epoch, "molecules": len(req.smiles),
                    "targets": len(req.targets), "shards": len(workers),
                    "elapsed_s": elapsed}, rows)
    log.info("%s miner=%s epoch=%s targets=%d molecules=%d predictions=%d/%d "
             "shards=%d %.1fs", request_id, req.miner, req.epoch,
             len(req.targets), len(req.smiles), len(rows), len(units),
             len(workers), elapsed)
    return {"request_id": request_id, "elapsed_s": round(elapsed, 2),
            "predictions": len(rows), "targets": digests, "results": results}


def _dispatch(workers, units, request_id) -> dict:
    """One blocking call per shard, round-robin over a GPU-interleaved list."""
    chunks: list[list[dict]] = [[] for _ in workers]
    for i, unit in enumerate(units):
        chunks[i % len(workers)].append(unit)

    def run(worker, batch) -> dict:
        if not batch:
            return {}
        with worker.lock:
            try:
                message = worker.score(batch, CFG.request_timeout_s)
            except Exception as exc:  # noqa: BLE001 - one shard, not the request
                log.warning("%s shard %d failed: %s", request_id, worker.wid, exc)
                worker.kill()
                return {}
        return {name: (worker.gpu, metrics)
                for name, metrics in (message.get("results") or {}).items()}

    scored: dict[str, tuple[int, dict]] = {}
    with futures.ThreadPoolExecutor(max_workers=len(workers)) as pool:
        for part in pool.map(run, workers, chunks):
            scored.update(part)
    return scored


@app.get("/health")
def health():
    status = S.pool.health()
    if status["alive"] < status["total"]:
        raise HTTPException(503, status)
    return status
