# nova-oracle

Boltz-2 protein–ligand affinity scoring as a service, for NOVA (Bittensor
subnet 68).

Each **shard** is a worker process holding both Boltz-2 checkpoints resident,
so a request costs compute only. Every request uses the whole pool, and Boltz's
dataloader is `batch_size=1`, so a shard scores serially and **the caller's
batch size is the throughput lever**. Proteins are addressed by sequence and
their alignments cached under a digest of it; a round is prepared once, then
scored against.

## Quickstart

```bash
./deploy/provision.sh          # bare GPU box -> ready, ~4 min
cp .env.example .env           # then edit
./deploy/run.sh                # start (or restart)
python3 tests/smoke.py         # verify every route
```

Startup is ~4 min for 24 shards and is paid once.

## API

Every route requires `Authorization: Bearer $ORACLE_TOKEN` except `/health`.

### `POST /v1/proteins`

Caches each target's alignment and warms the shards. Idempotent and cheap once
cached, so run it at the top of every round. An uncached sequence costs an MSA
search of tens of seconds.

| field | type | constraint |
|---|---|---|
| `sequences` | `[str]` | 1–16 items, 1–4096 residues in `ACDEFGHIKLMNPQRSTVWYXUBZO` |
| `labels` | `[str]?` | optional, positional — e.g. a UniProt accession, recorded against the protein |

```json
{"proteins": [{"sequence_hash": "d1ebf10f3b117814", "residues": 258,
               "depth": 3048, "cached": true}],
 "warm_s": 31.0, "elapsed_s": 31.2}
```

### `POST /v1/score`

| field | type | constraint |
|---|---|---|
| `miner` | `str` | 1–128 chars. Audit label — **not** authorization |
| `targets` | `[str]` | 1–8 sequences, each already registered |
| `smiles` | `[str]` | ≥1; `len(targets) × len(smiles)` ≤ `ORACLE_MAX_BATCH` |
| `epoch` | `str?` | optional round label, stored with each prediction |

Results are positional: `results[i].scores[t]` is `smiles[i]` against
`targets[t]`, `null` where that prediction failed. `predictions` counts what was
delivered, which is lower than requested if a shard failed.

```json
{"request_id": "dd76e017e09b", "elapsed_s": 17.5, "predictions": 4,
 "targets": ["d1ebf10f3b117814", "e6a24c903c1ef211"],
 "results": [{"smiles": "CC(=O)Nc1nnc(s1)S(N)(=O)=O",
              "scores": [{"affinity_pred_value": 1.04, "...": "..."},
                         {"affinity_pred_value": -1.94, "...": "..."}]}]}
```

Each score carries Boltz's metrics verbatim: `affinity_pred_value` and
`affinity_probability_binary` (plus their `1`/`2` ensemble members),
`confidence_score`, `ptm`, `iptm`, `ligand_iptm`, `protein_iptm`,
`complex_plddt`, `complex_iplddt`, `complex_pde`, `complex_ipde`, and the three
per-chain values flattened from its `chains_ptm` / `pair_chains_iptm` maps:
`chains_ptm_protein`, `chains_ptm_ligand`, `pair_chains_iptm_protein_ligand`.

### `GET /health`

No token — it is what a caller checks before it has a round to run.

```json
{"alive": 24, "total": 24, "gpus": [0,1,2,3,4,5,6,7], "rss_gb": 181.3}
```

### Errors

| code | meaning |
|---|---|
| `401` | missing or wrong bearer token |
| `409` | a target has no alignment — register it first |
| `413` | more predictions than `ORACLE_MAX_BATCH` |
| `422` | malformed body, or a sequence that is not residue codes |
| `502` | the MSA server could not be reached |
| `503` | no live shards (score), or a degraded pool (health) |

## Scoring settings

Fixed in `oracle/worker.py` (`PREDICT`), not per request: callers scoring under
different settings produce numbers that cannot be compared. Three are
deliberately not boltz's CLI defaults, so results are not interchangeable with
stock `boltz predict`:

| setting | here | boltz default |
|---|---|---|
| `sampling_steps` | 100 | 200 |
| `sampling_steps_affinity` | 100 | 200 |
| `affinity_mw_correction` | `True` | `False` |

`seed` is pinned where boltz leaves it random, which narrows variance but does
not remove it — output stays stochastic across batch compositions.

## Performance

8× RTX 4090, 3 shards/GPU, 258-residue target:

| batch | per shard | round-trip | s/molecule | mol/hr |
|------:|----------:|-----------:|-----------:|-------:|
| 24  | 1  | 32.1 s | 1.34 | 2,695 |
| 48  | 2  | 55.4 s | 1.15 | 3,120 |
| 96  | 4  | 101.5 s | 1.06 | 3,404 |
| 192 | 8  | 196.8 s | 1.02 | 3,513 |
| 384 | 16 | 380.8 s | 0.99 | 3,630 |

**Submit 4–8× the shard count** (96–192 here). The knee is at 4; a batch of 24
leaves 26% on the table and a single molecule costs ~17 s however many shards
sit idle.

Cost rises with target length — at batch 192, a 258-residue target runs 3,570
mol/hr and a 400-residue one (the NOVA dataset maximum) runs 2,515, so plan for
**2,500–3,600**. A 62-heavy-atom ligand costs ~25% more than a 13-atom one.

Concurrent requests queue per shard and give no throughput gain; the GPUs are
already saturated. Prefer one deep request at a time.

## Configuration

All settings come from the environment; `deploy/run.sh` sources `.env`.

| variable | default | notes |
|---|---|---|
| `ORACLE_GPUS` | `0` | comma-separated GPU ids |
| `ORACLE_SHARDS_PER_GPU` | `3` | see sizing below |
| `ORACLE_STAGGER_S` | `2` | spawn spacing |
| `ORACLE_HOST` / `ORACLE_PORT` | `127.0.0.1` / `8000` | loopback by default; terminate TLS at a proxy. Set a routable address only when the proxy cannot reach loopback |
| `ORACLE_TOKEN` | — | bearer token; required in production |
| `ORACLE_DB` | `oracle.db` | prediction store |
| `ORACLE_MSA_DIR` | `msa` | alignment cache, one `.a3m` per sequence digest |
| `ORACLE_RECYCLE_RSS_GB` | `17` | retire a shard above this |
| `ORACLE_REQUEST_TIMEOUT_S` | `900` | per-shard call timeout |
| `ORACLE_MAX_BATCH` | `256` | predictions per request |

Size `ORACLE_SHARDS_PER_GPU` by the lower of `hostRAM / 16 / nGPUs` and
`VRAM / 6` — a shard's RSS plateaus near 14 GB and its peak reserved VRAM is
5.7 GB. `ORACLE_RECYCLE_RSS_GB` must exceed that plateau, or shards are rebuilt
forever for no reason.

## The store

Every prediction is recorded: which miner asked, in which epoch, which molecule
against which target, and all eighteen scores.

| table | holds |
|---|---|
| `proteins` | digest, full sequence, residues, MSA depth, optional label |
| `requests` | miner, epoch, counts, shards, wall time |
| `predictions` | one row per (molecule, target), each metric its own column |

```bash
tail -f oracle.log
sqlite3 oracle.db 'SELECT miner, COUNT(*) FROM predictions GROUP BY miner'
```

## Requirements

Python ≥3.10, plus `boltz` and a CUDA-matched torch build installed separately;
`deploy/provision.sh` handles both. Alignment generation calls the ColabFold
MSA server, so the box needs outbound network on first use of a new sequence.

## License

MIT — see [LICENSE](LICENSE).
