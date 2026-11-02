# nova-oracle

Protein–ligand affinity scoring as a service, for NOVA (Bittensor subnet 68).

Each **shard** is a worker process holding the model resident, so a request
costs compute only. Every request uses the whole pool. Proteins are addressed by
sequence; a round is prepared once, then scored against.

| model | `ORACLE_MODEL` | |
|---|---|---|
| Boltz-2 | `boltz2` | [setup, scores, performance](oracle/models/boltz2/README.md) |

## Quickstart

```bash
./deploy/provision.sh boltz2   # bare GPU box -> ready
cp .env.example .env           # then edit
./deploy/run.sh                # start (or restart)
python3 tests/smoke.py         # verify every route
```

Startup loads every shard and is paid once.

## API

Every route requires `Authorization: Bearer $ORACLE_TOKEN` except `/health`.

### `POST /v1/proteins`

Prepares each target for the model and warms the shards. Idempotent and cheap
once prepared, so run it at the top of every round.

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

Each score carries the model's metrics; see its README.

### `GET /health`

No token — it is what a caller checks before it has a round to run.

```json
{"alive": 24, "total": 24, "gpus": [0,1,2,3,4,5,6,7], "rss_gb": 181.3}
```

### Errors

| code | meaning |
|---|---|
| `401` | missing or wrong bearer token |
| `409` | a target is not prepared — register it first |
| `413` | more predictions than `ORACLE_MAX_BATCH` |
| `422` | malformed body, or a sequence that is not residue codes |
| `502` | preparing a target failed upstream |
| `503` | no live shards (score), or a degraded pool (health) |

## Configuration

All settings come from the environment; `deploy/run.sh` sources `.env`.

| variable | default | notes |
|---|---|---|
| `ORACLE_MODEL` | `boltz2` | which model to serve |
| `ORACLE_GPUS` | `0` | comma-separated GPU ids |
| `ORACLE_SHARDS_PER_GPU` | `3` | sizing is in the model's README |
| `ORACLE_STAGGER_S` | `2` | spawn spacing |
| `ORACLE_HOST` / `ORACLE_PORT` | `127.0.0.1` / `8000` | loopback by default; terminate TLS at a proxy. Set a routable address only when the proxy cannot reach loopback |
| `ORACLE_TOKEN` | — | bearer token; required in production |
| `ORACLE_DB` | `oracle.db` | prediction store |
| `ORACLE_RECYCLE_RSS_GB` | `17` | retire a shard above this |
| `ORACLE_REQUEST_TIMEOUT_S` | `900` | per-shard call timeout |
| `ORACLE_MAX_BATCH` | `256` | predictions per request |

## The store

Every prediction is recorded: which miner asked, in which epoch, which molecule
against which target, and every metric the model returned.

| table | holds |
|---|---|
| `proteins` | digest, full sequence, residues, MSA depth (Boltz-2), optional label |
| `requests` | miner, epoch, counts, shards, wall time |
| `predictions` | one row per (molecule, target), each metric its own column |

```bash
tail -f oracle.log
sqlite3 oracle.db 'SELECT miner, COUNT(*) FROM predictions GROUP BY miner'
```

## Requirements

Each model's README lists what it needs; its `provision.sh` installs it.

## License

MIT — see [LICENSE](LICENSE).
