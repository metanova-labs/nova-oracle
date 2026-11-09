# nova-oracle

Protein–ligand affinity scoring as a service, for NOVA (Bittensor subnet 68).

Each **shard** is a worker process holding the model resident, so a request
costs compute only. Every request uses the whole pool. Proteins are addressed by
sequence; a round is prepared once, then scored against.

| model | `ORACLE_MODEL` | |
|---|---|---|
| Boltz-2 | `boltz2` | [setup, scores, performance](oracle/models/boltz2/README.md) |
| Nesso-1 | `nesso` | [setup, scores](oracle/models/nesso/README.md) |

## Quickstart

```bash
./deploy/provision.sh boltz2               # bare GPU box -> ready
cp oracle/models/boltz2/env.example .env   # then edit
./deploy/run.sh                            # start (or restart)
python3 tests/smoke.py                     # verify every route
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
| `labels` | `[str]?` | optional, positional, ≤128 chars — e.g. a UniProt accession, recorded against the protein |

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
| `smiles` | `[str]` | 1–10,000 of ≤2,000 chars; `len(targets) × len(smiles)` ≤ `ORACLE_MAX_BATCH` |
| `epoch` | `str?` | optional round label, ≤128 chars, stored with each prediction |

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
| `502` | preparing a target failed |
| `503` | no live shards (score), or a degraded pool (health) |

## Configuration

All settings come from the environment; `deploy/run.sh` sources `.env`. Start
from the model's `env.example`. The oracle logs every setting at startup.

Machine settings have no default:

| variable | notes |
|---|---|
| `ORACLE_MODEL` | which model to serve |
| `ORACLE_GPUS` | comma-separated GPU ids |
| `ORACLE_HOST` / `ORACLE_PORT` | bind address; use loopback and terminate TLS at a proxy. Set a routable address only when the proxy cannot reach loopback |
| `ORACLE_TOKEN` | bearer token for every route but `/health` |
| `ORACLE_DB` | prediction store; one per model |

Tuning settings default to the model's measured values, listed in its README:

| variable | notes |
|---|---|
| `ORACLE_SHARDS_PER_GPU` | shards per GPU |
| `ORACLE_STAGGER_S` | spacing between shard starts |
| `ORACLE_REQUEST_TIMEOUT_S` | per-shard call timeout |
| `ORACLE_RECYCLE_RSS_GB` | retire a shard above this RSS |
| `ORACLE_MAX_BATCH` | predictions per request |

## Running

`deploy/run.sh` (re)starts the oracle in the background and restarts it if it
crashes, unless it exits during startup (a missing setting, say). Output goes to
`oracle.log`, rotated at 50 MB with five old files kept. Stop it with
`kill -- -$(cat oracle.pid)`.

To start it at boot, run `deploy/autostart.sh`. Where systemd runs, it installs
and enables `nova-oracle.service`, which logs to `journalctl -u nova-oracle`. In
a container, pass it the script the container runs at start, e.g.
`deploy/autostart.sh /root/onstart.sh` on Vast.

## Tests

```bash
python3 -m unittest discover tests   # offline: settings, store, pool, API on a fake shard
python3 tests/smoke.py               # every route of a running oracle
python3 tests/bench.py               # throughput against batch depth
```

The offline tests need no GPU, only the service packages `pip install .` brings.
The live ones read the host, port and token from `.env`, or `ORACLE_URL`.

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
sqlite3 boltz2.db 'SELECT miner, COUNT(*) FROM predictions GROUP BY miner'
```

## Requirements

Each model's README lists what it needs; its `provision.sh` installs it.

## License

MIT — see [LICENSE](LICENSE).
