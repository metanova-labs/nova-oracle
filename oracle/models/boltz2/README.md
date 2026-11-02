# Boltz-2

Each shard holds both Boltz-2 checkpoints. Boltz's dataloader is
`batch_size=1`, so a shard scores serially and **the caller's batch size is the
throughput lever**. Every target needs an alignment: registering it searches the
ColabFold MSA server once (tens of seconds) and caches the result under the
sequence digest.

## Setup

```bash
./deploy/provision.sh boltz2               # bare GPU box -> ready, 10-20 min
cp oracle/models/boltz2/env.example .env   # then edit
```

Needs Python 3.12 and an NVIDIA driver ≥580. `provision.sh` installs the
packages locked in `requirements.txt`, including the patched boltz from
[nova](https://github.com/metanova-labs/nova) at a pinned commit. Changing any
version can change scores. The box needs outbound network the first time it
sees a sequence.

| variable | notes |
|---|---|
| `BOLTZ2_MSA_DIR` | alignment cache, one `.a3m` per sequence digest; required |

Tuning defaults, measured on 8× RTX 4090:

| variable | default |
|---|---|
| `ORACLE_SHARDS_PER_GPU` | 3 |
| `ORACLE_STAGGER_S` | 2 |
| `ORACLE_REQUEST_TIMEOUT_S` | 900 |
| `ORACLE_RECYCLE_RSS_GB` | 17 |
| `ORACLE_MAX_BATCH` | 256 |

## Scores

Each score carries Boltz's metrics verbatim: `affinity_pred_value` and
`affinity_probability_binary` (plus their `1`/`2` ensemble members),
`confidence_score`, `ptm`, `iptm`, `ligand_iptm`, `protein_iptm`,
`complex_plddt`, `complex_iplddt`, `complex_pde`, `complex_ipde`, and the three
per-chain values flattened from its `chains_ptm` / `pair_chains_iptm` maps:
`chains_ptm_protein`, `chains_ptm_ligand`, `pair_chains_iptm_protein_ligand`.

## Scoring settings

Fixed in `worker.py` (`PREDICT`), not per request: callers scoring under
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

Size `ORACLE_SHARDS_PER_GPU` by the lower of `hostRAM / 16 / nGPUs` and
`VRAM / 6` — a shard's RSS plateaus near 14 GB and its peak reserved VRAM is
5.7 GB. `ORACLE_RECYCLE_RSS_GB` must exceed that plateau, or shards are rebuilt
forever for no reason.
