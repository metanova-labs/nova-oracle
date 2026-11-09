# Nesso-1

Each shard holds [Nesso-1](https://github.com/recursionpharma/nesso). Registering
a target computes its ESM-2 embedding once, on the first GPU in `ORACLE_GPUS`
(about 10 s), and caches it.

## Setup

```bash
./deploy/provision.sh nesso               # bare GPU box -> ready, about 5 min
cp oracle/models/nesso/env.example .env   # then edit
```

Needs Python 3.12 and an NVIDIA driver ≥580. `provision.sh` installs the
packages locked in `requirements.txt`, including Nesso at a pinned commit, and
downloads the weights, residue dictionary and ESM-2 into `.cache/`. The weights
follow the Nesso version (`v1.0.0` on Hugging Face). Changing any version can
change scores. Shards run offline.

| variable | notes |
|---|---|
| `NESSO_EMBEDDINGS_DIR` | embedding cache, one `.safetensors` per sequence; required |

Tuning defaults:

| variable | default |
|---|---|
| `ORACLE_SHARDS_PER_GPU` | 4 |
| `ORACLE_STAGGER_S` | 2 |
| `ORACLE_REQUEST_TIMEOUT_S` | 900 |
| `ORACLE_RECYCLE_RSS_GB` | 8 |
| `ORACLE_MAX_BATCH` | 512 |

## Scores

Each score carries what Nesso's own writer reports: `affinity_pred_value`,
`affinity_probability_binary`, `affinity_pred_value1`, `affinity_pred_value2`,
`affinity_logits_binary`, and the six distogram entropies `entropy_pp`,
`entropy_pl`, `entropy_ll`, `entropy_crop_pp`, `entropy_crop_pl`,
`entropy_crop_ll`.

## Scoring settings

Fixed in `worker.py`, not per request. They are Nesso's CLI defaults: 5
recycling steps, pocket refinement (22 Å, 256 tokens), a 15 Å affinity cutoff,
bf16 autocast, full-precision float32 matmuls and the cuEquivariance kernels.
A shard will not start without the kernels.

Each molecule's conformer and the featurizer's random rotation are seeded from
its SMILES, so a molecule scores identically on every read. Nesso's CLI seeds
once per run instead. Scores differ slightly across GPU models and drivers.
