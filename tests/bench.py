"""Throughput against batch depth: a shard's fixed per-call cost is amortised
over the molecules in its chunk, so this finds where the curve flattens."""
import time

from harness import LIGANDS, NNMT, call

code, status = call("/health")
assert code == 200, status
shards = status["total"]
print("%d shards on %d gpus\n" % (shards, len(status["gpus"])))

code, body = call("/v1/proteins", {"sequences": [NNMT]})
assert code == 200, body
print("target %s, %d residues, depth %d\n"
      % (body["proteins"][0]["sequence_hash"], body["proteins"][0]["residues"],
         body["proteins"][0]["depth"]))

print("%7s %11s %11s %11s %9s" % ("batch", "per shard", "round-trip", "s/molecule", "mol/hr"))
for depth in (1, 2, 4, 8, 16):
    n = shards * depth
    smiles = [LIGANDS[i % len(LIGANDS)] for i in range(n)]
    started = time.time()
    code, body = call("/v1/score", {"miner": "bench", "epoch": "bench",
                                    "targets": [NNMT], "smiles": smiles})
    elapsed = time.time() - started
    if code != 200 or body.get("predictions") != n:
        print("%7d  FAILED code=%s predictions=%s %s"
              % (n, code, body.get("predictions"), body.get("detail", "")))
        continue
    print("%7d %11d %10.1fs %10.2fs %9.0f"
          % (n, depth, elapsed, elapsed / n, n / elapsed * 3600))
