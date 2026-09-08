"""End-to-end check of every route against a running oracle."""
import time

from harness import LIGANDS, NNMT, PDL1, call

FAILURES = []


def check(label, got, want):
    ok = got == want
    print("  %-32s %-20s %s" % (label, got, "ok" if ok else "EXPECTED %s" % (want,)))
    if not ok:
        FAILURES.append(label)


print("=== auth ===")
check("no token", call("/v1/score", {}, token="")[0], 401)
check("bad token", call("/v1/score", {}, token="wrong")[0], 401)
check("health needs no token", call("/health", token="")[0], 200)

print("=== validation ===")
check("non-residue sequence", call("/v1/proteins", {"sequences": ["not a protein"]})[0], 422)
check("empty sequence list", call("/v1/proteins", {"sequences": []})[0], 422)
check("missing miner", call("/v1/score", {"targets": [NNMT], "smiles": LIGANDS[:1]})[0], 422)
check("labels length mismatch",
      call("/v1/proteins", {"sequences": [NNMT], "labels": ["a", "b"]})[0], 422)

print("=== proteins ===")
started = time.time()
code, body = call("/v1/proteins", {"sequences": [NNMT, PDL1],
                                   "labels": ["P40261", "Q9NZQ7"]})
check("register 2 targets", code, 200)
for protein in body.get("proteins", []):
    print("    %s %4daa depth=%-6d cached=%s"
          % (protein["sequence_hash"], protein["residues"],
             protein["depth"], protein["cached"]))
print("    warm=%ss total=%.0fs" % (body.get("warm_s"), time.time() - started))
check("all cached", all(p["cached"] for p in body.get("proteins", [])), True)

print("=== scoring ===")
started = time.time()
code, body = call("/v1/score", {"miner": "smoke", "epoch": "e42",
                                "targets": [NNMT, PDL1], "smiles": LIGANDS[:2]})
check("2 molecules x 2 targets", code, 200)
check("predictions delivered", body.get("predictions"), 4)
check("target digests returned", len(body.get("targets", [])), 2)
check("results positional", [len(r["scores"]) for r in body.get("results", [])], [2, 2])
for row, molecule in zip(body.get("results", []), LIGANDS[:2]):
    for digest, score in zip(body["targets"], row["scores"]):
        print("    %s %s affinity=%6.3f p(bind)=%.3f"
              % (digest, molecule[:20].ljust(20), score["affinity_pred_value"],
                 score["affinity_probability_binary"]))
print("    %.1fs wall" % (time.time() - started))

print("=== refusals ===")
check("unregistered protein",
      call("/v1/score", {"miner": "smoke", "targets": ["MKVLAAGIVGLTKA"],
                         "smiles": LIGANDS[:1]})[0], 409)
check("batch over max",
      call("/v1/score", {"miner": "smoke", "targets": [NNMT],
                         "smiles": LIGANDS * 200})[0], 413)

print("=== health ===")
code, status = call("/health")
check("all shards alive", status.get("alive"), status.get("total"))
print("  ", status)

print("\n%s" % ("FAILED: " + ", ".join(FAILURES) if FAILURES else "all checks passed"))
raise SystemExit(1 if FAILURES else 0)
