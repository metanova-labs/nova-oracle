"""A stand-in shard for the offline tests, speaking the worker protocol.

SMILES "CRASH" exits with a traceback on stderr; "SLOW" answers after 3 s.
"""
import json
import sys
import time

print("library banner", flush=True)
print(json.dumps({"event": "ready"}), flush=True)
for line in sys.stdin:
    items = json.loads(line)["items"]
    smiles = {item["smiles"] for item in items}
    if "CRASH" in smiles:
        sys.stderr.write("Traceback (most recent call last):\nRuntimeError: CUDA error\n")
        sys.stderr.flush()
        sys.exit(1)
    if "SLOW" in smiles:
        time.sleep(3)
    results = {item["id"]: {"affinity_pred_value": float(len(item["smiles"])),
                            "affinity_probability_binary": 0.5}
               for item in items if item["input"]}
    sys.stdout.write("progress 100% " + json.dumps(
        {"event": "result", "results": results, "error": None}) + "\n")
    sys.stdout.flush()
