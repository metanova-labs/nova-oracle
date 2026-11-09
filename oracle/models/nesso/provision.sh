#!/usr/bin/env bash
# Provision a GPU box: the locked environment and the Nesso-1 weights, residue
# dictionary and ESM-2. Idempotent.
#
#   ./deploy/provision.sh nesso 2>&1 | tee provision.log
set -euo pipefail
cd "$(dirname "$0")/../../.."

VENV=.venv
step() { printf '\n=== %s (%s) ===\n' "$1" "$(date -u +%H:%M:%S)"; }

step "system packages"
need=()
command -v git >/dev/null || need+=(git)
command -v gcc >/dev/null || need+=(build-essential)
python3 -c "import ensurepip" 2>/dev/null || need+=(python3-venv)
python3 -c "import sysconfig, os; assert os.path.exists(sysconfig.get_path('include') + '/Python.h')" 2>/dev/null || need+=(python3-dev)
if [ ${#need[@]} -gt 0 ]; then
  sudo=$([ "$(id -u)" -eq 0 ] || echo sudo)
  $sudo apt-get update -qq
  $sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${need[@]}" >/dev/null
fi

step "python environment"
# Host pip config can pin other versions.
export PIP_CONFIG_FILE=/dev/null
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install -q --upgrade pip wheel

step "locked packages"
"$VENV/bin/pip" install -q --no-deps -r oracle/models/nesso/requirements.txt
"$VENV/bin/pip" check

step "Nesso-1 weights, residue dictionary and ESM-2"
"$VENV/bin/python" - <<'PY'
from nesso.data.esm import DEFAULT_ESM2_MODEL, setup_esm_model
from nesso.main import ensure_cache, get_cache_path, resolve_model_revision
cache = get_cache_path()
ccd, model = ensure_cache(cache, revision=resolve_model_revision(None))
setup_esm_model(DEFAULT_ESM2_MODEL, "cpu", cache_dir=cache / "huggingface")
print("  model", model.name, "| esm", DEFAULT_ESM2_MODEL)
PY

step "verification"
"$VENV/bin/python" - <<'PY'
import torch
print("  torch", torch.__version__, "cuda", torch.version.cuda,
      "| gpus", torch.cuda.device_count(),
      "| capability", torch.cuda.get_device_capability(0))
import nesso  # noqa: F401
import cuequivariance_torch  # noqa: F401
print("  nesso and kernels import OK")
PY
du -sh .cache | sed 's/^/  weights: /'
echo
echo "provisioned. start with: ./deploy/run.sh"
