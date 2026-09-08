#!/usr/bin/env bash
# Provision a fresh GPU box into a running oracle.
#
# Everything the service needs: the boltz source (from the nova repo), a CUDA
# torch environment, and the Boltz-2 weights. Alignments are not provisioned --
# the oracle generates and caches its own. Idempotent -- safe to re-run; each
# step skips if already done.
#
#   ./provision.sh 2>&1 | tee provision.log
set -euo pipefail

NOVA_DIR="${NOVA_DIR:-/root/nova}"
CACHE="${BOLTZ_CACHE:-/root/.boltz}"
VENV="$NOVA_DIR/.venv"
step() { printf '\n=== %s (%s) ===\n' "$1" "$(date -u +%H:%M:%S)"; }

step "system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git build-essential python3-venv python3-dev curl >/dev/null

step "nova checkout (boltz source)"
if [ ! -d "$NOVA_DIR/.git" ]; then
  git clone --depth 1 -q https://github.com/metanova-labs/nova.git "$NOVA_DIR"
fi
test -d "$NOVA_DIR/external_tools/boltz" || { echo "boltz source missing"; exit 1; }

step "python environment"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install -q --upgrade pip wheel

step "torch + boltz dependencies"
# requirements.txt resolves a CUDA-capable torch; we match cuequivariance to
# whatever CUDA that torch was built against rather than pinning a second stack.
"$VENV/bin/pip" install -q -r "$NOVA_DIR/requirements/requirements.txt"
"$VENV/bin/pip" install -q -e "$NOVA_DIR/external_tools/boltz"
TORCH_CUDA="$("$VENV/bin/python" -c 'import torch; print(torch.version.cuda)')"
case "$TORCH_CUDA" in 13*) SUF=cu13 ;; *) SUF=cu12 ;; esac
echo "  torch cuda=$TORCH_CUDA -> cuequivariance $SUF"
"$VENV/bin/pip" install -q "cuequivariance-ops-${SUF}" "cuequivariance-ops-torch-${SUF}" cuequivariance-torch

step "oracle service dependencies"
"$VENV/bin/pip" install -q fastapi uvicorn pydantic anyio

step "Boltz-2 weights"
mkdir -p "$CACHE"
fetch() {  # fetch <url> <dest>
  [ -s "$2" ] && { echo "  have $(basename "$2")"; return; }
  echo "  downloading $(basename "$2")"
  curl -sL --retry 5 --retry-all-errors -o "$2" "$1"
}
fetch https://model-gateway.boltz.bio/boltz2_conf.ckpt "$CACHE/boltz2_conf.ckpt"
fetch https://model-gateway.boltz.bio/boltz2_aff.ckpt  "$CACHE/boltz2_aff.ckpt"
fetch https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar "$CACHE/mols.tar"
[ -d "$CACHE/mols" ] || { echo "  extracting mols.tar"; tar -xf "$CACHE/mols.tar" -C "$CACHE"; }

step "verification"
"$VENV/bin/python" - <<'PY'
import torch
print("  torch", torch.__version__, "cuda", torch.version.cuda,
      "| gpus", torch.cuda.device_count(),
      "| capability", torch.cuda.get_device_capability(0))
import boltz  # noqa: F401
print("  boltz import OK")
try:
    from cuequivariance_torch.primitives.triangle import triangle_multiplicative_update  # noqa: F401
    print("  fused kernels available")
except Exception as exc:
    print("  fused kernels UNAVAILABLE:", type(exc).__name__, exc)
PY
du -sh "$CACHE" | sed 's/^/  weights: /'
echo
echo "provisioned. start with: ORACLE_GPUS=0,... ./deploy/run.sh"
