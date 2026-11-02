#!/usr/bin/env bash
# Provision a GPU box: the locked environment and the Boltz-2 weights.
# Idempotent.
#
#   ./deploy/provision.sh boltz2 2>&1 | tee provision.log
set -euo pipefail
cd "$(dirname "$0")/../../.."

VENV=.venv
CACHE="$HOME/.boltz"
step() { printf '\n=== %s (%s) ===\n' "$1" "$(date -u +%H:%M:%S)"; }

step "system packages"
need=()
command -v git >/dev/null || need+=(git)
command -v curl >/dev/null || need+=(curl)
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
"$VENV/bin/pip" install -q --no-deps -r oracle/models/boltz2/requirements.txt
"$VENV/bin/pip" check

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
echo "provisioned. start with: ./deploy/run.sh"
