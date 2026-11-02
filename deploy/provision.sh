#!/usr/bin/env bash
# Provision a GPU box for one model: ./deploy/provision.sh <model>
set -euo pipefail
cd "$(dirname "$0")/.."
script="oracle/models/${1:-}/provision.sh"
if [ -z "${1:-}" ] || [ ! -f "$script" ]; then
  echo "usage: $0 <model>, one of: $(ls oracle/models/*/provision.sh | cut -d/ -f3 | paste -sd' ' -)" >&2
  exit 2
fi
exec "$script"
