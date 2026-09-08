#!/usr/bin/env bash
# Start the oracle. Kills by pidfile, never by process-name pattern.
set -uo pipefail
cd "$(dirname "$0")/.."
source "${NOVA_DIR:-/root/nova}"/.venv/bin/activate
set -a; [ -f .env ] && . ./.env; set +a
if [ -f oracle.pid ]; then
  kill -TERM -"$(cat oracle.pid)" 2>/dev/null; sleep 4
  kill -KILL -"$(cat oracle.pid)" 2>/dev/null; rm -f oracle.pid
fi
setsid nohup python3 -m oracle > oracle.log 2>&1 < /dev/null &
echo $! > oracle.pid
echo "oracle pgid $(cat oracle.pid)"
