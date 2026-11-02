#!/usr/bin/env bash
# (Re)start the oracle in the background, restarting it if it crashes after
# startup. --foreground runs it once in this shell, for systemd. Kills by
# pidfile, never by process-name pattern.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
set -a; [ -f .env ] && . ./.env; set +a

supervise() {
  while true; do
    started=$(date +%s)
    python3 -m oracle
    status=$?
    if (( $(date +%s) - started < 60 )); then
      echo "oracle exited during startup (status $status); not restarting"
      return "$status"
    fi
    echo "oracle exited (status $status); restarting in 10 s"
    sleep 10
  done
}

case "${1:-}" in
  --foreground) exec python3 -m oracle ;;
  --supervise) supervise 2>&1 | python3 deploy/rotate_log.py oracle.log; exit ;;
esac

if [ -f oracle.pid ]; then
  kill -TERM -"$(cat oracle.pid)" 2>/dev/null; sleep 4
  kill -KILL -"$(cat oracle.pid)" 2>/dev/null; rm -f oracle.pid
fi
setsid nohup deploy/run.sh --supervise > /dev/null 2>&1 < /dev/null &
echo $! > oracle.pid
echo "oracle pgid $(cat oracle.pid)"
