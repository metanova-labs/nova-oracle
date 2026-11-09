#!/usr/bin/env bash
# Start the oracle at boot. Where systemd runs, installs and enables
# nova-oracle.service; in a container, adds deploy/run.sh to the script the
# container runs at start: ./deploy/autostart.sh /root/onstart.sh
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"

if [ -d /run/systemd/system ]; then
  sudo=$([ "$(id -u)" -eq 0 ] || echo sudo)
  $sudo tee /etc/systemd/system/nova-oracle.service >/dev/null <<UNIT
[Unit]
Description=nova-oracle
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=600
StartLimitBurst=5

[Service]
User=$(id -un)
WorkingDirectory=$repo
ExecStart=$repo/deploy/run.sh --foreground
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT
  $sudo systemctl daemon-reload
  $sudo systemctl enable nova-oracle
  echo "enabled nova-oracle.service; start it with: systemctl start nova-oracle"
elif [ -n "${1:-}" ]; then
  line="$repo/deploy/run.sh"
  grep -qxF "$line" "$1" 2>/dev/null || echo "$line" >> "$1"
  echo "$1 now starts the oracle"
else
  echo "no systemd here; pass the script this container runs at start, e.g. $0 /root/onstart.sh" >&2
  exit 2
fi
