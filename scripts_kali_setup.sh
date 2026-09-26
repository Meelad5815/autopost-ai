#!/usr/bin/env bash
set -euo pipefail

echo "[1/3] Starting Kali container..."
docker compose -f docker-compose.kali.yml up -d kali

echo "[2/3] Updating apt index..."
docker exec autopost-kali bash -lc 'apt-get update'

echo "[3/3] Installing common security tools..."
docker exec autopost-kali bash -lc 'DEBIAN_FRONTEND=noninteractive apt-get install -y nmap sqlmap nikto hydra john curl wget git python3 python3-pip'

echo "Done. Enter container with: docker exec -it autopost-kali bash"
