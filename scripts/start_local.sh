#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [ -x "./mudra_env/bin/python" ]; then
  PY="./mudra_env/bin/python"
else
  PY="python3"
fi

echo "[MUDRA] Running preflight..."
$PY scripts/preflight.py || true

echo "[MUDRA] Seeding database..."
$PY -m database.seed.seed_database

echo "[MUDRA] Starting backend API on http://127.0.0.1:8000 ..."
$PY -m backend.run_api &
API_PID=$!

cleanup() {
  echo "[MUDRA] Stopping backend API..."
  kill "$API_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[MUDRA] Starting desktop UI..."
$PY mudra_app.py
