#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

VENV_PATH=".venv"
if [ ! -d "$VENV_PATH" ]; then
  python3 -m venv "$VENV_PATH"
fi

source "$VENV_PATH/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt

python -m database.seed.seed_database
python scripts/preflight.py || true

echo "Bootstrap complete. Activate with: source .venv/bin/activate"
