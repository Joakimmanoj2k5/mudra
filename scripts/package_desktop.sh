#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PY="python3"
if [ -x "./.venv/bin/python" ]; then
  source ./.venv/bin/activate
  PY="python"
elif [ -x "./mudra_env/bin/python" ]; then
  PY="./mudra_env/bin/python"
fi

if ! $PY -c "import PyInstaller" >/dev/null 2>&1; then
  echo "PyInstaller not installed. Install with: $PY -m pip install pyinstaller"
  exit 1
fi

rm -rf build dist
$PY -m PyInstaller \
  --name MUDRA \
  --windowed \
  --noconfirm \
  --add-data "config:config" \
  --add-data "models:models" \
  --add-data "database:database" \
  mudra_app.py

echo "Desktop package created under dist/MUDRA"
