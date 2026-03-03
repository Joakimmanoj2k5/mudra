#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

RELEASE_TAG="${1:-v0.1.0}"
OUT_DIR="release/${RELEASE_TAG}"
mkdir -p "$OUT_DIR"

# Optional desktop package
if [ -d "dist/MUDRA" ]; then
  cp -R dist/MUDRA "$OUT_DIR/"
fi

mkdir -p "$OUT_DIR/deploy" "$OUT_DIR/scripts" "$OUT_DIR/config" "$OUT_DIR/database"
cp -R deploy "$OUT_DIR/"
cp -R scripts "$OUT_DIR/"
cp -R config "$OUT_DIR/"
cp database/schema.sql "$OUT_DIR/database/"
cp README.md requirements.txt requirements-dev.txt "$OUT_DIR/"

(
  cd "$OUT_DIR"
  find . -type f -print0 | xargs -0 shasum -a 256 > SHA256SUMS
)

echo "Release bundle created: $OUT_DIR"
