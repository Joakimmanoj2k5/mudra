"""Restore SQLite DB from backup file."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup", required=True)
    parser.add_argument("--db", default="database/mudra.db")
    args = parser.parse_args()

    backup = Path(args.backup)
    if not backup.exists():
        raise FileNotFoundError(f"Backup not found: {backup}")

    target = Path(args.db)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(backup, target)
    print(f"Restored {backup} -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
