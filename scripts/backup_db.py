"""Create timestamped SQLite backup for release safety."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="database/mudra.db")
    parser.add_argument("--out", default="backups")
    args = parser.parse_args()

    src = Path(args.db)
    if not src.exists():
        raise FileNotFoundError(f"DB not found: {src}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    dest = out_dir / f"mudra_{stamp}.db"
    shutil.copy2(src, dest)
    print(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
