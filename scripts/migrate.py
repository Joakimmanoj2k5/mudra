"""Apply database migrations for release/deployment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database.migrations.runner import apply_migrations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="database/mudra.db")
    parser.add_argument("--migrations", default="database/migrations")
    parser.add_argument("--schema", default="database/schema.sql")
    args = parser.parse_args()

    applied = apply_migrations(args.db, args.migrations, args.schema)
    if applied:
        print("Applied migrations:", ", ".join(applied))
    else:
        print("No pending migrations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
