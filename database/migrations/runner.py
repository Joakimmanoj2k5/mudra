"""Simple SQL migration runner for SQLite release flow."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _ensure_migration_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
          migration_id TEXT PRIMARY KEY,
          applied_at TEXT NOT NULL
        )
        """
    )


def _bootstrap_schema_if_needed(conn: sqlite3.Connection, schema_path: Path) -> None:
    row = conn.execute("SELECT COUNT(*) AS c FROM sqlite_master WHERE type='table' AND name='users'").fetchone()
    if row and int(row["c"]) > 0:
        return
    schema_sql = schema_path.read_text(encoding="utf-8")
    conn.executescript(schema_sql)


def _applied_ids(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT migration_id FROM schema_migrations").fetchall()
    return {r["migration_id"] for r in rows}


def apply_migrations(db_path: str, migrations_dir: str, schema_path: str) -> List[str]:
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    mig_dir = Path(migrations_dir)
    mig_dir.mkdir(parents=True, exist_ok=True)

    conn = _connect(str(db_file))
    applied_now: List[str] = []
    try:
        _ensure_migration_table(conn)
        _bootstrap_schema_if_needed(conn, Path(schema_path))

        if "000_schema_baseline" not in _applied_ids(conn):
            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations(migration_id, applied_at) VALUES('000_schema_baseline', datetime('now'))"
            )

        applied = _applied_ids(conn)
        migration_files = sorted(mig_dir.glob("*.sql"))
        for path in migration_files:
            mig_id = path.stem
            if mig_id in applied:
                continue
            sql = path.read_text(encoding="utf-8")
            conn.executescript(sql)
            conn.execute(
                "INSERT INTO schema_migrations(migration_id, applied_at) VALUES(?, datetime('now'))",
                (mig_id,),
            )
            applied_now.append(mig_id)

        conn.commit()
        return applied_now
    finally:
        conn.close()
