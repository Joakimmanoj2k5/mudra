"""Deterministic CI checks for MUDRA."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


def assert_ast_clean() -> None:
    for path in Path(".").rglob("*.py"):
        if "mudra_env" in str(path) or ".venv" in str(path):
            continue
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def assert_catalog_counts() -> None:
    from utils.common.gesture_catalog import ALPHABETS, WORD_SPECS

    assert len(ALPHABETS) == 26, f"expected 26 alphabets, got {len(ALPHABETS)}"
    assert 50 <= len(WORD_SPECS) <= 100, f"expected 50-100 words, got {len(WORD_SPECS)}"


def assert_database_seed() -> None:
    from database.db import DatabaseManager

    db = DatabaseManager("database/mudra.db")
    db.seed_core_data()
    gestures = db.get_gestures()
    assert len(gestures) >= 76, f"expected >=76 gestures, got {len(gestures)}"


def assert_migration_runner() -> None:
    from database.migrations.runner import apply_migrations

    applied = apply_migrations("database/mudra.db", "database/migrations", "database/schema.sql")
    assert isinstance(applied, list)


def run_preflight() -> None:
    code, out = run([sys.executable, "scripts/preflight.py"])
    print(out.strip())
    if code not in (0, 1):
        raise RuntimeError("preflight script failed unexpectedly")


def run_pytest() -> None:
    try:
        import pytest  # noqa: F401
    except Exception:
        print("pytest not installed in current interpreter; skipping test execution in local check.")
        return
    code, out = run([sys.executable, "-m", "pytest", "tests/unit", "tests/integration", "-q"])
    print(out.strip())
    if code != 0:
        raise RuntimeError("pytest failed")


def main() -> int:
    try:
        assert_ast_clean()
        assert_catalog_counts()
        assert_database_seed()
        assert_migration_runner()
        run_preflight()
        run_pytest()
    except Exception as exc:
        print(f"CI checks failed: {exc}")
        return 1
    print("CI checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
