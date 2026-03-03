"""Deployment preflight checks for local MUDRA runtime."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def check_file(path: str) -> bool:
    return Path(path).exists()


def main() -> int:
    checks = {}
    checks["config"] = check_file("config/app.yaml")
    checks["db_schema"] = check_file("database/schema.sql")
    checks["ui_entry"] = check_file("mudra_app.py")
    checks["api_entry"] = check_file("backend/run_api.py")

    db_ok = False
    try:
        conn = sqlite3.connect("database/mudra.db")
        conn.execute("SELECT 1")
        db_ok = True
        conn.close()
    except Exception:
        db_ok = False
    checks["sqlite_access"] = db_ok

    model_paths = [
        "models/static/static_mlp_v001.pt",
        "models/dynamic/dynamic_bigru_v001.pt",
        "models/registry/label_map.json",
    ]
    checks["model_artifacts"] = {p: check_file(p) for p in model_paths}

    passed = all(v if isinstance(v, bool) else all(v.values()) for v in checks.values())
    print(json.dumps({"passed": passed, "checks": checks}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
