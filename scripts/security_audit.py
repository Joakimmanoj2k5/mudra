"""Basic security audit checks for MUDRA deployment."""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_SECRET = "mudra-dev-secret-change-me"


def check_secret() -> tuple[bool, str]:
    secret = os.getenv("MUDRA_SECRET_KEY", DEFAULT_SECRET)
    if secret == DEFAULT_SECRET:
        return False, "MUDRA_SECRET_KEY is default. Set a strong secret before production."
    if len(secret) < 24:
        return False, "MUDRA_SECRET_KEY is too short (<24 chars)."
    return True, "MUDRA_SECRET_KEY configured"


def check_db_exists() -> tuple[bool, str]:
    db = Path(os.getenv("MUDRA_DB_PATH", "database/mudra.db"))
    return (db.exists(), f"DB path {'exists' if db.exists() else 'missing'}: {db}")


def check_model_artifacts() -> tuple[bool, str]:
    required = [
        Path("models/registry/label_map.json"),
        Path("models/registry/norm_stats.json"),
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        return False, "Missing model metadata files: " + ", ".join(missing)
    return True, "Model metadata files present"


def main() -> int:
    checks = [check_secret(), check_db_exists(), check_model_artifacts()]
    failed = [msg for ok, msg in checks if not ok]
    for ok, msg in checks:
        print(("PASS" if ok else "FAIL") + " | " + msg)
    if failed:
        print("Security audit failed")
        return 1
    print("Security audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
