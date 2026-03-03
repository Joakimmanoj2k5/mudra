from pathlib import Path

from database.migrations.runner import apply_migrations


def test_apply_migrations(tmp_path):
    db_path = tmp_path / "m.db"
    migrations = tmp_path / "migrations"
    migrations.mkdir(parents=True)
    schema_path = tmp_path / "schema.sql"

    schema_path.write_text(
        """
        CREATE TABLE IF NOT EXISTS users (user_id TEXT PRIMARY KEY);
        """,
        encoding="utf-8",
    )
    (migrations / "001_extra.sql").write_text(
        "CREATE TABLE IF NOT EXISTS extra_table (id TEXT PRIMARY KEY);",
        encoding="utf-8",
    )

    applied = apply_migrations(str(db_path), str(migrations), str(schema_path))
    assert "001_extra" in applied

    # Idempotent second run
    applied_again = apply_migrations(str(db_path), str(migrations), str(schema_path))
    assert applied_again == []
