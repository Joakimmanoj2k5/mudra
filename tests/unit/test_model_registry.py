from database.db import DatabaseManager


def test_model_registry_activation(tmp_path):
    db = DatabaseManager(str(tmp_path / "models.db"))
    db.seed_core_data()

    rows = db.list_model_versions()
    assert len(rows) >= 2

    static_rows = [r for r in rows if r["model_name"] == "static_mlp"]
    assert static_rows
    model_id = static_rows[0]["model_version_id"]
    assert db.activate_model_version(model_id)

    active = db.get_active_model_paths()
    assert "static_model_path" in active


def test_model_register_and_rollback(tmp_path):
    db = DatabaseManager(str(tmp_path / "rollback.db"))
    db.seed_core_data()
    model_id = db.register_model_version(
        model_name="static_mlp",
        framework="pytorch",
        artifact_path="models/static/static_mlp_v999.pt",
        label_map_path="models/registry/label_map.json",
        norm_stats_path="models/registry/norm_stats.json",
        metrics={"accuracy": 0.91},
        activate=True,
    )
    assert model_id
    active = db.get_active_model_paths()
    assert active["static_model_path"].endswith("static_mlp_v999.pt")
    assert db.rollback_model_family("static_mlp")
