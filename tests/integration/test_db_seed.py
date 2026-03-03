from database.db import DatabaseManager


def test_seed_data(tmp_path):
    db_path = tmp_path / "test.db"
    db = DatabaseManager(str(db_path))
    db.seed_core_data()
    gestures = db.get_gestures()
    assert len(gestures) >= 76
