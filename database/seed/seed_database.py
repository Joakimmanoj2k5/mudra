from database.db import DatabaseManager
from utils.common.security import hash_password


def main() -> None:
    db = DatabaseManager()
    db.seed_core_data()

    admin = db.get_user_by_email("admin@mudra.local")
    if not admin:
        db.create_user(
            email="admin@mudra.local",
            password_hash=hash_password("admin123"),
            full_name="MUDRA Admin",
            role="admin",
        )

    learner = db.get_user_by_email("demo@mudra.local")
    if not learner:
        db.create_user(
            email="demo@mudra.local",
            password_hash=hash_password("demo123"),
            full_name="Demo Learner",
            role="learner",
        )

    print("Database seeded: admin@mudra.local/admin123, demo@mudra.local/demo123")


if __name__ == "__main__":
    main()
