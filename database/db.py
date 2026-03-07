"""SQLite-first database access for MUDRA."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.common.gesture_catalog import ALPHABETS, WORD_SPECS, all_gestures
from utils.common.security import hash_password as _hash_password


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class User:
    user_id: str
    email: str
    full_name: str
    role: str


class DatabaseManager:
    def __init__(self, db_path: str = "database/mudra.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        schema = Path("database/schema.sql").read_text(encoding="utf-8")
        with self.connect() as conn:
            conn.executescript(schema)

    def seed_core_data(self) -> None:
        now = utc_now()
        with self.connect() as conn:
            lessons = [
                (str(uuid.uuid4()), "ISL Alphabets", "alphabet", 1, 1, 1, now),
                (str(uuid.uuid4()), "ISL Core Words", "word", 1, 2, 1, now),
                (str(uuid.uuid4()), "Daily Conversation", "conversation", 2, 3, 1, now),
            ]
            existing = conn.execute("SELECT COUNT(*) c FROM lessons").fetchone()["c"]
            if not existing:
                conn.executemany(
                    "INSERT INTO lessons(lesson_id,title,lesson_type,level,sequence_order,is_active,created_at) VALUES(?,?,?,?,?,?,?)",
                    lessons,
                )

            lesson_rows = conn.execute("SELECT lesson_id,lesson_type FROM lessons").fetchall()
            lesson_map = {row["lesson_type"]: row["lesson_id"] for row in lesson_rows}
            gesture_count = conn.execute("SELECT COUNT(*) c FROM gestures").fetchone()["c"]
            if gesture_count == 0:
                # Try to load rich descriptions from the reference dataset
                ref_data = {}
                try:
                    import json as _json
                    ref_path = Path("data/assets/gestures/isl_reference_data.json")
                    if ref_path.exists():
                        ref_data = _json.loads(ref_path.read_text(encoding="utf-8"))
                except Exception:
                    pass

                rows = []
                for g in all_gestures():
                    lesson_type = "alphabet" if g.lesson_type == "alphabet" else "word"
                    # Get description from reference data
                    desc = f"Practice sign for {g.display_name}"
                    if g.lesson_type == "alphabet":
                        entry = ref_data.get("alphabets", {}).get(g.display_name, {})
                    else:
                        entry = ref_data.get("words", {}).get(g.display_name, {})
                    if entry and entry.get("description"):
                        desc = entry["description"]

                    rows.append(
                        (
                            str(uuid.uuid4()),
                            lesson_map[lesson_type],
                            g.code,
                            g.display_name,
                            g.gesture_mode,
                            g.category,
                            int(g.requires_two_hands),
                            desc,
                            None,
                            now,
                        )
                    )
                conn.executemany(
                    """
                    INSERT INTO gestures(
                      gesture_id,lesson_id,gesture_code,display_name,gesture_mode,category,
                      requires_two_hands,description,media_path,created_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?)
                    """,
                    rows,
                )

            mv_count = conn.execute("SELECT COUNT(*) c FROM model_versions").fetchone()["c"]
            if mv_count == 0:
                seed_models = [
                    (
                        str(uuid.uuid4()),
                        "baseline_rule_model",
                        "python",
                        "v001",
                        "models/registry/baseline_rule_model.json",
                        "models/registry/label_map.json",
                        "models/registry/norm_stats.json",
                        json.dumps({"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}),
                        0,
                        now,
                    ),
                    (
                        str(uuid.uuid4()),
                        "static_mlp",
                        "pytorch",
                        "v001",
                        "models/static/static_mlp_v001.pt",
                        "models/registry/label_map.json",
                        "models/registry/norm_stats.json",
                        json.dumps({"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}),
                        1,
                        now,
                    ),
                    (
                        str(uuid.uuid4()),
                        "dynamic_bigru",
                        "pytorch",
                        "v001",
                        "models/dynamic/dynamic_bigru_v001.pt",
                        "models/registry/label_map.json",
                        "models/registry/dynamic_norm_stats.json",
                        json.dumps({"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}),
                        1,
                        now,
                    ),
                ]
                conn.executemany(
                    """
                    INSERT INTO model_versions(
                        model_version_id,model_name,framework,version_tag,artifact_path,
                        label_map_path,norm_stats_path,metrics_json,is_active,trained_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?)
                    """,
                    seed_models,
                )
            else:
                self._ensure_default_model_versions(conn, now)

            # Seed default users (admin + demo learner) if no users exist
            user_count = conn.execute("SELECT COUNT(*) c FROM users").fetchone()["c"]
            if user_count == 0:
                admin_id = str(uuid.uuid4())
                demo_id = str(uuid.uuid4())
                conn.execute(
                    "INSERT INTO users(user_id,email,password_hash,full_name,role,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                    (admin_id, "admin@mudra.local", _hash_password("admin123"), "MUDRA Admin", "admin", now, now),
                )
                conn.execute(
                    "INSERT INTO users(user_id,email,password_hash,full_name,role,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                    (demo_id, "demo@mudra.local", _hash_password("demo123"), "Demo Learner", "learner", now, now),
                )

    def _ensure_default_model_versions(self, conn: sqlite3.Connection, now: str) -> None:
        defaults = [
            ("static_mlp", "pytorch", "v001", "models/static/static_mlp_v001.pt", "models/registry/norm_stats.json"),
            (
                "dynamic_bigru",
                "pytorch",
                "v001",
                "models/dynamic/dynamic_bigru_v001.pt",
                "models/registry/dynamic_norm_stats.json",
            ),
        ]
        for model_name, framework, version_tag, artifact, norm_path in defaults:
            row = conn.execute(
                "SELECT model_version_id FROM model_versions WHERE model_name=? AND version_tag=?",
                (model_name, version_tag),
            ).fetchone()
            if not row:
                conn.execute(
                    """
                    INSERT INTO model_versions(
                        model_version_id,model_name,framework,version_tag,artifact_path,
                        label_map_path,norm_stats_path,metrics_json,is_active,trained_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        str(uuid.uuid4()),
                        model_name,
                        framework,
                        version_tag,
                        artifact,
                        "models/registry/label_map.json",
                        norm_path,
                        json.dumps({"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}),
                        1,
                        now,
                    ),
                )
            else:
                active = conn.execute(
                    "SELECT COUNT(*) c FROM model_versions WHERE model_name=? AND is_active=1",
                    (model_name,),
                ).fetchone()["c"]
                if active == 0:
                    conn.execute(
                        "UPDATE model_versions SET is_active=1 WHERE model_name=? AND version_tag=?",
                        (model_name, version_tag),
                    )

    def create_user(self, email: str, password_hash: str, full_name: str, role: str = "learner") -> User:
        uid = str(uuid.uuid4())
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO users(user_id,email,password_hash,full_name,role,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                (uid, email.lower(), password_hash, full_name, role, now, now),
            )
        return User(user_id=uid, email=email.lower(), full_name=full_name, role=role)

    def get_user_by_email(self, email: str) -> Optional[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM users WHERE email=?", (email.lower(),)).fetchone()

    def get_user_by_id(self, user_id: str) -> Optional[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()

    def get_lessons(self) -> List[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM lessons WHERE is_active=1 ORDER BY sequence_order").fetchall()

    def get_gestures(self, lesson_type: Optional[str] = None) -> List[sqlite3.Row]:
        catalog_codes = tuple(g.code for g in all_gestures())
        if not catalog_codes:
            return []

        placeholders = ",".join("?" for _ in catalog_codes)
        query = f"SELECT g.* FROM gestures g JOIN lessons l ON g.lesson_id=l.lesson_id WHERE g.gesture_code IN ({placeholders})"
        params: Tuple[str, ...] = catalog_codes
        if lesson_type:
            query += " AND l.lesson_type=?"
            params = params + (lesson_type,)
        query += " ORDER BY g.display_name"
        with self.connect() as conn:
            return conn.execute(query, params).fetchall()

    def get_random_gestures(self, limit: int = 10) -> List[sqlite3.Row]:
        catalog_codes = tuple(g.code for g in all_gestures())
        if not catalog_codes:
            return []
        placeholders = ",".join("?" for _ in catalog_codes)
        with self.connect() as conn:
            return conn.execute(
                f"SELECT * FROM gestures WHERE gesture_code IN ({placeholders}) ORDER BY RANDOM() LIMIT ?",
                catalog_codes + (limit,),
            ).fetchall()

    def get_active_model_version(self, model_name: Optional[str] = None) -> Optional[str]:
        with self.connect() as conn:
            if model_name:
                row = conn.execute(
                    "SELECT model_version_id FROM model_versions WHERE model_name=? AND is_active=1 ORDER BY trained_at DESC LIMIT 1",
                    (model_name,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT model_version_id FROM model_versions WHERE is_active=1 ORDER BY trained_at DESC LIMIT 1"
                ).fetchone()
            return row["model_version_id"] if row else None

    def record_attempt(
        self,
        user_id: str,
        gesture_id: str,
        target_gesture_id: str,
        predicted_label: str,
        confidence: float,
        is_correct: bool,
        latency_ms: int,
        fps: float,
        attempt_mode: str,
    ) -> None:
        now = utc_now()
        with self.connect() as conn:
            target_mode_row = conn.execute(
                "SELECT gesture_mode FROM gestures WHERE gesture_id=?",
                (target_gesture_id,),
            ).fetchone()
        model_name = "dynamic_bigru" if target_mode_row and target_mode_row["gesture_mode"] == "dynamic" else "static_mlp"
        model_version_id = self.get_active_model_version(model_name=model_name) or self.get_active_model_version()
        aid = str(uuid.uuid4())
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO attempts(
                  attempt_id,user_id,gesture_id,target_gesture_id,model_version_id,predicted_label,
                  confidence,is_correct,latency_ms,fps,attempt_mode,created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    aid,
                    user_id,
                    gesture_id,
                    target_gesture_id,
                    model_version_id,
                    predicted_label,
                    confidence,
                    int(is_correct),
                    latency_ms,
                    fps,
                    attempt_mode,
                    now,
                ),
            )

            lesson_row = conn.execute("SELECT lesson_id FROM gestures WHERE gesture_id=?", (target_gesture_id,)).fetchone()
            if lesson_row:
                lesson_id = lesson_row["lesson_id"]
                p = conn.execute(
                    "SELECT * FROM progress WHERE user_id=? AND lesson_id=?", (user_id, lesson_id)
                ).fetchone()
                if not p:
                    conn.execute(
                        "INSERT INTO progress(progress_id,user_id,lesson_id,attempts_count,correct_count,accuracy,mastery_score,last_attempt_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                        (str(uuid.uuid4()), user_id, lesson_id, 1, int(is_correct), float(is_correct), float(is_correct), now, now),
                    )
                else:
                    attempts_count = p["attempts_count"] + 1
                    correct_count = p["correct_count"] + int(is_correct)
                    accuracy = correct_count / attempts_count
                    conn.execute(
                        "UPDATE progress SET attempts_count=?,correct_count=?,accuracy=?,mastery_score=?,last_attempt_at=?,updated_at=? WHERE progress_id=?",
                        (attempts_count, correct_count, accuracy, accuracy, now, now, p["progress_id"]),
                    )

    def get_user_progress(self, user_id: str) -> List[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                """
                SELECT p.*, l.title, l.lesson_type
                FROM progress p
                JOIN lessons l ON p.lesson_id = l.lesson_id
                WHERE p.user_id=?
                ORDER BY l.sequence_order
                """,
                (user_id,),
            ).fetchall()

    def get_user_attempts(self, user_id: str, limit: int = 100) -> List[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                """
                SELECT a.*, g.display_name AS target_name
                FROM attempts a
                JOIN gestures g ON a.target_gesture_id = g.gesture_id
                WHERE a.user_id=?
                ORDER BY a.created_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()

    def record_study_session(self, user_id: str, gesture_id: str, study_time_seconds: int) -> None:
        if study_time_seconds <= 0:
            return
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO study_sessions(session_id,user_id,gesture_id,study_time_seconds,created_at)
                VALUES(?,?,?,?,?)
                """,
                (str(uuid.uuid4()), user_id, gesture_id, int(study_time_seconds), utc_now()),
            )

    def get_analytics_summary(self, user_id: str) -> Dict[str, float]:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT
                  COUNT(*) AS total_attempts,
                  COALESCE(AVG(is_correct), 0) AS accuracy,
                  COALESCE(AVG(confidence), 0) AS avg_confidence,
                  COALESCE(AVG(latency_ms), 0) AS avg_latency_ms
                FROM attempts
                WHERE user_id=?
                """,
                (user_id,),
            ).fetchone()
            return dict(row) if row else {
                "total_attempts": 0,
                "accuracy": 0.0,
                "avg_confidence": 0.0,
                "avg_latency_ms": 0.0,
            }

    def list_model_versions(self) -> List[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                """
                SELECT *
                FROM model_versions
                ORDER BY model_name, trained_at DESC
                """
            ).fetchall()

    def activate_model_version(self, model_version_id: str) -> bool:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT model_name FROM model_versions WHERE model_version_id=?",
                (model_version_id,),
            ).fetchone()
            if not row:
                return False
            model_name = row["model_name"]
            conn.execute("UPDATE model_versions SET is_active=0 WHERE model_name=?", (model_name,))
            conn.execute("UPDATE model_versions SET is_active=1 WHERE model_version_id=?", (model_version_id,))
            return True

    def get_active_model_paths(self) -> Dict[str, str]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT model_name, artifact_path, label_map_path, norm_stats_path
                FROM model_versions
                WHERE is_active=1
                """
            ).fetchall()
        result: Dict[str, str] = {}
        for row in rows:
            name = row["model_name"]
            if name == "static_mlp":
                result["static_model_path"] = row["artifact_path"]
                result["label_map_path"] = row["label_map_path"]
                result["norm_stats_path"] = row["norm_stats_path"]
            elif name == "dynamic_bigru":
                result["dynamic_model_path"] = row["artifact_path"]
        return result

    def _next_version_tag(self, conn: sqlite3.Connection, model_name: str) -> str:
        rows = conn.execute(
            "SELECT version_tag FROM model_versions WHERE model_name=?",
            (model_name,),
        ).fetchall()
        max_n = 0
        for r in rows:
            tag = r["version_tag"]
            m = re.match(r"^v(\d+)$", str(tag))
            if m:
                max_n = max(max_n, int(m.group(1)))
        return f"v{max_n + 1:03d}"

    def register_model_version(
        self,
        model_name: str,
        framework: str,
        artifact_path: str,
        label_map_path: str,
        norm_stats_path: str,
        metrics: Dict[str, float],
        activate: bool = False,
        version_tag: Optional[str] = None,
    ) -> str:
        now = utc_now()
        model_version_id = str(uuid.uuid4())
        with self.connect() as conn:
            if version_tag is None or not version_tag.strip():
                version_tag = self._next_version_tag(conn, model_name)
            if activate:
                conn.execute("UPDATE model_versions SET is_active=0 WHERE model_name=?", (model_name,))
            conn.execute(
                """
                INSERT INTO model_versions(
                    model_version_id,model_name,framework,version_tag,artifact_path,
                    label_map_path,norm_stats_path,metrics_json,is_active,trained_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    model_version_id,
                    model_name,
                    framework,
                    version_tag,
                    artifact_path,
                    label_map_path,
                    norm_stats_path,
                    json.dumps(metrics),
                    1 if activate else 0,
                    now,
                ),
            )
        return model_version_id

    def rollback_model_family(self, model_name: str) -> bool:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT model_version_id, is_active, trained_at
                FROM model_versions
                WHERE model_name=?
                ORDER BY trained_at DESC
                """,
                (model_name,),
            ).fetchall()
            if len(rows) < 2:
                return False

            active_idx = -1
            for i, r in enumerate(rows):
                if int(r["is_active"]) == 1:
                    active_idx = i
                    break
            if active_idx == -1:
                active_idx = 0
                conn.execute(
                    "UPDATE model_versions SET is_active=1 WHERE model_version_id=?",
                    (rows[0]["model_version_id"],),
                )
                return True

            if active_idx + 1 >= len(rows):
                return False

            current_id = rows[active_idx]["model_version_id"]
            previous_id = rows[active_idx + 1]["model_version_id"]
            conn.execute("UPDATE model_versions SET is_active=0 WHERE model_version_id=?", (current_id,))
            conn.execute("UPDATE model_versions SET is_active=1 WHERE model_version_id=?", (previous_id,))
            return True
