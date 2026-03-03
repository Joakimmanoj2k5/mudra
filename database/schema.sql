PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
  user_id TEXT PRIMARY KEY,
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  full_name TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('learner','admin')),
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS lessons (
  lesson_id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  lesson_type TEXT NOT NULL CHECK (lesson_type IN ('alphabet','word','conversation')),
  level INTEGER NOT NULL CHECK (level BETWEEN 1 AND 5),
  sequence_order INTEGER NOT NULL,
  is_active INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS gestures (
  gesture_id TEXT PRIMARY KEY,
  lesson_id TEXT NOT NULL,
  gesture_code TEXT UNIQUE NOT NULL,
  display_name TEXT NOT NULL,
  gesture_mode TEXT NOT NULL CHECK (gesture_mode IN ('static','dynamic')),
  category TEXT NOT NULL,
  requires_two_hands INTEGER NOT NULL DEFAULT 0,
  description TEXT,
  media_path TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY (lesson_id) REFERENCES lessons(lesson_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS model_versions (
  model_version_id TEXT PRIMARY KEY,
  model_name TEXT NOT NULL,
  framework TEXT NOT NULL,
  version_tag TEXT NOT NULL,
  artifact_path TEXT NOT NULL,
  label_map_path TEXT NOT NULL,
  norm_stats_path TEXT NOT NULL,
  metrics_json TEXT NOT NULL,
  is_active INTEGER NOT NULL DEFAULT 0,
  trained_at TEXT NOT NULL,
  UNIQUE(model_name, version_tag)
);

CREATE TABLE IF NOT EXISTS attempts (
  attempt_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  gesture_id TEXT NOT NULL,
  target_gesture_id TEXT NOT NULL,
  model_version_id TEXT,
  predicted_label TEXT NOT NULL,
  confidence REAL NOT NULL,
  is_correct INTEGER NOT NULL,
  latency_ms INTEGER NOT NULL,
  fps REAL,
  attempt_mode TEXT NOT NULL CHECK (attempt_mode IN ('practice','quiz','lesson')),
  created_at TEXT NOT NULL,
  FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
  FOREIGN KEY (gesture_id) REFERENCES gestures(gesture_id) ON DELETE RESTRICT,
  FOREIGN KEY (target_gesture_id) REFERENCES gestures(gesture_id) ON DELETE RESTRICT,
  FOREIGN KEY (model_version_id) REFERENCES model_versions(model_version_id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS progress (
  progress_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  lesson_id TEXT NOT NULL,
  attempts_count INTEGER NOT NULL DEFAULT 0,
  correct_count INTEGER NOT NULL DEFAULT 0,
  accuracy REAL NOT NULL DEFAULT 0,
  mastery_score REAL NOT NULL DEFAULT 0,
  last_attempt_at TEXT,
  updated_at TEXT NOT NULL,
  UNIQUE(user_id, lesson_id),
  FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
  FOREIGN KEY (lesson_id) REFERENCES lessons(lesson_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_attempts_user_time ON attempts(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_attempts_gesture ON attempts(gesture_id);
CREATE INDEX IF NOT EXISTS idx_progress_user ON progress(user_id);

CREATE TABLE IF NOT EXISTS study_sessions (
  session_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  gesture_id TEXT NOT NULL,
  study_time_seconds INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
  FOREIGN KEY (gesture_id) REFERENCES gestures(gesture_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_study_user_time ON study_sessions(user_id, created_at DESC);
