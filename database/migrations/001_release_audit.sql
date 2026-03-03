CREATE TABLE IF NOT EXISTS release_audit (
  audit_id TEXT PRIMARY KEY,
  release_tag TEXT NOT NULL,
  notes TEXT,
  created_at TEXT NOT NULL
);
