"""FastAPI backend for auth, lessons, attempts, and analytics."""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from database.db import DatabaseManager
from utils.common.security import create_access_token, decode_access_token, hash_password, verify_password


app = FastAPI(title="MUDRA API", version="1.0.0")
db = DatabaseManager(os.getenv("MUDRA_DB_PATH", "database/mudra.db"))
db.seed_core_data()
_RATE_BUCKET = {}
_RATE_LIMIT = int(os.getenv("MUDRA_RATE_LIMIT_PER_MIN", "120"))


class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    email: str
    password: str
    full_name: str


class AttemptRequest(BaseModel):
    gesture_id: str
    target_gesture_id: str
    predicted_label: str
    confidence: float
    is_correct: bool
    latency_ms: int
    fps: float
    attempt_mode: str = "practice"


class ModelRegisterRequest(BaseModel):
    model_name: str
    framework: str = "pytorch"
    artifact_path: str
    label_map_path: str = "models/registry/label_map.json"
    norm_stats_path: str = "models/registry/norm_stats.json"
    metrics: dict = {}
    activate: bool = True
    version_tag: Optional[str] = None


def _client_key(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.middleware("http")
async def security_and_rate_limit_middleware(request: Request, call_next):
    now = int(time.time())
    minute = now // 60
    key = f"{_client_key(request)}:{minute}"
    count = _RATE_BUCKET.get(key, 0) + 1
    _RATE_BUCKET[key] = count

    if count > _RATE_LIMIT:
        return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)

    # opportunistic cleanup of old buckets
    if len(_RATE_BUCKET) > 5000:
        for k in list(_RATE_BUCKET.keys())[:1000]:
            if not k.endswith(f":{minute}"):
                _RATE_BUCKET.pop(k, None)

    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    return response


def get_current_user(authorization: Optional[str] = Header(default=None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    token = authorization.split(" ", 1)[1]
    user_id = decode_access_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def require_admin(current=Depends(get_current_user)):
    if current["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    return current


@app.get("/health")
def health():
    active = db.get_active_model_paths()
    static_model = Path(active.get("static_model_path", ""))
    dynamic_model = Path(active.get("dynamic_model_path", ""))
    db_path = Path(db.db_path)
    return {
        "status": "ok",
        "database_exists": db_path.exists(),
        "active_models": active,
        "artifacts_exist": {
            "static": static_model.exists() if static_model.as_posix() else False,
            "dynamic": dynamic_model.exists() if dynamic_model.as_posix() else False,
        },
    }


@app.get("/ready")
def ready():
    db_path = Path(db.db_path)
    if not db_path.exists():
        raise HTTPException(status_code=503, detail="Database not ready")
    try:
        active = db.get_active_model_paths()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model registry unavailable: {exc}")
    return {"status": "ready", "active_models": active}


@app.post("/auth/login")
def login(payload: LoginRequest):
    user = db.get_user_by_email(payload.email)
    if not user or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(user["user_id"])
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "user_id": user["user_id"],
            "email": user["email"],
            "full_name": user["full_name"],
            "role": user["role"],
        },
    }


@app.post("/auth/register")
def register(payload: RegisterRequest):
    if db.get_user_by_email(payload.email):
        raise HTTPException(status_code=409, detail="User already exists")
    user = db.create_user(payload.email, hash_password(payload.password), payload.full_name)
    return {"user_id": user.user_id, "email": user.email, "full_name": user.full_name, "role": user.role}


@app.get("/lessons")
def lessons(current=Depends(get_current_user)):
    lesson_rows = db.get_lessons()
    result = []
    for row in lesson_rows:
        gestures = db.get_gestures(row["lesson_type"] if row["lesson_type"] != "conversation" else "word")
        result.append({
            "lesson_id": row["lesson_id"],
            "title": row["title"],
            "lesson_type": row["lesson_type"],
            "level": row["level"],
            "gesture_count": len(gestures),
        })
    return result


@app.get("/gestures")
def gestures(lesson_type: Optional[str] = None, current=Depends(get_current_user)):
    rows = db.get_gestures(lesson_type)
    return [dict(r) for r in rows]


@app.post("/attempts")
def create_attempt(payload: AttemptRequest, current=Depends(get_current_user)):
    db.record_attempt(
        user_id=current["user_id"],
        gesture_id=payload.gesture_id,
        target_gesture_id=payload.target_gesture_id,
        predicted_label=payload.predicted_label,
        confidence=payload.confidence,
        is_correct=payload.is_correct,
        latency_ms=payload.latency_ms,
        fps=payload.fps,
        attempt_mode=payload.attempt_mode,
    )
    return {"ok": True}


@app.get("/progress")
def progress(current=Depends(get_current_user)):
    rows = db.get_user_progress(current["user_id"])
    return [dict(r) for r in rows]


@app.get("/analytics")
def analytics(current=Depends(get_current_user)):
    summary = db.get_analytics_summary(current["user_id"])
    attempts = db.get_user_attempts(current["user_id"], limit=200)
    return {
        "summary": summary,
        "attempts": [dict(a) for a in attempts],
    }


@app.get("/models")
def list_models(current=Depends(get_current_user)):
    rows = db.list_model_versions()
    return [dict(r) for r in rows]


@app.post("/models/register")
def register_model(payload: ModelRegisterRequest, current=Depends(require_admin)):
    mv_id = db.register_model_version(
        model_name=payload.model_name,
        framework=payload.framework,
        artifact_path=payload.artifact_path,
        label_map_path=payload.label_map_path,
        norm_stats_path=payload.norm_stats_path,
        metrics=payload.metrics if isinstance(payload.metrics, dict) else {},
        activate=payload.activate,
        version_tag=payload.version_tag,
    )
    return {"model_version_id": mv_id, "activated": payload.activate}


@app.post("/models/{model_version_id}/activate")
def activate_model(model_version_id: str, current=Depends(require_admin)):
    ok = db.activate_model_version(model_version_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Model version not found")
    return {"ok": True}


@app.post("/models/{model_name}/rollback")
def rollback_model(model_name: str, current=Depends(require_admin)):
    ok = db.rollback_model_family(model_name)
    if not ok:
        raise HTTPException(status_code=400, detail="Rollback not possible")
    return {"ok": True}
