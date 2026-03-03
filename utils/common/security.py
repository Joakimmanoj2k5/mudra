"""Lightweight password and token helpers using stdlib only."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Optional


SECRET_KEY = os.getenv("MUDRA_SECRET_KEY", "mudra-dev-secret-change-me")
ACCESS_TOKEN_SECONDS = 60 * 60 * 2


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    return f"{salt}${digest}"


def verify_password(password: str, password_hash: str) -> bool:
    try:
        salt, digest = password_hash.split("$", 1)
    except ValueError:
        return False
    check = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    return hmac.compare_digest(check, digest)


def create_access_token(subject: str, expires_seconds: int = ACCESS_TOKEN_SECONDS) -> str:
    payload = {"sub": subject, "exp": int(time.time()) + expires_seconds}
    payload_b = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    payload_enc = base64.urlsafe_b64encode(payload_b).decode("utf-8")
    sig = hmac.new(SECRET_KEY.encode("utf-8"), payload_enc.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload_enc}.{sig}"


def decode_access_token(token: str) -> Optional[str]:
    try:
        payload_enc, sig = token.split(".", 1)
    except ValueError:
        return None

    expected = hmac.new(SECRET_KEY.encode("utf-8"), payload_enc.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return None

    try:
        payload = json.loads(base64.urlsafe_b64decode(payload_enc.encode("utf-8")).decode("utf-8"))
    except Exception:
        return None

    if int(payload.get("exp", 0)) < int(time.time()):
        return None
    return payload.get("sub")
