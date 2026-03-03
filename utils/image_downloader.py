"""Download and cache ISL reference images for offline display."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

CACHE_DIR = Path("data/assets/gestures/image_cache")


def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


def get_cached_image_path(gesture_name: str) -> Optional[str]:
    """Return cached image path for a gesture if it exists."""
    _ensure_cache_dir()
    # Look for any cached image for this gesture
    slug = gesture_name.strip().replace(" ", "_").lower()
    for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
        path = CACHE_DIR / f"{slug}{ext}"
        if path.exists():
            return str(path)
    return None


def download_image(url: str, gesture_name: str) -> Optional[str]:
    """Download an image from URL and cache it locally. Returns the local path."""
    _ensure_cache_dir()
    slug = gesture_name.strip().replace(" ", "_").lower()

    # Determine extension from URL
    ext = ".png"
    for possible_ext in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
        if possible_ext in url.lower():
            ext = possible_ext
            break

    local_path = CACHE_DIR / f"{slug}{ext}"
    if local_path.exists():
        return str(local_path)

    try:
        import urllib.request
        import ssl

        # Create SSL context that allows self-signed certs
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) MUDRA/1.0"
        })
        with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
            data = resp.read()
            if len(data) < 100:  # Too small, probably an error
                return None
            local_path.write_bytes(data)
            return str(local_path)
    except Exception as e:
        print(f"[ImageDownloader] Failed to download {url}: {e}")
        return None
