"""Resolve gesture names to reference media assets and descriptions."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

ASSET_ROOT = Path("data/assets/gestures")
IMAGE_CACHE = ASSET_ROOT / "image_cache"
_REF_DATA_PATH = ASSET_ROOT / "isl_reference_data.json"
_ref_cache: Optional[Dict] = None


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    return re.sub(r"_+", "_", cleaned).strip("_")


def _load_reference_data() -> Dict:
    global _ref_cache
    if _ref_cache is not None:
        return _ref_cache
    if _REF_DATA_PATH.exists():
        try:
            _ref_cache = json.loads(_REF_DATA_PATH.read_text(encoding="utf-8"))
        except Exception:
            _ref_cache = {}
    else:
        _ref_cache = {}
    return _ref_cache


def get_reference_image_path(gesture_name: str) -> Optional[str]:
    """Return path to a cached reference image for the gesture, or None."""
    if not gesture_name:
        return None

    slug = _slug(gesture_name)
    for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
        p = IMAGE_CACHE / f"{slug}{ext}"
        if p.exists():
            return str(p)

    # Single letter alphabets
    if len(gesture_name) == 1 and gesture_name.isalpha():
        for ext in (".png", ".jpg", ".jpeg"):
            p = IMAGE_CACHE / f"{gesture_name.lower()}{ext}"
            if p.exists():
                return str(p)
    return None


def get_media_path(gesture_name: str) -> Optional[str]:
    """Return the best available media file for *gesture_name*.

    Priority order:
        1. mp4 animation (generated reference)
        2. gif animation
        3. cached png image (from image_cache)
        4. placeholder image (None — caller handles fallback)
    """
    if not gesture_name:
        return None

    slug = _slug(gesture_name)

    # --- 1. MP4 animations ---
    if len(gesture_name) == 1 and gesture_name.isalpha():
        p = ASSET_ROOT / "alphabets" / f"{gesture_name.upper()}.mp4"
        if p.exists():
            return str(p)

    mp4_candidates = [
        ASSET_ROOT / "words" / f"{slug}.mp4",
        ASSET_ROOT / "alphabets" / f"{slug.upper()}.mp4",
    ]
    for c in mp4_candidates:
        if c.exists():
            return str(c)

    # --- 2. GIF animations ---
    gif_candidates = [
        ASSET_ROOT / "words" / f"{slug}.gif",
        ASSET_ROOT / "alphabets" / f"{slug}.gif",
    ]
    for c in gif_candidates:
        if c.exists():
            return str(c)

    # --- 3. Cached PNG image ---
    img = get_reference_image_path(gesture_name)
    if img is not None:
        return img

    # --- 4. No media found ---
    return None


def get_gesture_reference(gesture_name: str) -> Dict[str, str]:
    """Return reference info (description, tips, difficulty) for a gesture.

    Always returns a dict with at least 'description' and 'tips' keys.
    """
    if not gesture_name:
        return {"description": "No gesture selected.", "tips": "", "difficulty": ""}

    data = _load_reference_data()

    # Check alphabets
    if len(gesture_name) == 1 and gesture_name.isalpha():
        entry = data.get("alphabets", {}).get(gesture_name.upper(), {})
        if entry:
            return {
                "description": entry.get("description", ""),
                "tips": entry.get("tips", ""),
                "difficulty": entry.get("difficulty", "beginner"),
                "hand": entry.get("hand", "right"),
            }

    # Check words
    entry = data.get("words", {}).get(gesture_name, {})
    if entry:
        return {
            "description": entry.get("description", ""),
            "tips": entry.get("tips", ""),
            "difficulty": entry.get("difficulty", "beginner"),
            "hands": entry.get("hands", "right"),
            "mode": entry.get("mode", "static"),
            "category": entry.get("category", ""),
        }

    return {
        "description": f"Practice the ISL sign for '{gesture_name}'. Keep your hand clearly visible to the camera.",
        "tips": "Position your hand in the center of the frame with good lighting.",
        "difficulty": "",
    }


def get_gesture_description(gesture_name: str) -> str:
    """Convenience: return just the description string."""
    return get_gesture_reference(gesture_name).get("description", "")
