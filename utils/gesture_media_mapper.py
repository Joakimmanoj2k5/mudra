"""Resolve gesture names to reference media assets and descriptions."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Optional

from utils.common.gesture_catalog import all_gestures

ASSET_ROOT = Path("data/assets/gestures")
IMAGE_CACHE = ASSET_ROOT / "image_cache"
PLACEHOLDER_IMAGE = ASSET_ROOT / "placeholder.png"
_REF_DATA_PATH = ASSET_ROOT / "isl_reference_data.json"
_ref_cache: Optional[Dict] = None
_gesture_code_map_cache: Optional[Dict[str, str]] = None


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


def _gesture_code_map() -> Dict[str, str]:
    global _gesture_code_map_cache
    if _gesture_code_map_cache is not None:
        return _gesture_code_map_cache
    mapping: Dict[str, str] = {}
    for spec in all_gestures():
        mapping[spec.code.upper()] = spec.display_name
    _gesture_code_map_cache = mapping
    return mapping


def _normalize_gesture_name(gesture_code_or_name: str) -> str:
    """Normalize DB gesture_code or display_name to catalog display_name."""
    value = (gesture_code_or_name or "").strip()
    if not value:
        return ""

    code_key = value.upper()
    code_map = _gesture_code_map()
    if code_key in code_map:
        return code_map[code_key]

    if code_key.startswith("ALPHABET_"):
        tail = code_key.split("_", 1)[1].strip()
        if len(tail) == 1 and tail.isalpha():
            return tail

    if code_key.startswith("WORD_"):
        phrase = code_key.split("_", 1)[1].replace("_", " ").strip().lower()
        return " ".join(tok.capitalize() for tok in phrase.split())

    return value


def get_reference_image_path(gesture_code_or_name: str) -> Optional[str]:
    """Return path to a cached reference image for the gesture, or None."""
    gesture_name = _normalize_gesture_name(gesture_code_or_name)
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


def get_media_path(gesture_code_or_name: str) -> Optional[str]:
    """Return the best available media file for *gesture_name*.

    Priority order:
        1. mp4 animation (generated reference)
        2. gif animation
        3. cached png image (from image_cache)
        4. placeholder image
    """
    gesture_name = _normalize_gesture_name(gesture_code_or_name)
    if not gesture_name:
        return str(PLACEHOLDER_IMAGE) if PLACEHOLDER_IMAGE.exists() else None

    slug = _slug(gesture_name)

    # --- 1. MP4 animations ---
    if len(gesture_name) == 1 and gesture_name.isalpha():
        for name in (gesture_name.upper(), gesture_name.lower()):
            p = ASSET_ROOT / "alphabets" / f"{name}.mp4"
            if p.exists():
                return str(p)

    mp4_candidates = [
        ASSET_ROOT / "words" / f"{slug}.mp4",
        ASSET_ROOT / "alphabets" / f"{slug.upper()}.mp4",
        ASSET_ROOT / "alphabets" / f"{slug.lower()}.mp4",
    ]
    for c in mp4_candidates:
        if c.exists():
            return str(c)

    # --- 2. GIF animations ---
    gif_candidates = [
        ASSET_ROOT / "words" / f"{slug}.gif",
        ASSET_ROOT / "alphabets" / f"{slug}.gif",
        ASSET_ROOT / "alphabets" / f"{slug.upper()}.gif",
    ]
    for c in gif_candidates:
        if c.exists():
            return str(c)

    # --- 3. Cached PNG image ---
    img = get_reference_image_path(gesture_name)
    if img is not None:
        return img

    # --- 4. Placeholder image ---
    if PLACEHOLDER_IMAGE.exists():
        return str(PLACEHOLDER_IMAGE)
    return None


def get_gesture_reference(gesture_code_or_name: str) -> Dict[str, str]:
    """Return reference info (description, tips, difficulty) for a gesture.

    Always returns a dict with at least 'description' and 'tips' keys.
    """
    gesture_name = _normalize_gesture_name(gesture_code_or_name)
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


def get_gesture_description(gesture_code_or_name: str) -> str:
    """Convenience: return just the description string."""
    return get_gesture_reference(gesture_code_or_name).get("description", "")
