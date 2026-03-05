"""Resolve gesture names to reference media assets and descriptions."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Optional

from utils.common.gesture_catalog import all_gestures

ISL_VIDEO_ROOT = Path("isl_videos")
_REF_DATA_PATH = Path("data/assets/gestures/isl_reference_data.json")
_ref_cache: Optional[Dict] = None
_gesture_code_map_cache: Optional[Dict[str, str]] = None
_video_index_cache: Optional[Dict[str, Path]] = None

_NUMBER_STEM_ALIASES: Dict[str, tuple[str, ...]] = {
    "zero": ("0",),
    "one": ("1",),
    "two": ("2",),
    "three": ("3",),
    "four": ("4",),
    "five": ("5",),
    "six": ("6",),
    "seven": ("7",),
    "eight": ("8",),
    "nine": ("9",),
    "ten": ("10",),
}

_VIDEO_ALIASES: Dict[str, tuple[str, ...]] = {
    "i_am_fine": ("fine",),
}


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


def _video_index() -> Dict[str, Path]:
    global _video_index_cache
    if _video_index_cache is not None:
        return _video_index_cache
    index: Dict[str, Path] = {}
    if ISL_VIDEO_ROOT.exists():
        for path in sorted(ISL_VIDEO_ROOT.glob("*.mp4")):
            index[path.stem.lower()] = path
    _video_index_cache = index
    return _video_index_cache


def _candidate_video_stems(gesture_name: str) -> list[str]:
    slug = _slug(gesture_name)
    candidates = [slug]
    if len(gesture_name) == 1 and gesture_name.isalpha():
        candidates.append(gesture_name.lower())
    candidates.extend(_NUMBER_STEM_ALIASES.get(slug, ()))
    candidates.extend(_VIDEO_ALIASES.get(slug, ()))

    seen: set[str] = set()
    ordered: list[str] = []
    for stem in candidates:
        key = stem.lower()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(stem)
    return ordered


def get_media_path(gesture_code_or_name: str) -> Optional[str]:
    """Return the reference video path for a gesture from ``isl_videos/`` only."""
    gesture_name = _normalize_gesture_name(gesture_code_or_name)
    if not gesture_name:
        return None

    index = _video_index()
    for stem in _candidate_video_stems(gesture_name):
        path = index.get(stem.lower())
        if path and path.exists():
            return str(path)
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
