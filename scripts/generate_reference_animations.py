#!/usr/bin/env python3
"""Generate looping MP4 reference animations from cached gesture images.

Reads the gesture catalog and produces a 2-second looping animation for each
gesture using the corresponding PNG in ``data/assets/gestures/image_cache/``.

Usage::

    python scripts/generate_reference_animations.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# Ensure the project root is on sys.path so imports work when invoked directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np

from utils.common.gesture_catalog import all_gestures, GestureSpec
from utils.logging.logger import configure_logger

logger = configure_logger("mudra.generate_animations")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ASSET_ROOT = Path("data/assets/gestures")
IMAGE_CACHE = ASSET_ROOT / "image_cache"
ALPHABETS_DIR = ASSET_ROOT / "alphabets"
WORDS_DIR = ASSET_ROOT / "words"

RESOLUTION = 512
FPS = 30
DURATION_S = 2
TOTAL_FRAMES = FPS * DURATION_S  # 60

ZOOM_START = 1.0
ZOOM_END = 1.1

FOURCC = cv2.VideoWriter_fourcc(*"mp4v")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug(value: str) -> str:
    """Convert a display name to the slug used in image_cache filenames."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    return re.sub(r"_+", "_", cleaned).strip("_")


def _find_source_image(spec: GestureSpec) -> Path | None:
    """Locate the cached PNG for a gesture spec, or return None."""
    # For single-letter alphabets, the cache file is e.g. ``a.png``
    if spec.lesson_type == "alphabet" and len(spec.display_name) == 1:
        candidate = IMAGE_CACHE / f"{spec.display_name.lower()}.png"
        if candidate.exists():
            return candidate

    slug = _slug(spec.display_name)
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = IMAGE_CACHE / f"{slug}{ext}"
        if candidate.exists():
            return candidate

    return None


def _output_path(spec: GestureSpec) -> Path:
    """Determine where the animation MP4 should be saved."""
    if spec.lesson_type == "alphabet":
        return ALPHABETS_DIR / f"{spec.display_name.upper()}.mp4"
    slug = _slug(spec.display_name)
    return WORDS_DIR / f"{slug}.mp4"


def _apply_glow(frame: np.ndarray) -> np.ndarray:
    """Apply a soft glow effect around bright regions (hand area)."""
    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=15, sigmaY=15)
    glow = cv2.addWeighted(frame, 0.85, blurred, 0.15, 0)
    return glow


def _generate_animation(src_path: Path, dst_path: Path) -> bool:
    """Generate a 2-second looping MP4 from a source image.

    Effects applied per-frame:
    * slow zoom in (scale 1.0 → 1.1)
    * soft Gaussian glow
    * subtle fade-in / fade-out for seamless looping
    """
    img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if img is None:
        logger.warning("Could not read image: %s", src_path)
        return False

    # Resize source to a square working canvas (slightly larger for zoom headroom)
    work_size = int(RESOLUTION * ZOOM_END) + 2  # extra pixels to avoid border artifacts
    img_resized = cv2.resize(img, (work_size, work_size), interpolation=cv2.INTER_LANCZOS4)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dst_path), FOURCC, FPS, (RESOLUTION, RESOLUTION))
    if not writer.isOpened():
        logger.error("Could not open VideoWriter for: %s", dst_path)
        return False

    for i in range(TOTAL_FRAMES):
        t = i / max(TOTAL_FRAMES - 1, 1)  # 0.0 → 1.0

        # ---- zoom ----
        scale = ZOOM_START + (ZOOM_END - ZOOM_START) * t
        crop_size = int(RESOLUTION / scale)
        offset = (work_size - crop_size) // 2
        cropped = img_resized[offset:offset + crop_size, offset:offset + crop_size]
        frame = cv2.resize(cropped, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_LINEAR)

        # ---- glow ----
        frame = _apply_glow(frame)

        # ---- fade for seamless loop ----
        fade_frames = 8  # frames over which to fade
        if i < fade_frames:
            alpha = i / fade_frames
            frame = (frame.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
        elif i >= TOTAL_FRAMES - fade_frames:
            alpha = (TOTAL_FRAMES - 1 - i) / fade_frames
            frame = (frame.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)

        writer.write(frame)

    writer.release()
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ALPHABETS_DIR.mkdir(parents=True, exist_ok=True)
    WORDS_DIR.mkdir(parents=True, exist_ok=True)

    gestures = all_gestures()
    generated = 0
    skipped = 0
    missing = 0

    for spec in gestures:
        dst = _output_path(spec)

        # Skip if animation already exists
        if dst.exists():
            skipped += 1
            continue

        src = _find_source_image(spec)
        if src is None:
            logger.warning("Missing image for: %s (expected in %s)", spec.display_name, IMAGE_CACHE)
            missing += 1
            continue

        print(f"Generating animation for: {spec.display_name}")
        ok = _generate_animation(src, dst)
        if ok:
            generated += 1
        else:
            missing += 1

    # Summary
    print()
    print("=" * 48)
    print(f"Animations generated: {generated}")
    print(f"Animations skipped:   {skipped}")
    print(f"Missing images:       {missing}")
    print("=" * 48)
    logger.info(
        "Animation generation complete — generated=%d, skipped=%d, missing=%d",
        generated, skipped, missing,
    )


if __name__ == "__main__":
    main()
