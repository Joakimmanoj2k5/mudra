#!/usr/bin/env python3
"""Generate looping MP4 reference animations from cached ISL gesture images.

Reads the gesture catalog and produces a 3-second looping animation for each
gesture using the corresponding PNG in ``data/assets/gestures/image_cache/``.
The source cache is the project's ISL-only reference set.

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
DURATION_S = 3
TOTAL_FRAMES = FPS * DURATION_S  # 90

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


def _highlight_hand(frame: np.ndarray) -> np.ndarray:
    """Draw a subtle contour highlight around probable hand pixels."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0, 18, 40), (25, 255, 255))
    mask2 = cv2.inRange(hsv, (160, 18, 40), (179, 255, 255))
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.medianBlur(mask, 7)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return frame

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 1000:
        return frame

    overlay = frame.copy()
    cv2.drawContours(overlay, [largest], -1, (74, 222, 128), 2, cv2.LINE_AA)
    x, y, w, h = cv2.boundingRect(largest)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (34, 197, 94), 1, cv2.LINE_AA)
    return cv2.addWeighted(frame, 0.85, overlay, 0.15, 0)


def _draw_motion_arrow(frame: np.ndarray, t: float) -> np.ndarray:
    """Render a soft directional arrow to hint dynamic movement."""
    overlay = frame.copy()
    base_x = int(RESOLUTION * 0.18)
    base_y = int(RESOLUTION * 0.86)
    drift = int(18 * np.sin(2 * np.pi * t))
    start = (base_x + drift, base_y)
    end = (base_x + 120 + drift, base_y - 30)
    cv2.arrowedLine(overlay, start, end, (21, 204, 250), 3, cv2.LINE_AA, tipLength=0.25)
    cv2.putText(
        overlay,
        "ISL motion",
        (start[0] - 2, start[1] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (148, 163, 184),
        1,
        cv2.LINE_AA,
    )
    return cv2.addWeighted(frame, 0.82, overlay, 0.18, 0)


def _ensure_placeholder() -> None:
    """Create a fallback placeholder image used by the media mapper."""
    placeholder = ASSET_ROOT / "placeholder.png"
    if placeholder.exists():
        return
    img = np.zeros((RESOLUTION, RESOLUTION, 3), dtype=np.uint8)
    img[:] = (15, 23, 42)
    cv2.rectangle(img, (24, 24), (RESOLUTION - 24, RESOLUTION - 24), (51, 65, 85), 2)
    cv2.putText(img, "ISL Reference", (140, 236), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (34, 197, 94), 2, cv2.LINE_AA)
    cv2.putText(img, "Animation Missing", (126, 282), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (203, 213, 225), 2, cv2.LINE_AA)
    placeholder.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(placeholder), img)


def _generate_animation(src_path: Path, dst_path: Path) -> bool:
    """Generate a 3-second looping MP4 from a source image.

    Effects applied per-frame:
    * slow zoom in (scale 1.0 → 1.1)
    * hand contour highlight
    * movement hint arrow
    * soft Gaussian glow
    * subtle fade-in / fade-out for loop continuity
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
        t = i / max(TOTAL_FRAMES - 1, 1)  # 0.0 -> 1.0

        # ---- zoom ----
        scale = ZOOM_START + (ZOOM_END - ZOOM_START) * t
        crop_size = int(RESOLUTION / scale)
        offset = (work_size - crop_size) // 2
        cropped = img_resized[offset:offset + crop_size, offset:offset + crop_size]
        frame = cv2.resize(cropped, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_LINEAR)

        # ---- glow ----
        frame = _apply_glow(frame)
        frame = _highlight_hand(frame)
        frame = _draw_motion_arrow(frame, t)

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
    _ensure_placeholder()

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
