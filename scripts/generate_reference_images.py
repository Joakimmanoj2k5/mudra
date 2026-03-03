"""Generate ISL alphabet reference images programmatically.

Creates clean, annotated reference images for each ISL alphabet gesture
showing the hand configuration description on a styled card.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def _wrap_text(text: str, max_chars: int = 40) -> list[str]:
    """Word-wrap text to max_chars per line."""
    words = text.split()
    lines = []
    current = ""
    for w in words:
        if current and len(current) + 1 + len(w) > max_chars:
            lines.append(current)
            current = w
        else:
            current = f"{current} {w}".strip()
    if current:
        lines.append(current)
    return lines


def _generate_card(gesture_name: str, description: str, tips: str,
                   difficulty: str, hand: str, width: int = 480, height: int = 400) -> np.ndarray:
    """Generate a styled reference card image."""
    # Dark background
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (23, 23, 33)  # Dark blue-gray

    # Header background
    cv2.rectangle(img, (0, 0), (width, 70), (16, 185, 129), -1)  # Emerald green

    # Title
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_text = f"ISL: {gesture_name}"
    cv2.putText(img, title_text, (20, 48), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

    # Difficulty and hand info
    y = 105
    info_text = f"Difficulty: {difficulty.capitalize()} | Hand: {hand}"
    cv2.putText(img, info_text, (20, y), font, 0.5, (148, 163, 184), 1, cv2.LINE_AA)

    # Description
    y += 35
    cv2.putText(img, "How to sign:", (20, y), font, 0.55, (251, 191, 36), 1, cv2.LINE_AA)
    y += 10
    desc_lines = _wrap_text(description, 50)
    for line in desc_lines:
        y += 25
        if y > height - 60:
            break
        cv2.putText(img, line, (20, y), font, 0.48, (226, 232, 240), 1, cv2.LINE_AA)

    # Tips
    if tips and y < height - 80:
        y += 35
        cv2.putText(img, "Tips:", (20, y), font, 0.55, (251, 191, 36), 1, cv2.LINE_AA)
        tip_lines = _wrap_text(tips, 50)
        for line in tip_lines:
            y += 25
            if y > height - 20:
                break
            cv2.putText(img, line, (20, y), font, 0.45, (203, 213, 225), 1, cv2.LINE_AA)

    # Border
    cv2.rectangle(img, (0, 0), (width - 1, height - 1), (51, 65, 85), 2)

    return img


def generate_all_reference_images():
    """Generate reference card images for all gestures."""
    ref_path = Path("data/assets/gestures/isl_reference_data.json")
    if not ref_path.exists():
        print("Reference data not found!")
        return

    data = json.loads(ref_path.read_text(encoding="utf-8"))
    out_alpha = Path("data/assets/gestures/image_cache")
    out_alpha.mkdir(parents=True, exist_ok=True)

    count = 0

    # Alphabets
    for letter, info in data.get("alphabets", {}).items():
        img = _generate_card(
            gesture_name=letter,
            description=info.get("description", ""),
            tips=info.get("tips", ""),
            difficulty=info.get("difficulty", "beginner"),
            hand=info.get("hand", "right"),
        )
        path = out_alpha / f"{letter.lower()}.png"
        cv2.imwrite(str(path), img)
        count += 1

    # Words
    for word, info in data.get("words", {}).items():
        img = _generate_card(
            gesture_name=word,
            description=info.get("description", ""),
            tips=info.get("tips", ""),
            difficulty=info.get("difficulty", "beginner"),
            hand=info.get("hands", info.get("hand", "right")),
            height=440,  # Slightly taller for longer descriptions
        )
        slug = word.strip().replace(" ", "_").lower()
        path = out_alpha / f"{slug}.png"
        cv2.imwrite(str(path), img)
        count += 1

    print(f"Generated {count} reference images in {out_alpha}")


if __name__ == "__main__":
    generate_all_reference_images()
