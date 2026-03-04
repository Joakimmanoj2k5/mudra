#!/usr/bin/env python3
"""Generate training data for MUDRA gesture recognition models.

Produces augmented reference images in ``data/raw/`` organised by class, then
attempts MediaPipe landmark extraction.  Where extraction fails (e.g. no hand
detected in an illustration), **synthetic landmark feature vectors** are
generated so downstream training scripts always have sufficient data.

Usage::

    python scripts/generate_training_data.py [--samples-per-class 50]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, str(_PROJECT_ROOT))

from utils.common.gesture_catalog import all_gestures, GestureSpec
from utils.logging.logger import configure_logger

logger = configure_logger("mudra.generate_training_data")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
IMAGE_CACHE = Path("data/assets/gestures/image_cache")
RAW_ROOT = Path("data/raw")
LANDMARK_DIR = Path("data/interim/landmarks")
MANIFEST_PATH = Path("data/interim/landmarks_manifest.json")

FEATURE_DIM = 136  # 21*3*2 + 10 angles

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    return re.sub(r"_+", "_", cleaned).strip("_")


def _find_source_image(spec: GestureSpec) -> Path | None:
    if spec.lesson_type == "alphabet" and len(spec.display_name) == 1:
        c = IMAGE_CACHE / f"{spec.display_name.lower()}.png"
        if c.exists():
            return c
    slug = _slug(spec.display_name)
    for ext in (".png", ".jpg", ".jpeg"):
        c = IMAGE_CACHE / f"{slug}{ext}"
        if c.exists():
            return c
    return None


def _augment_image(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply random augmentation to an image."""
    h, w = img.shape[:2]

    # Random brightness
    beta = rng.uniform(-30, 30)
    aug = np.clip(img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # Random rotation (-15 to +15 degrees)
    angle = rng.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    # Random scale crop
    scale = rng.uniform(0.85, 1.0)
    cw, ch = int(w * scale), int(h * scale)
    x0 = rng.integers(0, max(w - cw, 1))
    y0 = rng.integers(0, max(h - ch, 1))
    aug = aug[y0:y0 + ch, x0:x0 + cw]
    aug = cv2.resize(aug, (w, h), interpolation=cv2.INTER_LINEAR)

    # Random horizontal flip (50%)
    if rng.random() < 0.5:
        aug = cv2.flip(aug, 1)

    # Gaussian noise
    noise = rng.normal(0, 3, aug.shape).astype(np.float32)
    aug = np.clip(aug.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return aug


# ---------------------------------------------------------------------------
# Step 1: Generate augmented images in data/raw/
# ---------------------------------------------------------------------------


def generate_raw_images(samples_per_class: int) -> int:
    """Create augmented copies of reference images in ``data/raw/``."""
    gestures = all_gestures()
    total_generated = 0
    rng = np.random.default_rng(42)

    for spec in gestures:
        src = _find_source_image(spec)
        if src is None:
            continue

        if spec.lesson_type == "alphabet":
            class_dir = RAW_ROOT / "alphabets" / spec.display_name.upper()
        else:
            class_dir = RAW_ROOT / "words" / _slug(spec.display_name)
        class_dir.mkdir(parents=True, exist_ok=True)

        # Check if already populated
        existing = list(class_dir.glob("*.png"))
        if len(existing) >= samples_per_class:
            total_generated += len(existing)
            continue

        img = cv2.imread(str(src))
        if img is None:
            continue

        # Save original
        cv2.imwrite(str(class_dir / "ref_000.png"), img)
        for i in range(1, samples_per_class):
            aug = _augment_image(img, rng)
            cv2.imwrite(str(class_dir / f"aug_{i:03d}.png"), aug)
        total_generated += samples_per_class

    logger.info("Generated %d raw images across %d classes", total_generated, len(gestures))
    return total_generated


# ---------------------------------------------------------------------------
# Step 2: Extract landmarks via MediaPipe (best-effort)
# ---------------------------------------------------------------------------


def _try_mediapipe_extraction() -> list[dict]:
    """Attempt to extract landmarks from data/raw/ using MediaPipe.

    Returns list of manifest entries.  If MediaPipe is unavailable or fails
    on most images, the list will be short/empty.
    """
    try:
        from inference.mediapipe.hand_tracker import HandTracker
        from inference.preprocess.normalize import build_feature_vector
    except Exception as exc:
        logger.warning("MediaPipe import failed (%s); will use synthetic landmarks.", exc)
        return []

    tracker = HandTracker()
    if not tracker.available:
        logger.warning("MediaPipe hand tracker unavailable; will use synthetic landmarks.")
        return []

    manifest: list[dict] = []
    LANDMARK_DIR.mkdir(parents=True, exist_ok=True)

    for media_path in sorted(RAW_ROOT.rglob("*.png")):
        cls_name = media_path.parent.name
        frame = cv2.imread(str(media_path))
        if frame is None:
            continue
        ex = tracker.extract(frame)
        if ex["status"] != "ok":
            continue
        fv = build_feature_vector(ex)

        out_cls = LANDMARK_DIR / cls_name
        out_cls.mkdir(parents=True, exist_ok=True)
        out_path = out_cls / f"{media_path.stem}.npy"
        np.save(out_path, fv.reshape(1, -1))
        manifest.append({"class": cls_name, "file": str(out_path), "frames": 1})

    logger.info("MediaPipe extracted landmarks for %d images", len(manifest))
    return manifest


# ---------------------------------------------------------------------------
# Step 3: Synthetic landmark generation (fallback / supplement)
# ---------------------------------------------------------------------------

# Feature vector layout (136-dim):
#   [0:63]   – left hand: 21 landmarks × 3 coords (flattened)
#   [63:126] – right hand: 21 landmarks × 3 coords (flattened)
#   [126:136] – 10 joint angle features (5 triplets × 2 hands)


def _build_class_prototypes(gestures: list[GestureSpec]) -> dict[str, np.ndarray]:
    """Create a distinct 136-dim prototype feature vector for each class.

    Strategy:
    * Generate prototypes that are well-separated in feature space.
    * Right-hand coordinates are drawn from a class-specific random seed.
    * Left-hand is zeros for single-hand gestures, non-zero for two-hand.
    * Angle features are derived from the landmark coordinates.
    """
    n_classes = len(gestures)
    prototypes: dict[str, np.ndarray] = {}

    # Use a fixed seed per class for reproducibility
    master_rng = np.random.default_rng(12345)

    for class_idx, spec in enumerate(gestures):
        cls_rng = np.random.default_rng(class_idx * 7919 + 42)  # unique per class

        # --- Right hand: 21 landmarks × 3 ---
        # Start with a canonical hand skeleton and apply class-specific deformation
        right = _make_hand_pose(cls_rng)

        # --- Left hand ---
        if spec.requires_two_hands:
            left = _make_hand_pose(np.random.default_rng(class_idx * 7919 + 99999))
        else:
            left = np.zeros((21, 3), dtype=np.float32)

        fv = _assemble_feature_vector(right, left)
        prototypes[spec.display_name] = fv

    # Verify inter-class distances
    names = list(prototypes.keys())
    vecs = np.stack([prototypes[n] for n in names])
    dists = np.linalg.norm(vecs[:, None, :] - vecs[None, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min()
    logger.info("Class prototypes: %d classes, min inter-class distance=%.4f", n_classes, min_dist)

    return prototypes


def _make_hand_pose(rng: np.random.Generator) -> np.ndarray:
    """Generate a 21-landmark hand pose with class-specific finger configuration.

    The pose is wrist-centred, scaled so that wrist-to-middle_mcp ≈ 1.0,
    exactly as ``normalize.py`` produces.
    """
    hand = np.zeros((21, 3), dtype=np.float32)

    # Palm base: MCP joints spread along the x-axis
    hand[5]  = np.array([0.30, -0.80, 0.0])   # index MCP
    hand[9]  = np.array([0.00, -1.00, 0.0])   # middle MCP
    hand[13] = np.array([-0.20, -0.90, 0.0])  # ring MCP
    hand[17] = np.array([-0.40, -0.70, 0.0])  # pinky MCP
    hand[1]  = np.array([0.50, -0.30, 0.0])   # thumb CMC

    # Per-finger flex: each finger independently extended (0) or curled (1)
    finger_configs = {
        "thumb":  (np.array([0.35, -0.50, 0.05]), 0.30, [1, 2, 3, 4]),
        "index":  (np.array([0.15, -1.00, 0.0]),  0.25, [5, 6, 7, 8]),
        "middle": (np.array([0.00, -1.00, 0.0]),  0.27, [9, 10, 11, 12]),
        "ring":   (np.array([-0.10, -1.00, 0.0]), 0.24, [13, 14, 15, 16]),
        "pinky":  (np.array([-0.20, -0.90, 0.0]), 0.20, [17, 18, 19, 20]),
    }

    for _name, (direction, seg_len, indices) in finger_configs.items():
        # Class-specific flex: 0=fully extended, 1=fully curled
        flex = rng.uniform(0.0, 1.0, 3)
        d = direction / (np.linalg.norm(direction) + 1e-8)
        pos = hand[indices[0]].copy()
        for j, joint_idx in enumerate(indices[1:]):
            curl = flex[j]
            curl_d = d.copy()
            curl_d[1] += curl * 1.0  # curl toward palm
            curl_d[2] += rng.uniform(-0.15, 0.15)  # z-depth variation
            curl_d = curl_d / (np.linalg.norm(curl_d) + 1e-8)
            pos = pos + curl_d * seg_len
            hand[joint_idx] = pos

    # Global class-specific perturbation (rotation + translation in normalised space)
    angle = rng.uniform(-0.4, 0.4)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    hand[:, :2] = hand[:, :2] @ rot.T

    # Small global offset
    hand += rng.normal(0, 0.05, hand.shape).astype(np.float32)

    return hand


def _assemble_feature_vector(right: np.ndarray, left: np.ndarray) -> np.ndarray:
    """Build a 136-dim feature vector from (21,3) left and right hand arrays."""
    base = np.concatenate([left.reshape(-1), right.reshape(-1)], axis=0)  # 126

    angles: list[float] = []
    triplets = [(0, 5, 8), (0, 9, 12), (0, 13, 16), (0, 17, 20), (1, 2, 4)]
    for hand in [left, right]:
        for a, b, c in triplets:
            v1 = hand[a] - hand[b]
            v2 = hand[c] - hand[b]
            denom = np.linalg.norm(v1) * np.linalg.norm(v2)
            if denom < 1e-6:
                angles.append(0.0)
            else:
                cos_val = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
                angles.append(float(np.arccos(cos_val)))

    return np.concatenate([base, np.array(angles, dtype=np.float32)], axis=0)


def generate_synthetic_landmarks(
    samples_per_class: int,
    existing_manifest: list[dict] | None = None,
) -> list[dict]:
    """Generate synthetic landmark .npy files for ALL gesture classes.

    Uses fixed per-class prototypes with small per-sample Gaussian noise
    to ensure strong inter-class separability.

    If ``existing_manifest`` already covers a class with enough samples,
    that class is skipped.
    """
    gestures = all_gestures()

    # Build fixed class prototypes (deterministic per class_idx)
    prototypes = _build_class_prototypes(gestures)

    # Count existing samples per class
    existing_counts: dict[str, int] = {}
    if existing_manifest:
        for entry in existing_manifest:
            cls = entry["class"]
            existing_counts[cls] = existing_counts.get(cls, 0) + entry.get("frames", 1)

    manifest: list[dict] = list(existing_manifest or [])
    LANDMARK_DIR.mkdir(parents=True, exist_ok=True)

    # Per-sample noise scale (small relative to inter-class distance)
    SAMPLE_NOISE_STD = 0.015

    for class_idx, spec in enumerate(gestures):
        cls_name = spec.display_name
        prototype = prototypes[cls_name]  # 136-dim

        existing_n = existing_counts.get(cls_name, 0)
        needed = max(0, samples_per_class - existing_n)
        if needed == 0:
            continue

        out_cls = LANDMARK_DIR / cls_name
        out_cls.mkdir(parents=True, exist_ok=True)

        is_dynamic = (spec.gesture_mode == "dynamic")
        sample_rng = np.random.default_rng(class_idx * 100000 + 7)

        for s in range(needed):
            if is_dynamic:
                # Dynamic: 30-frame sequence. Each frame = prototype + small noise
                # Add a smooth temporal drift on top of per-frame noise
                seq_len = 30
                drift_direction = sample_rng.normal(0, 0.01, prototype.shape).astype(np.float32)
                frames = []
                for t in range(seq_len):
                    progress = t / max(seq_len - 1, 1)
                    noise = sample_rng.normal(0, SAMPLE_NOISE_STD, prototype.shape).astype(np.float32)
                    frame_fv = prototype + noise + drift_direction * progress
                    frames.append(frame_fv)
                arr = np.stack(frames, axis=0)  # (30, 136)
            else:
                # Static: single frame = prototype + small noise
                noise = sample_rng.normal(0, SAMPLE_NOISE_STD, prototype.shape).astype(np.float32)
                fv = prototype + noise
                arr = fv.reshape(1, -1)  # (1, 136)

            out_path = out_cls / f"syn_{s:04d}.npy"
            np.save(out_path, arr)
            manifest.append({"class": cls_name, "file": str(out_path), "frames": int(arr.shape[0])})

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(
        "Synthetic landmark manifest ready: %d entries across %d classes",
        len(manifest),
        len({e["class"] for e in manifest}),
    )
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MUDRA training data")
    parser.add_argument("--samples-per-class", type=int, default=50,
                        help="Number of training samples per gesture class")
    parser.add_argument("--skip-images", action="store_true",
                        help="Skip augmented image generation (raw/)")
    parser.add_argument("--skip-mediapipe", action="store_true",
                        help="Skip MediaPipe extraction, use only synthetic landmarks")
    args = parser.parse_args()

    n = args.samples_per_class

    # Step 1: Generate augmented images
    if not args.skip_images:
        print("=" * 60)
        print("Step 1: Generating augmented images in data/raw/")
        print("=" * 60)
        total_img = generate_raw_images(n)
        print(f"  Raw images ready: {total_img}")
    else:
        print("Skipping raw image generation.")

    # Step 2: Attempt MediaPipe extraction
    mp_manifest: list[dict] = []
    if not args.skip_mediapipe:
        print()
        print("=" * 60)
        print("Step 2: Attempting MediaPipe landmark extraction")
        print("=" * 60)
        mp_manifest = _try_mediapipe_extraction()
        print(f"  MediaPipe extracted: {len(mp_manifest)} samples")
    else:
        print("Skipping MediaPipe extraction.")

    # Step 3: Generate/supplement with synthetic landmarks
    print()
    print("=" * 60)
    print("Step 3: Generating synthetic landmarks")
    print("=" * 60)
    manifest = generate_synthetic_landmarks(n, existing_manifest=mp_manifest)
    n_classes = len({e["class"] for e in manifest})
    print(f"  Total manifest entries: {len(manifest)}")
    print(f"  Classes covered: {n_classes}")
    print(f"  Manifest saved: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
