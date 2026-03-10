#!/usr/bin/env python3
"""Capture live webcam landmark templates for each ISL alphabet gesture.

Run this script, show each gesture when prompted, and it will save
feature templates that perfectly match your camera environment.
Press SPACE to capture (takes 15 frames), ESC to skip a letter.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.mediapipe.hand_tracker import HandTracker
from inference.preprocess.normalize import build_feature_vector


def main():
    tracker = HandTracker()
    if not tracker.available or tracker._landmarker is None:
        print("ERROR: MediaPipe not available")
        return

    label_map = json.loads(Path("models/registry/label_map.json").read_text())
    ns = json.loads(Path("models/registry/norm_stats.json").read_text())
    mean = np.array(ns["mean"], dtype=np.float32)
    std = np.array(ns["std"], dtype=np.float32)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    letters = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    all_features = {}  # class_idx -> list of feature vectors
    captures_per_letter = 20  # frames to capture per letter

    print("\n=== ISL Alphabet Calibration ===")
    print("For each letter, show the gesture and press SPACE to capture.")
    print("Press 'S' to skip a letter. Press 'Q' to quit early.\n")

    for letter in letters:
        if letter not in label_map:
            continue
        cls_idx = label_map[letter]
        all_features[cls_idx] = []

        print(f"\n>>> Show gesture for '{letter}' and press SPACE to capture...")

        waiting = True
        while waiting:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)

            extraction = tracker.extract(frame)
            status_text = "NO HAND" if extraction["status"] != "ok" else "HAND DETECTED"

            # Draw overlay
            display = frame.copy()
            cv2.putText(display, f"Letter: {letter}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(display, f"{status_text} - Press SPACE to capture",
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, "S=skip  Q=quit",
                        (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
            cv2.imshow("Calibration", display)

            key = cv2.waitKey(30) & 0xFF
            if key == ord(" "):  # SPACE - start capture
                print(f"  Capturing {captures_per_letter} frames for '{letter}'...")
                captured = 0
                for _ in range(captures_per_letter * 3):  # Try up to 3x frames
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    frame = cv2.flip(frame, 1)
                    extraction = tracker.extract(frame)
                    if extraction["status"] == "ok":
                        feat = build_feature_vector(extraction)
                        feat_norm = (feat - mean) / std
                        all_features[cls_idx].append(feat_norm)
                        captured += 1

                        display = frame.copy()
                        cv2.putText(display, f"Capturing {letter}: {captured}/{captures_per_letter}",
                                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                        cv2.imshow("Calibration", display)
                        cv2.waitKey(30)

                        if captured >= captures_per_letter:
                            break
                print(f"  Captured {captured} frames for '{letter}'")
                waiting = False
            elif key == ord("s") or key == ord("S"):
                print(f"  Skipped '{letter}'")
                waiting = False
            elif key == ord("q") or key == ord("Q"):
                print("  Quit early.")
                cap.release()
                cv2.destroyAllWindows()
                _save(all_features, label_map)
                return

    cap.release()
    cv2.destroyAllWindows()
    _save(all_features, label_map)


def _save(all_features, label_map):
    """Merge captured features with existing exemplars and rebuild centroids."""
    out_dir = Path("models/registry")
    n_cls = max(label_map.values()) + 1

    # Load existing exemplars
    existing = {}
    ep = out_dir / "static_exemplars.npz"
    if ep.exists():
        edata = np.load(ep)
        for key in edata.files:
            cls_idx = int(key.split("_")[1])
            existing[cls_idx] = list(edata[key])

    # Merge new captures
    total_new = 0
    for cls_idx, feats in all_features.items():
        if not feats:
            continue
        if cls_idx not in existing:
            existing[cls_idx] = []
        existing[cls_idx].extend(feats)
        total_new += len(feats)

    if total_new == 0:
        print("No new features captured.")
        return

    # Rebuild exemplars
    exemplar_arrays = {}
    for cls_idx, feats in existing.items():
        exemplar_arrays[f"class_{cls_idx}"] = np.stack(feats, axis=0).astype(np.float32)
    np.savez(out_dir / "static_exemplars.npz", **exemplar_arrays)

    # Rebuild centroids
    centroids = np.zeros((n_cls, 136), dtype=np.float32)
    counts = np.zeros(n_cls, dtype=np.int32)
    for cls_idx, feats in existing.items():
        arr = np.stack(feats, axis=0)
        centroids[cls_idx] = arr.mean(axis=0)
        norm = np.linalg.norm(centroids[cls_idx])
        if norm > 1e-8:
            centroids[cls_idx] /= norm
        counts[cls_idx] = len(feats)
    np.savez(out_dir / "static_centroids.npz", centroids=centroids, counts=counts)

    idx_to_label = {v: k for k, v in label_map.items()}
    print(f"\nSaved {total_new} new templates. Updated exemplars and centroids.")
    for cls_idx in sorted(all_features.keys()):
        name = idx_to_label.get(cls_idx, f"cls_{cls_idx}")
        print(f"  {name}: {len(all_features[cls_idx])} new, {counts[cls_idx]} total")


if __name__ == "__main__":
    main()
