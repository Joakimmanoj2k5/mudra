"""Extract MediaPipe landmarks from image/video datasets into .npy features."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from inference.mediapipe.hand_tracker import HandTracker
from inference.preprocess.normalize import build_feature_vector


def iter_media_files(root: Path):
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.mp4", "*.mov"):
        yield from root.rglob(ext)


def process_file(path: Path, tracker: HandTracker, sample_rate: int = 1, max_frames: int = 0):
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        frame = cv2.imread(str(path))
        if frame is None:
            return []
        ex = tracker.extract(frame)
        if ex["status"] != "ok":
            return []
        return [build_feature_vector(ex)]

    cap = cv2.VideoCapture(str(path))
    rows = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if sample_rate > 1 and frame_idx % sample_rate != 0:
            continue
        ex = tracker.extract(frame)
        if ex["status"] == "ok":
            rows.append(build_feature_vector(ex))
        if max_frames > 0 and len(rows) >= max_frames:
            break
    cap.release()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to data/raw")
    parser.add_argument("--output", default="data/interim/landmarks")
    parser.add_argument("--sample-rate", type=int, default=3,
                        help="Extract every Nth frame from videos (default: 3)")
    parser.add_argument("--max-frames", type=int, default=60,
                        help="Max frames to extract per video (default: 60, 0=unlimited)")
    args = parser.parse_args()

    input_root = Path(args.input)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    tracker = HandTracker()
    manifest = []
    media_files = list(iter_media_files(input_root))
    total = len(media_files)
    skipped = 0
    start_time = time.time()

    print(f"Processing {total} media files (sample_rate={args.sample_rate}, max_frames={args.max_frames})...")

    for i, media_path in enumerate(media_files):
        cls_name = media_path.parent.name
        out_cls = out_root / cls_name
        out_cls.mkdir(parents=True, exist_ok=True)

        feats = process_file(media_path, tracker, args.sample_rate, args.max_frames)
        if not feats:
            skipped += 1
            if (i + 1) % 10 == 0 or i == total - 1:
                elapsed = time.time() - start_time
                print(f"  [{i+1}/{total}] {cls_name}/{media_path.name} → 0 features (skipped)  [{elapsed:.0f}s]")
            continue
        arr = np.stack(feats, axis=0)
        out_path = out_cls / f"{media_path.stem}.npy"
        np.save(out_path, arr)
        manifest.append({"class": cls_name, "file": str(out_path), "frames": len(arr)})

        if (i + 1) % 10 == 0 or i == total - 1:
            elapsed = time.time() - start_time
            print(f"  [{i+1}/{total}] {cls_name}/{media_path.name} → {len(arr)} features  [{elapsed:.0f}s]")

    manifest_path = Path("data/interim/landmarks_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s — Extracted {len(manifest)} samples, skipped {skipped} files")


if __name__ == "__main__":
    main()
