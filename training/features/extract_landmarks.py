"""Extract MediaPipe landmarks from image/video datasets into .npy features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from inference.mediapipe.hand_tracker import HandTracker
from inference.preprocess.normalize import build_feature_vector


def iter_media_files(root: Path):
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.mp4", "*.mov"):
        yield from root.rglob(ext)


def process_file(path: Path, tracker: HandTracker):
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
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ex = tracker.extract(frame)
        if ex["status"] == "ok":
            rows.append(build_feature_vector(ex))
    cap.release()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to data/raw")
    parser.add_argument("--output", default="data/interim/landmarks")
    args = parser.parse_args()

    input_root = Path(args.input)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    tracker = HandTracker()
    manifest = []

    for media_path in iter_media_files(input_root):
        cls_name = media_path.parent.name
        out_cls = out_root / cls_name
        out_cls.mkdir(parents=True, exist_ok=True)

        feats = process_file(media_path, tracker)
        if not feats:
            continue
        arr = np.stack(feats, axis=0)
        out_path = out_cls / f"{media_path.stem}.npy"
        np.save(out_path, arr)
        manifest.append({"class": cls_name, "file": str(out_path), "frames": len(arr)})

    Path("data/interim/landmarks_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Extracted {len(manifest)} samples")


if __name__ == "__main__":
    main()
