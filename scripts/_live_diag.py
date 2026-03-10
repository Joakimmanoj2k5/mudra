"""Quick diagnostic: capture webcam hand and compare to training centroids."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import json

from inference.mediapipe.hand_tracker import HandTracker
from inference.preprocess.normalize import build_feature_vector, FeatureNormalizer

tracker = HandTracker()

# Load training data and norm stats
label_map = json.load(open("models/registry/label_map.json"))
idx_to_label = {v: k for k, v in label_map.items()}
norm = json.load(open("models/registry/norm_stats.json"))
normalizer = FeatureNormalizer(
    mean=np.array(norm["mean"], dtype=np.float32),
    std=np.array(norm["std"], dtype=np.float32),
)

centroids = np.load("models/registry/static_centroids.npz")["centroids"]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    sys.exit(1)

# Warm up
for _ in range(10):
    cap.grab()

print("Show a gesture (B, C, D, etc.) to the camera. Capturing for 10 seconds...")
print("=" * 70)

end = time.time() + 10
detected = 0
while time.time() < end:
    ok, frame = cap.read()
    if not ok:
        continue
    extraction = tracker.extract(frame)
    if extraction["status"] != "ok":
        continue
    detected += 1
    if detected > 20:
        break

    # Print hand info
    for h in extraction["hands"]:
        print(f"  Hand: {h['label']:6s}  score={h['score']:.2f}  "
              f"energy(coords)={np.abs(h['coords']).mean():.4f}")

    feature = build_feature_vector(extraction)
    left_e = np.abs(feature[:63]).mean()
    right_e = np.abs(feature[63:126]).mean()
    print(f"  Raw feature: left_slot_energy={left_e:.4f}  right_slot_energy={right_e:.4f}")

    feat_norm = normalizer.transform(feature)
    norm_val = np.linalg.norm(feat_norm)
    if norm_val > 1e-8:
        feat_unit = feat_norm / norm_val
    else:
        feat_unit = feat_norm

    sims = centroids @ feat_unit
    top5 = np.argsort(-sims)[:5]
    print(f"  Top-5 by centroid similarity:")
    for rank, idx in enumerate(top5):
        print(f"    {rank+1}. {idx_to_label[idx]:10s}  sim={sims[idx]:.4f}")

    # Also try hand-swapped version
    swapped = feature.copy()
    swapped[:63] = feature[63:126]
    swapped[63:126] = feature[:63]
    swapped[126:131] = feature[131:136]
    swapped[131:136] = feature[126:131]

    swap_norm = normalizer.transform(swapped)
    sn = np.linalg.norm(swap_norm)
    if sn > 1e-8:
        swap_unit = swap_norm / sn
    else:
        swap_unit = swap_norm

    swap_sims = centroids @ swap_unit
    swap_top5 = np.argsort(-swap_sims)[:5]
    print(f"  Top-5 SWAPPED (hand slots flipped):")
    for rank, idx in enumerate(swap_top5):
        print(f"    {rank+1}. {idx_to_label[idx]:10s}  sim={swap_sims[idx]:.4f}")
    print("-" * 70)
    time.sleep(0.3)

cap.release()
if detected == 0:
    print("No hand detected!")
else:
    print(f"\nCaptured {detected} frames")
