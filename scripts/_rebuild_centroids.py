"""Rebuild centroids and exemplars with hand-swap augmentation."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import json

data = np.load("data/processed/static_split_v1.npz")
X_train = data["X_train"]
y_train = data["y_train"]

label_map = json.load(open("models/registry/label_map.json"))
n_cls = max(label_map.values()) + 1

norm = json.load(open("models/registry/norm_stats.json"))
mean = np.array(norm["mean"], dtype=np.float32)
std = np.array(norm["std"], dtype=np.float32)


def swap_hand_slots(feat):
    s = feat.copy()
    s[:63] = feat[63:126]
    s[63:126] = feat[:63]
    s[126:131] = feat[131:136]
    s[131:136] = feat[126:131]
    return s


X_normed = (X_train - mean) / std

X_aug = np.vstack([X_normed, np.array([swap_hand_slots(x) for x in X_normed])])
y_aug = np.concatenate([y_train, y_train])

print(f"Original: {len(X_normed)} samples")
print(f"Augmented: {len(X_aug)} samples (with hand-swap)")

centroids = np.zeros((n_cls, X_aug.shape[1]), dtype=np.float32)
for c in range(n_cls):
    mask = y_aug == c
    if mask.sum() > 0:
        centroid = X_aug[mask].mean(axis=0)
        norm_val = np.linalg.norm(centroid)
        if norm_val > 1e-8:
            centroids[c] = centroid / norm_val

np.savez_compressed("models/registry/static_centroids.npz", centroids=centroids)
print(f"Saved centroids: {centroids.shape}")

exemplar_dict = {}
for c in range(n_cls):
    mask = y_aug == c
    if mask.sum() > 0:
        exemplar_dict[f"class_{c}"] = X_aug[mask]

np.savez_compressed("models/registry/static_exemplars.npz", **exemplar_dict)
total = sum(len(v) for v in exemplar_dict.values())
print(f"Saved exemplars: {total} total across {len(exemplar_dict)} classes")

idx_to_label = {v: k for k, v in label_map.items()}
for letter in ["A", "B", "C", "D", "E"]:
    idx = label_map[letter]
    c = centroids[idx]
    left_e = np.abs(c[:63]).mean()
    right_e = np.abs(c[63:126]).mean()
    print(f"{letter}: centroid left_energy={left_e:.4f}, right_energy={right_e:.4f}")
