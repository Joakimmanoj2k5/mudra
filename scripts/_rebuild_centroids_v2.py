"""Rebuild centroids and exemplars WITHOUT hand-swap augmentation.
The hand-swap will be handled at prediction time instead."""
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

X_normed = (X_train - mean) / std

print(f"Training samples: {len(X_normed)}")

centroids = np.zeros((n_cls, X_normed.shape[1]), dtype=np.float32)
for c in range(n_cls):
    mask = y_train == c
    if mask.sum() > 0:
        centroid = X_normed[mask].mean(axis=0)
        norm_val = np.linalg.norm(centroid)
        if norm_val > 1e-8:
            centroids[c] = centroid / norm_val

np.savez_compressed("models/registry/static_centroids.npz", centroids=centroids)
print(f"Saved centroids: {centroids.shape}")

exemplar_dict = {}
for c in range(n_cls):
    mask = y_train == c
    if mask.sum() > 0:
        exemplar_dict[f"class_{c}"] = X_normed[mask]

np.savez_compressed("models/registry/static_exemplars.npz", **exemplar_dict)
total = sum(len(v) for v in exemplar_dict.values())
print(f"Saved exemplars: {total} total across {len(exemplar_dict)} classes")
