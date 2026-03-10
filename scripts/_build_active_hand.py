"""Build a hand-agnostic centroid database using only the ACTIVE hand's features.
This eliminates the hand-slot assignment problem and normalization artifacts."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import json

data = np.load("data/processed/static_split_v1.npz")
X_train = data["X_train"]
y_train = data["y_train"]
X_test = data["X_test"]
y_test = data["y_test"]

label_map = json.load(open("models/registry/label_map.json"))
idx_to_label = {v: k for k, v in label_map.items()}
n_cls = max(label_map.values()) + 1


def extract_active_hand(feat_136):
    """Extract the dominant hand's features from a 136-dim vector.
    Returns 68-dim: 63 coords of the active hand + 5 joint angles."""
    left_raw = feat_136[:63]
    right_raw = feat_136[63:126]
    left_energy = np.abs(left_raw).sum()
    right_energy = np.abs(right_raw).sum()

    if left_energy > right_energy:
        coords = left_raw
        angles = feat_136[126:131]  # left hand angles
    else:
        coords = right_raw
        angles = feat_136[131:136]  # right hand angles

    return np.concatenate([coords, angles])


# Build active-hand features for training data
X_active = np.array([extract_active_hand(x) for x in X_train])
print(f"Active-hand features: {X_active.shape}")

# Compute normalization stats for the active-hand features
active_mean = X_active.mean(axis=0).astype(np.float32)
active_std = X_active.std(axis=0).astype(np.float32)
active_std[active_std < 1e-6] = 1.0

X_active_norm = (X_active - active_mean) / active_std

# Build centroids
centroids = np.zeros((n_cls, 68), dtype=np.float32)
for c in range(n_cls):
    mask = y_train == c
    if mask.sum() > 0:
        centroid = X_active_norm[mask].mean(axis=0)
        norm_val = np.linalg.norm(centroid)
        if norm_val > 1e-8:
            centroids[c] = centroid / norm_val

# Build exemplars
exemplar_dict = {}
for c in range(n_cls):
    mask = y_train == c
    if mask.sum() > 0:
        exemplar_dict[f"class_{c}"] = X_active_norm[mask]

# Save everything
np.savez_compressed(
    "models/registry/active_hand_centroids.npz",
    centroids=centroids,
    mean=active_mean,
    std=active_std,
)

feats_list = []
labels_list = []
for key in exemplar_dict:
    cls_idx = int(key.split("_")[1])
    for feat in exemplar_dict[key]:
        n = np.linalg.norm(feat)
        if n > 1e-8:
            feats_list.append(feat / n)
            labels_list.append(cls_idx)
exemplar_matrix = np.stack(feats_list)
exemplar_labels = np.array(labels_list, dtype=np.int32)
np.savez_compressed(
    "models/registry/active_hand_exemplars.npz",
    features=exemplar_matrix,
    labels=exemplar_labels,
)

print(f"Saved active_hand_centroids.npz: {centroids.shape}")
print(f"Saved active_hand_exemplars.npz: {exemplar_matrix.shape}")


def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / ex.sum()


def predict_active_hand(feat_136):
    """Predict using active-hand features."""
    active = extract_active_hand(feat_136)
    active_norm = (active - active_mean) / active_std
    fnorm = np.linalg.norm(active_norm)
    if fnorm < 1e-8:
        return 0, 0.0
    feat_unit = active_norm / fnorm

    csims = centroids @ feat_unit
    cprobs = softmax(csims / 0.07)

    asims = exemplar_matrix @ feat_unit
    k = min(15, len(asims))
    top_k = np.argpartition(-asims, k)[:k]
    knn = np.zeros(n_cls, dtype=np.float32)
    for idx in top_k:
        s = max(asims[idx], 0.0)
        knn[exemplar_labels[idx]] += s * s
    t = knn.sum()
    if t > 1e-8:
        knn /= t
    else:
        knn = cprobs

    probs = 0.7 * knn + 0.3 * cprobs
    pred = int(np.argmax(probs))
    return pred, float(probs[pred])


# Test on original test data
correct = 0
letter_correct = 0
letter_total = 0
for i in range(len(X_test)):
    pred, conf = predict_active_hand(X_test[i])
    if pred == y_test[i]:
        correct += 1
    lbl = idx_to_label.get(y_test[i], "")
    if len(lbl) == 1:
        letter_total += 1
        if pred == y_test[i]:
            letter_correct += 1

print(f"\nTest accuracy: {correct}/{len(X_test)} = {100*correct/len(X_test):.1f}%")
print(f"Letters only: {letter_correct}/{letter_total} = {100*letter_correct/max(letter_total,1):.1f}%")

# Per-letter detail
print("\nPer-letter results:")
for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    if letter not in label_map:
        continue
    idx = label_map[letter]
    mask = y_test == idx
    if mask.sum() == 0:
        continue
    correct_l = 0
    for i in np.where(mask)[0]:
        pred, _ = predict_active_hand(X_test[i])
        if pred == idx:
            correct_l += 1
    print(f"  {letter}: {correct_l}/{mask.sum()}")
