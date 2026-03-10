"""Build geometric hand-shape features and test accuracy.
Instead of raw coordinates, extract meaningful features like finger extension,
curl, spread - which are what actually distinguish ISL alphabet signs."""
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


def get_active_hand_coords(feat_136):
    """Get the 21x3 coords of the dominant hand."""
    left_raw = feat_136[:63].reshape(21, 3)
    right_raw = feat_136[63:126].reshape(21, 3)
    left_energy = np.abs(left_raw).sum()
    right_energy = np.abs(right_raw).sum()
    return left_raw if left_energy > right_energy else right_raw


def hand_shape_features(coords_21x3):
    """Extract geometric hand shape features from 21 landmarks.
    Returns a feature vector capturing finger states, distances, and angles."""
    c = coords_21x3
    feats = []

    # Wrist is landmark 0 (should be near origin after normalization)
    wrist = c[0]

    # 1. Finger extension ratios (5 features)
    # Compare fingertip-to-wrist dist with PIP-to-wrist dist
    fingers = {
        'thumb': (4, 3, 2, 1),    # tip, dip, pip, mcp
        'index': (8, 7, 6, 5),
        'middle': (12, 11, 10, 9),
        'ring': (16, 15, 14, 13),
        'pinky': (20, 19, 18, 17),
    }

    for name, (tip, dip, pip, mcp) in fingers.items():
        tip_dist = np.linalg.norm(c[tip] - wrist)
        pip_dist = np.linalg.norm(c[pip] - wrist)
        if pip_dist > 1e-6:
            feats.append(tip_dist / pip_dist)  # >1 = extended, <1 = folded
        else:
            feats.append(1.0)

    # 2. Finger curl angles at PIP joint (5 features)
    for name, (tip, dip, pip, mcp) in fingers.items():
        v1 = c[mcp] - c[pip]
        v2 = c[tip] - c[pip]
        d = np.linalg.norm(v1) * np.linalg.norm(v2)
        if d > 1e-6:
            cos = np.clip(np.dot(v1, v2) / d, -1.0, 1.0)
            feats.append(float(np.arccos(cos)))
        else:
            feats.append(0.0)

    # 3. Fingertip distances from palm center - landmark 9 (10 features: x,y for 5 fingers)
    palm_center = c[9]  # middle finger MCP
    for name, (tip, dip, pip, mcp) in fingers.items():
        diff = c[tip] - palm_center
        feats.append(diff[0])  # x relative
        feats.append(diff[1])  # y relative

    # 4. Inter-finger spread (4 features: angle between adjacent fingertips)
    tip_ids = [4, 8, 12, 16, 20]
    for i in range(4):
        v1 = c[tip_ids[i]] - palm_center
        v2 = c[tip_ids[i+1]] - palm_center
        d = np.linalg.norm(v1) * np.linalg.norm(v2)
        if d > 1e-6:
            cos = np.clip(np.dot(v1, v2) / d, -1.0, 1.0)
            feats.append(float(np.arccos(cos)))
        else:
            feats.append(0.0)

    # 5. Thumb-to-finger tip distances (4 features)
    thumb_tip = c[4]
    for tip_id in [8, 12, 16, 20]:
        feats.append(np.linalg.norm(thumb_tip - c[tip_id]))

    # 6. Finger height ordering (5 features: y-coordinate of each fingertip)
    for name, (tip, dip, pip, mcp) in fingers.items():
        feats.append(c[tip][1])

    # 7. Cross-finger contact features (3 features)
    # index-to-thumb, middle-to-thumb, ring-to-thumb
    for tip_id in [8, 12, 16]:
        feats.append(np.linalg.norm(c[4] - c[tip_id]))

    return np.array(feats, dtype=np.float32)


# Build features for all training samples
print("Building hand-shape features...")
X_shape_train = np.array([hand_shape_features(get_active_hand_coords(x)) for x in X_train])
X_shape_test = np.array([hand_shape_features(get_active_hand_coords(x)) for x in X_test])
print(f"Shape features: {X_shape_train.shape[1]} dims")

# Normalize
shape_mean = X_shape_train.mean(axis=0)
shape_std = X_shape_train.std(axis=0)
shape_std[shape_std < 1e-6] = 1.0

X_shape_train_norm = (X_shape_train - shape_mean) / shape_std
X_shape_test_norm = (X_shape_test - shape_mean) / shape_std

# Build centroids
centroids = np.zeros((n_cls, X_shape_train_norm.shape[1]), dtype=np.float32)
for c in range(n_cls):
    mask = y_train == c
    if mask.sum() > 0:
        centroid = X_shape_train_norm[mask].mean(axis=0)
        norm_val = np.linalg.norm(centroid)
        if norm_val > 1e-8:
            centroids[c] = centroid / norm_val

# Build exemplars
feats_list, labels_list = [], []
for c in range(n_cls):
    mask = y_train == c
    if mask.sum() == 0:
        continue
    for feat in X_shape_train_norm[mask]:
        n = np.linalg.norm(feat)
        if n > 1e-8:
            feats_list.append(feat / n)
            labels_list.append(c)
exemplar_matrix = np.stack(feats_list)
exemplar_labels = np.array(labels_list, dtype=np.int32)

# Save
np.savez_compressed(
    "models/registry/shape_centroids.npz",
    centroids=centroids,
    mean=shape_mean,
    std=shape_std,
)
np.savez_compressed(
    "models/registry/shape_exemplars.npz",
    features=exemplar_matrix,
    labels=exemplar_labels,
)
print(f"Saved shape_centroids.npz: centroids {centroids.shape}")
print(f"Saved shape_exemplars.npz: {exemplar_matrix.shape}")


def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / ex.sum()


def predict_shape(feat_136):
    active_coords = get_active_hand_coords(feat_136)
    shape_feat = hand_shape_features(active_coords)
    shape_norm = (shape_feat - shape_mean) / shape_std
    fnorm = np.linalg.norm(shape_norm)
    if fnorm < 1e-8:
        return 0, 0.0
    feat_unit = shape_norm / fnorm

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


# Test
correct = 0
letter_correct = 0
letter_total = 0
for i in range(len(X_test)):
    pred, conf = predict_shape(X_test[i])
    if pred == y_test[i]:
        correct += 1
    lbl = idx_to_label.get(y_test[i], "")
    if len(lbl) == 1:
        letter_total += 1
        if pred == y_test[i]:
            letter_correct += 1

print(f"\nTest accuracy: {correct}/{len(X_test)} = {100*correct/len(X_test):.1f}%")
print(f"Letters only: {letter_correct}/{letter_total} = {100*letter_correct/max(letter_total,1):.1f}%")

print("\nPer-letter results:")
for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    if letter not in label_map:
        continue
    idx = label_map[letter]
    mask = y_test == idx
    if mask.sum() == 0:
        continue
    correct_l = 0
    wrong_preds = []
    for i in np.where(mask)[0]:
        pred, _ = predict_shape(X_test[i])
        if pred == idx:
            correct_l += 1
        else:
            wrong_preds.append(idx_to_label.get(pred, "?"))
    wrong_str = ", ".join(wrong_preds) if wrong_preds else ""
    print(f"  {letter}: {correct_l}/{mask.sum()}  {f'(misclassified as: {wrong_str})' if wrong_str else ''}")
