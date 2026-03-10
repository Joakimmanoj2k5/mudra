"""Test combined approach: shape features + coordinate features ensemble."""
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

# Load shape data
shape_data = np.load("models/registry/shape_centroids.npz")
shape_centroids = shape_data["centroids"]
shape_mean = shape_data["mean"]
shape_std = shape_data["std"]
shape_ex = np.load("models/registry/shape_exemplars.npz")
shape_ex_feats = shape_ex["features"]
shape_ex_labels = shape_ex["labels"]

# Load active-hand data
active_data = np.load("models/registry/active_hand_centroids.npz")
active_centroids = active_data["centroids"]
active_mean = active_data["mean"]
active_std = active_data["std"]
active_ex = np.load("models/registry/active_hand_exemplars.npz")
active_ex_feats = active_ex["features"]
active_ex_labels = active_ex["labels"]

# Also load 136-dim data (with hand-swap at pred time)
norm = json.load(open("models/registry/norm_stats.json"))
full_mean = np.array(norm["mean"], dtype=np.float32)
full_std = np.array(norm["std"], dtype=np.float32)
full_centroids = np.load("models/registry/static_centroids.npz")["centroids"]
full_ex = np.load("models/registry/static_exemplars.npz")
full_ex_feats_list, full_ex_labels_list = [], []
for key in full_ex.files:
    cls_idx = int(key.split("_")[1])
    for feat in full_ex[key]:
        n = np.linalg.norm(feat)
        if n > 1e-8:
            full_ex_feats_list.append(feat / n)
            full_ex_labels_list.append(cls_idx)
full_ex_feats = np.stack(full_ex_feats_list)
full_ex_labels = np.array(full_ex_labels_list, dtype=np.int32)


def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / ex.sum()


def get_active_hand_coords(feat_136):
    left_raw = feat_136[:63].reshape(21, 3)
    right_raw = feat_136[63:126].reshape(21, 3)
    return left_raw if np.abs(left_raw).sum() > np.abs(right_raw).sum() else right_raw


def hand_shape_features(coords_21x3):
    c = coords_21x3
    feats = []
    wrist = c[0]
    fingers = {
        'thumb': (4, 3, 2, 1),
        'index': (8, 7, 6, 5),
        'middle': (12, 11, 10, 9),
        'ring': (16, 15, 14, 13),
        'pinky': (20, 19, 18, 17),
    }
    for name, (tip, dip, pip, mcp) in fingers.items():
        tip_dist = np.linalg.norm(c[tip] - wrist)
        pip_dist = np.linalg.norm(c[pip] - wrist)
        feats.append(tip_dist / pip_dist if pip_dist > 1e-6 else 1.0)
    for name, (tip, dip, pip, mcp) in fingers.items():
        v1 = c[mcp] - c[pip]
        v2 = c[tip] - c[pip]
        d = np.linalg.norm(v1) * np.linalg.norm(v2)
        if d > 1e-6:
            feats.append(float(np.arccos(np.clip(np.dot(v1, v2) / d, -1.0, 1.0))))
        else:
            feats.append(0.0)
    palm_center = c[9]
    for name, (tip, dip, pip, mcp) in fingers.items():
        diff = c[tip] - palm_center
        feats.append(diff[0])
        feats.append(diff[1])
    tip_ids = [4, 8, 12, 16, 20]
    for i in range(4):
        v1 = c[tip_ids[i]] - palm_center
        v2 = c[tip_ids[i+1]] - palm_center
        d = np.linalg.norm(v1) * np.linalg.norm(v2)
        if d > 1e-6:
            feats.append(float(np.arccos(np.clip(np.dot(v1, v2) / d, -1.0, 1.0))))
        else:
            feats.append(0.0)
    thumb_tip = c[4]
    for tip_id in [8, 12, 16, 20]:
        feats.append(np.linalg.norm(thumb_tip - c[tip_id]))
    for name, (tip, dip, pip, mcp) in fingers.items():
        feats.append(c[tip][1])
    for tip_id in [8, 12, 16]:
        feats.append(np.linalg.norm(c[4] - c[tip_id]))
    return np.array(feats, dtype=np.float32)


def extract_active_hand(feat_136):
    left_raw = feat_136[:63]
    right_raw = feat_136[63:126]
    if np.abs(left_raw).sum() > np.abs(right_raw).sum():
        return np.concatenate([left_raw, feat_136[126:131]])
    else:
        return np.concatenate([right_raw, feat_136[131:136]])


def swap_hand_slots(feat):
    s = feat.copy()
    s[:63] = feat[63:126]
    s[63:126] = feat[:63]
    s[126:131] = feat[131:136]
    s[131:136] = feat[126:131]
    return s


def knn_centroid_probs(feat, centroids, ex_feats, ex_labels):
    fnorm = np.linalg.norm(feat)
    if fnorm < 1e-8:
        return np.ones(n_cls, dtype=np.float32) / n_cls
    feat_unit = feat / fnorm
    csims = centroids @ feat_unit
    cprobs = softmax(csims / 0.07)
    asims = ex_feats @ feat_unit
    k = min(15, len(asims))
    top_k = np.argpartition(-asims, k)[:k]
    knn = np.zeros(n_cls, dtype=np.float32)
    for idx in top_k:
        s = max(asims[idx], 0.0)
        knn[ex_labels[idx]] += s * s
    t = knn.sum()
    if t > 1e-8:
        knn /= t
    else:
        knn = cprobs
    return 0.7 * knn + 0.3 * cprobs


def predict_ensemble(feat_136):
    """Combine 3 matching strategies:
    1. Shape features (most robust to distribution shift)
    2. Active-hand coordinates (medium)
    3. Full 136-dim with hand-swap (catches two-hand gestures)
    """
    # Strategy 1: Shape features
    active_coords = get_active_hand_coords(feat_136)
    sf = hand_shape_features(active_coords)
    sf_norm = (sf - shape_mean) / shape_std
    shape_probs = knn_centroid_probs(sf_norm, shape_centroids, shape_ex_feats, shape_ex_labels)

    # Strategy 2: Active-hand coordinates
    ah = extract_active_hand(feat_136)
    ah_norm = (ah - active_mean) / active_std
    active_probs = knn_centroid_probs(ah_norm, active_centroids, active_ex_feats, active_ex_labels)

    # Strategy 3: Full 136-dim with hand swap
    full_norm = (feat_136 - full_mean) / full_std
    full_swap = swap_hand_slots(full_norm)
    probs_orig = knn_centroid_probs(full_norm, full_centroids, full_ex_feats, full_ex_labels)
    probs_swap = knn_centroid_probs(full_swap, full_centroids, full_ex_feats, full_ex_labels)
    full_probs = np.maximum(probs_orig, probs_swap)
    full_probs /= full_probs.sum()

    # Ensemble: shape-dominant blend
    ensemble = 0.50 * shape_probs + 0.30 * active_probs + 0.20 * full_probs
    pred = int(np.argmax(ensemble))
    return pred, float(ensemble[pred])


# Test
correct = 0
letter_correct = 0
letter_total = 0
for i in range(len(X_test)):
    pred, conf = predict_ensemble(X_test[i])
    if pred == y_test[i]:
        correct += 1
    lbl = idx_to_label.get(y_test[i], "")
    if len(lbl) == 1:
        letter_total += 1
        if pred == y_test[i]:
            letter_correct += 1

print(f"ENSEMBLE test accuracy: {correct}/{len(X_test)} = {100*correct/len(X_test):.1f}%")
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
        pred, _ = predict_ensemble(X_test[i])
        if pred == idx:
            correct_l += 1
        else:
            wrong_preds.append(idx_to_label.get(pred, "?"))
    wrong_str = ", ".join(wrong_preds) if wrong_preds else ""
    print(f"  {letter}: {correct_l}/{mask.sum()}  {f'(miss: {wrong_str})' if wrong_str else ''}")
