"""Test hand-agnostic matching: simulate webcam with wrong hand slot."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import json

data = np.load("data/processed/static_split_v1.npz")
X_test = data["X_test"]
y_test = data["y_test"]

label_map = json.load(open("models/registry/label_map.json"))
idx_to_label = {v: k for k, v in label_map.items()}
n_cls = max(label_map.values()) + 1

norm = json.load(open("models/registry/norm_stats.json"))
mean = np.array(norm["mean"], dtype=np.float32)
std = np.array(norm["std"], dtype=np.float32)

centroids = np.load("models/registry/static_centroids.npz")["centroids"]
edata = np.load("models/registry/static_exemplars.npz")
feats_list, labels_list = [], []
for key in edata.files:
    cls_idx = int(key.split("_")[1])
    for feat in edata[key]:
        n = np.linalg.norm(feat)
        if n > 1e-8:
            feats_list.append(feat / n)
            labels_list.append(cls_idx)
exemplar_matrix = np.stack(feats_list)
exemplar_labels = np.array(labels_list, dtype=np.int32)


def swap_hand_slots(feat):
    s = feat.copy()
    s[:63] = feat[63:126]
    s[63:126] = feat[:63]
    s[126:131] = feat[131:136]
    s[131:136] = feat[126:131]
    return s


def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / ex.sum()


def predict_with_swap(feature):
    """Predict using both original and swapped feature (hand-agnostic)."""
    best_probs = np.zeros(n_cls, dtype=np.float32)
    swapped = swap_hand_slots(feature)

    for feat in [feature, swapped]:
        fnorm = np.linalg.norm(feat)
        if fnorm < 1e-8:
            continue
        feat_unit = feat / fnorm

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
        best_probs = np.maximum(best_probs, probs)

    best_probs /= best_probs.sum()
    return int(np.argmax(best_probs)), float(best_probs.max())


# Test 1: Normal test set (features in their original hand slots)
correct = 0
letter_correct = 0
letter_total = 0
for i in range(len(X_test)):
    feat = (X_test[i] - mean) / std
    pred, conf = predict_with_swap(feat)
    if pred == y_test[i]:
        correct += 1
    lbl = idx_to_label.get(y_test[i], "")
    if len(lbl) == 1:
        letter_total += 1
        if pred == y_test[i]:
            letter_correct += 1

print(f"NORMAL test set: {correct}/{len(X_test)} = {100*correct/len(X_test):.1f}%")
print(f"  Letters only: {letter_correct}/{letter_total} = {100*letter_correct/max(letter_total,1):.1f}%")

# Test 2: SWAPPED test set (simulate webcam with wrong hand slot)
correct2 = 0
letter_correct2 = 0
for i in range(len(X_test)):
    feat_orig = (X_test[i] - mean) / std
    feat_swapped = swap_hand_slots(feat_orig)  # Simulate wrong hand slot
    pred, conf = predict_with_swap(feat_swapped)
    if pred == y_test[i]:
        correct2 += 1
    lbl = idx_to_label.get(y_test[i], "")
    if len(lbl) == 1:
        if pred == y_test[i]:
            letter_correct2 += 1

print(f"\nSWAPPED test set (wrong hand slot): {correct2}/{len(X_test)} = {100*correct2/len(X_test):.1f}%")
print(f"  Letters only: {letter_correct2}/{letter_total} = {100*letter_correct2/max(letter_total,1):.1f}%")

# Test 3: Per-letter results for swapped input
print("\nPer-letter results with SWAPPED input:")
for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    if letter not in label_map:
        continue
    idx = label_map[letter]
    mask = y_test == idx
    if mask.sum() == 0:
        continue
    correct_l = 0
    for i in np.where(mask)[0]:
        feat = swap_hand_slots((X_test[i] - mean) / std)
        pred, _ = predict_with_swap(feat)
        if pred == idx:
            correct_l += 1
    print(f"  {letter}: {correct_l}/{mask.sum()}")
