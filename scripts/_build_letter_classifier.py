"""Build letter-specific classifier: only 26 classes instead of 71.
This dramatically reduces confusion and improves letter detection."""
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

# Identify letter classes
letter_indices = {}
for name, idx in label_map.items():
    if len(name) == 1 and name.isalpha():
        letter_indices[name] = idx

print(f"Letter classes: {len(letter_indices)}")
letter_idx_set = set(letter_indices.values())


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


# Build shape features for LETTER training samples only
letter_train_mask = np.array([y in letter_idx_set for y in y_train])
X_letters_train = X_train[letter_train_mask]
y_letters_train = y_train[letter_train_mask]

print(f"Letter training samples: {len(X_letters_train)}")

X_shape_letters = np.array([
    hand_shape_features(get_active_hand_coords(x)) for x in X_letters_train
])

# Also add active-hand raw coords for combined features
def extract_active_hand(feat_136):
    left_raw = feat_136[:63]
    right_raw = feat_136[63:126]
    if np.abs(left_raw).sum() > np.abs(right_raw).sum():
        return np.concatenate([left_raw, feat_136[126:131]])
    else:
        return np.concatenate([right_raw, feat_136[131:136]])


X_active_letters = np.array([extract_active_hand(x) for x in X_letters_train])

# Combined features: shape (36) + active hand coords (68) = 104 dims
X_combined = np.concatenate([X_shape_letters, X_active_letters], axis=1)
print(f"Combined feature dims: {X_combined.shape[1]}")

# Normalize
comb_mean = X_combined.mean(axis=0).astype(np.float32)
comb_std = X_combined.std(axis=0).astype(np.float32)
comb_std[comb_std < 1e-6] = 1.0
X_combined_norm = (X_combined - comb_mean) / comb_std

# Build centroids ONLY for letter classes
# Create a compact label space: original_idx -> 0..25
letter_idx_list = sorted(letter_indices.values())
original_to_compact = {orig: compact for compact, orig in enumerate(letter_idx_list)}
compact_to_original = {compact: orig for compact, orig in enumerate(letter_idx_list)}

n_letters = len(letter_idx_list)
y_compact = np.array([original_to_compact[y] for y in y_letters_train])

centroids = np.zeros((n_letters, X_combined_norm.shape[1]), dtype=np.float32)
for c in range(n_letters):
    mask = y_compact == c
    if mask.sum() > 0:
        centroid = X_combined_norm[mask].mean(axis=0)
        norm_val = np.linalg.norm(centroid)
        if norm_val > 1e-8:
            centroids[c] = centroid / norm_val

# Build exemplars
feats_list, labels_list = [], []
for c in range(n_letters):
    mask = y_compact == c
    if mask.sum() == 0:
        continue
    for feat in X_combined_norm[mask]:
        n = np.linalg.norm(feat)
        if n > 1e-8:
            feats_list.append(feat / n)
            labels_list.append(c)
exemplar_matrix = np.stack(feats_list)
exemplar_labels = np.array(labels_list, dtype=np.int32)

# Save everything
np.savez_compressed(
    "models/registry/letter_classifier.npz",
    centroids=centroids,
    mean=comb_mean,
    std=comb_std,
    exemplar_features=exemplar_matrix,
    exemplar_labels=exemplar_labels,
    letter_idx_list=np.array(letter_idx_list, dtype=np.int32),
)

print(f"Saved letter_classifier.npz:")
print(f"  centroids: {centroids.shape}")
print(f"  exemplars: {exemplar_matrix.shape}")
print(f"  letters mapped: {[idx_to_label[i] for i in letter_idx_list]}")


def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / ex.sum()


def predict_letter(feat_136):
    """Predict using letter-specific classifier."""
    active_coords = get_active_hand_coords(feat_136)
    sf = hand_shape_features(active_coords)
    ah = extract_active_hand(feat_136)
    combined = np.concatenate([sf, ah])
    combined_norm = (combined - comb_mean) / comb_std
    fnorm = np.linalg.norm(combined_norm)
    if fnorm < 1e-8:
        return 0, 0.0
    feat_unit = combined_norm / fnorm

    csims = centroids @ feat_unit
    cprobs = softmax(csims / 0.07)

    asims = exemplar_matrix @ feat_unit
    k = min(11, len(asims))
    top_k = np.argpartition(-asims, k)[:k]
    knn = np.zeros(n_letters, dtype=np.float32)
    for idx in top_k:
        s = max(asims[idx], 0.0)
        knn[exemplar_labels[idx]] += s * s
    t = knn.sum()
    if t > 1e-8:
        knn /= t
    else:
        knn = cprobs

    probs = 0.7 * knn + 0.3 * cprobs
    compact_pred = int(np.argmax(probs))
    original_idx = compact_to_original[compact_pred]
    return original_idx, float(probs[compact_pred])


# Test on letter test samples
letter_test_mask = np.array([y in letter_idx_set for y in y_test])
X_letters_test = X_test[letter_test_mask]
y_letters_test = y_test[letter_test_mask]

correct = 0
for i in range(len(X_letters_test)):
    pred, conf = predict_letter(X_letters_test[i])
    if pred == y_letters_test[i]:
        correct += 1

print(f"\nLetter-specific test accuracy: {correct}/{len(X_letters_test)} = {100*correct/len(X_letters_test):.1f}%")

print("\nPer-letter results:")
for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    if letter not in label_map:
        continue
    idx = label_map[letter]
    mask = y_letters_test == idx
    if mask.sum() == 0:
        continue
    correct_l = 0
    wrong_preds = []
    for i in np.where(mask)[0]:
        pred, conf = predict_letter(X_letters_test[i])
        if pred == idx:
            correct_l += 1
        else:
            wrong_preds.append(idx_to_label.get(pred, "?"))
    wrong_str = ", ".join(wrong_preds) if wrong_preds else ""
    print(f"  {letter}: {correct_l}/{mask.sum()}  {f'(miss: {wrong_str})' if wrong_str else ''}")
