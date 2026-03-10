"""Try Random Forest + heavy augmentation for letter classification."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pickle

data = np.load("data/processed/static_split_v1.npz")
X_train = data["X_train"]
y_train = data["y_train"]
X_test = data["X_test"]
y_test = data["y_test"]

label_map = json.load(open("models/registry/label_map.json"))
idx_to_label = {v: k for k, v in label_map.items()}

letter_indices = {}
for name, idx in label_map.items():
    if len(name) == 1 and name.isalpha():
        letter_indices[name] = idx
letter_idx_set = set(letter_indices.values())
letter_idx_list = sorted(letter_indices.values())
original_to_compact = {orig: compact for compact, orig in enumerate(letter_idx_list)}
compact_to_original = {compact: orig for compact, orig in enumerate(letter_idx_list)}
n_letters = len(letter_idx_list)


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


def combined_features(feat_136):
    active_coords = get_active_hand_coords(feat_136)
    sf = hand_shape_features(active_coords)
    ah = extract_active_hand(feat_136)
    return np.concatenate([sf, ah])


# Build letter training set
letter_mask_train = np.array([y in letter_idx_set for y in y_train])
X_lt = X_train[letter_mask_train]
y_lt = y_train[letter_mask_train]
y_lt_compact = np.array([original_to_compact[y] for y in y_lt])

X_feat = np.array([combined_features(x) for x in X_lt])
print(f"Letter training features: {X_feat.shape}")

# Heavy augmentation: 200x per original sample
rng = np.random.RandomState(42)
X_aug_list = [X_feat.copy()]
y_aug_list = [y_lt_compact.copy()]

for aug_round in range(200):
    noise_level = rng.uniform(0.01, 0.15)
    X_noisy = X_feat + rng.randn(*X_feat.shape).astype(np.float32) * noise_level
    X_aug_list.append(X_noisy)
    y_aug_list.append(y_lt_compact.copy())

# Mixup augmentation: blend same-class pairs
for _ in range(50):
    X_mix = np.zeros_like(X_feat)
    for c in range(n_letters):
        mask = y_lt_compact == c
        class_feats = X_feat[mask]
        if len(class_feats) < 2:
            X_mix[mask] = class_feats
            continue
        idxs = rng.randint(0, len(class_feats), size=len(class_feats))
        lam = rng.beta(0.4, 0.4, size=(len(class_feats), 1)).astype(np.float32)
        X_mix[mask] = lam * class_feats + (1 - lam) * class_feats[idxs]
    X_aug_list.append(X_mix)
    y_aug_list.append(y_lt_compact.copy())

X_aug = np.vstack(X_aug_list)
y_aug = np.concatenate(y_aug_list)
print(f"Augmented: {X_aug.shape[0]} samples ({X_aug.shape[0] // len(X_feat)}x)")

# Normalize
scaler = StandardScaler()
X_aug_scaled = scaler.fit_transform(X_aug)

# Train classifiers and compare
letter_mask_test = np.array([y in letter_idx_set for y in y_test])
X_test_feat = np.array([combined_features(x) for x in X_test[letter_mask_test]])
y_test_compact = np.array([original_to_compact[y] for y in y_test[letter_mask_test]])
X_test_scaled = scaler.transform(X_test_feat)

print(f"\nTest samples: {len(X_test_feat)}")

# 1. Random Forest
rf = RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_leaf=1, random_state=42, n_jobs=-1)
rf.fit(X_aug_scaled, y_aug)
rf_acc = rf.score(X_test_scaled, y_test_compact)
print(f"\nRandom Forest accuracy: {rf_acc*100:.1f}%")
rf_pred = rf.predict(X_test_scaled)
for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    idx = original_to_compact.get(label_map.get(letter, -1), -1)
    if idx < 0:
        continue
    mask = y_test_compact == idx
    if mask.sum() == 0:
        continue
    correct_l = (rf_pred[mask] == idx).sum()
    wrong = [idx_to_label[compact_to_original[p]] for p in rf_pred[mask] if p != idx]
    wrong_str = ", ".join(wrong) if wrong else ""
    print(f"  {letter}: {correct_l}/{mask.sum()}  {'(miss: ' + wrong_str + ')' if wrong_str else ''}")

# 2. Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, random_state=42)
gb.fit(X_aug_scaled, y_aug)
gb_acc = gb.score(X_test_scaled, y_test_compact)
print(f"\nGradient Boosting accuracy: {gb_acc*100:.1f}%")

# 3. KNN
knn = KNeighborsClassifier(n_neighbors=7, weights='distance')
knn.fit(X_aug_scaled, y_aug)
knn_acc = knn.score(X_test_scaled, y_test_compact)
print(f"\nKNN accuracy: {knn_acc*100:.1f}%")

# Save the best model
best_name = max([("RF", rf_acc, rf), ("GB", gb_acc, gb), ("KNN", knn_acc, knn)], key=lambda x: x[1])
print(f"\nBest: {best_name[0]} at {best_name[1]*100:.1f}%")

# Save RF (reliable) plus scaler
with open("models/registry/letter_rf_model.pkl", "wb") as f:
    pickle.dump({"model": rf, "scaler": scaler, "letter_idx_list": letter_idx_list}, f)
print("Saved letter_rf_model.pkl")
