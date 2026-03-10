"""Train and save just the Random Forest letter classifier."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import json
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

data = np.load("data/processed/static_split_v1.npz")
X_train = data["X_train"]
y_train = data["y_train"]

label_map = json.load(open("models/registry/label_map.json"))
idx_to_label = {v: k for k, v in label_map.items()}

letter_indices = {name: idx for name, idx in label_map.items() if len(name) == 1 and name.isalpha()}
letter_idx_set = set(letter_indices.values())
letter_idx_list = sorted(letter_indices.values())
original_to_compact = {orig: compact for compact, orig in enumerate(letter_idx_list)}
n_letters = len(letter_idx_list)


def get_active_hand_coords(feat_136):
    left = feat_136[:63].reshape(21, 3)
    right = feat_136[63:126].reshape(21, 3)
    return left if np.abs(left).sum() > np.abs(right).sum() else right


def hand_shape_features(c):
    feats = []
    wrist = c[0]
    fids = [(4,3,2,1),(8,7,6,5),(12,11,10,9),(16,15,14,13),(20,19,18,17)]
    for tip, dip, pip, mcp in fids:
        td = np.linalg.norm(c[tip] - wrist)
        pd = np.linalg.norm(c[pip] - wrist)
        feats.append(td / pd if pd > 1e-6 else 1.0)
    for tip, dip, pip, mcp in fids:
        v1, v2 = c[mcp] - c[pip], c[tip] - c[pip]
        d = np.linalg.norm(v1) * np.linalg.norm(v2)
        feats.append(float(np.arccos(np.clip(np.dot(v1, v2) / d, -1, 1))) if d > 1e-6 else 0.0)
    palm = c[9]
    for tip, dip, pip, mcp in fids:
        diff = c[tip] - palm
        feats.extend([diff[0], diff[1]])
    tips = [4, 8, 12, 16, 20]
    for i in range(4):
        v1, v2 = c[tips[i]] - palm, c[tips[i+1]] - palm
        d = np.linalg.norm(v1) * np.linalg.norm(v2)
        feats.append(float(np.arccos(np.clip(np.dot(v1, v2) / d, -1, 1))) if d > 1e-6 else 0.0)
    for tid in [8, 12, 16, 20]:
        feats.append(np.linalg.norm(c[4] - c[tid]))
    for tip, dip, pip, mcp in fids:
        feats.append(c[tip][1])
    for tid in [8, 12, 16]:
        feats.append(np.linalg.norm(c[4] - c[tid]))
    return np.array(feats, dtype=np.float32)


def extract_active_hand(feat_136):
    left, right = feat_136[:63], feat_136[63:126]
    if np.abs(left).sum() > np.abs(right).sum():
        return np.concatenate([left, feat_136[126:131]])
    return np.concatenate([right, feat_136[131:136]])


def combined_features(feat_136):
    return np.concatenate([
        hand_shape_features(get_active_hand_coords(feat_136)),
        extract_active_hand(feat_136),
    ])


# Build letter training set
mask = np.array([y in letter_idx_set for y in y_train])
X_lt = X_train[mask]
y_lt = np.array([original_to_compact[y] for y in y_train[mask]])

X_feat = np.array([combined_features(x) for x in X_lt])
print(f"Letter features: {X_feat.shape}")

# Heavy augmentation
rng = np.random.RandomState(42)
X_aug = [X_feat.copy()]
y_aug = [y_lt.copy()]

for _ in range(50):
    noise = rng.uniform(0.01, 0.15)
    X_aug.append(X_feat + rng.randn(*X_feat.shape).astype(np.float32) * noise)
    y_aug.append(y_lt.copy())

for _ in range(20):
    X_mix = np.zeros_like(X_feat)
    for c in range(n_letters):
        m = y_lt == c
        cf = X_feat[m]
        if len(cf) < 2:
            X_mix[m] = cf
            continue
        idxs = rng.randint(0, len(cf), size=len(cf))
        lam = rng.beta(0.4, 0.4, size=(len(cf), 1)).astype(np.float32)
        X_mix[m] = lam * cf + (1 - lam) * cf[idxs]
    X_aug.append(X_mix)
    y_aug.append(y_lt.copy())

X_aug = np.vstack(X_aug)
y_aug = np.concatenate(y_aug)
print(f"Augmented: {X_aug.shape[0]} samples")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_aug)

rf = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=2, random_state=42, n_jobs=-1)
rf.fit(X_scaled, y_aug)

# Test
X_test = data["X_test"]
y_test = data["y_test"]
test_mask = np.array([y in letter_idx_set for y in y_test])
X_test_feat = np.array([combined_features(x) for x in X_test[test_mask]])
y_test_compact = np.array([original_to_compact[y] for y in y_test[test_mask]])
X_test_scaled = scaler.transform(X_test_feat)

acc = rf.score(X_test_scaled, y_test_compact)
print(f"\nRF letter accuracy: {acc*100:.1f}%")

pred = rf.predict(X_test_scaled)
for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    idx = original_to_compact.get(label_map.get(letter, -1), -1)
    if idx < 0:
        continue
    m = y_test_compact == idx
    correct = (pred[m] == idx).sum()
    print(f"  {letter}: {correct}/{m.sum()}")

# Save
with open("models/registry/letter_rf_model.pkl", "wb") as f:
    pickle.dump({"model": rf, "scaler": scaler, "letter_idx_list": letter_idx_list}, f)
print("\nSaved letter_rf_model.pkl")
