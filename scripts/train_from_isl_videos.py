#!/usr/bin/env python3
"""
Train MUDRA gesture model from real ISL videos in isl_videos/ folder.

Steps:
  1. Map ISL video filenames to existing gesture classes
  2. Extract MediaPipe landmarks from each video
  3. Save landmark .npy files and manifest
  4. Build train/val/test split
  5. Train StaticMLP and DynamicBiGRU models
  6. Save model weights, label_map, norm_stats, class_modes

Usage:
    python scripts/train_from_isl_videos.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.mediapipe.hand_tracker import HandTracker
from inference.preprocess.normalize import build_feature_vector

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ISL_VIDEO_DIR = PROJECT_ROOT / "isl_videos"
LANDMARK_DIR = PROJECT_ROOT / "data" / "interim" / "landmarks_real"
MANIFEST_PATH = PROJECT_ROOT / "data" / "interim" / "landmarks_manifest.json"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
STATIC_SPLIT = PROCESSED_DIR / "static_split_v1.npz"
DYNAMIC_SPLIT = PROCESSED_DIR / "dynamic_split_v1.npz"
REGISTRY = PROJECT_ROOT / "models" / "registry"
STATIC_MODEL_PATH = PROJECT_ROOT / "models" / "static" / "static_mlp_v001.pt"
DYNAMIC_MODEL_PATH = PROJECT_ROOT / "models" / "dynamic" / "dynamic_bigru_v001.pt"


# ---------------------------------------------------------------------------
# Step 0: Build video-to-class mapping
# ---------------------------------------------------------------------------

# Manual mappings for videos whose filename doesn't directly match label
MANUAL_MAP = {
    # number digits → spelled-out names in label_map
    "0": None,          # no "Zero" class exists
    "1": "One",
    "2": "Two",
    "3": "Three",
    "4": "Four",
    "5": "Five",
    "6": "Six",
    "7": "Seven",
    "9": "Nine",
    # synonyms / alternate names
    "ok": "Okay",
    "good_morning": "Good Morning",
    "good_night": "Good Night",
    "good_afternoon": None,   # not in current label set
    "thank_you": "Thank You",
    "nice_to_meet_you": None, # not in current label set
    "excuse_me": None,
    "see_you_tomorrow": None,
    "what_happened": None,
    "what_time": None,
    "how_many": None,
    "every_time": None,
    "right_now": None,
    "be_brave": None,
    "be_brave_enough": None,
    "have_courage": None,
    "time_out": None,
}


def _norm(s: str) -> str:
    """Normalize a name for matching."""
    return s.strip().lower().replace(" ", "_").replace("-", "_")


def build_video_class_map() -> dict[str, str]:
    """Return {video_stem: class_display_name} for matchable videos."""
    # Load existing label map
    label_map_path = REGISTRY / "label_map.json"
    if label_map_path.exists():
        lm = json.loads(label_map_path.read_text(encoding="utf-8"))
    else:
        from utils.common.gesture_catalog import all_gestures
        lm = {g.display_name: i for i, g in enumerate(sorted(
            {g.display_name for g in all_gestures()}))}

    norm_to_label = {}
    for name in lm:
        norm_to_label[_norm(name)] = name

    video_to_class: dict[str, str] = {}
    for vpath in sorted(ISL_VIDEO_DIR.glob("*.mp4")):
        stem = vpath.stem
        # Check manual map first
        if stem in MANUAL_MAP:
            if MANUAL_MAP[stem] is not None:
                video_to_class[stem] = MANUAL_MAP[stem]
            continue

        # Try direct normalized match
        normed = _norm(stem)
        if normed in norm_to_label:
            video_to_class[stem] = norm_to_label[normed]
            continue

        # Try with first letter upper for alphabets (a-z)
        if len(stem) == 1 and stem.isalpha():
            upper = stem.upper()
            if upper in lm:
                video_to_class[stem] = upper
                continue

    return video_to_class


# ---------------------------------------------------------------------------
# Step 1: Extract landmarks from ISL videos
# ---------------------------------------------------------------------------

def extract_landmarks(video_to_class: dict[str, str], sample_every_n: int = 2) -> list[dict]:
    """Extract MediaPipe landmarks from ISL videos.

    Args:
        video_to_class: mapping from video stem to class name
        sample_every_n: extract every Nth frame (1 = all frames)

    Returns:
        List of manifest entries.
    """
    tracker = HandTracker()
    if not tracker.available:
        print("ERROR: MediaPipe HandTracker not available!")
        sys.exit(1)

    # Clean old landmarks
    if LANDMARK_DIR.exists():
        shutil.rmtree(LANDMARK_DIR)
    LANDMARK_DIR.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    total_frames_extracted = 0
    failed_videos = []

    for stem, class_name in sorted(video_to_class.items()):
        vpath = ISL_VIDEO_DIR / f"{stem}.mp4"
        if not vpath.exists():
            print(f"  WARN: {vpath} not found, skipping")
            continue

        cap = cv2.VideoCapture(str(vpath))
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        features: list[np.ndarray] = []
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % sample_every_n != 0:
                continue

            ex = tracker.extract(frame)
            if ex["status"] == "ok":
                fv = build_feature_vector(ex)
                features.append(fv)

        cap.release()

        if not features:
            failed_videos.append(stem)
            continue

        arr = np.stack(features, axis=0)  # (N, 136)
        out_cls = LANDMARK_DIR / class_name
        out_cls.mkdir(parents=True, exist_ok=True)
        out_path = out_cls / f"{stem}.npy"
        np.save(out_path, arr)

        manifest.append({
            "class": class_name,
            "file": str(out_path),
            "frames": int(arr.shape[0]),
        })
        total_frames_extracted += arr.shape[0]

    # Data augmentation: flip, noise, temporal shift
    augmented_entries = _augment_landmarks(manifest)
    manifest.extend(augmented_entries)

    # Save manifest
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    n_classes = len({e["class"] for e in manifest})
    print(f"\n  Landmark extraction complete:")
    print(f"    Classes with data: {n_classes}")
    print(f"    Total manifest entries: {len(manifest)}")
    print(f"    Total frames extracted: {total_frames_extracted}")
    if failed_videos:
        print(f"    Failed (no hand detected): {failed_videos}")

    return manifest


def _augment_landmarks(manifest: list[dict], copies: int = 4) -> list[dict]:
    """Create augmented copies of landmark data with noise + mirroring.

    This increases per-class sample count for better model generalization.
    """
    rng = np.random.default_rng(42)
    augmented: list[dict] = []

    for entry in manifest:
        arr = np.load(entry["file"])  # (N, 136)
        cls_name = entry["class"]
        out_cls = LANDMARK_DIR / cls_name

        for aug_i in range(copies):
            aug = arr.copy()

            # Add small Gaussian noise
            noise = rng.normal(0, 0.02, aug.shape).astype(np.float32)
            aug = aug + noise

            # Random scale perturbation (0.9 - 1.1)
            scale = rng.uniform(0.9, 1.1)
            aug[:, :126] *= scale  # scale only coordinate features, not angles

            # Random small rotation in xy plane
            angle = rng.uniform(-0.15, 0.15)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            for hand_offset in [0, 63]:
                for lm_i in range(21):
                    base = hand_offset + lm_i * 3
                    x, y = aug[:, base].copy(), aug[:, base + 1].copy()
                    aug[:, base] = x * cos_a - y * sin_a
                    aug[:, base + 1] = x * sin_a + y * cos_a

            out_path = out_cls / f"aug_{Path(entry['file']).stem}_{aug_i:02d}.npy"
            np.save(out_path, aug)
            augmented.append({
                "class": cls_name,
                "file": str(out_path),
                "frames": int(aug.shape[0]),
            })

    return augmented


# ---------------------------------------------------------------------------
# Step 2: Build train/val/test datasets
# ---------------------------------------------------------------------------

def build_datasets(manifest: list[dict], seq_len: int = 30):
    """Build static and dynamic split NPZ files from manifest."""
    from utils.common.gesture_catalog import all_gestures

    # Load class modes
    class_modes = {}
    cm_path = REGISTRY / "class_modes.json"
    if cm_path.exists():
        class_modes = json.loads(cm_path.read_text(encoding="utf-8"))
    else:
        class_modes = {g.display_name: g.gesture_mode for g in all_gestures()}

    # Build class list from manifest
    classes = sorted({e["class"] for e in manifest})
    class_to_idx = {c: i for i, c in enumerate(classes)}

    X_static = []
    y_static = []
    X_dynamic = []
    y_dynamic = []

    for entry in manifest:
        arr = np.load(entry["file"])
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        cls = entry["class"]
        idx = class_to_idx[cls]
        mode = class_modes.get(cls, "static")

        # Static: use mean of frames (for multi-frame) or the single frame
        if arr.shape[0] == 1:
            X_static.append(arr[0])
        else:
            # Use individual frames as separate static samples
            for frame_i in range(arr.shape[0]):
                X_static.append(arr[frame_i])
                y_static.append(idx)
            # Also add a dynamic sample if applicable
            if mode == "dynamic" and arr.shape[0] >= 3:
                padded = _sequence_pad(arr, seq_len)
                X_dynamic.append(padded)
                y_dynamic.append(idx)
            continue

        y_static.append(idx)

    X_static = np.array(X_static, dtype=np.float32)
    y_static = np.array(y_static, dtype=np.int64)

    print(f"\n  Dataset shapes:")
    print(f"    Static: X={X_static.shape}, y={y_static.shape}")
    print(f"    Unique static classes: {len(np.unique(y_static))}")

    # Train/val/test split (70/15/15)
    from sklearn.model_selection import train_test_split

    # Ensure enough samples per class
    unique_classes, class_counts = np.unique(y_static, return_counts=True)
    min_count = class_counts.min()
    print(f"    Min samples per class: {min_count}")
    print(f"    Max samples per class: {class_counts.max()}")

    # If any class has fewer than 3 samples, we can't stratify properly
    if min_count < 3:
        print(f"    WARNING: Some classes have <3 samples, using non-stratified split")
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X_static, y_static, test_size=0.30, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp, test_size=0.50, random_state=42)
    else:
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X_static, y_static, test_size=0.30, random_state=42, stratify=y_static)
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(STATIC_SPLIT,
             X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val,
             X_test=X_test, y_test=y_test)
    print(f"    Saved {STATIC_SPLIT}")
    print(f"    Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # Dynamic split
    if X_dynamic:
        X_dyn = np.array(X_dynamic, dtype=np.float32)
        y_dyn = np.array(y_dynamic, dtype=np.int64)
        print(f"    Dynamic: X={X_dyn.shape}, y={y_dyn.shape}")

        n_dyn_classes = len(np.unique(y_dyn))
        if n_dyn_classes >= 2 and len(y_dyn) >= 6:
            try:
                X_td, X_tmpd, y_td, y_tmpd = train_test_split(
                    X_dyn, y_dyn, test_size=0.30, random_state=42, stratify=y_dyn)
                X_vd, X_testd, y_vd, y_testd = train_test_split(
                    X_tmpd, y_tmpd, test_size=0.50, random_state=42, stratify=y_tmpd)
            except ValueError:
                X_td, X_tmpd, y_td, y_tmpd = train_test_split(
                    X_dyn, y_dyn, test_size=0.30, random_state=42)
                X_vd, X_testd, y_vd, y_testd = train_test_split(
                    X_tmpd, y_tmpd, test_size=0.50, random_state=42)
            np.savez(DYNAMIC_SPLIT,
                     X_train=X_td, y_train=y_td,
                     X_val=X_vd, y_val=y_vd,
                     X_test=X_testd, y_test=y_testd)
            print(f"    Saved {DYNAMIC_SPLIT}")
        else:
            print(f"    Not enough dynamic data for split (classes={n_dyn_classes}, samples={len(y_dyn)})")

    # Save label_map
    REGISTRY.mkdir(parents=True, exist_ok=True)
    label_map_path = REGISTRY / "label_map.json"
    label_map_path.write_text(
        json.dumps(class_to_idx, indent=2), encoding="utf-8")
    print(f"    Saved label_map.json ({len(class_to_idx)} classes)")

    # Update class_modes
    new_modes = {}
    for cls in classes:
        new_modes[cls] = class_modes.get(cls, "static")
    cm_path.write_text(json.dumps(new_modes, indent=2), encoding="utf-8")
    print(f"    Updated class_modes.json")

    return class_to_idx


def _sequence_pad(arr: np.ndarray, seq_len: int) -> np.ndarray:
    """Pad or truncate a sequence to fixed length."""
    if arr.shape[0] >= seq_len:
        return arr[:seq_len]
    pad = np.repeat(arr[-1:], seq_len - arr.shape[0], axis=0)
    return np.concatenate([arr, pad], axis=0)


# ---------------------------------------------------------------------------
# Step 3: Train static model
# ---------------------------------------------------------------------------

def train_static_model(epochs: int = 100, patience: int = 15):
    """Train StaticMLP on real ISL landmark data."""
    import torch
    import torch.nn as nn
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    print(f"\n  Training StaticMLP (epochs={epochs}, patience={patience})...")

    blob = np.load(STATIC_SPLIT)
    X_train, y_train = blob["X_train"], blob["y_train"]
    X_val, y_val = blob["X_val"], blob["y_val"]
    X_test, y_test = blob["X_test"], blob["y_test"]

    # Z-score normalization (computed from training set)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-6] = 1.0
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    n_cls = int(max(y_train.max(), y_val.max(), y_test.max()) + 1)
    input_dim = X_train.shape[1]

    print(f"    Input dim: {input_dim}, Classes: {n_cls}")
    print(f"    Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    class StaticMLP(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(128, output_dim),
            )
        def forward(self, x):
            return self.net(x)

    model = StaticMLP(input_dim, n_cls)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    best_f1 = -1.0
    stale = 0
    best_state = None

    for epoch in range(epochs):
        model.train()

        # Mini-batch training for better generalization
        batch_size = min(256, len(X_train_t))
        indices = torch.randperm(len(X_train_t))
        total_loss = 0.0
        n_batches = 0

        for i in range(0, len(X_train_t), batch_size):
            batch_idx = indices[i:i+batch_size]
            xb = X_train_t[batch_idx]
            yb = y_train_t[batch_idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Evaluate on validation
        model.eval()
        with torch.no_grad():
            val_logits = model(torch.tensor(X_val, dtype=torch.float32))
            val_pred = torch.argmax(val_logits, dim=1).cpu().numpy()
        val_acc = accuracy_score(y_val, val_pred)
        _, _, val_f1, _ = precision_recall_fscore_support(
            y_val, val_pred, average="macro", zero_division=0)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}: loss={total_loss/n_batches:.4f}, "
                  f"val_acc={val_acc:.4f}, val_f1={val_f1:.4f}")

        if val_f1 > best_f1 + 0.001:
            best_f1 = val_f1
            stale = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)

    # Final test evaluation
    model.eval()
    with torch.no_grad():
        test_logits = model(torch.tensor(X_test, dtype=torch.float32))
        test_pred = torch.argmax(test_logits, dim=1).cpu().numpy()
    test_acc = accuracy_score(y_test, test_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, test_pred, average="macro", zero_division=0)

    print(f"\n    Test results: acc={test_acc:.4f}, p={p:.4f}, r={r:.4f}, f1={f1:.4f}")

    # Save model
    STATIC_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), STATIC_MODEL_PATH)
    print(f"    Saved model: {STATIC_MODEL_PATH}")

    # Save norm stats
    norm_stats = {"mean": mean.tolist(), "std": std.tolist()}
    norm_path = REGISTRY / "norm_stats.json"
    norm_path.write_text(json.dumps(norm_stats, indent=2), encoding="utf-8")
    print(f"    Saved norm_stats.json")

    # Save metrics
    metrics = {"accuracy": test_acc, "precision": float(p),
               "recall": float(r), "f1_score": float(f1)}
    (REGISTRY / "metrics_static_v001.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


# ---------------------------------------------------------------------------
# Step 4: Train dynamic model
# ---------------------------------------------------------------------------

def train_dynamic_model(epochs: int = 80, patience: int = 15):
    """Train DynamicBiGRU on real ISL landmark data."""
    if not DYNAMIC_SPLIT.exists():
        print("\n  SKIPPED dynamic model training (no dynamic split)")
        return None

    import torch
    import torch.nn as nn
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    print(f"\n  Training DynamicBiGRU (epochs={epochs}, patience={patience})...")

    blob = np.load(DYNAMIC_SPLIT)
    X_train, y_train = blob["X_train"], blob["y_train"]
    X_val, y_val = blob["X_val"], blob["y_val"]
    X_test, y_test = blob["X_test"], blob["y_test"]

    # Z-score normalization
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
    std[std < 1e-6] = 1.0

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    # Use full class count from label_map (must match static model)
    lm_path = REGISTRY / "label_map.json"
    if lm_path.exists():
        lm = json.loads(lm_path.read_text(encoding="utf-8"))
        n_cls = max(lm.values()) + 1
    else:
        n_cls = int(max(y_train.max(), y_val.max(), y_test.max()) + 1)
    # Ensure n_cls covers all label indices in dynamic data
    data_max = int(max(y_train.max(), y_val.max(), y_test.max()) + 1)
    n_cls = max(n_cls, data_max)
    input_dim = X_train.shape[-1]
    print(f"    Input dim: {input_dim}, Seq len: {X_train.shape[1]}, Classes: {n_cls}")

    class DynamicBiGRU(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.gru = nn.GRU(input_dim, 128, num_layers=2, dropout=0.3,
                              bidirectional=True, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, output_dim))
        def forward(self, x):
            y, _ = self.gru(x)
            return self.head(y.mean(dim=1))

    model = DynamicBiGRU(input_dim, n_cls)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # Use class weights to handle sparse classes (many classes have 0 samples)
    class_counts = np.bincount(y_train, minlength=n_cls).astype(np.float32)
    class_counts[class_counts == 0] = 1.0  # avoid div-by-zero for absent classes
    weights = 1.0 / class_counts
    weights /= weights.sum()
    weights *= n_cls
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32),
        label_smoothing=0.05,
    )

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)

    best_f1 = -1.0
    stale = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_t)
        loss = criterion(logits, y_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            vl = model(torch.tensor(X_val, dtype=torch.float32))
            vp = torch.argmax(vl, dim=1).cpu().numpy()
        _, _, vf1, _ = precision_recall_fscore_support(
            y_val, vp, average="macro", zero_division=0)

        if (epoch + 1) % 10 == 0:
            vacc = accuracy_score(y_val, vp)
            print(f"    Epoch {epoch+1:3d}: loss={loss.item():.4f}, val_acc={vacc:.4f}, val_f1={vf1:.4f}")

        if vf1 > best_f1 + 0.001:
            best_f1 = vf1
            stale = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        tl = model(torch.tensor(X_test, dtype=torch.float32))
        tp = torch.argmax(tl, dim=1).cpu().numpy()
    tacc = accuracy_score(y_test, tp)
    p, r, f1, _ = precision_recall_fscore_support(y_test, tp, average="macro", zero_division=0)
    print(f"\n    Dynamic test: acc={tacc:.4f}, p={p:.4f}, r={r:.4f}, f1={f1:.4f}")

    DYNAMIC_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), DYNAMIC_MODEL_PATH)
    print(f"    Saved model: {DYNAMIC_MODEL_PATH}")

    dyn_norm = {"mean": mean.tolist(), "std": std.tolist()}
    (REGISTRY / "dynamic_norm_stats.json").write_text(
        json.dumps(dyn_norm, indent=2), encoding="utf-8")

    metrics = {"accuracy": tacc, "precision": float(p),
               "recall": float(r), "f1_score": float(f1)}
    (REGISTRY / "metrics_dynamic_v001.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train MUDRA from real ISL videos")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--sample-every", type=int, default=2,
                        help="Extract every Nth frame from videos (1=all)")
    parser.add_argument("--augment-copies", type=int, default=4,
                        help="Number of augmented copies per sample")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip extraction, use existing manifest")
    args = parser.parse_args()

    t0 = time.time()
    print()
    print("=" * 64)
    print("  MUDRA: Train from Real ISL Videos")
    print("=" * 64)

    # Step 0: Map videos to classes
    print("\n[Step 0] Mapping ISL videos to gesture classes...")
    video_to_class = build_video_class_map()
    print(f"  Matched {len(video_to_class)} videos to existing classes")
    for stem, cls in sorted(video_to_class.items()):
        print(f"    {stem}.mp4 → {cls}")

    if not video_to_class:
        print("ERROR: No videos could be mapped to classes!")
        sys.exit(1)

    # Step 1: Extract landmarks
    if args.skip_extract and MANIFEST_PATH.exists():
        print("\n[Step 1] Using existing manifest (--skip-extract)")
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    else:
        print("\n[Step 1] Extracting landmarks from ISL videos...")
        manifest = extract_landmarks(video_to_class, sample_every_n=args.sample_every)

    if not manifest:
        print("ERROR: No landmarks extracted!")
        sys.exit(1)

    # Step 2: Build datasets
    print("\n[Step 2] Building train/val/test datasets...")
    class_to_idx = build_datasets(manifest)

    # Step 3: Train static model
    print("\n[Step 3] Training static gesture model...")
    static_metrics = train_static_model(epochs=args.epochs)

    # Step 4: Train dynamic model
    print("\n[Step 4] Training dynamic gesture model...")
    dynamic_metrics = train_dynamic_model(epochs=args.epochs)

    # Summary
    elapsed = time.time() - t0
    print()
    print("=" * 64)
    print("  TRAINING COMPLETE")
    print("=" * 64)
    print(f"  Classes trained: {len(class_to_idx)}")
    print(f"  Static model: {static_metrics}")
    if dynamic_metrics:
        print(f"  Dynamic model: {dynamic_metrics}")
    print(f"  Total time: {elapsed:.1f}s")
    print()
    print("  Artifacts saved:")
    for p in [STATIC_MODEL_PATH, DYNAMIC_MODEL_PATH,
              REGISTRY / "label_map.json",
              REGISTRY / "norm_stats.json",
              REGISTRY / "class_modes.json",
              REGISTRY / "dynamic_norm_stats.json"]:
        exists = "✓" if p.exists() else "✗"
        print(f"    {exists} {p}")
    print()


if __name__ == "__main__":
    main()
