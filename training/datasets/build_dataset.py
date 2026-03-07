"""Build static and dynamic train/val/test datasets from extracted landmarks.

Handles low-data scenarios by augmenting samples to a minimum per class
using Gaussian noise and scale jitter so stratified splits work.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from utils.common.gesture_catalog import WORD_SPECS


def _norm_name(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def _dynamic_class_names() -> set[str]:
    names = set()
    for spec in WORD_SPECS:
        if spec.gesture_mode == "dynamic":
            names.add(_norm_name(spec.display_name))
            names.add(_norm_name(spec.code.replace("WORD_", "")))
    return names


def _sequence_pad(arr: np.ndarray, seq_len: int) -> np.ndarray:
    if arr.shape[0] >= seq_len:
        return arr[:seq_len]
    pad = np.repeat(arr[-1][None, :], seq_len - arr.shape[0], axis=0)
    return np.concatenate([arr, pad], axis=0)


def _augment_feature(feat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Create one augmented copy of a feature vector via noise + scale jitter."""
    noise = rng.normal(0, 0.012, feat.shape).astype(np.float32)
    scale = rng.uniform(0.93, 1.07)
    return (feat + noise) * scale


def _augment_sequence(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Augment a sequence of feature vectors."""
    noise = rng.normal(0, 0.012, seq.shape).astype(np.float32)
    scale = rng.uniform(0.93, 1.07)
    return (seq + noise) * scale


def _oversample_to_min(X_list, y_list, min_per_class: int, rng: np.random.Generator):
    """Augment underrepresented classes to at least min_per_class samples."""
    from collections import Counter
    counts = Counter(y_list)
    X_aug = list(X_list)
    y_aug = list(y_list)

    for cls, count in counts.items():
        if count >= min_per_class:
            continue
        # Collect indices of this class
        idxs = [i for i, lbl in enumerate(y_list) if lbl == cls]
        needed = min_per_class - count
        for _ in range(needed):
            src_idx = idxs[rng.integers(0, len(idxs))]
            X_aug.append(_augment_feature(X_list[src_idx], rng))
            y_aug.append(cls)

    return X_aug, y_aug


def _oversample_sequences(X_list, y_list, min_per_class: int, rng: np.random.Generator):
    """Augment underrepresented dynamic sequence classes."""
    from collections import Counter
    counts = Counter(y_list)
    X_aug = list(X_list)
    y_aug = list(y_list)

    for cls, count in counts.items():
        if count >= min_per_class:
            continue
        idxs = [i for i, lbl in enumerate(y_list) if lbl == cls]
        needed = min_per_class - count
        for _ in range(needed):
            src_idx = idxs[rng.integers(0, len(idxs))]
            X_aug.append(_augment_sequence(X_list[src_idx], rng))
            y_aug.append(cls)

    return X_aug, y_aug


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/interim/landmarks_manifest.json")
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--min-samples", type=int, default=20,
                        help="Minimum samples per class (augmented if fewer)")
    args = parser.parse_args()

    rng = np.random.default_rng(42)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    classes = sorted(set(row["class"] for row in manifest))
    class_map = {c: i for i, c in enumerate(classes)}

    X_raw = []
    y_raw = []
    X_seq_raw = []
    y_seq_raw = []
    dynamic_names = _dynamic_class_names()

    for row in manifest:
        arr = np.load(row["file"])
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        cls_label = class_map[row["class"]]

        # For static: use per-frame features (each frame is a sample)
        for frame_feat in arr:
            X_raw.append(frame_feat)
            y_raw.append(cls_label)

        # For dynamic:
        cls_norm = _norm_name(str(row["class"]))
        if cls_norm in dynamic_names and arr.shape[0] >= 2:
            X_seq_raw.append(_sequence_pad(arr, args.seq_len))
            y_seq_raw.append(cls_label)

    print(f"Raw samples: {len(X_raw)} static, {len(X_seq_raw)} dynamic sequences")
    print(f"Classes: {len(classes)}")

    # Augment to minimum per class
    X_aug, y_aug = _oversample_to_min(X_raw, y_raw, args.min_samples, rng)

    X = np.array(X_aug, dtype=np.float32)
    y = np.array(y_aug, dtype=np.int64)
    print(f"After augmentation: {len(X)} static samples")

    # Split — try stratified, fallback to non-stratified
    try:
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)
        print("Used stratified split for static data")
    except ValueError:
        print("Stratified split failed, using random split for static data")
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X, y, test_size=0.30, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp, test_size=0.50, random_state=42)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "static_split_v1.npz",
             X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val,
             X_test=X_test, y_test=y_test)

    has_dynamic = False
    if X_seq_raw:
        X_seq_aug, y_seq_aug = _oversample_sequences(
            X_seq_raw, y_seq_raw, max(4, args.min_samples // 4), rng)

        X_seq = np.array(X_seq_aug, dtype=np.float32)
        y_seq = np.array(y_seq_aug, dtype=np.int64)
        print(f"After augmentation: {len(X_seq)} dynamic sequences")

        try:
            X_train_d, X_tmp_d, y_train_d, y_tmp_d = train_test_split(
                X_seq, y_seq, test_size=0.30, random_state=42, stratify=y_seq)
            X_val_d, X_test_d, y_val_d, y_test_d = train_test_split(
                X_tmp_d, y_tmp_d, test_size=0.50, random_state=42, stratify=y_tmp_d)
            print("Used stratified split for dynamic data")
        except ValueError:
            print("Stratified split failed, using random split for dynamic data")
            X_train_d, X_tmp_d, y_train_d, y_tmp_d = train_test_split(
                X_seq, y_seq, test_size=0.30, random_state=42)
            X_val_d, X_test_d, y_val_d, y_test_d = train_test_split(
                X_tmp_d, y_tmp_d, test_size=0.50, random_state=42)

        has_dynamic = True
        np.savez(
            out / "dynamic_split_v1.npz",
            X_train=X_train_d, y_train=y_train_d,
            X_val=X_val_d, y_val=y_val_d,
            X_test=X_test_d, y_test=y_test_d,
        )

    Path("models/registry").mkdir(parents=True, exist_ok=True)
    Path("models/registry/label_map.json").write_text(
        json.dumps(class_map, indent=2), encoding="utf-8")

    print(f"\nSaved data/processed/static_split_v1.npz")
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    if has_dynamic:
        print(f"Saved data/processed/dynamic_split_v1.npz")
        print(f"  Train: {len(X_train_d)}, Val: {len(X_val_d)}, Test: {len(X_test_d)}")
    else:
        print("No dynamic sequences found in manifest; skipped dynamic split.")


if __name__ == "__main__":
    main()
