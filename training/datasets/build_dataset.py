"""Build static and dynamic train/val/test datasets from extracted landmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/interim/landmarks_manifest.json")
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--seq-len", type=int, default=30)
    args = parser.parse_args()

    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    df = pd.DataFrame(rows)
    classes = sorted(df["class"].unique().tolist())
    class_map = {c: i for i, c in enumerate(classes)}

    X = []
    y = []
    X_seq = []
    y_seq = []
    dynamic_names = _dynamic_class_names()
    has_dynamic = False

    for _, row in df.iterrows():
        arr = np.load(row["file"])
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        cls_norm = _norm_name(str(row["class"]))
        if arr.shape[0] == 1:
            X.append(arr[0])
            y.append(class_map[row["class"]])
        else:
            X.append(arr.mean(axis=0))
            y.append(class_map[row["class"]])

        if cls_norm in dynamic_names and arr.shape[0] >= 2:
            X_seq.append(_sequence_pad(arr, args.seq_len))
            y_seq.append(class_map[row["class"]])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "static_split_v1.npz", X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)

    if X_seq:
        has_dynamic = True
        X_seq = np.array(X_seq, dtype=np.float32)
        y_seq = np.array(y_seq, dtype=np.int64)
        X_train_d, X_tmp_d, y_train_d, y_tmp_d = train_test_split(
            X_seq, y_seq, test_size=0.30, random_state=42, stratify=y_seq
        )
        X_val_d, X_test_d, y_val_d, y_test_d = train_test_split(
            X_tmp_d, y_tmp_d, test_size=0.50, random_state=42, stratify=y_tmp_d
        )
        np.savez(
            out / "dynamic_split_v1.npz",
            X_train=X_train_d,
            y_train=y_train_d,
            X_val=X_val_d,
            y_val=y_val_d,
            X_test=X_test_d,
            y_test=y_test_d,
        )

    Path("models/registry").mkdir(parents=True, exist_ok=True)
    Path("models/registry/label_map.json").write_text(json.dumps({k: v for k, v in class_map.items()}, indent=2), encoding="utf-8")
    print("Saved data/processed/static_split_v1.npz")
    if has_dynamic:
        print("Saved data/processed/dynamic_split_v1.npz")
    else:
        print("No dynamic sequences were found in manifest; skipped dynamic split.")


if __name__ == "__main__":
    main()
