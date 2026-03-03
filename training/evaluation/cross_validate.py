"""Group-aware cross-validation entry point for static model features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from utils.metrics.classification import compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/static_split_v1.npz")
    parser.add_argument("--out", default="models/registry/cv_static_v001.json")
    args = parser.parse_args()

    blob = np.load(args.features)
    X = np.concatenate([blob["X_train"], blob["X_val"], blob["X_test"]], axis=0)
    y = np.concatenate([blob["y_train"], blob["y_val"], blob["y_test"]], axis=0)

    # Placeholder signer groups; replace with real signer_id array in production dataset builder.
    groups = np.arange(len(y)) % 5

    gkf = GroupKFold(n_splits=5)
    fold_metrics = []

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Nearest-centroid lightweight baseline for CV diagnostics.
        centroids = {}
        for cls in np.unique(y_tr):
            centroids[int(cls)] = X_tr[y_tr == cls].mean(axis=0)

        y_pred = []
        for row in X_te:
            dists = {cls: float(np.linalg.norm(row - ctr)) for cls, ctr in centroids.items()}
            y_pred.append(min(dists, key=dists.get))

        fold_metrics.append(compute_metrics(y_te.tolist(), y_pred))

    mean = {k: float(np.mean([f[k] for f in fold_metrics])) for k in fold_metrics[0]}
    std = {f"{k}_std": float(np.std([f[k] for f in fold_metrics])) for k in fold_metrics[0]}
    payload = {"folds": fold_metrics, "mean": mean, "std": std}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["mean"], indent=2))


if __name__ == "__main__":
    main()
