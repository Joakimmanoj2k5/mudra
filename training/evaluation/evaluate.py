"""Compute metrics and confusion matrix for static model outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from training.trainers.train_static import StaticMLP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/static_split_v1.npz")
    parser.add_argument("--model", default="models/static/static_mlp_v001.pt")
    parser.add_argument("--outdir", default="models/registry")
    args = parser.parse_args()

    blob = np.load(args.data)
    X_test, y_test = blob["X_test"], blob["y_test"]

    norm = json.loads(Path("models/registry/norm_stats.json").read_text(encoding="utf-8"))
    mean = np.array(norm["mean"], dtype=np.float32)
    std = np.array(norm["std"], dtype=np.float32)
    X_test = (X_test - mean) / std

    n_cls = int(y_test.max() + 1)
    model = StaticMLP(input_dim=X_test.shape[1], output_dim=n_cls)
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32))
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }
    (outdir / "evaluation_static_v001.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    cm = confusion_matrix(y_test, y_pred)
    np.savetxt(outdir / "confusion_static_v001.csv", cm, fmt="%d", delimiter=",")

    with open("models/registry/label_map.json", "r", encoding="utf-8") as f:
        label_map = json.load(f)
    idx_to_label = {v: k for k, v in label_map.items()}
    per_p, per_r, per_f1, per_sup = precision_recall_fscore_support(
        y_test, y_pred, labels=np.arange(n_cls), average=None, zero_division=0
    )
    per_class = []
    for idx in range(n_cls):
        per_class.append(
            {
                "class_index": idx,
                "class_name": idx_to_label.get(idx, f"class_{idx}"),
                "precision": float(per_p[idx]),
                "recall": float(per_r[idx]),
                "f1_score": float(per_f1[idx]),
                "support": int(per_sup[idx]),
            }
        )
    pd.DataFrame(per_class).to_csv(outdir / "per_class_metrics_static_v001.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(cm).plot(ax=ax, colorbar=False)
    ax.set_title("Static Model Confusion Matrix")
    fig.tight_layout()
    fig.savefig(outdir / "confusion_static_v001.png")
    print(metrics)


if __name__ == "__main__":
    main()
