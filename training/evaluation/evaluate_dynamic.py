"""Evaluate dynamic BiGRU model with confusion matrix and per-class metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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

from training.trainers.train_dynamic import DynamicBiGRU


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/dynamic_split_v1.npz")
    parser.add_argument("--model", default="models/dynamic/dynamic_bigru_v001.pt")
    parser.add_argument("--outdir", default="models/registry")
    args = parser.parse_args()

    blob = np.load(args.data)
    X_test, y_test = blob["X_test"], blob["y_test"]

    norm = json.loads(Path("models/registry/dynamic_norm_stats.json").read_text(encoding="utf-8"))
    mean = np.array(norm["mean"], dtype=np.float32)
    std = np.array(norm["std"], dtype=np.float32)
    X_test = (X_test - mean) / std

    label_map = json.loads(Path("models/registry/label_map.json").read_text(encoding="utf-8"))
    idx_to_label = {v: k for k, v in label_map.items()}
    n_classes = max(label_map.values()) + 1

    model = DynamicBiGRU(input_dim=X_test.shape[2], output_dim=n_classes)
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32))
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()

    acc = accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "evaluation_dynamic_v001.json").write_text(
        json.dumps({"accuracy": float(acc), "precision": float(p), "recall": float(r), "f1_score": float(f1)}, indent=2),
        encoding="utf-8",
    )

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(n_classes))
    np.savetxt(outdir / "confusion_dynamic_v001.csv", cm, fmt="%d", delimiter=",")

    per_p, per_r, per_f1, per_sup = precision_recall_fscore_support(
        y_test, y_pred, labels=np.arange(n_classes), average=None, zero_division=0
    )
    rows = []
    for idx in range(n_classes):
        rows.append(
            {
                "class_index": idx,
                "class_name": idx_to_label.get(idx, f"class_{idx}"),
                "precision": float(per_p[idx]),
                "recall": float(per_r[idx]),
                "f1_score": float(per_f1[idx]),
                "support": int(per_sup[idx]),
            }
        )
    pd.DataFrame(rows).to_csv(outdir / "per_class_metrics_dynamic_v001.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(cm).plot(ax=ax, colorbar=False)
    ax.set_title("Dynamic Model Confusion Matrix")
    fig.tight_layout()
    fig.savefig(outdir / "confusion_dynamic_v001.png")
    plt.close(fig)
    print({"accuracy": float(acc), "precision": float(p), "recall": float(r), "f1_score": float(f1)})


if __name__ == "__main__":
    main()
