"""Train dynamic gesture model (BiGRU) on sequence landmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class DynamicBiGRU(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, 128, num_layers=2, dropout=0.3, bidirectional=True, batch_first=True)
        self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, output_dim))

    def forward(self, x):
        y, _ = self.gru(x)
        pooled = y.mean(dim=1)
        return self.head(pooled)


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        pred = torch.argmax(logits, dim=1).cpu().numpy()
    acc = accuracy_score(y, pred)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    return float(acc), float(p), float(r), float(f1), pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/dynamic_split_v1.npz")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--out", default="models/dynamic/dynamic_bigru_v001.pt")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dynamic split not found at {data_path}. Run build_dataset.py first.")

    blob = np.load(data_path)
    X_train, y_train = blob["X_train"], blob["y_train"]
    X_val, y_val = blob["X_val"], blob["y_val"]
    X_test, y_test = blob["X_test"], blob["y_test"]

    feat_mean = X_train.mean(axis=(0, 1))
    feat_std = X_train.std(axis=(0, 1))
    feat_std[feat_std < 1e-6] = 1.0

    X_train = (X_train - feat_mean) / feat_std
    X_val = (X_val - feat_mean) / feat_std
    X_test = (X_test - feat_mean) / feat_std

    with open("models/registry/label_map.json", "r", encoding="utf-8") as f:
        label_map = json.load(f)
    n_classes = max(label_map.values()) + 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model = DynamicBiGRU(input_dim=X_train.shape[2], output_dim=n_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    best_state = None
    best_f1 = -1.0
    stale = 0
    for _ in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

        _, _, _, f1, _ = evaluate(model, X_val, y_val)
        if f1 > best_f1 + 0.002:
            best_f1 = f1
            stale = 0
            best_state = model.state_dict()
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    acc, p, r, f1, _ = evaluate(model, X_test, y_test)
    torch.save(model.state_dict(), args.out)
    Path("models/registry").mkdir(parents=True, exist_ok=True)
    Path("models/registry/dynamic_norm_stats.json").write_text(
        json.dumps({"mean": feat_mean.tolist(), "std": feat_std.tolist()}, indent=2),
        encoding="utf-8",
    )
    Path("models/registry/metrics_dynamic_v001.json").write_text(
        json.dumps({"accuracy": acc, "precision": p, "recall": r, "f1_score": f1}, indent=2),
        encoding="utf-8",
    )
    print({"accuracy": acc, "precision": p, "recall": r, "f1_score": f1})


if __name__ == "__main__":
    main()
