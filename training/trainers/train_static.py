"""Train static gesture model (MLP landmark baseline)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class StaticMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
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


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        pred = torch.argmax(logits, dim=1).cpu().numpy()
    acc = accuracy_score(y, pred)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    return float(acc), float(p), float(r), float(f1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/static_split_v1.npz")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--out", default="models/static/static_mlp_v001.pt")
    args = parser.parse_args()

    blob = np.load(args.data)
    X_train, y_train = blob["X_train"], blob["y_train"]
    X_val, y_val = blob["X_val"], blob["y_val"]
    X_test, y_test = blob["X_test"], blob["y_test"]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-6] = 1.0
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    n_cls = int(max(y_train.max(), y_val.max(), y_test.max()) + 1)
    model = StaticMLP(input_dim=X_train.shape[1], output_dim=n_cls)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    best_f1 = -1.0
    stale = 0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

        _, _, _, val_f1 = evaluate(model, X_val, y_val)
        if val_f1 > best_f1 + 0.002:
            best_f1 = val_f1
            stale = 0
            best_state = model.state_dict()
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    acc, p, r, f1 = evaluate(model, X_test, y_test)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)

    Path("models/registry").mkdir(parents=True, exist_ok=True)
    Path("models/registry/norm_stats.json").write_text(
        json.dumps({"mean": mean.tolist(), "std": std.tolist()}, indent=2),
        encoding="utf-8",
    )

    metrics = {"accuracy": acc, "precision": p, "recall": r, "f1_score": f1}
    Path("models/registry/metrics_static_v001.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(metrics)


if __name__ == "__main__":
    main()
