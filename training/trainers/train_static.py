"""Train static gesture model (MLP landmark baseline).

Improvements over baseline:
- Mini-batch training with DataLoader
- CosineAnnealingLR scheduler
- Runtime data augmentation (noise + dropout)
- Per-class metrics logging
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
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


def augment_batch(X: torch.Tensor, noise_std: float = 0.01, drop_p: float = 0.05) -> torch.Tensor:
    """Runtime augmentation: Gaussian noise + random feature dropout."""
    noise = torch.randn_like(X) * noise_std
    mask = (torch.rand_like(X) > drop_p).float()
    return (X + noise) * mask


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
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="models/static/static_mlp_v001.pt")
    args = parser.parse_args()

    blob = np.load(args.data)
    X_train, y_train = blob["X_train"], blob["y_train"]
    X_val, y_val = blob["X_val"], blob["y_val"]
    X_test, y_test = blob["X_test"], blob["y_test"]

    # Normalize
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-6] = 1.0
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    n_cls = int(max(y_train.max(), y_val.max(), y_test.max()) + 1)
    model = StaticMLP(input_dim=X_train.shape[1], output_dim=n_cls)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # Mini-batch DataLoader
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    best_f1 = -1.0
    stale = 0
    best_state = None

    print(f"Training: {len(X_train)} samples, {n_cls} classes, "
          f"batch={args.batch_size}, epochs={args.epochs}")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        batches = 0
        for X_batch, y_batch in train_loader:
            # Runtime augmentation
            X_aug = augment_batch(X_batch)
            optimizer.zero_grad()
            logits = model(X_aug)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1

        scheduler.step()

        val_acc, _, _, val_f1 = evaluate(model, X_val, y_val)
        avg_loss = epoch_loss / max(batches, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch+1:3d} | loss={avg_loss:.4f} | val_acc={val_acc:.3f} | "
                  f"val_f1={val_f1:.3f} | lr={lr_now:.6f}")

        if val_f1 > best_f1 + 0.002:
            best_f1 = val_f1
            stale = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                print(f"  Early stopping at epoch {epoch+1} (best val_f1={best_f1:.3f})")
                break

    if best_state:
        model.load_state_dict(best_state)

    acc, p, r, f1 = evaluate(model, X_test, y_test)
    print(f"\nTest metrics: acc={acc:.3f} | precision={p:.3f} | recall={r:.3f} | f1={f1:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"Saved model to {args.out}")

    Path("models/registry").mkdir(parents=True, exist_ok=True)
    Path("models/registry/norm_stats.json").write_text(
        json.dumps({"mean": mean.tolist(), "std": std.tolist()}, indent=2),
        encoding="utf-8",
    )

    metrics = {"accuracy": acc, "precision": p, "recall": r, "f1_score": f1}
    Path("models/registry/metrics_static_v001.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved norm_stats and metrics to models/registry/")


if __name__ == "__main__":
    main()
